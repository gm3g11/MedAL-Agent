"""Balanced, distributed dispatcher for the Stage 1 matrix on SGE (qrsh).

Environment: Sun Grid Engine. Each `qrsh -q gpu@@coba-a40 -l gpu=1` (or
`@csecri-v100`, or `@coba-h100 -l gpu=4`) gives a single-GPU (or 4-GPU) shell on
some node with CUDA_VISIBLE_DEVICES set by SGE. The 12 GPUs are SEPARATE
allocations on different nodes — so there is no single process that sees them all.

Design: a **shared NFS work queue**. Run THIS SAME command in every qrsh session;
each worker atomically claims jobs (mkdir lock under <out>/_claims) and runs them,
skipping any whose trajectory JSONL already exists. Faster GPUs (A40/H100) drain
more jobs → makespan self-balances. Fully resumable; safe across nodes.

Per-GPU batch is auto-detected from the card name (A40->12, V100->6, H100->16)
unless --batch is given. Multi-GPU allocations (H100 gpu=4) spawn one worker
thread per local card.

Usage — in EACH qrsh session (after SGE sets CUDA_VISIBLE_DEVICES):
    python -m medal_bench.runner.dispatch \
        --datasets busi,mmwhs,btcv_synapse,... --methods P0,P1,P2,P3,P4,P5,P6,P7,P8,P9 \
        --seeds 1000,2000,3000 --profile bench512 --out-dir runs/stage1
    # --plan prints the grid + remaining count without running.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import os
import random
import signal
import subprocess
import sys
import threading
import time

from medal_bench.runner.trajectory import read_jsonl


def _gpu_info(global_id: str) -> tuple[str, float]:
    """(name, total_GB) via nvidia-smi; ('GPU', 0.0) if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", str(global_id), "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        name, _, mem = out.stdout.strip().partition(",")
        return name.strip() or "GPU", float(mem.strip()) / 1024.0
    except Exception:
        return "GPU", 0.0


# A claim is stale (its holder died mid-cell) if there is no COMPLETE final .jsonl and
# its .partial trajectory has not grown for this long. MUST exceed the slowest single
# ROUND, else a healthy-but-slow holder is falsely stolen -> DUPLICATE run. The old claim
# that this is "non-corrupting" was FALSE: 2026-06-22 a v5 hippo P9 round legitimately took
# >60 min, the slow-but-healthy holder was falsely stolen, and the duplicate then clobbered
# the holder's good 7-round final with a broken 2-round one. The v5 grid (5000it, up to 20%
# budgets) has rounds far slower than the v3 canary's 27 min that justified 3600. Raised to
# 4h (> any single v5 round, << the 12h CELL_TIMEOUT), env-overridable.
STALE_CLAIM_SEC = int(os.environ.get("MEDAL_STALE_CLAIM_SEC", str(4 * 3600)))

# Per-cell wall-clock cap. A run_one child that hangs (the 'Loading CRC_default'
# env/NFS hang under saturation, or a wedged dataloader/GPU) would otherwise block its
# worker forever. On timeout the whole child process group is killed and the cell is
# released for retry. Must exceed the slowest FULL cell (P9/btcv/brats @512 @1000it);
# 4h is generous — calibrate down from the v3 canary's observed per-cell runtimes.
CELL_TIMEOUT_SEC = 4 * 3600


def _cell_done(path: str) -> bool:
    """A cell is DONE only when its final .jsonl exists AND carries all its rounds
    (round count == the total_rounds it logged). A short/truncated file is NOT done —
    so a lease-race or interrupted append can never make a cell look finished."""
    try:
        recs = read_jsonl(path)
    except Exception:
        return False
    return bool(recs) and len(recs) == recs[0].get("total_rounds", -1)


def _run_cell(cmd: list[str], env: dict, timeout: int) -> int:
    """Run one cell as its OWN process group with a wall-clock timeout. On timeout,
    SIGKILL the whole group (run_one + its dataloader/SAM children) so a hung cell
    can't wedge the worker. Returns the child rc, or -9 on timeout."""
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            pass
        try:
            proc.wait(timeout=30)
        except Exception:
            pass
        return -9

# job cost weights (higher = slower) for A40/V100 affinity
_METHOD_W = {"P9": 6, "P2": 4, "P4": 4, "P3": 2, "P5": 2, "P7": 2, "P8": 2, "P1": 1, "P6": 1, "P0": 0}
_BIG_DATASETS = {"mmwhs", "msd_task07_pancreas", "ext_brats2020", "ext_amos_ct", "ext_amos_mri",
                 "ext_abdoment1k", "kits19", "ext_word_ct", "flare22", "btcv_synapse"}


def _job_weight(d: str, m: str) -> int:
    return _METHOD_W.get(m, 1) * 10 + (5 if d in _BIG_DATASETS else 0)


def _auto_batch(name: str, mem_gb: float) -> int:
    """Memory-aware batch @512. Conservative: unknown/16GB V100 -> 6."""
    n = name.upper()
    if "H100" in n or "A100" in n:
        return 24
    if "A40" in n or "A6000" in n:
        return 16
    if "V100" in n:
        return 12 if mem_gb >= 24 else 6   # 32GB -> 12, 16GB -> 6 (conservative default)
    # unknown card: size by detected memory, conservative if undetectable
    if mem_gb >= 40:
        return 16
    if mem_gb >= 24:
        return 12
    return 6


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--methods", default="P0,P1,P2,P3,P4,P5,P6,P7,P8,P9")
    ap.add_argument("--seeds", default="1000,2000,3000")
    ap.add_argument("--profile", default="bench512_dry")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--foundation", default="sam")
    ap.add_argument("--sam-model-type", default="vit_h")
    ap.add_argument("--gpus", default=None,
                    help="override local GPU ids (default: CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--batch", type=int, default=None, help="override auto per-GPU batch")
    ap.add_argument("--num-iters", type=int, default=None)
    ap.add_argument("--save-predictions", action="store_true",
                    help="forward --save-predictions to each cell (always-on val masks)")
    ap.add_argument("--save-logits", action="store_true",
                    help="forward --save-logits to each cell (fp16 probs; canary storage estimate)")
    ap.add_argument("--defer-surface", action="store_true",
                    help="forward --defer-surface (skip inline HD95/ASSD; backfill offline)")
    ap.add_argument("--adaptive", action="store_true",
                    help="forward --adaptive to each cell (frozen_v4 train-to-plateau)")
    ap.add_argument("--min-iters", type=int, default=None)
    ap.add_argument("--max-iters", type=int, default=None)
    ap.add_argument("--cell-timeout", type=int, default=CELL_TIMEOUT_SEC,
                    help="per-cell wall-clock cap (s); on timeout the cell is killed + released for retry")
    ap.add_argument("--prefer", default="auto", choices=["auto", "heavy", "light", "shuffle"],
                    help="job pickup order: heavy-first (A40/H100), light-first (V100), or shuffle. "
                         "auto = by local GPU kind.")
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args(argv)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out = args.out_dir
    claims = os.path.join(out, "_claims")
    os.makedirs(claims, exist_ok=True)

    def jsonl(d, m, s):
        return os.path.join(out, f"{d}__{m}__s{s}.jsonl")
    grid = [(d, m, s) for d, m, s in itertools.product(datasets, methods, seeds)]
    remaining = [g for g in grid if not _cell_done(jsonl(*g))]
    print(f"[dispatch] grid {len(grid)} ({len(datasets)}ds x {len(methods)}m x {len(seeds)}s); "
          f"{len(grid)-len(remaining)} done, {len(remaining)} remaining")

    gpus = (args.gpus or os.environ.get("CUDA_VISIBLE_DEVICES") or "0")
    gpu_ids = [g.strip() for g in gpus.split(",") if g.strip()]
    # TRAIN batch must be UNIFORM across GPUs for a comparable benchmark (it changes
    # the optimization). Use the profile's fixed batch unless the user passes a single
    # explicit --batch (still applied uniformly). _auto_batch is only an advisory hint.
    first_name = "GPU"
    for i, gid in enumerate(gpu_ids):
        name, mem = _gpu_info(gid)
        if i == 0:
            first_name = name
        hint = _auto_batch(name, mem)
        note = "" if mem == 0 or mem >= 16 else f"  WARN: {mem:.0f}GB may OOM @512 batch>=12"
        print(f"   local gpu {gid}: {name} ({mem:.0f}GB) train_batch={args.batch or 'profile'} "
              f"(fits ~{hint}){note}")

    # A40/V100 affinity: A40/H100 take the SLOW jobs first; V100 take light first.
    prefer = args.prefer
    if prefer == "auto":
        u = first_name.upper()
        prefer = "heavy" if ("A40" in u or "H100" in u or "A100" in u or "A6000" in u) else \
                 ("light" if "V100" in u else "shuffle")
    print(f"[dispatch] pickup order = {prefer} (slow jobs -> A40/H100; light -> V100)")
    if args.plan:
        print(f"[dispatch] --plan only; this worker would process from {len(remaining)} remaining jobs.")
        return 0

    # order by preference, with per-pid jitter so same-kind workers don't all collide
    rng = random.Random(os.getpid())
    order = remaining[:]
    rng.shuffle(order)                                  # jitter
    if prefer == "heavy":
        order.sort(key=lambda j: -_job_weight(j[0], j[1]))
    elif prefer == "light":
        order.sort(key=lambda j: _job_weight(j[0], j[1]))
    counts = {}
    lock = threading.Lock()

    def claim(jobname: str) -> bool:
        path = os.path.join(claims, jobname)
        try:
            os.mkdir(path)                            # atomic across NFS
            return True
        except FileExistsError:
            pass
        # Claim exists. Steal it if STALE (holder died): no COMPLETE final .jsonl and no
        # per-PID .partial has grown for STALE_CLAIM_SEC.
        final = os.path.join(out, jobname + ".jsonl")
        if _cell_done(final):                         # truly done (all rounds present)
            return False
        partials = glob.glob(final + ".partial.*")
        try:
            last = (max(os.path.getmtime(p) for p in partials) if partials
                    else os.path.getmtime(path))
        except OSError:
            return False
        if time.time() - last < STALE_CLAIM_SEC:      # still making progress
            return False
        # Win the steal atomically: only one worker can rename the lock dir.
        stealing = f"{path}.steal.{os.getpid()}"
        try:
            os.rename(path, stealing)
        except OSError:
            return False                              # another worker won the steal
        try:
            os.rmdir(stealing)
        except OSError:
            pass
        for p in partials:                            # drop the dead worker's partials
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.mkdir(path)
            print(f"  [claim] stole stale claim {jobname} (idle {int(time.time()-last)}s)", flush=True)
            return True
        except FileExistsError:
            return False

    def worker(gid: str):
        n = 0; busy = 0.0
        timed_out: set = set()    # cells THIS worker killed on timeout; skip to avoid
                                  # livelock on a deterministic hang (peers may retry).
        # Loop until the queue is truly drained: re-scan after each pass so a worker
        # picks up cells freed later (e.g. a peer died and its dangling claim was
        # cleared). Without this, single-pass workers exit and orphan late-freed cells.
        while True:
            progressed = False
            for (d, m, s) in order:
                if (d, m, s) in timed_out:
                    continue
                if _cell_done(jsonl(d, m, s)):
                    continue
                job = f"{d}__{m}__s{s}"
                if not claim(job):
                    continue
                progressed = True
                cmd = [sys.executable, "-m", "medal_bench.runner.run_one",
                       "--policy", m, "--dataset", d, "--seed", str(s),
                       "--profile", args.profile, "--out-dir", out,
                       "--foundation", args.foundation, "--sam-model-type", args.sam_model_type,
                       "--device", "cuda:0"]
                if args.batch is not None:            # uniform explicit train-batch override
                    cmd += ["--batch", str(args.batch)]
                if args.num_iters is not None:
                    cmd += ["--num-iters", str(args.num_iters)]
                if args.save_predictions:
                    cmd += ["--save-predictions"]
                if args.save_logits:
                    cmd += ["--save-logits"]
                if args.defer_surface:
                    cmd += ["--defer-surface"]
                if args.adaptive:
                    cmd += ["--adaptive"]
                    if args.min_iters is not None:
                        cmd += ["--min-iters", str(args.min_iters)]
                    if args.max_iters is not None:
                        cmd += ["--max-iters", str(args.max_iters)]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gid)
                t0 = time.time()
                rc = _run_cell(cmd, env, args.cell_timeout)
                dt = time.time() - t0
                n += 1; busy += dt
                if rc == -9:        # timed out: release the claim so a peer can retry,
                                    # and skip it here so this worker doesn't re-grab a hang.
                    try:
                        os.rmdir(os.path.join(claims, job))
                    except OSError:
                        pass
                    timed_out.add((d, m, s))
                    with lock:
                        print(f"  [gpu{gid}] {job} TIMEOUT {dt:.0f}s -> released for retry", flush=True)
                    continue
                with lock:
                    print(f"  [gpu{gid}] {job} {'OK' if rc==0 else f'RC{rc}'} {dt:.0f}s", flush=True)
            if not progressed:
                # Nothing CLAIMABLE this pass — but that is NOT the same as "queue drained":
                # some incomplete cells may be held by claims of dead workers that have not
                # yet aged past STALE_CLAIM_SEC. Exiting now would orphan them (no live worker
                # left to steal the claim once it goes stale). So exit ONLY when every cell is
                # truly DONE; otherwise sleep and re-scan so dead-worker claims are always
                # eventually reclaimed.
                if all(_cell_done(jsonl(d, m, s)) for (d, m, s) in order):
                    break
                time.sleep(60)
        with lock:
            counts[gid] = (n, busy)
            print(f"  [gpu{gid}] drained {n} jobs, busy {busy/60:.1f} min", flush=True)

    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(gid,)) for gid in gpu_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"[dispatch] this worker done in {(time.time()-t0)/60:.1f} min; "
          f"per-gpu {[ (g, c[0]) for g,c in counts.items() ]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
