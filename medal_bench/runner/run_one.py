"""Single-cell entry point: one ``(policy, dataset, seed, profile)`` run.

Usage:
    python -m medal_bench.runner.run_one \\
        --policy P0 --dataset busi --seed 1000 --profile pilot \\
        --out-dir runs/pilot_v1

Writes a JSONL trajectory to:
    {out-dir}/{dataset}__{policy}__s{seed}.jsonl
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

DEFAULT_DATA_ROOT = "/groups/echambe2/datasets/data"
DEFAULT_HF_CACHE = "/groups/echambe2/gmeng/MedAL-Agent/cache/hf_hub"
DEFAULT_FOUND_CACHE = "/groups/echambe2/gmeng/MedAL-Agent/cache/foundation_features"


def _build_adapter(name: str, data_root: str):
    from medal_bench.data.adapters import DATASET_REGISTRY
    if name not in DATASET_REGISTRY:
        raise SystemExit(f"unknown dataset {name}; choose from {sorted(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name](data_root)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy",   required=True, choices=[f"P{i}" for i in range(10)])
    ap.add_argument("--dataset",  required=True)
    ap.add_argument("--seed",     type=int, required=True)
    ap.add_argument("--profile",  default="pilot",
                    choices=sorted(__import__("medal_bench.profiles", fromlist=["PROFILES"]).PROFILES))
    ap.add_argument("--out-dir",  required=True)
    ap.add_argument("--data-root",default=DEFAULT_DATA_ROOT)
    ap.add_argument("--device",   default="cuda:0")
    ap.add_argument("--foundation", default="sam", choices=["sam", "stub"],
                    help="P7/P8 foundation features source.")
    ap.add_argument("--sam-model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"],
                    help="SAM encoder size for P7/P8. vit_b=HF; vit_l/vit_h=original .pth.")
    ap.add_argument("--sam-checkpoint", default=None,
                    help="Original-format SAM .pth (required for vit_l; vit_h has a default).")
    ap.add_argument("--hf-cache", default=DEFAULT_HF_CACHE)
    ap.add_argument("--foundation-cache", default=DEFAULT_FOUND_CACHE)
    ap.add_argument("--batch", type=int, default=None, help="override train batch_size (e.g. smaller on V100)")
    ap.add_argument("--num-iters", type=int, default=None, help="override train iters (fixed mode)")
    ap.add_argument("--adaptive", action="store_true",
                    help="frozen_v4: train each round to a train-loss plateau instead of fixed iters")
    ap.add_argument("--min-iters", type=int, default=None, help="adaptive floor (default 500)")
    ap.add_argument("--max-iters", type=int, default=None, help="adaptive cap (default 3000)")
    ap.add_argument("--save-predictions", action="store_true",
                    help="save compressed val prediction masks + ids + valid masks every round")
    ap.add_argument("--save-logits", action="store_true",
                    help="additionally save fp16 softmax probs (heavier; for the canary storage estimate)")
    ap.add_argument("--defer-surface", action="store_true",
                    help="skip inline HD95/ASSD (the slow medpy part); compute them offline from saved "
                         "masks via `surface_offline` (forces --save-predictions; DSC+detection stay inline)")
    ap.add_argument("--force", action="store_true",
                    help="rerun even if a complete final .jsonl already exists (else this cell is skipped)")
    args = ap.parse_args(argv)

    os.environ["HF_HOME"] = args.hf_cache
    os.environ["HF_HUB_CACHE"] = args.hf_cache

    from medal_bench.profiles import build_run_config, PROFILES
    from medal_bench.runner.al_loop import run_al
    from medal_bench.runner.splits import make_split

    # Bucket A #2 — preflight GPU-memory guard: refuse memory-heavy policies on
    # too-small cards (fail fast with a clear message instead of a mid-run OOM).
    import torch as _torch
    if args.device.startswith("cuda") and _torch.cuda.is_available():
        _total_gb = _torch.cuda.get_device_properties(0).total_memory / 1e9
        _need = 24.0 if args.policy == "P9" else (
            22.0 if (args.policy in ("P7", "P8", "P8b") and args.foundation == "sam") else 0.0)
        if _need and _total_gb + 0.5 < _need:   # 0.5GB tolerance
            raise SystemExit(
                f"[run_one] REFUSING {args.policy} on {_torch.cuda.get_device_name(0)} "
                f"({_total_gb:.0f}GB < {_need:.0f}GB required). Route this cell to a larger GPU.")

    from medal_bench.runner.trajectory import read_jsonl

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    final_jsonl = os.path.join(args.out_dir, f"{args.dataset}__{args.policy}__s{args.seed}.jsonl")

    # Done-by-ROUND-COUNT idempotency (lease-race safe): if a COMPLETE final .jsonl
    # already exists, exit 0 without touching it (a truncated/short file is NOT done
    # and will be rerun). total_rounds is logged on every record. --force overrides.
    if not args.force and os.path.exists(final_jsonl):
        existing = read_jsonl(final_jsonl)
        if existing and len(existing) == existing[0].get("total_rounds", -1):
            print(f"[run_one] already complete ({len(existing)} rounds) -> skip "
                  f"(use --force to rerun): {final_jsonl}")
            return 0

    # Write rounds to a PER-PID partial and atomically rename to the final .jsonl only
    # on success, so two racing runs for the same cell never clobber a shared partial and
    # a just-finalized .jsonl is never truncated. A killed worker leaves only its own
    # *.partial.<pid>, so the cell stays re-runnable.
    out_jsonl = final_jsonl + f".partial.{os.getpid()}"
    if os.path.exists(out_jsonl):
        os.remove(out_jsonl)

    print(f"[run_one] dataset={args.dataset} policy={args.policy} seed={args.seed} "
          f"profile={args.profile} foundation={args.foundation}")
    adapter = _build_adapter(args.dataset, args.data_root)

    # pool_size = #train samples after split (so the budget % is over the
    # actual train pool, not the whole dataset)
    split = make_split(adapter, seed=args.seed)
    pool_size_full = len(split.train)
    prof = PROFILES[args.profile]
    pool_size = pool_size_full if prof.pool_cap is None else min(pool_size_full, prof.pool_cap)

    foundation_fn = None
    if args.foundation == "sam":
        from medal_bench.features.sam import make_sam_foundation_fn
        foundation_fn = make_sam_foundation_fn(
            cache_dir=args.foundation_cache,
            model_type=args.sam_model_type, checkpoint=args.sam_checkpoint,
        )
        print(f"[run_one] SAM features: model_type={args.sam_model_type} "
              f"checkpoint={args.sam_checkpoint or '(default/HF)'}")

    bc_kwargs = dict(
        profile_name=args.profile,
        policy_id=args.policy, policy_config={},
        dataset_name=args.dataset,
        seed=args.seed, out_jsonl=out_jsonl,
        device=args.device, foundation_features_fn=foundation_fn,
        num_classes=adapter.num_classes,
        preproc_cache_dir=os.path.join(
            os.path.dirname(DEFAULT_FOUND_CACHE), "preprocessed"),
    )
    cfg = build_run_config(pool_size=pool_size, **bc_kwargs)

    # M2 budget denominator: the fg-stratified cap can retain FEWER slices than
    # min(len(train), pool_cap) (the 1:1-balanced 5000-pool datasets resolve to ~2500),
    # so derive the TRUE accessible AL pool from the actual pool builder and rebuild the
    # budget grid on it. Mirrors run_full_supervised so fractions are honest.
    from medal_bench.runner.al_loop import _load_or_make_pool_indices, SplitView
    train_view = SplitView(adapter, split.train, "train")
    actual_AL_pool_N = len(_load_or_make_pool_indices(cfg, adapter.name, train_view))
    if actual_AL_pool_N != pool_size:
        cfg = build_run_config(pool_size=actual_AL_pool_N, **bc_kwargs)
    assert cfg.budget_plan[-1] <= actual_AL_pool_N, \
        f"budget plan max {cfg.budget_plan[-1]} > accessible AL pool {actual_AL_pool_N}"

    from dataclasses import replace
    if args.batch is not None:
        cfg.train = replace(cfg.train, batch_size=args.batch)
    if args.num_iters is not None:
        cfg.train = replace(cfg.train, num_iters=args.num_iters)
    if args.adaptive:
        _ad = {"adaptive_iters": True}
        if args.min_iters is not None:
            _ad["min_iters"] = args.min_iters
        if args.max_iters is not None:
            _ad["max_iters"] = args.max_iters
        cfg.train = replace(cfg.train, **_ad)
        print(f"[run_one] ADAPTIVE train: plateau, min_iters={cfg.train.min_iters} "
              f"max_iters={cfg.train.max_iters} window={cfg.train.plateau_window} "
              f"patience={cfg.train.plateau_patience} min_delta={cfg.train.plateau_min_delta}")
    if args.save_predictions or args.save_logits or args.defer_surface:
        cfg = replace(cfg, save_predictions=(args.save_predictions or args.save_logits or args.defer_surface),
                      save_logits=args.save_logits)
    if args.defer_surface:
        # empty set -> al_loop computes NO inline surface; surface_offline backfills HD95/ASSD
        # from the saved masks afterward (identical numbers, off the GPU loop).
        cfg = replace(cfg, surface_rounds=set())

    _last = cfg.budget_plan[-1]
    print(f"[run_one] budget denominator: full_train_N={pool_size_full} "
          f"requested_pool_cap={prof.pool_cap} actual_AL_pool_N={actual_AL_pool_N} "
          f"frac_of_AL_pool={_last / max(1, actual_AL_pool_N):.4f} "
          f"frac_of_full_train={_last / max(1, pool_size_full):.4f}")
    print(f"[run_one] budget plan (cumulative): {cfg.budget_plan}")

    try:
        recs = run_al(adapter, cfg)
        os.replace(out_jsonl, final_jsonl)   # atomic: cell is "done" only now
        last_metrics = recs[-1]["metrics"]
        print(f"[run_one] OK -> rounds={len(recs)}  last DSC_fg={last_metrics.get('mean_dsc_fg'):.4f}  "
              f"last HD95_fg={last_metrics.get('mean_hd95_fg', float('nan'))}  "
              f"wrote {final_jsonl}")
        return 0
    except Exception as e:
        print(f"[run_one] FAIL: {e}", file=sys.stderr)
        traceback.print_exc()
        # leave a .fail marker for the submit script to detect
        with open(final_jsonl + ".fail.txt", "w") as fh:
            fh.write(traceback.format_exc())
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
