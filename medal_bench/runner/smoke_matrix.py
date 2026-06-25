"""1-seed smoke matrix: 10 policies x N pilot datasets x 1 seed.

This is the gate the user mandated before launching the full 3-seed pilot:
verify that every (policy, dataset) tuple wires end-to-end through the AL
loop without crashing.

ROSE-1 is currently DEFERRED — its raw data is not on disk and requires a
credentialed download from the ROSE consortium. The adapter raises
FileNotFoundError when instantiated, so this matrix runs over the 5
available pilot datasets: ISIC2018, CVC-ClinicDB, BUSI, PROMISE12,
MSD07 Pancreas.

Usage:
    python -m medal_bench.runner.smoke_matrix \\
        --out-dir runs/smoke_matrix_v1 [--policies P0,P3,P8] [--datasets busi,cvc_clinicdb]
"""
from __future__ import annotations

import argparse
import os
import time
import traceback
from dataclasses import dataclass
from typing import Callable

from medal_bench.runner.al_loop import run_al, RunConfig, TrainConfig


DATA_ROOT_DEFAULT = "/groups/echambe2/datasets/data"


@dataclass
class _DsCfg:
    name: str
    factory: Callable
    pool_cap: int
    val_cap: int
    image_size: int


def _dataset_registry(data_root: str) -> dict[str, _DsCfg]:
    from medal_bench.data.adapters import DATASET_REGISTRY
    return {
        name: _DsCfg(name, (lambda f=factory: f(data_root)),
                     pool_cap=32, val_cap=8, image_size=128)
        for name, factory in DATASET_REGISTRY.items()
    }


ALL_POLICIES = ["P0","P1","P2","P3","P4","P5","P6","P7","P8","P9"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-root", default=DATA_ROOT_DEFAULT)
    ap.add_argument("--policies", default=",".join(ALL_POLICIES))
    ap.add_argument("--datasets", default="isic2018,cvc_clinicdb,busi,promise12,msd07_pancreas")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--num-iters", type=int, default=15)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--foundation", default="stub", choices=["stub", "sam"],
                    help="P7/P8 foundation features source.")
    ap.add_argument("--sam-model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    ap.add_argument("--sam-checkpoint", default=None)
    ap.add_argument("--foundation-cache", default="/groups/echambe2/gmeng/MedAL-Agent/cache/foundation_features")
    args = ap.parse_args(argv)

    foundation_fn = None
    if args.foundation == "sam":
        os.environ["HF_HOME"] = "/groups/echambe2/gmeng/MedAL-Agent/cache/hf_hub"
        os.environ["HF_HUB_CACHE"] = "/groups/echambe2/gmeng/MedAL-Agent/cache/hf_hub"
        from medal_bench.features.sam import make_sam_foundation_fn
        foundation_fn = make_sam_foundation_fn(
            cache_dir=args.foundation_cache,
            model_type=args.sam_model_type, checkpoint=args.sam_checkpoint,
        )

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "summary.txt")

    ds_reg = _dataset_registry(args.data_root)
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    results: list[tuple[str, str, bool, float, str]] = []  # (ds, pid, ok, sec, info)
    adapter_cache: dict = {}

    def _get_adapter(name: str):
        if name not in adapter_cache:
            adapter_cache[name] = ds_reg[name].factory()
        return adapter_cache[name]

    print(f"Smoke matrix: {len(datasets)} datasets x {len(policies)} policies = {len(datasets)*len(policies)} cells")
    t_start = time.time()

    for ds_name in datasets:
        dcfg = ds_reg[ds_name]
        try:
            ds = _get_adapter(ds_name)
        except Exception as e:
            print(f"[{ds_name}] adapter load failed: {e}")
            for pid in policies:
                results.append((ds_name, pid, False, 0.0, f"adapter-load: {e}"))
            continue

        for pid in policies:
            out_jsonl = os.path.join(args.out_dir, f"{ds_name}__{pid}__s{args.seed}.jsonl")
            if os.path.exists(out_jsonl):
                os.remove(out_jsonl)
            cfg = RunConfig(
                policy_id=pid, policy_config={},
                dataset_name=ds_name, seed=args.seed,
                budget_plan=[4, 8],
                train=TrainConfig(
                    num_iters=args.num_iters, batch_size=2,
                    image_size=dcfg.image_size,
                    features_per_stage=(8, 16, 32),
                ),
                foundation_features_fn=foundation_fn,
                out_jsonl=out_jsonl, device=args.device,
                pool_cap=dcfg.pool_cap, val_cap=dcfg.val_cap,
            )
            t0 = time.time()
            try:
                recs = run_al(ds, cfg)
                dt = time.time() - t0
                mfg = recs[-1]["metrics"]["mean_dsc_fg"]
                results.append((ds_name, pid, True, dt, f"mDSC_fg={mfg:.3f}"))
                print(f"  [{ds_name}/{pid}] OK {dt:.1f}s mDSC_fg={mfg:.3f}")
            except Exception as e:
                dt = time.time() - t0
                tb = traceback.format_exc()
                results.append((ds_name, pid, False, dt, str(e)))
                print(f"  [{ds_name}/{pid}] FAIL {dt:.1f}s -> {e}")
                with open(os.path.join(args.out_dir, f"{ds_name}__{pid}__s{args.seed}.fail.txt"), "w") as fh:
                    fh.write(tb)

    t_total = time.time() - t_start
    n_ok = sum(1 for r in results if r[2])
    print(f"\n=== {n_ok}/{len(results)} cells passed in {t_total:.1f}s ===")
    with open(summary_path, "w") as fh:
        fh.write(f"smoke matrix: {n_ok}/{len(results)} passed in {t_total:.1f}s\n")
        for ds_name, pid, ok, sec, info in results:
            fh.write(f"  {ds_name:18s} {pid:3s} {'OK ' if ok else 'FAIL'} {sec:6.1f}s  {info}\n")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
