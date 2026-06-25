"""Full-supervised reference: train on 100% of the labeled train pool.

This is the upper-bound baseline for relative_DSC = DSC_AL / DSC_full and
budget_to_90/95_full. It reuses ``run_al`` with a single-element budget plan
(`[pool_size]`) so round 0's initial labeled set IS the whole train pool, then
evaluates on val (with HD95). Same backbone / resolution / schedule as the AL
runs (driven by the same profile).

Usage:
    python -m medal_bench.runner.run_full_supervised --dataset busi --seed 1000 \
        --profile bench512_dry --out-dir runs/full_sup [--pool-cap N] [--num-iters K]

For very large pools, pass --pool-cap / --num-iters and document the capped protocol
(do not silently omit the baseline).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import replace
from pathlib import Path

DEFAULT_DATA_ROOT = "/groups/echambe2/datasets/data"
DEFAULT_FOUND_CACHE = "/groups/echambe2/gmeng/MedAL-Agent/cache/foundation_features"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--profile", default="bench512_dry",
                    choices=["smoke", "pilot", "bench512_dry", "bench512", "bench512_v4"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pool-cap", type=int, default=None, help="cap full-sup train pool (capped protocol)")
    ap.add_argument("--num-iters", type=int, default=None, help="override training iters")
    args = ap.parse_args(argv)

    from medal_bench.data.adapters import DATASET_REGISTRY
    from medal_bench.profiles import PROFILES
    from medal_bench.runner.al_loop import run_al, RunConfig
    from medal_bench.runner.splits import make_split

    if args.dataset not in DATASET_REGISTRY:
        raise SystemExit(f"unknown dataset {args.dataset}")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_jsonl = os.path.join(args.out_dir, f"{args.dataset}__FULL__s{args.seed}.jsonl")

    adapter = DATASET_REGISTRY[args.dataset](args.data_root)
    prof = PROFILES[args.profile]
    split = make_split(adapter, seed=args.seed)
    pool_size = len(split.train)
    cap = args.pool_cap if args.pool_cap is not None else prof.pool_cap
    if cap is not None:
        pool_size = min(pool_size, cap)

    train_cfg = prof.train if args.num_iters is None else replace(prof.train, num_iters=args.num_iters)

    from medal_bench.features.sam import make_sam_foundation_fn
    foundation_fn = make_sam_foundation_fn(cache_dir=DEFAULT_FOUND_CACHE, model_type="vit_h")

    cfg = RunConfig(
        policy_id="P0", policy_config={},          # no selection happens (single round)
        dataset_name=args.dataset, seed=args.seed,
        budget_plan=[pool_size],                    # provisional; corrected below
        train=train_cfg, out_jsonl=out_jsonl, device=args.device,
        pool_cap=cap, val_cap=prof.val_cap,
        foundation_features_fn=foundation_fn,
        surface_rounds={0},                          # HD95/ASD on the single round
        preproc_cache_dir=os.path.join(os.path.dirname(DEFAULT_FOUND_CACHE), "preprocessed"),
        stratify_pool_by_fg=prof.stratify_pool_by_fg, stratify_fg_ratio=prof.stratify_fg_ratio,
    )
    # The fg-stratified cap can yield FEWER slices than min(len(train), cap) — e.g. the
    # 1:1-balanced 5000-pool datasets resolve to 2500. The true full-pool size must come
    # from the actual pool builder, or run_al's "init size == budget_plan[0]" assert fails.
    from medal_bench.runner.al_loop import _load_or_make_pool_indices, SplitView
    train_view = SplitView(adapter, split.train, "train")
    pool_size = len(_load_or_make_pool_indices(cfg, adapter.name, train_view))
    cfg = replace(cfg, budget_plan=[pool_size])
    print(f"[full_sup] {args.dataset} seed={args.seed} full-pool N={pool_size} profile={args.profile}")
    try:
        recs = run_al(adapter, cfg)
        m = recs[-1]["metrics"]
        summary = {
            "dataset": args.dataset, "seed": args.seed, "profile": args.profile,
            "full_pool_size": pool_size, "capped": cap is not None,
            "num_classes": adapter.num_classes,
            "dsc_full_fg": m.get("mean_dsc_fg"),
            "dsc_full_per_class": m.get("dsc_per_class"),
            "hd95_full_fg": m.get("mean_hd95_fg"),
        }
        json.dump(summary, open(os.path.join(args.out_dir, f"{args.dataset}__FULL__s{args.seed}.json"), "w"), indent=2)
        print(f"[full_sup] OK dsc_full_fg={summary['dsc_full_fg']:.4f} hd95_full_fg={summary['hd95_full_fg']}")
        return 0
    except Exception as e:
        print(f"[full_sup] FAIL: {e}", file=sys.stderr)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
