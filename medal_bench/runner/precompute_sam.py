"""Warm the SAM-H feature cache for Stage 1 (B7).

Precomputes vit_h features for each (dataset, seed)'s FULL active-learning pool at
the chosen resolution, so every later P7/P8 run is a pure cache hit and the heavy
ViT-H @1024 forward happens ONCE, up front (ideally on the A40s). The pool is built
exactly as run_al builds it (split -> train_view -> pool_cap + fg-stratify ->
_IndexedSubset at the profile's resolution), and the subset NAME matches run_al's
unlabeled/labeled subset name so the cache key (which embeds name + encoder +
preprocess-hash + input-size) lines up.

Usage (one process per A40 is fine; they share the on-disk cache):
    python -m medal_bench.runner.precompute_sam \
        --datasets mmwhs,btcv_synapse,... --seeds 1000,2000,3000 \
        --profile bench512_dry --device cuda:0
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

DEFAULT_DATA_ROOT = "/groups/echambe2/datasets/data"
DEFAULT_FOUND_CACHE = "/groups/echambe2/gmeng/MedAL-Agent/cache/foundation_features"


def _build_pool(adapter, seed: int, prof, cache_dir: str):
    """Reproduce run_al's pool_subset + the wrapping subset run_al extracts on."""
    from medal_bench.runner.al_loop import _IndexedSubset, _stratified_pool_cap
    from medal_bench.runner.splits import make_split, SplitView

    split = make_split(adapter, seed=seed)
    train_view = SplitView(adapter, split.train, "train")
    rng = np.random.RandomState(seed)
    idx = list(range(len(train_view)))
    if prof.pool_cap is not None and len(idx) > prof.pool_cap:
        if prof.stratify_pool_by_fg:
            idx = _stratified_pool_cap(train_view, idx, prof.pool_cap, rng, fg_ratio=prof.stratify_fg_ratio)
        else:
            idx = rng.choice(idx, size=prof.pool_cap, replace=False).tolist()
    pool_subset = _IndexedSubset(train_view, idx, prof.train.image_size,
                                 prof.train.aspect_preserve, cache_dir=cache_dir)
    # run_al extracts SAM on _IndexedSubset(pool_subset, ...) -> name "..._subset_subset"
    full = _IndexedSubset(pool_subset, list(range(len(pool_subset))),
                          prof.train.image_size, prof.train.aspect_preserve)
    return full


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True, help="comma-separated dataset ids")
    ap.add_argument("--seeds", default="1000,2000,3000")
    ap.add_argument("--profile", default="bench512_dry", choices=["smoke", "pilot", "bench512_dry", "bench512"])
    ap.add_argument("--sam-model-type", default="vit_h", choices=["vit_b", "vit_l", "vit_h"])
    ap.add_argument("--sam-checkpoint", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--foundation-cache", default=DEFAULT_FOUND_CACHE)
    args = ap.parse_args(argv)
    os.environ.setdefault("HF_HOME", "/groups/echambe2/gmeng/MedAL-Agent/cache/hf_hub")

    from medal_bench.data.adapters import DATASET_REGISTRY
    from medal_bench.profiles import PROFILES
    from medal_bench.features.sam import extract_sam_features, resolve_sam_spec

    prof = PROFILES[args.profile]
    spec = resolve_sam_spec(args.sam_model_type, checkpoint=args.sam_checkpoint)
    preproc_cache = os.path.join(os.path.dirname(args.foundation_cache), "preprocessed")
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    rows = []
    for ds_id in datasets:
        for seed in seeds:
            t0 = time.time()
            try:
                adapter = DATASET_REGISTRY[ds_id](args.data_root)
                pool = _build_pool(adapter, seed, prof, preproc_cache)
                feats = extract_sam_features(pool, cache_dir=args.foundation_cache,
                                             spec=spec, device=args.device)
                dt = time.time() - t0
                rows.append((ds_id, seed, len(pool), feats.shape, round(dt, 1), "OK"))
                print(f"[warm] {ds_id} s{seed}: {len(pool)} slices -> {feats.shape} in {dt:.1f}s")
            except Exception as e:
                rows.append((ds_id, seed, 0, None, round(time.time() - t0, 1), f"FAIL:{e}"))
                print(f"[warm] {ds_id} s{seed} FAIL: {e}")
    n_ok = sum(1 for r in rows if r[-1] == "OK")
    print(f"\n=== SAM-H warm: {n_ok}/{len(rows)} (dataset,seed) cells cached ===")
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
