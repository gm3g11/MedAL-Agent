"""Serially PRE-WARM the preproc-array disk cache for one or more datasets.

When many Stage-2 cells for a big dataset launch in parallel and ALL cache-miss,
they each rebuild the multi-GB resized pool concurrently -> RAM spike + OOM
("rebuild storm"). This script builds the SAME ``_IndexedSubset`` pool + val
cache ONCE (serially) per (dataset, seed, profile) so every later cell READS the
cache (fast, low RAM). It changes no accuracy/metric behaviour -- it only writes
the identical .npz files ``run_al`` would write on a cold miss.

Key match is guaranteed BY CONSTRUCTION: it builds the exact same RunConfig a
cell builds (``build_run_config`` with the run_one kwargs) and runs the same
split / pool-builder / val-selection / ``_IndexedSubset`` code path, so the
content-hash filename is identical. The cache key is policy-INDEPENDENT (it
depends only on dataset + image_size + aspect_preserve + PREPROC_VERSION + the
pool/val sample_ids), so a single pre-warm is shared by all 10 policies of a
(dataset, seed, profile).

Usage:
    python -m medal_bench.runner.prewarm_cache \\
        --datasets busi,isic2018 --profile bench512_v4 --seed 1000
"""
from __future__ import annotations

import argparse
import os

DEFAULT_DATA_ROOT = "/groups/echambe2/datasets/data"
DEFAULT_CACHE_DIR = "/groups/echambe2/gmeng/MedAL-Agent/cache/preprocessed"


def prewarm_one(dataset: str, profile: str, seed: int, data_root: str,
                cache_dir: str) -> None:
    """Build (or reuse) the train-pool + val preproc cache for one dataset."""
    from medal_bench.data.adapters import DATASET_REGISTRY
    from medal_bench.profiles import build_run_config, PROFILES
    from medal_bench.runner.al_loop import (
        _IndexedSubset, _load_or_make_pool_indices, SplitView,
    )
    from medal_bench.runner.seeds import seed_all
    from medal_bench.runner.splits import make_split

    if dataset not in DATASET_REGISTRY:
        raise SystemExit(f"unknown dataset {dataset}; choose from {sorted(DATASET_REGISTRY)}")
    adapter = DATASET_REGISTRY[dataset](data_root)
    prof = PROFILES[profile]

    # Mirror run_one: pool_size = post-split train size clamped to pool_cap, used
    # only to build budget_plan (NOT in the cache key). The cfg fields the cache
    # key DOES depend on (pool_cap, stratify_*, val_cap, image_size, aspect) come
    # from the profile and are policy-independent.
    split = make_split(adapter, seed=seed)
    pool_size_full = len(split.train)
    pool_size = pool_size_full if prof.pool_cap is None else min(pool_size_full, prof.pool_cap)
    cfg = build_run_config(
        profile_name=profile, policy_id="P0", policy_config={},
        dataset_name=dataset, pool_size=pool_size, seed=seed,
        out_jsonl="/dev/null", num_classes=adapter.num_classes,
        preproc_cache_dir=cache_dir,
    )

    # Reproduce the run_al preproc prologue EXACTLY (al_loop.run_al lines ~403-422):
    # same seed_all -> split -> views -> pool indices (own RNG) -> val selection
    # (main RNG) -> _IndexedSubset. _load_or_make_pool_indices uses a dedicated RNG
    # so val selection below is unaffected by it -- identical to run_al's order.
    seed_all(cfg.seed)
    train_view = SplitView(adapter, split.train, "train")
    val_view = SplitView(adapter, split.val, "val")

    import numpy as np
    rng = np.random.RandomState(cfg.seed)
    train_indices = _load_or_make_pool_indices(cfg, adapter.name, train_view)
    val_indices = list(range(len(val_view)))
    if cfg.val_cap is not None and len(val_indices) > cfg.val_cap:
        val_indices = rng.choice(val_indices, size=cfg.val_cap, replace=False).tolist()

    sz = cfg.train.image_size
    print(f"[prewarm] {dataset}: profile={profile} seed={seed} sz={sz} "
          f"aspect_preserve={cfg.train.aspect_preserve} "
          f"pool_N={len(train_indices)} val_N={len(val_indices)}")

    for tag, view, indices in (("train", train_view, train_indices),
                               ("val", val_view, val_indices)):
        cpath = _cache_path(view.name, sz, cfg.train.aspect_preserve, indices,
                            view, cache_dir)
        existed = os.path.exists(cpath)
        # _IndexedSubset writes the cache as a side effect on a miss; on a hit it
        # only reads. Either way it leaves the file present with the right key.
        sub = _IndexedSubset(view, indices, sz, cfg.train.aspect_preserve,
                             cache_dir=cache_dir)
        verb = "REUSE" if (existed or sub.cache_status == "hit") else "CREATE"
        print(f"[prewarm]   {tag}: {verb} ({sub.cache_status}) {cpath}")


def _cache_path(base_name, image_size, aspect_preserve, indices, view, cache_dir):
    """Compute the cache filename for logging WITHOUT building the subset.

    Uses the IDENTICAL key recipe as _IndexedSubset (al_loop): sha256 over
    base.name | image_size | aspect_preserve | PREPROC_VERSION | join(sample_ids
    in index order), truncated to 16 hex chars."""
    import hashlib
    from medal_bench.runner.al_loop import _IndexedSubset
    sids = view.sample_ids()
    ids = [sids[i] for i in indices]
    key = hashlib.sha256(
        f"{base_name}|{image_size}|{aspect_preserve}|{_IndexedSubset.PREPROC_VERSION}|"
        f"{','.join(ids)}".encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"{base_name}__sz{image_size}__{key}.npz")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", required=True,
                    help="comma-separated dataset names (e.g. busi,isic2018)")
    ap.add_argument("--profile", default="bench512_v4", choices=list(__import__(
        "medal_bench.profiles", fromlist=["PROFILES"]).PROFILES))
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    args = ap.parse_args(argv)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for ds in datasets:
        prewarm_one(ds, args.profile, args.seed, args.data_root, args.cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
