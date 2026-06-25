"""frozen_v3 P8 fidelity: MIN_CLUSTER_SIZE filter (configurable + logged), round-robin
fill across clusters, graceful relax, and the P8c legacy ablation snapshot."""
from __future__ import annotations

import numpy as np

from medal_bench.policies import build, PolicyContext


def _ctx(pool, label=None, seed=0):
    feats = {"foundation_pool": pool.astype(np.float32)}
    if label is not None:
        feats["foundation_label"] = label.astype(np.float32)
    return PolicyContext(seed=seed, round_idx=0, num_classes=2, features=feats)


def test_min_cluster_size_excludes_small_clusters():
    rng = np.random.RandomState(0)
    big = np.array([0.0, 0.0]) + 0.01 * rng.randn(12, 2)   # one dense cluster (>5)
    mini = np.array([[20.0, 20.0], [20.1, 19.9]])          # size-2 satellite
    pool = np.concatenate([big, mini], axis=0)             # idx 12,13 = mini
    # default min_cluster_size=5 + k=1 -> pick from the big cluster, NOT the mini one.
    sel = build("P8").select(_ctx(pool), None, k=1)
    assert sel[0] < 12, f"picked a tiny-cluster point: {sel}"
    # disable the filter -> the param is wired (mini becomes eligible; logged size=0).
    ctx0 = _ctx(pool)
    build("P8", min_cluster_size=0).select(ctx0, None, k=1)
    assert ctx0.diagnostics_out["typiclust_min_cluster_size"] == 0


def test_filter_diagnostics_logged():
    rng = np.random.RandomState(1)
    ctx = _ctx(rng.randn(20, 8), rng.randn(4, 8))
    build("P8").select(ctx, None, k=3)
    for key in ("typiclust_min_cluster_size", "typiclust_num_filtered_clusters",
                "typiclust_num_singleton_clusters", "typiclust_min_cluster_relaxed"):
        assert key in ctx.diagnostics_out


def test_round_robin_spreads_and_relaxes_on_tiny_pool():
    # tiny all-small-cluster pool: filter would starve selection -> graceful relax fills k.
    rng = np.random.RandomState(2)
    pool = np.concatenate([np.array([0., 0.]) + 0.01 * rng.randn(3, 2),
                           np.array([9., 9.]) + 0.01 * rng.randn(3, 2)], axis=0)
    ctx = _ctx(pool, np.array([[0.0, 0.0]]))
    sel = build("P8").select(ctx, None, k=4)
    assert len(sel) == 4 and len(set(sel)) == 4
    assert ctx.diagnostics_out["typiclust_min_cluster_relaxed"] is True


def test_p8c_legacy_registered_and_distinct():
    p8, p8c = build("P8"), build("P8c")
    assert p8c.id == "P8c" and p8c.is_ablation is True
    assert "legacy" in p8c.name.lower()
    assert p8.name != p8c.name
