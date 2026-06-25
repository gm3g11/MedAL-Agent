"""P8 reference-faithful TypiClust tests (Section 8): density preference,
labeled-coverage priority, undercovered-before-covered, cluster-count logging,
unique selections, empty/duplicate robustness, and P8 vs P8b distinctness."""
from __future__ import annotations

import numpy as np

from medal_bench.policies import build, PolicyContext


def _ctx(pool_feats, label_feats=None, seed=0):
    feats = {"foundation_pool": pool_feats.astype(np.float32)}
    if label_feats is not None:
        feats["foundation_label"] = label_feats.astype(np.float32)
    return PolicyContext(seed=seed, round_idx=0, num_classes=2, features=feats)


def test_typiclust_reference_toy_density():
    rng = np.random.RandomState(0)
    dense = np.array([1.0, 1.0]) + 0.01 * rng.randn(10, 2)
    outlier = np.array([[ -5.0, 5.0]])
    pool = np.concatenate([dense, outlier], axis=0)   # idx 10 = outlier
    sel = build("P8").select(_ctx(pool), None, k=1)
    assert sel[0] != 10, f"TypiClust picked the outlier: {sel}"


def test_typiclust_labeled_coverage_affects_selection():
    # cluster A around (0,0) is covered by labeled; cluster B around (10,10) is
    # uncovered. With one query, TypiClust must pick from the uncovered B.
    A = np.array([[0., 0.], [0.2, -0.1], [-0.1, 0.2]])
    B = np.array([[10., 10.], [10.2, 9.9], [9.9, 10.1]])
    pool = np.concatenate([A, B], axis=0)             # pool idx 3,4,5 = B
    label = A.copy()                                  # labeled sit in A
    sel = build("P8").select(_ctx(pool, label), None, k=1)
    assert sel[0] in (3, 4, 5), f"should query uncovered cluster B, got {sel}"


def test_typiclust_undercovered_clusters_before_redundant():
    A = np.array([[0., 0.], [0.1, 0.1], [-0.1, 0.0]])
    B = np.array([[10., 10.], [10.1, 10.0]])
    C = np.array([[ -10., -10.], [-10.1, -9.9]])
    pool = np.concatenate([A, B, C], axis=0)          # B=idx3,4  C=idx5,6
    label = A.copy()                                  # only A covered
    sel = build("P8").select(_ctx(pool, label), None, k=2)
    assert set(sel).issubset({3, 4, 5, 6}), f"both picks should be from uncovered B/C: {sel}"
    assert len(set(sel)) == 2


def test_typiclust_cluster_count_policy_logged():
    rng = np.random.RandomState(1)
    ctx = _ctx(rng.randn(20, 8), rng.randn(4, 8))
    build("P8").select(ctx, None, k=3)
    assert ctx.diagnostics_out.get("typiclust_cluster_rule") == "labeled_plus_budget"
    assert isinstance(ctx.diagnostics_out.get("typiclust_n_clusters"), int)
    assert "typiclust_selected_clusters" in ctx.diagnostics_out


def test_typiclust_handles_duplicate_points_returns_unique():
    # duplicate points trigger sklearn empty-cluster warnings; must still return k unique.
    pool = np.tile(np.array([[1.0, 2.0]]), (12, 1)) + np.zeros((12, 2))
    pool[:3] += 5.0
    sel = build("P8").select(_ctx(pool, np.array([[0.0, 0.0]])), None, k=3)
    assert len(sel) == 3 and len(set(sel)) == 3


def test_p8_main_and_p8b_distinct():
    p8, p8b = build("P8"), build("P8b")
    assert p8.id == "P8" and p8b.id == "P8b"
    assert "typiclust" in p8.name.lower()
    assert "densityclust" in p8b.name.lower()
