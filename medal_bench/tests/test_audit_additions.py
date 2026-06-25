"""Audit-mandated tests (Part 5 of the P0–P9 correctness review, 2026-06-12).

Covers the minimum tests that were missing from test_policies.py:
  - budget increment counts (cumulative -> per-round deltas)
  - CoreSet k-center toy selection (farthest point)
  - entropy score monotonicity (uncertain > confident)
  - P5 entropy->coreset prefilter restriction (never select outside top 5k)
  - TypiClust prefers dense/typical points over outliers
  - SAM feature cache key separates encoder/model types (SAM-B vs SAM-H)
"""
from __future__ import annotations

import numpy as np
import torch

from medal_bench.policies import build, PolicyContext
from medal_bench.policies._helpers import kcenter_greedy
from medal_bench.runner.prediction_cache import PredictionCache


def _const_image_probs(fg_probs, h=2, w=2):
    """Build (N, 2, h, w) softmax probs where image i is constant [1-p, p]."""
    n = len(fg_probs)
    probs = torch.zeros(n, 2, h, w)
    for i, p in enumerate(fg_probs):
        probs[i, 0] = 1.0 - p
        probs[i, 1] = p
    argmax = torch.argmax(probs, dim=1)
    return probs, argmax


# ---- 1. budget increments -------------------------------------------------

def test_budget_increment_counts():
    from medal_bench.profiles import cumulative_budget_plan
    plan = cumulative_budget_plan(1000, [0.01, 0.02, 0.05, 0.10, 0.15, 0.20])
    assert plan == [10, 20, 50, 100, 150, 200], plan
    deltas = [plan[i + 1] - plan[i] for i in range(len(plan) - 1)]
    assert deltas == [10, 30, 50, 50, 50], deltas
    # tiny pool: clamps to [1, pool] and stays strictly increasing (no dup picks)
    tiny = cumulative_budget_plan(8, [0.01, 0.02, 0.05, 0.10, 0.15, 0.20])
    assert tiny == [1, 2, 3, 4, 5, 6], tiny
    assert len(set(tiny)) == len(tiny) and tiny[-1] <= 8


# ---- 2. CoreSet k-center toy ----------------------------------------------

def test_coreset_toy_selection():
    # helper directly: labeled at 0; unlabeled {1,2,10}; k=1 -> farthest = 10
    feats = np.array([[0.0], [1.0], [2.0], [10.0]])
    from sklearn.metrics import pairwise_distances
    dm = pairwise_distances(feats)
    new = kcenter_greedy(dm, init_idx=[0], k=1)
    assert new == [3], new  # global index 3 == value 10.0

    # through P3 (normalize=False so 1-D geometry survives)
    ctx = PolicyContext(
        seed=0, round_idx=0, num_classes=2,
        features={"task_unet_pool": np.array([[1.0], [2.0], [10.0]], np.float32),
                  "task_unet_label": np.array([[0.0]], np.float32)},
    )
    sel = build("P3", normalize=False).select(ctx, None, k=1)
    assert sel == [2], sel  # pool index 2 == value 10.0


# ---- 3. entropy score monotonicity ----------------------------------------

def test_entropy_score():
    probs, argmax = _const_image_probs([0.5, 0.99])  # img0 uncertain, img1 confident
    ctx = PolicyContext(seed=0, round_idx=0, num_classes=2,
                        pred_cache=PredictionCache(probs, argmax, ["a", "b"]))
    s = build("P1").score(ctx).cpu().numpy()
    assert s[0] > s[1], s
    assert s[1] < 0.1, f"confident prediction should score near 0: {s[1]}"


# ---- P1 normalized entropy (Section 3) ------------------------------------

def test_entropy_normalized_range():
    from medal_bench.policies._helpers import score_per_pixel
    rng = np.random.RandomState(0)
    probs = torch.softmax(torch.from_numpy(rng.randn(5, 4, 8, 8).astype(np.float32)), dim=1)
    hn = score_per_pixel(probs, "normalized_entropy")
    assert hn.min() >= -1e-6 and hn.max() <= 1.0 + 1e-4, (hn.min().item(), hn.max().item())


def test_entropy_formula_matches_manual():
    import math
    from medal_bench.policies._helpers import score_per_pixel
    vec = [0.7, 0.1, 0.1, 0.1]
    p = torch.tensor(vec).view(1, 4, 1, 1)
    hn = score_per_pixel(p, "normalized_entropy")[0, 0, 0].item()
    manual = -sum(pi * math.log2(pi) for pi in vec) / math.log2(4)
    assert abs(hn - manual) < 1e-5, (hn, manual)
    assert abs(score_per_pixel(torch.full((1, 4, 1, 1), 0.25), "normalized_entropy").item() - 1.0) < 1e-5
    assert score_per_pixel(torch.tensor([0.999, 0.0004, 0.0003, 0.0003]).view(1, 4, 1, 1),
                           "normalized_entropy").item() < 0.05


def test_entropy_binary_sigmoid_conversion():
    # Codebase uses 2-channel softmax for binary (probs[:,1]=P(fg) == the [1-p,p]
    # representation). For C=2, H_norm == binary entropy in bits.
    import math
    from medal_bench.policies._helpers import score_per_pixel
    p = 0.3
    hn = score_per_pixel(torch.tensor([1 - p, p]).view(1, 2, 1, 1), "normalized_entropy").item()
    manual = -((1 - p) * math.log2(1 - p) + p * math.log2(p))  # /log2(2)=1
    assert abs(hn - manual) < 1e-5, (hn, manual)


# ---- 4. P5 entropy->coreset prefilter restriction -------------------------

def test_entropy_coreset_prefilter():
    # 12 images, foreground prob strictly decreasing -> entropy strictly
    # decreasing in index, so the top-10 entropy candidates are indices 0..9.
    fg = [0.5 - 0.04 * i for i in range(12)]
    probs, argmax = _const_image_probs(fg)
    rng = np.random.RandomState(0)
    ctx = PolicyContext(
        seed=0, round_idx=0, num_classes=2,
        pred_cache=PredictionCache(probs, argmax, [str(i) for i in range(12)]),
        features={"task_unet_pool": rng.randn(12, 8).astype(np.float32),
                  "task_unet_label": rng.randn(3, 8).astype(np.float32)},
    )
    pol = build("P5")  # filter_ratio=5.0 -> keep top min(5k, n)=10 for k=2
    scores = pol.score(ctx)
    sel = pol.select(ctx, scores, k=2)
    assert len(sel) == 2 and len(set(sel)) == 2
    assert all(i < 10 for i in sel), f"P5 selected outside top-10 entropy pool: {sel}"


# ---- 5. TypiClust prefers dense/typical points ----------------------------

def test_typiclust_prefers_dense_points():
    rng = np.random.RandomState(0)
    dense = np.array([1.0, 1.0]) + 0.01 * rng.randn(10, 2)   # tight cluster
    outlier = np.array([[1.0, -1.0]])                         # opposite direction
    feats = np.concatenate([dense, outlier], axis=0).astype(np.float32)  # idx 10 = outlier
    ctx = PolicyContext(seed=0, round_idx=0, num_classes=2,
                        features={"foundation_pool": feats})
    sel = build("P8", m_neighbors=5).select(ctx, None, k=1)
    assert sel[0] != 10, f"TypiClust picked the outlier: {sel}"


# ---- 6. SAM cache key separates model types -------------------------------

def test_sam_cache_key_contains_model_type():
    from medal_bench.features.sam import _cache_path, SamPreprocessConfig, SamEncoderSpec
    cfg = SamPreprocessConfig()
    spec_b = SamEncoderSpec("vit_b", "hf", "facebook/sam-vit-base", None,
                            "facebook/sam-vit-base/vision_encoder")
    spec_h = SamEncoderSpec("vit_h", "original", None, "/x/sam_vit_h.pth",
                            "segment_anything/vit_h/image_encoder")
    p_b = _cache_path("/tmp/c", "busi", spec_b, cfg)
    p_h = _cache_path("/tmp/c", "busi", spec_h, cfg)
    assert "vit_h" in p_h.name, p_h.name
    assert p_b.name != p_h.name, "SAM-B and SAM-H must map to different cache files"


def test_sam_cache_key_separates_height():
    # frozen_v3: non-square inputs (H != W) must not collide on width alone.
    from medal_bench.features.sam import _cache_path, SamPreprocessConfig, SamEncoderSpec
    cfg = SamPreprocessConfig()
    spec = SamEncoderSpec("vit_h", "original", None, "/x/sam_vit_h.pth",
                          "segment_anything/vit_h/image_encoder")
    p1 = _cache_path("/tmp/c", "busi", spec, cfg, in_hw=(128, 256))
    p2 = _cache_path("/tmp/c", "busi", spec, cfg, in_hw=(256, 128))
    assert p1.name != p2.name, "H x W must both be in the cache key"


def test_sam_cache_checkpoint_mismatch_recomputes(tmp_path):
    # frozen_v3: a different checkpoint for the same encoder_id must invalidate the cache.
    from medal_bench.features.sam import (
        _cache_path, _read_cache, _write_cache, SamPreprocessConfig, SamEncoderSpec)
    cfg = SamPreprocessConfig()
    spec_a = SamEncoderSpec("vit_h", "original", None, "/a.pth",
                            "segment_anything/vit_h/image_encoder")
    spec_b = SamEncoderSpec("vit_h", "original", None, "/b.pth",
                            "segment_anything/vit_h/image_encoder")
    path = _cache_path(str(tmp_path), "busi", spec_a, cfg)
    _write_cache(path, spec_a, cfg, {"s0": __import__("numpy").zeros(256, dtype="float32")})
    assert _read_cache(path, spec_a, cfg) != {}        # same checkpoint -> hit
    assert _read_cache(path, spec_b, cfg) == {}        # different checkpoint -> recompute

