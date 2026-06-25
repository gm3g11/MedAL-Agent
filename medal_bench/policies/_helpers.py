"""Shared helpers for the v1 policies.

The four helper families:
  - score_per_pixel(probs, sub): lc | margin | entropy
  - aggregate(score_map, argmax, mode, ...): full | foreground | boundary
        with INDEPENDENT empty-mask fallbacks and per-mode counters
  - compute_class_weights(argmax, num_classes, ...): hard-class P4 weights
        with include_background, bg_weight_cap, and a normalized-ratio cap
  - kcenter_greedy / kmeans_plusplus / coverage_after_filter: sampling ops

All helpers are pure-functions (no policy state). Each policy composes them.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# per-pixel score (entropy, lc, margin)
# ---------------------------------------------------------------------------

def score_per_pixel(probs: torch.Tensor, sub: str) -> torch.Tensor:
    """probs: (N, C, H, W) softmaxed. Returns (N, H, W); larger == more uncertain."""
    if sub == "normalized_entropy":
        # H_norm = -sum_c p_c log p_c / log C  in [0, 1]. Log base cancels in the
        # ratio; eps guards log(0). For C=1 (degenerate) return raw H (==0).
        C = probs.shape[1]
        H = torch.sum(-probs * torch.log2(probs + 1e-12), dim=1)   # (N,H,W) bits
        return H / math.log2(C) if C > 1 else H
    if sub == "entropy":
        # legacy: mean-over-classes (= H/C); monotonic with H so ranking matches
        # normalized_entropy. Kept for back-compat / ablations.
        return torch.mean(-probs * torch.log2(probs + 1e-12), dim=1)
    if sub == "lc":
        return -probs.max(dim=1).values
    if sub == "margin":
        top2 = torch.topk(probs, 2, dim=1).values
        return -(top2[:, 0] - top2[:, 1])
    raise ValueError(f"unknown sub: {sub!r}")


# ---------------------------------------------------------------------------
# aggregation (full / foreground / boundary)
# ---------------------------------------------------------------------------

@dataclass
class FallbackCounters:
    fg_fallback_count: int = 0
    boundary_fallback_count: int = 0


def aggregate(
    score_map: torch.Tensor,
    argmax: torch.Tensor,
    mode: Optional[str],
    *,
    boundary_band_px: int = 3,
    empty_fg_tau: float = 0.005,
    empty_band_tau: float = 0.005,
    counters: Optional[FallbackCounters] = None,
    valid_bboxes: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Aggregate (N, H, W) score_map -> (N,) per-sample scalar.

    ``mode`` is one of {full, valid, foreground, boundary, None}; None == full.
    ``valid`` means over each sample's un-padded rectangle (``valid_bboxes`` is an
    (N,4) (y0,x0,h,w) array, or None -> full canvas); it stops letterbox-pad pixels
    from driving the query score. Foreground and boundary each have their own
    (independent) trigger and counter; on fallback that sample uses the full mean
    and the corresponding counter is incremented.
    """
    if mode is None or mode == "full":
        return _apply_full(score_map)
    if mode == "valid":
        return _apply_valid(score_map, valid_bboxes)
    if mode == "foreground":
        return _apply_foreground(score_map, argmax, empty_fg_tau, counters)
    if mode == "boundary":
        return _apply_boundary(score_map, argmax, boundary_band_px, empty_band_tau, counters)
    if mode == "foreground_boundary":
        out = _apply_foreground(score_map, argmax, empty_fg_tau, counters)
        out2 = _apply_boundary(score_map, argmax, boundary_band_px, empty_band_tau, counters)
        return 0.5 * (out + out2)
    raise ValueError(f"unknown aggregation mode: {mode!r}")


def _apply_full(score_map: torch.Tensor) -> torch.Tensor:
    return score_map.flatten(start_dim=1).mean(dim=1)


def _apply_valid(score_map: torch.Tensor, valid_bboxes: Optional[np.ndarray]) -> torch.Tensor:
    """Per-sample mean over the valid (un-padded) rectangle. None bboxes, or a bbox
    covering the whole canvas, fall back to the full-canvas mean (a no-op when there
    is no padding). Sliced per-sample (O(N)) rather than materializing an (N,H,W)
    mask, which would be hundreds of MB for a large pool."""
    if valid_bboxes is None:
        return _apply_full(score_map)
    N, H, W = score_map.shape
    out = torch.empty(N, dtype=score_map.dtype, device=score_map.device)
    for i in range(N):
        y0, x0, h, w = (int(v) for v in valid_bboxes[i])
        if y0 == 0 and x0 == 0 and h == H and w == W:
            out[i] = score_map[i].mean()
        else:
            region = score_map[i, y0:y0 + h, x0:x0 + w]
            out[i] = region.mean() if region.numel() > 0 else score_map[i].mean()
    return out


def _apply_foreground(score_map, argmax, empty_fg_tau, counters):
    N, H, W = argmax.shape
    fg_mask = (argmax > 0).float()
    fg_pixels = fg_mask.flatten(start_dim=1).sum(dim=1)
    total = float(H * W)
    fg_frac = fg_pixels / total
    fallback = fg_frac < empty_fg_tau
    masked_sum = (score_map * fg_mask).flatten(start_dim=1).sum(dim=1)
    masked_mean = torch.where(
        fg_pixels > 0,
        masked_sum / fg_pixels.clamp(min=1.0),
        torch.zeros_like(masked_sum),
    )
    full_mean = _apply_full(score_map)
    out = torch.where(fallback, full_mean, masked_mean)
    if counters is not None:
        counters.fg_fallback_count += int(fallback.sum().item())
    return out


def _apply_boundary(score_map, argmax, band_px, empty_band_tau, counters):
    from scipy.ndimage import binary_dilation, binary_erosion
    N, H, W = argmax.shape
    total = float(H * W)
    fg_np = (argmax > 0).cpu().numpy().astype(bool)
    band_np = np.empty_like(fg_np)
    for i in range(N):
        dil = binary_dilation(fg_np[i], iterations=band_px)
        ero = binary_erosion(fg_np[i], iterations=band_px)
        band_np[i] = dil & ~ero
    band_mask = torch.from_numpy(band_np.astype(np.float32)).to(score_map.device)
    band_pixels = band_mask.flatten(start_dim=1).sum(dim=1)
    band_frac = band_pixels / total
    fallback = band_frac < empty_band_tau
    masked_sum = (score_map * band_mask).flatten(start_dim=1).sum(dim=1)
    masked_mean = torch.where(
        band_pixels > 0,
        masked_sum / band_pixels.clamp(min=1.0),
        torch.zeros_like(masked_sum),
    )
    full_mean = _apply_full(score_map)
    out = torch.where(fallback, full_mean, masked_mean)
    if counters is not None:
        counters.boundary_fallback_count += int(fallback.sum().item())
    return out


# ---------------------------------------------------------------------------
# P4 class-frequency weights
# ---------------------------------------------------------------------------

def compute_class_weights(
    argmax: torch.Tensor,
    num_classes: int,
    *,
    include_background: bool = False,
    bg_weight_cap: float = 1.0,
    rebal_max_weight_ratio: float = 100.0,
    eps: float = 1e-6,
    min_pred_pixels_per_class: int = 5,
) -> torch.Tensor:
    """Per-class weights from predicted-class freqs on the unlabeled pool.

    ``argmax``: (N, H, W) int predicted class per pixel.

    Pipeline:
      1. Count predicted pixels per class.
      2. Drop classes with count < min_pred_pixels_per_class (speckle filter).
      3. weight = 1 / max(freq, eps).
      4. Normalize sum(weights[fg-classes]) == K (so uniform freqs -> w==1).
      5. Background weight: capped at bg_weight_cap (default 1.0).
         If include_background=False: background weight is forced to 0 (the
         caller will skip multiplying entropy on background pixels).
      6. Ratio cap: max(w_fg)/min(w_fg) <= rebal_max_weight_ratio.
    """
    argmax = argmax.long()
    counts = torch.bincount(argmax.view(-1), minlength=num_classes).float()  # (C,)
    # speckle filter
    counts = torch.where(counts < min_pred_pixels_per_class, torch.zeros_like(counts), counts)
    total = counts.sum().clamp(min=1.0)
    freq = counts / total
    w = 1.0 / torch.clamp(freq, min=eps)

    # split background vs foreground for separate handling
    fg_w = w[1:].clone()
    K = float(num_classes - 1)
    # normalize so sum(fg) == K (uniform -> 1)
    if fg_w.sum() > 0:
        fg_w = fg_w * (K / fg_w.sum().clamp(min=eps))
    # ratio cap
    fg_min = fg_w.min().clamp(min=1e-12)
    upper = fg_min * rebal_max_weight_ratio
    fg_w = torch.minimum(fg_w, upper)
    # re-normalize sum to K
    if fg_w.sum() > 0:
        fg_w = fg_w * (K / fg_w.sum().clamp(min=eps))

    w_out = torch.zeros(num_classes, dtype=torch.float32)
    if include_background:
        # cap background, keep separate
        w_out[0] = min(float(w[0]), bg_weight_cap)
    else:
        w_out[0] = 0.0
    w_out[1:] = fg_w
    return w_out


def apply_class_weights(score_map: torch.Tensor, argmax: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Multiply each pixel's score by the weight of its predicted class."""
    w = weights.to(score_map.device)
    return score_map * w[argmax.long()]


# ---------------------------------------------------------------------------
# sampling: top-k, k-center, k-means++, coverage-after-filter
# ---------------------------------------------------------------------------

def topk_indices(per_sample_score: torch.Tensor, k: int) -> List[int]:
    """Stable descending sort, take top-k indices."""
    scores = per_sample_score.detach().cpu().numpy()
    order = np.argsort(scores, kind="stable")[::-1]
    return order[:k].tolist()


def kcenter_greedy(dist_mat: np.ndarray, init_idx: Sequence[int], k: int) -> List[int]:
    """k-center greedy / farthest-first. Returns NEWLY selected indices
    (does not include init_idx). dist_mat shape (M, M) where M = N_labeled + N_pool.
    """
    # k-center is seeded from the labeled set; with an empty init the loop would
    # break on the first iteration (the labeled-column slice is empty) and silently
    # return [] (under-selection). Fail loudly so a future cold-start config is caught.
    assert len(init_idx) > 0, "kcenter_greedy needs a non-empty init/labeled set"
    M = dist_mat.shape[0]
    labeled = np.zeros(M, dtype=bool)
    labeled[list(init_idx)] = True
    new = []
    for _ in range(k):
        mat = dist_mat[~labeled, :][:, labeled]
        if mat.size == 0:
            break
        mat_min = mat.min(axis=1)
        loc = int(mat_min.argmax())
        chosen = np.arange(M)[~labeled][loc]
        labeled[chosen] = True
        new.append(int(chosen))
    return new


def kmeanspp_indices(X: np.ndarray, k: int, random_state: int = 0) -> List[int]:
    """sklearn k-means++ over feature matrix X (N, D). Seeded."""
    if k <= 0:                       # budget-exhausted round: nothing to pick (sklearn raises on k<=0)
        return []
    from sklearn.cluster import kmeans_plusplus
    _, idx = kmeans_plusplus(X=X, n_clusters=k, random_state=random_state)
    return [int(i) for i in idx]


def coverage_after_filter_indices(
    per_sample_score: torch.Tensor,
    pool_features: np.ndarray,
    label_features: np.ndarray,
    k: int,
    filter_ratio: float,
    metric: str,
) -> List[int]:
    """Top (filter_ratio * k) by score (clamped to pool size), then k-center."""
    from sklearn.metrics import pairwise_distances
    n_pool = pool_features.shape[0]
    keep = min(max(int(np.ceil(filter_ratio * k)), k), n_pool)
    scores = per_sample_score.detach().cpu().numpy()
    order = np.argsort(scores, kind="stable")[::-1]
    keep_idx = order[:keep]
    pool_filtered = pool_features[keep_idx]
    n_label = label_features.shape[0]
    all_feats = np.concatenate([label_features, pool_filtered], axis=0)
    dist_mat = pairwise_distances(all_feats, metric=metric)
    new_in_filtered = kcenter_greedy(dist_mat, init_idx=range(n_label), k=k)
    # map back to original pool indices
    return [int(keep_idx[i - n_label]) for i in new_in_filtered if (i - n_label) >= 0]
