"""P6 - Selective Uncertainty AL.

Faithful adaptation of Ma et al., "Breaking the Barrier: Selective
Uncertainty-based Active Learning for Medical Image Segmentation"
(ICASSP 2024; arXiv:2401.16298). Reference impl: HelenMa9998/Selective_Uncertainty_AL.

Motivation (from the paper): conventional uncertainty AL sums/averages a
per-pixel metric over ALL pixels. In imbalanced segmentation this drowns the
target (lesion/organ) signal in background and is redundant. Selective
Uncertainty instead FILTERS pixels to two prediction-derived regions and
scores each region separately, then DIVERSIFIES the batch by interleaving the
two rankings.

Per the reference (query_strategies/entropy_sampling.py), for the binary case:
  - target pixels   : predicted foreground prob >= 0.1
  - boundary pixels : |p - 0.5| <= 0.1   (low-confidence, near the boundary)
  - per-image score : MEAN per-pixel uncertainty inside each region (nan->0)
  - selection       : alternate between the target ranking and the boundary
                      ranking, appending unique sample ids until k.

Multi-class adaptation (documented, since the paper is binary):
  - foreground probability  = 1 - P(class 0 == background)
  - target region           = fg_prob >= tau_target            (default 0.1)
  - boundary region         = (top1_prob - top2_prob) <= delta_boundary
                              (default 0.2; for binary |p-0.5|<=0.1 <=> margin<=0.2)
  - per-pixel uncertainty   = predictive entropy (matches P1's score_per_pixel)

Uses only the per-round prediction cache (probs/argmax) — no extra forward
pass, no MC dropout, and NO ground-truth masks of unlabeled samples.

Diagnostics emitted:
  selu_target_frac          - mean fraction of pixels in the target region
  selu_boundary_frac        - mean fraction of pixels in the boundary region
  selu_score_target_mean    - mean per-image target-region uncertainty
  selu_score_boundary_mean  - mean per-image boundary-region uncertainty
"""
from __future__ import annotations

import numpy as np
import torch

from medal_bench.policies.base import Policy, PolicyContext
from medal_bench.policies.registry import register
from medal_bench.policies._helpers import score_per_pixel


def _masked_mean(score_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(N,H,W) score_map, (N,H,W) {0,1} mask -> (N,) mean over masked pixels.
    Empty mask -> 0.0 (matches the reference's nan_to_num)."""
    num = (score_map * mask).flatten(start_dim=1).sum(dim=1)
    den = mask.flatten(start_dim=1).sum(dim=1)
    return torch.where(den > 0, num / den.clamp(min=1.0), torch.zeros_like(num))


def _intersect_valid_(mask: torch.Tensor, valid_bboxes) -> torch.Tensor:
    """Zero ``mask`` (N,H,W) outside each sample's valid (un-padded) rectangle,
    in place. P6's target/boundary masks are prediction-derived and only SOFTLY
    exclude pad; this makes the exclusion a hard geometric guarantee. Sliced
    per-sample to avoid allocating an (N,H,W) valid mask. No-op when valid_bboxes
    is None (no padding)."""
    if valid_bboxes is None:
        return mask
    N, H, W = mask.shape
    for i in range(N):
        y0, x0, h, w = (int(v) for v in valid_bboxes[i])
        if y0 == 0 and x0 == 0 and h == H and w == W:
            continue
        mask[i, :y0, :] = 0; mask[i, y0 + h:, :] = 0
        mask[i, :, :x0] = 0; mask[i, :, x0 + w:] = 0
    return mask


def _interleave_rankings(rank_a: list[int], rank_b: list[int], k: int) -> list[int]:
    """Alternate a, b, a, b, ... appending unique indices until k are chosen.
    Matches the reference's target/boundary ranking interleave."""
    selected: list[int] = []
    seen: set[int] = set()
    ia = ib = 0
    take_a = True
    while len(selected) < k and (ia < len(rank_a) or ib < len(rank_b)):
        if take_a and ia < len(rank_a):
            idx = rank_a[ia]; ia += 1
            if idx not in seen:
                seen.add(idx); selected.append(idx)
        elif (not take_a) and ib < len(rank_b):
            idx = rank_b[ib]; ib += 1
            if idx not in seen:
                seen.add(idx); selected.append(idx)
        take_a = not take_a
    return selected[:k]


@register("P6")
class SelectiveUncertaintyAL(Policy):
    name = "Selective Uncertainty"
    needs_pred_cache = True
    needs_pred_cache_probs = False   # streaming reduction; full (N,C,H,W) never materialized

    def __init__(self, tau_target: float = 0.1, delta_boundary: float = 0.2,
                 sub: str = "normalized_entropy", **config):
        super().__init__(tau_target=tau_target, delta_boundary=delta_boundary,
                         sub=sub, **config)
        self.tau_target = float(tau_target)
        self.delta_boundary = float(delta_boundary)
        self.sub = sub
        self._s_target: np.ndarray | None = None
        self._s_boundary: np.ndarray | None = None

    @torch.no_grad()
    def per_batch_reduce(self, probs_b, argmax_b, valid_b, offset):
        """Per-batch (B,) reductions. The two region scores and the diagnostic
        sums/counts are all per-sample (or per-pixel) independent, so concatenating
        batches reproduces the whole-pool result exactly."""
        unc = score_per_pixel(probs_b, self.sub)               # (B, H, W)

        fg_prob = 1.0 - probs_b[:, 0]                          # (B, H, W) P(any foreground)
        target_mask = (fg_prob >= self.tau_target).float()

        top2 = torch.topk(probs_b, 2, dim=1).values            # (B, 2, H, W)
        margin = top2[:, 0] - top2[:, 1]                       # (B, H, W)
        boundary_mask = (margin <= self.delta_boundary).float()

        _intersect_valid_(target_mask, valid_b)
        _intersect_valid_(boundary_mask, valid_b)

        s_target = _masked_mean(unc, target_mask)              # (B,)
        s_boundary = _masked_mean(unc, boundary_mask)          # (B,)
        # per-pixel mask sum so the whole-pool mask fraction recombines exactly
        px = float(target_mask.shape[1] * target_mask.shape[2])
        tgt_frac = target_mask.flatten(start_dim=1).sum(dim=1) / px      # (B,)
        bnd_frac = boundary_mask.flatten(start_dim=1).sum(dim=1) / px    # (B,)
        return {"s_target": s_target, "s_boundary": s_boundary,
                "tgt_frac": tgt_frac, "bnd_frac": bnd_frac}

    def finalize_score(self, accum, ctx):
        s_target = accum["s_target"]                           # (N,)
        s_boundary = accum["s_boundary"]                       # (N,)
        self._s_target = s_target.cpu().numpy()
        self._s_boundary = s_boundary.cpu().numpy()

        ctx.diagnostics_out["selu_valid_intersected"] = bool(ctx.valid_bboxes is not None)
        ctx.diagnostics_out["selu_target_frac"] = float(accum["tgt_frac"].mean())
        ctx.diagnostics_out["selu_boundary_frac"] = float(accum["bnd_frac"].mean())
        ctx.diagnostics_out["selu_score_target_mean"] = float(s_target.mean())
        ctx.diagnostics_out["selu_score_boundary_mean"] = float(s_boundary.mean())

        # returned per-image score (for the trajectory's selected_scores log);
        # the actual batch selection interleaves the two component rankings.
        return 0.5 * (s_target + s_boundary)

    @torch.no_grad()
    def score(self, ctx: PolicyContext):
        accum = ctx.streamed_reduce
        if accum is None:
            accum = self.per_batch_reduce(
                ctx.pred_cache.probs, ctx.pred_cache.argmax, ctx.valid_bboxes, 0)
        return self.finalize_score(accum, ctx)

    def select(self, ctx: PolicyContext, scores, k: int) -> list[int]:
        if self._s_target is None or self._s_boundary is None:
            # score() must run first; fall back to top-k on the combined score
            s = scores.detach().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores)
            return np.argsort(s, kind="stable")[::-1][:k].tolist()
        rank_t = np.argsort(self._s_target, kind="stable")[::-1].tolist()    # high target-uncertainty first
        rank_b = np.argsort(self._s_boundary, kind="stable")[::-1].tolist()  # high boundary-uncertainty first
        return _interleave_rankings(rank_t, rank_b, k)
