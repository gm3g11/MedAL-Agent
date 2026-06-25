"""P1 - Normalized Entropy. Per-pixel normalized predictive entropy
H_norm = -sum_c p_c log p_c / log C  (in [0,1]), mean-aggregated over pixels, top-k."""
from __future__ import annotations
import numpy as np
import torch

from medal_bench.policies.base import Policy, PolicyContext
from medal_bench.policies.registry import register
from medal_bench.policies._helpers import score_per_pixel, aggregate, topk_indices


def _entropy_valid_batch(probs_b, argmax_b, valid_b):
    """Per-batch reduction for P1/P5: normalized-entropy per-pixel, aggregated over
    the valid (un-padded) region. Per-sample independent, so batching is exact."""
    sm = score_per_pixel(probs_b, "normalized_entropy")          # (B, H, W) in [0,1]
    return {"score": aggregate(sm, argmax_b, "valid", valid_bboxes=valid_b)}  # (B,)


@register("P1")
class EntropyFull(Policy):
    name = "Normalized Entropy"
    needs_pred_cache = True
    needs_pred_cache_probs = False   # uses streaming reduction; full (N,C,H,W) never materialized

    def per_batch_reduce(self, probs_b, argmax_b, valid_b, offset):
        return _entropy_valid_batch(probs_b, argmax_b, valid_b)

    def finalize_score(self, accum, ctx):
        return accum["score"]                                    # (N,)

    def score(self, ctx: PolicyContext):
        accum = ctx.streamed_reduce
        if accum is None:
            # fall back to whole-pool reduction (one batch == N); used by tests that
            # populate pred_cache.probs directly. Identical math to the streamed path.
            accum = self.per_batch_reduce(
                ctx.pred_cache.probs, ctx.pred_cache.argmax, ctx.valid_bboxes, 0)
        return self.finalize_score(accum, ctx)

    def select(self, ctx, scores, k):
        return topk_indices(scores, k)
