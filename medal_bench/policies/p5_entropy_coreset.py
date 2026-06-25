"""P5 - Entropy -> CoreSet (two-stage uncertainty-diversity hybrid).

Stage 1: P1 entropy scores; keep top ``filter_ratio * k`` candidates (pool-size-clamped).
Stage 2: k-center-greedy on the kept candidates' task features, seeded by the labeled set.
"""
from __future__ import annotations

import numpy as np

from medal_bench.policies.base import Policy, PolicyContext
from medal_bench.policies.registry import register
from medal_bench.policies._helpers import coverage_after_filter_indices
from medal_bench.policies.p1_entropy_full import _entropy_valid_batch


@register("P5")
class EntropyCoreSet(Policy):
    name = "Entropy -> CoreSet"
    needs_pred_cache = True
    needs_pred_cache_probs = False   # streaming reduction; full (N,C,H,W) never materialized
    needs_features = ("task_unet",)

    def __init__(self, filter_ratio: float = 5.0, metric: str = "l2", **config):
        super().__init__(filter_ratio=filter_ratio, metric=metric, **config)
        self.filter_ratio = float(filter_ratio)
        self.metric = metric

    def per_batch_reduce(self, probs_b, argmax_b, valid_b, offset):
        return _entropy_valid_batch(probs_b, argmax_b, valid_b)

    def finalize_score(self, accum, ctx):
        return accum["score"]

    def score(self, ctx: PolicyContext):
        accum = ctx.streamed_reduce
        if accum is None:
            accum = self.per_batch_reduce(
                ctx.pred_cache.probs, ctx.pred_cache.argmax, ctx.valid_bboxes, 0)
        return self.finalize_score(accum, ctx)

    def select(self, ctx, scores, k):
        pool_feats = ctx.features.get("task_unet_pool")
        label_feats = ctx.features.get("task_unet_label")
        assert pool_feats is not None and label_feats is not None
        return coverage_after_filter_indices(
            per_sample_score=scores,
            pool_features=pool_feats,
            label_features=label_feats,
            k=k,
            filter_ratio=self.filter_ratio,
            metric=self.metric,
        )
