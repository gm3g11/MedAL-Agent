"""P0 - Random. Uniform random sampling from the unlabeled pool."""
from __future__ import annotations
import numpy as np

from medal_bench.policies.base import Policy, PolicyContext
from medal_bench.policies.registry import register


@register("P0")
class RandomPolicy(Policy):
    name = "Random"
    needs_pred_cache = False

    def select(self, ctx: PolicyContext, scores, k: int) -> list[int]:
        n_pool = len(ctx.pool)
        rng = np.random.RandomState(ctx.seed + ctx.round_idx)
        return rng.choice(n_pool, size=k, replace=False).tolist()
