"""P3 - CoreSet (representativeness/diversity on task-model features).

k-center-greedy in the task model's encoder-bottleneck feature space,
seeded by the labeled set so each candidate is farthest from labeled+selected.
L2-normalized features; L2 distance (cosine is an ablation, not a separate policy).

The same algorithm runs on SAM foundation features in P7 (SAM-CoreSet) — this
policy is the task-model variant; the role is general representativeness.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import pairwise_distances

from medal_bench.policies.base import Policy, PolicyContext
from medal_bench.policies.registry import register
from medal_bench.policies._helpers import kcenter_greedy


@register("P3")
class CoreSet(Policy):
    name = "CoreSet"
    needs_pred_cache = False
    needs_features = ("task_unet",)

    def __init__(self, metric: str = "l2", normalize: bool = True, **config):
        super().__init__(metric=metric, normalize=normalize, **config)
        self.metric = metric
        self.normalize = bool(normalize)

    def select(self, ctx: PolicyContext, scores, k: int) -> list[int]:
        pool_feats = ctx.features.get("task_unet_pool")
        label_feats = ctx.features.get("task_unet_label")
        assert pool_feats is not None and label_feats is not None, \
            "P3 requires precomputed task_unet features (pool + label)"
        if self.normalize:
            pool_feats = _l2_norm(pool_feats)
            label_feats = _l2_norm(label_feats)
        n_label = label_feats.shape[0]
        all_feats = np.concatenate([label_feats, pool_feats], axis=0)
        dist_mat = pairwise_distances(all_feats, metric=self.metric)
        new_idx = kcenter_greedy(dist_mat, init_idx=range(n_label), k=k)
        return [int(i - n_label) for i in new_idx if (i - n_label) >= 0]


def _l2_norm(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, 1e-12, None)
