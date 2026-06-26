"""P7 - SAM-CoreSet (foundation-feature representativeness).

k-center-greedy in SAM ViT-H image-encoder feature space. The runner
pre-computes SAM embeddings (cached to HDF5) and stores them in
``ctx.features["foundation_pool"]`` / ``ctx.features["foundation_label"]``.

The foundation encoder identity (encoder_id, checkpoint, feature_layer,
pooling_rule, cache_version) is logged in the trajectory's ``foundation``
field per constraint #8.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import pairwise_distances

from medal_bench.policies.base import Policy, PolicyContext
from medal_bench.policies.registry import register
from medal_bench.policies._helpers import kcenter_greedy


@register("P7")
class SAMCoreSet(Policy):
    name = "Foundation-CoreSet"
    needs_pred_cache = False
    needs_features = ("foundation",)

    def __init__(self, metric: str = "l2", normalize: bool = True, **config):
        super().__init__(metric=metric, normalize=normalize, **config)
        self.metric = metric
        self.normalize = bool(normalize)

    def select(self, ctx: PolicyContext, scores, k: int) -> list[int]:
        pool_feats = ctx.features.get("foundation_pool")
        label_feats = ctx.features.get("foundation_label")
        assert pool_feats is not None and label_feats is not None, \
            "P7 requires precomputed foundation features (pool + label)"
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
