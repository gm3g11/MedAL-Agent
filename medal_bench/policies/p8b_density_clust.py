"""P8b - SAM-DensityClust / Foundation-TypiClust-lite (ABLATION, not core).

The earlier simplified P8: cluster the UNLABELED pool only into k clusters
(k-means++), pick the most typical (highest mean top-m cosine similarity)
unlabeled sample per cluster, round-robin to fill k. It ignores labeled coverage
and uses cosine density with n_clusters=k — a documented simplification of
canonical TypiClust. Retained ONLY as a named ablation.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from medal_bench.policies.base import Policy, PolicyContext
from medal_bench.policies.registry import register


@register("P8b")
class SAMDensityClust(Policy):
    name = "SAM-DensityClust"
    is_ablation = True
    needs_pred_cache = False
    needs_features = ("foundation",)

    def __init__(self, m_neighbors: int = 5, normalize: bool = True, **config):
        super().__init__(m_neighbors=m_neighbors, normalize=normalize, **config)
        self.m = int(m_neighbors)
        self.normalize = bool(normalize)

    def select(self, ctx: PolicyContext, scores, k: int) -> list[int]:
        pool_feats = ctx.features.get("foundation_pool")
        assert pool_feats is not None, "P8b requires precomputed foundation_pool"
        X = pool_feats.copy()
        if self.normalize:
            X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)

        n = X.shape[0]
        if k >= n:
            return list(range(n))
        k_eff = min(k, n)
        km = KMeans(n_clusters=k_eff, random_state=ctx.seed + ctx.round_idx, n_init=10)
        labels = km.fit_predict(X)

        ranked: dict[int, list[int]] = {}
        for c in range(k_eff):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                continue
            if len(members) == 1:
                ranked[c] = [int(members[0])]
                continue
            Xc = X[members]
            sim = cosine_similarity(Xc, Xc)
            np.fill_diagonal(sim, -np.inf)
            m_eff = min(self.m, len(members) - 1)
            partitioned = np.partition(sim, -m_eff, axis=1)[:, -m_eff:]
            typicality = partitioned.mean(axis=1)
            order = np.argsort(typicality, kind="stable")[::-1]
            ranked[c] = [int(members[j]) for j in order]

        cluster_order = sorted(ranked.keys(), key=lambda c: -len(ranked[c]))
        selected: list[int] = []
        ctx.diagnostics_out["p8b_n_clusters"] = k_eff
        while len(selected) < k and any(ranked[c] for c in cluster_order):
            for c in cluster_order:
                if ranked[c]:
                    selected.append(ranked[c].pop(0))
                    if len(selected) == k:
                        break
        return selected
