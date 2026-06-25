"""P8c - Foundation-TypiClust (LEGACY, deprecated ablation).

A verbatim snapshot of the pre-frozen_v3 P8 selection: single-pass one-per-cluster
+ global-typicality fallback, K-cap = min(knn, len-1), and NO MIN_CLUSTER_SIZE
filter. Kept ONLY so the v2 Stage-1/1.5 P8 results remain reproducible bit-for-bit;
the canonical paper-faithful method is the live P8 (min-cluster filter + round-robin
+ K-cap min(knn, len//2)). Not part of the core average (is_ablation=True).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from medal_bench.policies.base import Policy, PolicyContext
from medal_bench.policies.registry import register


@register("P8c")
class FoundationTypiClustLegacy(Policy):
    name = "Foundation-TypiClust-legacy"
    is_ablation = True
    needs_pred_cache = False
    needs_features = ("foundation",)

    def __init__(self, knn: int = 20, max_clusters: int = 500,
                 normalize: bool = True,
                 n_clusters_rule: str = "labeled_plus_budget", **config):
        super().__init__(knn=knn, max_clusters=max_clusters, normalize=normalize,
                         n_clusters_rule=n_clusters_rule, **config)
        self.knn = int(knn)
        self.max_clusters = int(max_clusters)
        self.normalize = bool(normalize)
        self.n_clusters_rule = n_clusters_rule

    def _typicality(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """typicality_i = 1 / mean distance to K nearest in-cluster neighbours."""
        typ = np.zeros(X.shape[0], dtype=np.float64)
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            if len(idx) == 1:
                typ[idx[0]] = 0.0
                continue
            D = pairwise_distances(X[idx])
            np.fill_diagonal(D, np.inf)
            K = min(self.knn, len(idx) - 1)
            nearest = np.partition(D, K - 1, axis=1)[:, :K]
            typ[idx] = 1.0 / (nearest.mean(axis=1) + 1e-12)
        return typ

    def select(self, ctx: PolicyContext, scores, k: int) -> list[int]:
        pool = ctx.features.get("foundation_pool")
        lab = ctx.features.get("foundation_label")
        assert pool is not None, "P8c requires precomputed foundation_pool"
        if lab is None:
            lab = np.zeros((0, pool.shape[1]), dtype=pool.dtype)

        n_pool, n_lab = pool.shape[0], lab.shape[0]
        if k >= n_pool:
            return list(range(n_pool))

        X = np.concatenate([lab, pool], axis=0).astype(np.float32)
        if self.normalize:
            X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
        n_total = X.shape[0]

        if self.n_clusters_rule != "labeled_plus_budget":
            raise ValueError(f"unknown n_clusters_rule {self.n_clusters_rule}")
        n_clusters = max(1, min(n_lab + k, self.max_clusters, n_total))
        km = KMeans(n_clusters=n_clusters, random_state=ctx.seed + ctx.round_idx, n_init=10)
        labels = km.fit_predict(X)

        covered = set(labels[:n_lab].tolist())
        cl_pool: dict[int, list[int]] = defaultdict(list)
        for ci in range(n_lab, n_total):
            cl_pool[int(labels[ci])].append(ci)
        typ = self._typicality(X, labels)

        order = sorted((c for c in cl_pool if cl_pool[c]),
                       key=lambda c: (c in covered, -len(cl_pool[c])))

        used: set[int] = set()
        selected_combined: list[int] = []
        chosen_clusters: list[int] = []
        for c in order:
            if len(selected_combined) >= k:
                break
            members = [i for i in cl_pool[c] if i not in used]
            if not members:
                continue
            best = max(members, key=lambda i: typ[i])
            selected_combined.append(best)
            used.add(best)
            chosen_clusters.append(c)

        if len(selected_combined) < k:
            rest = sorted((i for c in order for i in cl_pool[c] if i not in used),
                          key=lambda i: -typ[i])
            for i in rest:
                if len(selected_combined) >= k:
                    break
                selected_combined.append(i)
                used.add(i)

        selected = [int(i - n_lab) for i in selected_combined[:k]]
        ctx.diagnostics_out["typiclust_n_clusters"] = int(n_clusters)
        ctx.diagnostics_out["typiclust_cluster_rule"] = self.n_clusters_rule
        ctx.diagnostics_out["typiclust_knn"] = self.knn
        ctx.diagnostics_out["typiclust_n_uncovered_clusters"] = int(
            sum(1 for c in cl_pool if c not in covered and cl_pool[c]))
        ctx.diagnostics_out["typiclust_selected_from_uncovered"] = int(
            sum(1 for c in chosen_clusters if c not in covered))
        ctx.diagnostics_out["typiclust_selected_clusters"] = [int(c) for c in chosen_clusters[:k]]
        return selected
