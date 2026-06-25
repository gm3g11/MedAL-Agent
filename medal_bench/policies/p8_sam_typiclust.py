"""P8 - Foundation-TypiClust (reference-faithful, TypiClust on SAM features).

Faithful to avihu111/TypiClust, adapted to slice-level medical segmentation with
frozen foundation (SAM) image features:

  1. Cluster labeled ∪ unlabeled features into n_clusters = min(|L| + budget,
     max_clusters) with k-means (foundation features, L2-normalized).
  2. A cluster is "covered" if it contains any labeled sample; "uncovered"
     otherwise. With n_clusters = |L| + budget there are always ≥ budget
     uncovered clusters.
  3. Drop clusters whose TOTAL membership <= MIN_CLUSTER_SIZE (paper default 5;
     configurable) so no isolated/outlier cluster is queried; relax gracefully if
     that leaves nothing eligible. K-cap for typicality = min(knn, len//2) (paper).
  4. Process clusters UNCOVERED-first, each by size (descending), ROUND-ROBIN:
     cycle the clusters taking the next-most-TYPICAL unlabeled sample (typicality =
     1 / mean distance to its K nearest in-cluster neighbours — high density, not an
     outlier) from each until budget is filled. Round-robin keeps cluster-diversity
     even when budget > #eligible clusters.

This keeps TypiClust's two core ideas — prioritize under-covered regions, and
pick typical/high-density (not outlier) points — and is cluster-diverse by
construction. The simplified unlabeled-only/cosine-density/n_clusters=k variant
is the ablation P8b (SAM-DensityClust); the pre-v3 single-pass + global-fallback
behaviour (no min-cluster filter, K-cap=len-1) is preserved as the deprecated
ablation P8c. Uses the SAME SAM feature cache as P7. NO ground-truth masks are
read (clustering is purely on features).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from medal_bench.policies.base import Policy, PolicyContext
from medal_bench.policies.registry import register


@register("P8")
class FoundationTypiClust(Policy):
    name = "Foundation-TypiClust"
    needs_pred_cache = False
    needs_features = ("foundation",)

    def __init__(self, knn: int = 20, max_clusters: int = 500,
                 normalize: bool = True, min_cluster_size: int = 5,
                 n_clusters_rule: str = "labeled_plus_budget", **config):
        super().__init__(knn=knn, max_clusters=max_clusters, normalize=normalize,
                         min_cluster_size=min_cluster_size,
                         n_clusters_rule=n_clusters_rule, **config)
        self.knn = int(knn)
        self.max_clusters = int(max_clusters)
        self.normalize = bool(normalize)
        # paper MIN_CLUSTER_SIZE: clusters with TOTAL membership <= this are dropped
        # before selection (no isolated/outlier picks). Configurable; 0 disables.
        self.min_cluster_size = int(min_cluster_size)
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
            # paper K-cap = min(knn, len//2) (denser estimate on small clusters than
            # the legacy len-1, which averaged over too many neighbours).
            K = min(self.knn, max(1, len(idx) // 2))
            nearest = np.partition(D, K - 1, axis=1)[:, :K]
            typ[idx] = 1.0 / (nearest.mean(axis=1) + 1e-12)
        return typ

    def select(self, ctx: PolicyContext, scores, k: int) -> list[int]:
        pool = ctx.features.get("foundation_pool")
        lab = ctx.features.get("foundation_label")
        assert pool is not None, "P8 requires precomputed foundation_pool"
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
        seed = ctx.query_seed or (ctx.seed + ctx.round_idx)
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        labels = km.fit_predict(X)

        covered = set(labels[:n_lab].tolist())           # clusters with a labeled sample
        cl_pool: dict[int, list[int]] = defaultdict(list)  # cluster -> pool combined-indices
        for ci in range(n_lab, n_total):
            cl_pool[int(labels[ci])].append(ci)
        typ = self._typicality(X, labels)
        cluster_total = np.bincount(labels, minlength=n_clusters)   # TOTAL membership L∪U
        n_singleton = int((cluster_total == 1).sum())
        non_empty = [c for c in cl_pool if cl_pool[c]]              # clusters with >=1 unlabeled

        # MIN_CLUSTER_SIZE filter: keep only clusters whose TOTAL membership exceeds
        # the threshold (drops outlier/singleton clusters). Graceful fallback: if it
        # leaves nothing eligible, relax to all non-empty clusters so k can still fill.
        eligible = [c for c in non_empty if cluster_total[c] > self.min_cluster_size]
        n_filtered = len(non_empty) - len(eligible)
        relaxed = False
        if not eligible:
            eligible = non_empty
            relaxed = True

        # cycle order: UNCOVERED first, then descending size (under-coverage priority)
        order = sorted(eligible, key=lambda c: (c in covered, -len(cl_pool[c])))

        # ROUND-ROBIN: rank each cluster's unlabeled members by typicality (desc), then
        # cycle the clusters taking the next-most-typical from each until k filled. Keeps
        # cluster-diversity even when k > #eligible clusters (vs the legacy single-pass +
        # global fallback).
        ranked = {c: sorted(cl_pool[c], key=lambda i: -typ[i]) for c in order}
        ptr = {c: 0 for c in order}
        selected_combined: list[int] = []
        chosen_clusters: list[int] = []
        progressed = True
        while len(selected_combined) < k and progressed:
            progressed = False
            for c in order:
                if len(selected_combined) >= k:
                    break
                p = ptr[c]
                if p < len(ranked[c]):
                    selected_combined.append(ranked[c][p])
                    ptr[c] = p + 1
                    chosen_clusters.append(c)
                    progressed = True

        # last-resort fill (only if the filtered/eligible members couldn't supply k):
        # pull remaining unlabeled points by global typicality. Marks relax in diagnostics.
        if len(selected_combined) < k:
            relaxed = True
            used = set(selected_combined)
            rest = sorted((ci for ci in range(n_lab, n_total) if ci not in used),
                          key=lambda i: -typ[i])
            for i in rest:
                if len(selected_combined) >= k:
                    break
                selected_combined.append(i)

        selected = [int(i - n_lab) for i in selected_combined[:k]]
        ctx.diagnostics_out["typiclust_n_clusters"] = int(n_clusters)
        ctx.diagnostics_out["typiclust_cluster_rule"] = self.n_clusters_rule
        ctx.diagnostics_out["typiclust_knn"] = self.knn
        ctx.diagnostics_out["typiclust_min_cluster_size"] = self.min_cluster_size
        ctx.diagnostics_out["typiclust_num_filtered_clusters"] = int(n_filtered)
        ctx.diagnostics_out["typiclust_num_singleton_clusters"] = n_singleton
        ctx.diagnostics_out["typiclust_min_cluster_relaxed"] = bool(relaxed)
        ctx.diagnostics_out["typiclust_n_uncovered_clusters"] = int(
            sum(1 for c in cl_pool if c not in covered and cl_pool[c]))
        ctx.diagnostics_out["typiclust_selected_from_uncovered"] = int(
            sum(1 for c in chosen_clusters if c not in covered))
        ctx.diagnostics_out["typiclust_selected_clusters"] = [int(c) for c in chosen_clusters[:k]]
        return selected
