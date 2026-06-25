"""P9 - PAAL (Predictive Accuracy-Based Active Learning).

Canonical PAAL from Yi et al., "Predictive Accuracy-Based Active Learning for
Medical Image Segmentation" (IJCAI 2024). Reference impl: shijun18/PAAL-MedSeg.

Algorithm:
  1. Train Accuracy Predictor (AP) on the LABELED set:
       - For each labeled (image, mask): get the seg model's softmax probs P;
         compute per-class soft Dice T(P, mask) ∈ [0,1].
       - AP input = concat(image, P); AP target = T.
       - MSE loss, AdamW, from-scratch each AL round (since seg is from-scratch too).
  2. Score each UNLABELED sample:
       - Use pred_cache for seg probs.
       - AP output = predicted per-class accuracy â ∈ [0,1].
       - score = -mean(log(â + eps)) over foreground classes (higher = worse predicted accuracy = higher priority).
  3. Weighted Polling Strategy (WPS) over scores:
       - Cluster the unlabeled pool's task-encoder features with KMeans
         (k = log2(4·budget) + 1, clamped).
       - Sort each cluster by score desc.
       - Round-robin across clusters until budget filled.

This is the **fixed-budget** PAAL variant. Incremental Querying (IQ) from the
paper is NOT enabled here — fixed AL rounds keep the comparison with other
MedAL-Bench policies apples-to-apples. PAAL+IQ would be a separate policy.

Distinguished from P6 PEAL (perturbation-aware entropy) — see p6_peal.py.

Diagnostics emitted:
  paal_ap_epochs                - AP training epochs run
  paal_ap_loss_mean             - mean MSE loss over AP training
  paal_ap_loss_last             - last AP training loss
  paal_pred_acc_mean             - mean AP-predicted accuracy across pool (fg classes)
  paal_pred_acc_std              - std of pool predicted accuracy (fg classes)
  paal_score_mean / std          - score distribution
  paal_n_clusters                - WPS KMeans cluster count
  paal_cluster_sizes             - sizes of each cluster
  paal_selected_clusters         - cluster ID of each selected sample
  paal_ap_val_corr               - Pearson r between AP predictions and actual Dice on a held-out labeled split
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from medal_bench.policies.base import Policy, PolicyContext
from medal_bench.policies.registry import register
from medal_bench.policies._paal_ap import AccuracyPredictor, hard_dice_per_class


@register("P9")
class PAAL(Policy):
    name = "PAAL"
    needs_pred_cache = True
    needs_features = ("task_unet",)

    def __init__(self,
                 ap_epochs: int = 200,
                 ap_lr: float = 1e-3,
                 ap_batch_size: int = 4,
                 include_background: bool = True,
                 eps: float = 1e-6,
                 score_mode: str = "neg_log_mean_acc",
                 cluster_rule: str = "paper",
                 persist_ap_weights: bool = True,
                 **config):
        super().__init__(
            ap_epochs=ap_epochs, ap_lr=ap_lr, ap_batch_size=ap_batch_size,
            include_background=include_background, eps=eps,
            score_mode=score_mode, cluster_rule=cluster_rule,
            persist_ap_weights=persist_ap_weights, **config,
        )
        self.ap_epochs = int(ap_epochs)
        self.ap_lr = float(ap_lr)
        self.ap_batch_size = int(ap_batch_size)
        self.include_background = bool(include_background)
        self.eps = float(eps)
        self.score_mode = score_mode
        self.cluster_rule = cluster_rule
        # persist_ap_weights = True keeps the Accuracy Predictor (AP) weights
        # across AL rounds within a single (dataset, seed) cell. The official
        # PAAL paper trains AP jointly with seg for ~400 epochs; AP weights
        # naturally persist across query rounds. We approximate that by carrying
        # AP weights forward instead of re-initializing every round.
        # NOTE: this is unrelated to the AL "cold start" — the initial labeled
        # set at round 0 is still a uniform-random subset shared by all policies
        # at the same seed (see al_loop.run_al).
        self.persist_ap_weights = bool(persist_ap_weights)
        self._ap = None
        self._ap_signature = None  # (image_channels, num_classes) to detect arch mismatch
        self._last_pred_acc = None

    # ----- AP training over labeled samples -------------------------------

    def _score_class_idx(self, num_classes: int) -> list[int]:
        """Which class indices feed into the per-image score aggregation.
        Default `include_background=True` matches the official PAAL `mean(axis=1)`
        over all C output channels. Set False to mean fg only."""
        if self.include_background:
            return list(range(num_classes))
        return list(range(1, num_classes)) if num_classes > 1 else [0]

    def _train_ap(self, ctx: PolicyContext, ap: AccuracyPredictor,
                  image_channels: int, device: str) -> dict:
        """Train AP on ctx.labeled with the (already-trained) seg model frozen.
        Returns dict with ap_loss_mean, ap_loss_last, ap_val_corr."""
        num_classes = ctx.num_classes
        labeled = ctx.labeled
        n_lab = len(labeled)
        rng = np.random.RandomState(ctx.seed + ctx.round_idx + 12345)

        # tiny train/val split inside the labeled set for AP correlation diagnostic
        # (uses a few samples only as validation; falls back to train-only if too small)
        n_val_ap = min(8, max(0, n_lab // 10))
        order = list(range(n_lab))
        rng.shuffle(order)
        val_ap_idx = order[:n_val_ap]
        train_ap_idx = order[n_val_ap:]

        opt = torch.optim.AdamW(ap.parameters(), lr=self.ap_lr)
        ctx.model.eval()
        losses: list[float] = []
        # AP is trained to predict the FULL per-class accuracy vector; the
        # score aggregation later (over class_idx) is a separate concern.
        # Training on all classes matches the official MSE loss target shape.
        all_classes = list(range(num_classes))

        # SPEEDUP (MEDAL_P9_CACHE_AP=1): the seg model is FROZEN during AP training, so each
        # sample's softmax-probs + Dice target are CONSTANT across all ap_epochs. The default path
        # recomputes them every epoch -> ~ap_epochs x redundant seg-forward (the dominant P9 cost on
        # slow GPUs, which blew past the 4h cell timeout). With the flag we precompute them ONCE.
        # Off by default so already-run seeds stay byte-identical until the cache is verified to
        # produce identical selections.
        _cache_ap = os.environ.get("MEDAL_P9_CACHE_AP") == "1"
        _ap_cache: dict = {}
        if _cache_ap:
            for _s in range(0, len(train_ap_idx), self.ap_batch_size):
                _bi = train_ap_idx[_s:_s + self.ap_batch_size]
                _im, _ma = self._collate(labeled, _bi, device)
                with torch.no_grad():
                    _pr = F.softmax(ctx.model(_im), dim=1)
                    _dt = hard_dice_per_class(_pr, _ma, num_classes, eps=self.eps)
                for _k, _idx in enumerate(_bi):
                    _ap_cache[_idx] = (_im[_k], _pr[_k], _dt[_k])

        for epoch in range(self.ap_epochs):
            ep_loss = 0.0
            ep_n = 0
            shuffled = list(train_ap_idx)
            rng.shuffle(shuffled)
            for start in range(0, len(shuffled), self.ap_batch_size):
                batch_idx = shuffled[start:start + self.ap_batch_size]
                # ResNet-18 BatchNorm needs >1 sample in train mode; skip
                # 1-sample remainder batches.
                if len(batch_idx) < 2:
                    continue
                if _cache_ap:
                    imgs = torch.stack([_ap_cache[i][0] for i in batch_idx], dim=0)
                    probs = torch.stack([_ap_cache[i][1] for i in batch_idx], dim=0)
                    dice_tgt = torch.stack([_ap_cache[i][2] for i in batch_idx], dim=0)
                else:
                    imgs, masks = self._collate(labeled, batch_idx, device)
                    with torch.no_grad():
                        probs = F.softmax(ctx.model(imgs), dim=1)
                        dice_tgt = hard_dice_per_class(probs, masks, num_classes, eps=self.eps)
                ap.train()
                ap_in = torch.cat([imgs, probs], dim=1)
                pred = ap(ap_in)
                loss = F.mse_loss(pred[:, all_classes], dice_tgt[:, all_classes])
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += float(loss.detach().cpu()) * len(batch_idx)
                ep_n += len(batch_idx)
            if ep_n > 0:
                losses.append(ep_loss / ep_n)

        # Validation correlation on held-out labeled split
        val_corr = float("nan")
        if val_ap_idx:
            ap.eval()
            pred_vals: list[float] = []
            true_vals: list[float] = []
            with torch.no_grad():
                imgs, masks = self._collate(labeled, val_ap_idx, device)
                probs = F.softmax(ctx.model(imgs), dim=1)
                dice_tgt = hard_dice_per_class(probs, masks, num_classes, eps=self.eps)
                ap_in = torch.cat([imgs, probs], dim=1)
                pred = ap(ap_in)
                for b in range(imgs.shape[0]):
                    for c in all_classes:
                        pred_vals.append(float(pred[b, c].cpu()))
                        true_vals.append(float(dice_tgt[b, c].cpu()))
            if len(pred_vals) >= 2 and np.std(pred_vals) > 0 and np.std(true_vals) > 0:
                val_corr = float(np.corrcoef(pred_vals, true_vals)[0, 1])

        return {
            "ap_loss_mean": float(np.mean(losses)) if losses else float("nan"),
            "ap_loss_last": float(losses[-1]) if losses else float("nan"),
            "ap_val_corr": val_corr,
            "ap_n_train": len(train_ap_idx),
            "ap_n_val": len(val_ap_idx),
        }

    def _collate(self, ds, indices, device):
        """Stack samples at given indices into batched tensors on device.
        Images already resized to a uniform shape by _IndexedSubset."""
        imgs = []
        masks = []
        for i in indices:
            s = ds[i]
            img = s.image
            mask = s.mask
            if not isinstance(img, torch.Tensor):
                img = torch.from_numpy(np.asarray(img, dtype=np.float32))
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(np.asarray(mask, dtype=np.int64))
            imgs.append(img)
            masks.append(mask)
        imgs = torch.stack(imgs, dim=0).to(device, dtype=torch.float32)
        masks = torch.stack(masks, dim=0).to(device, dtype=torch.long)
        return imgs, masks

    # ----- public Policy API ---------------------------------------------

    def score(self, ctx: PolicyContext):
        device = next(ctx.model.parameters()).device
        num_classes = ctx.num_classes
        first_img = ctx.pool[0].image
        image_channels = int(first_img.shape[0])
        class_idx = self._score_class_idx(num_classes)

        # 1. Build + train AP. Warm-start across rounds (per official PAAL,
        # which trains AP jointly with seg for 100s of epochs and preserves it
        # across query rounds). Re-init only when image_channels/num_classes
        # change (cross-dataset reuse not supported anyway).
        sig = (image_channels, num_classes)
        if self.persist_ap_weights and self._ap is not None and self._ap_signature == sig:
            ap = self._ap.to(device)
        else:
            torch.manual_seed(ctx.seed + ctx.round_idx)
            ap = AccuracyPredictor(image_channels=image_channels,
                                   num_classes=num_classes).to(device)
            self._ap_signature = sig
        ap_stats = self._train_ap(ctx, ap, image_channels, device)
        if self.persist_ap_weights:
            self._ap = ap

        # 2. Score unlabeled pool via AP (using cached seg probs)
        ap.eval()
        per_sample_score = torch.zeros(len(ctx.pool))
        per_sample_pred_acc = torch.zeros(len(ctx.pool), num_classes)

        # batch through the pool for efficiency
        bs = self.ap_batch_size
        with torch.no_grad():
            for start in range(0, len(ctx.pool), bs):
                end = min(start + bs, len(ctx.pool))
                imgs = torch.stack([
                    torch.from_numpy(np.asarray(ctx.pool[i].image, dtype=np.float32))
                    if not isinstance(ctx.pool[i].image, torch.Tensor)
                    else ctx.pool[i].image.float()
                    for i in range(start, end)
                ], dim=0).to(device)
                probs = ctx.pred_cache.probs[start:end].to(device)
                ap_in = torch.cat([imgs, probs], dim=1)
                pred = ap(ap_in).clamp(self.eps, 1.0 - self.eps)
                # score: -mean(log(pred)) over fg classes (higher = lower pred acc = higher priority)
                fg = pred[:, class_idx]
                if self.score_mode == "neg_log_mean_acc":
                    s = -torch.log(fg).mean(dim=1)
                elif self.score_mode == "one_minus_mean_acc":
                    s = 1.0 - fg.mean(dim=1)
                else:
                    raise ValueError(f"unknown score_mode {self.score_mode}")
                per_sample_score[start:end] = s.cpu()
                per_sample_pred_acc[start:end] = pred.cpu()

        self._last_pred_acc = per_sample_pred_acc

        # diagnostics
        ctx.diagnostics_out["paal_ap_epochs"] = self.ap_epochs
        ctx.diagnostics_out["paal_ap_loss_mean"] = ap_stats["ap_loss_mean"]
        ctx.diagnostics_out["paal_ap_loss_last"] = ap_stats["ap_loss_last"]
        ctx.diagnostics_out["paal_ap_val_corr"] = ap_stats["ap_val_corr"]
        ctx.diagnostics_out["paal_ap_n_train"] = ap_stats["ap_n_train"]
        ctx.diagnostics_out["paal_ap_n_val"] = ap_stats["ap_n_val"]
        fg_acc = per_sample_pred_acc[:, class_idx]
        ctx.diagnostics_out["paal_pred_acc_mean"] = float(fg_acc.mean())
        ctx.diagnostics_out["paal_pred_acc_std"] = float(fg_acc.std())
        ctx.diagnostics_out["paal_score_mean"] = float(per_sample_score.mean())
        ctx.diagnostics_out["paal_score_std"] = float(per_sample_score.std())

        return per_sample_score

    def select(self, ctx, scores, k):
        """WPS: cluster pool's task features, sort each cluster by score desc, round-robin."""
        pool_feats = ctx.features.get("task_unet_pool")
        assert pool_feats is not None, "PAAL WPS requires task_unet_pool features"
        n = len(scores)
        if k <= 0:
            return []
        if k >= n:
            return list(range(n))

        # cluster count per the paper rule: int(log2(4k) + 1)
        if self.cluster_rule == "paper":
            n_clusters = int(math.log2(max(4 * k, 2)) + 1)
        elif self.cluster_rule.startswith("fixed:"):
            n_clusters = int(self.cluster_rule.split(":")[1])
        else:
            raise ValueError(f"unknown cluster_rule {self.cluster_rule}")
        n_clusters = max(1, min(n_clusters, n, k))

        # L2-normalize features for KMeans stability
        feats = pool_feats / np.clip(np.linalg.norm(pool_feats, axis=1, keepdims=True), 1e-12, None)
        km = KMeans(n_clusters=n_clusters,
                    random_state=ctx.seed + ctx.round_idx,
                    n_init=10)
        cluster_labels = km.fit_predict(feats)

        scores_arr = scores.numpy() if hasattr(scores, "numpy") else np.asarray(scores)
        clusters: dict[int, list[tuple[int, float]]] = {c: [] for c in range(n_clusters)}
        for i, c in enumerate(cluster_labels):
            clusters[int(c)].append((i, float(scores_arr[i])))
        for c in clusters:
            clusters[c].sort(key=lambda x: -x[1])

        selected: list[int] = []
        cursor = 0
        order = list(range(n_clusters))
        while len(selected) < k:
            if all(not clusters[c] for c in order):
                break
            c = order[cursor % n_clusters]
            cursor += 1
            if clusters[c]:
                idx, _ = clusters[c].pop(0)
                selected.append(idx)

        ctx.diagnostics_out["paal_n_clusters"] = n_clusters
        ctx.diagnostics_out["paal_cluster_sizes"] = [int((cluster_labels == c).sum()) for c in range(n_clusters)]
        ctx.diagnostics_out["paal_selected_clusters"] = [int(cluster_labels[i]) for i in selected]
        return selected
