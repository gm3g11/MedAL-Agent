"""P4 - BADGE (canonical CE pseudo-label gradient embedding + kmeans++).

Main P4 uses the canonical BADGE embedding: the closed-form gradient of the
CROSS-ENTROPY pseudo-label loss w.r.t. the final 1x1 seg-head weights,

    g_c(x) = mean_{h,w} [ (p_{c,h,w} - 1[yhat_{h,w}=c]) * z_{h,w} ],  g = concat_c g_c

(z = penultimate features; yhat = model argmax pseudo-label; NO ground truth).
Embedding dimension = num_classes * D. kmeans++ seeds the batch in this space.

The older CE+Dice backprop variant is the ablation P4b (BADGE-Seg-CE-Dice),
see p4b_badge_ce_dice.py — it is NOT the main P4.
"""
from __future__ import annotations

import torch

from medal_bench.policies.base import Policy, PolicyContext
from medal_bench.policies.registry import register
from medal_bench.policies._helpers import kmeanspp_indices


@register("P4")
class BADGE(Policy):
    name = "BADGE"
    needs_pred_cache = False

    def select(self, ctx: PolicyContext, scores, k: int) -> list[int]:
        from medal_bench.policies._badge_grad import canonical_ce_grad_embedding
        embeddings = [
            canonical_ce_grad_embedding(ctx.model, ctx.pool[i], num_classes=ctx.num_classes)
            for i in range(len(ctx.pool))
        ]
        X = torch.stack(embeddings, dim=0).cpu().numpy()
        ctx.diagnostics_out["badge_embedding_dim"] = int(X.shape[1])
        ctx.diagnostics_out["badge_kmeanspp_seed"] = int(ctx.seed + ctx.round_idx)
        return kmeanspp_indices(X, k, random_state=ctx.seed + ctx.round_idx)
