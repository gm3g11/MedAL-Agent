"""P4b - BADGE-Seg-CE-Dice (ABLATION, not a core baseline).

The original segmentation adaptation of BADGE used in earlier MedAL-Bench runs:
per-image gradient of a (CE + soft-Dice) pseudo-label loss w.r.t. the seg-head
weights (via backprop), flattened, then kmeans++. This deviates from canonical
BADGE (which is CE-only); it is retained ONLY as a named ablation with its own
method id, cache key, and result label — never as the main P4.
"""
from __future__ import annotations

import torch

from medal_bench.policies.base import Policy, PolicyContext
from medal_bench.policies.registry import register
from medal_bench.policies._helpers import kmeanspp_indices


@register("P4b")
class BADGESegCEDice(Policy):
    name = "BADGE-Seg-CE-Dice"
    is_ablation = True
    needs_pred_cache = False

    def select(self, ctx: PolicyContext, scores, k: int) -> list[int]:
        from medal_bench.policies._badge_grad import ce_dice_grad_embedding
        embeddings = [
            ce_dice_grad_embedding(ctx.model, ctx.pool[i], num_classes=ctx.num_classes)
            for i in range(len(ctx.pool))
        ]
        X = torch.stack(embeddings, dim=0).cpu().numpy()
        ctx.diagnostics_out["badge_ce_dice_embedding_dim"] = int(X.shape[1])
        return kmeanspp_indices(X, k, random_state=ctx.seed + ctx.round_idx)
