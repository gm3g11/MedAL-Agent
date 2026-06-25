"""P6 - PEAL (Perturbation-aware Entropy AL).

Per-image score = mean over pixels [ entropy(p_orig) × 1[argmax(p_orig) ≠ argmax(p_hflip)] ]

That is: base softmax entropy weighted by pixel-level disagreement between the
model's argmax on the original image and on its horizontal flip. Select top-K.

This is NOT canonical PAAL (IJCAI 2024, Yi et al. — see P9). Earlier drafts
called this policy "PAAL"; it is more accurately a perturbation-aware variant
of entropy with a single flip perturbation. The diagnostic key was renamed to
match (peal_mean_disagreement).

Uses the per-round prediction cache for the original pass (zero extra forward)
plus exactly one extra deterministic forward over the unlabeled pool with
horizontal flip. No MC dropout, no separate accuracy predictor.

Diagnostics emitted:
  peal_mean_disagreement   - mean fraction of pixels that flip class under hflip.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from medal_bench.policies.base import Policy, PolicyContext
from medal_bench.policies.registry import register
from medal_bench.policies._helpers import score_per_pixel, topk_indices


@register("P6")
class PEAL(Policy):
    name = "PEAL"
    needs_pred_cache = True

    def __init__(self, **config):
        super().__init__(**config)

    @torch.no_grad()
    def score(self, ctx: PolicyContext):
        device = next(ctx.model.parameters()).device
        was_training = ctx.model.training
        ctx.model.eval()

        ent = score_per_pixel(ctx.pred_cache.probs, "entropy")
        argmax_orig = ctx.pred_cache.argmax

        flipped_argmax: list[torch.Tensor] = []
        for i in range(len(ctx.pool)):
            x = ctx.pool[i].image
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(np.asarray(x, dtype=np.float32))
            x = x.to(device, dtype=torch.float32)
            if x.dim() == 2:
                x = x.unsqueeze(0)
            if x.dim() == 3:
                x = x.unsqueeze(0)
            x_flip = torch.flip(x, dims=[-1])
            p_flip = F.softmax(ctx.model(x_flip), dim=1)
            p_unflip = torch.flip(p_flip, dims=[-1])
            flipped_argmax.append(torch.argmax(p_unflip, dim=1).squeeze(0).cpu())

        argmax_flip = torch.stack(flipped_argmax, dim=0)
        disagreement = (argmax_orig != argmax_flip).float()

        per_pixel = ent * disagreement
        per_image = per_pixel.mean(dim=(-2, -1))

        ctx.diagnostics_out["peal_mean_disagreement"] = float(disagreement.mean())

        if was_training:
            ctx.model.train()
        return per_image

    def select(self, ctx, scores, k):
        return topk_indices(scores, k)
