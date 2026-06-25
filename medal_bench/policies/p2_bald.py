"""P2 - BALD / MC-dropout (calibration-gating skill).

K stochastic forward passes through the shared dropout-compatible nnU-Net,
with all dropout modules toggled to train mode (BN/InstanceNorm stay in eval
so running stats are not perturbed). Per-pixel BALD = entropy(mean_prob)
- mean(entropy_per_pass). Aggregate = full mean (default).

The two components are exposed in ``ctx.diagnostics_out`` so the
reduces-to-baseline test can verify them separately (Phase A note: don't
assert "BALD score == entropy" on T=1 + dropout-disabled - that's tautological).
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from medal_bench.policies.base import Policy, PolicyContext
from medal_bench.policies.registry import register
from medal_bench.policies._helpers import aggregate, topk_indices
from medal_bench.models.nnunet import enable_mc_dropout


@register("P2")
class BALD(Policy):
    name = "BALD"
    needs_pred_cache = False        # BALD does its own K-pass MC, not from cache

    def __init__(self, T: int = 10, **config):
        super().__init__(T=T, **config)
        self.T = int(T)

    def score(self, ctx: PolicyContext):
        model = ctx.model
        device = next(model.parameters()).device
        # seed BEFORE MC passes so determinism holds with same (seed, round)
        torch.manual_seed(ctx.seed + ctx.round_idx)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(ctx.seed + ctx.round_idx)

        # toggle dropout modules to train; BN/IN stay in eval
        n_dropouts = enable_mc_dropout(model)
        # diagnostics out
        ctx.diagnostics_out["bald_T"] = self.T
        ctx.diagnostics_out["bald_n_dropouts_enabled"] = n_dropouts

        # Preload pool images once (C,H,W); batch the T MC passes for throughput.
        # InstanceNorm is per-sample so batching only changes the dropout-mask RNG
        # order (still seeded/deterministic), not the validity of the MC estimate.
        imgs = []
        for i in range(len(ctx.pool)):
            x = ctx.pool[i].image
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(np.asarray(x, dtype=np.float32))
            x = x.float()
            if x.dim() == 2:
                x = x.unsqueeze(0)
            if x.dim() == 4:
                x = x.squeeze(0)
            imgs.append(x)
        bs = int(self.config.get("mc_batch_size", 32))

        # Accumulate the per-pixel running sums directly per batch. The forward
        # order (pass-outer, batch-inner) is UNCHANGED from the original, so the
        # dropout-mask RNG is consumed in the identical order -> byte-identical
        # scores. We just never hold a duplicate per-pass full (N,C,H,W): only the
        # running sums (one (N,C,H,W) + one (N,H,W)) plus the current batch live at
        # once, instead of the original's running-sum + per-pass copy (~2x).
        running_sum_probs: Optional[torch.Tensor] = None
        sum_per_pass_entropy: Optional[torch.Tensor] = None
        with torch.no_grad():
            for _ in range(self.T):
                start_row = 0
                for start in range(0, len(imgs), bs):
                    xb = torch.stack(imgs[start:start + bs], dim=0).to(device, dtype=torch.float32)
                    p = F.softmax(model(xb), dim=1)                     # (B, C, H, W)
                    e = torch.mean(-p * torch.log2(p + 1e-12), dim=1)   # (B, H, W)
                    p = p.detach().cpu()
                    e = e.detach().cpu()
                    B = p.shape[0]
                    if running_sum_probs is None:
                        N = len(imgs)
                        running_sum_probs = torch.zeros((N, *p.shape[1:]), dtype=p.dtype)
                        sum_per_pass_entropy = torch.zeros((N, *e.shape[1:]), dtype=e.dtype)
                    running_sum_probs[start_row:start_row + B] += p
                    sum_per_pass_entropy[start_row:start_row + B] += e
                    start_row += B

        mean_probs = running_sum_probs / self.T
        pred_ent = torch.mean(-mean_probs * torch.log2(mean_probs + 1e-12), dim=1)  # (N, H, W)
        mean_pp_ent = sum_per_pass_entropy / self.T                       # (N, H, W)
        bald = pred_ent - mean_pp_ent                                     # (N, H, W)

        # expose components for the reduces-to-baseline test
        ctx.diagnostics_out["bald_predictive_entropy"] = pred_ent
        ctx.diagnostics_out["bald_mean_per_pass_entropy"] = mean_pp_ent

        # return cache back to deterministic eval
        model.eval()
        # BALD builds its own per-pixel map over ctx.pool order, so ctx.valid_bboxes
        # (also pool-ordered) aligns; aggregate over the valid region (no-op if None).
        return aggregate(bald, torch.argmax(mean_probs, dim=1), "valid",
                         valid_bboxes=ctx.valid_bboxes)                    # (N,)

    def select(self, ctx, scores, k):
        return topk_indices(scores, k)


def _iter_pool_images(pool, device):
    """Yield (sample_id, image_tensor on device) for each pool item."""
    for i in range(len(pool)):
        s = pool[i]
        x = s.image
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        x = x.to(device, dtype=torch.float32)
        if x.dim() == 2:
            x = x.unsqueeze(0)              # add C
        if x.dim() == 3:
            x = x.unsqueeze(0)              # add B
        yield s.sample_id, x
