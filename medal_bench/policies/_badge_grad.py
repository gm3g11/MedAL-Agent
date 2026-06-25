"""Per-image gradient embeddings for BADGE (segmentation).

Two embeddings:

1. ``canonical_ce_grad_embedding`` (MAIN P4) — the analytic BADGE embedding for
   the CROSS-ENTROPY pseudo-label loss, which for a final 1x1 conv head equals
   the loss gradient w.r.t. the head weights, computed in closed form (no
   backprop). For penultimate features z_{h,w} (D-dim) and softmax p_{c,h,w}:

       g_c(x) = mean_{h,w} [ (p_{c,h,w} - 1[yhat_{h,w}=c]) * z_{h,w} ]   (D-dim)
       g(x)   = concat_c g_c(x)                                          (C*D-dim)

   yhat is the model's own argmax (pseudo-label) — NO ground truth. This matches
   canonical BADGE (JordanAsh/badge): the gradient embedding under the
   hallucinated label. Closed-form, so it's fast and exactly uses the
   current-round weights (no stale cache).

2. ``ce_dice_grad_embedding`` (ABLATION P4b) — the older segmentation adaptation:
   backprop of (CE + soft-Dice) pseudo-label loss to the head weights. Kept as a
   named ablation; documented deviation from canonical BADGE.

Both look up the segmentation head FRESH each call (the last Conv2d with
out_channels == num_classes) — no id(model) cache, so weight changes across AL
rounds are always reflected.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _find_seg_head(model, num_classes: int) -> torch.nn.Conv2d:
    """Return the final segmentation head conv (last Conv2d with
    out_channels == num_classes). Looked up fresh — no caching."""
    cand = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) and m.out_channels == num_classes:
            cand = m
    if cand is None:
        raise RuntimeError(
            f"BADGE: no Conv2d with out_channels={num_classes} found in model."
        )
    return cand


def _to_input(image, device) -> torch.Tensor:
    x = image
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x)
    x = x.to(device, dtype=torch.float32)
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    return x


@torch.no_grad()
def canonical_ce_grad_embedding(model, sample, num_classes: int) -> torch.Tensor:
    """MAIN P4: analytic CE pseudo-label gradient embedding -> (C*D,) on CPU."""
    device = next(model.parameters()).device
    model.eval()
    head = _find_seg_head(model, num_classes)

    captured = {}
    def _pre_hook(_mod, inp):
        captured["z"] = inp[0].detach()           # input to the head conv
    handle = head.register_forward_pre_hook(_pre_hook)
    try:
        x = _to_input(sample.image, device)
        logits = model(x)                          # (1, C, H, W)
    finally:
        handle.remove()

    z = captured["z"]                              # (1, D, h, w)
    p = torch.softmax(logits, dim=1)               # (1, C, H, W)
    # head input/output share spatial size for a 1x1 head; guard otherwise.
    if z.shape[-2:] != p.shape[-2:]:
        z = F.interpolate(z, size=p.shape[-2:], mode="nearest")
    yhat = torch.argmax(p, dim=1)                  # (1, H, W) pseudo-label
    onehot = F.one_hot(yhat, num_classes).permute(0, 3, 1, 2).float()  # (1,C,H,W)
    w = (p - onehot)[0]                            # (C, H, W)
    D = z.shape[1]
    zf = z[0].reshape(D, -1)                       # (D, HW)
    wf = w.reshape(num_classes, -1)                # (C, HW)
    HW = wf.shape[1]
    g = (wf @ zf.t()) / HW                         # (C, D)
    return g.reshape(-1).detach().cpu()            # (C*D,)


def ce_dice_grad_embedding(model, sample, num_classes: int) -> torch.Tensor:
    """ABLATION P4b: backprop of (CE + soft-Dice) pseudo-label loss to the head
    weight, flattened. Documented segmentation adaptation (NOT canonical BADGE)."""
    device = next(model.parameters()).device
    model.eval()
    head_weight = _find_seg_head(model, num_classes).weight

    x = _to_input(sample.image, device)
    model.zero_grad()
    logits = model(x)                              # (1, C, H, W)
    probs = F.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)             # pseudo-label
    ce = F.cross_entropy(logits, preds)
    preds_onehot = F.one_hot(preds, num_classes=num_classes).permute(0, 3, 1, 2).float()
    inter = (probs * preds_onehot).sum(dim=(0, 2, 3))
    union = probs.sum(dim=(0, 2, 3)) + preds_onehot.sum(dim=(0, 2, 3))
    dice = (2 * inter / union.clamp(min=1e-8)).mean()
    loss = ce + (1.0 - dice)
    loss.backward()
    g = head_weight.grad.detach().clone().flatten()
    model.zero_grad()
    return g.cpu()


# Back-compat alias (old name used CE+Dice).
image_wise_grad_embedding = ce_dice_grad_embedding
