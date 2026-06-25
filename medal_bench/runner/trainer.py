"""Minimal from-scratch trainer for the v1 smoke matrix.

Per constraint #9 ("From-scratch retraining per AL round, fine-tune later"),
the trainer takes a freshly-initialized model + the current labeled set and
trains for a fixed number of iterations.

Smoke defaults:
  - resize to 256x256 (bilinear for image, nearest for mask)
  - batch_size = 4, num_iters = 30
  - AdamW(lr=1e-3)
  - DiceCE loss (mean of CE + (1 - mean Dice over fg classes))

For the pilot we'll switch to nnU-Net's poly-LR schedule + per-dataset patch
size + ~250 iters/round.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from medal_bench.data.base import MedALDataset, Sample


def _letterbox_hw(h: int, w: int, size: int) -> tuple[int, int]:
    """Resized (h,w) so the long side == size, aspect preserved."""
    if h >= w:
        return size, max(1, round(w * size / h))
    return max(1, round(h * size / w)), size


def valid_bbox(orig_h: int, orig_w: int, size: int, aspect_preserve: bool) -> tuple[int, int, int, int]:
    """Un-padded (valid) rectangle of the resized canvas as (y0, x0, h, w).

    Single source of truth for the letterbox geometry: matches the top-left pad
    in ``_resize_image``/``_resize_mask`` (content at rows[0:nh], cols[0:nw]). For
    square resize (aspect_preserve=False) there is no padding -> the full canvas.
    Stored as a bbox (not just (nh,nw)) so callers never assume top-left anchoring."""
    if not aspect_preserve:
        return 0, 0, size, size
    nh, nw = _letterbox_hw(orig_h, orig_w, size)
    return 0, 0, nh, nw


def _resize_image(img: np.ndarray, size: int, aspect_preserve: bool = False) -> torch.Tensor:
    """(C, H, W) float32 -> (C, size, size) float32 via bilinear.

    aspect_preserve=False: square resize (distorts aspect — legacy smoke/pilot).
    aspect_preserve=True: resize long side to `size` (aspect kept), zero-pad to
    size x size (top-left). `size` is a multiple of 32 (e.g. 512)."""
    t = torch.from_numpy(img).unsqueeze(0)  # (1, C, H, W)
    if not aspect_preserve:
        if t.shape[2] == size and t.shape[3] == size:
            return t.squeeze(0)  # already canonical: same-size bilinear (scale=1) is identity
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
        return t.squeeze(0)
    nh, nw = _letterbox_hw(t.shape[2], t.shape[3], size)
    t = F.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False)
    out = torch.zeros((t.shape[1], size, size), dtype=t.dtype)
    out[:, :nh, :nw] = t[0]
    return out


def _resize_mask(mask: np.ndarray, size: int, aspect_preserve: bool = False) -> torch.Tensor:
    """(H, W) int64 -> (size, size) int64 via nearest (letterbox if aspect_preserve)."""
    t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
    if not aspect_preserve:
        if t.shape[2] == size and t.shape[3] == size:
            return t.squeeze().long()  # already canonical: same-size nearest is identity
        t = F.interpolate(t, size=(size, size), mode="nearest")
        return t.squeeze().long()
    nh, nw = _letterbox_hw(t.shape[2], t.shape[3], size)
    t = F.interpolate(t, size=(nh, nw), mode="nearest")
    out = torch.zeros((size, size), dtype=torch.long)
    out[:nh, :nw] = t[0, 0].long()
    return out


def collate_to_batch(samples: list[Sample], size: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    imgs = torch.stack([_resize_image(s.image, size) for s in samples], dim=0)
    masks = torch.stack([_resize_mask(s.mask, size) for s in samples], dim=0)
    return imgs, masks  # (B, C, H, W) float32; (B, H, W) int64


def _dice_ce_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    # reduction="none"+.mean() instead of the default fused reduction: the fused
    # nll_loss2d_forward CUDA kernel reduces with non-deterministic atomics (no
    # deterministic impl in torch 2.4.1), which made btcv round-0 non-reproducible
    # (0.0113 DSC spread across policies). Per-pixel CE then an explicit mean is a
    # deterministic reduction; mathematically identical (sum/N).
    ce = F.cross_entropy(logits, target, reduction="none").mean()
    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    # Dice on foreground classes only (skip bg=0)
    if num_classes > 1:
        p_fg = probs[:, 1:]
        t_fg = onehot[:, 1:]
        inter = (p_fg * t_fg).sum(dim=(0, 2, 3))
        denom = p_fg.sum(dim=(0, 2, 3)) + t_fg.sum(dim=(0, 2, 3))
        dice = (2.0 * inter / denom.clamp(min=1e-8)).mean()
    else:
        dice = torch.tensor(1.0, device=logits.device)
    return ce + (1.0 - dice)


def train_from_scratch(
    model: torch.nn.Module,
    labeled: MedALDataset,
    *,
    num_iters: int,
    batch_size: int,
    lr: float,
    image_size: int,
    num_classes: int,
    device: str,
    seed: int,
    dropout_seed: Optional[int] = None,
    adaptive: bool = False,
    min_iters: int = 500,
    max_iters: int = 3000,
    plateau_window: int = 100,
    plateau_patience: int = 5,
    plateau_min_delta: float = 0.003,
    plateau_rel_delta: float = 0.0,
) -> dict:
    """Train the model in-place. Returns iters run + mean/last loss + stop_reason.

    ``seed`` seeds the batch-sampler RNG (loader_seed). ``dropout_seed``, when given,
    anchors the torch global RNG just before the training loop so dropout masks are
    reproducible independently of weight-init draws (frozen_v3 component seeding).

    ``adaptive=False`` (default): fixed ``num_iters`` — identical to frozen_v3.
    ``adaptive=True`` (frozen_v4): train each round to a smoothed TRAIN-LOSS PLATEAU
    instead of a fixed count, so difficulty-based methods (which select harder data) are
    not differentially under-fit at a fixed budget. Stop when the mean loss over the last
    ``plateau_window`` iters has not improved by >= a combined threshold
    ``max(plateau_min_delta, plateau_rel_delta * |best_loss|)`` for ``plateau_patience``
    consecutive windows, bounded by [``min_iters``, ``max_iters``]. The absolute floor
    dominates near 0 loss (a pure relative test never plateaus there — 0.050->0.0495 reads
    as "1% better" forever); the relative term tolerates larger early-loss noise.
    Train loss is leak-free (never the eval val/test set); it plateaus at-or-after the
    val plateau, so this cannot stop too early (under-fit) — the max cap bounds over-fit.
    Same params for every method/dataset => still a clean control."""
    model.to(device).train()
    rng = np.random.RandomState(seed)
    if dropout_seed is not None:
        torch.manual_seed(int(dropout_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(dropout_seed))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    losses: list[float] = []
    n = len(labeled)
    if n == 0:
        return {"iters": 0, "mean_loss": float("nan"), "last_loss": float("nan"),
                "stop_reason": "empty"}

    cap = max_iters if adaptive else num_iters
    best_smooth = float("inf"); bad_checks = 0
    stop_reason = "plateau_max" if adaptive else "fixed"
    loss_curve: list[tuple[int, float]] = []   # (iter, smoothed train loss) per check, diagnostic
    for it in range(cap):
        idx = rng.choice(n, size=min(batch_size, n), replace=(n < batch_size))
        batch = [labeled[int(i)] for i in idx]
        imgs, masks = collate_to_batch(batch, size=image_size)
        imgs = imgs.to(device); masks = masks.to(device)
        opt.zero_grad()
        logits = model(imgs)
        loss = _dice_ce_loss(logits, masks, num_classes)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
        # adaptive early stop: smoothed train-loss plateau (combined abs/rel delta), after floor
        if adaptive and (it + 1) >= min_iters and (it + 1) % plateau_window == 0:
            smooth = float(np.mean(losses[-plateau_window:]))
            loss_curve.append((it + 1, smooth))
            if best_smooth == float("inf"):          # first check: accept as baseline
                best_smooth = smooth; bad_checks = 0
            else:
                threshold = max(plateau_min_delta, plateau_rel_delta * abs(best_smooth))
                if smooth < best_smooth - threshold:
                    best_smooth = smooth; bad_checks = 0
                else:
                    bad_checks += 1
                    if bad_checks >= plateau_patience:
                        stop_reason = "plateau"; break
    final_smooth = loss_curve[-1][1] if loss_curve else None
    best = None if best_smooth == float("inf") else best_smooth
    # Diagnostic-only divergence flag (audit P6): a round whose final loss ends well
    # above the best it reached has likely diverged late (e.g. care_LA P6 r4:
    # last_loss 0.81 vs best 0.34). Surfaced per-round so collapse-prone cells are
    # auditable/filterable; does NOT alter weights, metrics, or stopping.
    last = losses[-1] if losses else float("nan")
    diverged = bool(best is not None and last > max(0.05, 2.0 * best))
    return {"iters": len(losses), "stop_iter": len(losses),
            "mean_loss": float(np.mean(losses)),
            "last_loss": last, "stop_reason": stop_reason,
            "hit_max_iters": bool(adaptive and stop_reason == "plateau_max"),
            "final_smooth_loss": final_smooth, "best_smooth_loss": best,
            "diverged": diverged,
            "loss_curve": loss_curve}
