"""Task-encoder (nnU-Net bottleneck) feature extractor.

Used by policies whose ``needs_features = ('task_unet',)`` — currently
P3 (CoreSet), P5 (Entropy → CoreSet), and P9 (PAAL's WPS clustering step).
The PlainConvUNet encoder returns a list of skip-connection feature maps;
the last (bottleneck) is the deepest representation. We global-avg-pool it
to (C_bottleneck,).

Foundation features for P7/P8 come from ``medal_bench.features.sam`` (real
SAM ViT-B); ``foundation_stub`` is the seeded-random fallback the smoke
profile uses when SAM isn't wired in.
"""
from __future__ import annotations

import numpy as np
import torch

from medal_bench.data.base import MedALDataset
from medal_bench.runner.trainer import collate_to_batch


@torch.no_grad()
def extract_task_unet_features(
    model: torch.nn.Module,
    ds: MedALDataset,
    *,
    image_size: int,
    device: str,
    batch_size: int = 16,
) -> np.ndarray:
    """Return (N, D) task encoder features for every sample in ``ds``."""
    was_training = model.training
    model.eval()
    feats: list[np.ndarray] = []
    for start in range(0, len(ds), batch_size):
        chunk = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
        imgs, _ = collate_to_batch(chunk, size=image_size)
        imgs = imgs.to(device)
        skips = model.encoder(imgs)              # list of feature maps
        bottleneck = skips[-1]                   # (B, C, h, w)
        pooled = bottleneck.mean(dim=(-2, -1))   # (B, C)
        feats.append(pooled.cpu().numpy().astype(np.float32))
    if was_training:
        model.train()
    return np.concatenate(feats, axis=0)
