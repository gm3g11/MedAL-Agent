"""Tests for the adaptive-resolution letterbox + bench512_dry profile (Stage 0c)."""
from __future__ import annotations

import numpy as np

from medal_bench.runner.trainer import _resize_image, _resize_mask
from medal_bench.profiles import build_run_config


def test_letterbox_preserves_aspect_and_pads():
    # 1x100x200 image -> long side 64 -> resized 32x64 -> padded to 64x64
    img = np.ones((1, 100, 200), dtype=np.float32)
    out = _resize_image(img, 64, aspect_preserve=True)
    assert tuple(out.shape) == (1, 64, 64)
    # content occupies top 32 rows, all 64 cols; bottom rows are zero pad
    assert out[0, :32, :].min() > 0.5          # content
    assert out[0, 32:, :].sum().item() == 0.0  # zero-padded


def test_letterbox_mask_nearest_and_pad_is_background():
    mask = np.ones((100, 200), dtype=np.int64) * 3
    out = _resize_mask(mask, 64, aspect_preserve=True)
    import torch
    assert tuple(out.shape) == (64, 64) and out.dtype == torch.int64
    assert set(np.unique(out.numpy()).tolist()) == {0, 3}  # 3=content, 0=pad
    assert out[32:, :].sum().item() == 0


def test_square_resize_unchanged_default():
    img = np.ones((3, 50, 80), dtype=np.float32)
    out = _resize_image(img, 32)  # aspect_preserve defaults False
    assert tuple(out.shape) == (3, 32, 32)


def test_bench512_profile_budget_and_surface():
    cfg = build_run_config(
        profile_name="bench512_dry", policy_id="P0", policy_config={},
        dataset_name="x", pool_size=2000, seed=1000, out_jsonl="/tmp/x.jsonl",
        num_classes=8,
    )
    assert cfg.train.image_size == 512 and cfg.train.aspect_preserve is True
    assert len(cfg.budget_plan) == 3                 # init + 2 transitions
    assert cfg.budget_plan == sorted(set(cfg.budget_plan))  # strictly increasing
    assert cfg.surface_rounds == {0, 1, 2}           # first/mid/final over 3 rounds
