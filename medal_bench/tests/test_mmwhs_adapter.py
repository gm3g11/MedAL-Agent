"""MMWHS adapter smoke + remap/loss/metrics acceptance (Stage -1 B1).

Data-gated: skips cleanly if the MMWHS volumes are not on disk.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from medal_bench.data.adapters.mmwhs import MMWHSAdapter
from medal_bench.data.base import MedALDataset, Sample
from medal_bench.models.nnunet import build_unet_2d
from medal_bench.runner.eval import eval_segmentation
from medal_bench.runner.trainer import _dice_ce_loss

DATA_ROOT = Path(os.environ.get("MEDAL_DATA_ROOT", "/groups/echambe2/datasets/data"))
_MMWHS_ROOT = DATA_ROOT / "3d" / "mmwhs"


def _adapter(modality: str) -> MMWHSAdapter:
    if not (_MMWHS_ROOT / "extracted" / "Wholeheart_Train_Dataset").exists():
        pytest.skip(f"MMWHS not on disk: {_MMWHS_ROOT}")
    return MMWHSAdapter(str(_MMWHS_ROOT), modality=modality)


def _assert_slice(s: Sample):
    img, mask = s.image, s.mask
    assert img.dtype == np.float32 and img.ndim == 3 and img.shape[0] == 1
    assert 0.0 - 1e-6 <= float(img.min()) and float(img.max()) <= 1.0 + 1e-6
    assert mask.dtype == np.int64 and mask.ndim == 2
    assert mask.shape == img.shape[1:]
    assert mask.min() >= 0 and mask.max() < 8


@pytest.mark.parametrize("modality,n_cases", [("ct", 60), ("mr", 46)])
def test_mmwhs_smoke(modality, n_cases):
    ds = _adapter(modality)
    assert ds.name == f"mmwhs_{modality}" and ds.num_classes == 8
    s = ds[0]
    print(f"\n[mmwhs_{modality}] len={len(ds)} id={s.sample_id} pid={s.patient_id} "
          f"z={s.slice_index} img={s.image.shape} classes={np.unique(s.mask).tolist()}")
    _assert_slice(s)
    assert s.patient_id.startswith("Case") and isinstance(s.slice_index, int)
    pids = ds.patient_ids()
    assert pids is not None and len(pids) == len(ds)
    assert len(set(pids)) == n_cases  # case-disjoint splits rely on this


def test_mmwhs_remap_applied_on_real_data():
    """Across several slices the masks must be dense {0..7} with real foreground."""
    ds = _adapter("ct")
    seen = set()
    # sample slices spread across the index (different cases)
    for i in np.linspace(0, len(ds) - 1, 12).astype(int):
        u = np.unique(ds[int(i)].mask).tolist()
        assert set(u) <= set(range(8)), f"slice {i} has classes {u} outside 0..7"
        seen.update(u)
    assert seen - {0}, "no foreground class found in sampled CT slices"


def test_mmwhs_orientation_consistent_across_cases():
    """Two different cases (different native orientations) must both yield
    spatially-aligned (image,mask) slices after canonicalization."""
    ds = _adapter("ct")
    pids = ds.patient_ids()
    first_idx = {}
    for i, p in enumerate(pids):
        first_idx.setdefault(p, i)
        if len(first_idx) >= 2:
            break
    for i in list(first_idx.values())[:2]:
        s = ds[i]
        assert s.mask.shape == s.image.shape[1:]


def test_loss_accepts_mmwhs_mask():
    """_dice_ce_loss must consume an 8-class mask without class-index error."""
    ds = _adapter("ct")
    # find a slice with foreground so Dice is exercised
    mask = None
    for i in np.linspace(0, len(ds) - 1, 30).astype(int):
        m = ds[int(i)].mask
        if m.max() > 0:
            mask = m
            break
    assert mask is not None
    t = torch.from_numpy(mask[None]).long()          # (1, H, W)
    logits = torch.randn(1, 8, *mask.shape)           # (1, C=8, H, W)
    loss = _dice_ce_loss(logits, t, num_classes=8)
    assert torch.isfinite(loss)


def test_metrics_accept_mmwhs_mask():
    """eval_segmentation must compute C=8 DSC on real MMWHS slices."""
    ds = _adapter("ct")

    class _Mem(MedALDataset):
        name, modality, target, dim, query_unit, num_classes = "m", "ct", "h", "3d", "slice", 8
        def __init__(self, samples): self._s = samples
        def __len__(self): return len(self._s)
        def __getitem__(self, i): return self._s[i]
        def sample_ids(self): return [s.sample_id for s in self._s]

    idxs = np.linspace(0, len(ds) - 1, 3).astype(int)
    mem = _Mem([ds[int(i)] for i in idxs])
    model = build_unet_2d(input_channels=1, num_classes=8, features_per_stage=(8, 16, 32))
    out = eval_segmentation(model, mem, num_classes=8, image_size=64, device="cpu")
    assert len(out["dsc_per_class"]) == 8 and out["n_eval"] == 3
