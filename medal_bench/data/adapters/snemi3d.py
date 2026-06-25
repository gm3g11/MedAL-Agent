"""SNEMI3D (ISBI 2013 neurite EM) adapter -> 2D membrane slices.

Source: SNEMI3D challenge (mouse cortex ssTEM, 6x6x30 nm), on disk via Zenodo 7142003. A single
100-slice labeled training volume (1024x1024). Labels are neuron INSTANCE IDs (uint16); we convert
them to neuronal MEMBRANE boundaries (4-neighbour id change), matching isbi2012_em's target.

Single specimen -> like ISBI, contiguous slice BLOCKS of 10 are the leakage-aware group (floor(z/10)
-> 10 blocks), exposed via patient_ids(); ~80 train slices -> 2 acquisition rounds (better than ISBI's
0, but single-specimen so adjacency leakage across block edges is inherent). The 2nd EM dataset
alongside cremi (3 specimens, more rounds).

Layout (under root_dir):
  extracted/image/train-input.tif   - 100 grayscale slices (1024x1024, uint8)
  extracted/seg/train-labels.tif    - 100 instance-id slices (1024x1024, uint16)

Sample ID:  "slice_000" .. "slice_099"
Image:      grayscale -> (1, H, W) float32 in [0, 1]
Mask:       (H, W) int64 in {0, 1}; bg=0, membrane=1 (dilated 2px)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from medal_bench.data.base import MedALDataset, Sample

_BLOCK = 10   # slices per leakage-aware group -> 10 blocks for n=100


def _membrane(seg: np.ndarray) -> np.ndarray:
    """instance-id map (H,W) -> dense binary membrane {0,1}, 4-conn boundary dilated 2px."""
    seg = seg.astype(np.int64)
    b = np.zeros(seg.shape, dtype=bool)
    diff_v = seg[:-1, :] != seg[1:, :]
    b[:-1, :] |= diff_v; b[1:, :] |= diff_v
    diff_h = seg[:, :-1] != seg[:, 1:]
    b[:, :-1] |= diff_h; b[:, 1:] |= diff_h
    for _ in range(2):
        d = b.copy()
        d[:-1, :] |= b[1:, :]; d[1:, :] |= b[:-1, :]
        d[:, :-1] |= b[:, 1:]; d[:, 1:] |= b[:, :-1]
        b = d
    return b.astype(np.int64)


class SNEMI3DAdapter(MedALDataset):
    name = "snemi3d"
    modality = "electron_microscopy"
    target = "membrane"
    dim = "2d"
    query_unit = "image"
    num_classes = 2

    def __init__(self, root_dir: str):
        ex = Path(root_dir) / "extracted"
        vol_path = ex / "image" / "train-input.tif"
        lbl_path = ex / "seg" / "train-labels.tif"
        for p in (vol_path, lbl_path):
            if not p.exists():
                raise FileNotFoundError(f"SNEMI3D file not found: {p}")
        self._vol = tifffile.imread(str(vol_path))   # (Z, H, W) uint8
        self._lbl = tifffile.imread(str(lbl_path))   # (Z, H, W) uint16 instance ids
        if self._vol.shape != self._lbl.shape or self._vol.ndim != 3:
            raise ValueError(f"SNEMI3D shape mismatch: {self._vol.shape} vs {self._lbl.shape}")
        self._n = self._vol.shape[0]
        self._ids = [f"slice_{z:03d}" for z in range(self._n)]
        self._groups = [f"block_{z // _BLOCK}" for z in range(self._n)]

    def __len__(self) -> int:
        return self._n

    def sample_ids(self) -> list[str]:
        return list(self._ids)

    def patient_ids(self) -> list[str]:
        return list(self._groups)

    def __getitem__(self, i: int) -> Sample:
        img = (self._vol[i].astype(np.float32) / 255.0)[None]   # (1, H, W)
        mask = _membrane(self._lbl[i])
        return Sample(sample_id=self._ids[i], image=img,
                      mask=mask, patient_id=self._groups[i], slice_index=i)
