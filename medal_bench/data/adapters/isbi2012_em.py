"""ISBI 2012 EM (ssTEM membrane segmentation) adapter — multipage TIFF -> 2D slices.

Source: ISBI 2012 EM segmentation challenge (Drosophila ventral nerve cord, ssTEM).

Layout (under root_dir):
  raw/train-volume.tif   - 30 grayscale slices (512x512, uint8)
  raw/train-labels.tif   - 30 label slices    (512x512, uint8, values {0, 255})
  raw/test-volume.tif    - unlabeled test stack (NOT used; no ground-truth masks)

CRITICAL LABEL INVERSION: in the label TIFF, 0 = membrane (the FOREGROUND/positive
class) and 255 = background. We remap mask = (label == 0) so membrane -> 1, bg -> 0,
yielding a dense binary mask in {0, 1}.

There is a single ssTEM specimen, so all 30 slices come from one "patient". Adjacent
slices are physically correlated (consecutive sections of the same tissue), so to keep
train/val/test splits leakage-aware we group slices into contiguous index BLOCKS of 10
(floor(i / 10) -> 3 groups: blocks 0, 1, 2) and expose those via ``patient_ids()``.
NOTE: this only removes *block-boundary* leakage; mild adjacency correlation within a
block (and across block edges) is inherent to a single-specimen stack and unavoidable.

Sample ID:  "slice_NN" e.g. "slice_00" .. "slice_29"
Image:      Grayscale -> (1, H, W) float32 in [0, 1]
Mask:       (H, W) int64 in {0, 1}; bg=0, membrane=1
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from medal_bench.data.base import MedALDataset, Sample


_BLOCK = 10  # slices per leakage-aware group -> floor(i/10) gives 3 blocks for n=30


class ISBI2012EMAdapter(MedALDataset):
    name = "isbi2012_em"
    modality = "electron_microscopy"
    target = "membrane"
    dim = "2d"
    query_unit = "image"
    num_classes = 2

    def __init__(self, root_dir: str):
        raw = Path(root_dir) / "raw"
        vol_path = raw / "train-volume.tif"
        lbl_path = raw / "train-labels.tif"
        for p in (vol_path, lbl_path):
            if not p.exists():
                raise FileNotFoundError(f"ISBI2012-EM file not found: {p}")
        self._vol = tifffile.imread(str(vol_path))   # (Z, H, W) uint8
        self._lbl = tifffile.imread(str(lbl_path))   # (Z, H, W) uint8, {0, 255}
        if self._vol.shape != self._lbl.shape:
            raise ValueError(
                f"ISBI2012-EM volume/label shape mismatch: "
                f"{self._vol.shape} vs {self._lbl.shape}"
            )
        if self._vol.ndim != 3:
            raise ValueError(f"ISBI2012-EM expected (Z, H, W) stack, got {self._vol.shape}")
        self._n = self._vol.shape[0]
        self._ids = [f"slice_{z:02d}" for z in range(self._n)]
        self._groups = [f"block_{z // _BLOCK}" for z in range(self._n)]

    def __len__(self) -> int:
        return self._n

    def sample_ids(self) -> list[str]:
        return list(self._ids)

    def patient_ids(self) -> list[str]:
        # contiguous slice blocks are the leakage-aware grouping unit for splits
        return list(self._groups)

    def __getitem__(self, i: int) -> Sample:
        img = self._vol[i].astype(np.float32) / 255.0
        img = img[None]  # (1, H, W)
        # label inversion: 0 = membrane (foreground), 255 = background
        mask = (self._lbl[i] == 0).astype(np.int64)
        return Sample(
            sample_id=self._ids[i], image=img, mask=mask,
            patient_id=self._groups[i], slice_index=i,
        )
