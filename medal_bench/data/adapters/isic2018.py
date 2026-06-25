"""ISIC 2018 Task 1 (skin-lesion segmentation) adapter.

Layout (under root_dir):
  ISIC2018_Task1-2_{Training,Validation,Test}_Input/ISIC_*.jpg
  ISIC2018_Task1_{Training,Validation,Test}_GroundTruth/ISIC_*_segmentation.png

Image:  RGB JPG       -> (3, H, W) float32 in [0, 1]
Mask:   binary PNG    -> (H, W)    int64   in {0, 1}; bg=0, lesion=1
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from medal_bench.data.base import MedALDataset, Sample


_SPLIT_DIRS = {
    "train": ("ISIC2018_Task1-2_Training_Input", "ISIC2018_Task1_Training_GroundTruth"),
    "val":   ("ISIC2018_Task1-2_Validation_Input", "ISIC2018_Task1_Validation_GroundTruth"),
    "test":  ("ISIC2018_Task1-2_Test_Input", "ISIC2018_Task1_Test_GroundTruth"),
}


class ISIC2018Adapter(MedALDataset):
    name = "isic2018"
    modality = "dermoscopy"
    target = "skin_lesion"
    dim = "2d"
    query_unit = "image"
    num_classes = 2

    def __init__(self, root_dir: str, split: str = "train"):
        root = Path(root_dir) / "extracted"
        if not root.exists():
            raise FileNotFoundError(f"ISIC2018 extracted dir not found: {root}")
        if split not in _SPLIT_DIRS:
            raise ValueError(f"split must be one of {list(_SPLIT_DIRS)}, got {split}")
        img_sub, gt_sub = _SPLIT_DIRS[split]
        self._img_dir = root / img_sub
        self._gt_dir = root / gt_sub
        if not self._img_dir.exists() or not self._gt_dir.exists():
            raise FileNotFoundError(
                f"ISIC2018 split '{split}' missing: {self._img_dir} or {self._gt_dir}"
            )
        self._ids = sorted(p.stem for p in self._img_dir.glob("ISIC_*.jpg"))
        self.split = split

    def __len__(self) -> int:
        return len(self._ids)

    def sample_ids(self) -> list[str]:
        return list(self._ids)

    def __getitem__(self, i: int) -> Sample:
        sid = self._ids[i]
        img = np.asarray(Image.open(self._img_dir / f"{sid}.jpg").convert("RGB"))
        img = img.astype(np.float32).transpose(2, 0, 1) / 255.0  # (3, H, W)
        mask_path = self._gt_dir / f"{sid}_segmentation.png"
        m = np.asarray(Image.open(mask_path).convert("L"))
        mask = (m > 0).astype(np.int64)  # binarize: ground truth is 0 or 255
        return Sample(sample_id=sid, image=img, mask=mask, meta={"split": self.split})
