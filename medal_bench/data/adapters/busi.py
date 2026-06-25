"""BUSI (breast-ultrasound segmentation) adapter.

Layout (under root_dir):
  extracted/Dataset_BUSI_with_GT/{benign,malignant,normal}/
      <class> (N).png             - image
      <class> (N)_mask.png         - primary mask
      <class> (N)_mask_{1,2,...}.png - extra masks for multi-lesion images

Multiple mask files for the same image are OR-merged into one binary mask.
'normal' images have an all-zero mask (no lesion).

Sample ID:  "<class>_<N>" e.g. "benign_001"
Image:      Grayscale -> (1, H, W) float32 in [0, 1]
Mask:       Merged binary -> (H, W) int64 in {0, 1}; bg=0, lesion=1
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

from medal_bench.data.base import MedALDataset, Sample


_CLASSES = ("benign", "malignant", "normal")
# matches "benign (12).png" but not "benign (12)_mask.png"
_IMG_RE = re.compile(r"^(benign|malignant|normal) \((\d+)\)\.png$")


class BUSIAdapter(MedALDataset):
    name = "busi"
    modality = "ultrasound"
    target = "breast_lesion"
    dim = "2d"
    query_unit = "image"
    num_classes = 2

    def __init__(self, root_dir: str):
        root = Path(root_dir) / "extracted" / "Dataset_BUSI_with_GT"
        if not root.exists():
            raise FileNotFoundError(f"BUSI dir not found: {root}")
        self._index: list[tuple[str, int, Path, list[Path]]] = []  # (cls, n, img_path, mask_paths)
        for cls in _CLASSES:
            cls_dir = root / cls
            if not cls_dir.exists():
                continue
            for p in sorted(cls_dir.iterdir()):
                m = _IMG_RE.match(p.name)
                if not m:
                    continue
                n = int(m.group(2))
                masks = sorted(cls_dir.glob(f"{cls} ({n})_mask*.png"))
                self._index.append((cls, n, p, masks))
        # stable order by (class, n)
        self._index.sort(key=lambda r: (_CLASSES.index(r[0]), r[1]))
        self._ids = [f"{cls}_{n:03d}" for (cls, n, _, _) in self._index]

    def __len__(self) -> int:
        return len(self._ids)

    def sample_ids(self) -> list[str]:
        return list(self._ids)

    def __getitem__(self, i: int) -> Sample:
        sid = self._ids[i]
        cls, _n, img_path, mask_paths = self._index[i]
        img = np.asarray(Image.open(img_path).convert("L")).astype(np.float32) / 255.0
        img = img[None]  # (1, H, W)
        H, W = img.shape[-2:]
        if mask_paths:
            merged = np.zeros((H, W), dtype=np.uint8)
            for mp in mask_paths:
                m = np.asarray(Image.open(mp).convert("L"))
                merged |= (m > 0).astype(np.uint8)
            mask = merged.astype(np.int64)
        else:
            mask = np.zeros((H, W), dtype=np.int64)
        return Sample(sample_id=sid, image=img, mask=mask, meta={"class": cls})
