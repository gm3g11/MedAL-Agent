"""Generic 2D image+mask folder adapter.

Covers the common medical-segmentation layout where images and their masks are
paired by filename stem, either in two sibling directories (``images/`` +
``masks/``) or in one directory using a mask filename suffix (e.g. GlaS's
``foo.bmp`` / ``foo_anno.bmp``). One config line wires a new dataset; bespoke
formats (XML annotations, dual masks, NIfTI volumes, colour-coded multiclass
masks) still need their own adapter.

Contract (matches the other adapters):
  image -> (C, H, W) float32 in [0, 1]   (C=3 if to_rgb else 1)
  mask  -> (H, W)    int64               (binarized at bin_threshold, or raw
                                          class indices when binarize=False)

Image and mask may have different native resolutions (e.g. JSRT 2048 vs 1024);
the runner resizes both to the train image_size (image bilinear, mask nearest),
so alignment holds as long as they share the same field of view.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from medal_bench.data.base import MedALDataset, Sample

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}


class ImageMaskFolderAdapter(MedALDataset):
    dim = "2d"
    query_unit = "image"

    def __init__(
        self,
        *,
        name: str,
        modality: str,
        target: str,
        image_dir: str,
        mask_dir: Optional[str] = None,
        mask_suffix: Optional[str] = None,
        num_classes: int = 2,
        to_rgb: bool = True,
        binarize: bool = True,
        bin_threshold: int = 128,
    ):
        self.name = name
        self.modality = modality
        self.target = target
        self.num_classes = num_classes
        self._to_rgb = to_rgb
        self._binarize = binarize
        self._bin_threshold = bin_threshold

        img_dir = Path(image_dir)
        msk_dir = Path(mask_dir or image_dir)
        if not img_dir.exists():
            raise FileNotFoundError(f"{name}: image dir not found: {img_dir}")
        imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in _IMG_EXTS)

        items: list[tuple[Path, Path]] = []
        if mask_suffix:
            for p in imgs:
                if p.stem.endswith(mask_suffix):
                    continue  # this file IS a mask
                mp = msk_dir / f"{p.stem}{mask_suffix}{p.suffix}"
                if mp.exists():
                    items.append((p, mp))
        else:
            mask_by_stem = {p.stem: p for p in msk_dir.iterdir() if p.suffix.lower() in _IMG_EXTS}
            for p in imgs:
                mp = mask_by_stem.get(p.stem)
                if mp is not None:
                    items.append((p, mp))
        if not items:
            raise FileNotFoundError(f"{name}: no image/mask pairs under {img_dir} / {msk_dir}")
        self._items = items
        self._ids = [p.stem for p, _ in items]

    def __len__(self) -> int:
        return len(self._items)

    def sample_ids(self) -> list[str]:
        return list(self._ids)

    def __getitem__(self, i: int) -> Sample:
        ip, mp = self._items[i]
        if self._to_rgb:
            img = np.asarray(Image.open(ip).convert("RGB")).astype(np.float32).transpose(2, 0, 1) / 255.0
        else:
            img = (np.asarray(Image.open(ip).convert("L")).astype(np.float32) / 255.0)[None]
        m = np.asarray(Image.open(mp).convert("L"))
        if self._binarize:
            mask = (m >= self._bin_threshold).astype(np.int64)
        else:
            mask = m.astype(np.int64)
            if mask.max() >= self.num_classes:
                raise ValueError(
                    f"{self.name}: mask {mp.name} has label {int(mask.max())} "
                    f">= num_classes={self.num_classes}"
                )
        return Sample(sample_id=self._ids[i], image=img, mask=mask, meta={"image_path": str(ip)})
