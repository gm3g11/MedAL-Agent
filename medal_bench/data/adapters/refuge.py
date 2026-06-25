"""REFUGE fundus optic disc+cup adapter — 3-class (bg / disc / cup).

Layout (under root_dir = .../2d/refuge/raw/REFUGE):
  {train,val,test}/Images/<stem>.jpg   fundus photos (400 each)
  {train,val,test}/Masks/<stem>.png    dense class masks (400 each)

The ``Masks/`` PNGs are already remapped to dense class indices
{0=background, 1=optic disc, 2=optic cup} (the original REFUGE ``gts/*.bmp``
challenge codes {255=bg, 128=disc-rim, 0=cup} collapsed to this convention; cup
is the innermost region, nested inside the disc). So no LabelRemapper is needed.

The three native folders are pooled into a single AL universe; the runner's
``make_split`` re-carves train/val/test from this pool (the runner does not
honor native dataset splits). Same C=3 disc+cup task as ``origa``.

Contract (matches the other 2D adapters):
  image -> (3, H, W) float32 in [0, 1]
  mask  -> (H, W)    int64 in {0, 1, 2}
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from medal_bench.data.base import MedALDataset, Sample

_SPLITS = ("train", "val", "test")


class REFUGEAdapter(MedALDataset):
    name = "refuge"
    modality = "fundus"
    target = "optic_nerve_head"
    dim = "2d"
    query_unit = "image"
    num_classes = 3

    def __init__(self, root_dir: str):
        root = Path(root_dir)
        if not root.exists():
            raise FileNotFoundError(f"REFUGE dir not found: {root}")
        items: list[tuple[Path, Path]] = []
        ids: list[str] = []
        for split in _SPLITS:
            img_dir = root / split / "Images"
            msk_dir = root / split / "Masks"
            if not img_dir.exists():
                raise FileNotFoundError(f"REFUGE: image dir not found: {img_dir}")
            masks = {p.stem: p for p in msk_dir.glob("*.png")}
            for ip in sorted(img_dir.glob("*.jpg")):
                mp = masks.get(ip.stem)
                if mp is not None:
                    items.append((ip, mp))
                    ids.append(ip.stem)
        if not items:
            raise FileNotFoundError(f"REFUGE: no image/mask pairs under {root}")
        self._items = items
        self._ids = ids

    def __len__(self) -> int:
        return len(self._items)

    def sample_ids(self) -> list[str]:
        return list(self._ids)

    def __getitem__(self, i: int) -> Sample:
        ip, mp = self._items[i]
        img = np.asarray(Image.open(ip).convert("RGB")).astype(np.float32).transpose(2, 0, 1) / 255.0
        mask = np.asarray(Image.open(mp).convert("L")).astype(np.int64)
        if mask.max() >= self.num_classes:
            raise ValueError(
                f"refuge: mask {mp.name} has label {int(mask.max())} "
                f">= num_classes={self.num_classes}"
            )
        return Sample(sample_id=self._ids[i], image=img, mask=mask, meta={"image_path": str(ip)})
