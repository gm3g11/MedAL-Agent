"""CRAG (Colorectal Adenocarcinoma Gland) instance segmentation adapter.

Source: Warwick TIA MILD-Net dataset (Kaggle mirror hoinhnphm/crag-dataset),
213 H&E colorectal tiles with instance-level gland annotation.

Layout (under root_dir):
  raw/CRAG/{train_sup_16,train_unsup_137,val,test}/{image,mask}/<name>.png
    image = RGB H&E tile (~1512x1516)
    mask  = uint8 INSTANCE gland ids (0=bg, 1..K = distinct glands)

All 213 labeled image+mask pairs are exposed as the AL pool; the runner's make_split
produces the leakage-disjoint train/val/test. Masks are collapsed to a binary gland
target (id>0 -> 1). De-singletons the gland object alongside glas2015.

patient_id = per-tile: CRAG tiles carry no WSI id in their filenames, so WSI-level
grouping is unrecoverable; same-WSI tiles may land in different splits (a documented
limitation shared by the other histology tile sets here, not a bug).

Sample ID: "<split>__<stem>"; Image: (3, H, W) float32 in [0,1]; Mask: int64 {0,1}.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from medal_bench.data.base import MedALDataset, Sample

_SPLITS = ("train_sup_16", "train_unsup_137", "val", "test")


class CRAGAdapter(MedALDataset):
    name = "crag"
    modality = "histology"
    target = "gland"
    dim = "2d"
    query_unit = "image"
    num_classes = 2

    def __init__(self, root_dir: str):
        root = Path(root_dir) / "raw" / "CRAG"
        if not root.exists():
            raise FileNotFoundError(f"CRAG dir not found: {root}")
        self._index: list[tuple[str, Path, Path]] = []
        for split in _SPLITS:
            idir, mdir = root / split / "image", root / split / "mask"
            if not idir.exists():
                continue
            for ip in sorted(idir.glob("*.png")):
                mp = mdir / ip.name
                if not mp.exists():
                    raise FileNotFoundError(f"CRAG: missing mask {mp}")
                self._index.append((f"{split}__{ip.stem}", ip, mp))
        if not self._index:
            raise FileNotFoundError(f"No image/mask pairs under {root}")
        self._ids = [s for (s, _, _) in self._index]

    def __len__(self) -> int:
        return len(self._ids)

    def sample_ids(self) -> list[str]:
        return list(self._ids)

    def patient_ids(self) -> list[str]:
        return list(self._ids)  # per-tile (no WSI id available)

    def __getitem__(self, i: int) -> Sample:
        sid, ip, mp = self._index[i]
        img = np.asarray(Image.open(ip).convert("RGB")).astype(np.float32).transpose(2, 0, 1) / 255.0
        m = np.asarray(Image.open(mp))
        mask = (m > 0).astype(np.int64)   # instance ids -> binary gland
        return Sample(sample_id=sid, image=img, mask=mask, patient_id=sid,
                      meta={"image_path": str(ip)})
