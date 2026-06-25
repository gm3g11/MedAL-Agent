"""DRIVE (retinal vessel segmentation in fundus images) adapter.

Layout (under root_dir):
  raw/DRIVE/training/images/<NN>_training.tif      - RGB fundus image
  raw/DRIVE/training/1st_manual/<NN>_manual1.gif    - binary vessel annotation
  raw/DRIVE/training/mask/<NN>_training_mask.gif    - FOV mask (IGNORED)

Only the ~20 *training* images carry a ground-truth vessel mask (the DRIVE
test split ships no 1st_manual labels), so this adapter exposes the training
split only.

Image and mask share the 2-digit case id ``NN`` but use different filename
stems (``NN_training`` vs ``NN_manual1``), so they are paired by the parsed
``NN`` id, not by stem. The ``mask/`` field-of-view dir is deliberately ignored.

Each fundus image is a distinct subject, so ``patient_id = NN`` and the runner
gets one group per case (== one group per sample here; DRIVE has no
multi-image patients, so grouping equals identity and no cross-frame leakage
is possible).

Sample ID:  "<NN>"  e.g. "21"
Image:      RGB fundus     -> (3, H, W) float32 in [0, 1]
Mask:       Binary vessels -> (H, W)    int64   in {0, 1}; bg=0, vessel=1
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

from medal_bench.data.base import MedALDataset, Sample


# "21_training.tif" -> NN = "21"
_IMG_RE = re.compile(r"^(\d{2})_training\.tif$", re.IGNORECASE)


class DRIVEAdapter(MedALDataset):
    name = "drive"
    modality = "fundus"
    target = "retinal_vessels"
    dim = "2d"
    query_unit = "image"
    num_classes = 2

    def __init__(self, root_dir: str):
        train = Path(root_dir) / "raw" / "DRIVE" / "training"
        img_dir = train / "images"
        man_dir = train / "1st_manual"
        if not img_dir.exists() or not man_dir.exists():
            raise FileNotFoundError(
                f"DRIVE missing: {img_dir} or {man_dir}"
            )
        # index NN -> manual mask path, then pair against images by NN
        manuals: dict[str, Path] = {}
        for p in man_dir.iterdir():
            m = re.match(r"^(\d{2})_manual1\.gif$", p.name, re.IGNORECASE)
            if m:
                manuals[m.group(1)] = p

        self._index: list[tuple[str, Path, Path]] = []  # (NN, img_path, mask_path)
        for p in sorted(img_dir.iterdir()):
            m = _IMG_RE.match(p.name)
            if not m:
                continue
            nn = m.group(1)
            mp = manuals.get(nn)
            if mp is None:
                raise FileNotFoundError(
                    f"DRIVE: image {p.name} has no matching 1st_manual/{nn}_manual1.gif"
                )
            self._index.append((nn, p, mp))
        if not self._index:
            raise FileNotFoundError(f"DRIVE: no <NN>_training.tif images under {img_dir}")
        # stable order by integer case id
        self._index.sort(key=lambda r: int(r[0]))
        self._ids = [nn for (nn, _, _) in self._index]

    def __len__(self) -> int:
        return len(self._ids)

    def sample_ids(self) -> list[str]:
        return list(self._ids)

    def patient_ids(self) -> list[str]:
        # each fundus image is its own subject; case id is the grouping unit
        return [nn for (nn, _, _) in self._index]

    def __getitem__(self, i: int) -> Sample:
        nn, img_path, mask_path = self._index[i]
        img = np.asarray(Image.open(img_path).convert("RGB"))
        img = img.astype(np.float32).transpose(2, 0, 1) / 255.0  # (3, H, W)
        m = np.asarray(Image.open(mask_path).convert("L"))
        mask = (m > 0).astype(np.int64)  # bg=0, vessel=1
        return Sample(
            sample_id=nn, image=img, mask=mask,
            patient_id=nn, meta={"case": nn},
        )
