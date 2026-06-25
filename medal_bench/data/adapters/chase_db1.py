"""CHASE_DB1 (retinal vessel segmentation in fundus images) adapter.

Layout (under root_dir):
  raw/CHASEDB1/CHASEDB1/Image_NNL.jpg / Image_NNR.jpg   - RGB fundus image
  raw/CHASEDB1/CHASEDB1/Image_NN{L,R}_1stHO.png          - GT vessel mask (used)
  raw/CHASEDB1/CHASEDB1/Image_NN{L,R}_2ndHO.png          - 2nd observer (ignored)

28 images = both eyes (L, R) of 14 children (NN = 01..14). The first human
observer mask (_1stHO) is the ground truth; the _2ndHO second-observer mask is
ignored.

To keep splits leakage-disjoint, both eyes of one child share a patient group,
exposed via ``patient_ids()`` (the child id "NN") -> 14 groups for 28 images.

Sample ID:  "Image_NNL" / "Image_NNR" e.g. "Image_01L"
Image:      RGB -> (3, H, W) float32 in [0, 1]
Mask:       PIL mode '1' / L binarized -> (H, W) int64 in {0, 1}; bg=0, vessel=1
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

from medal_bench.data.base import MedALDataset, Sample


# matches "Image_01L.jpg" -> child id "01", eye "L"
_IMG_RE = re.compile(r"^Image_(\d{2})([LR])\.jpg$")


class ChaseDB1Adapter(MedALDataset):
    name = "chase_db1"
    modality = "fundus"
    target = "retinal_vessels"
    dim = "2d"
    query_unit = "image"
    num_classes = 2

    def __init__(self, root_dir: str):
        root = Path(root_dir) / "raw" / "CHASEDB1" / "CHASEDB1"
        if not root.exists():
            raise FileNotFoundError(f"CHASE_DB1 dir not found: {root}")
        # (child_id, image_path, mask_path)
        self._index: list[tuple[str, Path, Path]] = []
        for ip in sorted(root.glob("Image_*.jpg")):
            m = _IMG_RE.match(ip.name)
            if not m:
                continue
            child = m.group(1)
            mp = root / f"{ip.stem}_1stHO.png"
            if not mp.exists():
                raise FileNotFoundError(f"CHASE_DB1: no _1stHO mask {mp} for image {ip}")
            self._index.append((child, ip, mp))
        if not self._index:
            raise FileNotFoundError(f"No Image_*.jpg pairs under {root}")
        self._ids = [ip.stem for (_child, ip, _mp) in self._index]

    def __len__(self) -> int:
        return len(self._index)

    def sample_ids(self) -> list[str]:
        return list(self._ids)

    def patient_ids(self) -> list[str]:
        # child id ("NN") groups both eyes of one subject for leakage-disjoint splits
        return [child for (child, _, _) in self._index]

    def __getitem__(self, i: int) -> Sample:
        child, ip, mp = self._index[i]
        img = np.asarray(Image.open(ip).convert("RGB")).astype(np.float32).transpose(2, 0, 1) / 255.0
        m = np.asarray(Image.open(mp).convert("L"))
        mask = (m > 0).astype(np.int64)
        return Sample(
            sample_id=self._ids[i], image=img, mask=mask,
            patient_id=child, meta={"child": child, "image_path": str(ip)},
        )
