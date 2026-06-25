"""TNBC (Triple-Negative Breast Cancer nuclei segmentation) adapter.

Source: Zenodo record 1175282, TNBC_NucleiSegmentation.zip.

Layout (under root_dir):
  extracted/TNBC_NucleiSegmentation/Slide_NN/<file>.png   - H&E image  (RGBA)
  extracted/TNBC_NucleiSegmentation/GT_NN/<file>.png      - nuclei mask (L)

Each Slide_NN folder is one patient/slide; its images are paired to masks in
GT_NN by an identical filename. ~50 image+mask pairs across 11 slides.

To keep train/val/test splits leakage-disjoint at the slide level, the slide
folder is exposed via ``patient_ids()`` (sequence/case grouping unit).

Sample ID:  "Slide_NN_<stem>" e.g. "Slide_01_01_1"
Image:      RGBA -> RGB -> (3, H, W) float32 in [0, 1]
Mask:       L (0/255) binarized -> (H, W) int64 in {0, 1}; bg=0, nucleus=1
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

from medal_bench.data.base import MedALDataset, Sample


_SLIDE_RE = re.compile(r"^Slide_(\d+)$")
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class TNBCAdapter(MedALDataset):
    name = "tnbc"
    modality = "histology"
    target = "nucleus"
    dim = "2d"
    query_unit = "image"
    num_classes = 2

    def __init__(self, root_dir: str):
        root = Path(root_dir) / "extracted" / "TNBC_NucleiSegmentation"
        if not root.exists():
            raise FileNotFoundError(f"TNBC dir not found: {root}")
        # (slide_id, image_path, mask_path)
        self._index: list[tuple[str, Path, Path]] = []
        slide_dirs = sorted(
            (p for p in root.iterdir() if p.is_dir() and _SLIDE_RE.match(p.name)),
            key=lambda p: int(_SLIDE_RE.match(p.name).group(1)),
        )
        if not slide_dirs:
            raise FileNotFoundError(f"No Slide_NN dirs under {root}")
        for sdir in slide_dirs:
            nn = _SLIDE_RE.match(sdir.name).group(1)
            gdir = root / f"GT_{nn}"
            if not gdir.exists():
                raise FileNotFoundError(f"TNBC: missing mask dir {gdir} for {sdir.name}")
            for ip in sorted(p for p in sdir.iterdir() if p.suffix.lower() in _IMG_EXTS):
                mp = gdir / ip.name
                if not mp.exists():
                    raise FileNotFoundError(f"TNBC: no mask {mp} for image {ip}")
                self._index.append((sdir.name, ip, mp))
        self._ids = [f"{slide}_{ip.stem}" for (slide, ip, _) in self._index]

    def __len__(self) -> int:
        return len(self._ids)

    def sample_ids(self) -> list[str]:
        return list(self._ids)

    def patient_ids(self) -> list[str]:
        # slide folder is the patient/case grouping unit for splits
        return [slide for (slide, _, _) in self._index]

    def __getitem__(self, i: int) -> Sample:
        slide, ip, mp = self._index[i]
        img = np.asarray(Image.open(ip).convert("RGB")).astype(np.float32).transpose(2, 0, 1) / 255.0
        m = np.asarray(Image.open(mp).convert("L"))
        mask = (m > 0).astype(np.int64)
        return Sample(
            sample_id=self._ids[i], image=img, mask=mask,
            patient_id=slide, meta={"slide": slide, "image_path": str(ip)},
        )
