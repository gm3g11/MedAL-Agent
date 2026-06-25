"""NLM Montgomery (chest X-ray lung-field segmentation) adapter.

Layout (under root_dir):
  extracted/MontgomerySet/CXR_png/MCUCXR_{NNNN}_{c}.png        - chest X-ray
  extracted/MontgomerySet/ManualMask/leftMask/MCUCXR_*.png      - left-lung mask
  extracted/MontgomerySet/ManualMask/rightMask/MCUCXR_*.png     - right-lung mask

Each image has its left- and right-lung masks in SEPARATE directories; they are
OR-merged into one binary lung-field mask (model on the BUSI multi-mask merge).

Each X-ray is a distinct patient, so this is a native-2D dataset: there is no
sub-image grouping and ``patient_id`` is the image stem. (No frames/slices share
a patient, so splits are already leakage-disjoint at image granularity.)

Sample ID:  image stem, e.g. "MCUCXR_0001_0"
Image:      Grayscale -> (1, H, W) float32 in [0, 1]
Mask:       OR-merged left|right -> (H, W) int64 in {0, 1}; bg=0, lung=1
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from medal_bench.data.base import MedALDataset, Sample


class NLMMontgomeryAdapter(MedALDataset):
    name = "nlm_montgomery"
    modality = "chest_xray"
    target = "lung_field"
    dim = "2d"
    query_unit = "image"
    num_classes = 2

    def __init__(self, root_dir: str):
        root = Path(root_dir) / "extracted" / "MontgomerySet"
        self._img_dir = root / "CXR_png"
        self._left_dir = root / "ManualMask" / "leftMask"
        self._right_dir = root / "ManualMask" / "rightMask"
        for d in (self._img_dir, self._left_dir, self._right_dir):
            if not d.exists():
                raise FileNotFoundError(f"NLM Montgomery dir not found: {d}")

        # (stem, img_path, left_path, right_path); require BOTH masks per image.
        self._index: list[tuple[str, Path, Path, Path]] = []
        for p in sorted(self._img_dir.glob("*.png")):
            stem = p.stem
            lp = self._left_dir / f"{stem}.png"
            rp = self._right_dir / f"{stem}.png"
            if not lp.exists() or not rp.exists():
                raise FileNotFoundError(
                    f"NLM Montgomery: missing mask(s) for {stem} "
                    f"(left={lp.exists()}, right={rp.exists()})"
                )
            self._index.append((stem, p, lp, rp))
        if not self._index:
            raise FileNotFoundError(f"NLM Montgomery: no PNGs under {self._img_dir}")
        self._ids = [stem for (stem, _, _, _) in self._index]

    def __len__(self) -> int:
        return len(self._ids)

    def sample_ids(self) -> list[str]:
        return list(self._ids)

    def patient_ids(self) -> list[str]:
        # one X-ray == one patient; stem is the grouping unit for splits
        return list(self._ids)

    def __getitem__(self, i: int) -> Sample:
        stem, img_path, lp, rp = self._index[i]
        img = np.asarray(Image.open(img_path).convert("L")).astype(np.float32) / 255.0
        img = img[None]  # (1, H, W)
        H, W = img.shape[-2:]
        left = np.asarray(Image.open(lp).convert("L"))
        right = np.asarray(Image.open(rp).convert("L"))
        mask = ((left > 0) | (right > 0)).astype(np.int64)  # (H, W) in {0, 1}
        return Sample(
            sample_id=stem, image=img, mask=mask,
            patient_id=stem,
        )
