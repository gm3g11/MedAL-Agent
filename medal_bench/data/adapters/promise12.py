"""PROMISE12 (prostate MRI segmentation) adapter — 3D volumes -> 2D slices.

Layout (under root_dir):
  extracted/Case{00..49}.mhd / .raw                  - T2 MR volume
  extracted/Case{00..49}_segmentation.mhd / .raw     - whole-prostate mask

Each volume Case_NN contributes Z slices, indexed (case=NN, z=0..Z-1).
sample_id = "Case{NN}_{z:03d}"; patient_id = "Case{NN}" so the runner can
enforce volume-level (== patient-level) train/val/test splits.

Image:  (H, W) float32, percentile-clipped [0.5%, 99.5%] then min-max -> [0, 1]
Mask:   (H, W) int64    in {0, 1}; bg=0, prostate=1
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from medal_bench.data.base import MedALDataset, Sample


_CASE_RE = re.compile(r"^Case(\d{2})\.mhd$")


class PROMISE12Adapter(MedALDataset):
    name = "promise12"
    modality = "mri"
    target = "prostate"
    dim = "3d"
    query_unit = "slice"
    num_classes = 2

    def __init__(self, root_dir: str):
        root = Path(root_dir) / "extracted"
        if not root.exists():
            raise FileNotFoundError(f"PROMISE12 extracted dir not found: {root}")
        self._root = root
        # Pre-scan: read each volume's header to get Z without decoding voxels.
        case_files = sorted(p for p in root.glob("Case??.mhd") if _CASE_RE.match(p.name))
        if not case_files:
            raise FileNotFoundError(f"No Case??.mhd files under {root}")
        self._index: list[tuple[str, int]] = []  # (case_id, z)
        self._case_depth: dict[str, int] = {}
        for cp in case_files:
            case_id = cp.stem  # "Case00"
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(cp))
            reader.ReadImageInformation()
            depth = reader.GetSize()[2]  # X, Y, Z -> Z
            self._case_depth[case_id] = depth
            for z in range(depth):
                self._index.append((case_id, z))
        self._ids = [f"{c}_{z:03d}" for (c, z) in self._index]

    def __len__(self) -> int:
        return len(self._ids)

    def sample_ids(self) -> list[str]:
        return list(self._ids)

    def patient_ids(self) -> list[str]:
        return [c for (c, _z) in self._index]

    @lru_cache(maxsize=32)
    def _load_volume(self, case_id: str) -> tuple[np.ndarray, np.ndarray]:
        img_v = sitk.GetArrayFromImage(sitk.ReadImage(str(self._root / f"{case_id}.mhd")))
        seg_v = sitk.GetArrayFromImage(
            sitk.ReadImage(str(self._root / f"{case_id}_segmentation.mhd"))
        )
        return img_v, seg_v  # both (Z, H, W)

    def __getitem__(self, i: int) -> Sample:
        case_id, z = self._index[i]
        img_v, seg_v = self._load_volume(case_id)
        img = img_v[z].astype(np.float32)
        lo, hi = np.percentile(img, [0.5, 99.5])
        img = np.clip(img, lo, hi)
        denom = max(float(hi - lo), 1e-6)
        img = ((img - lo) / denom).astype(np.float32)
        img = img[None]  # (1, H, W)
        mask = (seg_v[z] > 0).astype(np.int64)
        return Sample(
            sample_id=self._ids[i], image=img, mask=mask,
            patient_id=case_id, slice_index=z,
        )
