"""Duke DME (Chiu 2015) retinal-fluid OCT segmentation adapter — .mat -> 2D B-scans.

Layout (under root_dir):
  extracted/2015_BOE_Chiu/Subject_{01..10}.mat

Each subject's .mat (scipy.io.loadmat) holds an OCT volume of 61 B-scans:
  images       : (496, 768, 61) uint8   - B-scan intensities (all 61 finite)
  manualFluid1 : (496, 768, 61) float64 - grader1's fluid annotation

Only 11 of the 61 B-scans per subject are manually annotated (10 subjects x 11
= 110 labeled slices). Un-annotated slices are entirely NaN in manualFluid1;
annotated slices are fully finite. We expose ONLY the 110 annotated slices.

manualFluid1 stores fluid as instance/region ids (0=background, 1..N=fluid
regions), so we binarize ``>0 -> 1`` to get a 2-class (bg / fluid) mask. Some
annotated slices have no fluid (legitimately all-background) — expected.

Sample ID:   "Subject_NN_z{zz}"  e.g. "Subject_01_z010"
Image:       Grayscale B-scan -> (1, H, W) float32 in [0, 1] (uint8 / 255)
Mask:        Binarized fluid  -> (H, W) int64 in {0, 1}; bg=0, fluid=1
patient_id:  "Subject_NN"  (subject-disjoint splits)
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.io as sio

from medal_bench.data.base import MedALDataset, Sample


_SUBJ_RE = re.compile(r"^Subject_(\d{2})\.mat$")


class DukeDMEChiu2015Adapter(MedALDataset):
    name = "duke_dme_chiu2015"
    modality = "oct"
    target = "retinal_fluid"
    dim = "3d"
    query_unit = "slice"
    num_classes = 2

    def __init__(self, root_dir: str):
        root = Path(root_dir) / "extracted" / "2015_BOE_Chiu"
        if not root.exists():
            raise FileNotFoundError(f"Duke DME extracted dir not found: {root}")
        self._root = root
        subj_files = sorted(p for p in root.glob("Subject_??.mat") if _SUBJ_RE.match(p.name))
        if not subj_files:
            raise FileNotFoundError(f"No Subject_??.mat files under {root}")
        # Pre-scan: find which slices each subject has a grader-1 fluid annotation
        # for (un-annotated slices are entirely NaN in manualFluid1).
        self._index: list[tuple[str, int]] = []  # (subject_id, z)
        for sp in subj_files:
            subject_id = sp.stem  # "Subject_01"
            mf = sio.loadmat(str(sp))["manualFluid1"]  # (H, W, Z) float64
            annotated = [z for z in range(mf.shape[2]) if np.isfinite(mf[:, :, z]).any()]
            for z in annotated:
                self._index.append((subject_id, z))
        if not self._index:
            raise RuntimeError(f"No annotated slices found under {root}")
        self._ids = [f"{s}_z{z:03d}" for (s, z) in self._index]

    def __len__(self) -> int:
        return len(self._ids)

    def sample_ids(self) -> list[str]:
        return list(self._ids)

    def patient_ids(self) -> list[str]:
        # subject id is the grouping unit for leakage-disjoint splits
        return [s for (s, _z) in self._index]

    @lru_cache(maxsize=4)
    def _load_subject(self, subject_id: str) -> tuple[np.ndarray, np.ndarray]:
        d = sio.loadmat(str(self._root / f"{subject_id}.mat"))
        return d["images"], d["manualFluid1"]  # (H, W, Z) uint8, (H, W, Z) float64

    def __getitem__(self, i: int) -> Sample:
        subject_id, z = self._index[i]
        imgs, mf = self._load_subject(subject_id)
        img = imgs[:, :, z].astype(np.float32) / 255.0
        img = img[None]  # (1, H, W)
        # NaN -> 0; fluid region ids (>0) -> foreground
        mask = (np.nan_to_num(mf[:, :, z], nan=0.0) > 0).astype(np.int64)
        return Sample(
            sample_id=self._ids[i], image=img, mask=mask,
            patient_id=subject_id, slice_index=z,
        )
