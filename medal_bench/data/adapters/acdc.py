"""ACDC (cardiac MRI segmentation) adapter — pre-sliced 2D slices from h5.

Layout (under root_dir):
  extracted/ACDC_h5/data/slices/patientNNN_frameNN_slice_K.h5

Each .h5 holds two datasets, both already a single 2D slice:
  'image' (H, W) float  - cine MR slice, already scaled to ~[0, 1]
  'label' (H, W) uint8  - dense class indices in {0, 1, 2, 3}

ACDC contains 100 patients; each has an ED and an ES frame (frame01/frame02),
and each frame is a short-axis stack sliced into K 2D slices. To prevent
train/val/test leakage, ``patient_ids()`` strips ``_frameNN_slice_K`` and
groups ED + ES + all SAX slices of a patient under "patientNNN".

sample_id = "patientNNN_frameNN_slice_K"; patient_id = "patientNNN".

Image:  (1, H, W) float32 in [0, 1] (source dtype cast only; no re-scaling)
Mask:   (H, W)    int64   in {0, 1, 2, 3}
        0=bg, 1=RV_cavity, 2=LV_myocardium, 3=LV_cavity
"""
from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np

from medal_bench.data.base import MedALDataset, Sample


# patient001_frame01_slice_10 -> ("patient001", 1, 10)
_SLICE_RE = re.compile(r"^(patient\d+)_frame(\d+)_slice_(\d+)$")


class ACDCAdapter(MedALDataset):
    name = "acdc"
    modality = "cardiac_mri"
    target = "cardiac_structures"
    dim = "3d"
    query_unit = "slice"
    num_classes = 4

    def __init__(self, root_dir: str):
        root = Path(root_dir) / "extracted" / "ACDC_h5" / "data" / "slices"
        if not root.exists():
            raise FileNotFoundError(f"ACDC slices dir not found: {root}")
        self._root = root
        # (path, patient_id, frame, slice_idx)
        index: list[tuple[Path, str, int, int]] = []
        for p in root.glob("*.h5"):
            m = _SLICE_RE.match(p.stem)
            if not m:
                continue
            index.append((p, m.group(1), int(m.group(2)), int(m.group(3))))
        if not index:
            raise FileNotFoundError(f"No patient*_frame*_slice_*.h5 under {root}")
        # stable, human-meaningful order: (patient, frame, integer slice)
        index.sort(key=lambda r: (r[1], r[2], r[3]))
        self._index = index
        self._ids = [p.stem for (p, _pid, _f, _s) in index]

    def __len__(self) -> int:
        return len(self._ids)

    def sample_ids(self) -> list[str]:
        return list(self._ids)

    def patient_ids(self) -> list[str]:
        return [pid for (_p, pid, _f, _s) in self._index]

    def __getitem__(self, i: int) -> Sample:
        path, pid, _frame, sidx = self._index[i]
        with h5py.File(path, "r") as h:
            img = h["image"][:].astype(np.float32)
            mask = h["label"][:].astype(np.int64)
        img = img[None]  # (1, H, W)
        return Sample(
            sample_id=self._ids[i], image=img, mask=mask,
            patient_id=pid, slice_index=sidx,
        )
