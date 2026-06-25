"""UMN DME OCT (retinal fluid segmentation) adapter — MATLAB .mat cell arrays.

Layout (under root_dir):
  raw/UMNDataset.mat        - University of Minnesota DME OCT dataset
  raw/ReadMe.pdf            - dataset description / citation

Structure of UMNDataset.mat (loadmat):
  AllSubjects   (1, 30) object array; AllSubjects[0, j] = (496, 1024, 25) uint8
                B-scan stack (25 B-scans per subject), intensities 0..255.
  ManualFluid1  (1, 29) object array; ManualFluid1[0, j] = (496, 1024, 25) uint8
                binary fluid mask {0, 1} from expert grader 1.
  ManualFluid2  (1, 29) second expert's annotation (unused; grader 1 is canonical).

There are 30 image cells but only 29 mask cells: subject index 29 (the 30th
subject) has no manual segmentation, so it is DROPPED. The remaining 29 subjects
contribute 25 B-scans each -> 725 slices. Many B-scans contain no fluid, so an
all-background mask is expected and valid (no fluid present in that slice).

sample_id = "subj{NN}_{z:02d}"; patient_id = "subj{NN}" so the runner enforces
subject-level (== patient-level) train/val/test splits and prevents leakage of
adjacent B-scans of the same eye across splits.

Image:  Grayscale -> (1, H, W) float32 in [0, 1]
Mask:   Binary fluid -> (H, W) int64 in {0, 1}; bg=0, retinal_fluid=1
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from medal_bench.data.base import MedALDataset, Sample


class UMNOCTAdapter(MedALDataset):
    name = "umn_oct"
    modality = "oct"
    target = "retinal_fluid"
    dim = "2d"
    query_unit = "slice"
    num_classes = 2

    def __init__(self, root_dir: str):
        self._mat_path = Path(root_dir) / "raw" / "UMNDataset.mat"
        if not self._mat_path.exists():
            raise FileNotFoundError(f"UMN OCT .mat not found: {self._mat_path}")
        # Read once to learn how many masked subjects/slices exist, then discard;
        # __getitem__ reloads lazily via the cached loader (loadmat is ~263 MB).
        m = loadmat(str(self._mat_path))
        imgs = m["AllSubjects"]      # (1, 30)
        masks = m["ManualFluid1"]    # (1, 29)
        n_subj = masks.shape[1]      # 29 (drop image-only subject index 29)
        self._index: list[tuple[int, int]] = []  # (subject_idx, z)
        for j in range(n_subj):
            vol = imgs[0, j]
            depth = vol.shape[2]     # 25 B-scans
            for z in range(depth):
                self._index.append((j, z))
        self._ids = [f"subj{j:02d}_{z:02d}" for (j, z) in self._index]

    def __len__(self) -> int:
        return len(self._ids)

    def sample_ids(self) -> list[str]:
        return list(self._ids)

    def patient_ids(self) -> list[str]:
        return [f"subj{j:02d}" for (j, _z) in self._index]

    @lru_cache(maxsize=1)
    def _load_mat(self):
        m = loadmat(str(self._mat_path))
        return m["AllSubjects"], m["ManualFluid1"]

    def __getitem__(self, i: int) -> Sample:
        j, z = self._index[i]
        imgs, masks = self._load_mat()
        img = imgs[0, j][:, :, z].astype(np.float32) / 255.0
        img = img[None]  # (1, H, W)
        mask = (masks[0, j][:, :, z] > 0).astype(np.int64)  # (H, W) in {0, 1}
        return Sample(
            sample_id=self._ids[i], image=img, mask=mask,
            patient_id=f"subj{j:02d}", slice_index=z,
            meta={"subject_idx": j},
        )
