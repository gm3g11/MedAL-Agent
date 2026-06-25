"""MSD Task05 Prostate (T2 + ADC MRI prostate-zone segmentation) adapter — 3D -> 2D slices.

Layout (under root_dir):
  raw/Task05_Prostate/imagesTr/prostate_NN.nii.gz   - 4D MR: (H, W, Z, modality)
  raw/Task05_Prostate/labelsTr/prostate_NN.nii.gz   - 3D label: (H, W, Z)

The image NIfTI carries two co-registered MR modalities (0=T2, 1=ADC). Per spec
we use the FIRST modality (T2, channel 0) as a 1-channel image.

Labels: 0=background, 1=PZ (peripheral zone), 2=TZ (transition zone). (3-class.)

Each volume contributes Z axial slices (mirrors msd07_pancreas: ALL slices are
indexed, including all-background ones). patient_id = volume basename
(e.g. "prostate_00") so the runner can enforce volume-level (leakage-disjoint)
splits.

Image: T2 MR has no fixed intensity window (unlike CT HU), so each volume's T2
       channel is normalized by its own 99.5th-percentile to [0, 1] (clipped),
       then (1, H, W) float32. Per-volume + deterministic.
Mask:  (H, W) int64 in {0, 1, 2}.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np

from medal_bench.data.base import MedALDataset, Sample


def _axial_axis(affine: np.ndarray) -> int:
    """Index of the superior-inferior (axial) axis among the 3 spatial axes;
    last spatial axis if affine degenerate. (Mirrors msd07_pancreas._axial_axis.)"""
    for ax, code in enumerate(nib.aff2axcodes(affine)):
        if code in ("S", "I"):
            return ax
    return 2


class MSDTask05ProstateAdapter(MedALDataset):
    name = "msd_task05_prostate"
    modality = "mri"
    target = "prostate_zones"
    dim = "3d"
    query_unit = "slice"
    num_classes = 3

    def __init__(self, root_dir: str):
        root = Path(root_dir) / "raw" / "Task05_Prostate"
        if not root.exists():
            raise FileNotFoundError(f"MSD05 dir not found: {root}")
        self._img_dir = root / "imagesTr"
        self._lbl_dir = root / "labelsTr"
        vols = sorted(
            p.name for p in self._img_dir.glob("prostate_*.nii.gz")
            if not p.name.startswith("._")
        )
        if not vols:
            raise FileNotFoundError(f"No prostate_*.nii.gz in {self._img_dir}")
        self._index: list[tuple[str, int]] = []  # (volume_id, z)
        for fn in vols:
            vol_id = fn[:-len(".nii.gz")]  # "prostate_00"
            # header gives shape without loading voxels; affine-derived axial axis
            # (over the 3 spatial axes) so re-oriented volumes slice correctly.
            nii = nib.load(str(self._img_dir / fn))
            depth = int(nii.header.get_data_shape()[_axial_axis(nii.affine)])
            for z in range(depth):
                self._index.append((vol_id, z))
        self._ids = [f"{v}_{z:04d}" for (v, z) in self._index]

    def __len__(self) -> int:
        return len(self._ids)

    def sample_ids(self) -> list[str]:
        return list(self._ids)

    def patient_ids(self) -> list[str]:
        return [v for (v, _z) in self._index]

    @lru_cache(maxsize=32)   # cover all 32 volumes (was 8 -> ~14x re-decode under random access)
    def _load_volume(self, vol_id: str) -> tuple[np.ndarray, np.ndarray]:
        img_nii = nib.load(str(self._img_dir / f"{vol_id}.nii.gz"))
        lbl_nii = nib.load(str(self._lbl_dir / f"{vol_id}.nii.gz"))
        img = img_nii.get_fdata()          # (H, W, Z, modality)
        img = img[..., 0]                  # first modality = T2 -> (H, W, Z)
        # move the affine-derived axial axis to front -> (Z, H, W).
        img = np.moveaxis(img, _axial_axis(img_nii.affine), 0)
        lbl = np.moveaxis(lbl_nii.get_fdata(), _axial_axis(lbl_nii.affine), 0)
        # per-volume robust normalization of T2 to [0, 1] (no fixed MR window).
        hi = np.percentile(img, 99.5)
        if hi <= 0:
            hi = float(img.max()) or 1.0
        img = np.clip(img / hi, 0.0, 1.0).astype(np.float32)
        return img, lbl

    def __getitem__(self, i: int) -> Sample:
        vol_id, z = self._index[i]
        img_v, lbl_v = self._load_volume(vol_id)
        img = img_v[z][None]  # (1, H, W) float32 in [0, 1]
        mask = lbl_v[z].astype(np.int64)
        return Sample(
            sample_id=self._ids[i], image=img, mask=mask,
            patient_id=vol_id, slice_index=z,
        )
