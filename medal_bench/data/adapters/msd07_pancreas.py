"""MSD Task07 Pancreas (CT pancreas + tumor segmentation) adapter — 3D -> 2D slices.

Layout (under root_dir):
  extracted/Task07_Pancreas/imagesTr/pancreas_NNN.nii.gz
  extracted/Task07_Pancreas/labelsTr/pancreas_NNN.nii.gz

Labels: 0=background, 1=pancreas, 2=tumor. (3-class segmentation.)

Each volume contributes Z axial slices; patient_id = volume basename
(e.g. "pancreas_001") so the runner can enforce volume-level splits.

Image: 16-bit CT HU clipped to a fixed soft-tissue window [-100, 240]
       -> normalized to [0, 1] then (1, H, W) float32.
Mask:  (H, W) int64 in {0, 1, 2}.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np

from medal_bench.data.base import MedALDataset, Sample


_HU_LO, _HU_HI = -100.0, 240.0


def _axial_axis(affine: np.ndarray) -> int:
    """Index of the superior-inferior (axial) axis; last axis if affine degenerate.
    (Mirrors mmwhs._axial_axis — robust to re-oriented NIfTI vs the hardcoded axis 2.)"""
    for ax, code in enumerate(nib.aff2axcodes(affine)):
        if code in ("S", "I"):
            return ax
    return 2


class MSD07PancreasAdapter(MedALDataset):
    name = "msd07_pancreas"
    modality = "ct"
    target = "pancreas_tumor"
    dim = "3d"
    query_unit = "slice"
    num_classes = 3

    def __init__(self, root_dir: str):
        root = Path(root_dir) / "extracted" / "Task07_Pancreas"
        if not root.exists():
            raise FileNotFoundError(f"MSD07 dir not found: {root}")
        self._img_dir = root / "imagesTr"
        self._lbl_dir = root / "labelsTr"
        vols = sorted(
            p.name for p in self._img_dir.glob("pancreas_*.nii.gz")
            if not p.name.startswith("._")
        )
        if not vols:
            raise FileNotFoundError(f"No pancreas_*.nii.gz in {self._img_dir}")
        self._index: list[tuple[str, int]] = []  # (volume_id, z)
        for fn in vols:
            vol_id = fn[:-len(".nii.gz")]  # "pancreas_001"
            # nibabel header gives shape without loading voxels; use the affine-derived
            # axial axis (not a hardcoded 2) so re-oriented volumes slice correctly.
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

    @lru_cache(maxsize=32)
    def _load_volume(self, vol_id: str) -> tuple[np.ndarray, np.ndarray]:
        img_nii = nib.load(str(self._img_dir / f"{vol_id}.nii.gz"))
        lbl_nii = nib.load(str(self._lbl_dir / f"{vol_id}.nii.gz"))
        # move each volume's affine-derived axial axis to front -> (Z, H, W). For the
        # current Task07 volumes (axial on axis 2) this == the old transpose(2,0,1).
        img = np.moveaxis(img_nii.get_fdata(), _axial_axis(img_nii.affine), 0)
        lbl = np.moveaxis(lbl_nii.get_fdata(), _axial_axis(lbl_nii.affine), 0)
        return img, lbl

    def __getitem__(self, i: int) -> Sample:
        vol_id, z = self._index[i]
        img_v, lbl_v = self._load_volume(vol_id)
        img = np.clip(img_v[z].astype(np.float32), _HU_LO, _HU_HI)
        img = (img - _HU_LO) / (_HU_HI - _HU_LO)
        img = img[None]  # (1, H, W)
        mask = lbl_v[z].astype(np.int64)
        return Sample(
            sample_id=self._ids[i], image=img, mask=mask,
            patient_id=vol_id, slice_index=z,
        )
