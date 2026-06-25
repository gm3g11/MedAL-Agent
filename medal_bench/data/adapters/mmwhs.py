"""MMWHS (Multi-Modality Whole Heart Segmentation) adapter — 3D -> 2D slices.

Layout (under root_dir):
  extracted/Wholeheart_Train_Dataset/{A,B,G} ct_train/CaseNNNN_{image,label}.nii.gz   (CT, 60 cases)
  extracted/Wholeheart_Train_Dataset/{C and D,E} mr_train/CaseNNNN_{image,label}.nii.gz (MR, 46 cases)

Wired as TWO registry entries (``mmwhs_ct``, ``mmwhs_mr``) sharing this class:
CT and MR have different intensity statistics and would confound a single AL
pool, so they are kept separate. Same label remap for both.

Native label codes {0,205,420,421,500,550,600,820,850} -> dense {0..7} via
MMWHS_REMAP (8 classes). Orientation varies per case (CT axial on axis 2; MR
axial on axis 1) and 2 CT cases (2009, 2017) have degenerate all-zero affines.
We pick the axial (S/I) axis from the affine's axis codes and slice along it,
falling back to the last axis when the affine is degenerate. (In-plane flips are
not canonicalized — image+mask always flip together, and this is a slice-level
benchmark; revisit only if cross-volume feature consistency demands it.)

Image: CT clipped to a fixed HU window [-200, 800] -> [0,1]; MR per-volume
percentile [0.5, 99.5] -> [0,1]. Output (1, H, W) float32. Mask (H, W) int64 in {0..7}.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np

from medal_bench.data.base import MedALDataset, Sample
from medal_bench.data.remap import LabelRemapper, MMWHS_REMAP

_CT_HU_LO, _CT_HU_HI = -200.0, 800.0
_MR_PLO, _MR_PHI = 0.5, 99.5


def _axial_axis(affine: np.ndarray) -> int:
    """Index of the superior-inferior (axial) axis; last axis if affine degenerate."""
    for ax, code in enumerate(nib.aff2axcodes(affine)):
        if code in ("S", "I"):
            return ax
    return 2


class MMWHSAdapter(MedALDataset):
    target = "whole_heart"
    dim = "3d"
    query_unit = "slice"
    num_classes = 8

    def __init__(self, root_dir: str, modality: str):
        if modality not in ("ct", "mr"):
            raise ValueError(f"modality must be 'ct' or 'mr', got {modality!r}")
        self.modality = modality
        self.name = f"mmwhs_{modality}"
        self._remap = LabelRemapper(MMWHS_REMAP, name=self.name)
        root = Path(root_dir) / "extracted" / "Wholeheart_Train_Dataset"
        if not root.exists():
            raise FileNotFoundError(f"MMWHS dir not found: {root}")
        tag = f"{modality}_train"
        imgs = sorted(
            p for p in root.glob(f"*{tag}*/*_image.nii.gz")
            if not p.name.startswith("._")
        )
        if not imgs:
            raise FileNotFoundError(f"No *_image.nii.gz for modality={modality} under {root}")
        self._img_paths: dict[str, Path] = {}
        self._lbl_paths: dict[str, Path] = {}
        self._index: list[tuple[str, int]] = []  # (case_id, z)
        for ip in imgs:
            case = ip.name[:-len("_image.nii.gz")]  # "Case1001"
            lp = ip.with_name(f"{case}_label.nii.gz")
            if not lp.exists():
                raise FileNotFoundError(f"missing label for {case}: {lp}")
            self._img_paths[case] = ip
            self._lbl_paths[case] = lp
            nii = nib.load(str(ip))
            depth = int(nii.header.get_data_shape()[_axial_axis(nii.affine)])
            for z in range(depth):
                self._index.append((case, z))
        self._ids = [f"{c}_{z:03d}" for (c, z) in self._index]

    def __len__(self) -> int:
        return len(self._ids)

    def sample_ids(self) -> list[str]:
        return list(self._ids)

    def patient_ids(self) -> list[str]:
        return [c for (c, _z) in self._index]

    @lru_cache(maxsize=8)
    def _load_volume(self, case: str) -> tuple[np.ndarray, np.ndarray]:
        img_nii = nib.load(str(self._img_paths[case]))
        lbl_nii = nib.load(str(self._lbl_paths[case]))
        # move the axial axis to front -> (Z, H, W)
        img = np.moveaxis(img_nii.get_fdata().astype(np.float32), _axial_axis(img_nii.affine), 0)
        lbl = self._remap.apply(np.moveaxis(lbl_nii.get_fdata(), _axial_axis(lbl_nii.affine), 0))
        if self.modality == "ct":
            img = np.clip(img, _CT_HU_LO, _CT_HU_HI)
            img = (img - _CT_HU_LO) / (_CT_HU_HI - _CT_HU_LO)
        else:  # mr: per-volume percentile normalization
            lo, hi = np.percentile(img, [_MR_PLO, _MR_PHI])
            hi = max(hi, lo + 1e-6)
            img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
        return img.astype(np.float32), lbl

    def __getitem__(self, i: int) -> Sample:
        case, z = self._index[i]
        img_v, lbl_v = self._load_volume(case)
        return Sample(
            sample_id=self._ids[i], image=img_v[z][None], mask=lbl_v[z].astype(np.int64),
            patient_id=case, slice_index=z,
        )
