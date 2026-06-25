"""CREMI (MICCAI 2016 Circuit Reconstruction from EM Images) adapter -> 2D membrane slices.

Source: cremi.org (HHMI Janelia; research-use, cite the CREMI challenge). 3 independent specimens
A/B/C, each a 125-slice ssTEM volume (1250x1250, 4x4x40 nm), stored as HDF5. The labels are neuron
INSTANCE IDs (/volumes/labels/neuron_ids); we convert them to neuronal MEMBRANE boundaries (a pixel
is membrane iff a 4-neighbour has a different neuron id), matching isbi2012_em's target="membrane".

The 3 specimens are SPECIMEN-DISJOINT leakage groups (patient_id = "cremi_A/B/C"), so make_split puts
whole specimens in train/val/test -- no within-specimen adjacency leakage (unlike a single-stack EM
set). 375 slices, specimen-grouped -> the budget grid yields multiple acquisition rounds.

Layout (under root_dir):  raw/sample_A.hdf, raw/sample_B.hdf, raw/sample_C.hdf
  each HDF5: /volumes/raw (125,1250,1250) uint8 ; /volumes/labels/neuron_ids (125,1250,1250) uint64

Sample ID:  "A_z000" .. "C_z124"
Image:      grayscale -> (1, H, W) float32 in [0, 1]
Mask:       (H, W) int64 in {0, 1}; bg=0, membrane=1 (dilated 2px so thin boundaries survive resize)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import h5py

from medal_bench.data.base import MedALDataset, Sample

_SENTINEL = np.uint64(0xFFFFFFFFFFFFFFFF)   # CREMI "transparent"/unlabeled marker


def _membrane(seg: np.ndarray) -> np.ndarray:
    """neuron-id map (H,W) -> dense binary membrane mask {0,1}, 4-conn boundary dilated 2px."""
    seg = seg.astype(np.uint64)
    b = np.zeros(seg.shape, dtype=bool)
    diff_v = seg[:-1, :] != seg[1:, :]
    b[:-1, :] |= diff_v; b[1:, :] |= diff_v
    diff_h = seg[:, :-1] != seg[:, 1:]
    b[:, :-1] |= diff_h; b[:, 1:] |= diff_h
    for _ in range(2):                                   # 4-conn dilation x2 (~thicker membrane)
        d = b.copy()
        d[:-1, :] |= b[1:, :]; d[1:, :] |= b[:-1, :]
        d[:, :-1] |= b[:, 1:]; d[:, 1:] |= b[:, :-1]
        b = d
    b[seg == _SENTINEL] = False                          # unlabeled voxels -> background
    return b.astype(np.int64)


class CREMIAdapter(MedALDataset):
    name = "cremi"
    modality = "electron_microscopy"
    target = "membrane"
    dim = "2d"
    query_unit = "image"
    num_classes = 2

    def __init__(self, root_dir: str):
        raw = Path(root_dir) / "raw"
        self._index: list[tuple[str, str, Path, int]] = []   # (sample_id, specimen, hdf, z)
        for s in ("A", "B", "C"):
            hp = raw / f"sample_{s}.hdf"
            if not hp.exists():
                raise FileNotFoundError(f"CREMI file not found: {hp}")
            with h5py.File(hp, "r") as f:
                nz = f["volumes/raw"].shape[0]
            for z in range(nz):
                self._index.append((f"{s}_z{z:03d}", f"cremi_{s}", hp, z))
        if not self._index:
            raise FileNotFoundError(f"CREMI: no slices found under {raw}")
        self._ids = [sid for (sid, _, _, _) in self._index]
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}   # hdf -> (raw_u8, membrane_u8)

    def __len__(self) -> int:
        return len(self._index)

    def sample_ids(self) -> list[str]:
        return list(self._ids)

    def patient_ids(self) -> list[str]:
        return [spec for (_, spec, _, _) in self._index]

    def _load(self, hp: Path):
        key = str(hp)
        if key not in self._cache:
            with h5py.File(hp, "r") as f:
                raw = np.asarray(f["volumes/raw"][:])                      # (Z,H,W) uint8
                seg = np.asarray(f["volumes/labels/neuron_ids"][:])        # (Z,H,W) uint64
            mem = np.stack([_membrane(seg[z]) for z in range(seg.shape[0])]).astype(np.uint8)
            self._cache[key] = (raw, mem)
        return self._cache[key]

    def __getitem__(self, i: int) -> Sample:
        sid, spec, hp, z = self._index[i]
        raw, mem = self._load(hp)
        img = (raw[z].astype(np.float32) / 255.0)[None]   # (1, H, W)
        mask = mem[z].astype(np.int64)
        return Sample(sample_id=sid, image=img, mask=mask, patient_id=spec, slice_index=z)
