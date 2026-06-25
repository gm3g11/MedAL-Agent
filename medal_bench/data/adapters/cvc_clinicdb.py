"""CVC-ClinicDB (polyp segmentation in colonoscopy) adapter.

Layout (under root_dir):
  extracted/PNG/Original/{1..612}.png
  extracted/PNG/Ground Truth/{1..612}.png
  extracted/metadata.csv                 - frame_id -> sequence_id map

CVC-ClinicDB ships 612 frames extracted from 29 colonoscopy sequences. To
prevent train/val/test leakage across frames of the same procedure, we
expose the sequence id via ``patient_ids()`` so ``make_split`` produces
sequence-disjoint splits.

Image:  RGB PNG  -> (3, H, W) float32 in [0, 1]
Mask:   RGB PNG  -> (H, W)    int64   in {0, 1}; bg=0, polyp=1
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from medal_bench.data.base import MedALDataset, Sample


class CVCClinicDBAdapter(MedALDataset):
    name = "cvc_clinicdb"
    modality = "endoscopy"
    target = "polyp"
    dim = "2d"
    query_unit = "image"
    num_classes = 2

    def __init__(self, root_dir: str):
        root = Path(root_dir) / "extracted"
        self._img_dir = root / "PNG" / "Original"
        self._gt_dir = root / "PNG" / "Ground Truth"
        meta_csv = root / "metadata.csv"
        if not self._img_dir.exists() or not self._gt_dir.exists():
            raise FileNotFoundError(f"CVC-ClinicDB missing: {self._img_dir} or {self._gt_dir}")
        if not meta_csv.exists():
            raise FileNotFoundError(
                f"CVC-ClinicDB metadata.csv missing: {meta_csv}. Sequence-level "
                "splits cannot be enforced without it; refusing to load."
            )
        # frame_id (string) -> sequence_id (string)
        self._seq_map: dict[str, str] = {}
        with open(meta_csv) as fh:
            for row in csv.DictReader(fh):
                self._seq_map[row["frame_id"]] = row["sequence_id"]
        # canonical iteration order: by integer frame_id
        self._ids = sorted(
            (p.stem for p in self._img_dir.glob("*.png")),
            key=lambda s: int(s) if s.isdigit() else s,
        )
        missing = [sid for sid in self._ids if sid not in self._seq_map]
        if missing:
            raise FileNotFoundError(
                f"CVC-ClinicDB: {len(missing)} frames missing from metadata.csv "
                f"(e.g. {missing[:3]})"
            )

    def __len__(self) -> int:
        return len(self._ids)

    def sample_ids(self) -> list[str]:
        return list(self._ids)

    def patient_ids(self) -> list[str]:
        # sequence id acts as the grouping unit for splits
        return [f"seq{self._seq_map[sid]}" for sid in self._ids]

    def __getitem__(self, i: int) -> Sample:
        sid = self._ids[i]
        img = np.asarray(Image.open(self._img_dir / f"{sid}.png").convert("RGB"))
        img = img.astype(np.float32).transpose(2, 0, 1) / 255.0
        m = np.asarray(Image.open(self._gt_dir / f"{sid}.png").convert("L"))
        mask = (m > 0).astype(np.int64)
        return Sample(
            sample_id=sid, image=img, mask=mask,
            patient_id=f"seq{self._seq_map[sid]}",
            meta={"sequence_id": self._seq_map[sid]},
        )
