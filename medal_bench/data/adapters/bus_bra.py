"""BUS-BRA (breast ultrasound lesion segmentation) adapter.

Source: BUS-BRA dataset, Zenodo record 8231412 (Gomez-Flores et al., Medical
Physics 2024). 1875 B-mode breast ultrasound images with binary lesion masks.

Layout (under root_dir):
  raw/BUSBRA/Images/bus_XXXX-{l,r}.png   - grayscale US image
  raw/BUSBRA/Masks/mask_XXXX-{l,r}.png   - binary lesion mask (PIL mode "1")
  raw/BUSBRA/bus_data.csv                - per-image ID -> Case (patient) + metadata

Each image's ``Case`` id (from the csv) is the patient: a case's left/right views
share one patient_id so train/val/test stay leakage-disjoint at the patient level.
Fills the breast-US-lesion object alongside busi.

Sample ID:  "bus_XXXX-side"; Image: grayscale -> (1, H, W) float32 in [0, 1];
Mask:       binary (>0 -> 1) -> (H, W) int64 in {0, 1}; bg=0, lesion=1.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from medal_bench.data.base import MedALDataset, Sample


class BUSBRAAdapter(MedALDataset):
    name = "bus_bra"
    modality = "ultrasound"
    target = "breast_lesion"
    dim = "2d"
    query_unit = "image"
    num_classes = 2

    def __init__(self, root_dir: str):
        root = Path(root_dir) / "raw" / "BUSBRA"
        img_dir, msk_dir = root / "Images", root / "Masks"
        if not img_dir.exists():
            raise FileNotFoundError(f"BUS-BRA Images dir not found: {img_dir}")
        # ID -> Case (patient) from the metadata csv (fallback: id itself)
        case_by_id: dict[str, str] = {}
        csv_path = root / "bus_data.csv"
        if csv_path.exists():
            with open(csv_path) as fh:
                for row in csv.DictReader(fh):
                    case_by_id[row["ID"]] = row.get("Case") or row["ID"]
        # (sample_id, image_path, mask_path, patient_id)
        self._index: list[tuple[str, Path, Path, str]] = []
        for ip in sorted(img_dir.glob("bus_*.png")):
            sid = ip.stem                       # bus_XXXX-side
            mp = msk_dir / f"mask_{sid[len('bus_'):]}.png"   # mask_XXXX-side.png
            if not mp.exists():
                raise FileNotFoundError(f"BUS-BRA: missing mask {mp}")
            self._index.append((sid, ip, mp, f"case_{case_by_id.get(sid, sid)}"))
        if not self._index:
            raise FileNotFoundError(f"No bus_*.png under {img_dir}")
        self._ids = [s for (s, _, _, _) in self._index]

    def __len__(self) -> int:
        return len(self._ids)

    def sample_ids(self) -> list[str]:
        return list(self._ids)

    def patient_ids(self) -> list[str]:
        return [pid for (_, _, _, pid) in self._index]

    def __getitem__(self, i: int) -> Sample:
        sid, ip, mp, pid = self._index[i]
        img = np.asarray(Image.open(ip).convert("L")).astype(np.float32)[None] / 255.0
        m = np.asarray(Image.open(mp).convert("L"))
        mask = (m > 0).astype(np.int64)
        return Sample(sample_id=sid, image=img, mask=mask, patient_id=pid,
                      meta={"image_path": str(ip)})
