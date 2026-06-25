"""PH2 (dermoscopy skin-lesion segmentation) adapter.

Source: PH2 Dataset (ADDI project), 200 dermoscopic images.

Layout (under root_dir):
  extracted/PH2Dataset/PH2 Dataset images/
      IMDxxx/
          IMDxxx_Dermoscopic_Image/IMDxxx.bmp   - RGB dermoscopy image
          IMDxxx_lesion/IMDxxx_lesion.bmp       - binary lesion mask (0/255)
          IMDxxx_roi/...                         - (ignored)

Each IMDxxx folder is one lesion = one patient (1 image per patient -> 200
groups). The folder id is exposed via ``patient_ids()`` so the runner can keep
train/val/test splits leakage-disjoint at the patient level.

Sample ID:  "IMDxxx" e.g. "IMD002"
Image:      RGB -> (3, H, W) float32 in [0, 1]
Mask:       binary (>0 -> 1) -> (H, W) int64 in {0, 1}; bg=0, lesion=1
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from medal_bench.data.base import MedALDataset, Sample


class PH2Adapter(MedALDataset):
    name = "ph2"
    modality = "dermoscopy"
    target = "skin_lesion"
    dim = "2d"
    query_unit = "image"
    num_classes = 2

    def __init__(self, root_dir: str):
        root = Path(root_dir) / "raw" / "extracted" / "PH2Dataset" / "PH2 Dataset images"
        if not root.exists():
            raise FileNotFoundError(f"PH2 dir not found: {root}")
        # (lesion_id, image_path, mask_path)
        self._index: list[tuple[str, Path, Path]] = []
        lesion_dirs = sorted(
            p for p in root.iterdir()
            if p.is_dir() and p.name.startswith("IMD") and not p.name.startswith("._")
        )
        if not lesion_dirs:
            raise FileNotFoundError(f"No IMDxxx dirs under {root}")
        for ldir in lesion_dirs:
            imd = ldir.name
            ip = ldir / f"{imd}_Dermoscopic_Image" / f"{imd}.bmp"
            mp = ldir / f"{imd}_lesion" / f"{imd}_lesion.bmp"
            if not ip.exists():
                raise FileNotFoundError(f"PH2: missing image {ip}")
            if not mp.exists():
                raise FileNotFoundError(f"PH2: missing mask {mp}")
            self._index.append((imd, ip, mp))
        self._ids = [imd for (imd, _, _) in self._index]

    def __len__(self) -> int:
        return len(self._ids)

    def sample_ids(self) -> list[str]:
        return list(self._ids)

    def patient_ids(self) -> list[str]:
        # one lesion folder == one patient (1 image per patient)
        return [imd for (imd, _, _) in self._index]

    def __getitem__(self, i: int) -> Sample:
        imd, ip, mp = self._index[i]
        img = np.asarray(Image.open(ip).convert("RGB")).astype(np.float32).transpose(2, 0, 1) / 255.0
        m = np.asarray(Image.open(mp).convert("L"))
        mask = (m > 0).astype(np.int64)
        return Sample(
            sample_id=imd, image=img, mask=mask,
            patient_id=imd, meta={"image_path": str(ip)},
        )
