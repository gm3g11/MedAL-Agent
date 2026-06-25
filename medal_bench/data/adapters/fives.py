"""FIVES (Fundus Image Dataset for AI-based Vessel Segmentation) adapter.

Source: Jin et al., Scientific Data 2022 (figshare 10.6084/m9.figshare.19688169, CC BY 4.0).
800 RGB fundus images (2048x2048) with pixel-wise binary vessel masks, 200 each across 4 disease
classes A(AMD)/D(diabetic-retinopathy)/G(glaucoma)/N(normal). Replaces the AL-degenerate DRIVE(20)
+ CHASE_DB1(28); 800 imgs -> the budget grid yields >=3 acquisition rounds.

Layout (under root_dir), after unrar of FIVES.rar into raw/:
  raw/<FIVES ...>/train/Original/*.png       - 600 RGB fundus images
  raw/<FIVES ...>/train/Ground truth/*.png   - 600 binary vessel masks
  raw/<FIVES ...>/test/Original/*.png        - 200 RGB images
  raw/<FIVES ...>/test/Ground truth/*.png    - 200 masks
The GT subfolder spelling varies across mirrors ("Ground truth"/"Groud truth", casing); matched by
the substring "truth". The benchmark re-splits (make_split), so the FIVES train/test split is merged
into one pool here; each fundus is an independent subject -> sample-level split (patient_ids=None).

Sample ID:  "{split}__{stem}" e.g. "train__1_N", "test__3_G" (split-prefixed to avoid collisions)
Image:      RGB -> (3, H, W) float32 in [0, 1]
Mask:       binary -> (H, W) int64 in {0, 1}; bg=0, vessel=1
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from medal_bench.data.base import MedALDataset, Sample


class FIVESAdapter(MedALDataset):
    name = "fives"
    modality = "fundus"
    target = "retinal_vessels"
    dim = "2d"
    query_unit = "image"
    num_classes = 2

    def __init__(self, root_dir: str):
        root = Path(root_dir) / "raw"
        orig_dirs = sorted(p for p in root.rglob("*") if p.is_dir() and p.name.lower() == "original")
        if not orig_dirs:
            raise FileNotFoundError(f"FIVES: no 'Original' image dir under {root}")
        self._index: list[tuple[str, Path, Path]] = []
        for od in orig_dirs:
            split = od.parent.name.lower()                     # "train" / "test"
            gts = [p for p in od.parent.iterdir() if p.is_dir() and "truth" in p.name.lower()]
            if not gts:
                raise FileNotFoundError(f"FIVES: no ground-truth dir beside {od}")
            gtd = gts[0]
            for ip in sorted(od.glob("*.png")):
                mp = gtd / ip.name
                if mp.exists():
                    self._index.append((f"{split}__{ip.stem}", ip, mp))
        if not self._index:
            raise FileNotFoundError(f"FIVES: no image/mask png pairs under {root}")
        self._ids = [sid for (sid, _, _) in self._index]

    def __len__(self) -> int:
        return len(self._index)

    def sample_ids(self) -> list[str]:
        return list(self._ids)

    def __getitem__(self, i: int) -> Sample:
        sid, ip, mp = self._index[i]
        img = np.asarray(Image.open(ip).convert("RGB")).astype(np.float32).transpose(2, 0, 1) / 255.0
        m = np.asarray(Image.open(mp).convert("L"))
        mask = (m > 127).astype(np.int64)
        return Sample(sample_id=sid, image=img, mask=mask, meta={"image_path": str(ip)})
