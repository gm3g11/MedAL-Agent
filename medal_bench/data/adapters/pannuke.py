"""PanNuke (pan-cancer histology nucleus segmentation) adapter.

Layout (under root_dir):
  extracted/Fold {1,2,3}/images/fold{N}/images.npy   - (P, 256, 256, 3) float64, RGB H&E, [0,255]
  extracted/Fold {1,2,3}/images/fold{N}/types.npy    - (P,) tissue-type strings
  extracted/Fold {1,2,3}/masks/fold{N}/masks.npy     - (P, 256, 256, 6) float64

PanNuke ships 7901 256x256 patches across 3 folds (2656 + 2523 + 2722). The
.npy arrays are float64 and several GB each, so they are opened with
``mmap_mode='r'`` and a single patch is materialized per ``__getitem__``.

Masks are 6 per-class INSTANCE-id channels:
  ch0 Neoplastic, ch1 Inflammatory, ch2 Connective, ch3 Dead, ch4 Epithelial,
  ch5 Background. (The fold README mislabels ch5 as index 6; in the data ch5 is
  the background channel, verified: ch5==0 iff some type channel is non-zero.)
Each type channel holds nucleus instance ids (e.g. 0..160), not a binary. We
collapse to a dense SEMANTIC label in {0..5}:
    cls = where(any(ch[0:5] > 0, axis=-1), argmax(ch[0:5], axis=-1) + 1, 0)
so 0=background, 1=Neoplastic, 2=Inflammatory, 3=Connective, 4=Dead,
5=Epithelial. (Type-channel overlap is essentially absent -- ~1px per fold --
and argmax resolves it deterministically.)

Grouping / leakage: PanNuke provides no patient ids. The three folds are the
dataset's own designated split units, so we expose the FOLD as the grouping key
via ``patient_ids()`` -> the runner produces fold-disjoint train/val/test
splits. (Only 3 groups, but it is the strongest leakage-disjoint unit
available; per-patch grouping would risk no leakage control at all.)

Sample ID:  "f{fold}_{patch_index:05d}"  e.g. "f1_00042"
Image:      RGB -> (3, H, W) float32 in [0, 1]
Mask:       (H, W) int64 in {0..5}; dense semantic labels
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from medal_bench.data.base import MedALDataset, Sample


_FOLDS = (1, 2, 3)


class PanNukeAdapter(MedALDataset):
    name = "pannuke"
    modality = "histology"
    target = "nucleus"
    dim = "2d"
    query_unit = "image"
    num_classes = 6

    def __init__(self, root_dir: str):
        root = Path(root_dir) / "extracted"
        if not root.exists():
            raise FileNotFoundError(f"PanNuke extracted dir not found: {root}")
        # mmap each fold's image/mask arrays (multi-GB float64); lazy per-patch.
        self._imgs: dict[int, np.ndarray] = {}
        self._masks: dict[int, np.ndarray] = {}
        self._index: list[tuple[int, int]] = []  # (fold, patch_index)
        for f in _FOLDS:
            img_p = root / f"Fold {f}" / "images" / f"fold{f}" / "images.npy"
            msk_p = root / f"Fold {f}" / "masks" / f"fold{f}" / "masks.npy"
            if not img_p.exists() or not msk_p.exists():
                raise FileNotFoundError(f"PanNuke fold {f} missing: {img_p} or {msk_p}")
            img = np.load(img_p, mmap_mode="r")
            msk = np.load(msk_p, mmap_mode="r")
            if img.shape[0] != msk.shape[0]:
                raise ValueError(
                    f"PanNuke fold {f}: images ({img.shape[0]}) and masks "
                    f"({msk.shape[0]}) patch counts differ"
                )
            self._imgs[f] = img
            self._masks[f] = msk
            for i in range(img.shape[0]):
                self._index.append((f, i))
        self._ids = [f"f{f}_{i:05d}" for (f, i) in self._index]

    def __len__(self) -> int:
        return len(self._ids)

    def sample_ids(self) -> list[str]:
        return list(self._ids)

    def patient_ids(self) -> list[str]:
        # fold is the leakage-disjoint grouping unit
        return [f"fold{f}" for (f, _i) in self._index]

    def __getitem__(self, i: int) -> Sample:
        f, p = self._index[i]
        img = np.asarray(self._imgs[f][p], dtype=np.float32) / 255.0  # (H, W, 3)
        img = img.transpose(2, 0, 1)  # (3, H, W)
        m6 = np.asarray(self._masks[f][p])  # (H, W, 6) float64
        types = m6[..., 0:5]
        fg = (types > 0).any(axis=-1)
        mask = np.where(fg, types.argmax(axis=-1) + 1, 0).astype(np.int64)
        return Sample(
            sample_id=self._ids[i], image=img, mask=mask,
            patient_id=f"fold{f}",
            meta={"fold": f},
        )
