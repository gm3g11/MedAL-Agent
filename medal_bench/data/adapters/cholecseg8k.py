"""CholecSeg8k (laparoscopic surgical-scene segmentation) adapter.

Layout (under root_dir, CholecSeg8k ships as per-video clip folders of
consecutive endoscopic frames):

  <video_dir>/<clip_dir>/frame_<k>_endo.png            - RGB frame
  <video_dir>/<clip_dir>/frame_<k>_endo_mask.png       - SPARSE mask (~74% gray-0) (NOT used)
  <video_dir>/<clip_dir>/frame_<k>_endo_color_mask.png - RGB color mask        (ignored)
  <video_dir>/<clip_dir>/frame_<k>_endo_watershed_mask.png - DENSE watershed GT (USED)

We index every ``*_endo_watershed_mask.png`` (the canonical DENSE ground truth) and pair
it with its sibling ``*_endo.png`` frame. The ``_endo_mask.png`` sibling is under-annotated
(~74% of every frame collapses to gray-0), so it is NOT used. The watershed grayscale mask
stores non-dense native codes {5,11,12,13,21,22,23,24,25,31,32,33,50}; CHOLECSEG8K_REMAP maps
them to dense {0..12} with Black-Background(50) -> bg. The gray-255 boundary line and any rare
stray code (35, 36) fall to background 0 via the LUT default.

VIDEO-disjoint grouping: each video folder is ~80 consecutive frames of one
procedure, so they MUST stay in one split. ``patient_id = "video<NN>"`` (the
top-level video folder under root) is exposed so ``make_split`` produces
video-disjoint train/val/test sets and frames of the same clip never leak.

Sample ID: "<clip>_frame_<k>" e.g. "video01_00080_frame_100" (globally unique)
Image:     RGB PNG -> (3, H, W) float32 in [0, 1]
Mask:      grayscale PNG remapped -> (H, W) int64 in {0..12}
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

from medal_bench.data.base import MedALDataset, Sample

# watershed native grayscale code -> dense class index (13 classes, incl. bg=0).
# The DENSE watershed mask has NO gray-0; Black-Background (gray-50) is the natural bg -> 0.
# The 12 tissue/tool classes (incl. Liver-Ligament=5, which the old endo_mask remap dropped) ->
# 1..12. The boundary line (gray-255) and any rare stray code (35, 36) fall to bg via the LUT default.
CHOLECSEG8K_REMAP = {
    50: 0,                                            # Black-Background -> bg
    5: 1, 11: 2, 12: 3, 13: 4, 21: 5, 22: 6,          # 12 tissue/tool foreground classes
    23: 7, 24: 8, 25: 9, 31: 10, 32: 11, 33: 12,
}

# Use the DENSE watershed GT, NOT _endo_mask.png (which collapses ~74% of every frame to gray-0).
_MASK_SUFFIX = "_endo_watershed_mask.png"
# top-level video folder name, e.g. "video01", "video12", "video55"
_VIDEO_RE = re.compile(r"video\d+", re.IGNORECASE)


def _build_lut() -> np.ndarray:
    """8-bit LUT: known native codes -> dense class; every stray code -> bg 0."""
    lut = np.zeros(256, dtype=np.int64)
    for native, dense in CHOLECSEG8K_REMAP.items():
        lut[native] = dense
    return lut


class CholecSeg8kAdapter(MedALDataset):
    name = "cholecseg8k"
    modality = "laparoscopy"
    target = "surgical_scene"
    dim = "2d"
    query_unit = "image"
    num_classes = 13

    def __init__(self, root_dir: str):
        root = Path(root_dir)
        if not root.exists():
            raise FileNotFoundError(f"CholecSeg8k dir not found: {root}")
        self._root = root
        self._lut = _build_lut()

        # discover dense-grayscale masks anywhere under root, paired with frames.
        index: list[tuple[str, str, Path, Path]] = []  # (sample_id, video_id, frame, mask)
        for mp in root.rglob(f"*{_MASK_SUFFIX}"):
            if mp.name.startswith("._"):
                continue
            frame = mp.with_name(mp.name[: -len(_MASK_SUFFIX)] + "_endo.png")
            if not frame.exists():
                continue
            video_id = self._video_id(mp)
            if video_id is None:
                continue
            stem = mp.name[: -len(_MASK_SUFFIX)]  # e.g. "frame_80"
            # clip folder (e.g. "video01_00080") makes the id unique even when
            # frame numbers repeat across clips of the same video.
            sample_id = f"{mp.parent.name}_{stem}"
            index.append((sample_id, video_id, frame, mp))

        if not index:
            raise FileNotFoundError(
                f"CholecSeg8k: no *{_MASK_SUFFIX} paired with *_endo.png under {root}"
            )
        # stable, deterministic order by sample_id
        index.sort(key=lambda r: r[0])
        self._index = index
        self._ids = [r[0] for r in index]

    def _video_id(self, mask_path: Path) -> str | None:
        """Top-level 'videoNN' folder of this mask (the leakage-grouping unit)."""
        rel = mask_path.relative_to(self._root)
        for part in rel.parts:
            m = _VIDEO_RE.fullmatch(part)
            if m:
                return part.lower()
        # fallback: any path component containing 'videoNN'
        for part in rel.parts:
            m = _VIDEO_RE.search(part)
            if m:
                return m.group(0).lower()
        return None

    def __len__(self) -> int:
        return len(self._ids)

    def sample_ids(self) -> list[str]:
        return list(self._ids)

    def patient_ids(self) -> list[str]:
        return [video_id for (_sid, video_id, _f, _m) in self._index]

    def __getitem__(self, i: int) -> Sample:
        sid, video_id, frame_path, mask_path = self._index[i]
        img = np.asarray(Image.open(frame_path).convert("RGB"))
        img = img.astype(np.float32).transpose(2, 0, 1) / 255.0  # (3, H, W)
        m = np.asarray(Image.open(mask_path).convert("L"))       # 8-bit grayscale
        mask = self._lut[m].astype(np.int64)                     # dense {0..12}
        return Sample(
            sample_id=sid, image=img, mask=mask,
            patient_id=video_id,
            meta={"video": video_id, "mask_path": str(mask_path)},
        )
