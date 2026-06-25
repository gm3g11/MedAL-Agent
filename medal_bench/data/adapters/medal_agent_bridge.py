"""Bridge: expose the audited ``medal_agent`` loader (21 pre-sliced datasets) as a
``medal_bench`` MedALDataset so the P0-P9 AL runner can consume them.

``medal_agent`` (at /groups/echambe2/datasets/medal_agent) is the single source of
truth for the expanded dataset set: pre-sliced 2D PNGs, case-disjoint split
manifests, strict dense remaps, 21 datasets — all audited (Check A/B + loader smoke
+ overlays). This bridge wraps a medal_agent split (default: ``train``) as the AL
universe; the runner's ``make_split`` then re-carves case-disjoint train/val/test by
``patient_id == case_id`` (leakage-free). Formal Stage 1 may instead honor
medal_agent's native splits via a runner hook (TODO, flagged).

This SUPERSEDES the interim ``mmwhs.py`` adapter + ``data/remap.py`` (raw-NIfTI path);
prefer bridged ids.
"""
from __future__ import annotations

import re
import sys
from typing import Optional

import numpy as np

from medal_bench.data.base import MedALDataset, Sample

_MA_PARENT = "/groups/echambe2/datasets"
if _MA_PARENT not in sys.path:
    sys.path.insert(0, _MA_PARENT)

_SLICE_RE = re.compile(r"_s(\d+)\.png$", re.IGNORECASE)


class MedalAgentBridge(MedALDataset):
    dim = "3d"          # all 21 medal_agent datasets are 3D volumes sliced to 2D
    query_unit = "slice"

    def __init__(self, ds_id: str, which: str = "train",
                 modality: Optional[str] = None, mask_subdir: Optional[str] = None):
        import medal_agent as ma
        spec = ma.get(ds_id)
        self._ds = ma.SlicedDataset(spec, which=which, modality=modality, mask_subdir=mask_subdir)
        # Drop slices with no mask file for the chosen subdir: the AL pool must be
        # labeled-able (the runner reveals a slice's mask on selection, and can't
        # reveal a label that doesn't exist). e.g. care_la has 11055 train images
        # but only 9712 atrium masks. SlicedDataset reads masks at masks_dir/<name>.
        md = self._ds.masks_dir
        if md is not None and md.exists():
            labeled = [p for p in self._ds.image_paths if (md / p.name).exists()]
            if labeled:
                self._ds.image_paths = labeled
        self.name = ds_id
        # Log the CONCRETE modality actually loaded (the per-dataset view override,
        # e.g. ext_brats2020 -> "t1ce") rather than the registry's generic tag
        # (e.g. "multi_modal_mri"), so provenance reflects the real input channel.
        self.modality = modality or spec.modality
        self.target = spec.classes[1] if len(spec.classes) > 1 else "foreground"
        self.num_classes = spec.C
        # enumerate stable ids + case ids without loading pixels
        self._ids: list[str] = []
        self._cases: list[str] = []
        self._slices: list[int] = []
        for p in self._ds.image_paths:
            case = ma.parse_case(p.name) or p.stem
            m = _SLICE_RE.search(p.name)
            self._ids.append(p.stem)
            self._cases.append(case)
            self._slices.append(int(m.group(1)) if m else -1)

    def __len__(self) -> int:
        return len(self._ids)

    def sample_ids(self) -> list[str]:
        return list(self._ids)

    def patient_ids(self) -> list[str]:
        return list(self._cases)

    def __getitem__(self, i: int) -> Sample:
        img_t, mask_t, _meta = self._ds[i]
        img = np.ascontiguousarray(img_t.numpy(), dtype=np.float32)   # (C, H, W) in [0,1]
        mask = None if mask_t is None else mask_t.numpy().astype(np.int64)  # (H, W)
        return Sample(
            sample_id=self._ids[i], image=img, mask=mask,
            patient_id=self._cases[i], slice_index=self._slices[i],
        )


# Per-dataset view overrides for multi-modal / multi-mask datasets (else first).
_VIEW_OVERRIDES: dict[str, dict] = {
    "mmwhs": {"modality": "ct"},          # cardiac CT (mr available as a variant)
    "ext_brats2020": {"modality": "t1ce"},  # contrast-enhanced T1
    "care_leftatrium_2026": {"mask_subdir": "atrium"},
    "myops": {"modality": "lge"},  # task-canonical scar/edema sequence (n=1760 vs c0=492; same C=5 masks)
}


def register_medal_agent_datasets(registry: dict) -> list[str]:
    """Add a bridge factory for every medal_agent dataset id. Returns the ids added.
    No-op (returns []) if medal_agent is unimportable, so medal_bench still loads."""
    try:
        import medal_agent as ma
    except Exception:
        return []
    added = []
    for ds_id in ma.list_ids():
        kw = _VIEW_OVERRIDES.get(ds_id, {})
        registry.setdefault(
            ds_id, (lambda dr, _i=ds_id, _k=kw: MedalAgentBridge(_i, **_k))
        )
        added.append(ds_id)
    return added
