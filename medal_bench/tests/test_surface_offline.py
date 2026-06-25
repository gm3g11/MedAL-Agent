"""Accuracy guard for the offline surface pass: the surface metrics computed offline
from saved masks (``surface_offline.compute_surface_from_preds``) must EXACTLY equal the
inline ones from ``eval_segmentation(compute_surface=True)``. If they ever diverge, the
offline-metric optimization is silently changing results -> this test fails.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from medal_bench.data.base import MedALDataset, Sample
from medal_bench.runner.eval import eval_segmentation
from medal_bench.runner.surface_offline import compute_surface_from_preds

C = 3
SZ = 24


class _Identity(torch.nn.Module):
    def forward(self, x):
        return x


def _img_for(pred_mask):
    img = np.zeros((C, SZ, SZ), dtype=np.float32)
    for c in range(C):
        img[c][pred_mask == c] = 1.0
    return img


class _MemDS(MedALDataset):
    name = "mem"; modality = "x"; target = "y"; dim = "3d"; query_unit = "slice"; num_classes = C
    def __init__(self, s): self._s = s
    def __len__(self): return len(self._s)
    def sample_ids(self): return [s.sample_id for s in self._s]
    def __getitem__(self, i): return self._s[i]


def _sample(sid, pred, gt, patient_id=None, slice_index=None):
    return Sample(sample_id=sid, image=_img_for(pred), mask=gt.astype(np.int64),
                  patient_id=patient_id, slice_index=slice_index, meta={})


def _box(mask, cls, y0, y1, x0, x1):
    mask[y0:y1, x0:x1] = cls


def _make_ds():
    z = lambda: np.zeros((SZ, SZ), dtype=np.int64)
    # Case A (2 slices, multi-class, shifted pred -> non-trivial HD95/ASSD on class 1 & 2)
    gA0 = z(); _box(gA0, 1, 4, 16, 4, 16); _box(gA0, 2, 7, 12, 7, 12)
    pA0 = z(); _box(pA0, 1, 5, 17, 5, 17); _box(pA0, 2, 8, 13, 8, 12)
    gA1 = z(); _box(gA1, 1, 5, 18, 6, 18); _box(gA1, 2, 9, 13, 9, 13)
    pA1 = z(); _box(pA1, 1, 4, 17, 5, 16); _box(pA1, 2, 8, 12, 9, 14)
    # Case B (1 slice): class 1 present in GT, pred all background -> TOTAL MISS (diagonal)
    gB = z(); _box(gB, 1, 6, 18, 6, 18)
    pB = z()
    # Case D (native-2D, patient_id=None -> case key = sample_id): partial overlap class 1
    gD = z(); _box(gD, 1, 3, 20, 3, 12)
    pD = z(); _box(pD, 1, 5, 19, 4, 13)
    return _MemDS([
        _sample("A0", pA0, gA0, patient_id="A", slice_index=0),
        _sample("A1", pA1, gA1, patient_id="A", slice_index=1),
        _sample("B0", pB, gB, patient_id="B", slice_index=0),
        _sample("D0", pD, gD, patient_id=None, slice_index=None),
    ])


def test_offline_surface_equals_inline():
    ds = _make_ds()
    out = eval_segmentation(_Identity(), ds, num_classes=C, image_size=SZ, device="cpu",
                            compute_surface=True, save_preds=True)
    offline = compute_surface_from_preds(out["_preds"], num_classes=C)

    keys = ["hd95_case_macro_fg", "assd_case_macro_fg", "mean_asd_fg_directed",
            "hd95_undefined", "surface_units", "mean_hd95_fg", "mean_asd_fg"]
    for k in keys:
        iv, ov = out[k], offline[k]
        if isinstance(iv, float) and math.isnan(iv):
            assert math.isnan(ov), f"{k}: inline nan, offline {ov}"
        else:
            assert iv == ov, f"{k}: inline {iv} != offline {ov}"

    # sanity: the total-miss case B contributed a diagonal -> undefined >= 1, finite means
    assert out["hd95_undefined"] >= 1
    assert not math.isnan(out["hd95_case_macro_fg"])
    assert out["mean_hd95_fg"] == out["hd95_case_macro_fg"]
