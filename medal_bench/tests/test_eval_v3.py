"""frozen_v3 per-case metric tests: per-case macro-fg DSC (primary) + native-2D
case fallback, pooled diagnostic + aliases, total-miss diagonal penalty, symmetric
ASSD, structure detection rate, valid-region masking, and eval_scope logging.

A FakeModel returns its input as logits, so a one-hot-per-pixel input image gives
fully controllable predictions independent of the GT mask.
"""
from __future__ import annotations

import numpy as np
import torch
from medpy.metric.binary import assd as _assd

from medal_bench.data.base import MedALDataset, Sample
from medal_bench.runner.eval import eval_segmentation

C = 2
SZ = 8


class _Identity(torch.nn.Module):
    def forward(self, x):           # x: (B, C, H, W) one-hot logits -> argmax = encoded class
        return x


def _img_for(pred_mask: np.ndarray) -> np.ndarray:
    """(C, SZ, SZ) one-hot so argmax over channels == pred_mask."""
    img = np.zeros((C, SZ, SZ), dtype=np.float32)
    for c in range(C):
        img[c][pred_mask == c] = 1.0
    return img


class _MemDS(MedALDataset):
    name = "mem"; modality = "x"; target = "y"; dim = "2d"; query_unit = "image"; num_classes = C

    def __init__(self, samples): self._s = samples
    def __len__(self): return len(self._s)
    def sample_ids(self): return [s.sample_id for s in self._s]
    def __getitem__(self, i): return self._s[i]


def _sample(sid, pred_mask, gt_mask, patient_id=None, slice_index=None, meta=None):
    return Sample(sample_id=sid, image=_img_for(pred_mask), mask=gt_mask.astype(np.int64),
                  patient_id=patient_id, slice_index=slice_index, meta=meta or {})


def _eval(ds, **kw):
    return eval_segmentation(_Identity(), ds, num_classes=C, image_size=SZ, device="cpu", **kw)


def test_per_case_macro_groups_by_patient_and_aliases():
    fg = np.ones((SZ, SZ), dtype=np.int64)          # GT all class-1
    empty = np.zeros((SZ, SZ), dtype=np.int64)
    # case A (2 slices): perfect prediction (DSC 1). case B (2 slices): predicts all bg (DSC 0).
    ds = _MemDS([
        _sample("A_0", fg, fg, patient_id="A", slice_index=0),
        _sample("A_1", fg, fg, patient_id="A", slice_index=1),
        _sample("B_0", empty, fg, patient_id="B", slice_index=0),
        _sample("B_1", empty, fg, patient_id="B", slice_index=1),
    ])
    out = _eval(ds)
    assert out["primary_metric"] == "mean_dsc_fg_case_macro"
    assert out["metric_version"] == "v3_case_macro"
    assert out["n_cases"] == 2
    assert abs(out["mean_dsc_fg_case_macro"] - 0.5) < 1e-6      # mean(1.0, 0.0)
    assert out["mean_dsc_fg"] == out["mean_dsc_fg_case_macro"]  # alias
    assert out["eval_scope"] == "case_retained_slices"          # slice_index present
    # pooled (micro) diagnostic present and != per-case here (B's slices dilute the pool)
    assert "mean_dsc_fg_pooled_diagnostic" in out and len(out["dsc_per_class"]) == C
    # detection: A detected class1, B missed -> 0.5
    assert abs(out["structure_detection_rate"] - 0.5) < 1e-6
    assert abs(out["missed_structure_rate"] - 0.5) < 1e-6


def test_native_2d_each_image_is_a_case():
    fg = np.ones((SZ, SZ), dtype=np.int64)
    ds = _MemDS([_sample("i0", fg, fg), _sample("i1", fg, fg)])   # patient_id=None
    out = _eval(ds)
    assert out["n_cases"] == 2                  # each image its own case
    assert out["eval_scope"] == "case_full_volume"
    assert abs(out["mean_dsc_fg_case_macro"] - 1.0) < 1e-6


def test_total_miss_diagonal_penalty_and_detection():
    fg = np.ones((SZ, SZ), dtype=np.int64)
    empty = np.zeros((SZ, SZ), dtype=np.int64)
    ds = _MemDS([_sample("c0", empty, fg, patient_id="c", slice_index=0)])  # GT present, pred empty
    out = _eval(ds, compute_surface=True)
    diag = float(np.sqrt(1 * 1 + SZ * SZ + SZ * SZ))            # (D=1, H=W=SZ)
    assert abs(out["hd95_case_macro_fg"] - diag) < 1e-6
    assert abs(out["assd_case_macro_fg"] - diag) < 1e-6
    assert out["hd95_undefined"] == 1
    assert out["structure_detection_rate"] == 0.0


def test_assd_is_symmetric_not_directed():
    # asymmetric: pred = top-left 2x2 block, gt = full row 0 of class1 -> asd(p,g) != asd(g,p)
    pred = np.zeros((SZ, SZ), dtype=np.int64); pred[:2, :2] = 1
    gt = np.zeros((SZ, SZ), dtype=np.int64);   gt[0, :] = 1
    ds = _MemDS([_sample("c0", pred, gt, patient_id="c", slice_index=0)])
    out = _eval(ds, compute_surface=True)
    # eval assembles a (1, SZ, SZ) volume; compare to medpy symmetric assd on the same.
    expect = float(_assd((pred == 1)[None], (gt == 1)[None]))
    assert abs(out["assd_case_macro_fg"] - expect) < 1e-6
    assert "mean_asd_fg_directed" in out       # directed kept as diagnostic


def test_valid_region_excludes_pad_from_dsc():
    # GT class1 only inside the valid top-left 4x4; pred is class1 EVERYWHERE. Without
    # valid masking the pred FPs in the pad region tank DSC; with masking DSC == 1.
    gt = np.zeros((SZ, SZ), dtype=np.int64); gt[:4, :4] = 1
    pred = np.ones((SZ, SZ), dtype=np.int64)
    ds = _MemDS([_sample("c0", pred, gt, patient_id="c", slice_index=0,
                         meta={"valid_bbox": (0, 0, 4, 4)})])
    out = _eval(ds)
    assert abs(out["mean_dsc_fg_case_macro"] - 1.0) < 1e-6      # pad FPs ignored
