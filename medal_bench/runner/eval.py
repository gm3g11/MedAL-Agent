"""Evaluation metrics — frozen_v3 per-case (per-volume) layer.

PRIMARY metric: per-case macro foreground DSC (``mean_dsc_fg_case_macro``).
Val slices are grouped by case (``patient_id``; native-2D images fall back to
``sample_id`` = each image is its own case), per-case DSC is computed and macro-
averaged over cases and over the foreground classes PRESENT in that case's GT.

Secondary (surface, opt-in via ``compute_surface``): per-case ``hd95_case_macro_fg``
and symmetric ``assd_case_macro_fg`` (medpy ``assd``, NOT directed ``asd``), plus
``structure_detection_rate`` / ``missed_structure_rate``. Total-miss (a class present
in a case's GT but predicted empty for the whole case) is charged the case volume's
DIAGONAL (worst-case) and counted in detection — never silently dropped.

Diagnostics (kept for audit / back-compat, never the headline): the old micro/pooled
DSC (``mean_dsc_fg_pooled_diagnostic`` / ``dsc_per_class``) and the directed ASD
(``mean_asd_fg_directed``).

Padding: each slice is restricted to its valid (un-padded) rectangle
(``meta['valid_bbox']``) before any metric, so letterbox pad never contaminates DSC
or surface distances. Distances are in PIXELS/VOXELS at ``image_size`` (native voxel
spacing is not threaded yet — frozen_v3 deferred); saved preds carry a spacing slot
so mm-metrics can be backfilled later without retraining.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from medpy.metric.binary import hd95 as _hd95, asd as _asd, assd as _assd

from medal_bench.data.base import MedALDataset
from medal_bench.runner.trainer import collate_to_batch


def _valid_bbox_of(sample, size: int) -> tuple[int, int, int, int]:
    vb = sample.meta.get("valid_bbox") if isinstance(sample.meta, dict) else None
    if vb is None:
        return 0, 0, size, size
    y0, x0, h, w = (int(v) for v in vb)
    return y0, x0, h, w


def _restrict_to_valid(arr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Set everything outside the valid rectangle to background (0). Returns ``arr``
    unchanged (no copy) when the bbox is the whole canvas (native-2D / square resize)."""
    y0, x0, h, w = bbox
    if y0 == 0 and x0 == 0 and h == arr.shape[0] and w == arr.shape[1]:
        return arr
    out = np.zeros_like(arr)
    out[y0:y0 + h, x0:x0 + w] = arr[y0:y0 + h, x0:x0 + w]
    return out


@torch.no_grad()
def eval_segmentation(
    model: torch.nn.Module,
    ds: MedALDataset,
    *,
    num_classes: int,
    image_size: int,
    device: str,
    batch_size: int = 16,
    compute_surface: bool = False,
    save_preds: bool = False,
    save_probs: bool = False,
) -> dict:
    """Per-case macro-fg DSC (primary) + pooled DSC (diagnostic), plus per-case
    HD95 / symmetric ASSD + detection rate when ``compute_surface``. When
    ``save_preds``, the per-sample val masks (+ optional fp16 probs when
    ``save_probs``) are returned under ``result['_preds']`` for the caller to dump.
    """
    was_training = model.training
    model.eval()

    # per-case accumulators (insertion-ordered for stable n_cases)
    cases: "OrderedDict[str, dict]" = OrderedDict()
    # pooled (micro) DSC — diagnostic only
    pooled_inter = np.zeros(num_classes, dtype=np.float64)
    pooled_denom = np.zeros(num_classes, dtype=np.float64)

    # prediction-saving collectors
    s_sids: list = []; s_pids: list = []; s_slis: list = []
    s_bbox: list = []; s_pred: list = []; s_gt: list = []; s_probs: list = []

    has_slice_index = False
    n_eval = 0

    for start in range(0, len(ds), batch_size):
        chunk = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
        imgs, gts = collate_to_batch(chunk, size=image_size)
        imgs = imgs.to(device); gts = gts.to(device)
        probs_t = F.softmax(model(imgs), dim=1)            # (B, C, H, W)
        pred_np = torch.argmax(probs_t, dim=1).cpu().numpy().astype(np.int64)
        gt_np = gts.cpu().numpy().astype(np.int64)
        probs_np = probs_t.cpu().numpy() if (save_preds and save_probs) else None

        for b, sample in enumerate(chunk):
            bbox = _valid_bbox_of(sample, image_size)
            pv = _restrict_to_valid(pred_np[b], bbox)
            gv = _restrict_to_valid(gt_np[b], bbox)

            if sample.slice_index is not None:
                has_slice_index = True
            ck = sample.patient_id if sample.patient_id is not None else sample.sample_id
            cur = cases.get(ck)
            if cur is None:
                cur = {"inter": np.zeros(num_classes), "denom": np.zeros(num_classes),
                       "gt_present": set(), "pred_present": set(), "slices": []}
                cases[ck] = cur

            for c in range(num_classes):
                pc = (pv == c); gc = (gv == c)
                inter_c = float(np.logical_and(pc, gc).sum())
                denom_c = float(pc.sum() + gc.sum())
                pooled_inter[c] += inter_c
                pooled_denom[c] += denom_c
                cur["inter"][c] += inter_c
                cur["denom"][c] += denom_c
                if gc.any():
                    cur["gt_present"].add(c)
                if pc.any():
                    cur["pred_present"].add(c)
            if compute_surface:
                sl_idx = sample.slice_index if sample.slice_index is not None else 0
                cur["slices"].append((sl_idx, pv.astype(np.uint8), gv.astype(np.uint8)))
            if save_preds:
                s_sids.append(sample.sample_id)
                s_pids.append(sample.patient_id or "")
                s_slis.append(-1 if sample.slice_index is None else int(sample.slice_index))
                s_bbox.append(bbox)
                s_pred.append(pv.astype(np.uint8))
                s_gt.append(gv.astype(np.uint8))
                if probs_np is not None:
                    s_probs.append(probs_np[b].astype(np.float16))
        n_eval += len(chunk)

    if was_training:
        model.train()

    # ---- PRIMARY: per-case macro fg DSC (over GT-present fg classes per case) ----
    case_dscs: list = []
    for cur in cases.values():
        present = sorted(c for c in cur["gt_present"] if c >= 1)
        if not present:
            continue   # case with no fg in GT -> excluded from the DSC macro
        per_class = []
        for c in present:
            d = cur["denom"][c]
            per_class.append(2.0 * cur["inter"][c] / d if d > 0 else 0.0)  # pred-empty -> 0
        case_dscs.append(float(np.mean(per_class)))
    mean_dsc_case_macro = float(np.mean(case_dscs)) if case_dscs else float("nan")

    # ---- DIAGNOSTIC: pooled (micro) DSC ----
    pooled = np.where(pooled_denom > 0, 2.0 * pooled_inter / np.maximum(pooled_denom, 1e-8),
                      float("nan"))
    pooled_fg = pooled[1:]
    mean_dsc_pooled = float(np.nanmean(pooled_fg)) if len(pooled_fg) else float("nan")

    # ---- structure detection (cheap; always emitted) ----
    det_total = det_hit = 0
    for cur in cases.values():
        for c in cur["gt_present"]:
            if c < 1:
                continue
            det_total += 1
            if c in cur["pred_present"]:
                det_hit += 1

    result = {
        "primary_metric": "mean_dsc_fg_case_macro",
        "metric_version": "v3_case_macro",
        "eval_scope": "case_retained_slices" if has_slice_index else "case_full_volume",
        "mean_dsc_fg_case_macro": mean_dsc_case_macro,
        "mean_dsc_fg": mean_dsc_case_macro,                  # alias = v3 primary (downstream compat)
        "mean_dsc_fg_pooled_diagnostic": mean_dsc_pooled,
        "dsc_per_class_pooled": [float(x) for x in pooled],
        "dsc_per_class": [float(x) for x in pooled],         # back-compat alias (pooled per-class)
        "n_cases": len(case_dscs),
        "n_eval": n_eval,
        "structure_detection_rate": (det_hit / det_total) if det_total else float("nan"),
        "missed_structure_rate": (1.0 - det_hit / det_total) if det_total else float("nan"),
    }

    if compute_surface:
        hd_vals: list = []; assd_vals: list = []; asd_dir_vals: list = []
        undefined = 0
        for cur in cases.values():
            if not cur["slices"]:
                continue
            sl = sorted(cur["slices"], key=lambda t: t[0])
            pred_vol = np.stack([s[1] for s in sl], axis=0)   # (D, H, W)
            gt_vol = np.stack([s[2] for s in sl], axis=0)
            D, H, W = pred_vol.shape
            diag = float(np.sqrt(D * D + H * H + W * W))
            for c in sorted(cur["gt_present"]):
                if c < 1:
                    continue
                p = (pred_vol == c); g = (gt_vol == c)
                if not g.any():
                    continue                                  # not a GT-present structure
                if not p.any():
                    # total miss -> diagonal worst-case (never dropped)
                    hd_vals.append(diag); assd_vals.append(diag); asd_dir_vals.append(diag)
                    undefined += 1
                    continue
                hd_vals.append(float(_hd95(p, g)))
                assd_vals.append(float(_assd(p, g)))          # symmetric (primary)
                asd_dir_vals.append(float(_asd(p, g)))        # directed (diagnostic)
        result["hd95_case_macro_fg"] = float(np.mean(hd_vals)) if hd_vals else float("nan")
        result["assd_case_macro_fg"] = float(np.mean(assd_vals)) if assd_vals else float("nan")
        result["mean_asd_fg_directed"] = float(np.mean(asd_dir_vals)) if asd_dir_vals else float("nan")
        result["hd95_undefined"] = int(undefined)
        result["surface_units"] = "pixels"
        # back-compat aliases (old per-slice keys; now per-case)
        result["mean_hd95_fg"] = result["hd95_case_macro_fg"]
        result["mean_asd_fg"] = result["assd_case_macro_fg"]

    if save_preds:
        result["_preds"] = {
            "sample_ids": s_sids,
            "patient_ids": s_pids,
            "slice_indices": s_slis,
            "valid_bbox": np.asarray(s_bbox, dtype=np.int32) if s_bbox
            else np.zeros((0, 4), dtype=np.int32),
            "pred": np.stack(s_pred) if s_pred
            else np.zeros((0, image_size, image_size), np.uint8),
            "gt": np.stack(s_gt) if s_gt
            else np.zeros((0, image_size, image_size), np.uint8),
            "probs": (np.stack(s_probs) if s_probs else None),
        }
    return result


# back-compat alias for the old smoke-only API
def eval_dsc(model, ds, *, num_classes, image_size, device, batch_size: int = 16) -> dict:
    return eval_segmentation(
        model, ds, num_classes=num_classes, image_size=image_size,
        device=device, batch_size=batch_size, compute_surface=False,
    )
