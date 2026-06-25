"""Extended evaluation: DSC, IoU, HD95, ASSD (filtered + penalty), empty-pred /
empty-GT rates. Used by the v3 replay runner for both val and test splits."""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F
from medpy.metric.binary import hd95 as _hd95, asd as _asd

from medal_bench.data.base import MedALDataset
from medal_bench.runner.trainer import collate_to_batch

PENALTY_PX = float(np.sqrt(256**2 + 256**2))  # image diagonal at 256x256 ≈ 362.04 px


def _binary_hd95_asd(pred: np.ndarray, gt: np.ndarray):
    p = pred.astype(bool); g = gt.astype(bool)
    if not p.any() and not g.any():
        return 0.0, 0.0
    if not p.any() or not g.any():
        return float("nan"), float("nan")
    return float(_hd95(p, g)), float(_asd(p, g))


@torch.no_grad()
def eval_full(model: torch.nn.Module, ds: MedALDataset, *,
              num_classes: int, image_size: int, device: str,
              batch_size: int = 4) -> dict:
    """Returns dict with DSC/IoU per class, HD95/ASSD per fg class (filtered + penalty),
    empty-pred / empty-GT rates."""
    was_training = model.training
    model.eval()
    n_eval = 0
    # DSC + IoU accumulators per class
    inter = np.zeros(num_classes, dtype=np.float64)
    pred_sum = np.zeros(num_classes, dtype=np.float64)
    gt_sum = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)
    # HD95 / ASSD per fg class — lists of per-sample values (NaN = undefined)
    hd_buckets = [[] for _ in range(num_classes)]
    asd_buckets = [[] for _ in range(num_classes)]
    # empty rates per fg class
    n_empty_pred = [0]*num_classes
    n_empty_gt = [0]*num_classes
    n_both_empty = [0]*num_classes
    n_undef = [0]*num_classes

    for start in range(0, len(ds), batch_size):
        chunk = [ds[i] for i in range(start, min(start+batch_size, len(ds)))]
        imgs, gts = collate_to_batch(chunk, size=image_size)
        imgs = imgs.to(device); gts = gts.to(device)
        logits = model(imgs)
        pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
        # DSC accumulators
        for c in range(num_classes):
            p_c = (pred == c).float(); g_c = (gts == c).float()
            inter[c] += float((p_c * g_c).sum().cpu())
            pred_sum[c] += float(p_c.sum().cpu())
            gt_sum[c] += float(g_c.sum().cpu())
            union[c] += float(((p_c + g_c) >= 1).float().sum().cpu())
        # Per-sample HD95 / ASSD + empty counters (fg classes only)
        pred_np = pred.cpu().numpy(); gt_np = gts.cpu().numpy()
        for b in range(pred_np.shape[0]):
            for c in range(1, num_classes):
                p_empty = not (pred_np[b] == c).any()
                g_empty = not (gt_np[b] == c).any()
                if p_empty and g_empty:
                    n_both_empty[c] += 1
                if p_empty: n_empty_pred[c] += 1
                if g_empty: n_empty_gt[c] += 1
                h, a = _binary_hd95_asd(pred_np[b] == c, gt_np[b] == c)
                if math.isnan(h):
                    n_undef[c] += 1
                hd_buckets[c].append(h)
                asd_buckets[c].append(a)
        n_eval += imgs.shape[0]

    if was_training: model.train()

    # DSC / IoU per class
    dsc = np.where((pred_sum + gt_sum) > 0,
                   2.0*inter / np.maximum(pred_sum + gt_sum, 1e-8), float("nan"))
    iou = np.where(union > 0, inter / np.maximum(union, 1e-8), float("nan"))

    # HD95 / ASSD per fg class — filtered + penalty
    hd95_filt_per_c = [None]
    hd95_pen_per_c = [None]
    asd_filt_per_c = [None]
    asd_pen_per_c = [None]
    for c in range(1, num_classes):
        finite_h = [x for x in hd_buckets[c] if not math.isnan(x)]
        finite_a = [x for x in asd_buckets[c] if not math.isnan(x)]
        # filtered: mean over finite
        hd95_filt_per_c.append(float(np.mean(finite_h)) if finite_h else float("nan"))
        asd_filt_per_c.append(float(np.mean(finite_a)) if finite_a else float("nan"))
        # penalty: replace NaN with PENALTY_PX
        pen_h = [PENALTY_PX if math.isnan(x) else x for x in hd_buckets[c]]
        pen_a = [PENALTY_PX if math.isnan(x) else x for x in asd_buckets[c]]
        hd95_pen_per_c.append(float(np.mean(pen_h)) if pen_h else float("nan"))
        asd_pen_per_c.append(float(np.mean(pen_a)) if pen_a else float("nan"))

    mean_dsc_fg = float(np.nanmean(dsc[1:])) if num_classes > 1 else float("nan")
    mean_iou_fg = float(np.nanmean(iou[1:])) if num_classes > 1 else float("nan")
    fg_filt = [v for v in hd95_filt_per_c[1:] if v is not None]
    fg_pen  = [v for v in hd95_pen_per_c[1:]  if v is not None]
    fg_filt_a = [v for v in asd_filt_per_c[1:] if v is not None]
    fg_pen_a  = [v for v in asd_pen_per_c[1:]  if v is not None]

    return {
        "n_eval": n_eval,
        "dsc_per_class": [float(x) for x in dsc],
        "iou_per_class": [float(x) for x in iou],
        "mean_dsc_fg": mean_dsc_fg,
        "mean_iou_fg": mean_iou_fg,
        "hd95_filtered_per_class": hd95_filt_per_c,
        "hd95_penalty_per_class": hd95_pen_per_c,
        "asd_filtered_per_class": asd_filt_per_c,
        "asd_penalty_per_class": asd_pen_per_c,
        "mean_hd95_filtered_fg": float(np.nanmean(fg_filt)) if fg_filt else float("nan"),
        "mean_hd95_penalty_fg":  float(np.nanmean(fg_pen))  if fg_pen  else float("nan"),
        "mean_asd_filtered_fg":  float(np.nanmean(fg_filt_a)) if fg_filt_a else float("nan"),
        "mean_asd_penalty_fg":   float(np.nanmean(fg_pen_a))  if fg_pen_a  else float("nan"),
        "n_empty_pred_per_class": n_empty_pred,
        "n_empty_gt_per_class":   n_empty_gt,
        "n_both_empty_per_class": n_both_empty,
        "n_hd95_undefined_per_class": n_undef,
        "empty_pred_rate_fg": float(sum(n_empty_pred[1:]) / max(1, n_eval * max(1, num_classes-1))),
        "empty_gt_rate_fg":   float(sum(n_empty_gt[1:])   / max(1, n_eval * max(1, num_classes-1))),
        "hd95_undef_rate_fg": float(sum(n_undef[1:])      / max(1, n_eval * max(1, num_classes-1))),
        "collapse_flag": int(mean_dsc_fg < 0.05),
        "penalty_px": PENALTY_PX,
    }
