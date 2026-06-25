"""Offline surface metrics (HD95 / symmetric ASSD / directed ASD) from saved val
prediction masks — identical numbers to the inline path in ``eval.py``, computed on
CPU after training so the GPU loop can skip them.

WHY this is accuracy-neutral: ``eval_segmentation(save_preds=True)`` saves exactly the
per-sample valid-restricted masks (``pv``/``gv``) that its own surface loop consumes
(``cur["slices"]``). So grouping those saved masks by case and running the SAME loop
here reproduces ``hd95_case_macro_fg`` / ``assd_case_macro_fg`` / ... byte-for-byte.
The surface math below is copied verbatim from ``eval.py`` (lines under
``if compute_surface:``); ``tests/test_surface_offline.py`` asserts inline == offline.

Usage:
    # patch the FINAL-round record of every cell in a run dir with offline surface metrics
    python -m medal_bench.runner.surface_offline --run-dir runs/stage2_wave2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import OrderedDict

import numpy as np
from medpy.metric.binary import hd95 as _hd95, asd as _asd, assd as _assd


def compute_surface_from_preds(preds: dict, num_classes: int) -> dict:
    """Per-case macro surface metrics from a saved-prediction dict.

    ``preds`` keys (as written by ``trajectory.write_predictions``): ``sample_ids``
    (str[N]), ``patient_ids`` (str[N], "" if none), ``slice_indices`` (int[N], -1 if
    none), ``pred`` (uint8 N,H,W), ``gt`` (uint8 N,H,W). Masks are already
    valid-restricted (outside the valid bbox = 0), matching the inline path.

    Returns the same surface keys ``eval_segmentation`` emits under ``compute_surface``.
    """
    sids = list(preds["sample_ids"])
    pids = list(preds["patient_ids"])
    slis = list(preds["slice_indices"])
    pred = np.asarray(preds["pred"])
    gt = np.asarray(preds["gt"])

    # group slices by case, exactly as eval.py: case key = patient_id else sample_id;
    # slice order key = slice_index if present else 0 (eval.py uses 0 for None; saved -1).
    cases: "OrderedDict[str, list]" = OrderedDict()
    for i in range(len(sids)):
        pid = str(pids[i]) if pids[i] is not None else ""
        ck = pid if pid != "" else str(sids[i])
        sl_idx = int(slis[i]) if int(slis[i]) >= 0 else 0
        cases.setdefault(ck, []).append((sl_idx, pred[i].astype(np.uint8), gt[i].astype(np.uint8)))

    # ---- surface loop: VERBATIM from eval.py ``if compute_surface:`` ----
    hd_vals: list = []; assd_vals: list = []; asd_dir_vals: list = []
    undefined = 0
    for sl_list in cases.values():
        if not sl_list:
            continue
        sl = sorted(sl_list, key=lambda t: t[0])
        pred_vol = np.stack([s[1] for s in sl], axis=0)   # (D, H, W)
        gt_vol = np.stack([s[2] for s in sl], axis=0)
        D, H, W = pred_vol.shape
        diag = float(np.sqrt(D * D + H * H + W * W))
        gt_present = sorted(int(c) for c in np.unique(gt_vol).tolist())
        for c in gt_present:
            if c < 1:
                continue
            p = (pred_vol == c); g = (gt_vol == c)
            if not g.any():
                continue
            if not p.any():
                hd_vals.append(diag); assd_vals.append(diag); asd_dir_vals.append(diag)
                undefined += 1
                continue
            hd_vals.append(float(_hd95(p, g)))
            assd_vals.append(float(_assd(p, g)))
            asd_dir_vals.append(float(_asd(p, g)))

    result = {
        "hd95_case_macro_fg": float(np.mean(hd_vals)) if hd_vals else float("nan"),
        "assd_case_macro_fg": float(np.mean(assd_vals)) if assd_vals else float("nan"),
        "mean_asd_fg_directed": float(np.mean(asd_dir_vals)) if asd_dir_vals else float("nan"),
        "hd95_undefined": int(undefined),
        "surface_units": "pixels",
    }
    result["mean_hd95_fg"] = result["hd95_case_macro_fg"]
    result["mean_asd_fg"] = result["assd_case_macro_fg"]
    return result


def _patch_cell(jsonl_path: str, preds_dir: str) -> str:
    """Patch a cell's FINAL-round record metrics with offline surface metrics. Returns
    a status string. Idempotent: skips if the final record already has surface keys."""
    with open(jsonl_path) as fh:
        recs = [json.loads(l) for l in fh if l.strip()]
    if not recs:
        return "empty"
    final = recs[-1]
    m = final.get("metrics", {})
    if "hd95_case_macro_fg" in m and not (isinstance(m["hd95_case_macro_fg"], float)
                                          and np.isnan(m["hd95_case_macro_fg"])):
        return "already-has-surface"
    npz_path = final.get("predictions_path") or ""
    if not npz_path or not os.path.exists(npz_path):
        # fall back to the conventional path by run_id + final round
        rid = final.get("run_id"); rnd = final.get("round")
        cand = os.path.join(preds_dir, f"{rid}__r{rnd}.npz")
        npz_path = cand if os.path.exists(cand) else ""
    if not npz_path:
        return "no-preds-npz"
    z = np.load(npz_path, allow_pickle=True)
    nc = len(m.get("dsc_per_class", [])) or int(np.asarray(z["pred"]).max()) + 1
    surf = compute_surface_from_preds(
        {k: z[k] for k in ("sample_ids", "patient_ids", "slice_indices", "pred", "gt")}, nc)
    m.update(surf)
    final["metrics"] = m
    tmp = jsonl_path + ".tmp"
    with open(tmp, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, jsonl_path)
    return "patched"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="dir with *__s*.jsonl cells + predictions/")
    args = ap.parse_args(argv)
    preds_dir = os.path.join(args.run_dir, "predictions")
    cells = sorted(glob.glob(os.path.join(args.run_dir, "*__s*.jsonl")))
    counts: dict[str, int] = {}
    for c in cells:
        st = _patch_cell(c, preds_dir)
        counts[st] = counts.get(st, 0) + 1
        print(f"  {os.path.basename(c)}: {st}")
    print(f"[surface_offline] {dict(sorted(counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
