"""v3 PEAL + PAAL diagnostics (from existing v1 JSONLs which logged per-round scalars).

Per-image PEAL disagreement and AP calibration curves require additional compute
(load checkpoint, forward over pool with flip / train AP, log per-sample) — they
are listed as next-step diagnostics in the v3 report and not generated here.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
from pathlib import Path

import numpy as np

REPO = Path("/groups/echambe2/gmeng/MedAL-Agent/repo/code")
V1_DIR = REPO / "runs" / "pilot_v1"
TABLES = REPO / "tables"


def load_v1(ds, pid, sd):
    path = V1_DIR / f"{ds}__{pid}__{sd}.jsonl"
    return [json.loads(l) for l in open(path)] if path.exists() else []


def main():
    datasets = ["busi", "cvc_clinicdb", "isic2018", "promise12"]
    seeds = ["s1000", "s2000", "s3000"]
    TABLES.mkdir(parents=True, exist_ok=True)

    # PEAL: per-round mean disagreement (scalar over the unlabeled pool, from v1 logs).
    # NOTE: v1 stage-1 (seed 1000) was logged BEFORE the PAAL→PEAL rename and uses
    # the legacy key `paal_mean_disagreement`; v1 stage-2 (seeds 2000+3000) uses
    # the new key `peal_mean_disagreement`. We accept either.
    peal_rows = []
    for ds in datasets:
        for sd in seeds:
            recs = load_v1(ds, "P6", sd)
            for r in recs:
                diag = r.get("selection_diagnostics", {})
                d = diag.get("peal_mean_disagreement", diag.get("paal_mean_disagreement"))
                peal_rows.append({
                    "dataset": ds, "seed": int(sd[1:]),
                    "round": r["round"],
                    "labeled_count": r["labeled_count"],
                    "selected_this_round": len(r.get("selected_ids", [])),
                    "peal_mean_disagreement": round(d, 5) if d is not None else "—",
                })
    with open(TABLES / "v3_peal_round_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(peal_rows[0].keys()))
        w.writeheader()
        for row in peal_rows: w.writerow(row)
    print(f"wrote tables/v3_peal_round_summary.csv ({len(peal_rows)} rows)")

    # PAAL: AP val_corr + pred_acc + ap_loss per round (from v1 logs)
    paal_rows = []
    for ds in datasets:
        for sd in seeds:
            recs = load_v1(ds, "P9", sd)
            for r in recs:
                d = r.get("selection_diagnostics", {})
                paal_rows.append({
                    "dataset": ds, "seed": int(sd[1:]),
                    "round": r["round"],
                    "labeled_count": r["labeled_count"],
                    "selected_this_round": len(r.get("selected_ids", [])),
                    "ap_loss_mean": (round(d.get("paal_ap_loss_mean"), 5)
                                     if d.get("paal_ap_loss_mean") is not None else "—"),
                    "ap_val_corr": (round(d.get("paal_ap_val_corr"), 4)
                                    if d.get("paal_ap_val_corr") is not None
                                    and not (isinstance(d.get("paal_ap_val_corr"), float)
                                             and math.isnan(d["paal_ap_val_corr"]))
                                    else "—"),
                    "pred_acc_mean": (round(d.get("paal_pred_acc_mean"), 4)
                                      if d.get("paal_pred_acc_mean") is not None else "—"),
                    "score_mean": (round(d.get("paal_score_mean"), 4)
                                   if d.get("paal_score_mean") is not None else "—"),
                    "n_clusters": d.get("paal_n_clusters", "—"),
                })
    with open(TABLES / "v3_paal_ap_round_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(paal_rows[0].keys()))
        w.writeheader()
        for row in paal_rows: w.writerow(row)
    print(f"wrote tables/v3_paal_ap_round_summary.csv ({len(paal_rows)} rows)")

    # Aggregate per-dataset summaries for the report
    peal_per_ds = {}
    paal_per_ds = {}
    for ds in datasets:
        peal_vals = [r["peal_mean_disagreement"] for r in peal_rows
                     if r["dataset"] == ds and r["peal_mean_disagreement"] != "—"]
        paal_corrs = [r["ap_val_corr"] for r in paal_rows
                      if r["dataset"] == ds and r["ap_val_corr"] != "—"]
        paal_accs = [r["pred_acc_mean"] for r in paal_rows
                     if r["dataset"] == ds and r["pred_acc_mean"] != "—"]
        peal_per_ds[ds] = peal_vals
        paal_per_ds[ds] = {"corrs": paal_corrs, "pred_accs": paal_accs}

    print("\nPEAL mean disagreement (mean ± std across rounds × seeds):")
    for ds in datasets:
        v = peal_per_ds[ds]
        if v: print(f"  {ds:14s}: {np.mean(v):.4f} ± {np.std(v):.4f}  (n={len(v)})")

    print("\nPAAL AP val correlation (where measurable):")
    for ds in datasets:
        c = paal_per_ds[ds]["corrs"]
        a = paal_per_ds[ds]["pred_accs"]
        c_str = f"{np.mean(c):.3f} ± {np.std(c):.3f} (n={len(c)})" if c else "—"
        a_str = f"{np.mean(a):.3f} ± {np.std(a):.3f}" if a else "—"
        print(f"  {ds:14s}: corr = {c_str}  |  pred_acc = {a_str}")

    return {"peal_per_ds": peal_per_ds, "paal_per_ds": paal_per_ds}


if __name__ == "__main__":
    main()
