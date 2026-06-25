"""v3 fair test-set evaluation — analysis + report builder.

Consumes:
  runs/test_eval_v3/*.jsonl  (120 cells, val + test metrics per round + ckpt path)

Produces:
  tables/v3_per_cell_metrics.csv          per (ds, policy, seed, round) — val+test metrics
  tables/v3_summary_final_round.csv        per (ds, policy, seed) — final-round summary
  tables/v3_val_vs_test_ranks.csv          per (ds, seed) ranks val vs test + Spearman ρ
  tables/v3_pairwise_stats.csv             paired vs Random/CoreSet/BADGE/SAM-TC
  tables/v3_failure_summary.csv            collapse / empty-pred / undef rates per cell
  reports/medal_agent_v3_fair_test_eval.md the answer to the 9 questions
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path("/groups/echambe2/gmeng/MedAL-Agent/repo/code")
V3_DIR = REPO / "runs" / "test_eval_v3"
REPORTS = REPO / "reports"
TABLES = REPO / "tables"

POLICIES = ["P0","P1","P2","P3","P4","P5","P6","P7","P8","P9"]
NAMES = {
    "P0":"Random","P1":"NormEnt","P2":"BALD","P3":"CoreSet","P4":"BADGE",
    "P5":"Ent→CS","P6":"SelUnc","P7":"SAM-CS","P8":"SAM-TC","P9":"PAAL",
}
SEEDS = [1000, 2000, 3000]
SEED_STRS = ["s1000","s2000","s3000"]
DATASETS = ["busi","cvc_clinicdb","isic2018","promise12"]
BUDGET_PCT = [1, 2, 5, 10, 15, 20]


def fmt_ms(vals, nd=3, missing="—"):
    if not vals: return missing
    if len(vals) == 1: return f"{vals[0]:.{nd}f}"
    return f"{np.mean(vals):.{nd}f} ± {np.std(vals):.{nd}f}"


def load_cells():
    """{(ds, pid, seed): [round records]}"""
    cells = {}
    for f in sorted(glob.glob(str(V3_DIR / "*.jsonl"))):
        name = os.path.basename(f).replace(".jsonl","")
        if name.startswith("_"): continue
        try:
            ds, pid, seed = name.split("__")
        except ValueError: continue
        if seed not in SEED_STRS: continue
        recs = [json.loads(l) for l in open(f)]
        cells[(ds, pid, seed)] = recs
    return cells


def bootstrap_ci(diffs, n_boot=10000, ci=0.95, rng=None):
    if rng is None: rng = np.random.default_rng(42)
    diffs = np.asarray(diffs)
    if len(diffs) == 0: return (float("nan"), float("nan"))
    boots = np.array([np.mean(rng.choice(diffs, len(diffs), replace=True)) for _ in range(n_boot)])
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 - (1 - ci) / 2) * 100)
    return float(lo), float(hi)


def paired_rank_biserial(diffs):
    """Paired rank-biserial correlation from signed-rank statistic.
    For pairs, r_rb = W_pos / (W_pos + W_neg) * 2 - 1, where W_pos/W_neg = sum of ranks
    of |diffs| for diffs > 0 and < 0 respectively (zeros split equally)."""
    diffs = np.asarray(diffs)
    diffs = diffs[diffs != 0]
    if len(diffs) == 0: return float("nan")
    ranks = stats.rankdata(np.abs(diffs))
    w_pos = ranks[diffs > 0].sum()
    w_neg = ranks[diffs < 0].sum()
    total = w_pos + w_neg
    if total == 0: return float("nan")
    return float((w_pos - w_neg) / total)


def cliffs_delta(a, b):
    """Distributional (unpaired) effect size. Ignores pairing."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0: return float("nan")
    n = len(a) * len(b)
    diff = a[:, None] - b[None, :]
    return float(((diff > 0).sum() - (diff < 0).sum()) / n)


def main():
    cells = load_cells()
    print(f"Loaded {len(cells)} cells")
    if len(cells) != 120:
        print(f"  WARN: expected 120 cells")
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # ----- 1. Per-cell-per-round metrics CSV -----
    per_round_rows = []
    for (ds, p, sd), recs in sorted(cells.items()):
        for r in recs:
            mv = r["metrics_val"]; mt = r["metrics_test"]
            per_round_rows.append({
                "dataset": ds, "policy": p, "seed": int(sd[1:]),
                "round": r["round"], "labeled_count": r["labeled_count"],
                "val_dsc": round(mv["mean_dsc_fg"], 5),
                "val_iou": round(mv["mean_iou_fg"], 5),
                "val_hd95_filt": round(mv["mean_hd95_filtered_fg"], 3) if not math.isnan(mv["mean_hd95_filtered_fg"]) else "nan",
                "val_hd95_pen":  round(mv["mean_hd95_penalty_fg"], 3) if not math.isnan(mv["mean_hd95_penalty_fg"]) else "nan",
                "val_asd_filt":  round(mv["mean_asd_filtered_fg"], 3) if not math.isnan(mv["mean_asd_filtered_fg"]) else "nan",
                "val_empty_pred_rate": round(mv["empty_pred_rate_fg"], 5),
                "val_hd95_undef_rate": round(mv["hd95_undef_rate_fg"], 5),
                "val_collapse": mv["collapse_flag"],
                "test_dsc": round(mt["mean_dsc_fg"], 5),
                "test_iou": round(mt["mean_iou_fg"], 5),
                "test_hd95_filt": round(mt["mean_hd95_filtered_fg"], 3) if not math.isnan(mt["mean_hd95_filtered_fg"]) else "nan",
                "test_hd95_pen":  round(mt["mean_hd95_penalty_fg"], 3) if not math.isnan(mt["mean_hd95_penalty_fg"]) else "nan",
                "test_asd_filt":  round(mt["mean_asd_filtered_fg"], 3) if not math.isnan(mt["mean_asd_filtered_fg"]) else "nan",
                "test_empty_pred_rate": round(mt["empty_pred_rate_fg"], 5),
                "test_hd95_undef_rate": round(mt["hd95_undef_rate_fg"], 5),
                "test_collapse": mt["collapse_flag"],
                "ckpt_path": r.get("ckpt_path",""),
            })
    with open(TABLES / "v3_per_cell_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_round_rows[0].keys()))
        w.writeheader()
        for row in per_round_rows: w.writerow(row)
    print(f"  wrote tables/v3_per_cell_metrics.csv ({len(per_round_rows)} rows)")

    # ----- 2. Per-cell final-round summary -----
    summary_rows = []
    for (ds, p, sd), recs in sorted(cells.items()):
        last = recs[-1]
        mv = last["metrics_val"]; mt = last["metrics_test"]
        bud = [r["labeled_count"] for r in recs]
        dscs_v = [r["metrics_val"]["mean_dsc_fg"] for r in recs]
        dscs_t = [r["metrics_test"]["mean_dsc_fg"] for r in recs]
        aulc_v = float(np.trapz(dscs_v, bud) / (bud[-1] - bud[0]))
        aulc_t = float(np.trapz(dscs_t, bud) / (bud[-1] - bud[0]))
        summary_rows.append({
            "dataset": ds, "policy": p, "seed": int(sd[1:]),
            "labeled_final": last["labeled_count"],
            "val_dsc_final": round(mv["mean_dsc_fg"], 5),
            "val_hd95_filt": round(mv["mean_hd95_filtered_fg"], 3) if not math.isnan(mv["mean_hd95_filtered_fg"]) else "nan",
            "val_aulc_dsc":  round(aulc_v, 5),
            "test_dsc_final": round(mt["mean_dsc_fg"], 5),
            "test_hd95_filt": round(mt["mean_hd95_filtered_fg"], 3) if not math.isnan(mt["mean_hd95_filtered_fg"]) else "nan",
            "test_aulc_dsc":  round(aulc_t, 5),
            "val_collapse": mv["collapse_flag"], "test_collapse": mt["collapse_flag"],
        })
    with open(TABLES / "v3_summary_final_round.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        for row in summary_rows: w.writerow(row)
    print(f"  wrote tables/v3_summary_final_round.csv ({len(summary_rows)} rows)")

    # ----- 3. Val vs Test rank-transfer -----
    transfer_rows = []
    for ds in DATASETS:
        for sd_int, sd in zip(SEEDS, SEED_STRS):
            # per-(ds, seed) rank policies by val DSC and test DSC
            vals = []
            tests = []
            pids_present = []
            for p in POLICIES:
                if (ds, p, sd) not in cells: continue
                v = cells[(ds, p, sd)][-1]["metrics_val"]["mean_dsc_fg"]
                t = cells[(ds, p, sd)][-1]["metrics_test"]["mean_dsc_fg"]
                vals.append(v); tests.append(t); pids_present.append(p)
            if len(vals) < 3: continue
            # ranks (1 = best = highest DSC)
            order_v = np.argsort(-np.asarray(vals))
            order_t = np.argsort(-np.asarray(tests))
            rank_v = np.empty(len(vals)); rank_t = np.empty(len(tests))
            for r, i in enumerate(order_v, 1): rank_v[i] = r
            for r, i in enumerate(order_t, 1): rank_t[i] = r
            # Spearman correlation between val and test ranks
            try:
                spear_rho, spear_p = stats.spearmanr(rank_v, rank_t)
            except Exception:
                spear_rho, spear_p = float("nan"), float("nan")
            for i, p in enumerate(pids_present):
                transfer_rows.append({
                    "dataset": ds, "seed": sd_int, "policy": p,
                    "val_dsc": round(vals[i], 5), "test_dsc": round(tests[i], 5),
                    "val_rank": int(rank_v[i]), "test_rank": int(rank_t[i]),
                    "rank_change": int(rank_v[i]) - int(rank_t[i]),
                    "spearman_rho_val_vs_test": round(float(spear_rho), 4),
                    "spearman_p": round(float(spear_p), 4),
                })
    with open(TABLES / "v3_val_vs_test_ranks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(transfer_rows[0].keys()))
        w.writeheader()
        for row in transfer_rows: w.writerow(row)
    print(f"  wrote tables/v3_val_vs_test_ranks.csv ({len(transfer_rows)} rows)")

    # Per-dataset average Spearman rho across seeds (overall val→test stability)
    spear_per_ds = defaultdict(list)
    for r in transfer_rows:
        spear_per_ds[r["dataset"]].append(r["spearman_rho_val_vs_test"])
    spear_per_ds_mean = {ds: float(np.mean(list(set(vs)))) for ds, vs in spear_per_ds.items()}

    # ----- 4. Pairwise stats (test split) vs baselines -----
    baselines = ["P0", "P3", "P4", "P8"]  # Random, CoreSet, BADGE, SAM-TC
    rng = np.random.default_rng(42)
    pairwise_rows = []
    for tgt in POLICIES:
        for base in baselines:
            if tgt == base: continue
            for split_label, mkey in [("val","metrics_val"), ("test","metrics_test")]:
                diffs = []
                t_vals = []; b_vals = []
                for ds in DATASETS:
                    for sd in SEED_STRS:
                        if (ds, tgt, sd) not in cells or (ds, base, sd) not in cells: continue
                        tv = cells[(ds, tgt, sd)][-1][mkey]["mean_dsc_fg"]
                        bv = cells[(ds, base, sd)][-1][mkey]["mean_dsc_fg"]
                        diffs.append(tv - bv)
                        t_vals.append(tv); b_vals.append(bv)
                if not diffs: continue
                diffs_arr = np.asarray(diffs)
                mean_d = float(diffs_arr.mean())
                ci_lo, ci_hi = bootstrap_ci(diffs_arr, n_boot=10000, ci=0.95, rng=rng)
                try:
                    w_stat, w_p = stats.wilcoxon(diffs_arr, zero_method="zsplit", alternative="two-sided")
                except Exception:
                    w_stat, w_p = float("nan"), float("nan")
                n_pos = int((diffs_arr > 0).sum())
                n_neg = int((diffs_arr < 0).sum())
                n_total = n_pos + n_neg
                try:
                    sign_p = float(stats.binomtest(n_pos, n=n_total, p=0.5).pvalue) if n_total > 0 else float("nan")
                except Exception:
                    sign_p = float("nan")
                rb = paired_rank_biserial(diffs_arr)
                cd = cliffs_delta(t_vals, b_vals)
                pairwise_rows.append({
                    "split": split_label,
                    "target_policy": tgt, "target_name": NAMES[tgt],
                    "baseline_policy": base, "baseline_name": NAMES[base],
                    "n_paired_cells": len(diffs),
                    "mean_diff_dsc": round(mean_d, 4),
                    "ci95_lo": round(ci_lo, 4), "ci95_hi": round(ci_hi, 4),
                    "wilcoxon_p": round(w_p, 4) if not math.isnan(w_p) else "nan",
                    "sign_test_p": round(sign_p, 4) if not math.isnan(sign_p) else "nan",
                    "paired_rank_biserial": round(rb, 3) if not math.isnan(rb) else "nan",
                    "cliffs_delta_distributional": round(cd, 3),
                    "n_pos": n_pos, "n_neg": n_neg,
                })
    with open(TABLES / "v3_pairwise_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pairwise_rows[0].keys()))
        w.writeheader()
        for row in pairwise_rows: w.writerow(row)
    print(f"  wrote tables/v3_pairwise_stats.csv ({len(pairwise_rows)} rows)")

    # ----- 5. Failure summary -----
    failure_rows = []
    for (ds, p, sd), recs in sorted(cells.items()):
        last = recs[-1]
        mv = last["metrics_val"]; mt = last["metrics_test"]
        failure_rows.append({
            "dataset": ds, "policy": p, "seed": int(sd[1:]),
            "val_collapse": mv["collapse_flag"],
            "val_empty_pred_rate": round(mv["empty_pred_rate_fg"], 4),
            "val_hd95_undef_rate": round(mv["hd95_undef_rate_fg"], 4),
            "test_collapse": mt["collapse_flag"],
            "test_empty_pred_rate": round(mt["empty_pred_rate_fg"], 4),
            "test_hd95_undef_rate": round(mt["hd95_undef_rate_fg"], 4),
        })
    with open(TABLES / "v3_failure_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(failure_rows[0].keys()))
        w.writeheader()
        for row in failure_rows: w.writerow(row)
    print(f"  wrote tables/v3_failure_summary.csv ({len(failure_rows)} rows)")

    # ----- 6. Final aggregates for report -----
    def best_per_ds_test():
        out = {}
        for ds in DATASETS:
            cands = []
            for p in POLICIES:
                ts = [cells[(ds, p, sd)][-1]["metrics_test"]["mean_dsc_fg"] for sd in SEED_STRS if (ds, p, sd) in cells]
                if ts: cands.append((p, float(np.mean(ts))))
            out[ds] = max(cands, key=lambda x: x[1]) if cands else None
        return out
    def best_per_ds_val():
        out = {}
        for ds in DATASETS:
            cands = []
            for p in POLICIES:
                vs = [cells[(ds, p, sd)][-1]["metrics_val"]["mean_dsc_fg"] for sd in SEED_STRS if (ds, p, sd) in cells]
                if vs: cands.append((p, float(np.mean(vs))))
            out[ds] = max(cands, key=lambda x: x[1]) if cands else None
        return out

    winners_val = best_per_ds_val()
    winners_test = best_per_ds_test()
    return {
        "cells": cells, "summary_rows": summary_rows,
        "pairwise_rows": pairwise_rows, "transfer_rows": transfer_rows,
        "failure_rows": failure_rows, "per_round_rows": per_round_rows,
        "spear_per_ds_mean": spear_per_ds_mean,
        "winners_val": winners_val, "winners_test": winners_test,
    }


if __name__ == "__main__":
    out = main()
    print("\nWinners (mean DSC across 3 seeds):")
    for ds in DATASETS:
        wv = out["winners_val"].get(ds)
        wt = out["winners_test"].get(ds)
        match = "✓ same" if wv and wt and wv[0] == wt[0] else "✗ different"
        print(f"  {ds:15s}: VAL = {wv[0]} ({wv[1]:.3f}), TEST = {wt[0]} ({wt[1]:.3f})  {match}")
    print("\nVal→Test Spearman rho (per dataset, mean across seeds):")
    for ds, rho in out["spear_per_ds_mean"].items():
        print(f"  {ds:15s}: rho = {rho:+.3f}")
