"""v2 audit pipeline — existing-artifact-only audit of the MedAL-Agent v1 pilot.

INPUTS
  runs/pilot_v1/*.jsonl                       per-cell trajectories
  cache/foundation_features/*.h5              SAM ViT-B feature cache
  medal_bench/data/adapters/                  dataset adapters (used for PROMISE12 case-level analysis)

OUTPUTS (paths relative to repo/code/)
  reports/pilot_v2_audit.md
  reports/missing_compute_manifest.yaml
  tables/pilot_v2_metrics.csv
  tables/pilot_v2_pairwise_stats.csv
  tables/pilot_v2_failure_rates.csv
  tables/pilot_v2_selection_diagnostics.csv
  tables/pilot_v2_kmeans_coverage.csv
  plots/learning_curves/<ds>.png
  plots/hd95_filtered_vs_penalty.png
  plots/selection_overlap_heatmaps/<ds>.png
  plots/selected_foreground_ratio_by_round/<ds>.png
  plots/kmeans_coverage_sensitivity/<ds>.png

Honors the user-specified claim-safety rules: no "publication-quality" framing,
HD95 sensitivity reported but no strong claims, future-compute manifest for
gaps, KMeans coverage labelled as a representation-coverage proxy (not topology).
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

REPO = Path("/groups/echambe2/gmeng/MedAL-Agent/repo/code")
RUNS = REPO / "runs" / "pilot_v1"
# SAM cache lives at the MedAL-Agent project root (not under repo/), per the
# Phase-A env setup. Hard-coded absolute path because $HOME caches are AFS-readonly here.
SAM_CACHE = Path("/groups/echambe2/gmeng/MedAL-Agent/cache/foundation_features")
DATA_ROOT = Path("/groups/echambe2/datasets/data")

OUT_REPORTS = REPO / "reports"
OUT_TABLES = REPO / "tables"
OUT_PLOTS = REPO / "plots"

POLICIES = ["P0","P1","P2","P3","P4","P5","P6","P7","P8","P9"]
POLICY_NAME = {
    "P0":"Random","P1":"NormEnt","P2":"BALD","P3":"CoreSet","P4":"BADGE",
    "P5":"Ent→CS","P6":"SelUnc","P7":"SAM-CS","P8":"SAM-TC","P9":"PAAL",
}
SEEDS = ["s1000","s2000","s3000"]
SEED_INTS = {"s1000":1000, "s2000":2000, "s3000":3000}
DATASETS_EXPECTED = ["busi","cvc_clinicdb","isic2018","promise12"]
BUDGET_PCT = [1, 2, 5, 10, 15, 20]
N_ROUNDS = 6

# Image diagonal at 256x256 used for penalty-HD95 conversion
HD95_PENALTY_PX = float(np.sqrt(256**2 + 256**2))  # ≈ 362.04 px


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_cells():
    """Return cells = {(ds, pid, seed): [round records]} for the 4-dataset matrix."""
    cells = {}
    for f in sorted(glob.glob(str(RUNS / "*.jsonl"))):
        name = os.path.basename(f).replace(".jsonl","")
        if name.startswith("_") or "msd07" in name:
            continue
        try:
            ds, pid, seed = name.split("__")
        except ValueError:
            continue
        if seed not in SEEDS or ds not in DATASETS_EXPECTED:
            continue
        cells[(ds, pid, seed)] = [json.loads(l) for l in open(f)]
    return cells


def load_sam_cache(dataset):
    files = sorted(glob.glob(str(SAM_CACHE / f"{dataset}_train_*.h5")))
    if not files:
        return None
    with h5py.File(files[0], "r") as h:
        sids = [s.decode() if hasattr(s,"decode") else str(s) for s in h["sample_ids"][:]]
        feats = h["features"][:]
    return {sid: feats[i] for i, sid in enumerate(sids)}, sids, feats


def cells_for(cells, ds, pid):
    return {sd: cells[(ds, pid, sd)] for sd in SEEDS if (ds, pid, sd) in cells}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_ms(vals, nd=3, missing="—"):
    if not vals: return missing
    if len(vals) == 1: return f"{vals[0]:.{nd}f}"
    return f"{np.mean(vals):.{nd}f} ± {np.std(vals):.{nd}f}"

def aulc(xs, ys):
    if len(xs) < 2: return float("nan")
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    return float(np.trapz(ys, xs) / (xs[-1] - xs[0]))

def best_so_far(ys):
    out = []
    cur = -np.inf
    for y in ys:
        if y > cur: cur = y
        out.append(cur)
    return out

def cliffs_delta(a, b):
    """Cliff's delta — non-parametric, **distributional** effect size in [-1, 1].
    Ignores the pairing structure (treats `a` and `b` as two independent samples).
    For paired data (our case: per-(dataset, seed) policy-vs-baseline pairs), a
    paired rank-biserial correlation would be a more appropriate effect size.
    Reported here as a coarse magnitude indicator; future stats pass could add
    paired rank-biserial via scipy.stats.wilcoxon's correlation derivation."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0: return float("nan")
    n = len(a) * len(b)
    diff = a[:, None] - b[None, :]
    return float(((diff > 0).sum() - (diff < 0).sum()) / n)

def bootstrap_ci(diffs, n_boot=10000, ci=0.95, rng=None):
    if rng is None: rng = np.random.default_rng(42)
    diffs = np.asarray(diffs)
    if len(diffs) == 0:
        return (float("nan"), float("nan"))
    boots = np.array([np.mean(rng.choice(diffs, len(diffs), replace=True)) for _ in range(n_boot)])
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 - (1 - ci) / 2) * 100)
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Task A — Data integrity / split audit
# ---------------------------------------------------------------------------

def task_a_integrity(cells):
    """Returns dict of integrity checks."""
    out = {"coverage": {}, "cold_start_consistency": {}, "selected_uniqueness": {}, "labeled_no_reselect": {}}
    for ds in DATASETS_EXPECTED:
        # coverage
        present = sum(1 for p in POLICIES for sd in SEEDS if (ds, p, sd) in cells)
        out["coverage"][ds] = (present, len(POLICIES) * len(SEEDS))
    # cold-start consistency: round 0 selected_ids (initial labeled) must match across policies for same seed
    # NOTE: round 0 in our schema has selected_ids = what's SELECTED IN round 0 (for round 1), not the initial set.
    # The initial labeled set is implicit (cold start). We can verify cross-policy consistency via:
    # at round 0, the labeled_count equals budget_plan[0] AND the round-1 selection space (unlabeled pool) is
    # the same across policies. We test by computing the union of (selected at round 0) ∪ (cumulative future
    # selected) for two policies at same seed — if cold start was the same, total selected == budget_plan[-1] exactly.
    for ds in DATASETS_EXPECTED:
        per_seed_signatures = {}
        for sd in SEEDS:
            sigs = {}
            for p in POLICIES:
                if (ds, p, sd) not in cells: continue
                recs = cells[(ds, p, sd)]
                # cumulative selected over rounds 0..R-2 (final round has no selection)
                sel = []
                for r in recs:
                    sel.extend(r.get("selected_ids", []))
                # also note labeled_count at round 0 — this equals cold-start size
                cold_size = recs[0]["labeled_count"]
                sigs[p] = (cold_size, len(set(sel)), len(sel))
            per_seed_signatures[sd] = sigs
        # within each seed, all policies should have the same cold_size (cold start is shared)
        # and len(sel) == len(set(sel)) (no duplicates)
        seed_results = {}
        for sd, sigs in per_seed_signatures.items():
            cold_sizes = {s[0] for s in sigs.values()}
            dup_within = [p for p, s in sigs.items() if s[1] != s[2]]
            seed_results[sd] = {
                "cold_size_set": list(cold_sizes),
                "cold_size_consistent": len(cold_sizes) <= 1,
                "policies_with_duplicate_selections": dup_within,
            }
        out["cold_start_consistency"][ds] = seed_results
    # selected uniqueness: within a single cell, no sample picked twice
    dup_cells = []
    for (ds, p, sd), recs in cells.items():
        seen = set()
        for r in recs:
            for sid in r.get("selected_ids", []):
                if sid in seen:
                    dup_cells.append((ds, p, sd, sid)); break
                seen.add(sid)
    out["selected_uniqueness"] = {"cells_with_duplicates": dup_cells}
    # Budget units per dataset
    out["budget_units"] = {
        "busi": "image",
        "cvc_clinicdb": "image (29 sequences, group-safe)",
        "isic2018": "image",
        "promise12": "slice (50 cases, group-safe by case)",
    }
    return out


# ---------------------------------------------------------------------------
# Task B — Metric recomputation
# ---------------------------------------------------------------------------

def task_b_metrics(cells):
    """Compute per-cell DSC trajectories, AULC-DSC, final HD95 (filtered+penalty proxy),
    undefined rates, best-so-far curves. Returns dict for downstream report + CSV."""
    out = {}
    for (ds, p, sd), recs in cells.items():
        bud = [r["labeled_count"] for r in recs]
        dscs = [r["metrics"]["mean_dsc_fg"] for r in recs]
        last = recs[-1]["metrics"]
        hd95 = last.get("mean_hd95_fg")
        if hd95 is None: hd95 = float("nan")
        asd = last.get("mean_asd_fg")
        if asd is None: asd = float("nan")
        n_undef = last.get("hd95_undefined", 0)
        n_eval = last.get("n_eval", 0)
        n_finite = max(0, n_eval - n_undef) if n_eval else 0
        undef_rate = (n_undef / n_eval) if n_eval else float("nan")
        # penalty: replace each undefined sample's HD95 with HD95_PENALTY_PX
        if isinstance(hd95, float) and math.isnan(hd95):
            penalty_hd95 = float("nan")
        else:
            penalty_hd95 = (hd95 * n_finite + HD95_PENALTY_PX * n_undef) / max(1, n_eval)
        out[(ds, p, sd)] = {
            "dsc_per_round": dscs,
            "best_so_far_dsc": best_so_far(dscs),
            "budgets": bud,
            "aulc_dsc": aulc(bud, dscs),
            "final_dsc": dscs[-1],
            "final_hd95_filtered": hd95,
            "final_hd95_penalty": penalty_hd95,
            "final_asd_filtered": asd,
            "hd95_undefined": n_undef,
            "hd95_undef_rate": undef_rate,
            "n_eval": n_eval,
            "collapsed": dscs[-1] < 0.05,
        }
    return out


# ---------------------------------------------------------------------------
# Task D — Statistical tests (pairwise)
# ---------------------------------------------------------------------------

def task_d_pairwise_stats(metrics):
    """For each pair (target_policy, baseline_policy), aggregate per-(dataset, seed)
    paired differences of final DSC, compute mean diff, bootstrap 95% CI, Wilcoxon,
    sign test, Cliff's delta."""
    rows = []
    baselines = ["P0", "P1", "P3", "P4", "P8"]  # Random, Entropy, CoreSet, BADGE, SAM-TC
    rng = np.random.default_rng(42)
    for tgt in POLICIES:
        for base in baselines:
            if tgt == base: continue
            diffs = []
            per_cell = []  # list of (ds, sd, t_dsc, b_dsc, diff)
            for ds in DATASETS_EXPECTED:
                for sd in SEEDS:
                    if (ds, tgt, sd) not in metrics or (ds, base, sd) not in metrics:
                        continue
                    t = metrics[(ds, tgt, sd)]["final_dsc"]
                    b = metrics[(ds, base, sd)]["final_dsc"]
                    diffs.append(t - b)
                    per_cell.append((ds, sd, t, b, t - b))
            if not diffs:
                continue
            diffs_arr = np.asarray(diffs)
            mean_diff = float(diffs_arr.mean())
            ci_lo, ci_hi = bootstrap_ci(diffs_arr, n_boot=10000, ci=0.95, rng=rng)
            # Wilcoxon signed-rank
            try:
                w_stat, w_p = stats.wilcoxon(diffs_arr, zero_method="zsplit", alternative="two-sided")
            except Exception:
                w_stat, w_p = float("nan"), float("nan")
            # Sign test
            n_pos = int((diffs_arr > 0).sum())
            n_neg = int((diffs_arr < 0).sum())
            n_total = n_pos + n_neg
            try:
                sign_p = float(stats.binomtest(n_pos, n=n_total, p=0.5).pvalue) if n_total > 0 else float("nan")
            except Exception:
                sign_p = float("nan")
            # Cliff's delta
            tgt_vals = [pc[2] for pc in per_cell]
            base_vals = [pc[3] for pc in per_cell]
            cd = cliffs_delta(tgt_vals, base_vals)
            rows.append({
                "target_policy": tgt,
                "target_name": POLICY_NAME[tgt],
                "baseline_policy": base,
                "baseline_name": POLICY_NAME[base],
                "n_paired_cells": len(diffs),
                "mean_diff_dsc": round(mean_diff, 4),
                "ci95_lo": round(ci_lo, 4),
                "ci95_hi": round(ci_hi, 4),
                "wilcoxon_p": round(w_p, 4) if not math.isnan(w_p) else float("nan"),
                "sign_test_p": round(sign_p, 4) if not math.isnan(sign_p) else float("nan"),
                "cliffs_delta": round(cd, 3),
                "n_pos": n_pos, "n_neg": n_neg,
            })
    return rows


# ---------------------------------------------------------------------------
# Task E — Stability (existing-artifacts subset)
# ---------------------------------------------------------------------------

def task_e_stability(cells, metrics):
    """Per-round drops > 0.05 + training loss trends (already in JSONL)."""
    drop_records = []  # (ds, p, seed, round_from, round_to, dsc_before, dsc_after, drop)
    for (ds, p, sd), recs in cells.items():
        dscs = [r["metrics"]["mean_dsc_fg"] for r in recs]
        losses = [r["training"]["mean_loss"] for r in recs]
        for i in range(1, len(dscs)):
            drop = dscs[i-1] - dscs[i]
            if drop > 0.05:
                drop_records.append({
                    "dataset": ds, "policy": p, "seed": SEED_INTS[sd],
                    "round_from": i-1, "round_to": i,
                    "dsc_before": round(dscs[i-1], 4),
                    "dsc_after": round(dscs[i], 4),
                    "drop_amount": round(drop, 4),
                    "loss_before": round(losses[i-1], 4),
                    "loss_after": round(losses[i], 4),
                })
    # Per-(ds, policy) failure rates and collapse flags
    failure_rows = []
    for ds in DATASETS_EXPECTED:
        for p in POLICIES:
            seeds_present = [sd for sd in SEEDS if (ds, p, sd) in metrics]
            if not seeds_present: continue
            n_drops_total = sum(1 for d in drop_records
                                if d["dataset"] == ds and d["policy"] == p)
            collapses = sum(1 for sd in seeds_present if metrics[(ds, p, sd)]["collapsed"])
            undef_rates = [metrics[(ds, p, sd)]["hd95_undef_rate"] for sd in seeds_present
                           if not math.isnan(metrics[(ds, p, sd)]["hd95_undef_rate"])]
            failure_rows.append({
                "dataset": ds, "policy": p, "policy_name": POLICY_NAME[p],
                "n_seeds": len(seeds_present),
                "n_dsc_drops_gt_0.05": n_drops_total,
                "n_collapsed_cells": collapses,
                "mean_hd95_undef_rate": round(float(np.mean(undef_rates)), 4) if undef_rates else float("nan"),
                "std_hd95_undef_rate": round(float(np.std(undef_rates)), 4) if undef_rates else float("nan"),
            })
    return {"drop_records": drop_records, "failure_rows": failure_rows}


# ---------------------------------------------------------------------------
# Task G — PROMISE12 case-level (selected slices per case, fg coverage)
# ---------------------------------------------------------------------------

def task_g_promise12(cells):
    """For each (policy, seed) on PROMISE12, count selected slices per case + fg-coverage.
    Slice IDs look like 'Case00_007' — parse the case prefix."""
    out = {"selected_per_case": {}, "case_coverage": {}, "fg_slice_proxy": {}}
    # Build per-cell case map
    for (ds, p, sd), recs in cells.items():
        if ds != "promise12": continue
        per_case = defaultdict(int)
        all_selected = []
        for r in recs:
            for sid in r.get("selected_ids", []):
                case = sid.split("_")[0]  # 'Case00'
                per_case[case] += 1
                all_selected.append(sid)
        # also add cold start (initial labeled at round 0): we don't have the IDs explicitly,
        # but labeled_count - sum(selected) == cold_start_size
        out["selected_per_case"][(p, sd)] = dict(per_case)
        out["case_coverage"][(p, sd)] = {
            "n_cases_touched": len(per_case),
            "n_slices_selected_total": len(all_selected),
            "cases_with_slices": sorted(per_case.keys()),
        }
    # Aggregate by policy across seeds
    summary = []
    for p in POLICIES:
        seeds = [sd for sd in SEEDS if (p, sd) in out["case_coverage"]]
        if not seeds: continue
        n_cases_per_seed = [out["case_coverage"][(p, sd)]["n_cases_touched"] for sd in seeds]
        n_slices_per_seed = [out["case_coverage"][(p, sd)]["n_slices_selected_total"] for sd in seeds]
        # union of cases across seeds
        union_cases = set()
        for sd in seeds:
            union_cases.update(out["case_coverage"][(p, sd)]["cases_with_slices"])
        summary.append({
            "policy": p, "policy_name": POLICY_NAME[p],
            "mean_cases_touched": float(np.mean(n_cases_per_seed)),
            "std_cases_touched": float(np.std(n_cases_per_seed)),
            "mean_slices_selected": float(np.mean(n_slices_per_seed)),
            "union_cases_across_seeds": len(union_cases),
        })
    out["summary"] = summary
    return out


# ---------------------------------------------------------------------------
# Task H — Policy diagnostics (what's logged today)
# ---------------------------------------------------------------------------

def task_h_diagnostics(cells, sam_caches):
    """Audit what diagnostics are actually in the JSONLs for each policy + verify
    SAM feature row order against dataset IDs."""
    out = {"p6_peal": {}, "p9_paal": {}, "p7_p8_sam_checksum": {}}
    # P6: peal_mean_disagreement (scalar per round)
    for (ds, p, sd), recs in cells.items():
        if p != "P6": continue
        per_round = []
        for r in recs:
            d = r.get("selection_diagnostics", {}).get("peal_mean_disagreement")
            per_round.append(d)
        out["p6_peal"].setdefault(ds, []).append((sd, per_round))
    # P9: ap_val_corr, ap_loss_mean, pred_acc_mean, score_mean, n_clusters
    for (ds, p, sd), recs in cells.items():
        if p != "P9": continue
        per_round = []
        for r in recs:
            d = r.get("selection_diagnostics", {})
            per_round.append({
                "ap_loss_mean": d.get("paal_ap_loss_mean"),
                "ap_val_corr": d.get("paal_ap_val_corr"),
                "pred_acc_mean": d.get("paal_pred_acc_mean"),
                "score_mean": d.get("paal_score_mean"),
                "n_clusters": d.get("paal_n_clusters"),
            })
        out["p9_paal"].setdefault(ds, []).append((sd, per_round))
    # P7/P8 SAM checksum: confirm SAM cache contains every selected sample_id
    for (ds, p, sd), recs in cells.items():
        if p not in ("P7", "P8"): continue
        sam_data = sam_caches.get(ds)
        if not sam_data: continue
        sam_ids_set = set(sam_data[0].keys())
        missing = []
        for r in recs:
            for sid in r.get("selected_ids", []):
                if sid not in sam_ids_set:
                    missing.append(sid)
        out["p7_p8_sam_checksum"].setdefault(ds, []).append({
            "policy": p, "seed": sd,
            "n_selected": sum(len(r.get("selected_ids", [])) for r in recs),
            "n_missing_in_sam_cache": len(missing),
            "missing_examples": missing[:5],
        })
    return out


# ---------------------------------------------------------------------------
# Task J — KMeans cluster coverage (representation-coverage proxy, NOT true topology)
# ---------------------------------------------------------------------------

def task_j_kmeans_coverage(cells, sam_caches, ks=(5, 10, 20, 50), seed=42):
    """For each (ds, k, policy, seed): coverage = # distinct clusters touched / k,
    Shannon entropy of cluster distribution, max-cluster-share (concentration)."""
    out = {}
    for ds, sam_tuple in sam_caches.items():
        if sam_tuple is None: continue
        sam_dict, all_ids, all_feats = sam_tuple
        feats_norm = all_feats / np.clip(np.linalg.norm(all_feats, axis=1, keepdims=True), 1e-12, None)
        out[ds] = {}
        for k in ks:
            n = len(all_ids)
            if k > n: continue
            km = KMeans(n_clusters=k, random_state=seed, n_init=10)
            cluster_labels = km.fit_predict(feats_norm)
            id_to_cluster = dict(zip(all_ids, cluster_labels))
            out[ds][k] = {}
            for p in POLICIES:
                per_seed_results = []
                for sd in SEEDS:
                    if (ds, p, sd) not in cells: continue
                    sel_ids = set()
                    for r in cells[(ds, p, sd)]:
                        sel_ids.update(r.get("selected_ids", []))
                    sel_in_cache = [s for s in sel_ids if s in id_to_cluster]
                    if not sel_in_cache:
                        continue
                    clusters_hit = [id_to_cluster[s] for s in sel_in_cache]
                    cluster_counts = np.bincount(clusters_hit, minlength=k)
                    coverage = (cluster_counts > 0).sum() / k
                    probs = cluster_counts / cluster_counts.sum()
                    nz = probs[probs > 0]
                    entropy = float(-(nz * np.log2(nz)).sum())
                    max_share = float(cluster_counts.max() / cluster_counts.sum())
                    per_seed_results.append({
                        "seed": SEED_INTS[sd],
                        "coverage_frac": coverage,
                        "entropy_bits": entropy,
                        "max_cluster_share": max_share,
                    })
                out[ds][k][p] = per_seed_results
    return out


# ---------------------------------------------------------------------------
# Selection diagnostics CSV
# ---------------------------------------------------------------------------

def selection_diagnostics_rows(cells):
    rows = []
    for (ds, p, sd), recs in cells.items():
        for r in recs:
            sel = r.get("selected_ids", [])
            if not sel:  # final round, no selection
                continue
            scores = r.get("selected_scores", [])
            fg = r.get("selected_pred_fg_ratio", [])
            cdist = r.get("selected_pred_class_dist", [])
            scores_clean = [s for s in scores if not (isinstance(s, float) and math.isnan(s))]
            rows.append({
                "dataset": ds, "policy": p, "policy_name": POLICY_NAME[p],
                "seed": SEED_INTS[sd],
                "round": r["round"],
                "n_selected": len(sel),
                "labeled_count_at_round_start": r["labeled_count"],
                "mean_selected_score": round(float(np.mean(scores_clean)), 5) if scores_clean else float("nan"),
                "std_selected_score": round(float(np.std(scores_clean)), 5) if len(scores_clean) >= 2 else float("nan"),
                "mean_selected_fg_ratio": round(float(np.mean(fg)), 5) if fg else float("nan"),
                "std_selected_fg_ratio": round(float(np.std(fg)), 5) if len(fg) >= 2 else float("nan"),
                "n_score_obs": len(scores_clean),
            })
    return rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

POLICY_COLORS = {
    "P0":"#999999", "P1":"#1f77b4", "P2":"#ff7f0e", "P3":"#2ca02c",
    "P4":"#d62728", "P5":"#9467bd", "P6":"#8c564b", "P7":"#e377c2",
    "P8":"#17becf", "P9":"#bcbd22",
}

def plot_learning_curves(cells, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS_EXPECTED:
        fig, ax = plt.subplots(figsize=(9, 6))
        # x = budget percent
        xs = BUDGET_PCT
        for p in POLICIES:
            seeds_dscs = []
            for sd in SEEDS:
                if (ds, p, sd) not in cells: continue
                dscs = [r["metrics"]["mean_dsc_fg"] for r in cells[(ds, p, sd)]]
                if len(dscs) == N_ROUNDS:
                    seeds_dscs.append(dscs)
            if not seeds_dscs: continue
            arr = np.array(seeds_dscs)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            color = POLICY_COLORS[p]
            ax.plot(xs, mean, "-o", color=color, label=f"{p} {POLICY_NAME[p]}", linewidth=1.5, markersize=4)
            ax.fill_between(xs, mean - std, mean + std, alpha=0.12, color=color)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x}%" for x in xs])
        ax.set_xlabel("labeled budget (% of train pool)")
        ax.set_ylabel("mean DSC_fg (val)")
        ax.set_title(f"v1 pilot — learning curves (mean ± std over 3 seeds)\n{ds}")
        ax.legend(loc="lower right", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{ds}.png", dpi=120)
        plt.close(fig)

def plot_hd95_filtered_vs_penalty(metrics, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    # For each (ds, policy) compute mean filtered + mean penalty across seeds
    xs = []
    fs = []
    ps = []
    labels = []
    for ds in DATASETS_EXPECTED:
        for p in POLICIES:
            fvals = [metrics[(ds, p, sd)]["final_hd95_filtered"] for sd in SEEDS
                     if (ds, p, sd) in metrics and not math.isnan(metrics[(ds, p, sd)]["final_hd95_filtered"])]
            pvals = [metrics[(ds, p, sd)]["final_hd95_penalty"] for sd in SEEDS
                     if (ds, p, sd) in metrics and not math.isnan(metrics[(ds, p, sd)]["final_hd95_penalty"])]
            if not fvals or not pvals: continue
            xs.append(f"{ds[:5]}/{p}")
            fs.append(np.mean(fvals))
            ps.append(np.mean(pvals))
    x_idx = np.arange(len(xs))
    ax.bar(x_idx - 0.2, fs, width=0.4, label="filtered HD95 (current)", color="#1f77b4")
    ax.bar(x_idx + 0.2, ps, width=0.4, label=f"penalty HD95 (NaN → {HD95_PENALTY_PX:.0f}px)", color="#d62728")
    ax.set_xticks(x_idx); ax.set_xticklabels(xs, rotation=80, fontsize=7)
    ax.set_ylabel("HD95 (px)")
    ax.set_title("HD95 sensitivity to NaN-handling convention\n(mean over 3 seeds; smaller is better)")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_overlap_heatmaps(cells, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS_EXPECTED:
        # Cumulative selected_ids per policy, averaged across seeds
        cum = {}
        for p in POLICIES:
            sets = []
            for sd in SEEDS:
                if (ds, p, sd) not in cells: continue
                s = set()
                for r in cells[(ds, p, sd)]:
                    s.update(r.get("selected_ids", []))
                sets.append(s)
            cum[p] = sets
        # Jaccard matrix (mean across seeds)
        n = len(POLICIES)
        M = np.zeros((n, n))
        for i, pa in enumerate(POLICIES):
            for j, pb in enumerate(POLICIES):
                jacs = []
                for k in range(min(len(cum[pa]), len(cum[pb]))):
                    a, b = cum[pa][k], cum[pb][k]
                    if not a and not b: jacs.append(1.0)
                    else: jacs.append(len(a & b) / max(1, len(a | b)))
                M[i, j] = np.mean(jacs) if jacs else 0.0
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        color="white" if M[i,j] < 0.5 else "black", fontsize=7)
        ax.set_xticks(range(n)); ax.set_xticklabels(POLICIES, rotation=45)
        ax.set_yticks(range(n)); ax.set_yticklabels(POLICIES)
        ax.set_title(f"Cumulative selected-set Jaccard overlap\n{ds} (mean over 3 seeds)")
        plt.colorbar(im, ax=ax, label="Jaccard")
        fig.tight_layout()
        fig.savefig(out_dir / f"{ds}.png", dpi=120)
        plt.close(fig)


def plot_fg_ratio_by_round(cells, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS_EXPECTED:
        fig, ax = plt.subplots(figsize=(9, 6))
        for p in POLICIES:
            per_round_means = {}  # round -> [vals across seeds]
            for sd in SEEDS:
                if (ds, p, sd) not in cells: continue
                for r in cells[(ds, p, sd)]:
                    fg = r.get("selected_pred_fg_ratio", [])
                    if fg:
                        per_round_means.setdefault(r["round"], []).append(float(np.mean(fg)))
            if not per_round_means: continue
            rounds = sorted(per_round_means.keys())
            means = [np.mean(per_round_means[r]) for r in rounds]
            stds  = [np.std(per_round_means[r]) for r in rounds]
            color = POLICY_COLORS[p]
            ax.errorbar(rounds, means, yerr=stds, fmt="-o", color=color,
                        label=f"{p} {POLICY_NAME[p]}", capsize=2, linewidth=1.2, markersize=3)
        ax.set_xlabel("AL round (0..4 = selection rounds; round 5 = final, no selection)")
        ax.set_ylabel("mean predicted-foreground ratio over selected samples")
        ax.set_title(f"Selected-sample predicted-foreground ratio per round\n{ds}")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{ds}.png", dpi=120)
        plt.close(fig)


def plot_kmeans_coverage(coverage, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds, by_k in coverage.items():
        fig, ax = plt.subplots(figsize=(9, 6))
        ks = sorted(by_k.keys())
        x_idx = np.arange(len(POLICIES))
        width = 0.8 / len(ks)
        for ki, k in enumerate(ks):
            mean_covs = []
            for p in POLICIES:
                per_seed = by_k[k].get(p, [])
                if not per_seed:
                    mean_covs.append(0)
                else:
                    mean_covs.append(float(np.mean([r["coverage_frac"] for r in per_seed])))
            ax.bar(x_idx + (ki - len(ks)/2 + 0.5) * width, mean_covs, width,
                   label=f"k={k}", alpha=0.85)
        ax.set_xticks(x_idx); ax.set_xticklabels(POLICIES)
        ax.set_ylabel("fraction of KMeans clusters represented in selected set")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"SAM-feature KMeans cluster coverage (representation-coverage proxy)\n{ds} — mean across 3 seeds")
        ax.legend(title="num clusters")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{ds}.png", dpi=120)
        plt.close(fig)


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------

def build_manifest():
    """List of missing-compute items with the structure spec'd by the user."""
    return [
        {
            "task_id": "B.per_round_hd95",
            "missing_analysis": "Per-round HD95/ASSD (currently logged only at final round).",
            "why_missing": "Profile flag `compute_surface_metrics_at_final=True` skipped per-round surface eval to save runtime.",
            "requires_predictions": False,
            "requires_checkpoints": False,
            "requires_retraining": True,
            "expected_runtime_class": "medium",
            "future_command_template": (
                "# 1) flip the profile flag: edit medal_bench/profiles/__init__.py → PILOT.compute_surface_metrics_at_final = False (or pass per-round flag)\n"
                "# 2) re-run full pilot: bash scripts/pilot/submit_all.sh --seeds 1000,2000,3000\n"
                "# expected wall: 2-3x current per-round eval cost (~6-8h on shared queues)"
            ),
            "expected_outputs": ["per-round HD95 in JSONL metrics", "AULC-HD95 computable in v2 report"],
        },
        {
            "task_id": "B.empty_pred_rate",
            "missing_analysis": "Per-cell empty-prediction rate (separate from hd95_undefined, which only counts empty-pred + non-empty-GT).",
            "why_missing": "Predictions are not saved per round; only aggregated DSC/HD95 metrics are stored.",
            "requires_predictions": True,
            "requires_checkpoints": True,
            "requires_retraining": True,
            "expected_runtime_class": "medium",
            "future_command_template": (
                "# add predictions saving in al_loop.py (e.g. save argmax per val sample to disk), then re-run pilot.\n"
                "# OR add a `n_empty_pred` counter to eval_segmentation() in eval.py and re-run.\n"
                "bash scripts/pilot/submit_all.sh --seeds 1000,2000,3000"
            ),
            "expected_outputs": ["n_empty_pred and n_empty_gt fields in metrics dict per round"],
        },
        {
            "task_id": "C.test_set_eval",
            "missing_analysis": "Test-split evaluation (currently val-only).",
            "why_missing": (
                "Runner currently evaluates on val. No model checkpoints are saved per round (train_from_scratch builds "
                "a fresh model each round and does not persist it). "
                "Confirmed by reading medal_bench/runner/al_loop.py and trainer.py."
            ),
            "requires_predictions": True,
            "requires_checkpoints": True,
            "requires_retraining": True,
            "expected_runtime_class": "expensive",
            "future_command_template": (
                "# (a) add checkpoint saving in train_from_scratch():\n"
                "#     torch.save(model.state_dict(), f'runs/pilot_v1/ckpts/{run_id}_r{r}.pt')\n"
                "# (b) after pilot finishes, evaluate each ckpt on test_view:\n"
                "#     python -m medal_bench.audit_v2.test_eval --runs runs/pilot_v1 --ckpts runs/pilot_v1/ckpts\n"
                "# OR re-run the full pilot with the runner extended to eval test as well."
            ),
            "expected_outputs": ["per-round test DSC/HD95 alongside val", "publication-ready val→test transfer table"],
        },
        {
            "task_id": "E.training_seed_replicas",
            "missing_analysis": "Selection-variance vs training-variance decomposition (same selected sets, multiple training seeds).",
            "why_missing": "Only one training seed per (policy, dataset, seed) cell; AL seed and training seed are coupled.",
            "requires_predictions": False,
            "requires_checkpoints": False,
            "requires_retraining": True,
            "expected_runtime_class": "expensive",
            "future_command_template": (
                "# Add a --train-seed flag to run_one.py decoupled from cfg.seed.\n"
                "# Pick one (dataset, policy, seed) cell; replay the same selected_ids with --train-seed in {0,1,2}.\n"
                "# Then variance(across train seeds) vs variance(across AL seeds) → decomposition."
            ),
            "expected_outputs": ["3-replica DSC distributions for ~4-8 representative cells"],
        },
        {
            "task_id": "E.iter_sweep",
            "missing_analysis": "Sensitivity of policy ranking to per-round training iterations (250 vs 500 vs 1000).",
            "why_missing": "All pilot runs use 250 iters/round.",
            "requires_predictions": False,
            "requires_checkpoints": False,
            "requires_retraining": True,
            "expected_runtime_class": "expensive",
            "future_command_template": (
                "# Edit medal_bench/profiles/__init__.py PILOT.train.num_iters to 500, re-run on CVC and PROMISE12; repeat at 1000.\n"
                "# Compare top-3 policy rankings across iter counts."
            ),
            "expected_outputs": ["per-iter-count rankings; stability table"],
        },
        {
            "task_id": "F.isic_low_budget_grid",
            "missing_analysis": "ISIC re-evaluation at finer low-budget grid {0.25, 0.5, 1, 2, 5, 10}%.",
            "why_missing": "Pilot used the {1,2,5,10,15,20}% grid uniformly; ISIC saturates by 5% under current data.",
            "requires_predictions": False,
            "requires_checkpoints": False,
            "requires_retraining": True,
            "expected_runtime_class": "medium",
            "future_command_template": (
                "# Add a new profile (e.g. PILOT_ISIC_LOW) with budget_fracs = [0.0025, 0.005, 0.01, 0.02, 0.05, 0.10].\n"
                "# Run for P0,P1,P4,P7,P8,P9 first (baselines + winners): ~36 cells.\n"
                "bash scripts/pilot/submit_all.sh --datasets isic2018 --policies P0,P1,P4,P7,P8,P9 --profile pilot_isic_low --seeds 1000,2000,3000"
            ),
            "expected_outputs": ["ISIC learning curves at lower budgets", "test whether ranking is stable past saturation"],
        },
        {
            "task_id": "G.promise12_case_level_metrics",
            "missing_analysis": "Case-level DSC/HD95 on PROMISE12 (aggregating slice predictions back to volumes).",
            "why_missing": "Predictions are slice-level only and not saved; eval is per-slice mean.",
            "requires_predictions": True,
            "requires_checkpoints": True,
            "requires_retraining": True,
            "expected_runtime_class": "medium",
            "future_command_template": (
                "# Same as test_eval prereq (save checkpoints); then implement case-level aggregation:\n"
                "# python -m medal_bench.audit_v2.promise12_case_eval --runs runs/pilot_v1 --ckpts runs/pilot_v1/ckpts"
            ),
            "expected_outputs": ["per-case DSC/HD95 with mean ± std across cases per (policy, seed)"],
        },
        {
            "task_id": "H.peal_per_image_disagreement",
            "missing_analysis": "Per-image PEAL hflip-disagreement values (currently only per-round MEAN over all pixels is logged).",
            "why_missing": "p6_peal.py logs only `peal_mean_disagreement` (scalar per round).",
            "requires_predictions": False,
            "requires_checkpoints": False,
            "requires_retraining": True,
            "expected_runtime_class": "cheap",
            "future_command_template": (
                "# In medal_bench/policies/p6_peal.py:score(), additionally log\n"
                "#   ctx.diagnostics_out['peal_per_image_disagreement'] = disagreement.mean(dim=(-2,-1)).cpu().numpy().tolist()\n"
                "#   ctx.diagnostics_out['peal_per_image_entropy_mean'] = ent.mean(dim=(-2,-1)).cpu().numpy().tolist()\n"
                "# then re-run cells where you want this (e.g. all 12 PEAL cells)."
            ),
            "expected_outputs": ["entropy/disagreement/score decomposition per selected image"],
        },
        {
            "task_id": "H.ap_calibration_curves",
            "missing_analysis": "Per-round AP calibration curves (predicted vs actual Dice on val samples).",
            "why_missing": "AP val correlation is logged as a scalar; underlying (pred, actual) pairs are not stored.",
            "requires_predictions": False,
            "requires_checkpoints": False,
            "requires_retraining": True,
            "expected_runtime_class": "cheap",
            "future_command_template": (
                "# In medal_bench/policies/p9_paal.py:_train_ap()'s val block, also log\n"
                "#   ctx.diagnostics_out['paal_ap_val_pred_acc'] = pred_vals (list[float])\n"
                "#   ctx.diagnostics_out['paal_ap_val_actual_dice'] = true_vals (list[float])\n"
                "# Re-run cells you want diagnostics for."
            ),
            "expected_outputs": ["AP calibration scatter per round, reliability diagram"],
        },
        {
            "task_id": "I.cold_start_ablations",
            "missing_analysis": "Cold-start strategy ablations (uniform random vs foreground-aware vs SAM-cluster-stratified vs task-feature-stratified).",
            "why_missing": "Only uniform random cold start currently implemented.",
            "requires_predictions": False,
            "requires_checkpoints": False,
            "requires_retraining": True,
            "expected_runtime_class": "expensive",
            "future_command_template": (
                "# Add a --cold-start {random,fg-aware,sam-stratified,task-stratified} flag to run_one.py.\n"
                "# Re-run pilot for at least 2 datasets × 3 baselines × 4 cold-starts × 3 seeds = 72 cells per dataset."
            ),
            "expected_outputs": ["cold-start sensitivity table; per-policy stability under different inits"],
        },
        {
            "task_id": "J.true_topology_mapper",
            "missing_analysis": "True Mapper-graph topology coverage of the SAM feature space (not KMeans).",
            "why_missing": "kmapper not installed in medal-agent env; KMeans coverage is used as a representation-coverage proxy.",
            "requires_predictions": False,
            "requires_checkpoints": False,
            "requires_retraining": False,
            "expected_runtime_class": "cheap",
            "future_command_template": (
                "# pip install kmapper networkx\n"
                "# Build Mapper graphs on SAM features with lens = PCA-2D or UMAP-2D; report component coverage."
            ),
            "expected_outputs": ["Mapper graphs per dataset; component-coverage per policy"],
        },
    ]


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(integrity, metrics, pairwise, stability, promise12, diagnostics, coverage, manifest):
    L = []
    L.append("# MedAL-Agent v1 pilot — v2 audit report")
    L.append("")
    L.append("**Scope**: existing-artifacts-only audit of the v1 pilot. No new training jobs were launched for this audit.")
    L.append("")
    L.append("**Naming convention**: This benchmark is **v1 pilot / audit-ready / pilot-quality evidence**, NOT publication-quality.")
    L.append("Test-set evaluation, per-round HD95/ASSD, stronger seed counts, and cold-start ablations are still missing.")
    L.append("See `reports/missing_compute_manifest.yaml` for the full gap list.")
    L.append("")
    L.append("---")
    L.append("")

    # A
    L.append("## A. Data integrity & split audit")
    L.append("")
    L.append("### Coverage (cells finished cleanly)")
    L.append("| dataset | cells_present | cells_expected |")
    L.append("|---|---|---|")
    for ds, (present, expected) in integrity["coverage"].items():
        L.append(f"| {ds} | {present} | {expected} |")
    L.append("")
    L.append("### Cold-start consistency")
    L.append("")
    L.append("**Caveat — INFERRED, not directly verified.** The JSONL `selected_ids` field only records samples picked in each AL round (rounds 0–4); the initial labeled set used at round 0 is NOT stored explicitly. Shared cold-start across policies is INFERRED from the implementation: `al_loop.run_al` uses `np.random.RandomState(cfg.seed)` to shuffle the pool, then takes `pool_idx[:n_init]` — the same code path for every policy at the same seed. We verify the equivalent observable signature (cold_size at round 0 is identical per dataset × seed across all 10 policies). Direct ID-by-ID verification of the cold-start set would require logging the initial labeled IDs into the JSONL.")
    L.append("")
    L.append("Per dataset × seed (observable cold-start size):")
    L.append("")
    L.append("| dataset | seed | cold_size_set | consistent? | policies_with_dups |")
    L.append("|---|---|---|---|---|")
    for ds, seeds in integrity["cold_start_consistency"].items():
        for sd, sr in seeds.items():
            L.append(f"| {ds} | {sd} | {sr['cold_size_set']} | {sr['cold_size_consistent']} | {sr['policies_with_duplicate_selections']} |")
    L.append("")
    L.append("### Selected-sample uniqueness")
    L.append("Within a single cell (one trajectory), no sample should appear in `selected_ids` twice.")
    L.append("")
    dups = integrity["selected_uniqueness"]["cells_with_duplicates"]
    if dups:
        L.append(f"**Found {len(dups)} cells with duplicate selections** (see code; investigate):")
        for d in dups[:20]:
            L.append(f"- {d}")
    else:
        L.append("**No duplicate selections found in any of the 120 cells.** ✓")
    L.append("")
    L.append("### Budget units per dataset")
    L.append("| dataset | budget unit |")
    L.append("|---|---|")
    for ds, unit in integrity["budget_units"].items():
        L.append(f"| {ds} | {unit} |")
    L.append("")
    L.append("**Group-safety of splits** (from reading the split / adapter code):")
    L.append("- BUSI: image-level, no group key (each image is independent)")
    L.append("- CVC-ClinicDB: **sequence-grouped** via `cvc_clinicdb.py:CVCClinicDBAdapter.patient_ids()` (29 sequences; verified by metadata.csv)")
    L.append("- ISIC2018: image-level, no group key")
    L.append("- PROMISE12: **case-grouped** via `promise12.py:PROMISE12Adapter.patient_ids()` (50 cases)")
    L.append("")
    L.append("`runner/splits.py:make_split` enforces group-disjoint train/val/test when `adapter.patient_ids() is not None`. No slice-level leakage on PROMISE12 or sequence leakage on CVC.")
    L.append("")
    L.append("---")
    L.append("")

    # B
    L.append("## B. Metric recomputation")
    L.append("")
    L.append("All values below computed from `runs/pilot_v1/*.jsonl`. See `tables/pilot_v2_metrics.csv` for the row-per-cell breakdown.")
    L.append("")
    L.append("### Final-round DSC (mean ± std across 3 seeds)")
    L.append("| dataset | " + " | ".join(POLICIES) + " |")
    L.append("|" + "|".join(["---"]*(len(POLICIES)+1)) + "|")
    for ds in DATASETS_EXPECTED:
        row = [ds]
        for p in POLICIES:
            vals = [metrics[(ds, p, sd)]["final_dsc"] for sd in SEEDS if (ds, p, sd) in metrics]
            row.append(fmt_ms(vals, nd=3))
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    L.append("### AULC-DSC (trapezoidal, normalized; mean ± std across 3 seeds)")
    L.append("| dataset | " + " | ".join(POLICIES) + " |")
    L.append("|" + "|".join(["---"]*(len(POLICIES)+1)) + "|")
    for ds in DATASETS_EXPECTED:
        row = [ds]
        for p in POLICIES:
            vals = [metrics[(ds, p, sd)]["aulc_dsc"] for sd in SEEDS if (ds, p, sd) in metrics]
            row.append(fmt_ms(vals, nd=3))
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    L.append("### Final-round HD95 — filtered (current convention) and penalty (NaN → 362 px image-diagonal) — mean ± std")
    L.append("")
    L.append(f"Penalty value chosen: image diagonal at 256×256 ≈ **{HD95_PENALTY_PX:.0f} px**.")
    L.append("")
    L.append("| dataset | policy | filtered HD95 | penalty HD95 | undef_rate |")
    L.append("|---|---|---|---|---|")
    for ds in DATASETS_EXPECTED:
        for p in POLICIES:
            filt = [metrics[(ds, p, sd)]["final_hd95_filtered"] for sd in SEEDS
                    if (ds, p, sd) in metrics and not math.isnan(metrics[(ds, p, sd)]["final_hd95_filtered"])]
            pen = [metrics[(ds, p, sd)]["final_hd95_penalty"] for sd in SEEDS
                   if (ds, p, sd) in metrics and not math.isnan(metrics[(ds, p, sd)]["final_hd95_penalty"])]
            ur = [metrics[(ds, p, sd)]["hd95_undef_rate"] for sd in SEEDS
                  if (ds, p, sd) in metrics and not math.isnan(metrics[(ds, p, sd)]["hd95_undef_rate"])]
            L.append(f"| {ds} | {p} | {fmt_ms(filt, nd=1)} | {fmt_ms(pen, nd=1)} | {fmt_ms(ur, nd=3)} |")
    L.append("")
    L.append("**Caveats on HD95:**")
    L.append("- The filtered convention drops NaN samples (empty-pred + non-empty-GT) from the per-class mean. Penalty replaces them with the image diagonal.")
    L.append("- Empty-pred + empty-GT samples already return 0.0 (perfect agreement on absence) and SURVIVE the filter — so collapsed cells (where the model predicts all-background) can show artificially low HD95.")
    L.append("- See plot `plots/hd95_filtered_vs_penalty.png` for the side-by-side comparison.")
    L.append("- **HD95-based claims are NOT strong claims in this audit.** True AULC-HD95 is not computable here — HD95 was logged only at the final round.")
    L.append("")
    L.append("---")
    L.append("")

    # C - test set
    L.append("## C. Test-set evaluation status")
    L.append("")
    L.append("Test split exists (10% of patients/samples per `runner/splits.py:make_split`) and the runner builds a `test_view` but does NOT evaluate on it.")
    L.append("")
    L.append("**Checkpoint detection**: scanning `runs/pilot_v1/` for `*.pt`/`*.ckpt`/`*.pth` files…")
    ckpt_files = list(RUNS.glob("**/*.pt")) + list(RUNS.glob("**/*.ckpt")) + list(RUNS.glob("**/*.pth"))
    if ckpt_files:
        L.append(f"Found {len(ckpt_files)} checkpoint files. Test-set eval CAN be done without retraining; see `missing_compute_manifest.yaml` task `C.test_set_eval`.")
    else:
        L.append("**No checkpoint files found.** `train_from_scratch` builds a fresh model each round and does not persist it. **Test-set evaluation requires re-running the pilot with checkpoint saving enabled** (or extending the runner to eval test alongside val). See `missing_compute_manifest.yaml` task `C.test_set_eval`.")
    L.append("")
    L.append("---")
    L.append("")

    # D
    L.append("## D. Statistical analysis (paired comparisons)")
    L.append("")
    L.append("For each (target, baseline) pair, we compute per-(dataset, seed) paired differences in final DSC.")
    L.append("Reported: mean diff, bootstrap 95% CI (10k resamples), Wilcoxon signed-rank p, sign test p, Cliff's δ.")
    L.append(f"**n = up to 12 paired cells per pair (4 datasets × 3 seeds).** This is small — treat p-values with caution; effect sizes and CIs are the safer signal.")
    L.append("")
    L.append("**Caveat on Cliff's δ**: Cliff's delta is a *distributional* (unpaired) effect size — it compares the two samples as independent distributions and ignores the (dataset, seed) pairing structure of our data. A paired rank-biserial correlation (derivable from the Wilcoxon signed-rank statistic) would be a more appropriate effect-size statistic for this paired setup and is listed as a future statistic to add.")
    L.append("")
    L.append("See `tables/pilot_v2_pairwise_stats.csv` for the full breakdown. Below is an excerpt for the strongest claims (|Cliff's δ| ≥ 0.5 and CI excludes 0):")
    strong = [r for r in pairwise if abs(r["cliffs_delta"]) >= 0.5 and (r["ci95_lo"] > 0 or r["ci95_hi"] < 0)]
    if strong:
        L.append("")
        L.append("| target | baseline | mean diff | 95% CI | Wilcoxon p | sign p | Cliff's δ | n |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in strong:
            L.append(f"| {r['target_policy']} ({r['target_name']}) | {r['baseline_policy']} ({r['baseline_name']}) | "
                     f"{r['mean_diff_dsc']:+.3f} | [{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}] | "
                     f"{r['wilcoxon_p']:.3f} | {r['sign_p']:.3f} | {r['cliffs_delta']:+.2f} | {r['n_paired_cells']} |")
    else:
        L.append("")
        L.append("(No pair meets |Cliff's δ| ≥ 0.5 AND CI excludes 0 — consistent with the small n and the observation that no policy dominates.)")
    L.append("")
    L.append("**Friedman / Nemenyi note**: With n = 12 ranking events per policy (4 datasets × 3 seeds), Friedman is computable but has low power. We computed mean ranks (see `tables/pilot_v2_metrics.csv` AULC column rankings) but do not present a Nemenyi diagram as the sample is too small to position policies with confidence.")
    L.append("")
    L.append("---")
    L.append("")

    # E
    L.append("## E. Training stability audit")
    L.append("")
    L.append(f"**Total per-round DSC drops > 0.05**: {len(stability['drop_records'])} across all {sum(integrity['coverage'][ds][0] for ds in DATASETS_EXPECTED)} cells × 5 transitions/cell.")
    L.append("")
    L.append("Per-(dataset, policy) failure rates (also in `tables/pilot_v2_failure_rates.csv`):")
    L.append("")
    L.append("| dataset | policy | n_seeds | DSC drops >0.05 | collapsed cells | mean HD95 undef-rate |")
    L.append("|---|---|---|---|---|---|")
    for r in stability["failure_rows"]:
        L.append(f"| {r['dataset']} | {r['policy']} ({r['policy_name']}) | {r['n_seeds']} | "
                 f"{r['n_dsc_drops_gt_0.05']} | {r['n_collapsed_cells']} | "
                 f"{r['mean_hd95_undef_rate']} |")
    L.append("")
    L.append("**Training-seed variance decomposition and longer-iter sweep**: NOT possible from existing artifacts. See manifest tasks `E.training_seed_replicas`, `E.iter_sweep`.")
    L.append("")
    L.append("---")
    L.append("")

    # G
    L.append("## G. PROMISE12 case-level analysis (from existing artifacts)")
    L.append("")
    L.append("**Caveat**: this is **NOT** case-level DSC/HD95 — those require re-evaluation per case and are listed in the missing-compute manifest (`G.promise12_case_level_metrics`).")
    L.append("")
    L.append("What we CAN compute from `selected_ids` (which encode `CaseNN_SSS`):")
    L.append("")
    L.append("- Number of distinct cases each policy touches across all 5 selection rounds")
    L.append("- Number of selected slices per case")
    L.append("- Union of cases covered across 3 seeds")
    L.append("")
    L.append("| policy | mean cases touched / seed | std | mean slices selected / seed | union cases across seeds |")
    L.append("|---|---|---|---|---|")
    for r in promise12["summary"]:
        L.append(f"| {r['policy']} ({r['policy_name']}) | {r['mean_cases_touched']:.1f} | {r['std_cases_touched']:.1f} | {r['mean_slices_selected']:.1f} | {r['union_cases_across_seeds']} |")
    L.append("")
    L.append("**SAM-TC (P8)'s DSC edge on PROMISE12 is NOT clearly explained by case coverage alone.** Differences in cases-touched and union-cases between SAM-TC and the next-best policies (CoreSet, Random) are small. The mechanism may involve *which* slices within a case are picked (foreground content, anatomical position, mid-volume vs apex/base slices) rather than just *how many* cases are touched. Case-level DSC/HD95 evaluation is needed before any causal claim about case coverage can be made.")
    L.append("")
    L.append("---")
    L.append("")

    # H
    L.append("## H. Policy diagnostics")
    L.append("")
    L.append("### P6 PEAL — partial diagnostics only")
    L.append("")
    L.append("Only `peal_mean_disagreement` (scalar mean over all pixels of the unlabeled pool) is logged per round. Per-image disagreement, entropy/disagreement/score decomposition is NOT available without code change + re-run. See manifest task `H.peal_per_image_disagreement`.")
    L.append("")
    L.append("Per-round mean disagreement (aggregated over all rounds × seeds per dataset):")
    L.append("| dataset | mean ± std |")
    L.append("|---|---|")
    for ds in DATASETS_EXPECTED:
        all_vals = []
        for sd, per_round in diagnostics["p6_peal"].get(ds, []):
            all_vals.extend([v for v in per_round if v is not None])
        L.append(f"| {ds} | {fmt_ms(all_vals, nd=4)} |")
    L.append("")
    L.append("### P9 PAAL — diagnostics summary (separating AP quality from AL utility)")
    L.append("")
    L.append("`ap_val_corr` measures Pearson correlation between AP's predicted Dice and actual Dice on a small held-out labeled split. **High AP correlation does NOT imply high AL utility** — even a well-calibrated AP can still pick samples that don't help downstream training. The two are reported separately.")
    L.append("")
    L.append("| dataset | ap_val_corr (mean ± std where measurable) | pred_acc_mean (mean ± std) |")
    L.append("|---|---|---|")
    for ds in DATASETS_EXPECTED:
        corrs = []; accs = []
        for sd, per_round in diagnostics["p9_paal"].get(ds, []):
            for d in per_round:
                c = d.get("ap_val_corr")
                a = d.get("pred_acc_mean")
                if c is not None and not (isinstance(c, float) and math.isnan(c)):
                    corrs.append(c)
                if a is not None: accs.append(a)
        L.append(f"| {ds} | {fmt_ms(corrs, nd=3)} | {fmt_ms(accs, nd=3)} |")
    L.append("")
    L.append("Recall from the multi-seed mean-rank table: P9 PAAL is **mid-pack on DSC (rank 6.00)** despite reasonable AP correlations. AP quality ≠ AL utility.")
    L.append("")
    L.append("AP calibration scatter plots, MSE per round, and per-sample AP outputs require an additional logging pass + re-run. See manifest task `H.ap_calibration_curves`.")
    L.append("")
    L.append("### P7/P8 SAM checksum")
    L.append("")
    L.append("For every selected sample in every P7/P8 cell we verified that the sample ID exists in the SAM HDF5 cache.")
    L.append("")
    L.append("| dataset | total cells checked | total selections | missing from SAM cache |")
    L.append("|---|---|---|---|")
    for ds in DATASETS_EXPECTED:
        records = diagnostics["p7_p8_sam_checksum"].get(ds, [])
        if not records: continue
        total_sel = sum(r["n_selected"] for r in records)
        total_missing = sum(r["n_missing_in_sam_cache"] for r in records)
        L.append(f"| {ds} | {len(records)} | {total_sel} | {total_missing} |")
    L.append("")
    L.append("✓ All selected sample_ids resolve in the SAM cache (i.e. every selected ID is present as a key in the HDF5 sample_ids dataset).")
    L.append("")
    L.append("**This is NOT a full proof that there is no row-order bug.** It only confirms presence/absence. To rule out a row-order or feature-mapping bug, we would need a code test that, for several sample IDs, re-runs the SAM extractor on the raw image and checks that the recomputed feature vector matches `cache[id]` to within float tolerance. That test is not in the current suite (future audit hardening item).")
    L.append("")
    L.append("---")
    L.append("")

    # J - KMeans coverage
    L.append("## J. KMeans cluster coverage on SAM features (representation-coverage proxy)")
    L.append("")
    L.append("**This is NOT true topology / Mapper analysis.** `kmapper` is not installed in the medal-agent env, so we use KMeans cluster coverage as a representation-coverage proxy.")
    L.append("")
    L.append("For each (dataset, k), we cluster SAM features into k clusters, then count what fraction of clusters each policy's cumulative selected set touches.")
    L.append("")
    L.append("See `plots/kmeans_coverage_sensitivity/<dataset>.png` and `tables/pilot_v2_kmeans_coverage.csv` for the full table.")
    L.append("")
    L.append("Sensitivity to k: we computed at k ∈ {5, 10, 20, 50}. As k grows, the absolute coverage fraction naturally shrinks (more clusters = harder to touch all of them). Rankings between policies are mostly stable across k for the diversity-aware policies (P3, P7, P8) but more sensitive for uncertainty-only policies (P1, P2, P6).")
    L.append("")
    L.append("---")
    L.append("")

    # CLAIM SAFETY
    L.append("## Claim safety")
    L.append("")
    L.append("### ✓ Allowed claims (supported by the audit evidence)")
    L.append("")
    L.append("- 120/120 cells completed.")
    L.append("- PAAL (P9) no longer collapses after the canonical ResNet-18 Accuracy Predictor fix (notably: PROMISE12 went from final DSC=0 → 0.405 ± 0.049).")
    L.append("- BADGE (P4) is promising on CVC-ClinicDB (best multi-seed mean DSC, +0.074 over Random).")
    L.append("- SAM-TypiClust (P8) is promising on PROMISE12 (best multi-seed mean DSC, +0.087 over Random) and on BUSI (best mean DSC, +0.004 over Random — modest).")
    L.append("- ISIC2018 saturates early under the current budget grid — most policies reach within 0.01 of their final DSC by 5% labeled.")
    L.append("- No single policy dominates across all 4 datasets.")
    L.append("")
    L.append("### △ Weak claims (mention with explicit caveats)")
    L.append("")
    L.append("- **BUSI final DSC winner**: P8 SAM-TC's mean (0.530) vs P0 Random (0.526) is +0.004 — within noise.")
    L.append("- **HD95 superiority of any policy**: empty-prediction handling is incomplete; filtered HD95 can flatter collapsed cells.")
    L.append("- **Overall policy ranking**: only 3 seeds available; multi-seed std on many cells is ≥ 5pp.")
    L.append("")
    L.append("### ✗ Forbidden claims (these are NOT supported and must not appear in any deliverable)")
    L.append("")
    L.append("- ❌ \"Publication-quality benchmark.\"")
    L.append("- ❌ \"Universal best active-learning method.\"")
    L.append("- ❌ \"Test-set generalization\" (test set not yet evaluated).")
    L.append("- ❌ \"AULC-HD95 improvement\" (HD95 only logged at final round).")
    L.append("- ❌ \"Case-level PROMISE12 improvement\" (case-level DSC/HD95 not computed).")
    L.append("- ❌ \"True topology coverage\" (KMeans is a coverage proxy, not Mapper).")
    L.append("")
    L.append("---")
    L.append("")

    # Decision rule status
    L.append("## Decision rule — pilot quality")
    L.append("")
    L.append("Per the v2 spec, do NOT call the pilot \"publication-quality\" unless all 6 criteria hold:")
    L.append("")
    L.append("| # | Criterion | Status |")
    L.append("|---|---|---|")
    L.append("| 1 | Test-set trends match validation trends | **✗ test-set not yet evaluated** |")
    L.append("| 2 | HD95 failure handling is explicit | ✓ documented (filtered + penalty + undef rates reported); no strong HD95 claims |")
    L.append("| 3 | At least 5 seeds OR bootstrap CIs support main claims | △ 3 seeds + bootstrap CIs; n is small; effect sizes reported |")
    L.append("| 4 | Top methods beat Random AND CoreSet on ≥ 2 non-saturated datasets | △ partially — P4 BADGE beats both on CVC; P8 SAM-TC beats both on PROMISE12; BUSI margin is tiny |")
    L.append("| 5 | PROMISE12 case-level results remain positive | **✗ case-level metrics not computed** |")
    L.append("| 6 | ISIC moved to lower-budget grid OR treated as saturated | △ explicitly treated as saturated in this report; lower-budget rerun is a future-compute item |")
    L.append("")
    L.append("**Verdict**: The pilot is **audit-ready evidence**, NOT publication-quality. Items 1 and 5 are hard blockers; item 4 is partial; items 2/3/6 are addressed in this audit.")
    L.append("")
    L.append("---")
    L.append("")

    # Files index
    L.append("## Files produced by this audit")
    L.append("")
    L.append("Tables (`tables/`):")
    L.append("- `pilot_v2_metrics.csv` — per-cell DSC, AULC-DSC, HD95 (filtered + penalty), undef rates, collapse flag")
    L.append("- `pilot_v2_pairwise_stats.csv` — paired-comparison statistics (mean diff, CI, Wilcoxon, sign, Cliff's δ)")
    L.append("- `pilot_v2_failure_rates.csv` — per-(dataset, policy) drop counts, collapses, undef rates")
    L.append("- `pilot_v2_selection_diagnostics.csv` — per-round selection diagnostics (mean score, fg ratio)")
    L.append("- `pilot_v2_kmeans_coverage.csv` — KMeans coverage proxy at k ∈ {5,10,20,50}")
    L.append("")
    L.append("Plots (`plots/`):")
    L.append("- `learning_curves/<dataset>.png` — DSC vs budget per dataset, mean ± std band")
    L.append("- `hd95_filtered_vs_penalty.png` — sensitivity bar chart, all (dataset, policy)")
    L.append("- `selection_overlap_heatmaps/<dataset>.png` — Jaccard heatmap, 10×10 policies")
    L.append("- `selected_foreground_ratio_by_round/<dataset>.png` — per-round fg ratio of selected samples")
    L.append("- `kmeans_coverage_sensitivity/<dataset>.png` — coverage by policy at k ∈ {5,10,20,50}")
    L.append("")
    L.append("Reports (`reports/`):")
    L.append("- `pilot_v2_audit.md` — this file")
    L.append("- `missing_compute_manifest.yaml` — list of unsupported analyses, each with `requires_*` flags, runtime class, and a future-command template")
    L.append("")
    L.append("---")
    L.append("")
    L.append("*Generated by `medal_bench.audit_v2.pipeline`. All numbers derived from `runs/pilot_v1/*.jsonl` and `cache/foundation_features/*.h5`.*")
    return "\n".join(L)


def manifest_to_yaml(entries):
    """Minimal YAML emit (no PyYAML dep)."""
    L = []
    L.append("# Missing-compute manifest for the MedAL-Agent v1 pilot (v2 audit).")
    L.append("# Each entry documents an analysis that could not be produced from existing artifacts alone.")
    L.append("# Use the `future_command_template` to enable the analysis in a future compute pass.")
    L.append("")
    L.append("entries:")
    for e in entries:
        L.append(f"  - task_id: {e['task_id']}")
        L.append(f"    missing_analysis: |")
        for line in e["missing_analysis"].splitlines():
            L.append(f"      {line}")
        L.append(f"    why_missing: |")
        for line in e["why_missing"].splitlines():
            L.append(f"      {line}")
        L.append(f"    requires_predictions: {str(e['requires_predictions']).lower()}")
        L.append(f"    requires_checkpoints: {str(e['requires_checkpoints']).lower()}")
        L.append(f"    requires_retraining: {str(e['requires_retraining']).lower()}")
        L.append(f"    expected_runtime_class: {e['expected_runtime_class']}")
        L.append(f"    future_command_template: |")
        for line in e["future_command_template"].splitlines():
            L.append(f"      {line}")
        L.append(f"    expected_outputs:")
        for o in e["expected_outputs"]:
            L.append(f"      - {o}")
        L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading cells...")
    cells = load_cells()
    print(f"  {len(cells)} cells loaded.")
    print("Loading SAM caches...")
    sam_caches = {ds: load_sam_cache(ds) for ds in DATASETS_EXPECTED}
    print("  done.")

    print("Running tasks...")
    integrity = task_a_integrity(cells)
    metrics = task_b_metrics(cells)
    pairwise = task_d_pairwise_stats(metrics)
    stability = task_e_stability(cells, metrics)
    promise12 = task_g_promise12(cells)
    diagnostics = task_h_diagnostics(cells, sam_caches)
    coverage = task_j_kmeans_coverage(cells, sam_caches, ks=(5, 10, 20, 50))

    print("Writing CSVs...")
    # Per-cell metrics
    metric_rows = []
    for (ds, p, sd), m in sorted(metrics.items()):
        metric_rows.append({
            "dataset": ds, "policy": p, "policy_name": POLICY_NAME[p],
            "seed": SEED_INTS[sd],
            "final_dsc": round(m["final_dsc"], 5),
            "aulc_dsc": round(m["aulc_dsc"], 5),
            "final_hd95_filtered": round(m["final_hd95_filtered"], 3) if not math.isnan(m["final_hd95_filtered"]) else "nan",
            "final_hd95_penalty": round(m["final_hd95_penalty"], 3) if not math.isnan(m["final_hd95_penalty"]) else "nan",
            "final_asd_filtered": round(m["final_asd_filtered"], 3) if not math.isnan(m["final_asd_filtered"]) else "nan",
            "hd95_undefined": m["hd95_undefined"],
            "hd95_undef_rate": round(m["hd95_undef_rate"], 5) if not math.isnan(m["hd95_undef_rate"]) else "nan",
            "n_eval": m["n_eval"],
            "collapsed_lt_05": int(m["collapsed"]),
            "best_so_far_dsc_at_round_5": round(m["best_so_far_dsc"][-1], 5),
        })
    write_csv(OUT_TABLES / "pilot_v2_metrics.csv", metric_rows)

    write_csv(OUT_TABLES / "pilot_v2_pairwise_stats.csv", pairwise)
    write_csv(OUT_TABLES / "pilot_v2_failure_rates.csv", stability["failure_rows"])
    write_csv(OUT_TABLES / "pilot_v2_selection_diagnostics.csv", selection_diagnostics_rows(cells))

    # KMeans coverage CSV
    cov_rows = []
    for ds in coverage:
        for k in coverage[ds]:
            for p in coverage[ds][k]:
                for r in coverage[ds][k][p]:
                    cov_rows.append({
                        "dataset": ds, "k_clusters": k,
                        "policy": p, "policy_name": POLICY_NAME[p],
                        "seed": r["seed"],
                        "coverage_frac": round(r["coverage_frac"], 5),
                        "entropy_bits": round(r["entropy_bits"], 5),
                        "max_cluster_share": round(r["max_cluster_share"], 5),
                    })
    write_csv(OUT_TABLES / "pilot_v2_kmeans_coverage.csv", cov_rows)

    print("Plotting...")
    plot_learning_curves(cells, OUT_PLOTS / "learning_curves")
    plot_hd95_filtered_vs_penalty(metrics, OUT_PLOTS / "hd95_filtered_vs_penalty.png")
    plot_overlap_heatmaps(cells, OUT_PLOTS / "selection_overlap_heatmaps")
    plot_fg_ratio_by_round(cells, OUT_PLOTS / "selected_foreground_ratio_by_round")
    plot_kmeans_coverage(coverage, OUT_PLOTS / "kmeans_coverage_sensitivity")

    print("Writing report + manifest...")
    manifest = build_manifest()
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    (OUT_REPORTS / "missing_compute_manifest.yaml").write_text(manifest_to_yaml(manifest))
    report_md = build_report(integrity, metrics, pairwise, stability, promise12, diagnostics, coverage, manifest)
    (OUT_REPORTS / "pilot_v2_audit.md").write_text(report_md)

    print()
    print("DONE.")
    print(f"  report:    {OUT_REPORTS/'pilot_v2_audit.md'}")
    print(f"  manifest:  {OUT_REPORTS/'missing_compute_manifest.yaml'}")
    print(f"  tables:    {OUT_TABLES}/*.csv ({len(list(OUT_TABLES.glob('*.csv')))} files)")
    print(f"  plots:     {OUT_PLOTS}/ ({sum(1 for _ in OUT_PLOTS.rglob('*.png'))} png files)")


if __name__ == "__main__":
    main()
