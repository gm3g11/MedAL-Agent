"""v3 plots — minimal essential set:
- plots/v3_learning_curves_val_test/<ds>.png   per-policy mean ± std DSC over budget, val solid + test dashed
- plots/v3_val_vs_test_scatter.png             one big scatter, color by dataset
- plots/v3_spearman_per_seed.png               bar of Spearman ρ per (ds, seed)
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

REPO = Path("/groups/echambe2/gmeng/MedAL-Agent/repo/code")
V3_DIR = REPO / "runs" / "test_eval_v3"
OUT_PLOTS = REPO / "plots"

POLICIES = ["P0","P1","P2","P3","P4","P5","P6","P7","P8","P9"]
POLICY_NAME = {"P0":"Random","P1":"NormEnt","P2":"BALD","P3":"CoreSet","P4":"BADGE",
               "P5":"Ent→CS","P6":"SelUnc","P7":"SAM-CS","P8":"SAM-TC","P9":"PAAL"}
POLICY_COLORS = {"P0":"#999999","P1":"#1f77b4","P2":"#ff7f0e","P3":"#2ca02c","P4":"#d62728",
                 "P5":"#9467bd","P6":"#8c564b","P7":"#e377c2","P8":"#17becf","P9":"#bcbd22"}
DS_COLORS = {"busi":"#1f77b4","cvc_clinicdb":"#ff7f0e","isic2018":"#2ca02c","promise12":"#d62728"}
DATASETS = ["busi","cvc_clinicdb","isic2018","promise12"]
SEEDS = ["s1000","s2000","s3000"]
BUDGET_PCT = [1, 2, 5, 10, 15, 20]


def load_cells():
    cells = {}
    for f in sorted(glob.glob(str(V3_DIR / "*.jsonl"))):
        name = os.path.basename(f).replace(".jsonl","")
        if name.startswith("_"): continue
        try: ds, p, sd = name.split("__")
        except ValueError: continue
        if sd not in SEEDS: continue
        cells[(ds, p, sd)] = [json.loads(l) for l in open(f)]
    return cells


def plot_learning_curves_val_test(cells, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        fig, ax = plt.subplots(figsize=(10, 6.5))
        for p in POLICIES:
            vals_per_seed = []
            tests_per_seed = []
            for sd in SEEDS:
                if (ds, p, sd) not in cells: continue
                recs = cells[(ds, p, sd)]
                if len(recs) != 6: continue
                vals_per_seed.append([r["metrics_val"]["mean_dsc_fg"] for r in recs])
                tests_per_seed.append([r["metrics_test"]["mean_dsc_fg"] for r in recs])
            if not vals_per_seed: continue
            v_arr = np.array(vals_per_seed); t_arr = np.array(tests_per_seed)
            v_mean = v_arr.mean(0); v_std = v_arr.std(0)
            t_mean = t_arr.mean(0); t_std = t_arr.std(0)
            color = POLICY_COLORS[p]
            ax.plot(BUDGET_PCT, v_mean, "-", color=color, linewidth=1.5,
                    label=f"{p} {POLICY_NAME[p]} (val)")
            ax.plot(BUDGET_PCT, t_mean, "--", color=color, linewidth=1.2, alpha=0.7,
                    label=f"{p} {POLICY_NAME[p]} (test)")
            ax.fill_between(BUDGET_PCT, v_mean - v_std, v_mean + v_std, alpha=0.10, color=color)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xticks(BUDGET_PCT); ax.set_xticklabels([f"{x}%" for x in BUDGET_PCT])
        ax.set_xlabel("labeled budget (% of train pool)")
        ax.set_ylabel("mean DSC_fg")
        ax.set_title(f"v3 learning curves (val solid, test dashed) — mean ± std over 3 seeds\n{ds}")
        ax.legend(loc="lower right", fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{ds}.png", dpi=120)
        plt.close(fig)


def plot_val_vs_test_scatter(cells, out_path):
    fig, ax = plt.subplots(figsize=(7.5, 7))
    for ds in DATASETS:
        xs = []; ys = []
        for p in POLICIES:
            for sd in SEEDS:
                if (ds, p, sd) not in cells: continue
                last = cells[(ds, p, sd)][-1]
                xs.append(last["metrics_val"]["mean_dsc_fg"])
                ys.append(last["metrics_test"]["mean_dsc_fg"])
        ax.scatter(xs, ys, color=DS_COLORS[ds], label=ds, alpha=0.7, s=40)
    # identity line
    lo, hi = 0, 1
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, linewidth=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("val DSC_fg (final round)")
    ax.set_ylabel("test DSC_fg (final round)")
    ax.set_title("v3 val vs test DSC — final round, all 120 cells\n(dashed: y = x)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_spearman_per_seed(cells, out_path):
    # Compute Spearman ρ per (ds, seed) over the 10 policies' val vs test final-round DSC
    fig, ax = plt.subplots(figsize=(9, 5))
    n_ds = len(DATASETS)
    width = 0.25
    x_idx = np.arange(n_ds)
    for j, sd in enumerate(SEEDS):
        rhos = []
        for ds in DATASETS:
            vals = []; tests = []
            for p in POLICIES:
                if (ds, p, sd) not in cells: continue
                last = cells[(ds, p, sd)][-1]
                vals.append(last["metrics_val"]["mean_dsc_fg"])
                tests.append(last["metrics_test"]["mean_dsc_fg"])
            if len(vals) < 3:
                rhos.append(float("nan"))
            else:
                rho, _ = stats.spearmanr(vals, tests)
                rhos.append(rho)
        ax.bar(x_idx + (j - 1) * width, rhos, width, label=sd)
    ax.set_xticks(x_idx); ax.set_xticklabels(DATASETS)
    ax.set_ylabel("Spearman ρ (val rank vs test rank)")
    ax.set_ylim(-0.2, 1.0)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_title("Val→Test rank stability per (dataset, seed)")
    ax.legend(title="seed")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    cells = load_cells()
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)
    plot_learning_curves_val_test(cells, OUT_PLOTS / "v3_learning_curves_val_test")
    plot_val_vs_test_scatter(cells, OUT_PLOTS / "v3_val_vs_test_scatter.png")
    plot_spearman_per_seed(cells, OUT_PLOTS / "v3_spearman_per_seed.png")
    print(f"v3 plots written under {OUT_PLOTS}/")


if __name__ == "__main__":
    main()
