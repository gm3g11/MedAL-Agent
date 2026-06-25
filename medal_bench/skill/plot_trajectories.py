"""Plot the DSC-vs-budget trajectory for every method on every 19-set dataset
(seed-averaged, with ±1 std band on the two reference methods). One subplot per
dataset; solid = ALLOWED methods, dashed = BLOCKLIST; Random (P0) and BADGE (P4)
drawn bold as references.

Run:  python -m medal_bench.skill.plot_trajectories
Writes: runs/frozen_v5/skill/trajectories_19set.png
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from medal_bench.skill import schema as S

COLORS = {
    "P0": "#000000", "P1": "#8c8c8c", "P2": "#1f77b4", "P3": "#9467bd",
    "P4": "#d62728", "P5": "#ff7f0e", "P6": "#7f7f7f", "P7": "#2ca02c",
    "P8": "#17becf", "P9": "#bcbd22",
}


def _curves():
    """(ds, method) -> (ratios[R], dsc_mean[R], dsc_std[R]) seed-averaged by round."""
    by = defaultdict(lambda: defaultdict(list))   # (ds,m) -> round -> [(ratio,dsc)]
    for path in sorted(glob.glob(os.path.join(S.RUNS_DIR, "*.jsonl"))):
        b = os.path.basename(path)
        if ".partial" in b:
            continue
        name = b[:-6]
        try:
            ds, rest = name.split("__P", 1)
            m = "P" + rest.split("__s")[0]
            seed = int(rest.split("__s")[1])
        except (ValueError, IndexError):
            continue
        if ds not in S.DS19 or m not in S.METHODS or seed not in S.SEEDS:
            continue
        for r in (json.loads(l) for l in open(path)):
            by[(ds, m)][r["round"]].append(
                (r["labeled_ratio"], r["metrics"]["mean_dsc_fg_case_macro"]))
    out = {}
    for key, rounds in by.items():
        rs = sorted(rounds)
        ratio = [float(np.mean([x[0] for x in rounds[r]])) for r in rs]
        mean = [float(np.mean([x[1] for x in rounds[r]])) for r in rs]
        std = [float(np.std([x[1] for x in rounds[r]])) for r in rs]
        out[key] = (np.array(ratio), np.array(mean), np.array(std))
    return out


def plot(out_path: str | None = None):
    cur = _curves()
    meta = {r["dataset"]: r for r in __import__("csv").DictReader(
        open(os.path.join(S.SKILL_DIR, "dataset_features.csv")))}
    # order datasets by difficulty (final-round Random DSC) for a readable layout
    order = sorted(S.DS19, key=lambda d: -cur[(d, "P0")][1][-1])

    ncol, nrow = 5, 4
    fig, axes = plt.subplots(nrow, ncol, figsize=(22, 16))
    axes = axes.ravel()
    for ax, ds in zip(axes, order):
        for m in S.METHODS:
            ratio, mean, std = cur[(ds, m)]
            blk = m in S.BLOCKLIST
            ref = m in ("P0", "P4")
            ax.plot(ratio * 100, mean, color=COLORS[m],
                    ls="--" if blk else "-",
                    lw=2.6 if ref else (1.0 if blk else 1.6),
                    alpha=1.0 if ref else (0.55 if blk else 0.9),
                    marker="o" if ref else None, ms=3, zorder=5 if ref else 2)
            if ref:  # ±1 std band only on the two references to avoid clutter
                ax.fill_between(ratio * 100, mean - std, mean + std,
                                color=COLORS[m], alpha=0.10, zorder=1)
        md = meta[ds]
        ax.set_title(f"{ds}\n{md['modality']} · C{md['n_classes']} · "
                     f"pool {int(float(md['n_images']))} · {'3D' if md.get('slices_per_case','1')!='1.0' else '2D'}",
                     fontsize=9)
        ax.set_xlabel("labeled %", fontsize=8)
        ax.set_ylabel("case-macro fg DSC", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    for ax in axes[len(order):]:
        ax.axis("off")

    # shared legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=COLORS[m],
                      ls="--" if m in S.BLOCKLIST else "-",
                      lw=2.6 if m in ("P0", "P4") else 1.6,
                      label=f"{m} {S.METHOD_NAME[m]}{' [block]' if m in S.BLOCKLIST else ''}")
               for m in S.METHODS]
    fig.legend(handles=handles, loc="lower center", ncol=10, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("frozen_v5 19-set — DSC vs labeling budget (seed-averaged; solid=allowed, "
                 "dashed=blocklist; bold=Random/BADGE ±1σ)", fontsize=13, y=0.997)
    fig.tight_layout(rect=(0, 0.03, 1, 0.985))

    if out_path is None:
        out_path = os.path.join(S.SKILL_DIR, "trajectories_19set.png")
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"wrote {out_path}")
    return out_path


def _raw_curves():
    """(ds, method, seed) -> (ratios[R], dsc[R])."""
    out = {}
    for path in sorted(glob.glob(os.path.join(S.RUNS_DIR, "*.jsonl"))):
        b = os.path.basename(path)
        if ".partial" in b:
            continue
        name = b[:-6]
        try:
            ds, rest = name.split("__P", 1)
            m = "P" + rest.split("__s")[0]
            seed = int(rest.split("__s")[1])
        except (ValueError, IndexError):
            continue
        if ds not in S.DS19 or m not in S.METHODS or seed not in S.SEEDS:
            continue
        rows = sorted((json.loads(l) for l in open(path)), key=lambda r: r["round"])
        out[(ds, m, seed)] = (np.array([r["labeled_ratio"] for r in rows]),
                              np.array([r["metrics"]["mean_dsc_fg_case_macro"] for r in rows]))
    return out


def plot_per_dataset(out_dir: str | None = None):
    """One full-size figure per dataset: 3 thin seed lines + bold seed-mean per method,
    legend sorted by AUBC. Writes <out_dir>/<dataset>.png for all 19."""
    import csv
    raw = _raw_curves()
    mean_cur = _curves()
    meta = {r["dataset"]: r for r in csv.DictReader(
        open(os.path.join(S.SKILL_DIR, "dataset_features.csv")))}
    aubc = {(r["dataset"], r["method"]): float(r["aubc_mean"])
            for r in csv.DictReader(open(os.path.join(S.SKILL_DIR, "skill_rows.csv")))}
    if out_dir is None:
        out_dir = os.path.join(S.SKILL_DIR, "trajectories")
    os.makedirs(out_dir, exist_ok=True)

    paths = []
    for ds in S.DS19:
        fig, ax = plt.subplots(figsize=(11, 7))
        for m in S.METHODS:
            blk = m in S.BLOCKLIST
            ls = "--" if blk else "-"
            for seed in S.SEEDS:
                if (ds, m, seed) in raw:
                    ratio, dsc = raw[(ds, m, seed)]
                    ax.plot(ratio * 100, dsc, color=COLORS[m], ls=ls, lw=0.8, alpha=0.22, zorder=1)
            ratio, mean, _ = mean_cur[(ds, m)]
            ax.plot(ratio * 100, mean, color=COLORS[m], ls=ls, lw=2.4, alpha=0.95,
                    marker="o", ms=4, zorder=3)
        md = meta[ds]
        is3d = md.get("slices_per_case", "1.0") not in ("1.0", "1")
        ax.set_title(f"{ds}  —  {md['modality']} · C{md['n_classes']} · "
                     f"{'3D-as-slice' if is3d else '2D'} · pool {int(float(md['n_images']))} · "
                     f"fg {float(md['fg_frac_mean'])*100:.1f}%   (thin=3 seeds, bold=mean)", fontsize=11)
        ax.set_xlabel("labeled budget (%)", fontsize=11)
        ax.set_ylabel("case-macro foreground DSC", fontsize=11)
        ax.grid(alpha=0.3)
        # legend sorted by AUBC desc, with values + block tag
        from matplotlib.lines import Line2D
        ordm = sorted(S.METHODS, key=lambda m: -aubc[(ds, m)])
        handles = [Line2D([0], [0], color=COLORS[m], ls="--" if m in S.BLOCKLIST else "-", lw=2.4,
                          marker="o", ms=4,
                          label=f"{m} {S.METHOD_NAME[m]:9s} AUBC {aubc[(ds, m)]:.3f}"
                                f"{'  [block]' if m in S.BLOCKLIST else ''}")
                   for m in ordm]
        ax.legend(handles=handles, fontsize=9, loc="lower right", framealpha=0.9, title="ranked by AUBC")
        fig.tight_layout()
        p = os.path.join(out_dir, f"{ds}.png")
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    print(f"wrote {len(paths)} per-dataset figures -> {out_dir}")
    return paths


if __name__ == "__main__":
    import sys
    if "--per-dataset" in sys.argv:
        plot_per_dataset()
    else:
        plot()
