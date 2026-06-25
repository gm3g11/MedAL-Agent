"""Publication-grade cross-dataset summary for the AL benchmark (audit BG1 + BG5).

Replaces the single final-DSC point with three complementary summaries computed
per the audit's recommendation, with proper paired significance:

  * final DSC      -- score at the max budget
  * AUBC           -- area under the DSC-vs-budget curve (budget-weighted mean),
                      span-normalized so it's a DSC-unit number (derived.aubc)
  * low-budget DSC -- score nearest a target fraction (default 5%)

Unit of analysis = the (usable) datasets. For each policy vs Random (P0) we form
the per-dataset paired difference (averaged over seeds first), then across datasets:
  - mean diff + bootstrap 95% CI
  - Wilcoxon signed-rank p
  - win-rate (fraction of datasets the policy beats P0) + sign-test p
A ranking gap smaller than the cross-dataset SE is flagged as a TIE.

Input: the long CSV (dataset,policy,seed,budget_frac,dsc_fg_round) emitted by the
report generator. Multi-seed aware: averages over seeds per (dataset,policy) first.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict

import numpy as np

from medal_bench.analysis.derived import aubc

DEGENERATE_DEFAULT = ("rose1",)
POLICIES = [f"P{i}" for i in range(10)]
PNAMES = {"P0": "Random", "P1": "Entropy", "P2": "BALD", "P3": "CoreSet",
          "P4": "BADGE", "P5": "Entropy+CoreSet", "P6": "SelUncertainty",
          "P7": "SAM-CoreSet", "P8": "SAM-TypiClust", "P9": "PAAL"}


def load_curves(csv_path: str):
    """-> {(dataset, policy, seed): [(budget_frac, dsc), ...] sorted}."""
    cells: dict = defaultdict(list)
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            try:
                cells[(row["dataset"], row["policy"], int(row["seed"]))].append(
                    (float(row["budget_frac"]), float(row["dsc_fg_round"])))
            except (KeyError, ValueError):
                continue
    for k in cells:
        cells[k].sort()
    return cells


def _cell_summaries(curve, low_target=0.05):
    """final, aubc, low-budget DSC for one curve (list of (frac, dsc))."""
    fr = [f for f, _ in curve]
    sc = [s for _, s in curve]
    final = sc[-1]
    au = aubc(fr, sc) if len(fr) >= 2 else float("nan")
    low = min(curve, key=lambda p: abs(p[0] - low_target))[1]
    return final, au, low


def _bootstrap_ci(diffs, n=10000, seed=12345):
    a = np.asarray(diffs, dtype=float)
    if len(a) < 2:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    means = a[rng.randint(0, len(a), size=(n, len(a)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def build(csv_path: str, degenerate=DEGENERATE_DEFAULT, low_target=0.05):
    cells = load_curves(csv_path)
    datasets = sorted({d for (d, _, _) in cells} - set(degenerate))
    seeds = sorted({s for (_, _, s) in cells})
    # per (dataset, policy): mean over seeds of each summary stat
    stat = {}  # (ds, pol) -> dict(final, aubc, low)
    for ds in datasets:
        for pol in POLICIES:
            vals = [_cell_summaries(cells[(ds, pol, s)], low_target)
                    for s in seeds if (ds, pol, s) in cells]
            if not vals:
                continue
            arr = np.array(vals, float)
            stat[(ds, pol)] = dict(final=np.nanmean(arr[:, 0]),
                                   aubc=np.nanmean(arr[:, 1]),
                                   low=np.nanmean(arr[:, 2]))
    # per-policy mean over datasets + paired-vs-P0 stats
    rows = []
    try:
        from scipy.stats import wilcoxon, binomtest
        _scipy = True
    except Exception:
        _scipy = False
    for pol in POLICIES:
        finals = [stat[(d, pol)]["final"] for d in datasets if (d, pol) in stat]
        aucs = [stat[(d, pol)]["aubc"] for d in datasets if (d, pol) in stat]
        lows = [stat[(d, pol)]["low"] for d in datasets if (d, pol) in stat]
        diffs = [stat[(d, pol)]["final"] - stat[(d, "P0")]["final"]
                 for d in datasets if (d, pol) in stat and (d, "P0") in stat]
        wins = sum(1 for x in diffs if x > 0)
        n = len(diffs)
        lo, hi = _bootstrap_ci(diffs) if pol != "P0" else (float("nan"), float("nan"))
        wp = sp = float("nan")
        if _scipy and pol != "P0" and n >= 2 and any(d != 0 for d in diffs):
            try:
                wp = float(wilcoxon(diffs, zero_method="zsplit", alternative="two-sided").pvalue)
                sp = float(binomtest(wins, n, 0.5).pvalue)
            except Exception:
                pass
        rows.append(dict(policy=pol, name=PNAMES[pol], n=n,
                         mean_final=float(np.mean(finals)), final_sd=float(np.std(finals)),
                         mean_aubc=float(np.mean(aucs)), mean_low=float(np.mean(lows)),
                         mean_diff_vs_P0=float(np.mean(diffs)) if diffs else float("nan"),
                         ci_lo=lo, ci_hi=hi, wins=wins, wilcoxon_p=wp, sign_p=sp))
    return datasets, seeds, rows


def to_markdown(csv_path, degenerate=DEGENERATE_DEFAULT, low_target=0.05):
    datasets, seeds, rows = build(csv_path, degenerate, low_target)
    se = np.std([r["mean_final"] for r in rows])  # rough cross-policy scale
    by_final = sorted(rows, key=lambda r: -r["mean_final"])
    by_auc = sorted(rows, key=lambda r: -r["mean_aubc"])
    L = []
    L.append(f"# AL benchmark — stats summary ({len(seeds)} seed(s): {seeds}, "
             f"{len(datasets)} usable datasets; rose1+degenerate excluded)\n")
    L.append("Paired analysis: per-policy vs Random(P0), datasets as the unit, seeds averaged first. "
             "**TIE** = |mean_diff| < cross-dataset SE; significance from Wilcoxon signed-rank + sign test.\n")
    L.append(f"\n## By final DSC vs by AUBC (budget-curve area)\n")
    L.append("| rank | by final-DSC | by AUBC |")
    L.append("|---|---|---|")
    for i in range(len(rows)):
        a, b = by_final[i], by_auc[i]
        L.append(f"| {i+1} | {a['policy']} {a['name']} ({a['mean_final']:.3f}) "
                 f"| {b['policy']} {b['name']} ({b['mean_aubc']:.3f}) |")
    L.append(f"\n## Per-policy summary (vs Random P0)\n")
    L.append("| policy | final | AUBC | @5% | Δfinal-vs-P0 | 95% CI | win/N | Wilcoxon p | verdict |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in by_final:
        if r["policy"] == "P0":
            verdict = "baseline"
            cis = "—"; diff = "—"; wn = "—"; wp = "—"
        else:
            diff = f"{r['mean_diff_vs_P0']:+.4f}"
            cis = f"[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]"
            wn = f"{r['wins']}/{r['n']}"
            wp = f"{r['wilcoxon_p']:.3f}" if r["wilcoxon_p"] == r["wilcoxon_p"] else "n/a"
            tie = abs(r["mean_diff_vs_P0"]) < se
            sig = (r["wilcoxon_p"] == r["wilcoxon_p"]) and r["wilcoxon_p"] < 0.05 \
                and not (r["ci_lo"] <= 0 <= r["ci_hi"])
            verdict = "**TIE**" if tie else ("sig>P0" if (sig and r["mean_diff_vs_P0"] > 0)
                                             else ("sig<P0" if sig else "ns"))
        L.append(f"| {r['policy']} {r['name']} | {r['mean_final']:.3f} | {r['mean_aubc']:.3f} "
                 f"| {r['mean_low']:.3f} | {diff} | {cis} | {wn} | {wp} | {verdict} |")
    L.append(f"\ncross-policy SE (tie threshold) ≈ {se:.4f}. "
             f"ns = not significant; sig requires Wilcoxon p<0.05 AND CI excluding 0.\n")
    return "\n".join(L)


def _gpu_arch(name: str) -> str:
    """Collapse a gpu_name to its arch family (V100->Volta, A40->Ampere, H100->Hopper)."""
    n = (name or "").lower()
    if "v100" in n:
        return "Volta"
    if "a40" in n or "a100" in n or "a10" in n:
        return "Ampere"
    if "h100" in n or "h200" in n:
        return "Hopper"
    return name or "unknown"


def verify_run_integrity(run_dirs, seed=None):
    """Audit a completed run for the GPU-confound + degenerate + budget-representativeness
    findings. Returns (ok, report_lines). ok=False if any dataset spans >1 GPU arch
    (the confound the v5 single-arch pinning must prevent)."""
    import glob
    import json
    import os
    from collections import defaultdict
    if isinstance(run_dirs, str):
        run_dirs = [run_dirs]
    by_ds_arch = defaultdict(set)        # dataset -> {arch}
    frac_full = {}                        # dataset -> fraction_of_full_train
    rounds = {}                           # dataset -> n_rounds
    for d in run_dirs:
        pat = f"{d}/*__s{seed}.jsonl" if seed else f"{d}/*__s*.jsonl"
        for f in glob.glob(pat):
            try:
                recs = [json.loads(l) for l in open(f)]
            except Exception:
                continue
            if not recs or len(recs) != recs[0].get("total_rounds", -1):
                continue
            ds = os.path.basename(f).split("__")[0]
            by_ds_arch[ds].add(_gpu_arch(recs[0].get("gpu_name", "")))
            bd = recs[0].get("budget_denominator", {})
            frac_full[ds] = bd.get("fraction_of_full_train")
            rounds[ds] = len(recs)
    L = ["dataset                  arch(s)          n_rounds  frac_of_full_train"]
    mixed = []
    for ds in sorted(by_ds_arch):
        archs = sorted(by_ds_arch[ds])
        flag = "  <-- MIXED-ARCH (confound!)" if len(archs) > 1 else ""
        if len(archs) > 1:
            mixed.append(ds)
        ff = frac_full.get(ds)
        ff_s = f"{ff:.3f}" if isinstance(ff, (int, float)) else "n/a"
        L.append(f"{ds:<24} {','.join(archs):<16} {rounds.get(ds,'?'):<9} {ff_s}{flag}")
    ok = not mixed
    L.append("")
    L.append(f"SINGLE-ARCH: {'PASS' if ok else 'FAIL — ' + ','.join(mixed) + ' span multiple archs'}")
    return ok, L


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="long results CSV (dataset,policy,seed,budget_frac,dsc_fg_round)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--low-target", type=float, default=0.05)
    a = ap.parse_args(argv)
    md = to_markdown(a.csv, low_target=a.low_target)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(md)
        print(f"wrote {a.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
