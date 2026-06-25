"""Stage S3 -- DESCRIPTIVE phase-family analysis (hypotheses only).

Fixed-policy trajectories cannot identify a per-round switching policy (different
labeled sets / checkpoints / pools per method). This module only describes how the
method FAMILIES (baseline / coverage / boundary / refinement) behave by budget,
their regret and collapse risk, and whether early/mid/late rankings are stable
across seeds. It explicitly does NOT claim any switch (e.g. P8->P5) works -- that
needs S4 branching rollouts.

Run:  python -m medal_bench.skill.phase_family_analysis
Writes: reports/phase_family_analysis.md
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np

from medal_bench.skill import schema as S

FAMILY = {m: S.METHOD_DESC[m][0] for m in S.METHODS}
FAMS = ["baseline", "coverage", "boundary", "refinement"]


def _load_rounds():
    """(ds, method, seed) -> [(ratio, dsc), ...] for the 19-set."""
    cur = {}
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
        rows = [json.loads(l) for l in open(path)]
        cur[(ds, m, seed)] = [(r["labeled_ratio"], r["metrics"]["mean_dsc_fg_case_macro"]) for r in rows]
    return cur


def analyze():
    import csv
    cur = _load_rounds()
    cells = list(csv.DictReader(open(os.path.join(S.SKILL_DIR, "cells_raw.csv"))))
    for r in cells:
        r["aubc"] = float(r["aubc"]); r["is_collapse"] = int(float(r["is_collapse"]))

    lines = ["# Phase-family analysis (frozen_v5 19-set) -- DESCRIPTIVE, hypotheses only", ""]

    # family AUBC + collapse
    fam_aubc = defaultdict(list); fam_coll = defaultdict(list)
    for r in cells:
        fam_aubc[FAMILY[r["method"]]].append(r["aubc"])
        fam_coll[FAMILY[r["method"]]].append(r["is_collapse"])
    lines.append("## Family AUBC + collapse rate (over all 19-set cells)")
    lines.append(f"{'family':12s} {'meanAUBC':>9s} {'collapse%':>10s} {'n_cells':>8s}")
    for f in FAMS:
        lines.append(f"{f:12s} {np.mean(fam_aubc[f]):9.4f} {100*np.mean(fam_coll[f]):9.1f}% {len(fam_aubc[f]):8d}")

    # budget-resolved family rank: bin rounds into early/mid/late by ratio
    def stage(ratio):
        return "early" if ratio <= 0.04 else "mid" if ratio <= 0.10 else "late"
    stage_fam = {st: defaultdict(list) for st in ("early", "mid", "late")}
    for (ds, m, s), pts in cur.items():
        for ratio, dsc in pts:
            stage_fam[stage(ratio)][(ds, FAMILY[m])].append(dsc)
    lines += ["", "## Family mean DSC by budget stage (avg over datasets/seeds; rank in parens)"]
    lines.append(f"{'stage':8s} " + " ".join(f"{f:>12s}" for f in FAMS))
    for st in ("early", "mid", "late"):
        # mean per family across datasets
        per_fam = {f: np.mean([np.mean(v) for (ds, ff), v in stage_fam[st].items() if ff == f])
                   for f in FAMS}
        rk = {f: r + 1 for r, f in enumerate(sorted(FAMS, key=lambda x: -per_fam[x]))}
        lines.append(f"{st:8s} " + " ".join(f"{per_fam[f]:.3f}(#{rk[f]})".rjust(12) for f in FAMS))

    # cross-seed stability of family ranking by stage
    lines += ["", "## Cross-seed stability of family ranking (per stage)"]
    for st in ("early", "mid", "late"):
        seed_orders = []
        for s in S.SEEDS:
            pf = {}
            for f in FAMS:
                vals = []
                for (ds, m, ss), pts in cur.items():
                    if ss != s or FAMILY[m] != f:
                        continue
                    vals += [dsc for ratio, dsc in pts if stage(ratio) == st]
                pf[f] = np.mean(vals) if vals else 0
            seed_orders.append(tuple(sorted(FAMS, key=lambda x: -pf[x])))
        same = len(set(seed_orders)) == 1
        lines.append(f"  {st:6s}: {'STABLE' if same else 'VARIES'} across seeds -> {seed_orders}")

    lines += ["", "## Caveat", "These are fixed-policy observations. A family looking strong at a budget"
              " stage is a HYPOTHESIS for an S4 branching experiment, NOT evidence that switching into it"
              " mid-session helps. No switch is claimed here."]
    txt = "\n".join(lines)
    print(txt)
    os.makedirs("reports", exist_ok=True)
    open("reports/phase_family_analysis.md", "w").write(txt + "\n")
    print("\nwrote reports/phase_family_analysis.md")


if __name__ == "__main__":
    analyze()
