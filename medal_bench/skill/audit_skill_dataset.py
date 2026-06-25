"""Stage S1 -- audit the exported Query-Strategy-Skill dataset.

Structural QC + leakage firewall + a cross-check that the skill table reproduces
the independently-derived ceiling numbers (always-Random 0.6738, always-BADGE
0.6795, per-dataset oracle 0.6902 from scratch_skill_ceiling.py). Any FAIL means
do not train.

Run:  python -m medal_bench.skill.audit_skill_dataset
"""
from __future__ import annotations

import csv
import os

import numpy as np

from medal_bench.skill import schema as S


def _load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v) if ("." in v or "e" in v.lower() or v.lstrip("-").isdigit()) else v
            except (ValueError, AttributeError):
                pass
    return rows


def audit(skill_dir: str = S.SKILL_DIR) -> bool:
    cells = _load(os.path.join(skill_dir, "cells_raw.csv"))
    rows = _load(os.path.join(skill_dir, "skill_rows.csv"))
    ok = True
    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")

    print("== structural ==")
    check("cells_raw row count == 570", len(cells) == 570, f"got {len(cells)}")
    check("skill_rows row count == 190", len(rows) == 190, f"got {len(rows)}")
    dsm = {(r["dataset"], r["method"]) for r in rows}
    check("every (dataset x method) present", len(dsm) == 190 and
          all((d, m) in dsm for d in S.DS19 for m in S.METHODS))

    print("== leakage firewall ==")
    leak = set(S.FEATURE_COLS) & S.FORBIDDEN_FEATURE_COLS
    check("no forbidden/target col in FEATURE_COLS", not leak, str(leak))
    check("dataset_id not a feature (group key only)", "dataset" not in S.FEATURE_COLS)
    feat_present = all(c in rows[0] for c in S.FEATURE_COLS)
    check("all declared FEATURE_COLS exist in table", feat_present)

    print("== targets ==")
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    regret_ok = rank_ok = within_ok = True
    for ds, rs in by_ds.items():
        best = max(x["aubc_mean"] for x in rs)
        for x in rs:
            if abs(x["regret"] - (best - x["aubc_mean"])) > 1e-5:
                regret_ok = False
            if x["within_eps"] != int(x["regret"] <= S.EPS_AUBC + 1e-9):
                within_ok = False
        ranks = sorted(int(x["rank"]) for x in rs)
        if ranks != list(range(1, 11)):
            rank_ok = False
        if abs(min(x["regret"] for x in rs)) > 1e-9:
            regret_ok = False
    check("regret == best_aubc_mean - aubc_mean (>=0, min 0)", regret_ok)
    check("rank is a 1..10 permutation per dataset", rank_ok)
    check("within_eps == (regret <= eps)", within_ok)
    check("collapse_prob in [0,1]", all(0 <= r["collapse_prob"] <= 1 for r in rows))

    print("== round-0 shared-state consistency ==")
    # r0_dsc must be identical across a dataset's 10 method-rows (it's shared)
    r0_shared = all(len({round(x["r0_dsc"], 5) for x in rs}) == 1 for rs in by_ds.values())
    check("r0_dsc shared across methods within a dataset", r0_shared)

    print("== ceiling cross-check (vs scratch_skill_ceiling.py) ==")
    am = {(r["dataset"], r["method"]): r["aubc_mean"] for r in rows}
    rand = np.mean([am[(d, "P0")] for d in S.DS19])
    badge = np.mean([am[(d, "P4")] for d in S.DS19])
    oracle = np.mean([max(am[(d, m)] for m in S.METHODS) for d in S.DS19])
    check("always-Random AUBC ~ 0.6738", abs(rand - 0.6738) < 0.001, f"{rand:.4f}")
    check("always-BADGE AUBC ~ 0.6795", abs(badge - 0.6795) < 0.001, f"{badge:.4f}")
    check("per-dataset oracle AUBC ~ 0.6902", abs(oracle - 0.6902) < 0.001, f"{oracle:.4f}")

    print("== collapse signal (sanity) ==")
    cp = {}
    for r in rows:
        cp.setdefault(r["method"], []).append(r["collapse_prob"])
    print("    per-method mean collapse_prob:",
          {m: round(float(np.mean(cp[m])), 3) for m in S.METHODS})

    print(f"\n=== AUDIT {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if audit() else 1)
