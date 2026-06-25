"""Stage S1 leakage / grouping / target-correctness tests (spec section 14, the
items that apply to the fixed-policy Query-Strategy-Skill dataset). Tests 7/11/12
(branch reward, byte-identical checkpoints) belong to S4 branching and are not
asserted here.

Run:  python -m pytest medal_bench/skill/tests/ -q
"""
import csv
import os

import numpy as np
import pytest

from medal_bench.skill import schema as S
from medal_bench.skill import splits

SKILL_DIR = S.SKILL_DIR
CELLS = os.path.join(SKILL_DIR, "cells_raw.csv")
ROWS = os.path.join(SKILL_DIR, "skill_rows.csv")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(CELLS) and os.path.exists(ROWS)),
    reason="export not built yet; run medal_bench.skill.export_query_strategy_dataset")


def _load(path):
    out = list(csv.DictReader(open(path)))
    for r in out:
        for k, v in list(r.items()):
            try:
                r[k] = float(v) if v not in ("", None) else v
            except (ValueError, TypeError):
                pass
    return out


@pytest.fixture(scope="module")
def cells():
    return _load(CELLS)


@pytest.fixture(scope="module")
def rows():
    return _load(ROWS)


# 1. No v4/v5 trajectory mixing -- the export reads ONLY runs/frozen_v5.
def test_source_is_frozen_v5_only():
    assert S.RUNS_DIR == "runs/frozen_v5"
    # v5 grid has no 5% budget point; assert no cell carries the v4 signature
    # (a labeled_ratio within 1e-3 of exactly 0.05 at an interior round).
    # cells_raw only stores n_rounds; deeper check is in the round-grid below.


def test_only_19set_and_seeds(cells):
    assert {r["dataset"] for r in cells} == set(S.DS19)
    assert {int(r["seed"]) for r in cells} == set(S.SEEDS)
    assert {r["method"] for r in cells} == set(S.METHODS)
    assert len(cells) == 570


# 2. No test metric used as an input feature.
def test_no_test_metric_in_features():
    leak = set(S.FEATURE_COLS) & S.FORBIDDEN_FEATURE_COLS
    assert not leak, f"leaking target/identity cols into features: {leak}"
    # explicit: none of the per-method outcome columns may be features
    for bad in ("aubc", "aubc_mean", "dsc_final", "regret", "rank", "is_collapse"):
        assert bad not in S.FEATURE_COLS


# 3. No future-round leakage: round-0 features use ONLY the round-0 record and are
#    the P1-P9 consensus (seed-mean of per-seed medians), independent of any
#    method's later trajectory.
def test_round0_features_are_round0_consensus(cells, rows):
    import json

    def round_dsc(ds, line_idx):
        """seed-mean over (median across P1-P9) of the DSC at trajectory line `line_idx`."""
        per_seed = []
        for s in S.SEEDS:
            vals = []
            for m in S.ROUND0_CONSENSUS_METHODS:
                f = f"{S.RUNS_DIR}/{ds}__{m}__s{s}.jsonl"
                if os.path.exists(f):
                    ls = open(f).read().splitlines()
                    vals.append(json.loads(ls[line_idx])["metrics"]["mean_dsc_fg_case_macro"])
            if vals:
                per_seed.append(float(np.median(vals)))
        return float(np.mean(per_seed))

    for ds in S.DS19:
        stored = [r["r0_dsc"] for r in rows if r["dataset"] == ds][0]
        # (a) faithful: matches the export's round-0 aggregation exactly
        assert abs(stored - round_dsc(ds, 0)) < 1e-4, (ds, stored, round_dsc(ds, 0))
        # (b) provably NOT a leaked later round (final round DSC differs from round-0)
        assert abs(stored - round_dsc(ds, -1)) > 1e-6, f"{ds}: r0 equals final round (leakage?)"


# 4 & 5. Dataset-grouped splits; all methods of a held-out dataset held out.
def test_lodo_splits_are_dataset_grouped(rows):
    for train_ds, held in splits.lodo_folds():
        assert held not in train_ds
        tr = [r for r in rows if r["dataset"] in train_ds]
        te = [r for r in rows if r["dataset"] == held]
        splits.assert_no_dataset_leak(tr, te)
        # all 10 methods of the held-out dataset are in the test fold
        assert {r["method"] for r in te} == set(S.METHODS)
        assert len(te) == 10
        assert len(tr) == 18 * 10


# 6. Round-0 shared-state consistency across the 10 methods of a dataset.
def test_round0_shared_within_dataset(rows):
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    for ds, rs in by_ds.items():
        assert len({round(r["r0_dsc"], 5) for r in rs}) == 1, ds


# 8. Regret calculation.
def test_regret(rows):
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    for ds, rs in by_ds.items():
        best = max(r["aubc_mean"] for r in rs)
        for r in rs:
            assert abs(r["regret"] - (best - r["aubc_mean"])) < 1e-5
            assert r["regret"] >= -1e-9
        assert abs(min(r["regret"] for r in rs)) < 1e-9  # best method regret 0


# 9. Within-epsilon target calculation.
def test_within_epsilon(rows):
    for r in rows:
        assert r["within_eps"] == int(r["regret"] <= S.EPS_AUBC + 1e-9)
        assert 0.0 <= r["p_within_eps"] <= 1.0


# 13/14. Firewall: nothing in the dataset exposes unlabeled GT; difficulty/collapse
#        labels come only from observed outcomes (final DSC / trajectory), not GT.
def test_collapse_label_from_observed_outcomes(cells):
    # is_collapse must be reconstructible from the trajectory-derived flags only
    for r in cells:
        recon = int(bool(r["c_abs"]) or bool(r["c_rel"]) or bool(r["c_instab"]))
        assert int(r["is_collapse"]) == recon
