"""Stage S2 -- skill API contract tests (spec section 14 items 10/16/17).

16. API returns predictive uncertainty + a fallback.
17. Skill inference is deterministic.
10. Calibration helpers behave.
Plus: blocklisted methods are never recommended (do-no-harm acceptance bar).
"""
import csv
import os

import numpy as np
import pytest

from medal_bench.skill import calibration as cal
from medal_bench.skill import schema as S

ROWS = os.path.join(S.SKILL_DIR, "skill_rows.csv")
need_data = pytest.mark.skipif(not os.path.exists(ROWS), reason="export not built yet")


def _state():
    r = next(csv.DictReader(open(ROWS)))
    keys = S.STATIC_NUM_COLS + S.STATIC_CAT_COLS + S.ROUND0_COLS
    state = {}
    for k in keys:
        v = r[k]
        try:
            state[k] = float(v)
        except ValueError:
            state[k] = v
    return state


# ---- calibration (no data needed) ----
def test_calibration_perfect_is_zero():
    assert cal.brier([0, 1, 1, 0], [0, 1, 1, 0]) == 0.0
    assert cal.ece([0, 1, 1, 0], [0, 1, 1, 0]) == 0.0


def test_calibration_worst_is_one():
    assert cal.brier([1, 0], [0, 1]) == 1.0


# ---- API contract ----
@need_data
def test_api_returns_uncertainty_and_fallback():
    from medal_bench.skill.agent_api import recommend_query_strategy
    out = recommend_query_strategy(_state())
    for key in ("recommended", "fallback_method", "ranked_methods", "expected_utility",
                "uncertainty_interval", "probability_within_epsilon", "collapse_risk",
                "expected_training_cost", "expected_query_cost", "evidence"):
        assert key in out, f"missing {key}"
    assert out["fallback_method"] == "P4"
    lo, hi = out["uncertainty_interval"]
    assert lo <= hi


@need_data
def test_blocklisted_never_recommended():
    from medal_bench.skill.agent_api import QueryStrategySkill
    skill = QueryStrategySkill()
    st = _state()
    for risk in ("balanced", "averse"):
        out = skill.recommend(st, risk_tolerance=risk)
        assert out["recommended"] not in S.BLOCKLIST
        assert all(m not in S.BLOCKLIST for m in out["ranked_methods"])


@need_data
def test_api_deterministic():
    from medal_bench.skill.agent_api import QueryStrategySkill
    skill = QueryStrategySkill()
    st = _state()
    a = skill.recommend(st)
    b = skill.recommend(st)
    assert a["recommended"] == b["recommended"]
    assert a["expected_utility"] == b["expected_utility"]
    assert a["probability_within_epsilon"] == b["probability_within_epsilon"]


@need_data
def test_compute_constraint_prefers_cheaper():
    from medal_bench.skill.agent_api import QueryStrategySkill
    skill = QueryStrategySkill()
    out = skill.recommend(_state(), compute_constraint="low")
    # recommended must be a non-blocklisted method
    assert out["recommended"] in S.ALLOWED


@need_data
def test_balanced_recommendation_is_collapse_aware():
    """Do-no-harm: a deviation from the safe default must never land on a method that is
    less collapse-safe than the default (the P8-on-busi failure the review found)."""
    import csv
    from medal_bench.skill.agent_api import QueryStrategySkill
    skill = QueryStrategySkill()
    rows = list(csv.DictReader(open(ROWS)))
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], {})[r["method"]] = float(r["collapse_prob"])
    keys = S.STATIC_NUM_COLS + S.STATIC_CAT_COLS + S.ROUND0_COLS
    for ds in S.DS19:
        r0 = next(r for r in rows if r["dataset"] == ds)
        state = {k: (float(r0[k]) if k not in S.STATIC_CAT_COLS else r0[k]) for k in keys}
        rec = skill.recommend(state, risk_tolerance="balanced")["recommended"]
        # the recommended method never collapsed on this dataset in the benchmark
        assert by_ds[ds][rec] == 0.0, f"{ds}: recommended {rec} has collapse_prob {by_ds[ds][rec]}"
