"""Derived benchmark-metric tests (Stage -1 B5)."""
from __future__ import annotations

import math

import pytest

from medal_bench.analysis.derived import (
    aubc, gain_over_random, budget_to_target_full,
    regret_to_best, average_rank, win_rate,
)


def test_aubc_formula():
    # constant 0.8 over [0,1] -> normalized area 0.8
    assert math.isclose(aubc([0.0, 0.5, 1.0], [0.8, 0.8, 0.8]), 0.8)
    # straight line 0->1 over [0,1] -> mean 0.5
    assert math.isclose(aubc([0.0, 1.0], [0.0, 1.0]), 0.5)


def test_aubc_needs_two_points():
    with pytest.raises(ValueError):
        aubc([0.1], [0.5])


def test_gain_over_random():
    assert gain_over_random([0.6, 0.7], [0.5, 0.5]) == [pytest.approx(0.1), pytest.approx(0.2)]
    with pytest.raises(ValueError):
        gain_over_random([0.6], [0.5, 0.5])


def test_budget_to_90_full():
    # full=0.9, target 90% -> 0.81; curve crosses 0.81 between 0.1(0.7) and 0.2(0.85)
    b = budget_to_target_full([0.05, 0.1, 0.2], [0.5, 0.7, 0.85], dsc_full=0.9, target=0.9)
    # linear interp: 0.1 + (0.81-0.7)*(0.2-0.1)/(0.85-0.7)
    assert math.isclose(b, 0.1 + 0.11 * 0.1 / 0.15)


def test_budget_to_full_never_reached():
    assert budget_to_target_full([0.1, 0.2], [0.3, 0.4], dsc_full=0.9, target=0.9) is None


def test_regret_to_best_fixed_method():
    r = regret_to_best({"P0": 0.6, "P1": 0.8, "P4": 0.75})
    assert r["P1"] == 0.0 and math.isclose(r["P0"], 0.2) and math.isclose(r["P4"], 0.05)


def test_average_rank():
    # budget 1: P1>P0; budget 2: P0>P1  -> both average rank 1.5
    ranks = average_rank([{"P0": 0.5, "P1": 0.7}, {"P0": 0.8, "P1": 0.6}])
    assert ranks["P0"] == 1.5 and ranks["P1"] == 1.5


def test_average_rank_ties():
    ranks = average_rank([{"P0": 0.5, "P1": 0.5}])  # tie -> both rank 1.5
    assert ranks["P0"] == 1.5 and ranks["P1"] == 1.5


def test_win_rate():
    cells = [{"P0": 0.5, "P1": 0.7}, {"P0": 0.9, "P1": 0.6}, {"P0": 0.5, "P1": 0.5}]
    wr = win_rate(cells)
    # P1 wins cell1, P0 wins cell2, tie cell3 (0.5 each) -> P0=1.5/3, P1=1.5/3
    assert math.isclose(wr["P0"], 1.5 / 3) and math.isclose(wr["P1"], 1.5 / 3)
