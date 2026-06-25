"""Pool-size-dependent budget policy tests (Stage -1 B3)."""
from __future__ import annotations

import math

from medal_bench.profiles.budget import budget_grid


def test_budget_grid_case_A():
    # N<500: absolute counts, last = min(120, floor(0.2N))
    g = budget_grid(N=300, num_classes=2)
    assert g.case == "A"
    assert g.max_count == min(120, math.floor(0.2 * 300))  # 60
    assert all(b < 300 for b in g.cumulative_counts)


def test_budget_grid_case_B():
    g = budget_grid(N=2000, num_classes=2)
    assert g.case == "B"
    # 1% of 2000 = 20, but initial floor max(8,4,20)=20
    assert g.initial_count == 20
    assert g.max_count == 400  # 20%


def test_budget_grid_case_C():
    g = budget_grid(N=10_000, num_classes=3)
    assert g.case == "C"
    assert g.initial_count == math.ceil(0.0025 * 10_000)  # 25
    assert g.max_count == 1000  # 10%


def test_budget_grid_case_D():
    g = budget_grid(N=50_000, num_classes=2)
    assert g.case == "D"
    assert g.cumulative_counts[0] == max(8, 4, math.ceil(0.0005 * 50_000))  # 25
    assert g.max_count == math.ceil(0.02 * 50_000)  # 1000
    g5 = budget_grid(N=50_000, num_classes=2, case_d_add_5pct=True)
    assert g5.max_count == math.ceil(0.05 * 50_000)  # 2500


def test_initial_count_floor():
    # tiny pool, many classes -> floor 2*num_classes dominates
    g = budget_grid(N=400, num_classes=8)
    assert g.initial_count == max(8, 16, 5)  # 16
    assert g.initial_count == 16


def test_initial_count_cap():
    g = budget_grid(N=100_000, num_classes=2, cap_initial=128)
    assert g.initial_count <= 128


def test_incremental_counts_cumulative_strictly_increasing():
    for N in (300, 2000, 10_000, 50_000):
        g = budget_grid(N=N, num_classes=4)
        c = g.cumulative_counts
        assert all(c[i] < c[i + 1] for i in range(len(c) - 1)), c
        assert all(b <= N for b in c)


def test_budget_counts_and_fractions_logged():
    g = budget_grid(N=2000, num_classes=2)
    assert len(g.fractions) == len(g.cumulative_counts)
    assert abs(g.fractions[-1] - g.max_count / 2000) < 1e-9
