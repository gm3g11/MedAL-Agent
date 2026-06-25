"""Pool-size-dependent budget policy (Stage -1 B3).

Replaces the flat 1/2/5/10/15/20% grid (which is meaningless for huge slice
pools, where 1% is already thousands of slices) with a grid that adapts to the
train-pool size N. Returns cumulative *counts* (what the AL loop consumes) plus
both fractions and absolute counts for logging.

  Case A  N < 500          absolute [5,10,20,40,80, min(120, floor(0.20 N))]
  Case B  500  <= N < 5k   fracs    [1, 2, 5, 10, 15, 20] %
  Case C  5k   <= N < 30k  fracs    [0.25, 0.5, 1, 2, 5, 10] %
  Case D  N >= 30k         fracs    [0.05, 0.1, 0.25, 0.5, 1, 2] %  (+5% opt-in)

Initial labeled count: max(8, 2*num_classes, first_budget_count), optionally
capped (documented in frozen v2). The cumulative plan is strictly increasing and
clamped to [1, N].
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_CASE_B = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
_CASE_C = [0.0025, 0.005, 0.01, 0.02, 0.05, 0.10]
_CASE_D = [0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02]
_CASE_A_ABS = [5, 10, 20, 40, 80]  # 6th entry = min(120, floor(0.20 N)), appended below


@dataclass
class BudgetGrid:
    N: int
    case: str
    cumulative_counts: list[int]   # what the AL loop iterates (strictly increasing)
    fractions: list[float]         # count / N, parallel to cumulative_counts
    initial_count: int             # == cumulative_counts[0]
    max_count: int                 # == cumulative_counts[-1]

    @property
    def n_al_rounds(self) -> int:
        """Rounds with an actual acquisition step (everything past the seed round)."""
        return len(self.cumulative_counts) - 1

    @property
    def is_degenerate(self) -> bool:
        """True when the pool is too small for active learning: the seed set already
        meets/exceeds the budget so NO acquisition ever runs (e.g. rose1: N=20 -> the
        min seed of 8 is 40% of the pool -> a single round, selected_ids always empty).
        Such cells carry no AL signal and should be excluded from method comparison."""
        return self.n_al_rounds < 1


def _raw_targets(N: int, *, case_d_add_5pct: bool,
                 case_b_fracs: list[float] | None = None) -> tuple[str, list[int]]:
    if N < 500:
        # absolute low-budget curve, truncated at the 20% ceiling so it stays
        # monotonic for small N (e.g. N=300 -> ceiling 60 < the 80 entry).
        cap = min(120, math.floor(0.20 * N))
        targets = [c for c in _CASE_A_ABS if c < cap] + [cap]
        return "A", targets
    if N < 5_000:
        # frozen_v5 may override the Case-B grid (low-budget-weighted) via the
        # profile; v4 and earlier pass None -> the original flat [1,2,5,10,15,20]%.
        fracs = case_b_fracs or _CASE_B
        case = "B"
    elif N < 30_000:
        fracs = _CASE_C
        case = "C"
    else:
        fracs = list(_CASE_D) + ([0.05] if case_d_add_5pct else [])
        case = "D"
    return case, [max(1, math.ceil(f * N)) for f in fracs]


def budget_grid(N: int, num_classes: int, *, cap_initial: int | None = None,
                case_d_add_5pct: bool = False,
                case_b_fracs: list[float] | None = None) -> BudgetGrid:
    if N < 1:
        raise ValueError(f"pool size N must be >= 1, got {N}")
    case, targets = _raw_targets(N, case_d_add_5pct=case_d_add_5pct,
                                 case_b_fracs=case_b_fracs)

    initial = max(8, 2 * num_classes, targets[0])
    if cap_initial is not None:
        initial = min(initial, cap_initial)
    initial = min(initial, N)

    # build a strictly-increasing cumulative plan, clamped to N
    plan: list[int] = [initial]
    for t in targets[1:]:
        t = min(max(t, plan[-1] + 1), N)
        if t > plan[-1]:
            plan.append(t)
        if plan[-1] >= N:
            break
    return BudgetGrid(
        N=N, case=case, cumulative_counts=plan,
        fractions=[c / N for c in plan],
        initial_count=plan[0], max_count=plan[-1],
    )
