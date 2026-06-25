"""Benchmark-level derived metrics (Stage -1 B5).

Pure post-hoc functions over per-method DSC-vs-budget curves (no model, no GPU).
A "curve" is a list of (cumulative_fraction, score) points sorted by fraction.

Implemented: AUBC, gain-over-random, budget-to-X%-full, regret-to-best,
average-rank, win-rate. `load_curves` builds curves from trajectory JSONLs.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict


def aubc(fractions: list[float], scores: list[float]) -> float:
    """Area under the (fraction, score) curve, normalized to the fraction span
    -> a budget-weighted mean score in the curve's score units."""
    if len(fractions) != len(scores) or len(fractions) < 2:
        raise ValueError("need >=2 parallel (fraction, score) points")
    pts = sorted(zip(fractions, scores))
    area = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        area += 0.5 * (y0 + y1) * (x1 - x0)
    span = pts[-1][0] - pts[0][0]
    return area / span if span > 0 else float(pts[0][1])


def gain_over_random(method: list[float], random_: list[float]) -> list[float]:
    """Per-budget score difference method - random (parallel, equal length)."""
    if len(method) != len(random_):
        raise ValueError("curves must be aligned to the same budgets")
    return [m - r for m, r in zip(method, random_)]


def budget_to_target_full(fractions: list[float], scores: list[float],
                          dsc_full: float, target: float = 0.90) -> float | None:
    """Smallest cumulative fraction at which the curve reaches target*dsc_full
    (linear interpolation between budgets). None if never reached."""
    thresh = target * dsc_full
    pts = sorted(zip(fractions, scores))
    if pts[0][1] >= thresh:
        return pts[0][0]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y1 >= thresh:
            if y1 == y0:
                return x1
            return x0 + (thresh - y0) * (x1 - x0) / (y1 - y0)
    return None


def regret_to_best(scores_by_method: dict[str, float]) -> dict[str, float]:
    """At one budget: best_score - method_score for each method (>=0)."""
    best = max(scores_by_method.values())
    return {m: best - s for m, s in scores_by_method.items()}


def average_rank(scores_by_method_per_budget: list[dict[str, float]]) -> dict[str, float]:
    """Mean rank (1=best) of each method across budgets. Ties share the mean rank."""
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for budget in scores_by_method_per_budget:
        # rank: highest score -> rank 1; average ranks within ties
        order = sorted(budget.items(), key=lambda kv: -kv[1])
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and order[j + 1][1] == order[i][1]:
                j += 1
            avg_rank = (i + 1 + j + 1) / 2  # 1-indexed mean of tied positions
            for k in range(i, j + 1):
                sums[order[k][0]] += avg_rank
                counts[order[k][0]] += 1
            i = j + 1
    return {m: sums[m] / counts[m] for m in sums}


def win_rate(scores_by_method_per_cell: list[dict[str, float]]) -> dict[str, float]:
    """Fraction of cells (dataset x budget) where a method has the top score.
    Ties split the win equally among tied methods."""
    wins: dict[str, float] = defaultdict(float)
    methods: set[str] = set()
    for cell in scores_by_method_per_cell:
        methods.update(cell)
        best = max(cell.values())
        winners = [m for m, s in cell.items() if s == best]
        for m in winners:
            wins[m] += 1.0 / len(winners)
    n = len(scores_by_method_per_cell)
    return {m: wins[m] / n for m in methods}


def load_curves(jsonl_dir: str, score_key: str = "mean_dsc_fg") -> dict[tuple, list[tuple[float, float]]]:
    """Build {(dataset, policy, seed): [(cumulative_fraction, score), ...]} from
    trajectory JSONLs written by the AL runner."""
    curves: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(jsonl_dir, "*.jsonl"))):
        for line in open(path):
            r = json.loads(line)
            metrics = r.get("metrics") or {}
            score = metrics.get(score_key)
            if score is None:
                continue
            n = r.get("labeled_count")
            cum = r.get("cumulative_budget")
            ratio = r.get("labeled_ratio")
            frac = ratio if ratio is not None else (n / cum if cum else 0.0)
            key = (r["dataset"], r["policy_id"], r["seed"])
            curves[key].append((float(frac), float(score)))
    return {k: sorted(v) for k, v in curves.items()}
