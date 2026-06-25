"""Calibration metrics for the skill's probabilistic outputs (within-eps / collapse
risk). Pure functions, no model state -- used by evaluate_lodo and the API tests.
"""
from __future__ import annotations

import numpy as np


def brier(p, y) -> float:
    """Brier score = mean squared error of probabilistic predictions (lower better)."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


def ece(p, y, n_bins: int = 10) -> float:
    """Expected calibration error over equal-width probability bins."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    if not len(p):
        return float("nan")
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def reliability_table(p, y, n_bins: int = 10):
    """(bin_center, predicted_mean, empirical_rate, count) rows."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            rows.append(((lo + hi) / 2, float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return rows
