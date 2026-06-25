"""Group-aware evaluation splits. The independent generalization unit is the
DATASET -- never split rows. All seeds/methods/rounds of a held-out dataset stay
held out together.
"""
from __future__ import annotations

from medal_bench.skill import schema as S


def lodo_folds(datasets=None):
    """Yield (train_datasets, held_out_dataset) -- one fold per dataset."""
    datasets = list(datasets or S.DS19)
    for held in datasets:
        yield [d for d in datasets if d != held], held


def loo_group_folds(rows, key):
    """Leave-one-<key>-out over a categorical (e.g. modality / object_family).
    Yields (train_groups, held_group)."""
    groups = sorted({r[key] for r in rows})
    for held in groups:
        yield [g for g in groups if g != held], held


def out_of_seed_folds(seeds=None):
    """Yield (train_seeds, held_out_seed) for nested out-of-seed evaluation."""
    seeds = list(seeds or S.SEEDS)
    for held in seeds:
        yield [s for s in seeds if s != held], held


def assert_no_dataset_leak(train_rows, test_rows):
    """Raise if any dataset appears in both train and test."""
    tr = {r["dataset"] for r in train_rows}
    te = {r["dataset"] for r in test_rows}
    overlap = tr & te
    if overlap:
        raise AssertionError(f"dataset leak across split: {overlap}")
