"""Patient/volume-level train/val/test splits for v1 datasets.

For 2D-native datasets (no patient_id), split by sample_id.
For 3D-source-as-2D-slice datasets (patient_id set), group slices by
patient_id and split GROUPS disjointly into train / val / test so no
patient's slices appear in two splits (constraint #3).

Returns a SplitView: a thin proxy that mimics the adapter but only exposes
the indices belonging to the split. The wrapped adapter does the actual
file/volume loading.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from medal_bench.data.base import MedALDataset, Sample


@dataclass
class SplitIndices:
    train: list[int]
    val: list[int]
    test: list[int]


class SplitView(MedALDataset):
    """View over a subset of indices in an underlying adapter."""

    def __init__(self, base: MedALDataset, indices: list[int], tag: str):
        self._base = base
        self._idx = list(indices)
        self._tag = tag
        # mirror class-level metadata so policies see the right num_classes
        self.name = f"{base.name}_{tag}"
        self.modality = base.modality
        self.target = base.target
        self.dim = base.dim
        self.query_unit = base.query_unit
        self.num_classes = base.num_classes

    def __len__(self) -> int:
        return len(self._idx)

    def sample_ids(self) -> list[str]:
        return [self._base.sample_ids()[i] for i in self._idx]

    def __getitem__(self, i: int) -> Sample:
        return self._base[self._idx[i]]

    def patient_ids(self) -> Optional[list[str]]:
        pids = self._base.patient_ids()
        if pids is None:
            return None
        return [pids[i] for i in self._idx]


def make_split(
    adapter: MedALDataset,
    seed: int,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> SplitIndices:
    """Build patient-grouped split if adapter.patient_ids() is non-None, else
    sample-id-level random split. Deterministic given (seed, adapter)."""
    rng = np.random.RandomState(seed)
    n = len(adapter)
    pids = adapter.patient_ids()

    if pids is None:
        order = np.arange(n)
        rng.shuffle(order)
        n_test = max(1, int(round(test_frac * n)))
        n_val = max(1, int(round(val_frac * n)))
        test_idx = order[:n_test].tolist()
        val_idx = order[n_test : n_test + n_val].tolist()
        train_idx = order[n_test + n_val :].tolist()
        return SplitIndices(train=train_idx, val=val_idx, test=test_idx)

    # patient-grouped
    unique = sorted(set(pids))
    p_order = np.array(unique)
    rng.shuffle(p_order)
    n_p = len(p_order)
    n_p_test = max(1, int(round(test_frac * n_p)))
    n_p_val = max(1, int(round(val_frac * n_p)))
    p_test = set(p_order[:n_p_test].tolist())
    p_val = set(p_order[n_p_test : n_p_test + n_p_val].tolist())
    p_train = set(p_order[n_p_test + n_p_val :].tolist())
    train_idx, val_idx, test_idx = [], [], []
    for i, pid in enumerate(pids):
        if pid in p_train:
            train_idx.append(i)
        elif pid in p_val:
            val_idx.append(i)
        else:
            test_idx.append(i)
    return SplitIndices(train=train_idx, val=val_idx, test=test_idx)
