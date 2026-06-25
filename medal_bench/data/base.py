"""MedALDataset interface + sample metadata.

Each pilot dataset has an adapter under ``medal_bench/data/adapters/<name>.py``
that returns a MedALDataset for its train/val/test splits.

Key invariants:
- ``__getitem__(i)`` returns a ``Sample`` with image, mask, sample_id, meta.
- ``sample_id`` is stable across runs and seeds (it becomes the JSONL key).
- ``patient_id`` is set for 3D-source datasets used as 2D slices, so the
  runner can enforce patient-level splits.
- ``label`` is only present on labeled samples; the pool returns ``None``
  for label until the runner reveals it after a selection round.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Sample:
    sample_id: str                   # globally unique within (dataset, split)
    image: np.ndarray                # (H, W) or (C, H, W); float32; range = adapter contract
    mask: Optional[np.ndarray]       # (H, W) int (class indices); None if not revealed
    meta: dict = field(default_factory=dict)
    # 3D-source-used-as-2D-slice metadata; required for patient-level splits
    patient_id: Optional[str] = None
    slice_index: Optional[int] = None


class MedALDataset(abc.ABC):
    """A read-only-during-AL dataset.

    The runner builds three of these per dataset (train_pool, val, test).
    Labels in train_pool are only revealed after the policy queries them.
    """
    name: str                        # adapter-set short name, e.g. "isic2018"
    modality: str                    # "dermoscopy" | "endoscopy" | "ultrasound" | ...
    target: str                      # "skin_lesion" | "polyp" | ...
    dim: str                         # "2d" | "3d"
    query_unit: str                  # "image" | "slice" | "volume"
    num_classes: int                 # incl. background = 0

    @abc.abstractmethod
    def __len__(self) -> int: ...
    @abc.abstractmethod
    def __getitem__(self, i: int) -> Sample: ...
    @abc.abstractmethod
    def sample_ids(self) -> list[str]: ...

    # patient-level grouping (returns None for native-2D datasets)
    def patient_ids(self) -> Optional[list[str]]:
        return None
