"""Unit tests for the label-remap infrastructure (Stage -1 B1)."""
from __future__ import annotations

import numpy as np
import pytest

from medal_bench.data.remap import (
    LabelRemapper, MMWHS_REMAP, BTCV_REMAP, MYOPS_REMAP, BRATS_REMAP,
)


def test_remap_dense_labels_mmwhs():
    r = LabelRemapper(MMWHS_REMAP, "mmwhs")
    native = np.array([[0, 205, 420], [421, 500, 550], [600, 820, 850]], dtype=np.int64)
    out = r.apply(native)
    assert set(np.unique(out).tolist()) <= set(range(8))
    # 420 and 421 must both collapse to class 2
    assert out[0, 2] == 2 and out[1, 0] == 2


def test_unknown_native_label_raises():
    r = LabelRemapper(MMWHS_REMAP, "mmwhs")
    with pytest.raises(ValueError, match="unknown native label|outside the remap"):
        r.apply(np.array([[0, 999]], dtype=np.int64))


def test_background_zero_preserved():
    r = LabelRemapper(MMWHS_REMAP, "mmwhs")
    assert r.apply(np.zeros((4, 4), dtype=np.int64)).sum() == 0


def test_remap_requires_background_mapping():
    with pytest.raises(ValueError, match="preserve background"):
        LabelRemapper({1: 0, 2: 1}, "bad")


def test_remap_rounds_float_masks():
    # BTCV stores float32 NIfTI; remapper must round to nearest int before lookup.
    r = LabelRemapper(BTCV_REMAP, "btcv")
    out = r.apply(np.array([[0.0, 16.0], [12.0, 1.0]], dtype=np.float32))
    assert out[0, 1] == 13 and out[1, 0] == 12


def test_btcv_dense_remap_after_semantic_confirmation():
    r = LabelRemapper(BTCV_REMAP, "btcv")
    native = np.array([list(range(13)) + [16]], dtype=np.int64)
    out = r.apply(native)
    assert set(np.unique(out).tolist()) == set(range(14))  # dense 0..13
    assert out[0, -1] == 13  # 16 -> 13


def test_remap_num_classes():
    assert LabelRemapper(MMWHS_REMAP, "mmwhs").num_classes == 8
    assert LabelRemapper(BTCV_REMAP, "btcv").num_classes == 14
    assert LabelRemapper(MYOPS_REMAP, "myops").num_classes == 5
    assert LabelRemapper(BRATS_REMAP, "brats").num_classes == 4


def test_native_high_value_labels_preserved_before_remap():
    """A mask read as true int (850) remaps fine; the SAME value truncated to
    uint8 (850 % 256 == 82) is NOT a known code -> must raise. Documents why
    masks must be read as integer NIfTI, never via an 8-bit decoder."""
    r = LabelRemapper(MMWHS_REMAP, "mmwhs")
    assert r.apply(np.array([[850]], dtype=np.int64))[0, 0] == 7
    truncated = np.array([[850]], dtype=np.int64).astype(np.uint8).astype(np.int64)  # 82
    assert truncated[0, 0] == 82
    with pytest.raises(ValueError):
        r.apply(truncated)
