"""frozen_v3 valid-region aggregation: aggregate('valid') over a bbox, the no-pad
fall-back, the BADGE k<=0 guard, and P6's hard valid-region intersection."""
from __future__ import annotations

import numpy as np
import torch

from medal_bench.policies._helpers import aggregate, _apply_full, kmeanspp_indices


def test_aggregate_valid_ignores_pad_region():
    # high scores in the bottom (pad) rows, low in the valid top-left -> 'valid' mean
    # must be LOW (ignores pad) and below the full-canvas mean.
    sm = torch.zeros((1, 8, 8))
    sm[0, :4, :4] = 0.1          # valid region
    sm[0, 4:, :] = 0.9           # pad region (excluded)
    bbox = np.array([[0, 0, 4, 4]])
    v = aggregate(sm, None, "valid", valid_bboxes=bbox)
    full = aggregate(sm, None, "full")
    assert abs(float(v[0]) - 0.1) < 1e-6
    assert float(v[0]) < float(full[0])


def test_aggregate_valid_none_is_full():
    sm = torch.rand((3, 8, 8))
    v = aggregate(sm, None, "valid", valid_bboxes=None)
    assert torch.allclose(v, _apply_full(sm))


def test_kmeanspp_k_le_zero_returns_empty():
    X = np.random.RandomState(0).randn(5, 4)
    assert kmeanspp_indices(X, 0) == []
    assert kmeanspp_indices(X, -1) == []
    assert len(kmeanspp_indices(X, 3)) == 3      # normal path unaffected


def test_p6_intersects_valid_region():
    from medal_bench.policies.p6_selective_uncertainty import _intersect_valid_
    mask = torch.ones((1, 8, 8))
    _intersect_valid_(mask, np.array([[0, 0, 4, 4]]))
    assert float(mask[0, :4, :4].sum()) == 16.0   # valid region kept
    assert float(mask[0, 4:, :].sum()) == 0.0     # pad zeroed
    assert float(mask[0, :, 4:].sum()) == 0.0
