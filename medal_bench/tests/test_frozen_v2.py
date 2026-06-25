"""Frozen config v2 draft tests (Stage -1 B8)."""
from __future__ import annotations

from medal_bench.profiles.frozen import FROZEN_CONFIG_HASH
from medal_bench.profiles.frozen_v2 import FROZEN_CONFIG_V2, FROZEN_V2_HASH


def test_frozen_v2_hash_changes_from_v1():
    assert FROZEN_V2_HASH != FROZEN_CONFIG_HASH


def test_frozen_v2_contains_resolution_policy():
    rp = FROZEN_CONFIG_V2["resolution_policy"]
    assert rp["modes"]["bench_res"] == 512 and rp["pad_to_multiple_of"] == 32
    assert rp["preserve_aspect_ratio"] is True


def test_frozen_v2_contains_budget_policy():
    bp = FROZEN_CONFIG_V2["budget_policy"]
    assert set(bp["cases"]) == {"A_lt_500", "B_500_5k", "C_5k_30k", "D_ge_30k"}


def test_frozen_v2_contains_remap_version():
    assert FROZEN_CONFIG_V2["remap_version"].startswith("v1-")
    assert "mmwhs" in FROZEN_CONFIG_V2["remaps"]


def test_frozen_v2_contains_metric_policy():
    mp = FROZEN_CONFIG_V2["metric_policy"]
    assert "empty_mask" in mp and "both_empty" in mp["empty_mask"]
    assert mp["no_test_labels_for_query_or_model_selection"] is True


def test_frozen_v2_ablations_remain_ablation_only():
    assert FROZEN_CONFIG_V2["ablations_are_ablation_only"] is True
    assert set(FROZEN_CONFIG_V2["ablation_methods"]) == {"P4b", "P8b"}
