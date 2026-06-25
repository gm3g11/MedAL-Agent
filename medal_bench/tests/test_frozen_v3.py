"""frozen_v3 config: distinct hash, supersedes v2, num_iters=1000, registry
consistency (incl. P8c), and the v3 decision blocks are present."""
from __future__ import annotations

from medal_bench.profiles.frozen import FROZEN_CONFIG_HASH
from medal_bench.profiles.frozen_v2 import FROZEN_V2_HASH
from medal_bench.profiles.frozen_v3 import FROZEN_CONFIG_V3, FROZEN_V3_HASH
from medal_bench.policies import all_ids, build
from medal_bench.runner.trajectory import TRAJECTORY_SCHEMA_VERSION


def test_hash_distinct_and_supersedes_v2():
    assert FROZEN_V3_HASH not in (FROZEN_V2_HASH, FROZEN_CONFIG_HASH)
    assert len(FROZEN_V3_HASH) == 64
    assert FROZEN_CONFIG_V3["supersedes"]["v2_hash"] == FROZEN_V2_HASH


def test_num_iters_and_schema():
    assert FROZEN_CONFIG_V3["train"]["num_iters"] == 1000
    assert FROZEN_CONFIG_V3["logging_schema_version"] == TRAJECTORY_SCHEMA_VERSION == "v3"


def test_registry_consistency_includes_p8c():
    declared = set(FROZEN_CONFIG_V3["core_methods"]) | set(FROZEN_CONFIG_V3["ablation_methods"])
    assert declared == set(all_ids())
    assert "P8c" in FROZEN_CONFIG_V3["ablation_methods"]
    for pid in FROZEN_CONFIG_V3["ablation_methods"]:
        assert build(pid).is_ablation is True
    for pid in FROZEN_CONFIG_V3["core_methods"]:
        assert build(pid).is_ablation is False


def test_v3_decision_blocks_present():
    mp = FROZEN_CONFIG_V3["metric_policy"]
    assert mp["primary_metric"] == "mean_dsc_fg_case_macro"
    assert mp["metric_version"] == "v3_case_macro"
    assert "DIAGONAL" in mp["total_miss"]
    bp = FROZEN_CONFIG_V3["budget_policy"]
    assert "actual_AL_pool_N" in bp["denominator"]
    for f in ("fraction_of_AL_pool", "fraction_of_full_train"):
        assert f in bp["log"]
    assert FROZEN_CONFIG_V3["seeding"]["component_seeds"] == [
        "model_init_seed", "loader_seed", "query_seed", "dropout_seed"]
