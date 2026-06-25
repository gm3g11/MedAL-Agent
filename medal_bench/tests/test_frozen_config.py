"""Frozen-config consistency (task 4): the frozen benchmark config must agree
with the live registry, ablation flags, feature dim, and logging schema."""
from __future__ import annotations

from medal_bench.profiles.frozen import FROZEN_CONFIG, FROZEN_CONFIG_HASH
from medal_bench.policies import all_ids, build
from medal_bench.runner.trajectory import TRAJECTORY_SCHEMA_VERSION


def test_frozen_core_and_ablations_match_registry():
    ids = set(all_ids())
    declared = set(FROZEN_CONFIG["core_methods"]) | set(FROZEN_CONFIG["ablation_methods"])
    assert declared == ids, declared ^ ids


def test_frozen_ablation_flags():
    for pid in FROZEN_CONFIG["ablation_methods"]:
        assert build(pid).is_ablation is True, pid
    for pid in FROZEN_CONFIG["core_methods"]:
        assert build(pid).is_ablation is False, pid


def test_frozen_schema_and_dim():
    # v1 freeze used logging schema "v2"; the live constant has since moved to "v3"
    # (frozen_v3). Pin the historical config to the schema it actually declared.
    assert FROZEN_CONFIG["logging_schema_version"] == "v2"
    assert FROZEN_CONFIG["foundation_feature_extractor"]["feature_dim"] == 256
    assert FROZEN_CONFIG["foundation_feature_extractor"]["sam_model_type"] in ("vit_b", "vit_l", "vit_h")
    assert len(FROZEN_CONFIG_HASH) == 64
    assert build("P6").name.lower().startswith("selective")
    assert "normalized" in build("P1").name.lower()
