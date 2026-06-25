"""Method-registry cleanup tests (Section 2/6): P6 is Selective Uncertainty,
PEAL is gone from the core registry, and core/ablation ids are distinct."""
from __future__ import annotations

from medal_bench.policies import all_ids, build


CORE = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]


def test_p6_registry_maps_to_selective_uncertainty():
    pol = build("P6")
    assert type(pol).__name__ == "SelectiveUncertaintyAL"
    assert "selective uncertainty" in pol.name.lower()


def test_no_peal_in_core_registry():
    for pid in all_ids():
        assert "peal" not in build(pid).name.lower(), f"{pid} still named PEAL"


def test_p1_named_normalized_entropy():
    assert "normalized entropy" in build("P1").name.lower()


def test_core_and_ablation_ids_present_and_distinct():
    ids = set(all_ids())
    assert set(CORE).issubset(ids)
    assert {"P4b", "P8b"}.issubset(ids)
    # ablations are not part of the core list
    assert "P4b" not in CORE and "P8b" not in CORE
    # canonical P4 != ablation P4b, main P8 != lite P8b
    assert build("P4").name != build("P4b").name
    assert build("P8").name != build("P8b").name
