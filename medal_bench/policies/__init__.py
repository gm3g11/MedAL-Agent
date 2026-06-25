"""Policy registry entry point.

Importing this module triggers registration of all v1 policies (P0..P9).

Core methods P0–P9 (P6 = Selective Uncertainty AL since 2026-06-12; PEAL removed
from the core list). Ablation methods carry distinct ids:
  P4b = BADGE-Seg-CE-Dice  (CE+Dice gradient; main P4 is canonical CE-only BADGE)
  P8b = SAM-DensityClust    (simplified unlabeled-only density; main P8 is
                             reference-faithful Foundation-TypiClust)
  P8c = Foundation-TypiClust-legacy (pre-v3 single-pass + global fallback, no
                             min-cluster filter; deprecated, kept for v2 repro)
"""
from medal_bench.policies.base import Policy, PolicyContext
from medal_bench.policies.registry import register, get, build, all_ids

from medal_bench.policies import (
    p0_random,
    p1_entropy_full,
    p2_bald,
    p3_coreset,
    p4_badge,
    p4b_badge_ce_dice,
    p5_entropy_coreset,
    p6_selective_uncertainty,
    p7_sam_coreset,
    p8_sam_typiclust,
    p8b_density_clust,
    p8c_typiclust_legacy,
    p9_paal,
)

__all__ = ["Policy", "PolicyContext", "register", "get", "build", "all_ids"]
