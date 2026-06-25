"""Run profiles: smoke (gate-only) and pilot (real).

A profile bundles all the knobs the runner needs that are NOT specific to
``(policy, dataset, seed)``: training hyperparameters, the cumulative budget
plan, eval-time flags. ``run_one`` reads a profile + the (policy, dataset,
seed) tuple and builds a RunConfig.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from medal_bench.runner.al_loop import RunConfig, TrainConfig


PILOT_BUDGET_FRACS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
PILOT_SEEDS = [1000, 2000, 3000]


@dataclass
class ProfileConfig:
    name: str
    train: TrainConfig
    budget_fracs: list[float]
    pool_cap: int | None
    val_cap: int | None
    compute_surface_metrics_at_final: bool
    stratify_pool_by_fg: bool = False
    stratify_fg_ratio: float = 0.5
    # B3: use pool-size-dependent budget_grid instead of flat budget_fracs
    pool_dependent_budget: bool = False
    # frozen_v5: override the Case-B (500<=N<5000) budget grid; None -> default
    # flat [1,2,5,10,15,20]%. v5 uses a low-budget-weighted grid.
    budget_case_b: list[float] | None = None
    # truncate the budget plan to this many cumulative points (dry runs)
    n_budget_points: int | None = None
    # HD95/ASD rounds: "final" | "first_mid_final" | "none"
    surface_policy: str = "final"


SMOKE = ProfileConfig(
    name="smoke",
    train=TrainConfig(
        num_iters=15, batch_size=2, lr=1e-3,
        image_size=128,
        features_per_stage=(8, 16, 32),
        dropout_p=0.1,
    ),
    budget_fracs=[0.5, 1.0],
    pool_cap=32, val_cap=8,
    compute_surface_metrics_at_final=False,
)

PILOT = ProfileConfig(
    name="pilot",
    train=TrainConfig(
        num_iters=250, batch_size=8, lr=1e-3,
        image_size=256,
        features_per_stage=(32, 64, 128, 256, 320),
        dropout_p=0.1,
    ),
    budget_fracs=PILOT_BUDGET_FRACS,
    # pool_cap=5000: only MSD07 (~22500 train slices) is affected; ISIC2018
    # (2594), BUSI (624), CVC (612), PROMISE12 (~1100) all stay below the cap.
    # Without this, eager-loading the resized pool + numpy temporaries push
    # MSD07 cells to ~80GB RSS and trigger the A40 node's OOM killer (observed
    # 7/9 MSD07 stage-1 cells got SIGKILLed at ~38min wall, gpu_usage=NONE).
    pool_cap=5000, val_cap=None,
    compute_surface_metrics_at_final=True,
    # When pool_cap kicks in (i.e. MSD07 only), pre-scan masks and cap to
    # 50% fg-containing + 50% bg slices so the labeled set isn't starved of
    # pancreas (without this, the model collapses to all-bg → DSC_fg=0).
    stratify_pool_by_fg=True,
    stratify_fg_ratio=0.5,
)


# Stage 0c formal-profile DRY RUN: adaptive long-side 512, pool-dependent budgets
# truncated to 3 points (init -> 2 transitions), short-but-nontrivial training,
# HD95 at first/mid/final. pool_cap keeps the dry run bounded.
BENCH512_DRY = ProfileConfig(
    name="bench512_dry",
    train=TrainConfig(
        num_iters=120, batch_size=8, lr=1e-3,
        image_size=512, aspect_preserve=True,
        features_per_stage=(32, 64, 128, 256, 320),
        dropout_p=0.1,
    ),
    budget_fracs=PILOT_BUDGET_FRACS,   # unused when pool_dependent_budget=True
    pool_cap=600, val_cap=120,
    compute_surface_metrics_at_final=True,
    stratify_pool_by_fg=True, stratify_fg_ratio=0.5,
    pool_dependent_budget=True,
    n_budget_points=3,
    surface_policy="first_mid_final",
)

# Stage 2 FORMAL profile (frozen_v3): adaptive 512, FULL pool-dependent budget (all 6
# points), real training. num_iters=1000 (Stage 1.5 proved 250 is severely undertrained
# and its method rankings don't survive to 1000 iters). pool_cap=5000 keeps huge slice
# pools tractable; val_cap=500 bounds HD95 cost. HD95 at first/mid/final. (BTCV may need
# 2000 — run the BTCV-2000 check via --num-iters 2000 before adopting any exception.)
BENCH512 = ProfileConfig(
    name="bench512",
    train=TrainConfig(
        num_iters=1000, batch_size=12, lr=1e-3,
        image_size=512, aspect_preserve=True,
        features_per_stage=(32, 64, 128, 256, 320),
        dropout_p=0.1,
    ),
    budget_fracs=PILOT_BUDGET_FRACS,   # unused (pool_dependent_budget=True)
    pool_cap=5000, val_cap=500,
    compute_surface_metrics_at_final=True,
    stratify_pool_by_fg=True, stratify_fg_ratio=0.5,
    pool_dependent_budget=True,
    n_budget_points=None,              # FULL budget grid (all points)
    surface_policy="first_mid_final",
)

# Stage 2 FORMAL profile (frozen_v4). Differs from bench512 (frozen_v3) in two ways the v3
# probe + canary established: (1) ADAPTIVE train-to-plateau iters instead of fixed 1000 —
# the probe showed the fixed-iter regime differentially under-trains difficulty-based methods
# (isic Δ@2000: Random +0.015 < entropy +0.048 < BADGE +0.066 < PAAL +0.089), so fixed iters
# artifactually favours diversity/random; adaptive removes that bias per-dataset without tuning.
# (2) Surface metrics at FINAL round only (per-case HD95/ASSD ~17 min on 14-class). Prob saving
# is masks-always + fp16-probs-for-a-debug-subset, set at launch (the ~1 TB all-cell gate).
BENCH512_V4 = ProfileConfig(
    name="bench512_v4",
    train=TrainConfig(
        num_iters=1000, batch_size=12, lr=1e-3,        # num_iters ignored under adaptive
        image_size=512, aspect_preserve=True,
        features_per_stage=(32, 64, 128, 256, 320),
        dropout_p=0.1,
        adaptive_iters=True, min_iters=1000, max_iters=3000,
        plateau_window=100, plateau_patience=5,
        plateau_min_delta=0.005, plateau_rel_delta=0.0,   # primary = abs train-loss delta
    ),
    budget_fracs=PILOT_BUDGET_FRACS,
    pool_cap=5000, val_cap=500,
    compute_surface_metrics_at_final=True,
    stratify_pool_by_fg=True, stratify_fg_ratio=0.5,
    pool_dependent_budget=True,
    n_budget_points=None,
    surface_policy="final",
)

# frozen_v5 (2026-06-20 audit): identical to v4 EXCEPT
#   (1) max_iters 3000 -> 5000  : ~50% of v4 round-trainings hit the 3000 cap without
#       plateauing (hard multi-class still descending at loss 0.11-0.23); adaptive
#       stopping means well-behaved cells still stop early, so this only un-truncates
#       the under-trained cells. Subsumes the old per-dataset btcv/msd07=5000 hack.
#   (2) Case-B budget grid [1,2,5,10,15,20]% -> [1,2,4,7,10,15,20]% (7 pts): the v4
#       grid under-resolved the steep 2-10% region (every dataset re-ranked between the
#       5% and 10% points) and over-resolved the flat 10-20% region.
# GPU/TF32 confound is fixed by SINGLE-ARCH PINNING at launch (each dataset's full matrix
# on one arch; not a profile field). TF32 is kept ON in seeds.py (deterministic + ~2x on
# Ampere/Hopper, consistent within a single arch). Determinism = deterministic cuDNN + a
# deterministic CE (trainer.py reduction='none'.mean(); the fused nll_loss2d reduction was
# non-deterministic). Run all 3 seeds (1000/2000/3000) under THIS profile.
BENCH512_V5 = ProfileConfig(
    name="bench512_v5",
    train=TrainConfig(
        num_iters=1000, batch_size=12, lr=1e-3,
        image_size=512, aspect_preserve=True,
        features_per_stage=(32, 64, 128, 256, 320),
        dropout_p=0.1,
        adaptive_iters=True, min_iters=1000, max_iters=5000,
        plateau_window=100, plateau_patience=5,
        plateau_min_delta=0.005, plateau_rel_delta=0.0,
    ),
    budget_fracs=PILOT_BUDGET_FRACS,
    budget_case_b=[0.01, 0.02, 0.04, 0.07, 0.10, 0.15, 0.20],
    pool_cap=5000, val_cap=500,
    compute_surface_metrics_at_final=True,
    stratify_pool_by_fg=True, stratify_fg_ratio=0.5,
    pool_dependent_budget=True,
    n_budget_points=None,
    surface_policy="final",
)

PROFILES = {"smoke": SMOKE, "pilot": PILOT, "bench512_dry": BENCH512_DRY,
            "bench512": BENCH512, "bench512_v4": BENCH512_V4, "bench512_v5": BENCH512_V5}


def cumulative_budget_plan(pool_size: int, fracs: list[float]) -> list[int]:
    """{1,2,5,10,15,20}% (default) of pool_size, clamped, strictly increasing."""
    plan: list[int] = []
    last = -1
    for f in fracs:
        n = max(1, int(math.ceil(f * pool_size)))
        n = min(n, pool_size)
        n = max(n, last + 1)
        plan.append(n)
        last = n
    return plan


def _surface_rounds(policy: str, n: int) -> set | None:
    if policy == "first_mid_final":
        return {0, n // 2, n - 1}
    if policy == "final":            # frozen_v4: HD95/ASSD only at the final budget
        return {n - 1}               # (per-case surface is ~17 min on 14-class datasets)
    if policy == "none":
        return set()
    return None  # fall back to compute_surface_metrics_at_final


def build_run_config(
    *,
    profile_name: str,
    policy_id: str,
    policy_config: dict,
    dataset_name: str,
    pool_size: int,
    seed: int,
    out_jsonl: str,
    device: str = "cuda:0",
    foundation_features_fn=None,
    num_classes: int = 2,
    preproc_cache_dir: str | None = None,
) -> RunConfig:
    if profile_name not in PROFILES:
        raise ValueError(f"unknown profile {profile_name}; choose from {list(PROFILES)}")
    prof = PROFILES[profile_name]
    if prof.pool_dependent_budget:
        from medal_bench.profiles.budget import budget_grid
        plan = budget_grid(pool_size, num_classes,
                           case_b_fracs=prof.budget_case_b).cumulative_counts
        if prof.n_budget_points:
            plan = plan[: prof.n_budget_points]
    else:
        plan = cumulative_budget_plan(pool_size, prof.budget_fracs)
    return RunConfig(
        policy_id=policy_id, policy_config=policy_config,
        dataset_name=dataset_name, seed=seed,
        budget_plan=plan, train=prof.train,
        out_jsonl=out_jsonl, device=device,
        pool_cap=prof.pool_cap, val_cap=prof.val_cap,
        foundation_features_fn=foundation_features_fn,
        compute_surface_metrics_at_final=prof.compute_surface_metrics_at_final,
        surface_rounds=_surface_rounds(prof.surface_policy, len(plan)),
        preproc_cache_dir=preproc_cache_dir,
        stratify_pool_by_fg=prof.stratify_pool_by_fg,
        stratify_fg_ratio=prof.stratify_fg_ratio,
    )
