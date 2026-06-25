"""Frozen benchmark configuration v3 (Stage 2 launch config).

Supersedes v2 (``profiles/frozen_v2.py``). v3 records the decisions locked after the
Stage 1.5 iteration-sensitivity study + the P0-P9 correctness review:

  - num_iters = 1000 (250 was severely undertrained; its method rankings did not
    survive more training). BTCV may get a 2000 exception pending the BTCV-2000 check.
  - PRIMARY metric = per-case macro foreground DSC (mean_dsc_fg_case_macro); per-case
    HD95 + symmetric ASSD with a DIAGONAL total-miss penalty + structure detection rate;
    old micro/pooled DSC + directed ASD kept as diagnostics only.
  - valid-region query aggregation for P1/P2/P5 (and a hard valid-region intersection in
    P6) so letterbox pad never drives selection or the metric.
  - budget denominator = the TRUE accessible AL pool (actual_AL_pool_N), logging both
    fraction_of_AL_pool and fraction_of_full_train.
  - component-level per-round seeding (model_init/loader/query/dropout), all logged.
  - P8 TypiClust made paper-faithful (MIN_CLUSTER_SIZE filter + round-robin + K-cap
    min(20,len//2)); the pre-v3 P8 is preserved as the deprecated ablation P8c.
  - always-on prediction saving (compressed val masks + ids + valid masks + fp16 probs);
    saved-prob storage is gated by the canary estimate before Wave 2.

Distances are in PIXELS/VOXELS at image_size (native voxel spacing not threaded yet —
deferred; saved preds carry a spacing slot for later mm backfill).
"""
from __future__ import annotations

import hashlib
import json

from medal_bench.profiles.frozen_v2 import FROZEN_V2_HASH as _V2_HASH

FROZEN_CONFIG_V3 = {
    "config_version": "3.0",
    "frozen_date": "2026-06-15",
    "supersedes": {
        "v2_hash": _V2_HASH,
        "reason": "num_iters=1000 + per-case metrics + diagonal total-miss + detection rate + "
                  "valid-region aggregation + actual_AL_pool_N budget denominator + component "
                  "per-round seeding + paper-faithful P8 (min-cluster+round-robin) + prediction saving",
    },
    "status": "FROZEN for Stage 2 (gated on: full pytest green, v3 canary pass, prob-storage "
              "estimate review, BTCV-2000 review).",
    "profile_name": "bench512",
    "train": {"num_iters": 1000, "train_batch_size": 12, "lr": 1e-3,
              "pool_cap": 5000, "val_cap": 500,
              "btcv_2000_check": "run btcv_synapse @ --num-iters 2000; adopt a documented "
                                 "family rule OR keep global 1000 with an underfit caveat — no "
                                 "silent one-off exception"},

    # --- core methods (code frozen; P8 selection now paper-faithful) ---
    "core_methods": {f"P{i}": n for i, n in enumerate([
        "Random", "Normalized Entropy", "BALD / MC-dropout", "CoreSet",
        "BADGE (canonical CE-only gradient embedding)",
        "Entropy -> CoreSet (Uncertainty-Filtered CoreSet)",
        "Selective Uncertainty AL", "Foundation-CoreSet",
        "Foundation-TypiClust (min-cluster + round-robin)", "PAAL",
    ])},
    "ablation_methods": {
        "P4b": "BADGE-Seg-CE-Dice",
        "P8b": "SAM-DensityClust",
        "P8c": "Foundation-TypiClust-legacy (pre-v3 single-pass; kept for v2 repro)",
    },
    "ablations_are_ablation_only": True,

    # --- foundation extractor (unchanged from v2) ---
    "foundation_feature_extractor": {
        "model": "SAM", "sam_model_type": "vit_h", "checkpoint": "sam_vit_h_4b8939.pth",
        "no_silent_fallback_to_vit_b": True,
        "cache_key_includes": ["encoder_id", "checkpoint", "preprocess_version",
                               "resolution_policy", "input_h", "input_w"],
    },

    # --- resolution (unchanged from v2: adaptive 512 letterbox) ---
    "resolution_policy": {
        "mode_default": "bench_res", "bench_res": 512,
        "preserve_aspect_ratio": True, "resize_by": "long_side_cap", "pad_to_multiple_of": 32,
        "mask_interpolation": "nearest", "image_interpolation": "bilinear",
        "valid_region": "letterbox pad excluded from P1/P2/P5 query aggregation, P6 region "
                        "intersection, and DSC/HD95/ASSD metrics (loss masking deferred)",
    },

    # --- metric policy (v3 per-case) ---
    "metric_policy": {
        "primary_metric": "mean_dsc_fg_case_macro",
        "metric_version": "v3_case_macro",
        "per_case": "group val slices by patient_id (native-2D image = its own case); macro over "
                    "cases and over GT-present foreground classes",
        "surface": "per-case HD95 + symmetric ASSD (medpy assd); units = pixels/voxels at image_size",
        "total_miss": "GT-present class predicted empty -> HD95/ASSD = case volume DIAGONAL (never "
                      "dropped) + counted in structure_detection_rate",
        "detection": ["structure_detection_rate", "missed_structure_rate"],
        "diagnostics_only": ["mean_dsc_fg_pooled_diagnostic", "dsc_per_class (pooled)",
                             "mean_asd_fg_directed"],
        "eval_scope": "logged per cell as case_full_volume (native-2D) | case_retained_slices "
                      "(3D-as-slice uses fg-positive retained slices — NOT the full native volume)",
        "spacing": "voxel spacing/affine not threaded yet; saved preds carry a spacing slot for mm backfill",
        "no_test_labels_for_query_or_model_selection": True,
    },

    # --- budget denominator (M2) ---
    "budget_policy": {
        "module": "medal_bench.profiles.budget.budget_grid",
        "denominator": "actual_AL_pool_N (the realized post-cap, fg-stratified pool — NOT min(len,cap))",
        "log": ["full_train_N", "requested_pool_cap", "actual_AL_pool_N",
                "fraction_of_AL_pool", "fraction_of_full_train", "budget_plan (absolute)"],
        "foreground_only_pool_caveat": "3D-as-slice pools are fg-positive RETAINED slices (the fg-stratify "
                                       "cap fills the fg half; ~no bg-only slices) — disclosed, acceptable",
    },

    # --- component seeding (M3) ---
    "seeding": {
        "per_round": "seed_all(seed + r) floor at the top of each round",
        "component_seeds": ["model_init_seed", "loader_seed", "query_seed", "dropout_seed"],
        "derived_from": "numpy.SeedSequence(seed + r)",
        "logged": "round_seed + component_seeds on every trajectory record",
    },

    # --- prediction saving (item 7) ---
    "prediction_saving": {
        "always": ["compressed val pred masks (uint8)", "gt", "sample/patient/slice ids",
                   "valid_bbox", "spacing slot"],
        "fp16_probs": "saved for the v3 canary to MEASURE size; all-cells saving is a HARD GATE on "
                      "the canary storage estimate before Wave 2 (else canary/debug/headline subsets only)",
        "purpose": "metric revisions never require retraining (recompute from saved preds/probs)",
    },

    # --- protocol (unchanged from v2 unless noted) ---
    "query_unit": "image/slice (2D); 3D volumes sliced to 2D",
    "seeds": [1000, 2000, 3000],
    "preprocessing_version": "v3-adaptive-letterbox-validbbox",
    "remap_policy_version": "v1-2026-06-13",
    "logging_schema_version": "v3",
    "determinism": "cuDNN deterministic by default; MEDAL_NONDETERMINISTIC=1 to opt out",
    "initial_labeled_set": "persisted per (dataset, seed, n_init), shared across methods",
}


def _hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()


FROZEN_V3_HASH = _hash(FROZEN_CONFIG_V3)


if __name__ == "__main__":
    print(json.dumps(FROZEN_CONFIG_V3, indent=2))
    print("FROZEN_V3_HASH =", FROZEN_V3_HASH)
    print("V2_HASH        =", _V2_HASH)
