"""Frozen benchmark configuration (task 4).

The single declarative source of truth for the 42-dataset launch. ``FROZEN_CONFIG_HASH``
pins it; any change to the dict changes the hash and must be a deliberate re-freeze.

Launch the benchmark consistent with this config, e.g.:
    python -m medal_bench.runner.run_one --policy P7 --dataset busi --seed 1000 \
        --profile pilot --foundation sam --sam-model-type vit_h
"""
from __future__ import annotations

import hashlib
import json

FROZEN_CONFIG = {
    "config_version": "1.0",
    "frozen_date": "2026-06-12",

    # --- core methods (the published P0-P9 baselines) ---
    "core_methods": {
        "P0": "Random",
        "P1": "Normalized Entropy",
        "P2": "BALD / MC-dropout",
        "P3": "CoreSet",
        "P4": "BADGE (canonical CE-only gradient embedding)",
        "P5": "Entropy -> CoreSet (Uncertainty-Filtered CoreSet)",
        "P6": "Selective Uncertainty AL",
        "P7": "Foundation-CoreSet",
        "P8": "Foundation-TypiClust",
        "P9": "PAAL",
    },

    # --- ablation methods (NOT part of the core comparison) ---
    "ablation_methods": {
        "P4b": "BADGE-Seg-CE-Dice (CE+Dice gradient)",
        "P8b": "SAM-DensityClust (simplified unlabeled-only TypiClust-lite)",
        "P8c": "Foundation-TypiClust-legacy (pre-v3 single-pass; kept for v2 repro)",
    },
    "ablations_are_ablation_only": True,

    # --- method definitions that were corrected this cycle ---
    "p1_entropy_formula": "H_norm = -sum_c p_c log p_c / log(C); mean over pixels; range [0,1]",
    "p4_main": "canonical CE-only analytic gradient embedding g_c=mean(p_c-1[yhat=c])z, dim C*D",
    "p6_method": "Selective Uncertainty AL (Ma et al., ICASSP 2024, arXiv:2401.16298); PEAL removed from core",
    "p8_main": "reference TypiClust: cluster labeled∪unlabeled into |L|+budget, uncovered-first, 1/mean-KNN typicality",

    # --- foundation feature extractor for P7/P8 ---
    "foundation_feature_extractor": {
        "model": "SAM",
        "sam_model_type": "vit_h",                       # RECOMMENDED final choice
        "checkpoint": "sam_vit_h_4b8939.pth",
        "feature_dim": 256,                              # SAM neck output (same for vit_b/l/h)
        "pooling": "global average pool over the 64x64 image-encoder grid",
        "grayscale_to_rgb": "replicate single channel to 3 (deterministic)",
        "fallback": "vit_b permitted ONLY if explicitly configured for compute; selections differ "
                    "materially from vit_h (pilot overlap 0.19-0.38) and never fall back silently",
    },

    # --- AL protocol ---
    "budget_curve_cumulative_fracs": [0.01, 0.02, 0.05, 0.10, 0.15, 0.20],
    "rounding": "ceil(frac * pool_size), clamped to [1, pool], strictly increasing",
    "query_unit": "image/slice (2D); 3D volumes sliced to 2D",
    "slice_id_format": "selected IDs are slice IDs encoding (dataset_name, case_id, slice_index); "
                       "case-level only if explicitly configured",
    "seeds": [1000, 2000, 3000],

    # --- preprocessing & reproducibility ---
    "preprocessing_version": "v1",
    "mask_interpolation": "nearest-neighbour",
    "image_interpolation": "bilinear",
    "logging_schema_version": "v2",
    "determinism": "cuDNN deterministic by default; MEDAL_NONDETERMINISTIC=1 to opt out",
    "initial_labeled_set": "persisted per (dataset, seed, n_init) and shared across all methods",
}


def _hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()


FROZEN_CONFIG_HASH = _hash(FROZEN_CONFIG)


if __name__ == "__main__":
    print(json.dumps(FROZEN_CONFIG, indent=2))
    print("FROZEN_CONFIG_HASH =", FROZEN_CONFIG_HASH)
