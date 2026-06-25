"""Frozen benchmark configuration v2 (DRAFT — Stage -1 B8).

v1 (``profiles/frozen.py``) pinned forced-256 square resize + a flat 1-20% budget.
v2 records the Stage -1 decisions: adaptive resolution, pool-size-dependent budgets,
a remap version, an explicit metric/empty-mask policy, and the Stage 1 dataset list.

STATUS: DRAFT. Some referenced machinery is not yet implemented at freeze time —
adaptive resolution (B2), full-supervised baseline (B4), derived metrics (B5), and
the loader-cache (B6). The resolution is provisionally 512 and is NOT finally frozen
until the Stage 1 512/640/768 sensitivity + throughput results are in. Do not treat
FROZEN_V2_HASH as the launch hash until those land and this header is removed.
"""
from __future__ import annotations

import hashlib
import json

from medal_bench.profiles.frozen import FROZEN_CONFIG_HASH as _V1_HASH

FROZEN_CONFIG_V2 = {
    "config_version": "2.0",
    "frozen_date": "2026-06-13",
    "supersedes": {"v1_hash": _V1_HASH, "reason": "adaptive resolution + pool-dependent budgets"},
    "status": "FROZEN for Stage 1. Resolution default = 512 (user-confirmed); the Stage 1 "
              "512/640/768 sensitivity may revise resolution for Stage 2 only.",
    "profile_name": "bench512",
    "train": {"num_iters": 250, "train_batch_size": 12, "lr": 1e-3,
              "pool_cap": 5000, "val_cap": 500,
              "note": "train_batch_size is FIXED across all GPUs (comparability — it changes "
                      "the optimization); inference batches (eval/feature=16, SAM=8) are "
                      "quality-neutral and sized for GPU headroom"},

    # --- core methods (unchanged from v1; method CODE is frozen + 124 tests) ---
    "core_methods": {f"P{i}": n for i, n in enumerate([
        "Random", "Normalized Entropy", "BALD / MC-dropout", "CoreSet",
        "BADGE (canonical CE-only gradient embedding)",
        "Entropy -> CoreSet (Uncertainty-Filtered CoreSet)",
        "Selective Uncertainty AL", "Foundation-CoreSet", "Foundation-TypiClust", "PAAL",
    ])},
    "ablation_methods": {"P4b": "BADGE-Seg-CE-Dice", "P8b": "SAM-DensityClust"},
    "ablations_are_ablation_only": True,
    "policy_code_unchanged_from_v1": True,

    # --- foundation extractor ---
    "foundation_feature_extractor": {
        "model": "SAM", "sam_model_type": "vit_h", "checkpoint": "sam_vit_h_4b8939.pth",
        "no_silent_fallback_to_vit_b": True,
        "cache_key_includes": ["encoder_id", "preprocess_version", "resolution_policy"],
    },

    # --- adaptive resolution policy (B2; provisional) ---
    "resolution_policy": {
        "mode_default": "bench_res",
        "modes": {"smoke_res": 384, "bench_res": 512, "hires_res": 768, "max_res": 1024},
        "preserve_aspect_ratio": True,
        "resize_by": "long_side_cap",
        "pad_to_multiple_of": 32,
        "mask_interpolation": "nearest",
        "image_interpolation": "bilinear",
        "log_sizes": ["orig", "resized", "padded"],
        "sensitivity_study": "512 vs 640 vs 768 on {isic2018, msd07_pancreas, mmwhs|btcv} before Stage 2",
        "default_frozen_for_stage1": True,
    },

    # --- determinism (verified at 512; see seeds.py) ---
    "determinism": {
        "cudnn_deterministic": True, "cudnn_benchmark": False,
        "use_deterministic_algorithms": "True (warn_only)",
        "cublas_workspace_config": ":4096:8",
        "verified": "same seed -> identical selected_ids + ckpt_hash @512 (busi/P1)",
        "opt_out_env": "MEDAL_NONDETERMINISTIC=1",
    },

    # --- pool-size-dependent budget policy (B3; implemented) ---
    "budget_policy": {
        "module": "medal_bench.profiles.budget.budget_grid",
        "cases": {
            "A_lt_500": "absolute [5,10,20,40,80] truncated at min(120, floor(0.2N))",
            "B_500_5k": "[1,2,5,10,15,20]%",
            "C_5k_30k": "[0.25,0.5,1,2,5,10]%",
            "D_ge_30k": "[0.05,0.1,0.25,0.5,1,2]% (+5% opt-in)",
        },
        "initial_count": "max(8, 2*num_classes, ceil(first_frac*N)), optional cap 128/256",
        "log": ["N", "fractions", "cumulative_counts", "incremental_counts", "initial_count", "max_count"],
    },

    # --- remap policy (B1; implemented) ---
    "remap_version": "v1-2026-06-13",
    "remaps": {
        "mmwhs": "{0,205,420,421,500,550,600,820,850}->{0..7} (8 cls); 421->2",
        "btcv": "{0..12,16}->{0..13} (14 cls; 16->13, code-16 identity UNCONFIRMED — see btcv note)",
        "prepared_not_wired": ["myops {..2221->4} (5 cls)", "brats {0,1,2,4}->{0..3} (4 cls)"],
        "rule": "read masks as integer NIfTI; LabelRemapper hard-errors on unknown native codes",
    },

    # --- metric policy (B5; partly implemented) ---
    "metric_policy": {
        "dsc": "per-class + foreground-mean every budget",
        "hd95_asd": "first/middle/final budget in Stage 1; every budget for Stage 2 headline if feasible",
        "surface_dice_nsd": "not implemented",
        "empty_mask": {
            "both_empty": "DSC undefined(NaN); HD95/ASD = 0.0 (perfect absence agreement)",
            "one_empty_one_not": "DSC per accumulation; HD95/ASD = NaN, counted in hd95_undefined",
        },
        "derived": ["AUBC", "gain_over_random", "budget_to_90/95_full", "regret", "avg_rank", "win_rate"],
        "no_test_labels_for_query_or_model_selection": True,
    },

    # --- split / eval policy (B-splits; implemented) ---
    "split_policy": {
        "frac": {"val": 0.1, "test": 0.1},
        "case_disjoint_for_3d_slices": True,
        "official_images_only_test_excluded_from_dsc_hd95": True,
        "stage1_reports": "validation metrics (pilot); internal labeled test preferred for Stage 2",
    },

    # --- protocol ---
    "query_unit": "image/slice (2D); 3D volumes sliced to 2D",
    "slice_id_format": "(dataset_name, case_id, slice_index); case-disjoint splits",
    "seeds": [1000, 2000, 3000],
    "preprocessing_version": "v2-adaptive-letterbox-pad32",
    "remap_policy_version": "v1-2026-06-13",
    "loader_cache_version": "v1 (preproc-array npz + pool-selection json + sam-input-size key __in{sz})",
    "logging_schema_version": "v2",
    "determinism": "cuDNN deterministic by default; MEDAL_NONDETERMINISTIC=1 to opt out",
    "initial_labeled_set": "persisted per (dataset, seed, n_init), shared across methods",

    # --- Stage 1 dataset inclusion list ---
    "stage1_datasets": [
        "busi", "kvasir_seg", "isic2018", "glas2015", "origa",
        "promise12", "msd07_pancreas", "mmwhs_ct", "btcv_synapse",
    ],
}


def _hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()


FROZEN_V2_HASH = _hash(FROZEN_CONFIG_V2)


if __name__ == "__main__":
    print(json.dumps(FROZEN_CONFIG_V2, indent=2))
    print("FROZEN_V2_HASH =", FROZEN_V2_HASH)
    print("V1_HASH        =", _V1_HASH)
