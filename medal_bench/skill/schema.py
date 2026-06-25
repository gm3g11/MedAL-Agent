"""Shared constants + the Query-Strategy-Skill dataset contract (Stage S1).

This module is the single source of truth for: the 19-set, the methods, the
method descriptor table, the blocklist, the collapse definition thresholds, and
the column groups of the exported skill dataset. Everything downstream
(export / audit / evaluate_lodo / agent_api / tests) imports from here so the
contract is defined once.

Scientific guardrails encoded here (see reports/query_skill_data_card.md):
  * The independent generalization unit is the DATASET, not a row.
  * Features are decision-time only: static dataset descriptors + the SHARED
    round-0 state (consensus over P1-P9, never P0 -- see ROUND0_CONSENSUS_METHODS).
  * Targets are derived from the post-acquisition trajectory; the PRIMARY target
    is regret (offset-invariant to the shared round-0 level).
"""
from __future__ import annotations

# ----------------------------------------------------------------------------
# The frozen_v5 19-dataset primary benchmark (exclude the in-progress 42-set).
# ----------------------------------------------------------------------------
DS19 = [
    "btcv_synapse", "flare22", "mmwhs_ct", "hvsmr2016", "ext_abdoment1k",
    "ext_brats2020", "msd_task07_pancreas", "isic2018", "care_leftatrium_2026",
    "kits19", "msd_task03_liver", "msd_task04_hippocampus", "refuge", "glas2015",
    "msd_task09_spleen", "origa", "kvasir_seg", "liqa_mri", "busi",
]

METHODS = [f"P{i}" for i in range(10)]
SEEDS = [1000, 2000, 3000]

METHOD_NAME = {
    "P0": "Random", "P1": "Entropy", "P2": "BALD", "P3": "CoreSet", "P4": "BADGE",
    "P5": "Ent+Core", "P6": "SelUnc", "P7": "SAM-Core", "P8": "TypiClust", "P9": "PAAL",
}

# Primary method family (STATE.md S2 phase grouping; primary assignment where a
# method appears in two phases). Plus capability flags used as method features.
#   family in {baseline, coverage, boundary, refinement}
METHOD_DESC = {
    #        family        unc   div    hyb    found  pred   stoch
    "P0": ("baseline",   False, False, False, False, False, True),
    "P1": ("boundary",   True,  False, False, False, False, False),
    "P2": ("boundary",   True,  False, False, False, False, True),   # MC-dropout
    "P3": ("coverage",   False, True,  False, False, False, False),
    "P4": ("boundary",   True,  True,  True,  False, False, True),   # BADGE (grad k-means++)
    "P5": ("boundary",   True,  True,  True,  False, False, False),  # Entropy->CoreSet
    "P6": ("refinement", True,  False, False, False, False, False),  # Selective Uncertainty
    "P7": ("coverage",   False, True,  False, True,  False, False),  # SAM-CoreSet
    "P8": ("coverage",   False, True,  False, True,  False, False),  # SAM-TypiClust
    "P9": ("refinement", False, True,  True,  False, True,  True),   # PAAL (acc-predictor)
}
METHOD_FLAG_COLS = ["m_unc", "m_div", "m_hyb", "m_found", "m_pred", "m_stoch"]

# The durable, statistically-robust signal (paired Wilcoxon+Holm p<=5e-4 for
# P6/P9; P1/P3 weakly-but-consistently worse than Random). A "do-no-harm" skill
# never selects these. Defined from observed 3-seed outcomes, NOT a model.
BLOCKLIST = ["P1", "P3", "P6", "P9"]
ALLOWED = [m for m in METHODS if m not in BLOCKLIST]   # P0,P2,P4,P5,P7,P8

# Round-0 is policy-independent by construction, but P0/seed-3000 diverges from
# P1-P9 on ext_brats2020 + msd_task07_pancreas (a re-submit artifact). Derive the
# SHARED round-0 state features from the P1-P9 consensus only.
ROUND0_CONSENSUS_METHODS = [m for m in METHODS if m != "P0"]

# ----------------------------------------------------------------------------
# Collapse definition (from OBSERVED training outcomes only -- gate test #15).
# A (dataset, method, seed) cell is flagged collapse if ANY hold:
#   (a) absolute catastrophic : final DSC < COLLAPSE_ABS
#   (b) relative collapse     : final DSC < Random(same ds,seed) - COLLAPSE_REL_GAP
#   (c) mid-trajectory instab : max running-peak-to-later drop > COLLAPSE_DROP
# ----------------------------------------------------------------------------
COLLAPSE_ABS = 0.15
COLLAPSE_REL_GAP = 0.10
COLLAPSE_DROP = 0.15

# Within-epsilon band, calibrated to measured cross-seed AUBC std (~0.017-0.020).
# At this epsilon nearly the whole good cluster ties the best -- which is itself
# the headline finding.
EPS_AUBC = 0.016

# ----------------------------------------------------------------------------
# Column groups of the exported modeling table (one row per dataset x method).
# ----------------------------------------------------------------------------
# Block A: static dataset descriptors (same for every method-row of a dataset).
STATIC_NUM_COLS = [
    "n_classes", "is_multiclass", "is_3d", "pool_N", "full_train_N",
    "n_groups", "slices_per_case", "fg_frac_mean", "fg_frac_median",
    "rarest_class_frac", "class_imbalance", "img_h", "img_w", "aspect_ratio",
]
STATIC_CAT_COLS = ["modality", "object_family"]

# Block B: shared round-0 state (consensus over P1-P9, seed-mean).
ROUND0_COLS = [
    "r0_dsc", "r0_detection_rate", "r0_missed_rate",
    "r0_dsc_class_min", "r0_dsc_class_mean", "r0_labeled_frac",
]

# Block D: method descriptors.
METHOD_COLS = ["family"] + METHOD_FLAG_COLS + ["in_blocklist", "exp_query_cost_z"]

# Feature columns the LODO models may consume (dataset_id is a GROUP KEY ONLY).
FEATURE_COLS = STATIC_NUM_COLS + STATIC_CAT_COLS + ROUND0_COLS + METHOD_COLS

# Target columns (never used as features).
TARGET_COLS = [
    "aubc_mean", "aubc_std", "dsc_final_mean", "regret", "rank",
    "within_eps", "p_within_eps", "collapse_prob",
    "train_cost_mean", "query_cost_mean",
]

# Columns that must NEVER appear in FEATURE_COLS (leakage firewall -- test #2).
FORBIDDEN_FEATURE_COLS = set(TARGET_COLS) | {
    "dataset", "seed", "aubc", "dsc_final", "is_collapse", "best_method",
}

SKILL_DATASET_VERSION = "qss_v1_frozenv5_19set"
RUNS_DIR = "runs/frozen_v5"
SKILL_DIR = "runs/frozen_v5/skill"
