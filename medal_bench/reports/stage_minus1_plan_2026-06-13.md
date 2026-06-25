# Stage −1 — Implementation Plan (build before formal Stage 1)

Date 2026-06-13. Deliverable B. Status legend: ⬜ todo · 🔨 in progress · ✅ done.

Goal: build the 8 missing prerequisites (B1–B8) so the formal benchmark can run.
Each item lists: files to change, tests to add, risk, dependencies. Ordered by
dependency + value. All changes are **additive** where possible (new modules,
new profile) to avoid disturbing the validated P0–P9 method code (102 tests).

Hard rule: **do not alter P0–P9 method behavior.** Resolution/budget/remap/metric
changes go behind a new profile + frozen config v2, leaving `smoke`/`pilot` intact
until v2 is reviewed.

---

## Implementation order & dependency graph

```
B1 remap+MMWHS ──┐
B2 adaptive-res ─┼─► B8 frozen v2 ─► Stage 0b ─► Stage 1
B3 budget policy ┘                    ▲
B5 derived metrics ───────────────────┤ (post-hoc; needed for Stage 1 report)
B4 full-sup runner ───────────────────┤ (needed for Stage 1 report)
B7 SAM-H precompute ──────────────────┘ (throughput convenience)
B6 throughput fix ──► gating Stage 1 at scale (most invasive; do carefully)
```

B1+B2+B3+B8 are the critical path to **Stage 0b**. B4/B5/B7 are needed for the
Stage 1 *report* but not for Stage 0b. B6 is needed before Stage 1 *at scale*.

---

## B1 — Remap infrastructure + MMWHS adapter (+ BTCV)  🔨 [keystone]

**New files**
- `data/remap.py` — `LabelRemapper(mapping: dict[int,int], name: str)` with a
  vectorized LUT, `apply(mask)->mask`, raising `ValueError` on any native code
  not in `mapping`. Named constants: `MMWHS_REMAP`, `BTCV_REMAP`, `MYOPS_REMAP`,
  `BRATS_REMAP`, `CARE_LA_BINARY`, plus stubs for AbdomenCT-1K/CHAOS (commented,
  pending semantic confirmation).
- `data/adapters/mmwhs.py` — `MMWHSAdapter(root_dir, modality)`, 3D→2D sliced,
  `nib.as_closest_canonical` orientation, modality-aware normalization (CT: HU
  window; MR: per-volume 0.5–99.5 percentile → [0,1]), applies `MMWHS_REMAP` in
  `__getitem__`. `patient_id = "Case{NNNN}"`, `sample_id = "{case}_{z:03d}"`.
  `num_classes=8`.
- `data/adapters/btcv.py` — `BTCVAdapter(root_dir)`, CT HU window, applies
  `BTCV_REMAP` (`16→13`, dense 14 classes). `num_classes=14`. (Wire AFTER E,
  the BTCV decision note.)

**Registry** (`data/adapters/__init__.py`): add `mmwhs_ct`, `mmwhs_mr`,
`btcv_synapse` (BTCV after note E). Data roots:
`{dr}/3d/mmwhs/extracted/Wholeheart_Train_Dataset/...`,
`{dr}/3d/btcv_synapse/extracted/btcv_ct/...`.

**Design decision (documented):** MMWHS is split into `mmwhs_ct` (60 cases) and
`mmwhs_mr` (46 cases) — NOT one combined pool — because CT/MR intensity stats and
orientations differ and a mixed AL pool would confound the benchmark. Same remap
for both.

**Tests** (`tests/test_remap.py`, `tests/test_mmwhs_adapter.py`):
- `test_remap_dense_labels_mmwhs` — apply MMWHS_REMAP → unique ⊆ {0..7}.
- `test_loss_accepts_mmwhs_mask` — CE+Dice loss runs on a remapped (C=8) mask.
- `test_metrics_accept_mmwhs_mask` — eval.py DSC runs on C=8.
- `test_unknown_native_label_raises` — code 999 → ValueError.
- `test_background_zero_preserved` — 0→0.
- `test_native_high_value_labels_preserved_before_remap` — load via nibabel keeps
  850; assert not truncated to uint8.
- `test_mmwhs_orientation_canonical` — two cases with different native orientation
  slice along a consistent axial axis.
- `test_btcv_dense_remap_after_semantic_confirmation` — (after E) 16→13, unique ⊆ {0..13}.

**Risk:** Low-medium. Additive new files + registry lines. Risk is in MR
normalization choice and orientation handling — covered by a loader-level smoke +
mask/image overlay sanity check (deliverable D).

**Dependencies:** none (uses existing nibabel). Unblocks Stage 0b.

---

## B2 — Adaptive resolution policy  ⬜

**Files**
- New `data/preprocess.py` — `resize_long_side(img, mask, long_side, pad_multiple=32)`:
  preserve aspect ratio, resize by long-side cap (bilinear img / nearest mask),
  pad to multiple of 32, return arrays + a `SizeLog{orig,resized,padded}` dict.
- `runner/al_loop.py` `_IndexedSubset` + `trainer.py` `_resize_image/_resize_mask`:
  route through `preprocess.py` when the profile selects an adaptive mode; keep the
  legacy square path for `smoke`/`pilot` (back-compat).
- `profiles/__init__.py`: add `ResolutionPolicy{mode, long_side, pad_multiple}` and
  a new `bench` profile (long_side=512). Modes: smoke_res=384, bench_res=512,
  hires_res=768, max_res=1024.
- `runner/trajectory.py`: log `orig/resized/padded` size + resolution mode + hash.

**Tests** (`tests/test_preprocess.py`): aspect_ratio_preserved, padding_multiple_of_32,
mask_nearest, image_bilinear, resolution_metadata_logged,
no_hidden_256_dependency_in_formal_profile.

**Risk:** Medium. Touches the hot resize path. Mitigate: legacy path untouched;
adaptive path behind profile flag; variable-size tensors require per-sample
(batch=1) or same-size-bucketed collation — confirm collate handles padded sizes.

**Dependencies:** interacts with B6 (cache key must include resolution) and B8.

---

## B3 — Pool-size-dependent budget policy  ⬜

**Files**
- New `profiles/budget.py` — `budget_grid(N, num_classes)` returning
  (fracs_or_counts, absolute_cumulative_counts, initial_count) per Cases A–D;
  `initial_count = max(8, 2*num_classes, ceil(first_frac*N))`, optional cap
  128/256 (documented in frozen v2).
- `profiles/__init__.py`: `bench` profile uses `budget.budget_grid` instead of flat
  `cumulative_budget_plan`.
- `runner/trajectory.py`: log N, fracs, absolute cumulative counts, incremental
  counts, initial_count, max_count (some already logged; add fraction + initial).

**Tests** (`tests/test_budget.py`): case_A/B/C/D grids, initial_count_floor,
incremental_counts_cumulative, budget_counts_logged.

**Risk:** Low. New module; `cumulative_budget_plan` stays for legacy profiles.

**Dependencies:** B8 (records the policy).

---

## B4 — Full-supervised baseline runner  ⬜

**Files**
- New `runner/run_full_supervised.py` — train on 100% of `split.train` with the
  same backbone/resolution/schedule/aug as the `bench` profile; eval on val (+
  internal test); write a `{dataset}__FULL__s{seed}.json` with DSC/HD95.
  Reuses `al_loop` training internals (factor a `_train_and_eval` helper, or call
  `run_al` with a single round at budget=N).

**Tests** (`tests/test_full_supervised.py`): uses_all_labeled_train_pool,
excludes_val_test, metrics_saved, relative_dsc_computable.

**Risk:** Low-medium. Mostly reuse. Watch: huge-N datasets (msd07 capped) need the
capped-protocol note.

**Dependencies:** B2/B3 (same preprocessing+schedule as AL runs). Needed for Stage 1 report, not 0b.

---

## B5 — Derived metrics module  ⬜

**Files**
- New `analysis/derived.py` — read trajectory JSONLs across methods/seeds, compute:
  AUBC (trapezoid over cumulative-fraction vs DSC), final-budget DSC/HD95,
  gain-over-random, budget_to_90/95_full (interpolated vs B4 baseline), regret to
  best-fixed, average rank, win rate, per-budget/per-dataset winner. Output tidy
  CSV/Parquet.
- `runner/eval.py`: allow HD95 at first/middle/final budgets (Stage 1), not just
  final (add a `surface_at_rounds` list to the eval call).

**Tests** (`tests/test_derived.py`): aubc_formula, gain_over_random,
budget_to_90_full, regret_to_best_fixed_method, average_rank, win_rate,
empty_mask_metric_policy (assert documented behavior in eval.py).

**Risk:** Low. Pure post-hoc analysis on logged data.

**Dependencies:** B4 (full-sup for relative metrics). Needed for Stage 1 report.

---

## B6 — Data-loading throughput fix  ⬜ [most invasive]

**Problem:** `_IndexedSubset` (al_loop.py:100) eager-loads + resizes the whole pool
from NFS every round (~15 min/2k slices).

**Approach (lowest-risk):** disk cache of preprocessed `(image,mask)` arrays keyed
by `(dataset, preprocess_version, remap_version, resolution_policy, sample_id)`,
written as a per-dataset `.npz`/memmap under `cache/preprocessed/`. First run
builds it; later rounds/methods/seeds memmap-read. Preserve deterministic ordering.

**Files:** new `data/cache.py`; `runner/al_loop.py` `_IndexedSubset` reads from cache
when present. Log cache-build vs cache-hit time separately; add GPU-mem logging
(closes Stage 0a W1).

**Tests** (`tests/test_loader_cache.py`): cache_key_includes_preprocess_remap_resolution,
cached_and_uncached_samples_match, deterministic_loader_order,
no_eager_full_pool_read_in_cached_mode.

**Risk:** High — central hot path. Mitigate: cache behind a flag, fall back to eager
on miss, byte-equality test cached-vs-uncached. Benchmark on busi/isic2018/msd07.

**Dependencies:** B2 (resolution in cache key). Gates Stage 1 at scale.

---

## B7 — SAM-H feature cache warming  ⬜

**Files:** new `runner/precompute_sam.py --datasets ... --sam-model-type vit_h
--resolution bench` looping `extract_sam_features` over Stage-1 train splits;
extend the SAM cache key to include resolution (currently keyed on
encoder_id+preprocess_hash; add resolution). No silent vit_b fallback (already
enforced — verify a hard error if vit_h ckpt missing).

**Tests** (`tests/test_sam_precompute.py`): precompute_command, no_silent_fallback,
cache_key_includes_resolution, cache_invalidation_on_resolution_change.

**Risk:** Low. SAM caching already exists + is collision-safe (Stage 0a confirmed).

**Dependencies:** B2 (resolution in key).

---

## B8 — Frozen config v2  ⬜

**Files:** new `profiles/frozen_v2.py` (leave `frozen.py` as the v1 record). Include:
P0–P9 ids+versions, P4b/P8b ablation-only, SAM-H/vit_h, adaptive resolution policy,
pool-dependent budget policy, metric policy, HD95 policy (first/mid/final Stage 1),
empty-mask policy, split policy, official images-only test exclusion, remap version,
preprocessing version, loader/cache version, seeds, query_unit, slice-id format,
Stage 1 dataset inclusion list. New `FROZEN_V2_HASH`.

**Tests** (`tests/test_frozen_v2.py`): hash_changes_from_v1, contains_resolution_policy,
contains_budget_policy, contains_remap_version, contains_metric_policy.

**Risk:** Low. Declarative.

**Dependencies:** B1,B2,B3,B5 (records their final form). Last before Stage 0b/1.

---

## Summary of new test files
test_remap.py, test_mmwhs_adapter.py, test_preprocess.py, test_budget.py,
test_full_supervised.py, test_derived.py, test_loader_cache.py,
test_sam_precompute.py, test_frozen_v2.py — target ~40 new tests on top of the 102.
