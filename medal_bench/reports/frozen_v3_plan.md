# frozen_v3 — Decisions & Implementation Plan

## ★ FINALIZED SPEC (user-locked 2026-06-15) — implementation checklist
Stage 1.5 validated the decision (see `stage1p5_report.md`). Implement these, then a v3 canary; do NOT
launch full Stage 2 until canary review. "No more broad code debugging beyond these v3 protocol fixes."

1. **num_iters = 1000 global default.** Before final Stage 2, run a **BTCV-only 2000-iter check**;
   report whether 2000 materially improves BTCV or changes its method ranking → if so, BTCV gets a
   2000-iter exception, else keep global 1000.
2. **Metrics — per-case macro foreground DSC = PRIMARY.** For 3D-as-slice datasets, group slices by
   `case_id`, reconstruct per-case prediction/label volumes, compute per-case DSC, macro over cases &
   fg classes. Native-2D: each image = one case. Secondary: `HD95_case_macro_fg`,
   `symmetric_ASSD_case_macro_fg`, `structure_detection_rate`, `missed_structure_rate`. Total-miss →
   **diagonal penalty + detection rate** (never silently drop). Keep old micro/pooled DSC as a diagnostic only.
3. **Valid-region query aggregation** for P1 (entropy), P2 (BALD), P5 (entropy→coreset). Verify P6
   already ignores padding via its fg/boundary mask.
4. **Budget denominator = actual accessible AL_pool_N.** Log: full_train_N, requested_pool_cap,
   actual_AL_pool_N, budget_fraction_of_AL_pool, budget_fraction_of_full_train, absolute counts.
5. **P8 TypiClust fix** before Stage 2: min-cluster-size filter + round-robin selection + no singleton-
   outlier picks. Keep old P8 as deprecated/ablation variant.
6. **Component-level deterministic seeding; log all seeds.**
7. **Prediction saving (always):** compressed val prediction masks + case/slice IDs + valid-region masks
   + spacing/affine metadata if available. Save logits if storage allows, else logits for selected/debug
   subsets only.
8. **Document foreground-only pool:** 3D-slice benchmark uses fg-positive retained slices — acceptable
   for Stage 2 but stated clearly.
9. **v3 validation canary (before Stage 2):** datasets {busi, isic2018, mmwhs, btcv_synapse,
   msd_task07_pancreas}; methods {P0,P1,P4,P5,P8,P9}; seed 1000; first/mid/final budgets (or full curve
   if cheap). Report: per-case DSC; HD95/ASSD w/ total-miss penalty; valid-region score behavior;
   corrected budget fractions; P8 corrected behavior; prediction saving; runtime; whether P0 no longer
   collapses on MSD07; whether full_sup_pool is adequate.
10. **Stage 2 dataset list:** curated 18–24 (proposal in `stage2_dataset_list.md`); after canary passes,
    run Stage 2 in waves.

---


Decisions locked 2026-06-15 after the Stage 1.5 code review (`stage1p5_code_review.md`). **No core code
edited yet** — the v2 Stage 1.5 sweep is still running and shares the codebase, so implementation waits
until it completes (else cells would mix v2/v3 and corrupt the iters experiment).

## Locked decisions (user-approved)

1. **Metrics → per-case (per-volume) evaluation.** Group val slices by `patient_id`, assemble per-class
   3D masks, compute **DSC + HD95 + symmetric ASSD per case**, then macro-average over cases and over
   foreground classes. Replaces the current per-slice micro (pooled) DSC and directed ASD. Fixes review
   items **C1 + M1 + M4** together.
2. **Missed-structure (total-miss) convention → diagonal penalty + detection rate.** When a class is
   present in a case's GT but the predicted volume is empty for it, set HD95/ASSD = the case's image/
   volume **diagonal** (worst-case) instead of dropping it; additionally report **detection rate**
   (fraction of GT-present structures the model predicted non-empty) as a separate diagnostic.
3. **Roll-out → batch into frozen_v3; small v3 validation smoke (NOT a full Stage 1.5 re-run); save val
   logits in Stage 2.** *Refined 2026-06-15 per user:* Stage 1.5 is scaffolding to *decide* v3, not a
   downstream deliverable (skill learning consumes Stage 2/3, never 1.5). The v3 *decisions* (iters,
   valid-region, budget) come from the current **v2** Stage 1.5 — relative comparisons robust to the
   metric change — so a full v3 re-run of the 6×7×3 matrix would just duplicate Stage 2. Instead, after
   implementing v3, run a **small v3 validation smoke** (≈1–2 datasets × a few methods × 1 seed) to
   confirm correctness (per-case metrics sane, per-round seeding deterministic, faithful-P8 runs, logits
   save, no crashes), then go straight to Stage 2 under v3. Persist val logits in Stage 2 so future
   metric revisions never require retraining. v2 Stage-1/1.5 absolute DSC/HD95 numbers stay v2 (internal
   go/no-go only).
4. **P8 TypiClust → make paper-faithful.** Add the `MIN_CLUSTER_SIZE` (>5) filter and round-robin
   (`i % n_clusters`) selection (review **Med1 + Med2**); re-run P8 under v3.

## Full frozen_v3 change set

**Result-changing (require the v3 re-run; do NOT apply while v2 Stage 1.5 runs):**
- [A] Per-case volumetric metrics in `runner/eval.py` (DSC/HD95/symmetric ASSD per case; diagonal
  penalty for missed structures; detection rate). Needs val slices grouped by `patient_id` + 3D assembly.
- [M3] Per-round seeding: `seed_all(cfg.seed + r)` at the top of each round in `al_loop.py` before model
  build + train (currently seeded once → per-round init relies on carried-over global RNG).
- [P8] `p8_sam_typiclust.py`: MIN_CLUSTER_SIZE filter + round-robin selection.
- [valid-region] Aggregate P1/P2/P5 query scores over the valid (un-padded) region — **pending the
  1.5B forward-pass prototype** confirming it changes selections; if confirmed, add a "valid" mode in
  `_helpers.aggregate` and switch P1/P2/P5.
- [iters] num_iters for v3 — **pending Stage 1.5A**: decide the number and whether global or
  dataset-family-specific from the iteration-sensitivity table.
- [logits] Save val logits/preds in Stage 2 (and the v3 re-run) so metrics are recomputable without retrain.

**Reporting-only (safe, no re-run needed):**
- [M2] Budget denominator: compute the grid / log fractions against the **true post-cap pool N**
  (busi624 kvasir800 isic2076 glas133 origa520 btcv1494 mmwhs2500 msd2500 brats2500), keeping the
  absolute count schedule. Fix in `run_one.py` to mirror `run_full_supervised.py`'s pool-size derivation.

**Robustness (no result change; can apply anytime):**
- BADGE `kmeanspp_indices` k≤0 guard; SAM cache-key include checkpoint id + input H; msd07 affine-based
  axial axis; `_IndexedSubset.patient_ids()` per-sample membership.

**Deferred (decide after iters):**
- Loss design (CE over bg+pad, 1:1 CE:Dice weight, full Dice penalty for batch-absent class) — revisit
  only if longer training alone doesn't fix multiclass, to avoid confounding the iters result.

**No action (documented):** BADGE gradient sign (sign-invariant), CoreSet/feature L2-normalization
(deliberate), P9 minor deviations (WPS L2-norm / AP val split / KMeans seed — documented), softmax-before-
argmax (harmless), P6 score-logging nit.

## Sequence
1. Finish current **Stage 1.5 (v2)** → iters signal (valid; relative comparison) + full-sup adequacy.
2. Run **1.5B** valid-region forward-pass prototype → confirm/deny the aggregation change.
3. Implement **frozen_v3** (all the above), bump FROZEN hash, document.
4. **Small v3 validation smoke** (≈1–2 datasets × a few methods × 1 seed) — correctness/sanity only, NOT
   a full Stage 1.5 re-run (that would just duplicate Stage 2).
5. **Stage 2** = full 3-seed P0–P9 matrix under frozen_v3, saving val logits → feeds skill learning.

## Guardrails
- Stage 2 stays blocked until frozen_v3 is reviewed (per user).
- Existing Stage-1/1.5 **DSC rankings** remain indicative (DSC bias is reweighting, not leakage); the
  **HD95/ASD numbers** in v2 reports are superseded by the v3 per-case metrics.
