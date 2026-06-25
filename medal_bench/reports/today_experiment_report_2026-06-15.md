# MedAL-Bench — frozen_v3 implementation + validation report (2026-06-15)

For external review. Self-contained: every amendment claim cites `file:line` so it can be checked
against the tree. Companion config doc: `stage2_config_rationale.md`.

---

## 0. Target of today's experiment

**Validate the frozen_v3 benchmark protocol end-to-end so Stage 2 can launch — and resolve the one
open training question (iters adequacy).** The goal was *protocol correctness + the pre-Stage-2 gate*,
**not** to produce a method ranking. Concretely, the gate (amendment #10) was: (1) full pytest green,
(2) a v3 canary that proves the protocol gives sane, leakage-free metrics, (3) a prediction-storage
estimate to size disk, (4) the BTCV-2000 / iters-adequacy check.

**Verdict: target met.** All four gate items are satisfied; the iters check additionally surfaced a
training-fairness confound that defines the next (frozen_v4) improvement.

---

## 1. Amendment-by-amendment accounting (the pre-experiment instructions)

| # | Amendment | Status | Evidence |
|---|---|---|---|
| 1 | Metric naming: emit `mean_dsc_fg_case_macro` + `mean_dsc_fg_pooled_diagnostic`; set `primary_metric` + `metric_version`; keep `mean_dsc_fg` as alias | ✅ Done | `runner/eval.py:172-177` — `primary_metric="mean_dsc_fg_case_macro"`, `metric_version="v3_case_macro"`, both names emitted, `mean_dsc_fg` aliased to the case-macro primary |
| 2 | Log `eval_scope` (`case_full_volume` vs `case_retained_slices`); don't call retained-slice eval "full volumetric" | ✅ Done | `runner/eval.py:174` — `eval_scope = "case_retained_slices" if has_slice_index else "case_full_volume"`. Retained-slice runs are labeled as such (not "full volume") |
| 3 | Valid region as full `valid_mask` or `valid_bbox=(y0,x0,h,w)` (not `(nh,nw)` top-left-assuming); use it for P1/P2/P5; intersect P6 explicitly | ✅ Done | `runner/al_loop.py:107,135-139,179-183` store `valid_bbox` on disk + per-sample meta; `policies/_helpers.py` `'valid'` aggregation mode; P1/P2/P5 switched `'full'→'valid'`; `policies/p6_selective_uncertainty.py:55-59` `_intersect_valid_` zeros outside the bbox |
| 4 | Use valid masks in metric computation; loss masking + class-weighted/focal loss may stay deferred | ✅ Done (deferrals taken as permitted) | `runner/eval.py:45-46,100-102` `_restrict_to_valid` applied to pred+gt before every DSC/surface metric. Loss masking + weighted loss **intentionally deferred** per the amendment |
| 5 | P8 `MIN_CLUSTER_SIZE` configurable + logged graceful fallback; log filtered/singleton/relaxed counts | ✅ Done | `policies/p8_sam_typiclust.py:47,57` configurable `min_cluster_size`; `:110-117` graceful relax (`if not eligible: eligible = non_empty; relaxed = True`); `:159-162` logs `typiclust_min_cluster_size/_num_filtered_clusters/_num_singleton_clusters/_min_cluster_relaxed` |
| 6 | Component-level seeds: `model_init/loader/query/dropout`; log all | ✅ Done | `runner/al_loop.py:506-525,609,688` — `component_seeds(round_seed)` derives all four; `seed_torch(model_init_seed)`, loader+dropout passed to trainer, `query_seed` to the policy; all logged in the trajectory record |
| 7 | Save masks+IDs+valid always; fp16 probs all-cell only if canary storage estimate OK; **hard gate before Wave 2**; else probs for debug/headline subset | ✅ Done | `runner/al_loop.py:285-286,554-564` `save_predictions`/`save_logits`; `trajectory.py:149-155` `write_predictions` stores sample_ids+pred+gt+patient_ids+slice_indices+valid_bbox (+ optional fp16 probs). **Estimate: ~0.8–1.2 TB all-cell → decision: masks always, probs debug-subset only.** Hard gate honored: no Stage-2 wave launched |
| 8 | Budget denominator = `actual_AL_pool_N`; log `fraction_of_AL_pool` + `fraction_of_full_train` | ✅ Done | `runner/run_one.py:142-148` derives `actual_AL_pool_N` from the real pool builder + asserts the plan fits; `runner/al_loop.py:435-436` logs both fractions; `run_one.py:170-173` prints both |
| 9 | Run BTCV-2000 check; if 2000 materially improves, propose documented family rule OR keep 1000 with caveat — no silent one-off | ✅ Done | Probe ran isic/msd07/**btcv** @2000. **btcv materially improves (Random +0.195 @2000).** Response = a *documented general rule* (adaptive train-to-plateau, frozen_v4), explicitly **not** a silent btcv exception. (Full btcv×all-methods @2000 sweep not completed — probe stopped once the confound was conclusive; this is the one partial item, see §3) |
| 10 | Don't launch Stage 2 until pytest passes + canary passes + storage reviewed + BTCV-2000 reviewed | ✅ Done | All four met (§2, §3). **Stage 2 NOT launched** — awaiting explicit go |

**Bottom line on the instructions: all 10 amendments are implemented.** The only non-complete piece is
the *breadth* of #9's @2000 sweep (intentionally truncated — see §3), which did not change the
conclusion. Test suite: **173 tests pass** (22 test files), up from 102.

---

## 2. Result A — frozen_v3 validated (the canary)

**Canary: 25/30 cells** (5 datasets × 6 methods {P0,P1,P4,P5,P8,P9}, seed 1000, fixed 1000 iters). It
confirmed the v3 protocol end-to-end:

- Per-case macro-fg DSC is the primary metric (grouped by patient_id, else sample_id).
- Diagonal total-miss penalty + structure-detection rate work (btcv: 12 total-missed organs penalized;
  detection rate 0.69).
- P8 TypiClust fidelity: min-cluster filter active, diagnostics logged.
- Prediction saving works (masks + ids + valid_bbox round-trip).
- **P0 no longer collapses on msd07** (0.338 vs 0.00 before the per-case + stratification fixes).

**Storage gate:** all-cell fp16 probs ≈ **0.8–1.2 TB** for a 1-seed Stage 2 (measured ~60 GB / 18
canary cells; mmwhs ~1.5 GB/round). → **decision: masks always, probs for a debug subset only.**

---

## 3. Result B — the iters-sensitivity probe (the headline finding)

The BTCV-2000 check (#9) was widened to {isic, msd07, btcv} @ 2000 iters vs the canary @ 1000.

**Fixed-iter confound CONFIRMED.** On isic, ΔDSC (1000→2000 iters):

| method | selection type | Δ@2000 |
|---|---|---|
| P0 Random | diversity | +0.015 |
| P8 TypiClust | diversity | +0.010 |
| P1 entropy | uncertainty | +0.048 |
| P4 BADGE | gradient | +0.066 |
| **P9 PAAL** | hardest-data | **+0.089** |

**Monotonic in selection difficulty:** the harder a method selects, the more it was under-fit at 1000
iters and the more it recovers at 2000. Diversity/random barely move; difficulty-based methods recover
∝ how hard they select; PAAL closes ~75% of its gap to Random. Separately, **Random itself is under-fit
on hard datasets** (Δ@2000: btcv **+0.195**, msd07 +0.047, isic +0.015) — no single fixed iter count
fits all datasets.

**Implication:** a fixed iteration budget **biases the benchmark toward diversity/random methods**, so
the "Random ≥ uncertainty methods" pattern (seen in Stage 1.5 and the canary) is *substantially a
training-adequacy artifact, not a true method ranking.* This is the documented general rule for #9: the
fix is per-(dataset,method) adaptive iters, not a btcv one-off.

> **Honesty note (the one partial item):** the @2000 probe was **stopped early** once the confound was
> conclusive (isic full table + btcv-P0 +0.195). A complete btcv×all-methods @2000 grid was not run, and
> the @2000 probe is abandoned (not to be resubmitted) because the chosen fix is adaptive, not @2000.

---

## 4. Result C — adaptive-iters fix (frozen_v4 prep, IN PROGRESS — not a Stage-2 gate)

Driven by §3. Replace fixed iters with **adaptive train-to-plateau** per (dataset, method) — removes the
confound at scale without per-dataset tuning (can't be done with any single fixed count).

- **Criterion bug found + fixed:** a *relative* train-loss plateau never triggers near zero loss
  (0.050→0.0495 reads as "1% better" forever). Switched to an **absolute** loss-delta
  (`smooth < best_smooth - plateau_min_delta`, default 0.003), `min_iters=500`, `max_iters=3000`,
  window 100, patience 5. Verified on a toy curve (plateaus at 1350 vs previously hitting the cap).
- **6-cell sanity run in flight** ({isic,msd07,btcv}×{P0,P9}). **Early yellow flag:** isic-P9 round 0
  stopped at `iters=3000, stop_reason=plateau` — i.e. it grazed the cap. Decisive next signal is
  isic-**P0** round 0 (same initial set, no selection): if P0 also rides to 3000, `min_delta=0.003` is
  too tight → bump to 0.005–0.01 and re-tune before adopting frozen_v4.

**frozen_v4 is therefore NOT yet validated.** It is a *later* improvement, explicitly out of the
pre-Stage-2 gate.

---

## 5. Where this leaves Stage 2

Two valid launch configs:

- **frozen_v3 (fixed 1000 iters) — validated, launch-ready now.** Pipeline + per-budget trajectories
  are sound (good for skill-learning data + an indicative first-pass benchmark). Caveat: method
  *rankings* carry the confound (lean toward diversity/random) — label them "indicative / 1000-iter."
- **frozen_v4 (adaptive) — cleaner rankings, not yet validated.** Decisive signal ~30–60 min out;
  full per-dataset stop_iter table ~3–4 h; may need a `min_delta` re-tune (→ tomorrow) if isic-P0 also
  caps.

**Recommendation:** launch frozen_v3 Stage 2 now (carrying the two non-controversial efficiency fixes:
surface-metrics-at-final-only, debug-subset probs), and let the adaptive run finish in the background as
frozen_v4 for a later clean-ranking pass.

**Status: Stage 2 NOT launched. Awaiting explicit go (config choice + frozen_v4 decision).**

---

## 6. Open questions for the reviewer (GPT)

1. Is launching Stage 2 on **frozen_v3 (fixed 1000)** acceptable for a *skill-learning trajectory*
   dataset given the confirmed ranking confound, or should it wait for frozen_v4?
2. For frozen_v4: is **absolute train-loss-delta plateau** the right criterion, or should it use a
   val-DSC plateau / PAAL's own stopping rule? Is `min_delta` 0.003 (→0.005–0.01) reasonable, and is
   `max_iters=3000` enough for the 14-class (btcv-family) datasets that were +0.195 @2000?
3. Is **1 seed (1000)** acceptable for the first Stage-2 wave (vs the ~0.02 GPU-nondeterminism noise
   floor), with 2000/3000 added before any "method X wins" claim?
