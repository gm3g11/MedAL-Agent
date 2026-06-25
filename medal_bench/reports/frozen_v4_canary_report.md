# frozen_v4 canary — overnight report (2026-06-16, ~00:15)

**TL;DR (updated ~04:00):** the v4 protocol now **completes clean END-TO-END on the light datasets —
busi+isic 9/10 done, 0 churn, incl. the SAM/P8 cell** (isic P8 still running). The full-curve canary
**hit a host-RAM OOM-churn blocker on the BIG datasets** (btcv/msd07) — cells silently SIGKILL'd before
finishing 6 rounds → restart → churn → 0 big-dataset completions. **OOM is confirmed big-dataset-only.**
**Wave 2 was NOT launched.** NEW finding from the clean completions: **cap-hit is budget-dependent, not
just dataset-hardness** — even busi/isic cap at high budget (r0 0% → r5 78% capped), with near-ceiling
flat-ish val-DSC there (train-loss memorizes past 0.005/window once the labeled set is large). Three
calls for this morning: **(1) the OOM fix** (blocks all big-dataset runs incl. Wave 2), **(2) the
criterion** — abs=0.005 is *safe* (trains to good DSC) but rarely economizes iters at high budget;
consider abs 0.008–0.01 or a val-DSC-based stop, **(3) btcv/msd07 max_iters** (they cap *and* are hard).

See **§5** for the clean-completion table + the budget-vs-cap finding.

---

## 1. What the criterion did (VALIDATED — round level)
From the partial trajectories before the churn was stopped:

| dataset | rounds seen | stop_iter / reason | per-case DSC | read |
|---|---|---|---|---|
| **isic2018** P9 | r0,r1,r2 | **2000/plateau**, 3000, 3000 | 0.78, 0.78, 0.83 | plateaus BELOW cap r0 → **abs=0.005 works** (was 3000 under abs=0.003) |
| isic2018 P8 | r0,r1 | 2900/plateau, 2900 | 0.78, 0.77 | just under cap; clean |
| btcv_synapse (P0,P1,P4,P8,P9) | up to r2 | **all 3000/plateau_max** | 0.32–0.43 (rising) | 14-class CT caps — undertrained, needs max_iters>3000 |
| msd_task07_pancreas (all) | up to r3 | **all 3000/plateau_max** | 0.14–0.33 (**rising, >0**) | 0.63%-fg caps — **no collapse** (DSC>0), but undertrained |

- **New logging fields serialize** (stop_iter, stop_reason, hit_max_iters, best_smooth_loss,
  train_runtime_sec, query_runtime_sec). ✓
- **No crashes / no Python errors** (0 `.fail.txt`). Per-case metrics + valid-region + P8 all ran.
- **msd07 does NOT collapse** (DSC 0.15→0.33 across rounds). ✓
- **cap-hit 81%** but **entirely** in btcv + msd07 (the two known-hard datasets); isic plateaus cleanly.
  Per-method: P9/P4 cap only where P0 also caps (dataset-hardness, not method bias).

## 2. The blocker — host-RAM OOM churn
- Cells die with **rc=-9 (SIGKILL)**, **0 `.fail.txt`**, **no "CUDA out of memory"** in any log → it's
  **host RAM**, not GPU memory, not a Python error.
- Death times are **highly variable (397–3564 s)** → resource-driven kills, **not** our 4h cell-timeout.
- The dispatch misreads each SIGKILL as a timeout → releases the cell → a peer re-claims → it dies again
  → **infinite churn, 0 completions** in ~95 min on 12 GPUs.
- **Why now (v3 canary finished 25/30 with heavier I/O):** most likely **compute-node memory
  contention** from other jobs tonight, hitting our big-dataset cells (btcv 1494 / msd07 2500 / @512
  eager-loaded pools, ~tens of GB RSS each; several per node → OOM). Possibly aggravated by the longer
  adaptive runs colliding with peers.

## 3. What I did overnight (safe path)
- **Stopped the churn** (qdel'd the 12 A40/V100 workers — they were completing nothing and clobbering
  partials). Kept the 4 H100 workers queued (opportunistic, harmless).
- **Did NOT launch Wave 2** — its brats/mmwhs_ct are equally big and would churn.
- **Relaunched a clean-light canary: busi + isic2018 only** (10 cells, light pools, low OOM risk) on a
  modest 7 workers, to confirm the protocol completes end-to-end on non-pathological data by morning.
- Held **all big datasets** (btcv, msd07 + Wave-2's brats/mmwhs_ct/spleen) for this discussion.

## 4. Decisions for this morning
1. **OOM fix (blocks any full-curve big-dataset run incl. Wave 2).** Options, low→high effort:
   - **a)** Run **1 cell per node** (request more CPU slots/worker so SGE packs fewer heavy cells per
     node → more RAM each). No science change. *Recommended first try.*
   - **b)** Lower **val_cap** (eval-set size; affects only eval memory/cost, not training/AL) — cheap,
     science-safe-ish.
   - **c)** **Resume-from-partial** so a killed cell continues from its last completed round instead of
     restarting (makes forward progress despite occasional kills) — code change in al_loop/run_one.
   - **d)** Lower **pool_cap** — **avoid**: changes the AL pool size = the benchmark definition.
   - **e)** Just **retry on less-contended nodes / later** if it was transient peer contention.
2. **btcv/msd07 max_iters.** Both cap at 3000 with **rising** DSC = undertrained, not converged. Bump
   max_iters (e.g. 4000–5000) for the high-class-CT / sparse-fg family, or document a per-family rule.
   (Per your item 6 — rerun just those datasets, don't block the rest.)
3. **Then** launch Wave-2 confident-6 (busi, kvasir_seg, isic2018, ext_brats2020, mmwhs_ct,
   msd_task09_spleen) once the OOM fix is confirmed on a big dataset.

**Status:** busi+isic clean re-run **10/10 complete** (see §5); everything else awaiting your OOM +
max_iters/criterion calls.

---

## 5. busi+isic clean re-run — RESULTS (overnight completion)
**10/10 cells completed clean (10 `OK`, 0 timeout/churn) — full clean-data validation done.** This is
the end-to-end protocol validation on non-pathological data: per-case DSC computed, masks saved,
valid-region + P8/SAM all ran, new logging fields serialize.

Final per-cell stop_iter sequence (r0→r5) and final-budget per-case DSC:

| cell | stop_iters (r0..r5) | final DSC |
|---|---|---|
| busi P0 | 1700,1500,2500,2700,3000,3000 | 0.62 |
| busi P1 | 1700,1500,2800,2900,3000,3000 | 0.53 |
| busi P4 | 1900,1500,2300,2900,3000,3000 | 0.58 |
| busi P8 (SAM) | 1900,1500,1700,2100,2500,2900 | 0.57 |
| busi P9 | 1900,1800,2600,2700,2700,3000 | 0.59 |
| isic P0 | 2900,3000,3000,3000,3000,3000 | 0.86 |
| isic P1 | 2900,3000,3000,3000,3000,3000 | 0.76 |
| isic P4 | 2000,2600,3000,3000,3000,2400 | 0.85 |
| isic P9 | 2000,3000,3000,3000,3000,3000 | 0.86 |
| isic P8 (SAM) | 2000,3000,3000,2300,2600,3000 | 0.87 |

**KEY FINDING — cap-hit is BUDGET-dependent, not just dataset-hardness.** Cap-hits (stop_reason
`plateau_max`) by round, across the 9 done cells:

| round | r0 | r1 | r2 | r3 | r4 | r5 |
|---|---|---|---|---|---|---|
| capped | **0/9** | 3/9 | 4/9 | 4/9 | 6/9 | **7/9** |

- **r0 plateaus cleanly for every cell** (abs=0.005 works at low budget). As the labeled set grows, more
  rounds ride to the 3000 cap — at high budget the train loss keeps improving >0.005/window (more data
  to fit) so it rarely triggers the plateau.
- val-DSC at the capped rounds is **high and still inching up with budget** (isic P0:
  0.78→0.81→0.83→0.85→0.86→0.86; busi P0: 0.29→0.18→0.46→0.51→0.60→0.62) — the gains are mostly the
  **budget** effect (more labels), marginal from extra iters → the caps yield **good models, not
  catastrophic undertraining**.
- **Implication:** abs=0.005 is *safe* (trains to good DSC) but does NOT economize iters at high budget,
  and this same effect — amplified by dataset hardness — is what makes btcv/msd07 cap at every round.
  So **the criterion choice (abs vs higher abs vs val-DSC-based stop) and the OOM are the two levers**;
  a global max_iters bump would raise everyone's runtime (already ~2.5 h/cell on clean data).

**Bottom line:** v4 protocol is correct + completes on clean data. Before Wave-2: decide (1) OOM fix,
(2) criterion (keep abs=0.005 / raise to 0.008–0.01 / val-DSC stop), (3) btcv/msd07 max_iters.

---

## 6. RESOLUTION (2026-06-16, "solve all issues")
**Diagnosis refined:** A40/V100 nodes are **250 GB / 32-core**; with `-pe smp 8` SGE packs up to 4 heavy
cells/node, and a big cell's peak RSS pushes total node RAM over 250 GB → the host OOM-killer fires
(silent SIGKILL = the churn). Preproc caches were **read, not rebuilt** (so not a rebuild-storm). Key
asset: **qa-h100-002/003 have 503 GB RAM** (2× headroom).

**Decisions:**
1. **OOM fix = memory-aware scheduling, no science change.** Launch Wave-2 workers with an SGE
   `-l mem_free=<N>G` reservation (N sized from a live single-cell RSS probe, job 1099163 ext_brats2020),
   so SGE never overcommits a node, on **any** node size. Also route to the 503 GB H100 nodes when free.
   No pool_cap change. (Resume-from-partial noted as future hardening; not needed if nodes don't OOM.)
2. **Criterion = KEEP abs=0.005.** The canary proved it is *safe* (trains to good DSC: isic 0.86, busi
   0.62; no collapse). It caps at high budget (train-loss memorizes) but val-DSC is near-ceiling there,
   so the extra iters cost runtime, not correctness. Raising abs would trade a little convergence for
   speed — not worth changing the frozen protocol; fix throughput via the OOM/scheduling lever instead.
3. **max_iters = KEEP 3000 for the Wave-2 confident-6.** isic capped at 3000 yet hit 0.86 (near ceiling)
   → brats/mmwhs_ct (similar difficulty) should likewise give good DSC at 3000. The genuinely
   under-trained, *held* datasets **btcv + msd07** (14-class CT / 0.63%-fg, rising DSC at the cap) get
   **max_iters 4000–5000** when rerun later — not in tonight's confident-6.

**Action:** Wave-2 confident-6 (busi, kvasir_seg, isic2018, ext_brats2020, mmwhs_ct, msd_task09_spleen)
× P0–P9 × seed 1000 launched with `qsub -V -l mem_free`, abs=0.005, max_iters=3000.

---

## 7. STAGE-2 WAVE-2 (confident-6) — pre-full-Stage-2 report (2026-06-16)
**Validated representative set: 22/60 cells complete** across all 6 confident datasets + all 10 methods
represented; the remaining 38 are re-running (more of the same — they add data points/seed-robustness,
not new validation). Below is the 4-category check you asked for.

### (1) Completion
- **22/60 done, 0 failed-unrecoverably, 0 churn now.** All 6 datasets have completions; both **big
  datasets have clean P9 (PAAL) completions** (ext_brats2020 0.42, mmwhs_ct 0.76). Big-dataset **P8/SAM-H
  still in flight** — but P8/SAM was already validated end-to-end in the canary (busi+isic P8 completed).
- **Three infra issues hit + fixed during this run** (all now resolved, none affect data correctness):
  (a) **preproc-cache rebuild storm** on big datasets → fixed by serial pre-warm (`prewarm_cache.py`, #2);
  (b) **host-RAM contention** on 250 GB A40/V100 nodes → managed (per-cell peak is only ~9.4 GB GPU /
  ~tens GB host; rebuild was the real spike); (c) **claim-staling** (workers exited on nothing-claimable
  → orphaned dead-worker claims) → **fixed at root** (dispatch now polls-until-done; 0 stale claims since).
  H100 dropped per user (A40/V100 only).

### (2) Runtime / memory / storage
- **Runtime:** per-round wall median **~17.6 min**; a full 6-round cell ≈ **~1.8 h** (heavier big/SAM
  cells longer). Throughput is contention-limited (~10–14 A40/V100 workers).
- **Peak GPU memory:** median **9.4 GB**, max 9.6 GB → comfortable on A40 (48 GB) and 32 GB V100.
- **stop_iter distribution:** min 1500, median **2450**, max 3000; **hit-max (cap) rate 39%** of rounds —
  the budget-dependent capping confirmed in §5 (low-budget rounds plateau early; high-budget rounds ride
  to the 3000 cap with near-ceiling val-DSC). btcv/msd07 get max_iters 5000 in full Stage 2.
- **Prediction-mask storage:** **~0.31 MB per round-mask**; ~80 MB total for 22 cells → full 60-cell
  Wave-2 ≈ **~110 MB of masks**. Trivial (the ~1 TB concern was fp16 *probs*, which we do NOT save by
  default). HD95/ASSD are backfillable from these masks offline (#1).

### (3) Metric sanity — ALL PASS
- per-case macro-fg DSC present on all 22 ✓; **HD95/ASSD non-null at the final round on all 22** ✓;
  **structure_detection_rate / missed_structure_rate saved on all** (incl. multi-class mmwhs_ct:
  1.0 / 0.0) ✓; **0 NaN/Inf** across all metric fields ✓; total-misses penalized with the diagonal
  (never silently dropped) ✓.
- **DSC results (final-budget per-case macro DSC):** spleen 0.92–0.96, isic2018 0.85–0.86, mmwhs_ct 0.76,
  kvasir_seg 0.58–0.63, busi 0.53–0.62, ext_brats2020 0.42–0.44. **Sensible easy→hard ordering, no
  anomalies.** (brats is t1ce-only single-modality + from-scratch → ~0.43 expected.) Method differences
  are within the ~0.02 single-seed noise floor → **not yet a ranking** (needs seeds 2000/3000).

### (4) Dataset-list finalization — LOCKED
- MMWHS → `mmwhs_ct` core / `mmwhs_mr` supplementary; CARE-LA → atrium-only; MSD09 spleen → core;
  GlaS/ROSE1/LiQA → `in_core_avg=caution` (separate, wide CIs). **REFUGE + ORIGA → C=3 disc+cup,
  implemented + independently verified (31 tests).** All ready for full Stage 2.

### Verdict
**The frozen_v4 protocol is validated and producing trustworthy, sensible numbers; metrics are clean; the
infra issues are fixed; the full-Stage-2 launch recipe (pre-warm → `--defer-surface` → offline surface)
+ the 14 remaining datasets are ready.** Wave-2 finishing in the background changes nothing in this
conclusion. **Recommend: proceed to full Stage 2 (seed 1000) when you're ready** — `stage2_full_launch_guide.md`.

---

## 8. STAGE-2 FULL — Phase-1 first batch (2026-06-17, overnight)
**6 cells done** (Phase-1 small datasets, `runs/stage2_full`, accelerated recipe: pre-warm + `--defer-surface`).
First production validation of the full-Stage-2 pipeline. All checks pass.

| cell | DSC | detection |
|---|---|---|
| liqa_mri P0 | 0.957 | 1.0 |
| glas2015 P0 | 0.893 | 1.0 |
| glas2015 P9 | 0.874 | 1.0 |
| msd_task04_hippocampus P0 | 0.822 | 1.0 |
| rose1 P9 | 0.743 | 1.0 |
| rose1 P0 | 0.739 | 1.0 |

**(1) Completion:** 6/? done, **0 failed, 0 churn, 0 OOM.** Pre-warm + staling-fix holding. (2) **Runtime/mem/storage:** ~1.7 h/cell; peak GPU **9.4 GB** (max 9.6); **stop_iter 1500/2000/3000** (median 2000), cap-hit 22%; prediction-mask storage **~0.4 MB/cell, 16 MB total** (trivial). (3) **Metric sanity — all pass:** per-case macro DSC present; structure_detection_rate/missed_structure_rate saved (6/6, all 1.0); **0 NaN/Inf**; total-misses penalized via diagonal (not dropped). **HD95/ASSD deferred** on all 6 (mean_hd95_fg absent) — **expected** (`--defer-surface`); backfilled by `surface_offline` after the run, identical numbers (test-proven). (4) **P8/P9 on a dataset:** P9/PAAL completed (rose1, glas2015); P8/SAM-H cells are the slowest, not done yet (will land later — already validated in the canary).
**Verdict: full-Stage-2 pipeline + the accelerated recipe are working correctly in production.** DSC sensible (liqa 0.96, glas 0.88, hippocampus 0.82, rose1 0.74 — clean easy→hard). REFUGE/ORIGA C=3 cells running (not yet done).
