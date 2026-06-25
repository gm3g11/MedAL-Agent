# Stage 1.5 Report — Iteration Sensitivity, Full-Sup Adequacy, Valid-Region, Budget → frozen_v3

Seed 1000. Finalized 2026-06-15. Stage 1.5 is **COMPLETE** for all decision-relevant analyses.

## §0 Coverage (FINAL)
- **it250** (= Stage-1 runs): 42/42 ✓ (all 6 datasets × 7 methods P0,P1,P3,P4,P5,P8,P9).
- **it1000**: **42/42 ✓** — the decision-critical 250-vs-1000 comparison is complete.
- **it500**: **40/42** — ext_brats2020 P4 & P8 @500 excluded (the cluster module-load system hung every
  fresh job at startup this morning; ~2 retries each on A40+V100 all stuck at "Loading CRC_default").
  These are the **intermediate** 500-iter point for one dataset and **do NOT affect any conclusion**
  (the 250↔1000 result and brats's 250/1000 values are complete). The 2 cells stay queued to backfill.
- **full-sup**: 12/12 ✓ (250/500/1000; plus the 9-dataset 250-iter set from Stage 1).
- **Total AL: 82/84** (it1000 42/42 + it500 40/42).

---

## §1 Bucket A — applied & verified (no metric/behavior change)
1. **GPU name + total memory** logged per cell (`gpu_name`, `gpu_total_mem_mb` in TrajectoryRecord).
2. **Preflight guard**: refuses P9 on <24 GB, SAM-H/P7/P8 on <22 GB (fail-fast; no-op on 32/48 GB GPUs).
3. **BraTS modality** logged as `t1ce` (was the misleading `multi_modal_mri`).
4. **`read_trajectory_deduped()`** export helper (collapses duplicate rounds + dedups selected_ids).
All verified (parse, fields present, dedup 12→6 rounds DSC-identical, modality='t1ce'); none touches the
training/selection/metric path.

---

## §2 Iteration sensitivity (THE headline) — final-budget DSC at 250 / 500 / 1000 iters

| dataset | P0 | P1 | P3 | P4 | P5 | P8 | P9 |
|---|---|---|---|---|---|---|---|
| busi | .477/.521/.527 | .463/.466/.525 | .481/.555/.530 | .408/.523/.549 | .483/.437/.571 | .448/.540/.516 | .419/.491/.532 |
| isic2018 | .830/.838/.871 | .750/.835/.867 | .837/.851/.845 | .822/.843/.871 | .839/.860/.874 | .826/.849/.866 | .839/.853/.846 |
| mmwhs | .224/.300/.557 | .232/.308/.484 | .212/.307/.444 | .209/.335/.507 | .140/.376/.531 | .187/.328/.510 | .244/.377/.535 |
| btcv_synapse | .061/.119/.367 | .093/.242/.493 | .060/.134/.333 | .056/.134/.446 | .072/.168/.424 | .058/.116/.354 | .083/.235/.321 |
| ext_brats2020 | .319/.411/.457 | .303/.410/.467 | .382/.365/.393 | .363/**–**/.435 | .352/.389/.469 | .322/**–**/.362 | .341/.355/.405 |
| msd_task07_pancreas | .000/.225/.451 | .005/.223/.269 | .000/.240/.310 | .000/.238/.338 | .072/.248/.288 | .179/.250/.305 | .188/.268/.322 |

(All 500-iter cells filled except ext_brats2020 P4/P8 — **–** = excluded, cluster startup-hang; see §0.)

**Mean DSC 250 → 1000 (full it1000, avg over the 7 methods):**
busi +18% (.45→.54) · isic +5% (.82→.86) · **mmwhs +146% (.21→.51)** · **btcv +467% (.07→.39)** ·
brats +26% (.34→.43) · **msd07 +417% (.06→.33)**.

**Monotonicity (250 → 500 → 1000, mean over methods present at all three):** strictly increasing on every
dataset — busi .454→.505→.536 · isic .820→.847→.863 · mmwhs .207→.333→.510 · btcv .069→.164→.391 ·
brats .339→.386→.438 · msd07 .063→.242→.326. The 500-iter point sits cleanly between, so the gains are
**training adequacy**, not a 1000-iter artifact.

**Does DSC improve with iters?** YES, dramatically — and most for the undertrained datasets (multiclass
mmwhs/btcv, sparse-FG msd07). msd07's "collapse" (0.00 @250) fully resolves to ~0.27–0.45 @1000 — it
was **undertraining, not an intrinsic failure**.

**Do method RANKINGS change 250 → 1000?** YES — decisively, now confirmed on the COMPLETE it1000 set.
Spearman of the 7-method DSC ordering, 250 vs 1000 (all n=7): busi **+0.11**, isic **+0.04**, mmwhs
**+0.11**, btcv **+0.07**, brats **−0.29**, msd07 **−0.41** (mean ≈ **−0.06**). I.e. **the 250-iter
rankings are essentially uncorrelated (even anti-correlated) with the 1000-iter rankings on every
dataset.** Example flips: at 1000 iters **Random (P0) is the top method on mmwhs (.557) and msd07
(.451)** — vs P9 winning at 250; P5 (Entropy→CoreSet) tops busi/isic/brats; btcv best flips to P1.
**⟹ the Stage-1 (250-iter) method comparison is an undertraining artifact that does NOT survive more
training and would mislead Stage 2.** This is the central, now-final justification for raising iters
before scaling.

---

## §3 Full-supervision adequacy — full-sup DSC at 250 / 500 / 1000 (capped AL pool)

| dataset | 250 | 500 | 1000 | AL-best@250 | undertrained@250? | plateaued@1000? |
|---|---|---|---|---|---|---|
| busi | 0.475 | 0.555 | 0.584 | 0.483 | yes (≈AL) | ~ (slowing) |
| isic2018 | 0.825 | 0.859 | 0.872 | 0.839 | yes (≈AL) | yes |
| mmwhs | 0.216 | 0.431 | 0.575 | 0.244 | **severe** (<AL) | not yet (rising) |
| btcv_synapse | 0.059 | 0.112 | 0.384 | 0.093 | **severe** (<AL) | **NO — still climbing steeply** |
| ext_brats2020 | 0.328 | 0.403 | 0.455 | 0.382 | yes | ~ |
| msd_task07_pancreas | 0.075 | 0.242 | 0.347 | 0.188 | **severe** | ~ (msd07 P0 probe plateaued .43@1k/.40@2k) |

**At 250 iters every dataset is undertrained** (full-sup ≈ or < low-budget AL). At 1000, full-sup
recovers strongly. **btcv (14-class) is still climbing at 1000** (0.06→0.11→0.38) — likely still
underfit at 1000; may warrant 2000 (see §6). msd07's longer-iter probe plateaued ~0.40–0.43 (1000–2000),
so 1000 is near its ceiling.

---

## §4 Valid-region query-score analysis
Full-canvas uncertainty scores are correlated with the empty/letterbox-pad fraction (Spearman: isic
**−0.547**, brats **+0.486**, mmwhs −0.126), i.e. selection is partly driven by image geometry, not pure
uncertainty (P1/P2/P5 use full aggregation; P6 is fg/boundary-masked and robust). A forward-pass
full-canvas-vs-valid-region prototype on a trained model is the remaining confirmation step (deferred —
runs during frozen_v3 dev). Recommendation stands: aggregate P1/P2/P5 scores over the valid region.

---

## §5 Corrected budget table
| dataset | full_train_N | requested_cap | actual_AL_pool_N | fg/bg quota | cumulative counts | incremental |
|---|---|---|---|---|---|---|
| busi | 624 | 5000 | 624 | all fg | 8,13,32,63,94,125 | 8,5,19,31,31,31 |
| kvasir_seg | 800 | 5000 | 800 | all fg | 8,16,40,80,120,160 | 8,8,24,40,40,40 |
| isic2018 | 2076 | 5000 | 2076 | all fg | 21,42,104,208,312,416 | 21,21,62,104,104,104 |
| glas2015 | 133 | 5000 | 133 | all fg | 8,10,20,26 | 8,2,10,6 |
| origa | 520 | 5000 | 520 | all fg | 8,11,26,52,78,104 | 8,3,15,26,26,26 |
| mmwhs | ≥5000 | 5000 | **2500** | 2500 fg / 0 bg | 16,25,50,100,250,500 | 16,9,25,50,150,250 |
| btcv_synapse | 1494 | 5000 | 1494 | all fg | 28,30,75,150,225,299 | 28,2,45,75,75,74 |
| msd_task07_pancreas | ≥5000 | 5000 | **2500** | 2500 fg / 0 bg | 13,25,50,100,250,500 | 13,12,25,50,150,250 |
| ext_brats2020 | ≥5000 | 5000 | **2500** | 2500 fg / 0 bg | 13,25,50,100,250,500 | 13,12,25,50,150,250 |

**Why 5000→2500:** `_stratified_pool_cap` targets 50% fg / 50% bg, but after slicing 3D volumes
*every retained slice contains foreground* (fg-slice ratio = 1.000), so the bg quota can't be filled and
the pool caps at the 2500 fg half. The budget grid was built on N=min(train,5000)=5000 (Case C), so the
realized max budget is 20% of the true 2500 pool, not 10%. **Fix:** budget grid + logged fractions must
use the true post-cap pool N (frozen_v3 / `run_one.py`, mirroring the `run_full_supervised.py` fix).

---

## §6 Recommended frozen_v3

**num_iters — RAISE; the data demands it.** 250 is severely undertrained and its method rankings are
non-predictive (§2 Spearman ≈0). Recommendation: **1000 iters as the global default.** Even binary
gains modestly and their rankings also reshuffle, so a uniform schedule is cleanest for comparability.
Caveat: **btcv (14-class) is still climbing at 1000** — run a btcv-only 2000-iter check; if it keeps
rising, either use 2000 for the multiclass family or accept 1000 with a documented note. (Family-specific
{binary 250–500, multiclass/hard 1000–2000} is an option but uniform 1000 is simpler and safer.)

**Metric layer (from the code review — C1/M1/M4):** adopt **per-case (per-volume) metrics** — DSC +
HD95 + symmetric **ASSD** per case, macro over cases & fg classes; **diagonal penalty + detection rate**
for total-miss (no silent NaN-drop). Replaces per-slice micro-DSC + directed ASD.

**Valid-region aggregation (§4):** aggregate P1/P2/P5 query scores over the valid (un-padded) region
(pending the forward-pass prototype confirmation).

**Budget denominator (§5):** use the true post-cap pool N.

**Other code-review items:** M2 (run_one pool-size fix), M3 (per-round `seed_all(seed+r)`), P8 TypiClust
fidelity (MIN_CLUSTER_SIZE + round-robin), lease-race fix (atomic per-PID partial / longer stale TTL),
**dispatch subprocess timeout/watchdog** (the overnight wedging root cause — no per-cell timeout), and
the `selected_ids` dedup for skill export.

**Schedules global vs family-specific:** recommend **global** (num_iters=1000, per-case metrics,
valid-region agg, true-pool budget) for comparability; only btcv may need a 2000-iter exception.

---

## §7 Stage-2 go / no-go
**NO-GO until frozen_v3 is implemented and reviewed.** Stage 1.5 proved that the Stage-1 (250-iter)
configuration produces undertrained models whose method rankings do not survive more training — running
the full 3-seed Stage-2 matrix at 250 iters would yield untrustworthy comparisons. **Go path:** implement
frozen_v3 (1000 iters [btcv 2000 TBD] + per-case metrics + valid-region agg + true-pool budget +
robustness fixes) → small v3 validation smoke (1–2 datasets × few methods × 1 seed, correctness/sanity)
→ then Stage-2 (full 3-seed P0–P9 under v3, saving val logits). Do not re-run the full Stage 1.5 under v3
(scaffolding only). Holding for user review of frozen_v3.
