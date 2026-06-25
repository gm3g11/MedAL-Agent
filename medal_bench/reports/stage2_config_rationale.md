# Stage 2 config + today's experiment summary (for review)

Date 2026-06-15. Reviews the v3 canary + iters-probe + adaptive findings and the resulting
**Stage-2 / frozen_v4 configuration** (`profiles.bench512_v4`). Read before launching Stage 2.

---

## Part 1 — What ran today, and what we found

### A. frozen_v3 implemented + validated (canary)
- All 10 frozen_v3 amendments implemented; **173 tests pass** (was 102).
- **Canary** (25/30 cells, 5 datasets × 6 methods, seed 1000, bench512 @1000it) validated the v3 protocol:
  per-case macro-fg DSC primary metric ✓; diagonal total-miss penalty + detection rate ✓ (btcv: 12
  total-missed organs penalized, detection 0.69); P8 TypiClust fidelity ✓ (min-cluster filter active);
  prediction saving ✓; **P0 no longer collapses on msd07** (0.338 vs 0.00 @250it) ✓.

### B. The iters-sensitivity probe — THE headline finding
Question: is the fixed-1000-iter regime fair across methods? Ran {isic, msd07, btcv} @ **2000** iters,
compared to the canary @1000.

**Fixed-iter confound CONFIRMED.** On isic (all methods, Δ DSC 1000→2000):

| method | type | Δ@2000 |
|---|---|---|
| P0 Random | diversity | +0.015 |
| P8 TypiClust | diversity | +0.010 |
| P1 entropy | uncertainty | +0.048 |
| P4 BADGE | gradient | +0.066 |
| **P9 PAAL** | hardest-data selection | **+0.089** |

**Monotonic:** the harder a method selects, the more it was under-fit at 1000 and the more it recovers at
2000. Diversity methods (Random, TypiClust) barely move; uncertainty methods recover ∝ selection
difficulty; PAAL closes ~75% of its gap to Random. Also **Random itself is under-fit on hard datasets**
(Δ: btcv +0.195, msd07 +0.047, isic +0.015) — no single fixed iter count fits all datasets.

**Implication:** the "Random ≥ uncertainty methods at 1000 iters" pattern (Stage 1.5 + canary) is
**substantially a training-adequacy artifact, not a true method ranking.** A fixed iter budget biases the
whole benchmark toward diversity/random methods. → motivates adaptive iters.

### C. Adaptive-iters fix
- Replace fixed iters with **adaptive train-to-plateau** (per (dataset, method), no tuning) → removes the
  confound at scale; can't be done with any single fixed count.
- Criterion bug found + fixed: a *relative* train-loss plateau never triggers near 0 loss (memorization →
  0.050→0.0495 is "1% better" forever) → switched to an **absolute** loss-delta. Verified (toy plateaus at
  1350 iters vs previously hitting the cap). A 6-cell adaptive validation run is confirming the stop-iters
  are sensible (isic low, btcv high) and that the Random-vs-PAAL gap closes.

### D. Orchestration lessons
Slow final rounds (per-case surface metrics ~17 min + 1.5 GB fp16-prob writes) false-tripped the 20-min
stale threshold → duplicate runs. Fixed for Stage 2 (below).

---

## Part 2 — Stage-2 config (`bench512_v4` / frozen_v4) and WHY

| Knob | Stage-2 setting | Reason |
|---|---|---|
| **Training** | **Adaptive train-to-plateau** — abs loss-delta 0.003, min_iters 500, max_iters 3000 | The probe proved fixed iters differentially under-train difficulty-based methods → biased ranking. Adaptive trains each (dataset, method) to its own convergence → removes the bias **per-dataset without tuning**, and is *cheaper* than a fixed-high budget (easy datasets stop early). |
| **Metric** | per-case macro-fg DSC (v3) + **surface (HD95/ASSD) at FINAL round only** | Surface is ~17 min/cell on 14-class datasets; computing it at first/mid/final triples that for ~no extra signal. Final-budget surface is the headline anyway. |
| **Storage** | masks + ids + valid_bbox **always**; **fp16 probs for a debug subset only** | All-cell prob saving ≈ **0.8–1.2 TB** for 1-seed Stage 2 (measured ~60 GB/18 canary cells; mmwhs 1.5 GB/round). Masks round-trip to any region/detection/overlap metric; probs are only needed for calibration spot-checks → save them for a small headline/debug subset. |
| **Seeds** | **1 seed (1000) first** (your call) | 240 cells vs 720. Caveat: 1 seed gives *indicative* rankings against a ~0.02 GPU-nondeterminism noise floor, **not statistically robust** — add 2000/3000 later before any "method X wins" claim. |
| **Backbone** | from-scratch U-Net (unchanged) | Method-agnostic fairness — every method shares the identical backbone. PAAL's paper ResNet-50 + Incremental-Querying is deliberately *out of scope* (giving it to PAAL alone breaks cross-method comparability). |
| **Resolution** | 512 aspect-preserving letterbox (unchanged) | Resolution affects all methods *equally* (no fairness confound, unlike iters) → a fixed value is fine; 512 is a sound default. |
| **Orchestration** | STALE_CLAIM_SEC 1200→**3600** + done-by-round-count + per-PID partials | Canary surfaced false-steals from slow final rounds; fixed. (A finer within-cell watchdog is recommended but not blocking.) |
| **Dataset scope** | 20-Core, seed 1000, in waves | Per `stage2_dataset_list.md` (already locked). |

---

## Part 3 — Outstanding before launch
1. **Adaptive sanity confirming** — isic stop-iter sensible (plateaus low, not the cap) + gap closes.
   First signal ~30–45 min. If isic still hits the cap, bump `plateau_min_delta` to ~0.005–0.01.
2. **btcv-family max_iters** — btcv was +0.195 at 2000 and may still climb; the adaptive run shows whether
   3000 is enough or the 14-class family needs a higher cap.
3. `frozen_v4.py` freeze record — finalize once the caps above are confirmed.
4. **Launch:** bulk `dispatch --profile bench512_v4 --save-predictions` (masks always); a separate small
   debug-subset dispatch adds `--save-logits`.

**Bottom line:** the config is `bench512_v4` = frozen_v3 + (adaptive iters, surface-final-only,
debug-subset probs). The one thing gating launch is confirming the adaptive criterion plateaus sensibly on
real data (~30–45 min); everything else is settled and implemented.
