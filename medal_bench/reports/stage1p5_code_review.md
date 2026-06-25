# Stage 1.5 — P0–P9 + Infrastructure Correctness Review

Read-only audit (no code changed) of all AL-relevant code, each method cross-checked against its
source paper / reference implementation. Conducted 2026-06-15 while Stage 1.5 runs. **Decisions to be
discussed before any change.**

## Verdict

**The active-learning science core is sound.** Specifically verified correct (with paper citations
and numeric checks):

- **Case-disjoint splits** — `splits.py:82-100` groups by patient/case id and partitions the *unique
  patient set*; empirically patient-disjoint across train/val/test, deterministic per seed. **No
  patient leakage.** (The #1 validity risk — clean.)
- **Index-space translation** — `al_loop.py:434` `unlabeled_indices` ↔ pool-local mapping is consistent
  in all four uses (firewall, sample_id, pred-cache, labeled.add). **No off-by-one.** (The #2 risk — clean.)
- **Pool/init sharing** — pool & init caches are keyed by (dataset, seed, …), **never policy_id**, so
  every method gets the identical cold-start + candidate pool. Init set excluded from later scoring.
- **Label remap** — `remap.py` hard-errors on unknown codes (never silent→0); MMWHS/BTCV/BRATS LUTs correct.
- **Policy fidelity** — P1 normalized entropy [0,1] ✓; **P2 BALD** MI formula + MC-dropout genuinely
  enabled (10 Dropout2d, T=10, stochastic) ✓; P3 k-center greedy (init=labeled, 2-approx) ✓; **P4 BADGE**
  CE pseudo-label gradient *validated against autograd* + k-means++ D² ✓ (CE-only; P4b correctly ablation);
  P5 entropy→k-center ✓; P6 = Ma et al. ICASSP 2024 selective uncertainty ✓; P7 = clean k-center on SAM
  features ✓; **P9 PAAL** faithful to official `shijun18/PAAL-MedSeg` (AP arch, hard-Dice target incl. bg,
  `log_mean` score, WPS round-robin, cluster count) ✓; P8b correctly flagged ablation ✓.
- **SAM-H features** — correct pixel mean/std, 1024 input, 256-dim neck pooling, frozen/eval/no-grad,
  resolution-keyed cache.
- **Budget grid logic** — monotonic, ≤N, `initial = max(8, 2C, ⌈f·N⌉)` (the *input N* is the issue, see M2).

Issues cluster in four areas: **metric rigor** (C1, M1, M4), **reproducibility robustness** (M3), the
**budget denominator** (M2, already known), and **P8 TypiClust fidelity** (Med1, Med2). None indicate
leakage or a broken AL loop; the existing Stage-1 *DSC* rankings are not invalidated, but the
*HD95/ASD* numbers and cross-method comparison need the metric fixes before Stage 2.

---

## CRITICAL

### C1 — HD95/ASD silently drop total-miss cases → biased primary metric, biased *per method*
`eval.py:88-91, 109-111`. When a class is present in GT but the model predicts it empty, the distance
is `NaN`, counted in `hd95_undefined`, **but excluded from `mean_hd95_fg`/`mean_asd_fg`** (line 109
filters `not isnan`). So the headline HD95 averages only samples where the model already located the
object — the *worst* (total-miss) cases vanish. This is optimistic **and** differentially biased: a
policy whose model produces more empty predictions gets a better HD95. HD95 is the **primary** surface
metric (eval.py:3), so this distorts the method comparison Stage 2 is meant to establish.
*Decision:* pick a documented total-miss convention (image-diagonal penalty, fixed large value, or a
fixed denominator that includes undefined) and always report `hd95_undefined` beside HD95.

---

## MAJOR

### M1 — DSC is micro-averaged (volume-pooled), not per-image; dataset-absent classes dropped
`eval.py:65, 81-82, 97-99`. `inter`/`denom` accumulate over the entire eval set per class → one pooled
(micro) Dice per class, then `nanmean` over fg. Consequences: (a) large slices dominate, per-case
variance is lost — most med-seg papers (and likely the Table-8 protocol) report **per-image/per-case**
mean Dice; (b) a class absent across the whole eval set → `denom=0 → NaN`, silently excluded. Note this
"Dice" also differs from P9's training target convention (`_paal_ap.hard_dice_per_class` defines
absent-in-both = 1.0). Changes headline DSC and cross-paper comparability.
*Decision:* choose macro (per-image) vs micro and an explicit empty-class convention; align with the
Table-8 reference protocol; apply uniformly.

### M2 — Budget denominator: `run_one` feeds N=5000 to the grid but the true pool is 2500 (KNOWN; now located)
`run_one.py:90-92` sets `pool_size = min(len(train), pool_cap)=5000`, but `_stratified_pool_cap`
(`al_loop.py:263-269`) under-fills to **2500** for mmwhs/msd07/brats (no bg-only slices to fill the 50%
bg quota). So `budget_grid` and every logged `labeled_ratio` (`al_loop.py:597`) / fraction are ~2× off
for the capped datasets. **`run_full_supervised.py:76-82` already fixes this** (derives the true size via
`_load_or_make_pool_indices`) — the main AL runner was not given the same fix. This is the Stage-1.5C
budget-denominator item. *Decision:* derive `pool_size` from the actual pool builder in `run_one`; assert
`budget_plan[-1] <= len(pool_subset)`.

### M3 — Per-round model-init/dropout reproducibility relies on carried-over global torch RNG
`al_loop.py:347` calls `seed_all` once; `trainer.py:98` seeds only a NumPy `RandomState` for batch
sampling — **torch weight init (each round rebuilds the model) and training dropout masks draw from the
leftover global torch RNG**. Same-seed runs are reproducible *only* if the entire op sequence is byte-
identical (true today → determinism tests pass), but any code-path change shifts every later round's
weights, and the `seed=cfg.seed+r` arg gives a false impression of per-round seeding.
*Decision:* `seed_all(cfg.seed + r)` (or `torch.manual_seed`) at the top of each round before build+train.
Low-risk, makes the 3-seed benchmark robust.

### M4 — ASD is the directed surface distance, not symmetric ASSD
`eval.py:21, 34`. `medpy.metric.binary.asd(p,g)` is one-directional (pred→gt); the standard reported
metric is symmetric ASSD. HD95 is fine (medpy hd95 is symmetric internally). ASD is the *secondary*
metric, so lower impact, but it's nonstandard/mislabeled. *Decision:* use `assd`, or rename to
"directed ASD (pred→gt)" and disclose.

---

## MEDIUM — P8 TypiClust fidelity (vs Hacohen et al., ICML 2022 + ref impl)

### Med1 — P8 omits the `MIN_CLUSTER_SIZE` (>5) filter → can query isolated/size-1 clusters
`p8_sam_typiclust.py:95`. Canonical TypiClust drops clusters with ≤5 members before selection; P8 keeps
every non-empty cluster, so it can pick the lone point of a size-1 cluster (typicality hard-set to 0.0)
— an outlier, the opposite of "typical." *Decision:* add the min-cluster-size filter (keep a fallback
for degenerate tiny pools).

### Med2 — P8 selection is single-pass one-per-cluster + global fallback, not round-robin
`p8_sam_typiclust.py:101-120`. Canonical cycles clusters `i % n_clusters`, taking the next-most-typical
from each in priority order; P8 takes ≤1 per cluster then fills the remainder by *global* typicality.
Differs when #non-empty clusters < budget (e.g. duplicate-collapsed k-means). *Decision:* replace the
fallback with round-robin re-cycling over the sorted cluster order.

(Both only bite when k-means yields empty/tiny clusters, but they make P8 deviate from the paper exactly
in the low-budget regime TypiClust targets.)

---

## MINOR / robustness (no impact on current results unless noted)

- **Loss design** `trainer.py:69,76-81` — CE includes background **and padded pixels** (no ignore_index),
  1:1 CE:Dice weight, and the Dice term charges the *full* penalty for a class absent in the batch. With
  sparse FG this lets bg-CE swamp Dice and contributes to the multiclass/msd07 collapse. Ties into the
  frozen_v3 loss/iters discussion (not a standalone bug). *(Overlaps known padding + undertraining items.)*
- **BADGE k=0 crash latent** `_helpers.py:235-239` — `kmeans_plusplus` raises on k=0/k>pool; P4/P4b lack
  the graceful `return []` the k-center helper has. Can't trigger in the normal increasing-budget path,
  but is a robustness gap. *Direction:* early `if k<=0: return []`.
- **SAM cache key** `sam.py:203-210` — validation omits the checkpoint identity and uses width-only for
  input size; benign (checkpoint frozen, square inputs) but technically incomplete.
- **P8 K-cap** `p8_sam_typiclust.py:61` — uses `min(20, len-1)` vs paper's `min(20, len//2)`; minor.
- **P9 minor deviations from official** `p9_paal.py:296,117-122,298` — WPS L2-normalizes features before
  KMeans (official uses raw), carves a small AP-val split for a diagnostic, KMeans `random_state` differs.
  All documented/seeded; for exact parity drop the WPS L2-norm + train AP on full labeled set.
- **CoreSet/feature L2-normalization** `p3_coreset.py` (and P5/P7) — normalized features vs raw L2 in
  Sener-Savarese; **documented deliberate choice**, consistent across the coreset family.
- **BADGE gradient sign** `_badge_grad.py:82` — `(p−onehot)` vs ref `(onehot−p)`; sign-invariant under
  k-means++ (verified), harmless.
- **softmax-before-argmax** `eval.py:78`, `prediction_cache.py:50` — redundant (monotonic), harmless.
- **msd07 axial axis hardcoded** `msd07_pancreas.py:54,73` — `transpose(2,0,1)` not affine-derived;
  verified safe for all 281 current volumes (axis-2 = S), but unguarded vs future re-oriented data.
- **`_IndexedSubset.patient_ids()`** `al_loop.py:192-193` — returns None for the whole subset if sample[0]
  has falsy patient_id; latent (splitting happens on the raw adapter, not the subset), but fragile.
- **P6 `score()` log value** `p6_selective_uncertainty.py:116` — logged `selected_scores` don't drive the
  (interleaved) selection; cosmetic logging-fidelity nit.

---

## Already-known (confirmed, tracked elsewhere)
- Padded letterbox pixels included in loss / DSC / HD95 / full-canvas query scores (verification §4).
- 250-iter undertraining (verification §7 / diagnostics) — Stage 1.5A is measuring the fix.

---

## Proposed discussion agenda (no changes made yet)

**A. Metric correctness — fix before Stage 2 (these bias the comparison):**
  C1 HD95/ASD total-miss convention · M1 DSC per-image-macro + empty-class convention · M4 symmetric ASSD.
  → These redefine reported numbers, so they belong in the **frozen_v3 metric spec** and ideally a
  re-eval of the Stage-1/1.5 trajectories (metrics are recomputable from saved preds? — note: preds are
  NOT saved, so a metric change requires re-running, or recomputing from cached logits if we add that).

**B. frozen_v3 (already in scope):** M2 budget denominator (use actual AL_pool_N) · loss/iters (Minor loss
  item + Stage 1.5A result) · valid-region aggregation (verification §4).

**C. Method fidelity (decide if we want paper-faithful P8 in Stage 2):** Med1 + Med2 TypiClust.

**D. Robustness (low-risk, do anytime):** M3 per-round seeding · BADGE k=0 guard · SAM cache key · msd07
  affine axis · `_IndexedSubset.patient_ids`.

**E. No action / document-only:** BADGE sign, CoreSet L2-norm, P9 minor deviations, softmax-argmax, P6 log.

Open question for A: a metric-definition change means the Stage-1 seed-1000 HD95/ASD (and possibly DSC)
numbers in the existing reports would change. Since model predictions aren't persisted, this needs either
re-running eval or adding a one-off logits dump. Worth deciding before Stage 2 so all seeds use one metric.
