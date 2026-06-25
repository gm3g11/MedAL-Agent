# Stage 1 — Seed 1000 Diagnostics Addendum

Companion to `stage1_seed1000_summary.md`. Answers the 5 requested checks. Single seed; treat as
indicative. Sources: existing trajectory JSONL, per-round `candidate_scores/`, and the 512-preproc
mask cache (remapped labels) — **no model re-run** for Checks 2 & 4. Checks 1 & 3 use fresh baseline
runs launched 2026-06-14 19:39 (status noted inline).

> **frozen_v2 is NOT changed.** These are diagnostics to inform the A/B/C/D decision; one seed is not
> enough to re-freeze the config.

---

## Check 1 — Full-supervised sanity  *(baselines re-running; busi done)*

The earlier full-sup outputs were **stale** (run 10:11, before the resolution/letterbox/determinism
fixes) — the old `busi` baseline scored `0.0003` on an `80`-sample pool, which is the buggy old
pipeline, not a real number. All 9 are re-running with current `bench512` code. Protocol: full-sup
trains on the **fg-stratified train pool capped at 5000** (`budget_plan=[pool_size]`), i.e. the *same
pool the AL methods select from* — so for datasets with <5000 train slices it is the full train set;
for mmwhs/msd07/brats it is the (fg-stratified) capped pool. Same backbone/resolution/250-iter as AL.

> **Bug found & fixed in `run_full_supervised.py`.** It set `budget_plan=[min(len(train),5000)]=5000`
> for the big datasets, but the fg-stratified cap actually yields **2500** slices (1:1 fg:bg balance),
> so the full-sup runs crashed with `initial set size 2500 != budget_plan[0]=5000`. Fixed to take the
> true pool size from the actual pool builder (`_load_or_make_pool_indices`). This bug only ever hit
> full-sup (AL never sets `budget_plan[0]` above the pool), so **no AL/Stage-1 result is affected.**
> mmwhs/msd07/brats full-sup re-launched with the fix.

| dataset | full-sup DSC | per-class DSC | pool used | AL best (final) | 250-iter undertrained? |
|---|---|---|---|---|---|
| busi | **0.475** | [bg 0.965, lesion 0.475] | full train (624 < 5000 cap) | 0.483 (P5) | **Yes** — full-sup(624) ≈ AL(~70). 10× data, ~0 gain. |
| kvasir_seg | **0.479** | [bg .895, fg .479] | full train (800) | 0.502 (P8) | **Yes** — full-sup < AL best |
| isic2018 | **0.825** | [bg .962, fg .825] | full train (2076) | 0.839 (P5/P9) | **Yes** — 0.825 vs 0.839 (Δ<0.03) |
| glas2015 | **0.899** | [bg .946, fg .899] | full train (133) | 0.894 (P0) | **Yes** — full-sup ≈ AL best (saturated) |
| origa | **0.934** | [bg .999, fg .934] | full train (520) | 0.943 (P8) | **Yes** — 0.934 vs 0.943 (Δ<0.03) |
| mmwhs | **0.216** | [bg .965, .423, .14, .515, .001, .059, .358, .017] | 2500 (fg-strat cap) | 0.244 (P9) | **Yes** — full-sup *below* AL best; 4/7 fg classes ≈0 |
| btcv_synapse | **0.059** | [bg .977, liver .77, all other 12 organs ≈0] | full train (1494) | 0.093 (P1) | **Yes** — full-sup *below* AL best; only liver learns |
| msd_task07_pancreas | **0.075** | [bg .989, .151, .0] | 2500 (fg-strat cap) | 0.188 (P9) | **Yes** — near-collapse @250 even on 100% data |
| ext_brats2020 | **0.328** | [bg .984, NCR .24, edema .183, ET .559] | 2500 (fg-strat cap) | 0.382 (P3) | **Yes** — full-sup *below* AL best |

**busi readout:** full supervision (624 imgs) reaches **0.475**, essentially equal to AL at ~70 imgs
(0.48). When the 100%-data upper bound is no better than a tiny labeled fraction, the binding
constraint is the **optimizer budget (250 iters), not labels.** This is the headline undertraining
signal; the rest of the table will confirm whether it generalizes. (Table fills as jobs land.)

---

## Check 2 — Multiclass diagnosis (btcv_synapse, mmwhs, ext_brats2020)

**Headline: the multiclass weakness is "never *learned*," not "never *selected*."** Every fg class is
present in the pool, in the *initial* labeled set, and in every method's final set. The model simply
fails to fit most classes in 250 iters, and on btcv it actively **regresses as budget grows**.

### Class availability (rules out a selection/coverage cause)
- **No rare classes.** Min pool frequency: btcv pancreas 14.4% / gallbladder 15.2% / Radrenal 16.4%;
  mmwhs all 47–67%; brats all 58–99%. None below 5%.
- **Initial labeled set already covers ALL classes** — btcv 28 slices → all 13 fg; mmwhs 16 → all 7
  fg; brats 13 → all 3 fg.
- **Final labeled set covers all fg classes for every method** (P0–P9). So "rare classes never
  selected" is **false** — coverage is complete from round 0.

### Per-class DSC by budget — the actual failure
**btcv_synapse (P0 Random):** only **liver** (~0.79) and **spleen** (sporadic ~0.3) ever learn. Right/
left kidney, gallbladder, esophagus, aorta, IVC, portal vein, pancreas, both adrenals = **0.000 at
every budget**. And it **gets worse with data**: spleen 0.288→0.006, stomach 0.264→0.000; val
mean-DSC 0.101→0.061 while mean train loss **rises** 1.67→1.75. Best method (P1) learns kidneys
transiently (R/L kidney 0.57/0.47 at budget 75) then loses them — unstable, not progressing.

**mmwhs (P0):** classes learn partially but **oscillate** (LVblood 0.43→0.59, others flip between
~0.4 and 0.000 across budgets); val DSC **flat ~0.22** across all budgets; loss rises 1.24→1.69. No
budget trend. Best method P9 = 0.244, same flat/unstable pattern.

**ext_brats2020 (P0):** the *least* broken — 3 classes, well-balanced (edema in 98.8% of slices). ET
~0.5–0.6, edema ~0.2–0.4, NCR 0→0.27. Shows mild budget improvement (val DSC 0.22→0.37 peak). Best
method P3 = 0.382.

### Diagnosis
The signature — **flat/declining val DSC vs. budget, flat/rising train loss, collapse onto the 1–2
dominant classes (bg/liver/LVblood)** — is **underfitting + class-imbalance collapse**, not a data or
selection problem. 250 iters at batch 12 is too few to fit a 14-/8-class dense problem; adding more
multi-organ slices worsens the imbalance the under-trained model can't absorb. Severity scales with
class count (btcv 14 ≫ mmwhs 8 ≫ brats 4 / binary datasets fine). **This points to longer training
for multiclass (option B/C), not a selection or frozen-resolution change.**

*(Caveat: `selected_ids` over-logs for coreset-family policies on large pools — see Logging findings.
This does not affect coverage conclusions, which only need "≥1 occurrence per class.")*

---

## Check 3 — msd07 longer-training probe  *(running)*

Per request, msd07 is labeled **"hard-task collapse under the current Stage 1 slice-level config,"**
not Tier-C. Probe launched: `P0 Random @ 1000 iters` and `@ 2000 iters` (separate out-dirs), plus the
msd07 full-supervised baseline (@250, in Check 1). Results table fills when they land:

| run | iters | DSC_fg | per-class | verdict |
|---|---|---|---|---|
| msd07 P0 Random (Stage 1) | 250 | 0.000 | — | collapsed |
| msd07 P0 Random probe | 1000 | **0.4286** | [bg .998, .589, .268] | **RECOVERS** |
| msd07 P0 Random probe | 2000 | **0.4009** | [bg .998, .583, .219] | RECOVERS (plateaus ~0.40–0.43) |
| msd07 full-supervised | 250 | 0.075 | [bg .989, .151, .0] | near-collapse @250 |

**Verdict: NOT intrinsic collapse — it is undertraining.** P0 Random jumps **0.000 → 0.4286** purely by
training 1000 instead of 250 iters (same data, same slice-level config). The @250 full-sup near-collapse
(0.075) was also just too few steps for the extreme-FG-sparsity pancreas task. This **reverses the
earlier Tier-C/option-D lean**: msd07 should **stay in the benchmark**; the real lever is training
length (shared with the multiclass undertraining) → **supports option B/C, not D.**

---

## Check 4 — ISIC P1 (Normalized Entropy) diagnosis

ISIC `selected_ids` are **clean for all policies** (sum=unique=395), so this analysis is exact.

- **P1 ranks by entropy correctly.** Selected-slice mean entropy far exceeds pool mean every round
  (r0 0.197 vs 0.077; r2 0.412 vs 0.241; r4 0.372 vs 0.296). It is doing what it claims.
- **It is NOT picking artifacts/empty masks.** 0/395 selected slices are near-empty (<50 px FG).
- **It over-selects LARGE lesions.** P1-selected median FG = 32,797 px vs pool median 24,310; **82/395
  selected are in the pool's top-10% largest** (>96,756 px). Plain normalized entropy rewards the big,
  heterogeneous, ambiguous-boundary lesions.
- **Its set diverges from the others.** Jaccard vs P0 0.121, P4 0.155, P8 0.122 — only ~85–106 of 395
  shared. P1 specializes in a distinct, hard subpopulation.

**Diagnosis:** P1's −0.080 vs Random is the classic *uncertainty-sampling-over-focuses-on-hard-
outliers* failure: by loading the training set with large, ambiguous lesions it under-represents the
typical easy lesions that dominate the val set, lowering **mean** DSC. The **letterbox-padding-inflates-
entropy** hypothesis is *not* supported — if padding drove the score, P1 would favor small/empty
slices, but it favors the largest ones. (A definitive valid-region-vs-full-canvas entropy split is not
recoverable offline — `candidate_scores` are pre-aggregated scalars — and would need a re-run logging
both maskings; the lesion-size evidence already argues against the padding artifact.)

---

## Logging findings (do not affect DSC; fix before skill-learning export)

1. **`selected_ids` over-logged for coreset-family policies on large pools.** On brats/mmwhs, P3 logs
   exactly 2× the true selections; P5/P9 log partial duplicates. `labeled_count` and training are
   **correct** (always = budget), so all DSC/HD95 results are valid and comparable. But the per-round
   *selection set* is unreliable for P3/P5/P9 (and should be checked for P7/P8) on the 5000-pool
   datasets. The **Query-Strategy skill needs accurate per-round selections**, so this must be fixed
   (dedup / log only the newly-committed ids) before Stage 3 export. Clean on all small-pool datasets
   (e.g. isic) and for non-coreset policies.
2. **Per-round artifacts available for free:** `candidate_scores/<run>__r<k>.json` (all candidate
   scores), `init_sets/`, `pool_sets/` — useful for the skills; keep them.

---

## Check 5 — Decision (pending Checks 1 & 3 completion)

frozen_v2 unchanged for now. Provisional read from current evidence:

- **Binary datasets are healthy** (busi/kvasir/isic/glas/origa: sensible DSC, AL≈/≥Random) → keep
  as-is.
- **Multiclass under-fits at 250 iters** (Check 2) and **busi full-sup shows 250 iters is saturated**
  (Check 1) → the leading remedy is **(C) a longer-training sensitivity for the multiclass datasets**
  (btcv/mmwhs/brats, and possibly all) on seed 1000 *before* committing seeds 2000/3000 — cheaper than
  re-running the whole matrix and directly tests whether method rankings change once the model can fit.
- **msd07**: probe now in — **NOT option D.** It recovers 0.000→0.4286 at 1000 iters, so it is
  undertraining, not intrinsic collapse → fold into the same iters fix (B/C), keep in the benchmark.

**Updated recommendation:** the unifying finding across Checks 1–3 is that **250 iters is undertrained**
(binary saturated, multiclass underfit, msd07 collapsed-then-recovered). Run **(C)** a seed-1000 longer-
training sensitivity — 1000 iters on the multiclass + msd07 (binary likely benefits too) — confirm
whether DSC and method ordering move, then decide **(B)** a re-freeze with higher iters vs **(A)**
as-is. **Do not retire msd07 (D).** Holding — **not** launching seeds 2000/3000, **not** changing
frozen_v2.
