# Stage 1 — Seed-1000 Pre-Scaling Verification (7 checks)

Requested before launching seeds 2000/3000. The 90-cell seed-1000 wave is **complete (90/90, 0
failures)**. This report answers the 7 verification checks. Companion: `stage1_seed1000_summary.md`
(results) and `stage1_seed1000_diagnostics.md` (training/multiclass/ISIC deep-dives).

**Bottom line:** the AL mechanics are sound (budgets, shared pools, splits, determinism), but three
issues should be decided before scaling to 3 seeds: **(a) 250-iter undertraining** (now confirmed to
even *collapse* msd07, recoverable at 1000 iters), **(b) full-canvas score aggregation is biased by
padding/background**, and **(c) the capped-pool budget grid uses N=5000 while the real pool is 2500**.
None corrupts existing results' comparability; all are frozen_v2-level decisions held for your call.

---

## Check 1 — Budget table  ✅ grids valid; ⚠️ capped-pool N mismatch

All grids: **sorted, unique, ≤ pool_N, initial ≥ max(8, 2C)**. glas2015 is sane (`[8,10,20,26]`, no
non-monotonic tail). Fractions below are vs the **true AL pool N** (post-cap).

| dataset | C | full_train | AL_pool_N | initial | cumulative counts | incremental | max | max % of pool |
|---|---|---|---|---|---|---|---|---|
| busi | 2 | 624 | 624 | 8 | 8,13,32,63,94,125 | 8,5,19,31,31,31 | 125 | 20.0% |
| kvasir_seg | 2 | 800 | 800 | 8 | 8,16,40,80,120,160 | 8,8,24,40,40,40 | 160 | 20.0% |
| isic2018 | 2 | 2076 | 2076 | 21 | 21,42,104,208,312,416 | 21,21,62,104,104,104 | 416 | 20.0% |
| glas2015 | 2 | 133 | 133 | 8 | 8,10,20,26 | 8,2,10,6 | 26 | 19.5% |
| origa | 2 | 520 | 520 | 8 | 8,11,26,52,78,104 | 8,3,15,26,26,26 | 104 | 20.0% |
| mmwhs | 8 | ≥5000 | **2500** | 16 | 16,25,50,100,250,500 | 16,9,25,50,150,250 | 500 | **20.0%** |
| btcv_synapse | 14 | 1494 | 1494 | 28 | 28,30,75,150,225,299 | 28,2,45,75,75,74 | 299 | 20.0% |
| msd_task07_pancreas | 3 | ≥5000 | **2500** | 13 | 13,25,50,100,250,500 | 13,12,25,50,150,250 | 500 | **20.0%** |
| ext_brats2020 | 4 | ≥5000 | **2500** | 13 | 13,25,50,100,250,500 | 13,12,25,50,150,250 | 500 | **20.0%** |

**Verification:** counts sorted ✓ unique ✓ ≤ pool_N ✓; `initial = max(8, 2C, ⌈first_frac·N⌉)` holds
(e.g. mmwhs 16 = 2·8; isic 21 = ⌈1%·2076⌉; btcv 28 = 2·14). No 5,10,20,40,80,33-style bug.

**⚠️ Finding (capped datasets):** the grid for mmwhs/msd07/brats was built on `N = min(full_train,
5000) = 5000` (→ Case C `[0.25,0.5,1,2,5,10]%` ⇒ counts `13,25,50,100,250,500`), but the **fg-stratified
pool is only 2500**. So the realized max budget is **20% of the actual pool, not the intended 10%**, and
their initial 13/16 is ~0.5–0.6% of pool, not 0.25%. Counts are still valid and identical across all
methods (comparability intact), but the *fraction semantics and logged N are 2× off*. Fix = build the
budget grid on the true post-cap pool size (same root cause as the full-sup bug fixed earlier). This
changes budget counts → **frozen_v2-level, held for your decision.**

---

## Check 2 — Pool-cap manifests (mmwhs, msd07, brats)  ✅

- **How the 5000-cap is sampled:** `_stratified_pool_cap` (al_loop.py:244) — **foreground-stratified**,
  not random, not case-stratified. It pre-scans every train slice's mask and prefers slices containing
  any foreground at a target `fg_ratio=0.5`. (Case-disjointness is enforced earlier at the split level
  via `make_split`; the cap is fg-stratified *within* the train split.)
- **Shared across P0–P9:** **Yes.** The pool is built once and cached at
  `cache/al_state/pool_sets/<ds>__s1000__cap5000_fg0.5.json`, keyed by (dataset, seed, cap, fg_ratio),
  and **every policy loads the identical manifest** (verified: same `candidate_count` + initial = 2500
  across methods).
- **Manifest paths:** `mmwhs__s1000__cap5000_fg0.5.json`, `msd_task07_pancreas__s1000__cap5000_fg0.5.json`,
  `ext_brats2020__s1000__cap5000_fg0.5.json` (under `cache/al_state/pool_sets/`). Datasets with
  train ≤ cap (busi/kvasir/isic/glas/origa/btcv) have **no manifest** — the full train split is the pool.
- **fg ratio & class histogram in the capped pool:**

| dataset | pool_N | fg-slice ratio | fg-pixel ratio | notes |
|---|---|---|---|---|
| mmwhs | 2500 | 1.000 | 0.132 | all 7 fg classes 47–67% of slices (see diagnostics §2) |
| msd_task07_pancreas | 2500 | 1.000 | **0.006** | extreme FG sparsity (0.6% of pixels) |
| ext_brats2020 | 2500 | 1.000 | 0.027 | edema in ~99% of slices |
| btcv_synapse | 1494 | 1.000 | 0.078 | 13 organs, pancreas/gallbladder/adrenals ~14–16% |

(All capped pools end up 100% fg-slice because the fg-stratify pre-scan fills to target with fg slices;
the per-class histograms are in `stage1_seed1000_diagnostics.md` §2 — every class is present.)

---

## Check 3 — Full-supervision baseline definition  ✅ = Option A

Full-sup trains on **100% of the capped AL pool** (`budget_plan=[pool_size]`, where `pool_size` is the
true fg-stratified post-cap pool after the fix) — i.e. **Option A**, exactly the pool the AL methods
query from. This is your recommended main reference. For the 6 datasets with train ≤ cap, the capped
pool *is* the full train set, so A = B there. **No separate full-train (Option B) upper-bound was run**
— I can add it for the 3 capped datasets if you want the true upper bound reported separately.

---

## Check 4 — Padding / valid-region handling  ⚠️ padding biases selection

Letterbox = aspect-preserve to long-side 512, then **zero-pad to 512²**; padded mask pixels = class 0
(background). There is **no pad mask / valid-region mask / ignore_index anywhere** (Sample, batch,
PredictionCache, PolicyContext carry no pad info).

| included in… | padded pixels? | evidence |
|---|---|---|
| Training loss | **YES** (as bg) | `_dice_ce_loss` has no ignore_index; `_resize_mask` pads with 0 (trainer.py) |
| DSC / HD95 | **YES** | computed over full 512² (eval.py); *fg* DSC is robust (no fg in pad), but bg/HD95 affected |
| Query scores P1/P2/P5 | **YES** (full mean) | `aggregate(...,"full")` = mean over all pixels (_helpers.py `_apply_full`) |
| Query score P6 | mostly no | P6 uses fg-target/boundary masks (`_masked_mean`), padding ≈ excluded |
| Valid-region-only aggregation | **NO** | no such mechanism exists |

**Empirical impact (the requested sanity check).** Because pad pixels contribute ~zero entropy, the
full-canvas mean ≈ `valid_mean·(1−pad_frac)` — i.e. partly a function of image geometry. Correlating
each candidate's empty/pad fraction against its logged full-canvas entropy (P1, round 0):

| dataset | empty_frac mean/std | Spearman(empty_frac, entropy) |
|---|---|---|
| isic2018 | 0.300 / 0.050 | **−0.547** (p≈1e-160) |
| ext_brats2020 | 0.687 / 0.100 | **+0.486** (p≈2e-147) |
| mmwhs | 0.141 / 0.134 | −0.126 (p≈3e-10) |

The selection of the full-aggregation uncertainty policies is **materially correlated with padding/
background geometry** (|ρ| up to 0.55), and the sign is dataset-dependent. This is a real confound —
e.g. it explains ISIC-P1's large-lesion bias (diagnostics §Check 4): more content ⇒ less pad ⇒ higher
full-canvas entropy ⇒ selected. **Per-dataset letterbox pad fraction** (geometric, ≈ true bars):
isic ~0.30, brats ~0.35 (t1ce 240×155), glas ~0.32, btcv/mmwhs variable. **Recommendation:** aggregate
query scores over the **valid (un-padded) region** at minimum, ideally the predicted-fg region, for
P1/P2/P5. This changes method behavior → **frozen_v2-level, held for your decision.** (`fg`/`boundary`
modes already exist in `_helpers.aggregate`; switching P1/P2/P5 to a valid/fg mode is small.)

---

## Check 5 — P9 / P7 / SAM-H scheduling & GPU logging  ⚠️ no hard guard; GPU type not logged

- **GPU logged per cell:** only `gpu_mem_mb` (peak used). **GPU name/type and total memory are NOT
  logged** (trajectory.py). The cluster V100s are confirmed **32 GB** (logged at dispatch, and all P9
  cells ran without OOM), so seed-1000 is safe — but per your request, type+total should be logged.
- **P9 on 16 GB:** **No hard guard** — dispatch only *warns* (`WARN: <16GB may OOM`) and `_auto_batch`
  hints a smaller batch; nothing refuses/skips P9. In this cluster all V100s are 32 GB so it never hit
  a 16 GB card, but the protection is advisory, not enforced.
- **P7/P8/SAM-H:** SAM-H batch is **memory-aware** (`features/sam.py default_batch_size` sizes by *free*
  GPU memory: vit_h → 8 if ≥40 GB, 4 if ≥22 GB, else 2), so it won't pick an OOM batch — but there is
  **no job-level skip** if a card is simply too small.
- **Recommended additive fixes (do not affect results):** (1) log `gpu_name` + `gpu_total_mem_mb` in
  each record; (2) add a hard pre-flight guard: refuse P9 if total GPU mem < ~24 GB and P7/P8/SAM-H if
  < ~22 GB (fail fast with a clear message instead of mid-run OOM). These are safe to apply now.

---

## Check 6 — BraTS modality  ✅ t1ce-only; ⚠️ mislogged

- **Modality: contrast-enhanced T1 (`t1ce`) only**, single channel. On disk:
  `/groups/echambe2/datasets/external_drops/brats2020/sliced_2d/train/images_t1ce/` (cached images are
  (N,1,512,512) — one channel). Set via the bridge override `"ext_brats2020": {"modality": "t1ce"}`
  (medal_agent_bridge.py).
- **Same modality + split across all methods:** **Yes** — one adapter instance per `run_one`, shared
  by P0–P9; `make_split` is deterministic per seed (case-disjoint).
- **⚠️ Mislogged:** the trajectory `modality` field records the registry tag **`"multi_modal_mri"`**, not
  the actual `t1ce`. Misleading for provenance. **Recommended fix (metadata only):** log the concrete
  loaded modality (`t1ce`) — additive, doesn't affect results. Optionally encode it in the dataset id.

---

## Check 7 — Training adequacy  ⚠️ 250 iters is undertrained

| dataset | full-sup DSC (250it) | AL best | P0 loss trend | verdict |
|---|---|---|---|---|
| busi | 0.475 | 0.483 | ~flat | **undertrained** (100% data ≈ AL ~70 imgs) |
| mmwhs | 0.216 | 0.244 | **rises** 1.24→1.69 | **undertrained** (full-sup < AL best) |
| msd_task07_pancreas | 0.075 | 0.188 | — | **undertrained** (near-collapse @250) |
| kvasir_seg | 0.479 | 0.502 | — | **undertrained** (full-sup < AL best) |
| isic2018 | 0.825 | 0.839 | — | **undertrained** (Δ<0.03) |
| glas2015 | 0.899 | 0.894 | — | **saturated** (full-sup ≈ AL best) |
| origa | 0.934 | 0.943 | — | **undertrained** (Δ<0.03) |
| btcv_synapse | 0.059 | 0.093 | rises | **undertrained** (full-sup < AL best; only liver learns) |
| ext_brats2020 | 0.328 | 0.382 | — | **undertrained** (full-sup < AL best) |

**Stage-1 full-sup is now 9/9 and EVERY dataset is undertrained/saturated at 250 iters** — full
supervision never meaningfully beats low-budget AL, and on btcv/mmwhs/brats it is *below* the AL best.
The msd07 longer-training probe (P0 Random) confirms the cause: 0.000 @250 → **0.4286 @1000 → 0.4009
@2000** (recovers then plateaus). The binding constraint across the board is **training length**.

- **Loss decreasing?** No for multiclass — mean train loss **rises** with budget (btcv 1.67→1.75, mmwhs
  1.24→1.69), and val DSC is flat/declining (diagnostics §Check 2). Binary datasets reach usable DSC
  but full-sup saturates at the low-budget AL level.
- **All-methods-near-zero?** msd07 at 250 iters (P0/P3/P4/P7 = 0.000).
- **Is full supervision obviously undertrained?** **Yes** — wherever measured, 100% of the pool barely
  beats (or underperforms) a tiny AL fraction.
- **🔑 Decisive new evidence:** **msd07 P0 @1000 iters = 0.4286** (vs 0.000 @250). Longer training
  *rescues* the "collapse" → it was **undertraining, not an intrinsic hard-task failure.** (@2000 probe
  running to confirm the ceiling.) This **reverses** the earlier "retire msd07 (option D)" lean: msd07
  should **stay in**, and the real lever is **training length.**

---

## Recommended actions before scaling to seeds 2000/3000

**A. Safe, additive — can apply now (no effect on existing results' comparability):**
1. Log `gpu_name` + `gpu_total_mem_mb` per cell (Check 5).
2. Hard pre-flight memory guard: refuse P9 < ~24 GB, SAM-H/P7/P8 < ~22 GB (Check 5).
3. Log the concrete BraTS modality (`t1ce`) instead of `multi_modal_mri` (Check 6).
4. Dedup `selected_ids` for coreset-family policies (diagnostics logging finding) — needed for the
   eventual skill export; does not affect DSC.

**B. frozen_v2-level — your decision, not changed yet:**
5. **Training iters** (Check 7) — the highest-leverage change. msd07 0.00→0.43 and the multiclass
   underfitting both say 250 is too low. Options: raise globally, or scale by difficulty (e.g. 1000 for
   multiclass/hard, keep 250 for binary).
6. **Valid/fg-region score aggregation** for P1/P2/P5 (Check 4) — removes the padding/geometry confound.
7. **Budget grid on true post-cap pool N=2500** for mmwhs/msd07/brats (Check 1) — honest fractions.

I'm **holding** — not launching seeds 2000/3000, not modifying frozen_v2. Tell me which of A (apply now)
and B (re-freeze) you want, and I'll proceed.
