# Stage 2 — FINAL plan (frozen_v3): 20-dataset Core + waves + task definitions

User-locked 2026-06-15. Counts from `/groups/echambe2/datasets/DATASET_TABLE_FINAL.md` + medal_agent
registry; `actual_AL_pool_N` **measured** for Stage-1 datasets, **(build)** = measured in Wave-0/1 under
v3. `query_dim = slice` for all (a native-2D image = one slice = one case). `metric_split = val`
(case-disjoint); MSD/KiTS/AMOS ship unlabeled test, so val is **derived 15%** from train cases.
Budget schedule = `budget_grid(actual_AL_pool_N, C)` (small→absolute; 500≤N<5k→[1,2,5,10,15,20]%;
2500-cap→[~13,25,50,100,250,500]).

## Wave plan
- **Wave 0** — v3 canary {busi, isic2018, mmwhs, btcv_synapse, msd_task07_pancreas} × {P0,P1,P4,P5,P8,P9} × seed 1000.
- **Wave 1** — core-9 (the Stage-1 datasets) rerun under v3, seed 1000 (sanity vs Stage-1.5).
- **Wave 2** — full 20-dataset Core, seed 1000.
- **Wave 3** — seeds 2000 + 3000 (Core).
- **Wave 4** — supplementary / hard datasets.

## Stage 2-CORE (20)

| dataset_id | formal_task_id | tier | modality | object | native_dim | C | train/val/test cases | actual_AL_pool_N | metric_split | test_has_labels | remap_key | task_variant | caveats | in_core_avg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| busi | busi_lesion | A | Ultrasound | breast lesion | 2D | 2 | 780 / derived / – | **624 (meas)** | val | no | binary_255 | binary | no fixed split | Y |
| kvasir_seg | kvasir_polyp | A | Endoscopy | GI polyp | 2D | 2 | 880 / 120 / – | **800 (meas)** | val | no | binary_255 | binary | — | Y |
| refuge | refuge_disc_cup | A | Fundus | optic disc+cup | 2D | 3 | 400 / 400 / 400 | (build) | val (+test) | **yes (400)** | photo_3class | disc+cup (C=3) | **verify wiring (fundus, like origa); fallback promise12** | Y |
| isic2018_task1 | isic2018_lesion | A | Dermoscopy | skin lesion | 2D | 2 | 2594 / 100 / 1000 | **2076 (meas)** | val (+test) | **yes (1000)** | binary_255 | binary | — | Y |
| glas2015 | glas_gland_binary | A | Histopathology | gland | 2D | 2 | 85 / der / 80+20 | **133 (meas)** | val (+test) | yes | binary_255 | **binary (collapse instances)** | **tiny pool→wide CI** | caution |
| origa | origa_disc_cup | A | Fundus | optic disc+cup | 2D | **3** | 650 / der / – | (rebuild @C3) | val | no | photo_3class | **disc+cup (C=3, NOT binary)** | **Stage-1 ran C=2 → re-task to C=3** | Y |
| rose1 | rose1_vessel_svc | A | OCT-A | retinal vessel | 2D | 2 | 30 / – / 9 | (build, ~30) | val | yes(9) | binary_255 | SVC complex | **tiny pool→wide CI; best-balanced (fg 19.6%)** | caution |
| msd_task09_spleen | msd09_spleen | A | CT | spleen | 3D | 2 | 41 / der / 20(nolbl) | (build) | val | no | (identity) | binary | clean single organ | Y |
| msd_task04_hippocampus | msd04_hippocampus | A | MRI | hippocampus | 3D | 3 | 260 / der / 130(nolbl) | (build) | val | no | (identity) | ant+post | small ROI (fg 8.2%) | Y |
| msd_task03_liver | msd03_liver | B | CT | liver+tumor | 3D | 3 | 131 / der / 70(nolbl) | (build, cap) | val | no | (identity) | organ+tumor | tumor sparse | Y |
| mmwhs_ct | mmwhs_ct_wholeheart | B | CT | whole heart (7) | 3D | 8 | 60 / der / – | (build; CT-only) | val | no | mmwhs | **CT-only (LOCKED split)** | mmwhs_mr → supplementary; combined NOT used | Y |
| hvsmr2016 | hvsmr_cardiac | B | MRI | blood/myo/vessels (9) | 3D | 9 | 60 / der / – | (build) | val | no | (identity) | 9-class | high class count; needs 1000it | Y |
| care_leftatrium_2026 | care_la_atrium | B | MRI(LGE) | left atrium | 3D | 2 | 190 / der / – | (build) | val | no | care_la | **atrium-only (rec.)** | 3 variants — see TD#5; scar→supp | Y |
| btcv_synapse | btcv_13organ | B | CT | 13 abd organs | 3D | 14 | 27 / 3 / – | **1494 (meas)** | val | no | btcv | **may need 2000it** | 14-class | Y |
| flare22 | flare22_13organ | B | CT | 13 abd organs | 3D | 14 | 45 / 5 / – | (build) | val | no | (identity) | 14-class | btcv-like; maybe 2000it | Y |
| kits19 | kits19_kidney | B | CT | kidney+tumor | 3D | 3 | 210 / der / 90(nolbl) | (build, cap) | val | no | (identity) | organ+tumor | tumor sparse | Y |
| ext_abdoment1k | abdoment1k_4organ | B | CT | 4 abd organs | 3D | 6 | 892 / 100 / 7 | (build, cap 2500) | val | yes(7) | abdoment1k | dense 6 | huge→cap; sparse codes | Y |
| ext_brats2020 | **ext_brats2020_t1ce** | B | MRI(t1ce) | brain tumor | 3D | 4 | 369 / der / 125(gated) | **2500 (meas)** | val | no | brats | **t1ce-only (TD#3)** | high empty-frac | Y |
| msd_task07_pancreas | msd07_pancreas | C | CT | pancreas+tumor | 3D | 3 | 281 / der / 139(nolbl) | **2500 (meas)** | val | no | (identity) | organ+tumor | **fg 0.63%; 0.00@250→0.43@1000** | caution |
| liqa_mri | liqa_liver | C | MRI(multiparam) | liver | 3D | 2 | 30 labeled / – / – | (build, ~30) | val | no | (identity) | GED4 binary | **only 30 labeled (TD#6); tiny→wide CI** | caution |

**CORE is LOCKED (20)**: busi, kvasir_seg, refuge, isic2018_task1, glas_gland_binary, origa_disc_cup,
rose1, btcv_synapse, flare22, kits19, msd_task03_liver, msd_task07_pancreas, msd_task09_spleen,
ext_brats2020_t1ce, msd_task04_hippocampus, mmwhs_ct, hvsmr2016, care_la_atrium, ext_abdoment1k, liqa_mri.

`in_core_avg`: **Y** = include in the headline core mean; **caution** = report separately / with CIs
(tiny pool ⇒ noisy per-case macro DSC): glas_gland_binary, rose1, msd07_pancreas, liqa_mri.

## Task-definition resolutions (TD#1–7)
1. **ORIGA → C=3** (bg/disc/cup; native {0,128,255}). Stage-1 ran it binary (C=2) — **that was a
   simplification; re-task to 3-class** `origa_disc_cup` with a 3-class photo remap. (Pool rebuilds @C3.)
2. **GlaS → binary gland mask** (`glas_gland_binary`); collapse the instance IDs to fg. (Instance seg is
   out of scope for this semantic-AL benchmark.)
3. **BraTS → t1ce only**, logged as **`ext_brats2020_t1ce`** (Bucket-A modality fix already emits t1ce);
   C=4 dense {bg,necrotic,edema,enhancing} (native {0,1,2,4}, 4→3).
4. **MMWHS → SPLIT (LOCKED).** Core = **`mmwhs_ct`** (CT-only, cleaner single-domain); **`mmwhs_mr`** →
   supplementary (domain-shift study). The combined CT+MR task is **NOT used in the core average.**
5. **CARE-LA → atrium-only** (`care_la_atrium`) for Core; scar / atrium+scar variants are sparse/harder →
   supplementary / Stage-3. (Native atrium codes {255,420} both→1.)
6. **LiQA → labeled-only.** Supervised AL reveals a label on query, so the pool = the **30 labeled GED4
   cases** (the 430 unlabeled subjects are not usable as labeled-on-query). Tiny pool ⇒ small-budget AL,
   noisy metric → `in_core_avg=caution`.
7. **3D-slice pool = foreground-positive retained slices** (the fg-stratify cap fills the fg half; ~no
   bg-only slices). Documented; acceptable, disclosed.

## SUPPLEMENTARY / hard (Wave 4, LOCKED) — also Stage-3 skill-learning data
hyperkvasir_seg, cvc_clinicdb, mmwhs_mr, rose2, msd_task06_lung (fg 0.22%),
msd_task08_hepaticvessel (0.60%), msd_task10_colon (0.47%), ext_word_ct (17-cls, 4 sparse),
ext_amos_ct (16-cls), ext_amos_mri, myops (scar/edema), promise12.

## HOLD OUT (not in current Stage 2)
totalsegmentator (117-cls, not sliced), snemi3d (EM instance), cubs_v1/cubs_extended (boundary coords),
duke_dme_chiu2015 / umn_oct (OCT layers/fluid — non-dense), drive (no test GT in drop),
rsna_boneage_zhao2021b (proxy masks), cholecseg8k (13-cls surgical, color-watershed labels).

## Locked decisions (2026-06-15)
- **20-Core final** (see list above); MMWHS→`mmwhs_ct` (split, CT core); hyperkvasir_seg→supplementary;
  **REFUGE** takes the 20th Core slot (PROMISE12 fallback only if REFUGE wiring fails).
- Task IDs locked: `origa_disc_cup` (C=3), `glas_gland_binary`, `ext_brats2020_t1ce`, `care_la_atrium`,
  `mmwhs_ct`; LiQA = 30 labeled cases only; 3D-slice pools = fg-positive retained slices (documented).
- **Wiring to verify before Wave 2** (not in medal_agent registry / not Stage-1-proven): refuge,
  hyperkvasir_seg (supp), promise12 (fallback). busi/kvasir_seg/isic2018/glas/origa use medal_bench 2D
  adapters; the 13 3D/registry datasets are bridge-ready.
- `in_core_avg=caution` for tiny-pool Core (glas 85 / rose1 30 / liqa 30 / msd09 41 cases) — report with CIs.
