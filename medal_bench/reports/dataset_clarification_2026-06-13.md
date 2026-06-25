# Dataset clarification (pre-Stage-0) — 2026-06-13

Requested before Stage 0. Three tables: (1) 21 loader-ready vs 24 sliced, (2) classes
by split + missing-by-split, (3) ext_word_ct recommendation.

---

## Table 1 — 21 training-loader-ready vs 24 sliced datasets

There are **24 `sliced_2d` dirs on disk** but the `medal_agent` registry exposes **21**.
The 5 registry `ext_*` ids map to disk dirs `abdoment1k/amos_ct/amos_mri/brats2020/word_ct`.
The **3 sliced-but-not-registered** are `chaos`, `snemi3d`, `promise12`.

| dataset_id (disk) | status | reason |
|---|---|---|
| abdoment1k → `ext_abdoment1k` | ✅ included | registered, audited |
| amos_ct → `ext_amos_ct` | ✅ included | registered, audited |
| amos_mri → `ext_amos_mri` | ✅ included | registered, audited |
| brats2020 → `ext_brats2020` | ✅ included | registered, audited |
| word_ct → `ext_word_ct` | ✅ included | registered; see Table 3 caveat |
| btcv_synapse | ✅ included | registered (remap 16→13, C=14) |
| care_leftatrium_2026 | ✅ included | registered (binary; bridge drops 1189 no-atrium-mask slices) |
| flare22 | ✅ included | registered (C=14) |
| hvsmr2016 | ✅ included | registered (C=9) |
| kits19 | ✅ included | registered (C=3) |
| liqa_mri | ✅ included | registered (C=2) |
| mmwhs | ✅ included | registered (C=8; bridge uses CT view) |
| myops | ✅ included | registered (C=5) |
| msd_task02_heart … msd_task10_colon (8) | ✅ included | registered (MSD subset) |
| **chaos** | ⏸ excluded/pending | **inferred:** multi-modal (CT {0,255} + MR {0,63,126,189,252}); cross-modality remap/registration not in registry — confirm |
| **snemi3d** | ⛔ excluded | **inferred:** EM neuron membrane/instance segmentation — different task type, out of scope for semantic-organ AL — confirm |
| **promise12** | ⏸ excluded (dup) | **inferred:** already wired natively in `medal_bench` (2D MHD adapter `promise12`); excluded from `medal_agent` to avoid a duplicate source — confirm |

→ 21 registered = 24 sliced − {chaos, snemi3d, promise12}. The three "reason" entries are
my inference from on-disk structure; please confirm/correct.

---

## Table 2 — classes by split + missing-by-split

Computed fresh from `medal_agent` with dense remap (`runs/stage_pre1/class_by_split.json`).
Legend: `[]` = all dense classes present · `[n,…]` = dense classes **missing** from that split ·
`derived` = no physical val dir (the runner's `make_split` carves val from train at run time) ·
`unlabeled` = physical test dir but no masks (`test_has_labels=False`) · `absent` = no such dir.
(The on-disk `audit_reports/check_A_B_summary.json` is STALE — predates the btcv 16→13 fix; this
table supersedes it.)

| dataset_id | C | train missing | val (physical) | test |
|---|---|---|---|---|
| btcv_synapse | 14 | `[]` | `[]` (native) | absent |
| care_leftatrium_2026 | 2 | `[]` | derived | absent |
| ext_abdoment1k | 6 | `[]` | `[]` (native) | `[]` **labeled** |
| ext_amos_ct | 16 | `[]` | `[]` (native) | unlabeled |
| **ext_amos_mri** | 16 | `[]` | **`[14, 15]`** (native) | unlabeled |
| ext_brats2020 | 4 | `[]` | derived | absent |
| **ext_word_ct** | 17 | **`[8, 9, 15, 16]`** | **`[8, 9, 15, 16]`** | absent |
| flare22 | 14 | `[]` | `[]` (native) | absent |
| hvsmr2016 | 9 | `[]` | derived | absent |
| kits19 | 3 | `[]` | derived | unlabeled |
| liqa_mri | 2 | `[]` | derived | absent |
| mmwhs | 8 | `[]` | derived | absent |
| myops | 5 | `[]` | derived | absent |
| msd_task02_heart | 2 | `[]` | derived | unlabeled |
| msd_task03_liver | 3 | `[]` | derived | unlabeled |
| msd_task04_hippocampus | 3 | `[]` | derived | unlabeled |
| msd_task06_lung | 2 | `[]` | derived | unlabeled |
| msd_task07_pancreas | 3 | `[]` | derived | unlabeled |
| msd_task08_hepaticvessel | 3 | `[]` | derived | unlabeled |
| msd_task09_spleen | 2 | `[]` | derived | unlabeled |
| msd_task10_colon | 2 | `[]` | derived | unlabeled |

**Two datasets have missing-class issues** (rest are clean — every dense class present in train):
- **ext_word_ct**: `{8,9,15,16}` missing from train **and** val → see Table 3.
- **ext_amos_mri**: `{14,15}` present in train but **missing from its native val split** → val DSC can't
  score classes 14/15. **Recommend:** for amos_mri, let the runner re-derive val from train
  (case-disjoint) instead of using the native val, OR report present-class val metrics. Flag Tier B.

Only **ext_abdoment1k** has a labeled test split among the bridged set (test_has_labels); all MSD/amos
tests are **unlabeled** (images-only) → must NOT be used for DSC/HD95 (use val). This matches the
`test_has_labels` gate.

---

## Table 3 — ext_word_ct missing train+val classes {8, 9, 15, 16}

`ext_word_ct` is a 17-class whole-body CT dataset. Dense classes **{8, 9, 15, 16} are absent
from train+val** (per your audit + my re-scan): rare organs not present in every patient
(gallbladder, esophagus, head-of-femur L/R). A class absent from train+val can be neither
learned nor validated.

**Options:**
- **A) Exclude from Stage 1** — loses a valuable 17-class whole-body CT dataset. Overkill: 13
  classes are fully usable.
- **B) Include with present-class metrics only (RECOMMENDED for Stage 1)** — keep the loader as
  is (C=17, no remap change); report macro/per-class DSC over the **present** classes only, and
  mark {8,9,15,16} as `N/A` (no train signal). Lowest-risk, no re-slicing, usable now. If those
  classes appear in the labeled test split, exclude them from test metrics too (the model has no
  signal for them → spurious 0 DSC).
- **C) Revise slicing/splitting so all 17 classes appear in train+val** — the clean long-term
  fix, but requires re-running the slicer/split with class-stratified case assignment; delays
  Stage 1.

**Recommendation:** **B now** (present-class metrics, {8,9,15,16} marked N/A), and schedule **C**
before any *headline* Stage 2 use of ext_word_ct. Flag ext_word_ct **Tier B/C** (caveat:
incomplete class coverage) in the inclusion table.
