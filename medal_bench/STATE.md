# MedAL-Bench — Current State & Handoff (read before MedAL skill learning)

Last updated 2026-06-15. This is the authoritative state of the P0–P9 active-learning benchmark.
Read this before building the **Round-Adaptive AL Skill**. Trust this over older reports/slides/memory
if they conflict (notably: **P6 is Selective Uncertainty, NOT PEAL**).

> **frozen_v3 IMPLEMENTED (2026-06-15).** The Stage-2 launch config is now **`profiles/frozen_v3.py`**
> (`FROZEN_V3_HASH`); `bench512` profile = **num_iters=1000** (was 250 — Stage 1.5 proved 250 undertrains).
> Spec + rationale: `reports/frozen_v3_plan.md` + `reports/NEXT_SESSION_frozen_v3.md`. What changed vs v2:
> (1) **PRIMARY metric = per-case macro-fg DSC** (`mean_dsc_fg_case_macro`; `mean_dsc_fg` is now an ALIAS
> to it; `metric_version=v3_case_macro`); per-case HD95 + **symmetric ASSD** with a **diagonal total-miss
> penalty** + `structure_detection_rate`; old micro/pooled DSC + directed ASD kept as `*_diagnostic`.
> (2) **valid-region** query aggregation for P1/P2/P5 + hard valid-region intersection in P6 + valid-region
> masking in eval (letterbox pad excluded; `meta['valid_bbox']` plumbed through `_IndexedSubset`→
> `PolicyContext.valid_bboxes`). (3) **budget denominator = actual_AL_pool_N** (`run_one` mirrors
> `run_full_supervised`; logs full_train_N / requested_pool_cap / actual_AL_pool_N / both fractions).
> (4) **component per-round seeding** (`seed_all(seed+r)` + model_init/loader/query/dropout seeds, all
> logged). (5) **P8 paper-faithful** (MIN_CLUSTER_SIZE + round-robin + K-cap min(20,len//2)); pre-v3 P8
> preserved as ablation **P8c** (registry now **14 ids**). (6) **always-on prediction saving** (compressed
> val masks + ids + valid_bbox + fp16 probs; `--save-predictions/--save-logits`). (7) orchestration:
> dispatch **per-cell timeout** + **done-by-round-count** + per-PID partials + `run_one --force`.
> **Logging schema bumped v2→v3** (`TRAJECTORY_SCHEMA_VERSION`). Tests: **173 pass** (was 102). NEXT: run
> the **v3 canary** (`submit/v3_canary_worker.sh`) + BTCV-2000 check, review the prob-storage estimate
> (HARD GATE), then Stage 2. **Stage 2 stays blocked until the canary passes + storage/BTCV-2000 reviewed.**

> **Stage 0a/0b PASSED (2026-06-13).** 4-stage plan (0a→−1→0b→1→2→3). **medal_agent (21 datasets)
> bridged into the runner** (`data/adapters/medal_agent_bridge.py`; DATASET_REGISTRY now 32). Stage 0a
> (busi/kvasir/msd07) + Stage 0b (mmwhs C=8 / btcv_synapse C=14 / ext_abdoment1k C=6) = 40 smoke cells
> all green; full checklist verified (determinism replay, case-disjoint, no leakage, **gpu_mem_mb now
> logged**, HD95 skipped-under-smoke). Stage −1 so far: **B1** remap+MMWHS (now superseded by the bridge),
> **B3** budgets, **B5** derived metrics. Stage −1 now also: **B2** adaptive 512 letterbox
> (`TrainConfig.aspect_preserve`) + **bench512_dry** profile (pool-dependent budget, HD95 first/mid/final);
> **B4** full-sup runner (`runner/run_full_supervised.py`); **B6** preproc disk-cache (hit/miss logged;
> dtype-shrink to int16/fp16 still TODO); **B7** SAM-H warm cmd (`runner/precompute_sam.py`, resolution-keyed
> cache `__in512`); **balanced multi-GPU dispatcher** (`runner/dispatch.py`, pull-queue for 4×A40+8×V100).
> **Stage 0c** (formal-profile dry run, busi/msd07/mmwhs/btcv × P0,P1,P4,P8,P9 @512) IN PROGRESS. 143 tests.
> Decide before Stage 1: F1 (honor medal_agent native splits), F2 (drop interim mmwhs_ct/mmwhs_mr),
> V100 mem (→ v100 batch), submit system (persistent dispatcher vs SLURM array → dispatch --emit-manifest).
> Reports: `reports/{stage0a_0b,stage1_launch_plan,dataset_clarification,medal_agent_integration,stage_minus1_plan}_2026-06-13.md`.

---

## 0. Can the methods be trusted?

YES for the **implementations** — P0–P9 are correct, reference-faithful, leakage-free, reproducible,
and fairly comparable (validated by 102 passing tests + a 2k-pool stress test, all invariants green).
NOT yet for **accuracy conclusions** — the multi-dataset DSC-vs-budget benchmark has **not been run**;
do not cite "which method wins" yet.

## 1. Method registry (canonical)

Core (`medal_bench.policies.all_ids()` → 12 ids incl. ablations):

| ID | name (`build(id).name`) | what it selects on |
|----|------|------|
| P0 | Random | nothing (uniform) |
| P1 | Normalized Entropy | `H/log C ∈[0,1]` per-pixel, mean over pixels |
| P2 | BALD | MC-dropout mutual information, T=10 |
| P3 | CoreSet | k-center on U-Net **bottleneck** features |
| P4 | BADGE | **canonical CE-only** gradient embedding (dim C·D) + k-means++ |
| P5 | Entropy -> CoreSet | top-5k entropy → k-center (hybrid baseline) |
| P6 | Selective Uncertainty | entropy on predicted **target+boundary** pixels (arXiv:2401.16298) |
| P7 | Foundation-CoreSet | k-center on **SAM** image features |
| P8 | Foundation-TypiClust | TypiClust on SAM features (L∪U clusters, uncovered-first) |
| P9 | PAAL | Accuracy-Predictor + Weighted-Polling (IJCAI 2024) |

Ablations (NOT core; `is_ablation=True`): **P4b** BADGE-Seg-CE-Dice, **P8b** SAM-DensityClust.
PEAL is removed from core (archived at `policies/_archived_peal/`).

## 2. Phase grouping (the basis for round-adaptive skill learning)

| AL phase | methods | intuition |
|---|---|---|
| **Coverage Formation** | P3, P7, P8 | spread over the feature manifold (diversity/representativeness) |
| **Boundary Discovery** | P1, P2, P4, P5, P6 | find uncertain / decision-boundary samples |
| **Error Refinement** | P6, P9 (opt. P4/P5) | target predicted-low-quality / boundary regions |
| **Saturation Gate** | — | no new AL method; stop / fallback / random-audit behavior |

The code cleanly separates **score** (uncertainty/boundary) from **select** (coverage/diversity), so a
skill can mix-and-match per round.

## 3. How to call any method from an AL state (the interface a skill uses)

```python
from medal_bench.policies import build, PolicyContext
pol = build("P6")                          # or any id, with **config kwargs
ctx = PolicyContext(
    seed=..., round_idx=...,               # determinism is derived from (seed, round_idx)
    model=task_unet,                       # current-round trained model
    pred_cache=pred_cache,                 # softmax probs/argmax over the unlabeled pool (P1/P2/P5/P6/P9)
    pool=unlabeled_ds, labeled=labeled_ds, # read-only; NEVER read pool/unlabeled masks
    features={"task_unet_pool":..., "task_unet_label":...,   # P3/P5/P9
              "foundation_pool":..., "foundation_label":...},# P7/P8 (SAM)
    num_classes=C,
)
scores = pol.score(ctx)                    # np array or None (None for pure-diversity methods)
selected = pol.select(ctx, scores, k)      # list[int] pool-LOCAL indices, k unique
diag = ctx.diagnostics_out                 # per-method signals (see §4)
```

The runner that builds all of this each round is `medal_bench.runner.al_loop.run_al`.
**Firewall:** policies must never read `ctx.pool[i].mask` (unlabeled GT). P9 reads only `ctx.labeled` masks.

## 4. Per-round signals a skill can learn from

- **Trajectory JSONL (schema v2)** — one record/round: `selected_ids`, `candidate_count`,
  `candidate_scores_path` (sidecar with every candidate's score), `selection_diagnostics`,
  `metrics` (val DSC), `labeled_count`, `cumulative_budget`, `incremental_query_count`, `ckpt_hash`,
  `feature_cache_keys`, `sam_model_type`. Written by `runner/trajectory.py`.
- **Per-method `selection_diagnostics`** (in `ctx.diagnostics_out`): e.g. P2 `bald_*`; P6
  `selu_target_frac/selu_boundary_frac`; P8 `typiclust_n_clusters/selected_from_uncovered`;
  P9 `paal_ap_loss_mean/paal_pred_acc_mean/paal_selected_clusters`; P4 `badge_embedding_dim`.
  These are exactly the kind of state a round-adaptive skill can use to decide which method to run.

## 5. Frozen benchmark config (locked) — `medal_bench/profiles/frozen.py`

- Budget curve cumulative **1→2→5→10→15→20%**; seeds 1000/2000/3000; query unit image/slice
  (3D sliced to 2D; IDs encode dataset+case+slice).
- **Foundation extractor = SAM-H / vit_h** (checkpoint `sam_vit_h_4b8939.pth`); pass
  `--sam-model-type vit_h`. vit_b only if explicitly configured (selections differ materially).
- Determinism on by default (`MEDAL_NONDETERMINISTIC=1` to opt out); initial labeled set persisted &
  shared across methods. Preprocessing v1 (mask nearest, image bilinear); logging schema v2.

## 6. Datasets

**Registry now has 32 datasets** (`data/adapters/__init__.py` `DATASET_REGISTRY`):
- 11 medal_bench-native: isic2018, cvc_clinicdb, busi, promise12, msd07_pancreas, kvasir_seg,
  hyperkvasir_seg, glas2015, origa + interim mmwhs_ct/mmwhs_mr.
- **21 bridged from `medal_agent`** (`/groups/echambe2/datasets/medal_agent`, the user's audited
  pre-sliced loader — Check A/B + smoke + overlays all green): btcv_synapse, care_leftatrium_2026,
  ext_abdoment1k, ext_amos_ct, ext_amos_mri, ext_brats2020, ext_word_ct, flare22, hvsmr2016, kits19,
  liqa_mri, mmwhs, myops, msd_task02/03/04/06/07/08/09/10. Via
  `data/adapters/medal_agent_bridge.py` (`MedalAgentBridge`); all 21 pass a P0 loader smoke through
  the runner (case-disjoint). **medal_agent is the single source of truth for the expanded set;**
  interim `mmwhs_ct/mmwhs_mr` + `data/remap.py` are superseded (deprecate — see
  `reports/medal_agent_integration_2026-06-13.md`). TWO open decisions there: F1 honor medal_agent
  native splits for formal Stage 1 (runner hook) and F2 deprecate the interim MMWHS adapter.

## 7. Known concerns / not-yet-done

1. No accuracy benchmark run yet (DSC curves) — gated on datasets + throughput.
2. **Throughput:** pool eager-load ≈15 min for 2k images over NFS (`_IndexedSubset` per run) — fix with
   a disk-cache/memmap/lazy loader before launching ~1,500 cells.
3. Dataset coverage 10/42.
4. P4 per-candidate forward (9 s @2k) is batchable like the prediction cache (cheap win).

## 8. Key entry points & commands

```bash
PY=/groups/echambe2/gmeng/conda_envs/medal-agent/bin/python   # torch 2.4.1+cu121, CUDA
$PY -m pytest medal_bench/tests/ -q                            # 102 tests
$PY -m medal_bench.runner.run_one --policy P6 --dataset busi --seed 1000 --profile pilot \
   --foundation sam --sam-model-type vit_h --out-dir runs/x    # one cell
$PY -m medal_bench.runner.smoke_matrix --out-dir runs/smoke --policies P0,...,P9 --datasets busi
$PY -m medal_bench.profiles.frozen                             # print frozen config + hash
```

Full reports: `reports/audit_p0_p9_2026-06-12.md`, `reports/benchmark_readiness_2026-06-12.md`,
`reports/warnings_closure_2026-06-13.md`, `reports/EXPERIMENT_SUMMARY_for_sharing.md`.
