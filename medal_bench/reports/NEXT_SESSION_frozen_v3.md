# Kickoff prompt for the fresh CC chat — frozen_v3 implementation

Copy the block below into a new Claude Code chat (working dir `/groups/echambe2/gmeng/MedAL-Agent`).

---

```
Project: MedAL-Bench — fixed-method active-learning benchmark for 2D medical image segmentation.
Working dir: /groups/echambe2/gmeng/MedAL-Agent  (code at repo/code/, run commands from repo/code/)
Test env python: /groups/echambe2/gmeng/conda_envs/medal-agent/bin/python (torch 2.4.1+cu121)

Stage 1.5 is COMPLETE and the decision is locked. Implement frozen_v3, run a v3 canary, then WAIT for
my Stage-2 go. Do NOT launch full Stage 2 / seeds 2000/3000 until the canary passes and I approve.

READ FIRST, in this priority order (these are the authoritative handoff):
1. repo/code/medal_bench/reports/frozen_v3_plan.md   ← ★ the FINALIZED SPEC + sequencing (start here)
2. repo/code/medal_bench/reports/stage1p5_report.md  ← final results/evidence (+ stage1p5_diagnostics if needed)
3. repo/code/medal_bench/reports/stage2_dataset_list.md  ← Stage-2 selection/tiers
4. /groups/echambe2/datasets/DATASET_TABLE_FINAL.md  ← authoritative dataset catalog (verified cases/slices/labels/test-labels/imbalance)
5. the current runner/config files: repo/code/medal_bench/runner/{eval,trainer,al_loop,run_one,dispatch,seeds}.py,
   medal_bench/policies/{_helpers,p1_entropy_full,p2_bald,p5_entropy_coreset,p6_selective_uncertainty,p8_sam_typiclust}.py,
   medal_bench/profiles/{__init__,budget,frozen_v2}.py
IMPORTANT: do NOT rely on the older STATE.md or frozen_v2 reports where they CONFLICT with frozen_v3_plan.md
— frozen_v3_plan.md (+ this prompt) win. Update STATE.md to point at frozen_v3 once implemented.

IMPLEMENT frozen_v3 exactly per frozen_v3_plan.md §FINALIZED SPEC (10 items):
 1. num_iters=1000 global default; + a BTCV-only 2000-iter check before final Stage 2 (report if BTCV
    needs a 2000 exception, else keep global 1000).
 2. PRIMARY metric = per-case macro foreground DSC. 3D-as-slice: group slices by case_id, reconstruct
    per-case pred/label volumes. Native-2D: each image = one case. Secondary: HD95_case_macro_fg,
    symmetric_ASSD_case_macro_fg, structure_detection_rate, missed_structure_rate. Total-miss →
    diagonal penalty + detection rate (NEVER silently drop). Keep old micro/pooled DSC as diagnostic only.
 3. Valid-region query aggregation for P1, P2, P5; verify P6 already ignores padding via fg/boundary mask.
 4. Budget denominator = actual accessible AL_pool_N; log full_train_N, requested_pool_cap, actual_AL_pool_N,
    budget_fraction_of_AL_pool, budget_fraction_of_full_train, absolute counts.
 5. P8 TypiClust fix: min-cluster-size filter + round-robin selection + no singleton-outlier picks;
    keep old P8 as deprecated/ablation variant.
 6. Component-level deterministic seeding; log all seeds.
 7. Always save compressed val prediction masks + case/slice IDs + valid-region masks + spacing/affine if
    available; save logits if storage allows, else logits for selected/debug subsets only.
 8. Document the foreground-only-pool caveat (3D-slice uses fg-positive retained slices).
Also fix the orchestration bugs that wedged the overnight run: dispatch subprocess timeout/watchdog +
lease-race truncation (see stage1p5_code_review.md). Bump the frozen hash; document the change. No broad
code debugging beyond these v3 protocol fixes.

THEN, in order:
 1. implement frozen_v3 (above);
 2. run the v3 CANARY: datasets {busi, isic2018, mmwhs, btcv_synapse, msd_task07_pancreas}, methods
    {P0,P1,P4,P5,P8,P9}, seed 1000, first/mid/final budgets (full curve if cheap);
 3. run the BTCV-2000 check;
 4. produce a v3 canary report: per-case DSC; HD95/ASSD with total-miss penalty; valid-region score
    behavior; corrected budget fractions; P8 corrected behavior; prediction saving works; runtime;
    whether P0 no longer collapses on MSD07; whether full_sup_pool is adequate;
 5. STOP and wait for my Stage-2 go.

STAGE-2 PLAN IS LOCKED in stage2_dataset_list.md — a **20-dataset Core** + supplementary/holdout + a
5-wave rollout (Wave 0 = the v3 canary above; Wave 1 = core-9 rerun under v3; Wave 2 = 20-Core seed 1000;
Wave 3 = seeds 2000/3000; Wave 4 = supplementary/hard). Do NOT run all ~50 datasets. Before Wave 2,
resolve the dataset-readiness + task-definition items in stage2_dataset_list.md §TD:
 All LOCKED (see stage2_dataset_list.md):
 - **ORIGA = C=3 disc+cup** (`origa_disc_cup`, 3-class remap) — NOT the binary C=2 Stage-1 used.
 - **GlaS = binary gland** (`glas_gland_binary`); **BraTS = t1ce-only** (`ext_brats2020_t1ce`, C=4 dense).
 - **MMWHS = SPLIT: `mmwhs_ct` in Core, `mmwhs_mr` supplementary** (combined CT+MR NOT used in core avg).
 - **CARE-LA = atrium-only** (`care_la_atrium`); **LiQA = 30 labeled cases only**; 3D-slice pools =
   fg-positive retained slices (documented).
 - **20th Core slot = REFUGE** (hyperkvasir_seg moved to supplementary). **Wiring to verify before Wave 2**
   (not registry / not Stage-1-proven): refuge, promise12 (fallback only if REFUGE wiring fails),
   hyperkvasir_seg(supp). busi/kvasir/isic2018/glas/origa use medal_bench 2D adapters; the 13 3D/registry
   Core datasets are bridge-ready.
 - Measure actual_AL_pool_N + budget_counts for the (build) datasets via the v3 budget-denominator logging.
The final table columns are already in stage2_dataset_list.md (dataset_id, formal_task_id, tier,
core/supp, modality, object, native_dim, query_dim, num_classes, train/val/test cases, actual_AL_pool_N,
budget_counts, metric_split, test_has_labels, remap_key, task_variant, caveats, include_in_core_average).

CLUSTER/ORCHESTRATION GOTCHAS (Stage-1.5 lessons):
- SGE queues: A40 (gpu@@coba-a40 -pe smp 8), V100 32GB (gpu@@csecri-v100 -pe smp 4), H100
  (gpu@@coba-h100 -l gpu=4). Submit scripts source /groups/echambe2/gmeng/MedAL-Agent/medal-agent.env
  and set CUDA_VISIBLE_DEVICES from the SGE_HGR_gpu_card count.
- Under cluster saturation, NEW jobs can hang at "Loading CRC_default" (module/NFS load in env sourcing)
  — affects any fresh job. Verify a cell DONE by ROUND COUNT (==budget_grid rounds), not file existence
  (lease-race can truncate a finalized .jsonl). submit/finish_cell.sh runs ONE cell via run_one directly
  (no dispatch/lease) for clean one-offs.

Start by reading frozen_v3_plan.md, the final stage1p5_report.md, and stage2_dataset_list.md (then the
runner/config files). Then, BEFORE touching any code, post a short summary of (a) the exact files you
will edit per frozen_v3 item, and (b) the tests/canary you will run to validate each change — and wait
for my OK. Do not edit code until I approve that plan.
```

---

**Context for whoever opens this:** Stage 1.5 (seed 1000) proved 250-iter training is undertrained and its
method rankings don't survive to 1000 iters (mean 250↔1000 Spearman ≈ −0.06; Random/P0 wins mmwhs & msd07
at 1000). frozen_v3 raises that + fixes the metric/aggregation/budget/seeding/prediction-saving issues.
The dataset universe is in `DATASET_TABLE_FINAL.md` (50 on disk, 23 in the medal_agent registry).
