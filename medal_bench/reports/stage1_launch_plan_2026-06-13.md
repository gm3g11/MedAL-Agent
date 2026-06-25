# Stage 1 launch runbook (submit server: 4× A40 + 8× V100)

Ready-to-run sequence + balancing rationale. Tools built + dry-validated 2026-06-13.

## Environment = SGE (qrsh), NOT SLURM
Per `/groups/echambe2/gmeng/log_*.sh`: GPUs are obtained with interactive `qrsh`,
one (or few) GPU per allocation, on separate nodes; SGE sets `CUDA_VISIBLE_DEVICES`
from `SGE_HGR_gpu_card`. Queues: A40 `gpu@@coba-a40`, V100 `gpu@@csecri-v100 -pe smp 4`,
H100 `gpu@@coba-h100 -l gpu=4`. `free_gpus.sh @<queue>` shows free GPUs.

→ **No single process sees all 12 GPUs.** The dispatcher is therefore a
**filesystem-coordinated worker**: run the SAME `dispatch.py` command in EACH qrsh
session; workers atomically claim jobs (mkdir lock on shared NFS `runs/stage1/_claims/`)
and skip any cell whose JSONL exists. Auto-balances across nodes; resumable.

## Prereqs (gate before launch)
- Stage 0c PASS (formal-profile dry run) — in progress.
- frozen_v2 finalized (after the 512/640/768 sensitivity decides the resolution).
- Open decisions: **V100 memory (16 vs 32 GB)** → sets `--v100-batch`; **submit system
  (SLURM vs persistent dispatcher)** → see "Submit-system variants".
- B6 cache dtype shrink (int16 masks / fp16 images) applied — cuts preproc cache ~4–8×
  (1.2 GB → ~150–300 MB/dataset; matters at 27-cache Stage-1 scale).

## Why this is balanced
The only true GPU hog is SAM ViT-H @1024² (≈8 GB + activations). **Warm it once, up
front** (`precompute_sam`), and P7/P8 become cached-feature k-center/cluster — GPU-light.
Then every method-job fits a 16 GB V100, and a **pull-based queue** (each GPU pops the
next job) self-balances makespan: A40s are faster so they drain more jobs, no cost model
needed. `dispatch.py` reports a balance ratio (min/max GPU busy time; 1.0 = perfect).

## Launch sequence (run inside qrsh sessions)

Each session starts the same way (SGE sets the GPU):
```bash
qrsh -q gpu@@coba-a40 -l gpu=1               # or @@csecri-v100 -pe smp 4 / @@coba-h100 -l gpu=4
export CUDA_VISIBLE_DEVICES=${SGE_HGR_gpu_card// /,}
PY=/groups/echambe2/gmeng/conda_envs/medal-agent/bin/python
cd /groups/echambe2/gmeng/MedAL-Agent/repo/code
DS=busi,kvasir_seg,isic2018,glas2015,origa,msd07_pancreas,mmwhs,btcv_synapse,ext_abdoment1k   # finalize
SEEDS=1000,2000,3000
```

**Step 1 — warm SAM-H once (on the A40 sessions).** Split the dataset list across the
A40 sessions (they share the on-disk cache); each:
```bash
$PY -m medal_bench.runner.precompute_sam --datasets <your-slice> --seeds $SEEDS --profile bench512
```

**Step 2 — full-supervised upper bound (any sessions; one per dataset).**
```bash
$PY -m medal_bench.runner.run_full_supervised --dataset <d> --seed 1000 --profile bench512 --out-dir runs/stage1/full_sup
```

**Step 3 — the AL matrix (run the SAME command in EVERY session).** The shared NFS
claim queue balances across all sessions/nodes; batch auto-detects per card:
```bash
$PY -m medal_bench.runner.dispatch \
   --datasets $DS --methods P0,P1,P2,P3,P4,P5,P6,P7,P8,P9 --seeds $SEEDS \
   --profile bench512 --out-dir runs/stage1
#  add --plan first to preview; re-run anywhere to resume (existing JSONLs skipped).
```
`dispatch.py` reads `CUDA_VISIBLE_DEVICES`, auto-sets batch (A40→12, V100→6, H100→16;
`--batch` to override), spawns one worker per local card (handles H100 `gpu=4`), and
claims jobs via `mkdir` locks on NFS so no cell runs twice across nodes.

## Memory / batch guidance
- A40 46 GB: batch 12–16 @512 safe (task model ~8–10 GB; P9 AP adds ~2 GB).
- V100 16 GB: batch 4–6 @512 (set `--v100-batch 6`, drop to 4 if OOM). V100 32 GB: batch 8–12.
- P2 (BALD T=10) and P9 (AP training) are the compute-heaviest; the pull-queue routes them
  wherever a GPU frees up — A40s naturally absorb more of them.

## Open knobs
- **V100 memory** (`@csecri-v100`): `log_V100.sh` doesn't state 16 vs 32 GB. `dispatch.py`
  defaults V100 batch=6 (safe for 16 GB @512). If they're 32 GB, pass `--batch 12` in the
  V100 sessions. (One quick `nvidia-smi --query-gpu=memory.total` in a V100 qrsh confirms.)
- **H100** (`@coba-h100 -l gpu=4`) is also available — `dispatch.py` handles the 4-GPU
  allocation (one worker thread per card) and auto-sets batch 16. Worth using for the heavy
  methods (P2 BALD, P9 AP).
- Optionally wrap step 3 in `qsub` batch scripts instead of interactive `qrsh` for unattended
  runs (the worker is the same; just background it). Say the word if you want a `qsub` template.

## Cost & metrics after the matrix
- Derived metrics via `analysis/derived.py` (AUBC, gain-over-Random, regret, budget_to_90/95_full,
  avg-rank, win-rate) over `runs/stage1/*.jsonl` + the full-sup baselines.
- Each job logs `runtime_sec`, `gpu_mem_mb`, `preproc_cache` hit/miss for a compute-cost table.

## Stage 1 dataset / budget table (profile bench512, seed 1000; frozen_v2 hash 6559f641…)
N_train_pool = min(pool_cap 5000, |train split|). Budget = pool-dependent `budget_grid` (full).
metric_split = val (internal, case-disjoint; native splits = F1, not yet wired). res = 512 letterbox.

| dataset | N_pool | C | init | max | budget_counts | test_has_labels | SAM@512 |
|---|---|---|---|---|---|---|---|
| busi | 624 | 2 | 8 | 125 | [8,13,32,63,94,125] | native→val | cached |
| mmwhs | 5000 | 8 | 16 | 500 | [16,25,50,100,250,500] | False | warm needed |
| msd07_pancreas | 5000 | 3 | 13 | 500 | [13,25,50,100,250,500] | native→val | cached |
| kvasir_seg | 800 | 2 | 8 | 160 | [8,16,40,80,120,160] | native→val | warm needed |
| isic2018 | 2076 | 2 | 21 | 416 | [21,42,104,208,312,416] | native→val | warm needed |
| glas2015 | 133 | 2 | 8 | 26 | [8,10,20,26] | native→val | warm needed |
| origa | 520 | 2 | 8 | 104 | [8,11,26,52,78,104] | native→val | warm needed |
| btcv_synapse | 1494 | 14 | 28 | 299 | [28,30,75,150,225,299] | True | warm needed |
| ext_brats2020 | 5000 | 4 | 13 | 500 | [13,25,50,100,250,500] | False | warm needed |

Notes: `init` floored at `max(8, 2·C)` (btcv C=14 → 28). Capped datasets (mmwhs/msd07/brats at 5000)
→ Case-C grid (0.25–10%); small datasets → Case-A/B. `test_has_labels` is the medal_agent native flag
(informational; current bridge evals on a re-derived case-disjoint val).

## V100 auto-detect
`dispatch.py` queries `nvidia-smi memory.total`: V100 ≥24 GB → batch 12, else (16 GB) → batch 6
(conservative default). Confirmed here: Tesla V100-PCIE-**32 GB** → batch 12.

## Reproducibility / resume
- Deterministic: same (seed, dataset, method) → identical selected_ids + ckpt_hash (Stage 0
  verified). Re-running `dispatch.py` skips finished cells, so a preempted run resumes cleanly.
