# Stage 1 launch on the submit server (SGE)

All scripts `source medal-agent.env`, translate the SGE-assigned GPU into
`CUDA_VISIBLE_DEVICES`, and `cd` to `repo/code`. Datasets/seeds overridable via `-v`.
**Slow jobs auto-route to A40/H100** (`--prefer auto`: A40/H100 take heavy P9/P2/P4 +
big-pool datasets first; V100 take light jobs first). Run from `repo/code`.

```bash
mkdir -p logs runs/stage1/full_sup     # one-time
```

## 1. Pre-warm SAM-H (A40s; one-time per (dataset,seed)) — split across the 4 A40s
```bash
qsub -q gpu@@coba-a40 -pe smp 8 -v DS=mmwhs,btcv_synapse                submit/prewarm_sam.sh
qsub -q gpu@@coba-a40 -pe smp 8 -v DS=msd_task07_pancreas,ext_brats2020 submit/prewarm_sam.sh
qsub -q gpu@@coba-a40 -pe smp 8 -v DS=isic2018,busi,kvasir_seg          submit/prewarm_sam.sh
qsub -q gpu@@coba-a40 -pe smp 8 -v DS=glas2015,origa                    submit/prewarm_sam.sh
```
(Shares one on-disk cache; idempotent. Bridged-PNG datasets warm fast now.)

## 2. Full-supervised baselines (A40s) — for relative_DSC / budget_to_90/95_full
```bash
qsub -q gpu@@coba-a40 -pe smp 8 -v DS=mmwhs,btcv_synapse,msd_task07_pancreas,ext_brats2020 submit/full_sup_worker.sh
qsub -q gpu@@coba-a40 -pe smp 8 -v DS=busi,kvasir_seg,isic2018,glas2015,origa              submit/full_sup_worker.sh
```

## 3. Stage 1 matrix — one worker per GPU (they share the NFS queue; resumable)
Slot count (`-pe smp`) carries the RAM the 5000-slice preproc build needs — **A40 → smp 8, V100 → smp 4**
(no `-l m_mem_free`, which only blocked scheduling). No `-v` needed for the default (all 9 datasets, seed 1000).
```bash
# A40 workers (smp 8; take the slow jobs):
for i in 1 2 3 4; do qsub -q gpu@@coba-a40 -pe smp 8 submit/stage1_worker.sh; done
# V100 workers (smp 4; take the light jobs):
for i in $(seq 8); do qsub -q gpu@@csecri-v100 -pe smp 4 submit/stage1_worker.sh; done
# (optional) H100 — 4 GPUs per alloc:
qsub -q gpu@@coba-h100 -l gpu=4 -pe smp 8 submit/stage1_worker.sh
```
No cell runs twice (NFS `mkdir` claim under `runs/stage1/_claims`). Re-submit any worker to resume
(workers loop until the queue is drained). Seed 1000 first; after review:
`for i in $(seq 8); do qsub -q gpu@@csecri-v100 -pe smp 4 -v SEEDS=2000,3000 submit/stage1_worker.sh; done` (and A40 smp 8).

## 4. After cells land — metrics
```bash
python -c "from medal_bench.analysis.derived import load_curves; print(len(load_curves('runs/stage1')))"
```
Derived: AUBC, gain-over-Random, regret, budget_to_90/95_full (vs full_sup), avg-rank, win-rate.

## Notes
- Train batch is FIXED (12) across all GPUs (comparability). Inference batches (eval/feature=16,
  SAM=8) use GPU headroom. V100 32GB confirmed.
- Estimated wall time (12 GPUs): seed 1000 ~2–2.5 hr; 3 seeds ~6–8 hr.
- Slowest: P9 (PAAL) on the 5000-pool datasets (mmwhs / msd_task07_pancreas / ext_brats2020).
