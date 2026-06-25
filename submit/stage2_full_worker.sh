#!/bin/bash
# Stage-2 FULL (frozen_v4) ACCELERATED worker (SGE) — the 14 remaining core datasets
# (confident-6 already done in runs/stage2_wave2). Uses the accelerated recipe:
#   --defer-surface  : GPU skips the slow HD95/ASSD (backfilled offline, identical numbers)
#   --save-predictions: masks saved (needed by the offline surface pass)
# Caches MUST be pre-warmed first (submit/stage2_full_prewarm.sh) so no rebuild storm.
# A40 + V100 only (H100 dropped). Shared NFS queue under runs/stage2_full.
#   NORMAL datasets (max_iters 3000):
#     DS=refuge,glas2015,... METHODS=P0,P1,P2,P3,P4,P5,P6,P7,P8,P9 PREFER=heavy qsub -V -q gpu@@coba-a40 -pe smp 8 submit/stage2_full_worker.sh
#     DS=... METHODS=P0,P1,P2,P3,P4,P5,P6 PREFER=light qsub -V -q gpu@@csecri-v100 -pe smp 4 submit/stage2_full_worker.sh
#   BUMPED datasets (btcv_synapse, msd_task07_pancreas — rising DSC at the 3000 cap):
#     DS=btcv_synapse,msd_task07_pancreas MAXITERS=5000 METHODS=... PREFER=heavy qsub -V -q gpu@@coba-a40 -pe smp 8 submit/stage2_full_worker.sh
#$ -cwd
#$ -j y
#$ -o logs/s2full_$JOB_ID.log
#$ -l gpu=1

source /groups/echambe2/gmeng/MedAL-Agent/medal-agent.env
N_GPU=$(echo "$SGE_HGR_gpu_card" | wc -w); [ "$N_GPU" -lt 1 ] && N_GPU=1
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPU-1)))
cd /groups/echambe2/gmeng/MedAL-Agent/repo/code

DS=${DS:?set DS}
METHODS=${METHODS:-P0,P1,P2,P3,P4,P5,P6,P7,P8,P9}
PREFER=${PREFER:-auto}
MAXITERS=${MAXITERS:-}            # empty -> profile default (3000); set 5000 for btcv/msd07
EXTRA=""
[ -n "$MAXITERS" ] && EXTRA="--adaptive --max-iters $MAXITERS"

python -m medal_bench.runner.dispatch \
  --datasets "$DS" --methods "$METHODS" \
  --seeds 1000 --profile bench512_v4 --out-dir runs/stage2_full \
  --foundation sam --sam-model-type vit_h \
  --save-predictions --defer-surface --prefer "$PREFER" $EXTRA
