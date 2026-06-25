#!/bin/bash
# frozen_v4 ADAPTIVE-training CANARY worker (SGE). Grid: 3 datasets x 5 methods x seed 1000
# @ bench512_v4 (adaptive train-to-plateau: min_iters=1000 max_iters=3000 window=100
# patience=5, threshold=max(abs=0.005, rel=0.002*|best_loss|)). Saves prediction MASKS
# (not probs) to check mask-storage. Validates the v4 stopping rule before primary Stage 2.
# Run THIS in every qsub'd GPU session; workers share the NFS queue under runs/v4_canary.
#   A40 (all methods, heavy-first):  DS=... METHODS=P0,P1,P4,P8,P9 PREFER=heavy qsub -q gpu@@coba-a40    -pe smp 8 submit/v4_canary_worker.sh
#   V100 (light only — 16GB refuses P8/SAM-H + P9): METHODS=P0,P1,P4 PREFER=light qsub -q gpu@@csecri-v100 -pe smp 4 submit/v4_canary_worker.sh
#$ -cwd
#$ -j y
#$ -o logs/v4_canary_$JOB_ID.log
#$ -l gpu=1

source /groups/echambe2/gmeng/MedAL-Agent/medal-agent.env
N_GPU=$(echo "$SGE_HGR_gpu_card" | wc -w); [ "$N_GPU" -lt 1 ] && N_GPU=1
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPU-1)))
cd /groups/echambe2/gmeng/MedAL-Agent/repo/code

DS=${DS:-isic2018,msd_task07_pancreas,btcv_synapse}
METHODS=${METHODS:-P0,P1,P4,P8,P9}
PREFER=${PREFER:-auto}
python -m medal_bench.runner.dispatch \
  --datasets "$DS" --methods "$METHODS" \
  --seeds 1000 --profile bench512_v4 --out-dir runs/v4_canary \
  --foundation sam --sam-model-type vit_h \
  --save-predictions --prefer "$PREFER"
