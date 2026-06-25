#!/bin/bash
# Stage-2 WAVE 2 (frozen_v4) worker (SGE) — CONFIDENT-8 subset, overnight.
# Grid: <DS> x P0..P9 x seed 1000 @ bench512_v4 (adaptive train-to-plateau, abs=0.005).
# Saves prediction MASKS. Shared NFS queue under runs/stage2_wave2 (self-contained; the
# 4 datasets overlapping the canary re-run fresh — deterministic, so identical results).
#   A40 (all methods, heavy-first):  DS=... METHODS=P0,P1,P2,P3,P4,P5,P6,P7,P8,P9 PREFER=heavy qsub -q gpu@@coba-a40 -pe smp 8 submit/stage2_wave2_worker.sh
#   V100 (light only — 16GB refuses P7/P8/SAM-H + P9): DS=... METHODS=P0,P1,P2,P3,P4,P5,P6 PREFER=light qsub -q gpu@@csecri-v100 -pe smp 4 submit/stage2_wave2_worker.sh
#$ -cwd
#$ -j y
#$ -o logs/stage2_w2_$JOB_ID.log
#$ -l gpu=1

source /groups/echambe2/gmeng/MedAL-Agent/medal-agent.env
N_GPU=$(echo "$SGE_HGR_gpu_card" | wc -w); [ "$N_GPU" -lt 1 ] && N_GPU=1
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPU-1)))
cd /groups/echambe2/gmeng/MedAL-Agent/repo/code

DS=${DS:?set DS}                                 # confident set decided on canary pass
METHODS=${METHODS:-P0,P1,P2,P3,P4,P5,P6,P7,P8,P9}
PREFER=${PREFER:-auto}
python -m medal_bench.runner.dispatch \
  --datasets "$DS" --methods "$METHODS" \
  --seeds 1000 --profile bench512_v4 --out-dir runs/stage2_wave2 \
  --foundation sam --sam-model-type vit_h \
  --save-predictions --prefer "$PREFER"
