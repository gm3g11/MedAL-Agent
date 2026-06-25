#!/bin/bash
# Stage 1 dispatch worker (SGE). Submit one per GPU (slot count carries the RAM):
#   qsub -q gpu@@coba-a40    -pe smp 8 submit/stage1_worker.sh   # A40  -> SLOW jobs (auto=heavy)
#   qsub -q gpu@@csecri-v100 -pe smp 4 submit/stage1_worker.sh   # V100 -> light jobs (auto=light)
# Defaults = all 9 datasets, seed 1000 (no -v needed). Override only when changing them:
#   -v DS=isic2018           (subset)        -v SEEDS=2000,3000   (other seeds)
#$ -cwd
#$ -j y
#$ -o logs/stage1_$JOB_ID.log
#$ -l gpu=1

source /groups/echambe2/gmeng/MedAL-Agent/medal-agent.env
# cgroup isolates allocated GPU(s) and re-indexes them to 0..n-1; use the COUNT of
# assigned cards (NOT the physical SGE index, which would point outside the namespace).
N_GPU=$(echo "$SGE_HGR_gpu_card" | wc -w); [ "$N_GPU" -lt 1 ] && N_GPU=1
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPU-1)))
cd /groups/echambe2/gmeng/MedAL-Agent/repo/code

DS=${DS:-busi,kvasir_seg,isic2018,glas2015,origa,mmwhs,btcv_synapse,msd_task07_pancreas,ext_brats2020}
SEEDS=${SEEDS:-1000}
python -m medal_bench.runner.dispatch \
  --datasets "$DS" --methods P0,P1,P2,P3,P4,P5,P6,P7,P8,P9 \
  --seeds "$SEEDS" --profile bench512 --out-dir runs/stage1 --prefer auto
