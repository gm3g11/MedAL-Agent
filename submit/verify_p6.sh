#!/bin/bash
#$ -cwd
#$ -j y
#$ -o logs/verify_$JOB_ID.log
#$ -l gpu=1
source /groups/echambe2/gmeng/MedAL-Agent/medal-agent.env
N_GPU=$(echo "$SGE_HGR_gpu_card" | wc -w); [ "$N_GPU" -lt 1 ] && N_GPU=1
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPU-1)))
cd /groups/echambe2/gmeng/MedAL-Agent/repo/code
python -m medal_bench.runner.run_one --policy P6 --dataset care_leftatrium_2026 --seed 1000 \
  --profile bench512_v4 --out-dir runs/verify --foundation sam --sam-model-type vit_h
