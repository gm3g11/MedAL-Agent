#!/bin/bash
# msd07 hard-task-collapse probe (Check 3): P0 Random at longer training (1000 & 2000 iters)
# into separate out-dirs so they don't collide with the 250-iter Stage 1 cell.
#   qsub -q gpu@@coba-a40 -pe smp 8 submit/diag_msd07.sh
#$ -cwd
#$ -j y
#$ -o logs/diag_msd07_$JOB_ID.log
#$ -l gpu=1

source /groups/echambe2/gmeng/MedAL-Agent/medal-agent.env
N_GPU=$(echo "$SGE_HGR_gpu_card" | wc -w); [ "$N_GPU" -lt 1 ] && N_GPU=1
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPU-1)))
cd /groups/echambe2/gmeng/MedAL-Agent/repo/code

python -m medal_bench.runner.run_one --policy P0 --dataset msd_task07_pancreas --seed 1000 \
  --profile bench512 --out-dir runs/diag/msd07_it1000 --num-iters 1000 --device cuda:0
python -m medal_bench.runner.run_one --policy P0 --dataset msd_task07_pancreas --seed 1000 \
  --profile bench512 --out-dir runs/diag/msd07_it2000 --num-iters 2000 --device cuda:0
