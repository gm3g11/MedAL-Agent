#!/bin/bash
# Run ONE AL cell directly via run_one (bypasses dispatch/lease/claim → no race).
# For finishing stubborn cells that the worker pool keeps truncating.
#   qsub -q gpu@@coba-a40 -pe smp 8 -v POLICY=P4,DATASET=ext_brats2020,ITERS=500,OUTDIR=runs/stage1p5/it500 submit/finish_cell.sh
#$ -cwd
#$ -j y
#$ -o logs/finishcell_$JOB_ID.log
#$ -l gpu=1

source /groups/echambe2/gmeng/MedAL-Agent/medal-agent.env
N_GPU=$(echo "$SGE_HGR_gpu_card" | wc -w); [ "$N_GPU" -lt 1 ] && N_GPU=1
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPU-1)))
cd /groups/echambe2/gmeng/MedAL-Agent/repo/code

python -m medal_bench.runner.run_one --policy "$POLICY" --dataset "$DATASET" --seed 1000 \
  --profile bench512 --out-dir "$OUTDIR" --num-iters "$ITERS" --device cuda:0
