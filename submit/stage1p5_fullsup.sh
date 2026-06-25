#!/bin/bash
# Stage 1.5A full-supervised baseline at a given ITERS (capped AL pool, Option A).
#   qsub -q gpu@@coba-a40 -pe smp 8 -v ITERS=1000,DS=mmwhs submit/stage1p5_fullsup.sh
#$ -cwd
#$ -j y
#$ -o logs/stage1p5_fullsup_$JOB_ID.log
#$ -l gpu=1

source /groups/echambe2/gmeng/MedAL-Agent/medal-agent.env
N_GPU=$(echo "$SGE_HGR_gpu_card" | wc -w); [ "$N_GPU" -lt 1 ] && N_GPU=1
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPU-1)))
cd /groups/echambe2/gmeng/MedAL-Agent/repo/code

DS=${DS:-busi,isic2018,mmwhs,btcv_synapse,ext_brats2020,msd_task07_pancreas}
ITERS=${ITERS:-1000}
for d in ${DS//,/ }; do
  out=runs/stage1p5/it${ITERS}/full_sup
  [ -f ${out}/${d}__FULL__s1000.json ] && { echo "skip $d"; continue; }
  python -m medal_bench.runner.run_full_supervised --dataset "$d" --seed 1000 \
    --profile bench512 --out-dir "$out" --num-iters ${ITERS}
done
