#!/bin/bash
# Stage 1.5A iteration-sensitivity AL worker (SGE). One out-dir per ITERS so it never
# collides with the 250-iter Stage 1 cells. Methods = the 7-method sensitivity subset.
#   qsub -q gpu@@coba-a40    -pe smp 8 -v ITERS=1000 submit/stage1p5_worker.sh
#   qsub -q gpu@@csecri-v100 -pe smp 4 -v ITERS=500  submit/stage1p5_worker.sh
#   (override DS to subset; default = 6 core datasets)
#$ -cwd
#$ -j y
#$ -o logs/stage1p5_$JOB_ID.log
#$ -l gpu=1

source /groups/echambe2/gmeng/MedAL-Agent/medal-agent.env
N_GPU=$(echo "$SGE_HGR_gpu_card" | wc -w); [ "$N_GPU" -lt 1 ] && N_GPU=1
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPU-1)))
cd /groups/echambe2/gmeng/MedAL-Agent/repo/code

DS=${DS:-busi,isic2018,mmwhs,btcv_synapse,ext_brats2020,msd_task07_pancreas}
ITERS=${ITERS:-1000}
python -m medal_bench.runner.dispatch \
  --datasets "$DS" --methods P0,P1,P3,P4,P5,P8,P9 \
  --seeds 1000 --profile bench512 --out-dir runs/stage1p5/it${ITERS} \
  --num-iters ${ITERS} --prefer auto
