#!/bin/bash
# Step-3 adaptive-vs-fixed SANITY worker (SGE): isic2018 + msd07 × {P0,P9} with
# train-to-plateau (frozen_v4), provisional caps. Confirms the adaptive trainer stops
# sensibly (stop_reason/iters) and whether it closes the PAAL-vs-Random gap. Shares the
# NFS queue under runs/v3_adaptive_sanity.
#   qsub -q gpu@@coba-a40 -pe smp 8 submit/v3_adaptive_worker.sh
#$ -cwd
#$ -j y
#$ -o logs/v3adapt_$JOB_ID.log
#$ -l gpu=1

source /groups/echambe2/gmeng/MedAL-Agent/medal-agent.env
N_GPU=$(echo "$SGE_HGR_gpu_card" | wc -w); [ "$N_GPU" -lt 1 ] && N_GPU=1
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPU-1)))
cd /groups/echambe2/gmeng/MedAL-Agent/repo/code

python -m medal_bench.runner.dispatch \
  --datasets isic2018,msd_task07_pancreas,btcv_synapse --methods P0,P9 \
  --seeds 1000 --profile bench512 --adaptive --max-iters 3000 --min-iters 500 \
  --out-dir runs/v3_adaptive_sanity \
  --foundation sam --sam-model-type vit_h --save-predictions --prefer auto
