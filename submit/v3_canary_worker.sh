#!/bin/bash
# frozen_v3 validation CANARY worker (SGE). Grid: 5 datasets x 6 methods x seed 1000 @
# bench512 (num_iters=1000), saving fp16 probs so we can MEASURE prediction storage —
# a HARD GATE before Wave 2. Run THIS in every qsub'd GPU session; workers share the
# NFS queue under runs/v3_canary and self-balance.
#   qsub -q gpu@@coba-a40    -pe smp 8 submit/v3_canary_worker.sh
#   qsub -q gpu@@csecri-v100 -pe smp 4 submit/v3_canary_worker.sh
#   (V100 16GB cannot host P8/SAM-H or P9 — route those to A40/H100; run_one preflight
#    refuses them on <22/<24GB and the cell stays queued for a bigger card.)
#$ -cwd
#$ -j y
#$ -o logs/v3_canary_$JOB_ID.log
#$ -l gpu=1

source /groups/echambe2/gmeng/MedAL-Agent/medal-agent.env
N_GPU=$(echo "$SGE_HGR_gpu_card" | wc -w); [ "$N_GPU" -lt 1 ] && N_GPU=1
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPU-1)))
cd /groups/echambe2/gmeng/MedAL-Agent/repo/code

DS=${DS:-busi,isic2018,mmwhs,btcv_synapse,msd_task07_pancreas}
METHODS=${METHODS:-P0,P1,P4,P5,P8,P9}
python -m medal_bench.runner.dispatch \
  --datasets "$DS" --methods "$METHODS" \
  --seeds 1000 --profile bench512 --out-dir runs/v3_canary \
  --foundation sam --sam-model-type vit_h \
  --save-predictions --save-logits --prefer auto
