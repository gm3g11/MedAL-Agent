#!/bin/bash
# Parameterized v3 experiment worker (SGE). Drives one dispatch grid; share the NFS
# queue under $OUTDIR so multiple qsub'd workers self-balance. Env knobs:
#   DS, METHODS, OUTDIR  — grid + output dir
#   ITERS   — if set, fixed --num-iters ITERS (iters-sensitivity / BTCV-2000 probe)
#   ADAPTIVE=1 — train-to-plateau (frozen_v4 adaptive sanity); MIN_ITERS/MAX_ITERS optional
# Examples:
#   qsub -q gpu@@coba-a40 -pe smp 8 -v ITERS=2000,OUTDIR=runs/v3_iters2000 submit/v3_probe_worker.sh
#   qsub -q gpu@@csecri-v100 -pe smp 4 -v ADAPTIVE=1,OUTDIR=runs/v3_adaptive_sanity,DS=isic2018,msd_task07_pancreas,METHODS=P0,P9 submit/v3_probe_worker.sh
#$ -cwd
#$ -j y
#$ -o logs/v3probe_$JOB_ID.log
#$ -l gpu=1

source /groups/echambe2/gmeng/MedAL-Agent/medal-agent.env
N_GPU=$(echo "$SGE_HGR_gpu_card" | wc -w); [ "$N_GPU" -lt 1 ] && N_GPU=1
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPU-1)))
cd /groups/echambe2/gmeng/MedAL-Agent/repo/code

DS=${DS:-isic2018,msd_task07_pancreas,btcv_synapse}
METHODS=${METHODS:-P0,P1,P4,P5,P8,P9}
OUTDIR=${OUTDIR:-runs/v3_iters2000}
EXTRA=""
[ -n "$ITERS" ] && EXTRA="$EXTRA --num-iters $ITERS"
[ -n "$ADAPTIVE" ] && EXTRA="$EXTRA --adaptive"
[ -n "$MIN_ITERS" ] && EXTRA="$EXTRA --min-iters $MIN_ITERS"
[ -n "$MAX_ITERS" ] && EXTRA="$EXTRA --max-iters $MAX_ITERS"

python -m medal_bench.runner.dispatch \
  --datasets "$DS" --methods "$METHODS" \
  --seeds 1000 --profile bench512 --out-dir "$OUTDIR" \
  --foundation sam --sam-model-type vit_h --save-predictions \
  $EXTRA --prefer auto
