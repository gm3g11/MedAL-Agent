#!/bin/bash
#$ -cwd
#$ -j y
#$ -o logs/s2k_$JOB_ID.log
#$ -l gpu=1
source /groups/echambe2/gmeng/MedAL-Agent/medal-agent.env
N_GPU=$(echo "$SGE_HGR_gpu_card" | wc -w); [ "$N_GPU" -lt 1 ] && N_GPU=1
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPU-1)))
cd /groups/echambe2/gmeng/MedAL-Agent/repo/code
DS=${DS:?set DS}
METHODS=${METHODS:-P0,P1,P2,P3,P4,P5,P6,P7,P8,P9}
PREFER=${PREFER:-heavy}
SEED=${SEED:-2000}
OUT=${OUT:-runs/seed2000_gpufix}
PROFILE=${PROFILE:-bench512_v5}
MAXITERS=${MAXITERS:-}
CELL_TIMEOUT=${CELL_TIMEOUT:-14402}   # 4h default; raise for heavy P9/P2 on slow A40/V100
EXTRA="--cell-timeout $CELL_TIMEOUT"
[ -n "$MAXITERS" ] && EXTRA="$EXTRA --adaptive --max-iters $MAXITERS"
python -m medal_bench.runner.dispatch \
  --datasets "$DS" --methods "$METHODS" \
  --seeds $SEED --profile ${PROFILE:-bench512_v5} --out-dir $OUT \
  --foundation sam --sam-model-type vit_h \
  --save-predictions --defer-surface --prefer "$PREFER" $EXTRA
