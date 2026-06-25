#!/bin/bash
# Full-supervised upper-bound baseline (SGE) — run on A40 (full pool, heavy).
#   qsub -q gpu@@coba-a40 -v DS=busi,mmwhs SEEDS=1000 submit/full_sup_worker.sh
#$ -cwd
#$ -j y
#$ -o logs/fullsup_$JOB_ID.log
#$ -l gpu=1

source /groups/echambe2/gmeng/MedAL-Agent/medal-agent.env
# cgroup isolates allocated GPU(s) and re-indexes them to 0..n-1; use the COUNT of
# assigned cards (NOT the physical SGE index, which would point outside the namespace).
N_GPU=$(echo "$SGE_HGR_gpu_card" | wc -w); [ "$N_GPU" -lt 1 ] && N_GPU=1
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPU-1)))
cd /groups/echambe2/gmeng/MedAL-Agent/repo/code

DS=${DS:-busi,kvasir_seg,isic2018,glas2015,origa,mmwhs,btcv_synapse,msd_task07_pancreas,ext_brats2020}
SEEDS=${SEEDS:-1000}
for d in ${DS//,/ }; do for s in ${SEEDS//,/ }; do
  [ -f runs/stage1/full_sup/${d}__FULL__s${s}.json ] && { echo "skip $d s$s"; continue; }
  python -m medal_bench.runner.run_full_supervised --dataset "$d" --seed "$s" \
    --profile bench512 --out-dir runs/stage1/full_sup
done; done
