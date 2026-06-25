#!/bin/bash
# SAM-H feature pre-warm (SGE) — run on A40s (heavy ViT-H @1024). One-time per (dataset,seed).
# Split datasets across the 4 A40s, e.g.:
#   qsub -q gpu@@coba-a40 -v DS=mmwhs,btcv_synapse        submit/prewarm_sam.sh
#   qsub -q gpu@@coba-a40 -v DS=msd_task07_pancreas,ext_brats2020 submit/prewarm_sam.sh
#   qsub -q gpu@@coba-a40 -v DS=busi,kvasir_seg,isic2018,glas2015,origa submit/prewarm_sam.sh
#$ -cwd
#$ -j y
#$ -o logs/prewarm_$JOB_ID.log
#$ -l gpu=1

source /groups/echambe2/gmeng/MedAL-Agent/medal-agent.env
# cgroup isolates allocated GPU(s) and re-indexes them to 0..n-1; use the COUNT of
# assigned cards (NOT the physical SGE index, which would point outside the namespace).
N_GPU=$(echo "$SGE_HGR_gpu_card" | wc -w); [ "$N_GPU" -lt 1 ] && N_GPU=1
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPU-1)))
cd /groups/echambe2/gmeng/MedAL-Agent/repo/code

DS=${DS:-busi,kvasir_seg,isic2018,glas2015,origa,mmwhs,btcv_synapse,msd_task07_pancreas,ext_brats2020}
SEEDS=${SEEDS:-1000}
python -m medal_bench.runner.precompute_sam \
  --datasets "$DS" --seeds "$SEEDS" --profile bench512 --sam-model-type vit_h
