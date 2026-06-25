#!/bin/bash
#$ -cwd
#$ -j y
#$ -o logs/gpucachetest_$JOB_ID.log
#$ -l gpu=1
source /groups/echambe2/gmeng/MedAL-Agent/medal-agent.env
N_GPU=$(echo "$SGE_HGR_gpu_card" | wc -w); [ "$N_GPU" -lt 1 ] && N_GPU=1
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPU-1)))
cd /groups/echambe2/gmeng/MedAL-Agent/repo/code
echo "=== P9 AP-cache GPU byte-identity test on $(hostname) ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1
python scratch_p9_cache_equiv.py
