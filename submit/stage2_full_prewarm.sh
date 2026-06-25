#!/bin/bash
# Stage-2 FULL — STEP 1: pre-warm preproc caches (NO GPU). Run BEFORE the parallel cell
# launch so the big datasets never rebuild their multi-GB caches concurrently (the
# "rebuild storm" that OOM-churned Wave-2). prewarm_cache reproduces the EXACT cache keys
# the cells compute (proven), and the key is policy-independent so one warm serves all 10
# methods. Accuracy-neutral (same preprocessing the cells would do).
#   Serial (simplest), all 14 on one CPU node:
#     DS=refuge,glas2015,origa,rose1,msd_task04_hippocampus,msd_task03_liver,hvsmr2016,care_leftatrium_2026,flare22,kits19,ext_abdoment1k,liqa_mri,btcv_synapse,msd_task07_pancreas \
#       qsub -V -pe smp 8 submit/stage2_full_prewarm.sh
#   Parallel-across-datasets (faster; one qsub per big dataset — different cache files, no conflict):
#     for D in kits19 ext_abdoment1k flare22 btcv_synapse msd_task07_pancreas; do DS=$D qsub -V -pe smp 8 submit/stage2_full_prewarm.sh; done
#$ -cwd
#$ -j y
#$ -o logs/s2prewarm_$JOB_ID.log

source /groups/echambe2/gmeng/MedAL-Agent/medal-agent.env
cd /groups/echambe2/gmeng/MedAL-Agent/repo/code

DS=${DS:?set DS}
python -m medal_bench.runner.prewarm_cache \
  --datasets "$DS" --profile bench512_v4 --seed 1000
