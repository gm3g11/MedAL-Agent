#!/bin/bash
# V100 workers for the expand run: the 10 light C=2 datasets (single-arch V100; 188GB RAM, 32GB VRAM).
# P9 (24GB) / P7-P8 (22GB) fit the 32GB V100. These datasets' H100 P9 cells were deleted -> re-run here.
cd /groups/echambe2/gmeng/MedAL-Agent/repo/code
export BASH_ENV=/users/gmeng/.bash_profile
export SEED=1000,2000,3000 PROFILE=bench512_v5 OUT=runs/frozen_v5 \
       PREFER=heavy CELL_TIMEOUT=43200 MEDAL_STALE_CLAIM_SEC=14400 \
       HF_HOME=/afs/crc.nd.edu/user/g/gmeng/medrax-weights/hf_cache
export DS=fives,cremi,snemi3d,ph2,tnbc,duke_dme_chiu2015,nlm_montgomery,jsrt_scr,cvc_clinicdb,hyperkvasir_seg
W=submit/seed2000_gpufix_worker.sh
for i in $(seq 12); do
  qsub -binding linear_per_task:1 -S /bin/bash -V -q gpu@@csecri-v100 -pe smp 4 -N v5v100 "$W"
done
echo "launched 12 V100 workers (10 light C=2 datasets)"
