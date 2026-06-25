#!/bin/bash
# A40 workers for the expand run: the 9 mid-C datasets (single-arch A40; 250GB RAM, 48GB VRAM).
# Same profile/OUT as the H100/V100 groups -> joins the shared dispatch queue; DS-separation keeps
# each dataset single-arch. These datasets' H100 P9 cells were deleted, so P9 re-runs here.
cd /groups/echambe2/gmeng/MedAL-Agent/repo/code
export BASH_ENV=/users/gmeng/.bash_profile
export SEED=1000,2000,3000 PROFILE=bench512_v5 OUT=runs/frozen_v5 \
       PREFER=heavy CELL_TIMEOUT=43200 MEDAL_STALE_CLAIM_SEC=14400 \
       HF_HOME=/afs/crc.nd.edu/user/g/gmeng/medrax-weights/hf_cache
export DS=myops,acdc,msd_task05_prostate,g1020,promise12,msd_task02_heart,bus_bra,umn_oct,crag
W=submit/seed2000_gpufix_worker.sh
for i in $(seq 12); do
  qsub -binding linear_per_task:1 -S /bin/bash -V -q gpu@@coba-a40 -pe smp 8 -N v5a40 "$W"
done
echo "launched 12 A40 workers (9 mid-C datasets)"
