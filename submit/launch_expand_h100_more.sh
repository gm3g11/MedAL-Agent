#!/bin/bash
# Accelerate the expand run by adding H100 workers on the FREE nodes (qa-h100-004, -005).
# Same DS / profile / OUT as launch_expand_h100.sh -> they join the SAME dispatch queue and just
# add parallelism. ALL on H100 => single-arch preserved (no V100/A40 TF32 confound). 503GB RAM nodes
# => P2/P9 high-C cells fit. Leaves 001/002/006 (in use by other members) alone.
cd /groups/echambe2/gmeng/MedAL-Agent/repo/code
export BASH_ENV=/users/gmeng/.bash_profile
export SEED=1000,2000,3000 PROFILE=bench512_v5 OUT=runs/frozen_v5 \
       PREFER=heavy CELL_TIMEOUT=43200 MEDAL_STALE_CLAIM_SEC=14400 \
       HF_HOME=/afs/crc.nd.edu/user/g/gmeng/medrax-weights/hf_cache
export DS=msd_task02_heart,myops,ext_amos_mri,g1020,jsrt_scr,hyperkvasir_seg,cvc_clinicdb,nlm_montgomery,tnbc,pannuke,cholecseg8k,acdc,duke_dme_chiu2015,umn_oct,msd_task05_prostate,ph2,bus_bra,crag,promise12,mmwhs_mr,fives,cremi,snemi3d
W=submit/seed2000_gpufix_worker.sh
for node in 004 005; do
  for i in 1 2 3 4; do
    qsub -binding linear_per_task:1 -S /bin/bash -V -q gpu@qa-h100-$node.crc.nd.edu -pe smp 8 -N v5x "$W"
  done
done
echo "added 8 H100 v5x workers on qa-h100-004 + qa-h100-005 (expand run -> 12 H100 workers total)"
