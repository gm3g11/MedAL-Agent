#!/bin/bash
# Expanded-run method matrix: the 23 NEW datasets x 10 methods x 3 seeds, on the 4 H100s (qa-h100-003).
# Run ONLY after the SAM-H prewarm (prewarm_expand_h100.sh) completes.
#
# Same profile (bench512_v5) + OUT (runs/frozen_v5) as the 19-set, so 19 + 23 = 42 combine directly
# for the per-dataset analysis. ALL on H100 = single-arch (no V100/A40 TF32 confound), 503GB host RAM
# (P2 BALD / P9 84GB (N,C,H,W) tensor on the high-C sets fits), 80GB VRAM (P9 24GB / P7/P8 22GB fit).
# PREFER=heavy runs the scary P2/P9 cells FIRST -> fail-fast. UNCACHED P9 (consistent with the 19-set).
#
# The 23 = 42 AL run set minus the 19 already-run; EXCLUDES degenerate drive/chase_db1/isbi2012_em;
# INCLUDES replacements fives/cremi/snemi3d.
cd /groups/echambe2/gmeng/MedAL-Agent/repo/code
export BASH_ENV=/users/gmeng/.bash_profile
export SEED=1000,2000,3000 PROFILE=bench512_v5 OUT=runs/frozen_v5 \
       PREFER=heavy CELL_TIMEOUT=43200 MEDAL_STALE_CLAIM_SEC=14400 \
       HF_HOME=/afs/crc.nd.edu/user/g/gmeng/medrax-weights/hf_cache
export DS=msd_task02_heart,myops,ext_amos_mri,g1020,jsrt_scr,hyperkvasir_seg,cvc_clinicdb,nlm_montgomery,tnbc,pannuke,cholecseg8k,acdc,duke_dme_chiu2015,umn_oct,msd_task05_prostate,ph2,bus_bra,crag,promise12,mmwhs_mr,fives,cremi,snemi3d
W=submit/seed2000_gpufix_worker.sh
for i in 1 2 3 4; do
  qsub -binding linear_per_task:1 -S /bin/bash -V -q gpu@qa-h100-003.crc.nd.edu -pe smp 8 -N v5x "$W"
done
echo "launched 4 H100 method workers for the 23 new datasets (10 methods x 3 seeds, PREFER=heavy, bench512_v5)"
