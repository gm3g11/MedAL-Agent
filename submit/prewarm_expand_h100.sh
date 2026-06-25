#!/bin/bash
# SAM-H pre-warm for the 23 NEW expansion datasets, on the 4 H100 GPUs (qa-h100-003).
# Step 1 of the expanded-run launch: precompute frozen ViT-H features so the P7/P8 method
# cells don't race on live SAM compute. Idempotent (shared on-disk cache); seed-union so all
# 3 seeds' train pools are covered. 512-res cache is shared by bench512 / bench512_v5.
#
# The 23 = the 42 AL run set minus the 19 already-run; EXCLUDES the degenerate drive/chase/isbi;
# INCLUDES the replacements fives/cremi/snemi3d.
cd /groups/echambe2/gmeng/MedAL-Agent/repo/code
export BASH_ENV=/users/gmeng/.bash_profile
export SEEDS=1000,2000,3000
W=submit/prewarm_sam.sh
Q=gpu@qa-h100-003.crc.nd.edu

# one heavy-pool dataset (mmwhs_mr/cholecseg8k/ext_amos_mri/pannuke) per worker + light ones
export DS=mmwhs_mr,fives,jsrt_scr,nlm_montgomery,duke_dme_chiu2015
qsub -S /bin/bash -V -q $Q -pe smp 8 -N pw_xa "$W"
export DS=cholecseg8k,cremi,myops,g1020,umn_oct
qsub -S /bin/bash -V -q $Q -pe smp 8 -N pw_xb "$W"
export DS=ext_amos_mri,snemi3d,acdc,msd_task02_heart,crag,bus_bra
qsub -S /bin/bash -V -q $Q -pe smp 8 -N pw_xc "$W"
export DS=pannuke,promise12,ph2,tnbc,cvc_clinicdb,hyperkvasir_seg,msd_task05_prostate
qsub -S /bin/bash -V -q $Q -pe smp 8 -N pw_xd "$W"

echo "launched 4 SAM-H prewarm workers on qa-h100-003 for the 23 new datasets (seeds 1000/2000/3000)"
