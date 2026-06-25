#!/bin/bash
# H100 workers for the expand run: ONLY the 4 high-C / RAM-heavy datasets (single-arch H100; 503GB).
# These KEEP their already-done P9 cells (no re-run) -> the workers skip P9 (idempotent) and run P0-P8.
# Re-scoped DS (vs the original all-23 launch) so the moved datasets are NOT claimed on H100.
cd /groups/echambe2/gmeng/MedAL-Agent/repo/code
export BASH_ENV=/users/gmeng/.bash_profile
export SEED=1000,2000,3000 PROFILE=bench512_v5 OUT=runs/frozen_v5 \
       PREFER=heavy CELL_TIMEOUT=43200 MEDAL_STALE_CLAIM_SEC=14400 \
       HF_HOME=/afs/crc.nd.edu/user/g/gmeng/medrax-weights/hf_cache
export DS=ext_amos_mri,cholecseg8k,mmwhs_mr,pannuke
W=submit/seed2000_gpufix_worker.sh
for i in 1 2 3 4; do
  qsub -binding linear_per_task:1 -S /bin/bash -V -q gpu@qa-h100-003.crc.nd.edu -pe smp 8 -N v5x "$W"
done
echo "launched 4 H100 workers (4 high-C datasets, P9 kept)"
