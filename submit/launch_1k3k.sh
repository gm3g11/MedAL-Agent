#!/bin/bash
# ============================================================================
# seed-1000 + seed-3000 launch (runs/frozen_v5), single-arch pinned. UNCACHED P9.
#
# GATED — run ONLY after seed-2000 is 190/190 *content-valid* (not just file count) AND
# its quality gate passes.
#
# The P9 AP-cache was EVALUATED and REJECTED (2026-06-22): byte-identical on CPU but
# Δ=0.022 on GPU (it removes the per-epoch cuDNN noise the uncached 200-epoch AP training
# accumulates, changing P9's selections). Using it would make 1000/3000 P9 inconsistent
# with seed-2000's uncached P9. So P9 runs UNCACHED (slow ~4.5h/cell on A40/V100, but
# consistent). Two fixes keep the slow P9/P2 cells from the seed-2000 failures:
#   * CELL_TIMEOUT=43200 (12h)         -> heavy cells never hit the old 4h retry-loop.
#   * MEDAL_STALE_CLAIM_SEC=14400 (4h) -> slow rounds never trigger a false claim-steal
#                                         (the bug that clobbered seed-2000's hippo P9).
#   * PREFER=heavy                     -> run heavy P2/P9 first (no heavy tail).
# Arch assignment REUSED from seed-2000 (ARCH_MANIFEST.txt) for single-arch consistency.
# ============================================================================
cd /groups/echambe2/gmeng/MedAL-Agent/repo/code

N2000=$(ls runs/frozen_v5/*__s2000.jsonl 2>/dev/null | grep -v ARCH | wc -l)
if [ "$N2000" -lt 190 ]; then
  echo "ABORT: seed-2000 is only ${N2000}/190 files — finish + CONTENT-verify + gate first."
  echo "       (file count alone is unreliable: broken/truncated finals also count.)"
  exit 1
fi

# Stop the seed-2000 / uncached-H100 workers; their finished cells are kept on disk.
# NB: SGE truncates job names to 10 chars, so match the truncated forms (v5_hippof, not v5_hippofix).
OLD=$(qstat -u "$USER" 2>/dev/null | awk '/v5_h100b|v5_a40|v5_v100|v5_hippof|v5c_/{print $1}' | sort -u)
[ -n "$OLD" ] && { echo "qdel old workers: $OLD"; qdel $OLD; sleep 8; }

# SEED has a comma -> must ride -V as a shell var (a -v item would split on the comma).
export BASH_ENV=/users/gmeng/.bash_profile
export SEED=1000,3000 PROFILE=bench512_v5 OUT=runs/frozen_v5 \
       PREFER=heavy CELL_TIMEOUT=43200 MEDAL_STALE_CLAIM_SEC=14400 \
       HF_HOME=/afs/crc.nd.edu/user/g/gmeng/medrax-weights/hf_cache
W=submit/seed2000_gpufix_worker.sh

export DS=btcv_synapse,flare22,mmwhs_ct,hvsmr2016,ext_abdoment1k,ext_brats2020,msd_task07_pancreas,isic2018,care_leftatrium_2026
for i in 1 2 3 4; do qsub -binding linear_per_task:1 -S /bin/bash -V -q gpu@qa-h100-003.crc.nd.edu -pe smp 8 -N v5_h100 "$W"; done
export DS=kits19,msd_task03_liver,msd_task04_hippocampus
for i in 1 2 3 4; do qsub -binding linear_per_task:1 -S /bin/bash -V -q gpu@@coba-a40 -pe smp 8 -N v5_a40 "$W"; done
export DS=refuge,glas2015,msd_task09_spleen,origa,kvasir_seg,liqa_mri,busi
for i in $(seq 8); do qsub -binding linear_per_task:1 -S /bin/bash -V -q gpu@@csecri-v100 -pe smp 4 -N v5_v100 "$W"; done

echo "launched UNCACHED 1k/3k: 4 H100 + 4 A40 + 8 V100 (SEED=1000,3000 PREFER=heavy, 12h timeout, 4h stale-claim)"
