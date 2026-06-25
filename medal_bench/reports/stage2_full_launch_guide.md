# Stage-2 FULL launch guide (accelerated, frozen_v4) — READY, awaiting your go

Covers the **14 remaining core datasets** (the confident-6 — busi, kvasir_seg, isic2018,
msd_task09_spleen, mmwhs_ct, ext_brats2020 — are already done in `runs/stage2_wave2`). Output dir:
`runs/stage2_full`. Seed **1000 only** (seeds 2000/3000 held until seed-1000 core is reviewed).

**Accuracy:** the three accelerations are accuracy-neutral and test-guarded — `--defer-surface` backfills
HD95/ASSD offline with *identical numbers* (`test_surface_offline.py`), pre-warm reproduces the *exact*
cache keys cells compute (proven), resume (when added) is determinism-tested. DSC/detection/training are
untouched.

## Datasets
- **Normal (max_iters 3000, profile default):** refuge, glas2015, origa, rose1, msd_task04_hippocampus,
  msd_task03_liver, hvsmr2016, care_leftatrium_2026, flare22, kits19, ext_abdoment1k, liqa_mri.
  (refuge + origa = the new C=3 disc+cup adapters; glas2015/rose1/liqa_mri = `in_core_avg=caution`,
  reported separately with wide CIs.)
- **Bumped (max_iters 5000):** btcv_synapse, msd_task07_pancreas — they rose in DSC at the 3000 cap
  (under-trained); your 4000–5000 decision. (flare22/hvsmr2016 are also multi-class and *may* cap at
  3000 like mmwhs_ct did — mmwhs gave good DSC there, so left at 3000; revisit if their DSC looks
  under-trained.)

## Launch sequence (3 steps)
**STEP 1 — pre-warm caches (no GPU, do FIRST):**
```
for D in refuge glas2015 origa rose1 msd_task04_hippocampus msd_task03_liver hvsmr2016 \
         care_leftatrium_2026 flare22 kits19 ext_abdoment1k liqa_mri btcv_synapse msd_task07_pancreas; do
  DS=$D qsub -V -pe smp 8 submit/stage2_full_prewarm.sh
done
# wait until all prewarm jobs finish (cache files present) before STEP 2.
```

**STEP 2 — launch cells (A40 + V100, --defer-surface --save-predictions):**
```
# NORMAL-12 (max_iters 3000)
DSN=refuge,glas2015,origa,rose1,msd_task04_hippocampus,msd_task03_liver,hvsmr2016,care_leftatrium_2026,flare22,kits19,ext_abdoment1k,liqa_mri
for i in $(seq 1 8); do DS=$DSN METHODS=P0,P1,P2,P3,P4,P5,P6,P7,P8,P9 PREFER=heavy qsub -V -q gpu@@coba-a40    -pe smp 8 -N s2f_a submit/stage2_full_worker.sh; done
for i in $(seq 1 8); do DS=$DSN METHODS=P0,P1,P2,P3,P4,P5,P6          PREFER=light qsub -V -q gpu@@csecri-v100 -pe smp 4 -N s2f_v submit/stage2_full_worker.sh; done
# BUMPED-2 (max_iters 5000) — A40 only (P7/P8/P9 need >=22-24GB)
DSB=btcv_synapse,msd_task07_pancreas
for i in 1 2 3 4; do DS=$DSB MAXITERS=5000 METHODS=P0,P1,P2,P3,P4,P5,P6,P7,P8,P9 PREFER=heavy qsub -V -q gpu@@coba-a40 -pe smp 8 -N s2f_b submit/stage2_full_worker.sh; done
```

**STEP 3 — offline surface backfill (after ALL cells done, no GPU):**
```
python -m medal_bench.runner.surface_offline --run-dir runs/stage2_full
# patches each cell's final-round record with HD95/ASSD (identical to inline).
```

## Notes / guards
- **MUST run STEP 1 fully before STEP 2** (else the rebuild storm recurs).
- All `qsub` use **`-V`** (else the `DS` env doesn't propagate — the worker errors with `set DS`).
- `--defer-surface` keeps DSC + detection inline (cheap); only HD95/ASSD move offline.
- After STEP 3, the `runs/stage2_full` records are complete and identical to an all-inline run.
- Confident-6 results live in `runs/stage2_wave2`; combine both dirs for the 20-core headline.
- **Resume (#3):** once landed, add `--resume` to the worker so a killed cell continues from its last
  completed round instead of restarting.
