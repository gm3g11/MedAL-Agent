# frozen_v5 QC report  (19 completed cells, 11 datasets, seeds [1000, 2000, 3000])

## QC1 — TF32 flags disabled: PASS
  cudnn.allow_tf32=False  cuda.matmul.allow_tf32=False  ->  PASS

## QC2 — single GPU arch per dataset: PASS
  btcv_synapse             Hopper
  busi                     Volta
  flare22                  Hopper
  glas2015                 Volta
  hvsmr2016                Hopper
  kits19                   Ampere
  liqa_mri                 Volta
  mmwhs_ct                 Hopper
  msd_task03_liver         Ampere
  msd_task09_spleen        Volta
  refuge                   Volta

## QC3 — round-0 invariance across P0-P9: PASS
  dataset/seed                 n_pol seed/init id    arch r0 DSC spread
  glas2015/s2000                   3           OK   Volta        0.0000
  hvsmr2016/s2000                  2           OK  Hopper        0.0000
  msd_task09_spleen/s2000          2           OK   Volta        0.0000

## QC4 — all cells frozen_v5 (no v4 mixed): PASS
  Case-B cells: v5-grid (no 5% point) 15 | v4-grid (has 5% point) 0
  seeds present: [1000, 2000, 3000]

## QC5 — 3-seed stats (DSC/AUBC/HD95/ASSD/detect/regret/rank/collapse): PASS
  datasets=11 seeds=[1000, 2000, 3000]  (cells aggregated over available seeds)

  policy            meanDSC  meanAUBC  avgRank  meanRegret  collapses   n
  P2 BALD             0.877     0.860     3.00      0.0122          0   1
  P0 Random           0.755     0.714     1.33      0.0004          0   6
  P4 BADGE            0.705     0.646      nan         nan          0   2
  P9 PAAL             0.689     0.666     1.67      0.0727          0   6
  P1 Entropy            nan       nan      nan         nan          0   0
  P3 CoreSet            nan       nan      nan         nan          0   0
  P5 Ent+CoreSet        nan       nan      nan         nan          0   0
  P6 SelUnc             nan       nan      nan         nan          0   0
  P7 SAM-CoreSet        nan       nan      nan         nan          0   0
  P8 SAM-TypiClust      nan       nan      nan         nan          0   0

  catastrophic-collapse cells (final DSC<0.15): 0 / 19 completed

## QC6 — P6 canonical-baseline diagnostics: PASS
  P6 per-cell diagnostics (selection + divergence):
  dataset/seed               finalDSC  detect  sel_fg tgt_frac diverged_rounds
  (no completed P6 cells yet)
  Deep GT-based fg-size / unique-cases / adjacent-redundancy: see forensic (submit/p6_forensic.py) — P6 selects ~5-6x smaller fg, fewer cases, more adjacency.
