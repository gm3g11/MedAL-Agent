# MedAL-Bench v1 — image/slice-wise pool-based active learning

A compact, non-redundant policy action space (P0–P9) for image/slice-wise AL
in medical image segmentation. Each policy corresponds to a distinct skill the
MedAL-Agent Selector may need; the bench is the falsifier.

## Status

- Approved scope: scaffold + unit tests + per-adapter 1-image smoke
  + 1-seed 10×6 smoke matrix before any 3-seed pilot.
- Backend: nnU-Net v2 (used as a library; AL-round retraining is our own loop).
- Query unit: 2D image (native 2D datasets) or 2D slice (3D-source datasets used
  with 2D nnU-Net config). Volume-wise is v2.

## Layout

```
medal_bench/
├── configs/          # YAMLs: datasets/, policies/, budgets/, runs/
├── data/             # MedALDataset interface + per-dataset adapters
├── models/           # nnU-Net builder + AL-round trainer
├── features/         # task features + foundation encoders (DINOv2 / MedSAM)
├── policies/         # P0..P9 + base.py + registry.py
├── runner/           # al_loop.py + trajectory.py (JSONL) + seeds.py
├── metrics/          # segmentation + calibration + diagnostics + AL summary
├── profiles/         # dataset_profile.py (class freq, FG ratio, redundancy)
├── analysis/         # cross-run aggregation + skill cards
└── tests/            # pytest
```

See `analysis/skill_card.py` for the falsification baselines per skill.

## Pilot datasets (v1)

ISIC 2018 Task 1, CVC-ClinicDB, BUSI, ROSE-1, PROMISE12 (slice), MSD07
Pancreas (slice). 6 datasets × 10 policies × 3 seeds = 180 runs.

## Approved deviations from the original plan

1. Shared dropout-compatible nnU-Net for ALL policies in BALD-included runs
   (P2 toggles MC-dropout mode at inference; others use eval mode).
2. Cumulative-checkpoint budget {1%, 2%, 5%, 10%, 15%, 20%}; per-round = delta.
3. Patient/volume-level splits for PROMISE12 and MSD07 (no slice-level leakage).
4. P3 main = `foreground` scope; ablation on BUSI, ISIC, MSD07.
5. P4 hard-class weighting + `include_background` + `bg_weight_cap` configs.
6. HD95 is the primary surface metric; ASD is secondary.
7. 1-image read-and-print smoke per adapter is a hard gate before any training.
8. P8/P9 log encoder_id, checkpoint, layer, pooling rule, cache_version.
9. From-scratch retraining per AL round (fine-tune is a future ablation).
