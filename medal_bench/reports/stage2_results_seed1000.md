# MedAL-Bench — Stage 2 Results (seed 1000)

**Status: COMPLETE — 200/200 cells (20 datasets × 10 policies), frozen_v4 protocol, seed 1000.**
Per-case macro foreground DSC (`mean_dsc_fg`) at the final round (~20% labeling budget, 6 AL rounds;
glas2015 = 4 rounds, rose1 = 1 round). No NaNs, no <0.05 hard collapses (except the one documented
P6 case). **19 usable datasets** for method comparison (rose1 excluded — see below).

## Headline finding: random is a strong baseline

At 20% budget on 2D medical segmentation, **no AL method convincingly beats random selection (P0).**
Random is 2nd by mean DSC, wins the most individual datasets (4/19), and the best method (P5
Entropy+CoreSet) edges it by only **+0.006 mean DSC**. This reproduces the central conclusion of the
deep-AL survey literature.

| rank | policy | mean DSC (19) | #dataset wins | beats random |
|---|---|---|---|---|
| 1 | **P5 Entropy+CoreSet** | **0.736** | 2 | **10/19** |
| 2 | **P0 Random** | 0.730 | **4** | — |
| 3 | P4 BADGE | 0.726 | 2 | 9/19 |
| 4 | P8 SAM-TypiClust | 0.724 | 3 | 6/19 |
| 5 | P7 SAM-CoreSet | 0.723 | 1 | 7/19 |
| 6 | P2 BALD | 0.722 | 3 | 9/19 |
| 7 | P1 Entropy | 0.707 | 2 | 8/19 |
| 8 | P3 CoreSet (U-Net feats) | 0.705 | 1 | 5/19 |
| 9 | P9 PAAL | 0.692 | 1 | 5/19 |
| 10 | P6 Selective Uncertainty | 0.650 | 0 | 5/19 |

Only **P5** beats random on more than half the datasets. **P6 (Selective Uncertainty) is the weakest
and least stable** method (see below).

## Per-dataset final DSC (P0–P9)

| dataset | P0 | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 | best |
|---|---|---|---|---|---|---|---|---|---|---|---|
| btcv_synapse | 0.62 | 0.62 | 0.63 | 0.62 | 0.61 | 0.63 | 0.55 | 0.62 | 0.58 | 0.58 | P2 |
| busi | 0.62 | 0.53 | 0.49 | 0.62 | 0.58 | 0.53 | 0.56 | 0.62 | 0.57 | 0.59 | P3 |
| care_leftatrium_2026 | 0.89 | 0.62 | 0.84 | 0.87 | 0.89 | 0.85 | **0.10** | 0.88 | 0.87 | 0.76 | P4 |
| ext_abdoment1k | 0.78 | 0.76 | 0.78 | 0.72 | 0.78 | 0.77 | 0.71 | 0.77 | 0.79 | 0.74 | P8 |
| ext_brats2020 | 0.45 | 0.50 | 0.48 | 0.37 | 0.44 | 0.48 | 0.36 | 0.44 | 0.44 | 0.42 | P1 |
| flare22 | 0.78 | 0.87 | 0.85 | 0.73 | 0.85 | 0.87 | 0.80 | 0.81 | 0.85 | 0.82 | P5 |
| glas2015 | 0.89 | 0.87 | 0.89 | 0.87 | 0.87 | 0.89 | 0.89 | 0.87 | 0.86 | 0.87 | P0 |
| hvsmr2016 | 0.56 | 0.48 | 0.46 | 0.47 | 0.57 | 0.57 | **0.29** | 0.52 | 0.56 | 0.42 | P5 |
| isic2018 | 0.86 | 0.76 | 0.85 | 0.82 | 0.85 | 0.86 | 0.82 | 0.87 | 0.87 | 0.86 | P7 |
| kits19 | 0.57 | 0.56 | 0.56 | 0.51 | 0.54 | 0.56 | 0.52 | 0.53 | 0.51 | 0.51 | P0 |
| kvasir_seg | 0.60 | 0.61 | 0.61 | 0.58 | 0.58 | 0.65 | 0.63 | 0.58 | 0.67 | 0.63 | P8 |
| liqa_mri | 0.96 | 0.95 | 0.96 | 0.96 | 0.97 | 0.96 | 0.94 | 0.96 | 0.94 | 0.94 | P4 |
| mmwhs_ct | 0.84 | 0.85 | 0.88 | 0.83 | 0.86 | 0.87 | 0.84 | 0.85 | 0.82 | 0.76 | P2 |
| msd_task03_liver | 0.54 | 0.56 | 0.59 | 0.56 | 0.57 | 0.62 | 0.56 | 0.56 | 0.62 | 0.55 | P8 |
| msd_task04_hippocampus | 0.82 | 0.83 | 0.83 | 0.82 | 0.83 | 0.83 | 0.80 | 0.81 | 0.82 | 0.73 | P2 |
| msd_task07_pancreas | 0.43 | 0.40 | 0.34 | 0.39 | 0.37 | 0.39 | 0.34 | 0.36 | 0.34 | 0.30 | P0 |
| msd_task09_spleen | 0.96 | 0.92 | 0.94 | 0.96 | 0.93 | 0.94 | 0.93 | 0.95 | 0.92 | 0.93 | P0 |
| origa | 0.85 | 0.84 | 0.83 | 0.82 | 0.85 | 0.84 | 0.84 | 0.83 | 0.84 | 0.85 | P9 |
| refuge | 0.87 | 0.89 | 0.89 | 0.87 | 0.88 | 0.89 | 0.88 | 0.88 | 0.87 | 0.88 | P1 |
| ~~rose1~~ | 0.74 | 0.74 | 0.74 | 0.74 | 0.74 | 0.74 | 0.74 | 0.74 | 0.74 | 0.74 | **EXCLUDED** |

Policies: P0=Random, P1=Entropy, P2=BALD, P3=CoreSet (U-Net feats), P4=BADGE, P5=Entropy+CoreSet,
P6=Selective Uncertainty, P7=SAM-CoreSet, P8=SAM-TypiClust, P9=PAAL.

## Notable findings & caveats

- **rose1 EXCLUDED (degenerate).** Only 20 training images → the minimum seed set (8) is already 40%
  of the pool, so the protocol runs **1 round at 40% budget with no AL selection** → all methods tie
  at ~0.74. Not a bug, not re-runnable; too small for the 20%-budget AL protocol.
- **P6 (Selective Uncertainty) is unstable** — late-round collapses: care_leftatrium P6 = **0.10**
  (verified deterministic — a rerun reproduced 0.100 exactly), hvsmr2016 P6 = **0.29** (climbs to 0.39
  by round 4, drops to 0.29 at round 5). P6 works fine elsewhere (glas 0.89, mmwhs 0.84) but is the
  weakest, least reliable policy.
- **Low absolute DSC on hard datasets is intrinsic**, not error: pancreas (~0.30–0.43, sparse FG),
  kits19/brats (~0.45–0.57, small/diffuse targets). Consistent across all methods.
- **Single seed (1000) only** — no statistical significance / error bars yet. The +0.006 P5-vs-random
  gap is within plausible seed noise; multi-seed (2000/3000) needed before any ranking claim is firm.
- GPU heterogeneity (A40/V100/H100) introduces ~0.005 DSC FP noise (e.g. rose1's 0.739 vs 0.743).

## Infrastructure notes

- **Multi-class BALD RAM wall solved on H100.** P2/uncertainty methods on high-class-count datasets
  (btcv/flare22 C=14, hvsmr C=9, mmwhs C=8) need ~84–150 GB host RAM for the byte-identical BALD
  running sum — OOM on A40/V100. The H100 (qa-h100-003, ~444 GB RAM) completed all of them.
- **Chunked acquisition fix** (P1/P5/P6) bounded RAM to O(batch·C·H·W), equivalence-tested
  (byte-identical selection) — see `project-multiclass-acquisition-oom-fix`.
- V100 smp4 packing doubled V100 utilization (4→8 GPUs: 16 cores/node, 4 GPUs/node).
- mmwhs P2 required a re-submit (a worker enforced a ~32 min cell-timeout → BALD C=8 hung; re-run with
  the default 4h timeout on H100 completed it at DSC 0.88).

---

## Per-budget breakdown (DSC at ~5% / ~10% / ~20% labeling budget)

Budget = fraction of train pool labeled; columns pick each dataset's round nearest 5/10/20%.
rose1 (1 round @40%) and glas2015 (4 rounds, max ~19.5%) shown at their available budgets.


**btcv_synapse**

| policy | @5% | @10% | @20% |
|---|---|---|---|
| P0 | 0.43 | 0.52 | 0.62 |
| P1 | 0.46 | 0.54 | 0.62 |
| P2 | 0.45 | 0.56 | 0.63 |
| P3 | 0.40 | 0.51 | 0.62 |
| P4 | 0.42 | 0.52 | 0.61 |
| P5 | 0.48 | 0.57 | 0.63 |
| P6 | 0.41 | 0.43 | 0.55 |
| P7 | 0.46 | 0.57 | 0.62 |
| P8 | 0.42 | 0.53 | 0.58 |
| P9 | 0.40 | 0.51 | 0.58 |

**busi**

| policy | @5% | @10% | @20% |
|---|---|---|---|
| P0 | 0.46 | 0.51 | 0.62 |
| P1 | 0.50 | 0.49 | 0.53 |
| P2 | 0.38 | 0.54 | 0.49 |
| P3 | 0.38 | 0.53 | 0.62 |
| P4 | 0.39 | 0.52 | 0.58 |
| P5 | 0.40 | 0.49 | 0.53 |
| P6 | 0.43 | 0.47 | 0.56 |
| P7 | 0.45 | 0.50 | 0.62 |
| P8 | 0.49 | 0.60 | 0.57 |
| P9 | 0.47 | 0.47 | 0.59 |

**care_leftatrium_2026**

| policy | @5% | @10% | @20% |
|---|---|---|---|
| P0 | 0.78 | 0.86 | 0.89 |
| P1 | 0.68 | 0.64 | 0.62 |
| P2 | 0.45 | 0.61 | 0.84 |
| P3 | 0.63 | 0.77 | 0.87 |
| P4 | 0.84 | 0.87 | 0.89 |
| P5 | 0.78 | 0.83 | 0.85 |
| P6 | 0.70 | 0.73 | 0.10 |
| P7 | 0.82 | 0.87 | 0.88 |
| P8 | 0.84 | 0.87 | 0.87 |
| P9 | 0.75 | 0.78 | 0.76 |

**ext_abdoment1k**

| policy | @5% | @10% | @20% |
|---|---|---|---|
| P0 | 0.69 | 0.74 | 0.78 |
| P1 | 0.60 | 0.72 | 0.76 |
| P2 | 0.66 | 0.65 | 0.78 |
| P3 | 0.49 | 0.64 | 0.72 |
| P4 | 0.65 | 0.70 | 0.78 |
| P5 | 0.60 | 0.72 | 0.77 |
| P6 | 0.64 | 0.71 | 0.71 |
| P7 | 0.61 | 0.67 | 0.77 |
| P8 | 0.71 | 0.76 | 0.79 |
| P9 | 0.43 | 0.65 | 0.74 |

**ext_brats2020**

| policy | @5% | @10% | @20% |
|---|---|---|---|
| P0 | 0.42 | 0.42 | 0.45 |
| P1 | 0.42 | 0.47 | 0.50 |
| P2 | 0.40 | 0.46 | 0.48 |
| P3 | 0.35 | 0.40 | 0.37 |
| P4 | 0.40 | 0.44 | 0.44 |
| P5 | 0.40 | 0.45 | 0.48 |
| P6 | 0.40 | 0.42 | 0.36 |
| P7 | 0.42 | 0.45 | 0.44 |
| P8 | 0.38 | 0.45 | 0.44 |
| P9 | 0.39 | 0.42 | 0.42 |

**flare22**

| policy | @5% | @10% | @20% |
|---|---|---|---|
| P0 | 0.71 | 0.75 | 0.78 |
| P1 | 0.81 | 0.85 | 0.87 |
| P2 | 0.77 | 0.85 | 0.85 |
| P3 | 0.71 | 0.69 | 0.73 |
| P4 | 0.76 | 0.79 | 0.85 |
| P5 | 0.82 | 0.86 | 0.87 |
| P6 | 0.66 | 0.80 | 0.80 |
| P7 | 0.68 | 0.78 | 0.81 |
| P8 | 0.75 | 0.85 | 0.85 |
| P9 | 0.78 | 0.82 | 0.82 |

**glas2015**

| policy | @5% | @10% | @20% |
|---|---|---|---|
| P0 | 0.85 | 0.85 | 0.89 |
| P1 | 0.85 | 0.86 | 0.87 |
| P2 | 0.85 | 0.85 | 0.89 |
| P3 | 0.85 | 0.84 | 0.87 |
| P4 | 0.85 | 0.85 | 0.87 |
| P5 | 0.85 | 0.86 | 0.89 |
| P6 | 0.85 | 0.85 | 0.89 |
| P7 | 0.85 | 0.86 | 0.87 |
| P8 | 0.85 | 0.84 | 0.86 |
| P9 | 0.85 | 0.83 | 0.87 |

**hvsmr2016**

| policy | @5% | @10% | @20% |
|---|---|---|---|
| P0 | 0.36 | 0.48 | 0.56 |
| P1 | 0.18 | 0.26 | 0.48 |
| P2 | 0.17 | 0.30 | 0.46 |
| P3 | 0.27 | 0.37 | 0.47 |
| P4 | 0.43 | 0.45 | 0.57 |
| P5 | 0.31 | 0.47 | 0.57 |
| P6 | 0.22 | 0.31 | 0.29 |
| P7 | 0.32 | 0.39 | 0.52 |
| P8 | 0.43 | 0.49 | 0.56 |
| P9 | 0.23 | 0.40 | 0.42 |

**isic2018**

| policy | @5% | @10% | @20% |
|---|---|---|---|
| P0 | 0.83 | 0.85 | 0.86 |
| P1 | 0.79 | 0.78 | 0.76 |
| P2 | 0.80 | 0.82 | 0.85 |
| P3 | 0.83 | 0.85 | 0.82 |
| P4 | 0.86 | 0.86 | 0.85 |
| P5 | 0.82 | 0.85 | 0.86 |
| P6 | 0.82 | 0.83 | 0.82 |
| P7 | 0.82 | 0.85 | 0.87 |
| P8 | 0.80 | 0.85 | 0.87 |
| P9 | 0.83 | 0.86 | 0.86 |

**kits19**

| policy | @5% | @10% | @20% |
|---|---|---|---|
| P0 | 0.52 | 0.53 | 0.57 |
| P1 | 0.43 | 0.44 | 0.56 |
| P2 | 0.43 | 0.51 | 0.56 |
| P3 | 0.48 | 0.49 | 0.51 |
| P4 | 0.48 | 0.51 | 0.54 |
| P5 | 0.45 | 0.57 | 0.56 |
| P6 | 0.41 | 0.49 | 0.52 |
| P7 | 0.46 | 0.50 | 0.53 |
| P8 | 0.48 | 0.50 | 0.51 |
| P9 | 0.41 | 0.44 | 0.51 |

**kvasir_seg**

| policy | @5% | @10% | @20% |
|---|---|---|---|
| P0 | 0.48 | 0.51 | 0.60 |
| P1 | 0.46 | 0.55 | 0.61 |
| P2 | 0.43 | 0.53 | 0.61 |
| P3 | 0.40 | 0.49 | 0.58 |
| P4 | 0.33 | 0.49 | 0.58 |
| P5 | 0.31 | 0.44 | 0.65 |
| P6 | 0.42 | 0.51 | 0.63 |
| P7 | 0.47 | 0.53 | 0.58 |
| P8 | 0.50 | 0.57 | 0.67 |
| P9 | 0.44 | 0.48 | 0.63 |

**liqa_mri**

| policy | @5% | @10% | @20% |
|---|---|---|---|
| P0 | 0.90 | 0.92 | 0.96 |
| P1 | 0.91 | 0.92 | 0.95 |
| P2 | 0.87 | 0.91 | 0.96 |
| P3 | 0.94 | 0.93 | 0.96 |
| P4 | 0.90 | 0.95 | 0.97 |
| P5 | 0.93 | 0.95 | 0.96 |
| P6 | 0.92 | 0.92 | 0.94 |
| P7 | 0.93 | 0.95 | 0.96 |
| P8 | 0.91 | 0.93 | 0.94 |
| P9 | 0.91 | 0.95 | 0.94 |

**mmwhs_ct**

| policy | @5% | @10% | @20% |
|---|---|---|---|
| P0 | 0.75 | 0.79 | 0.84 |
| P1 | 0.73 | 0.79 | 0.85 |
| P2 | 0.76 | 0.83 | 0.88 |
| P3 | 0.65 | 0.76 | 0.83 |
| P4 | 0.79 | 0.85 | 0.86 |
| P5 | 0.82 | 0.82 | 0.87 |
| P6 | 0.76 | 0.77 | 0.84 |
| P7 | 0.75 | 0.78 | 0.85 |
| P8 | 0.79 | 0.82 | 0.82 |
| P9 | 0.56 | 0.74 | 0.76 |

**msd_task03_liver**

| policy | @5% | @10% | @20% |
|---|---|---|---|
| P0 | 0.52 | 0.56 | 0.54 |
| P1 | 0.48 | 0.53 | 0.56 |
| P2 | 0.42 | 0.54 | 0.59 |
| P3 | 0.50 | 0.56 | 0.56 |
| P4 | 0.51 | 0.56 | 0.57 |
| P5 | 0.54 | 0.61 | 0.62 |
| P6 | 0.45 | 0.48 | 0.56 |
| P7 | 0.52 | 0.53 | 0.56 |
| P8 | 0.54 | 0.58 | 0.62 |
| P9 | 0.46 | 0.49 | 0.55 |

**msd_task04_hippocampus**

| policy | @5% | @10% | @20% |
|---|---|---|---|
| P0 | 0.76 | 0.80 | 0.82 |
| P1 | 0.76 | 0.81 | 0.83 |
| P2 | 0.77 | 0.82 | 0.83 |
| P3 | 0.76 | 0.79 | 0.82 |
| P4 | 0.76 | 0.81 | 0.83 |
| P5 | 0.78 | 0.82 | 0.83 |
| P6 | 0.73 | 0.78 | 0.80 |
| P7 | 0.73 | 0.78 | 0.81 |
| P8 | 0.76 | 0.79 | 0.82 |
| P9 | 0.70 | 0.76 | 0.73 |

**msd_task07_pancreas**

| policy | @5% | @10% | @20% |
|---|---|---|---|
| P0 | 0.27 | 0.33 | 0.43 |
| P1 | 0.27 | 0.33 | 0.40 |
| P2 | 0.25 | 0.35 | 0.34 |
| P3 | 0.23 | 0.31 | 0.39 |
| P4 | 0.26 | 0.32 | 0.37 |
| P5 | 0.27 | 0.31 | 0.39 |
| P6 | 0.26 | 0.31 | 0.34 |
| P7 | 0.30 | 0.30 | 0.36 |
| P8 | 0.32 | 0.31 | 0.34 |
| P9 | 0.22 | 0.31 | 0.30 |

**msd_task09_spleen**

| policy | @5% | @10% | @20% |
|---|---|---|---|
| P0 | 0.89 | 0.94 | 0.96 |
| P1 | 0.84 | 0.93 | 0.92 |
| P2 | 0.89 | 0.93 | 0.94 |
| P3 | 0.90 | 0.95 | 0.96 |
| P4 | 0.82 | 0.89 | 0.93 |
| P5 | 0.91 | 0.94 | 0.94 |
| P6 | 0.87 | 0.93 | 0.93 |
| P7 | 0.88 | 0.94 | 0.95 |
| P8 | 0.91 | 0.95 | 0.92 |
| P9 | 0.86 | 0.91 | 0.93 |

**origa**

| policy | @5% | @10% | @20% |
|---|---|---|---|
| P0 | 0.82 | 0.83 | 0.85 |
| P1 | 0.78 | 0.81 | 0.84 |
| P2 | 0.81 | 0.82 | 0.83 |
| P3 | 0.78 | 0.81 | 0.82 |
| P4 | 0.82 | 0.82 | 0.85 |
| P5 | 0.79 | 0.81 | 0.84 |
| P6 | 0.79 | 0.83 | 0.84 |
| P7 | 0.80 | 0.82 | 0.83 |
| P8 | 0.79 | 0.80 | 0.84 |
| P9 | 0.84 | 0.82 | 0.85 |

**refuge**

| policy | @5% | @10% | @20% |
|---|---|---|---|
| P0 | 0.84 | 0.85 | 0.87 |
| P1 | 0.86 | 0.88 | 0.89 |
| P2 | 0.86 | 0.87 | 0.89 |
| P3 | 0.83 | 0.86 | 0.87 |
| P4 | 0.85 | 0.87 | 0.88 |
| P5 | 0.85 | 0.86 | 0.89 |
| P6 | 0.86 | 0.87 | 0.88 |
| P7 | 0.84 | 0.87 | 0.88 |
| P8 | 0.84 | 0.86 | 0.87 |
| P9 | 0.83 | 0.85 | 0.88 |

**rose1**

| policy | @5% | @10% | @20% |
|---|---|---|---|
| P0 | 0.74 | 0.74 | 0.74 |
| P1 | 0.74 | 0.74 | 0.74 |
| P2 | 0.74 | 0.74 | 0.74 |
| P3 | 0.74 | 0.74 | 0.74 |
| P4 | 0.74 | 0.74 | 0.74 |
| P5 | 0.74 | 0.74 | 0.74 |
| P6 | 0.74 | 0.74 | 0.74 |
| P7 | 0.74 | 0.74 | 0.74 |
| P8 | 0.74 | 0.74 | 0.74 |
| P9 | 0.74 | 0.74 | 0.74 |

---

## How AL-method ranking shifts with budget (mean DSC over 19 usable datasets)

| policy | mean@5% | mean@10% | mean@20% |
|---|---|---|---|
| P0 | 0.646 | 0.687 | 0.730 |
| P1 | 0.622 | 0.663 | 0.707 |
| P2 | 0.602 | 0.670 | 0.722 |
| P3 | 0.600 | 0.661 | 0.705 |
| P4 | 0.638 | 0.689 | 0.726 |
| P5 | 0.637 | 0.697 | 0.736 |
| P6 | 0.610 | 0.655 | 0.650 |
| P7 | 0.636 | 0.681 | 0.723 |
| P8 | 0.658 | 0.702 | 0.724 |
| P9 | 0.596 | 0.657 | 0.692 |

**Top-3 by budget:**
- @5%: P8(0.658), P0(0.646), P4(0.638)
- @10%: P8(0.702), P5(0.697), P4(0.689)
- @20%: P5(0.736), P0(0.730), P4(0.726)
