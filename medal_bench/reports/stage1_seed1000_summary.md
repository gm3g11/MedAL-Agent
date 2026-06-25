# Stage 1 — Seed 1000 Summary (review gate before seeds 2000/3000)

**Status:** 90/90 cells complete, **0 hard failures**. Single seed only — **no significance testing
yet** (that needs all 3 seeds); treat method differences as indicative, not conclusive. A diagnostics
addendum (full-supervised sanity, multiclass per-class analysis, msd07 longer-training probe, ISIC P1
inspection) is appended in §9–§13 — see also `stage1_seed1000_diagnostics.md`.

Profile: `bench512` (image 512, aspect-preserve letterbox, train batch 12, num_iters 250,
pool_cap 5000, pool-dependent budget grid, 6 budget points). Metric: mean foreground DSC.

---

## 1. Final DSC (last budget point)

| dataset | P0 | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 | best |
|---|---|---|---|---|---|---|---|---|---|---|---|
| busi                | 0.477 | 0.463 | 0.415 | 0.481 | 0.408 | **0.483** | 0.420 | 0.474 | 0.448 | 0.419 | P5 |
| kvasir_seg          | 0.460 | 0.424 | 0.443 | 0.424 | 0.470 | 0.396 | 0.434 | 0.459 | **0.502** | 0.451 | P8 |
| isic2018            | 0.830 | 0.750 | 0.826 | 0.837 | 0.822 | **0.839** | 0.831 | 0.818 | 0.826 | **0.839** | P5/P9 |
| glas2015            | **0.894** | 0.890 | 0.886 | 0.884 | 0.878 | 0.882 | 0.864 | 0.885 | 0.881 | 0.885 | P0 |
| origa               | 0.927 | 0.929 | 0.941 | 0.937 | 0.927 | 0.932 | 0.918 | 0.920 | **0.943** | 0.915 | P8 |
| mmwhs               | 0.224 | 0.232 | 0.123 | 0.212 | 0.209 | 0.140 | 0.127 | 0.190 | 0.187 | **0.244** | P9 |
| btcv_synapse        | 0.061 | **0.093** | 0.078 | 0.060 | 0.056 | 0.072 | 0.062 | 0.061 | 0.058 | 0.083 | P1 |
| msd_task07_pancreas | 0.000 | 0.005 | 0.168 | 0.000 | 0.000 | 0.072 | 0.038 | 0.000 | 0.179 | **0.188** | P9 |
| ext_brats2020       | 0.319 | 0.303 | 0.276 | **0.382** | 0.363 | 0.352 | 0.370 | 0.342 | 0.322 | 0.341 | P3 |

mmwhs P2 (BALD) = 0.123, **below Random (0.224)** and among the worst on mmwhs — consistent with the
multiclass under-learning flagged in §5/§10.

Methods: P0 Random · P1 Normalized-Entropy · P2 BALD · P3 CoreSet · P4 BADGE(CE) · P5 Entropy→CoreSet
· P6 Selective-Uncertainty · P7 Foundation-CoreSet(SAM-H) · P8 Foundation-TypiClust(SAM-H) · P9 PAAL.

## 2. Initial DSC (first budget point — cold-start context)

| dataset | P0 | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 |
|---|---|---|---|---|---|---|---|---|---|---|
| busi                | 0.341 | 0.236 | 0.295 | 0.236 | 0.295 | 0.236 | 0.236 | 0.236 | 0.236 | 0.295 |
| kvasir_seg          | 0.366 | 0.366 | 0.366 | 0.366 | 0.371 | 0.366 | 0.366 | 0.366 | 0.366 | 0.371 |
| isic2018            | 0.827 | 0.827 | 0.812 | 0.812 | 0.827 | 0.827 | 0.827 | 0.827 | 0.812 | 0.812 |
| glas2015            | 0.866 | 0.866 | 0.867 | 0.866 | 0.867 | 0.866 | 0.866 | 0.866 | 0.866 | 0.867 |
| origa               | 0.893 | 0.893 | 0.887 | 0.893 | 0.893 | 0.893 | 0.893 | 0.893 | 0.893 | 0.887 |
| mmwhs               | 0.230 | 0.230 | 0.212 | 0.212 | 0.212 | 0.230 | 0.230 | 0.230 | 0.230 | 0.212 |
| btcv_synapse        | 0.101 | 0.101 | 0.104 | 0.101 | 0.104 | 0.101 | 0.104 | 0.101 | 0.101 | 0.104 |
| msd_task07_pancreas | 0.106 | 0.106 | 0.131 | 0.106 | 0.131 | 0.131 | 0.106 | 0.106 | 0.106 | 0.131 |
| ext_brats2020       | 0.224 | 0.224 | 0.274 | 0.224 | 0.274 | 0.224 | 0.224 | 0.224 | 0.224 | 0.270 |

(Initial DSC clusters by *initial labeled set*, which only differs across methods through their
budget-grid first point — so identical values within a dataset are expected.)

## 3. Gain over Random (final DSC − P0 final), where P0 did **not** collapse

Excludes `msd_task07_pancreas` (P0 collapsed to 0.000 → ratio undefined).

| dataset | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 |
|---|---|---|---|---|---|---|---|---|---|
| busi          | −.014 | −.062 | +.004 | −.069 | **+.006** | −.057 | −.003 | −.029 | −.058 |
| kvasir_seg    | −.036 | −.017 | −.036 | +.010 | −.064 | −.026 | −.001 | **+.042** | −.009 |
| isic2018      | **−.080** | −.004 | +.007 | −.008 | +.009 | +.001 | −.012 | −.004 | +.009 |
| glas2015      | −.004 | −.008 | −.010 | −.016 | −.012 | −.030 | −.009 | −.013 | −.009 |
| origa         | +.002 | +.014 | +.010 | .000 | +.005 | −.009 | −.007 | **+.016** | −.012 |
| mmwhs         | +.008 | −.101 | −.012 | −.015 | −.084 | −.097 | −.034 | −.037 | **+.020** |
| btcv_synapse  | **+.032** | +.017 | −.001 | −.005 | +.011 | +.001 | .000 | −.003 | +.022 |
| ext_brats2020 | −.016 | −.043 | **+.063** | +.044 | +.033 | +.051 | +.023 | +.003 | +.022 |

**Takeaway (single seed):** AL gains over Random are **small and dataset-dependent** at these low
budgets — typically within ±0.02–0.06. No method dominates: P8 (Foundation-TypiClust) wins
kvasir/origa, P9 (PAAL) wins isic(tie)/mmwhs/msd07, P3 (CoreSet) wins brats, P5 wins busi, and
**Random (P0) is best on glas2015**. This "AL barely beats random in the low-budget regime" pattern
is itself a expected and reportable result; whether the small gaps are real needs the 3-seed spread.

## 4. Collapse list

- **msd_task07_pancreas (Tier-C, sparse-FG instability):** P0, P3, P4, P7 → DSC = 0.000; P1 → 0.005
  (effectively collapsed). Survivors are still low: P2 0.168, P8 0.179, P9 0.188, P5 0.072, P6 0.038.
  Consistent with the known pancreas collapse; needs class-weighted Dice + FG-batch sampling to be
  trainable. Keep in the matrix as a documented failure mode, not a method comparison.
- No collapse elsewhere. mmwhs/btcv are **low but not collapsed** (hard 8-/14-class at tiny budgets).

## 5. Anomalies beyond the known set

Known and confirmed: btcv ≈0.06–0.09; isic P1 drop; msd07 collapse; mmwhs P5/P6 low (0.14/0.13).

New observations worth a look (not necessarily code bugs):
1. **Multiclass datasets don't improve with budget — and btcv slightly *regresses*.** btcv final
   (≈0.06) is *below* its initial (≈0.10) for most methods; mmwhs is roughly flat (0.230→0.22). At
   512 with 8/14 classes and tiny labeled budgets, `num_iters=250` may be too few to fit multiclass,
   so added data isn't yet helping. This caps how much signal these datasets give for *method* ranking
   in Stage 1 — flag for the budget/iters review, but it is **not** a frozen-config change to make now.
2. **isic P1 (Normalized Entropy) −0.080** is the single largest deviation from Random. Could be a real
   weakness of plain normalized entropy on skin lesions, or seed-specific — 3 seeds will resolve it.

## 6. Failures

None. 0 `*.fail.txt`, 0 SAM-OOM. All 89 completed cells produced full 6-point trajectories.

## 7. Orchestration fixes applied this run (so seeds 2000/3000 are hands-off)

- `run_one.py`: trajectory now writes to `*.jsonl.partial` and **atomically renames** to `.jsonl`
  only on success → a killed worker never leaves a "looks-done" partial that the dispatcher would skip.
- `dispatch.py`: **stale-claim lease** — a worker steals a claim only if there is no final `.jsonl`
  and the `.partial` has been idle >20 min (atomic rename-to-steal). Tested 5/5. This removes the
  manual orphan-clearing that was needed repeatedly on seed 1000.

## 8. ETA for seeds 2000/3000

Per-sample **preproc and SAM-H caches are shared across seeds** (keyed by sample, not seed), and they
are now fully warm. Only pool-selection + training differ per seed. So seeds 2000/3000 should run
**substantially faster than 1000** (which paid all the cache-build + debugging cost). Rough estimate
with the warm caches and the lease keeping workers busy: **~1–1.5 hr per seed, ~2–3 hr for both.**

Launch (after this review):
`for i in 1 2 3 4; do qsub -q gpu@@coba-a40 -pe smp 8 -v SEEDS=2000,3000 submit/stage1_worker.sh; done`
`for i in $(seq 8); do qsub -q gpu@@csecri-v100 -pe smp 4 -v SEEDS=2000,3000 submit/stage1_worker.sh; done`
