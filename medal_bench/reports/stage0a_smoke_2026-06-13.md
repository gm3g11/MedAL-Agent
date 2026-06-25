# Stage 0a — Loop-Smoke Report (NON-SCIENTIFIC)

Date: 2026-06-13. Env: A40 (46 GB), torch 2.4.1+cu121, conda `medal-agent`.

> **Stage 0a uses the legacy `smoke` configuration (image_size=128, pool_cap=32,
> budget plan [16,32], 15 train iters) and is NOT a formal benchmark result.**
> No method comparison or accuracy claim may be drawn from these numbers.

## Verdict: READY WITH WARNINGS for Stage −1 / Stage 0b.

The end-to-end AL loop, logging, determinism, SAM-H feature path, and the
firewall are all working. Warnings are smoke-config artifacts + one real
logging gap (GPU memory), none of which block Stage −1.

## What was run

Commands (exact):
```bash
PY=/groups/echambe2/gmeng/conda_envs/medal-agent/bin/python
OUT=/groups/echambe2/gmeng/MedAL-Agent/repo/code/runs/stage0a

# Run A — representative methods x 3 wired datasets, SAM-H for P7/P8 path
for ds in busi kvasir_seg msd07_pancreas; do
  for pol in P0 P1 P4 P8 P9; do
    $PY -m medal_bench.runner.run_one --policy $pol --dataset $ds --seed 1000 \
      --profile smoke --foundation sam --sam-model-type vit_h --out-dir $OUT/runA
  done
done

# Run B — all 10 methods on BUSI, SAM-H
$PY -m medal_bench.runner.smoke_matrix --out-dir $OUT/runB_allmethods_busi \
  --datasets busi --seed 1000 --foundation sam --sam-model-type vit_h
```

Output paths:
- Run A trajectories: `runs/stage0a/runA/{dataset}__{policy}__s1000.jsonl` (15 cells)
- Run B: `runs/stage0a/runB_allmethods_busi/` (+ `summary.txt`)
- Candidate scores: `runs/stage0a/runA/candidate_scores/*.json`
- Persisted init sets: `runs/stage0a/runA/init_sets/*.json`
- vit_h feature caches: `cache/foundation_features/*__segment_anything_vit_h_image_encoder__*.h5`
- Driver log: `runs/stage0a/stage0a_driver.log`

## Results

- **Run A: 15/15 cells OK, 0 failures.**
- **Run B: 10/10 cells OK** (`summary.txt`: "10/10 passed in 12.7s").

## Pass-criteria checklist (§5)

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | Training loop completes | ✅ | 25/25 cells exit 0 |
| 2 | Loss computes | ✅ | no NaN, DSC produced |
| 3 | Metrics compute | ✅ | `dsc_per_class`, `mean_dsc_fg`, `n_eval` logged each round |
| 4 | Query methods run | ✅ | P0,P1,P4,P8,P9 + full P0–P9 on busi |
| 5 | Selected IDs unique | ✅ | per-cell `sel_unique=True` across all rounds |
| 6 | Selected IDs from train pool only | ✅ | drawn from `train_view` only (`al_loop.py`); 102 tests |
| 7 | Train/val/test disjoint (case-disjoint for 3D) | ✅ | `runner/splits.py:82-100` patient-grouped; msd07 grouped by `pancreas_NNN` |
| 8 | Candidate-score files saved | ✅ | `candidate_scores/*.json` sidecars written |
| 9 | Selected-ID files saved | ✅ | `selected_ids` in JSONL + init_sets/ persisted |
| 10 | Checkpoint hashes saved | ✅ | `ckpt_hash` per round (sha256) |
| 11 | SAM-H cache path exercised | ✅ | P8 r0 logs `sam_model_type='vit_h'`, `feature_cache_keys={'foundation':'segment_anything/vit_h/image_encoder__v1'}`; 3 vit_h .h5 caches created |
| 12 | No SAM-B/SAM-H cache collision | ✅ | vit_h keys = `segment_anything_vit_h_image_encoder`; vit_b keys = `facebook_sam-vit-base_vision_encoder` — disjoint files |
| 13 | Deterministic replay | ✅ | re-run busi/P1: identical `selected_ids`, `ckpt_hash`, init set |
| 14 | No unlabeled-mask access by query | ✅ | `test_firewall_no_unlabeled_pool_mask_access` (102 tests pass) |
| 15 | No val/test data used for querying | ✅ | policies receive `pool`/`labeled` views only; test split never read in loop |
| 16 | Runtime + GPU memory logged | ⚠️ | runtime logged as `runtime_sec={train,eval,select}`; **GPU memory NOT logged** → see W1 |

## Warnings

- **W1 (real gap): GPU memory is not logged by the runner.** Only `runtime_sec`
  (a `{train,eval,select}` dict) is recorded. `/usr/bin/time -v` peak RSS was
  captured at the batch level in the driver log, but per-cell peak GPU memory is
  not in the trajectory. → Fold GPU-mem logging into Stage −1 (B6 logging reqs).
- **W2 (smoke artifact, not a bug): all methods converge to identical DSC on
  busi/kvasir.** With `pool_cap=32` and budget `[16,32]`, round-1 trains on the
  *entire* 32-sample pool, so every method ends on the same labeled set →
  identical final model. This also makes the round-0 selection of k=16 from a
  16-sample unlabeled pool degenerate (select-all), which is why P8's
  `selection_diagnostics` is empty here. Real (pool-dependent) budgets in
  Stage 1 avoid this entirely.
- **W3 (smoke artifact): msd07_pancreas DSC_fg=0.000 for all methods.** The
  `smoke` profile does not foreground-stratify the capped pool, so 32 random
  pancreas slices are almost all background → model collapses to all-bg. The
  `pilot` profile already fixes this (`stratify_pool_by_fg=True`,
  `profiles/__init__.py:64`). Non-issue for a loop-smoke.

## Conclusion

The loop, logging, determinism, firewall, and SAM-H path are validated on
currently-wired datasets. Proceed to Stage −1 build. Carry W1 (GPU-mem logging)
into the Stage −1 logging work.
