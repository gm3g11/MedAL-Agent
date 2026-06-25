# Stage 0b — Interim Report: MMWHS-CT multiclass/remap validation (NON-SCIENTIFIC)

Date 2026-06-13. Env: A40, torch 2.4.1+cu121.

> **Interim:** uses the legacy `smoke` config (image_size=128, pool_cap=32, budget
> [16,32], 15 iters) — NOT the formal adaptive-512 Stage 0b (which waits on B2).
> Purpose: prove the **8-class remap path runs end-to-end** through training +
> DiceCE loss + 8-class DSC/HD95 metrics + query, with no class-index error.
> No accuracy claim.

## Verdict: PASS (interim). READY for the formal adaptive-512 Stage 0b once B2 lands.

## What was run

```bash
# runA: representative methods (SAM-H), legacy smoke
for pol in P0 P1 P4 P8 P9; do
  python -m medal_bench.runner.run_one --policy $pol --dataset mmwhs_ct --seed 1000 \
    --profile smoke --foundation sam --sam-model-type vit_h --out-dir runs/stage0b/runA
done
# runB: all 10 methods
python -m medal_bench.runner.smoke_matrix --out-dir runs/stage0b/runB_allmethods \
  --datasets mmwhs_ct --seed 1000 --foundation sam --sam-model-type vit_h
```

Outputs: `runs/stage0b/runA/*.jsonl`, `runs/stage0b/runB_allmethods/summary.txt`,
overlay `runs/stage0b/overlays/mmwhs_ct_overlay.png`, log `runs/stage0b/stage0b_driver.log`.

## Results

- **runA: 5/5 OK** (P0,P1,P4,P8,P9).
- **runB: 10/10 OK** (P0–P9), 725 s total (~72 s/cell).
- No `.fail.txt` markers.
- mDSC_fg ≈ 0.01–0.05 across methods — expected smoke artifact (random mostly-bg
  CT slices, 15 iters, img 128); NOT a method comparison.

## Pass-criteria checklist (§7)

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | MMWHS adapter loads | ✅ | all cells loaded `mmwhs_ct` |
| 2 | Remap applied in training loader | ✅ | masks dense {0..7} via `LabelRemapper` in `__getitem__` |
| 3 | Labels after remap dense 0..C-1 | ✅ | `dsc_per_class` length 8 |
| 4 | Loss accepts multiclass masks | ✅ | DiceCE trained on C=8, no class-index error |
| 5 | Metrics accept multiclass masks | ✅ | 8-class DSC computed; 7 fg classes scored |
| 6 | P0/P1/P4/P8/P9 (and all P0–P9) run | ✅ | 15/15 cells OK |
| 7 | Candidate scores saved | ✅ | `candidate_scores/*.json` |
| 8 | Selected IDs saved | ✅ | `selected_ids`; unique; format `Case1012_117` (case+slice) |
| 9 | Checkpoint hashes saved | ✅ | `ckpt_hash` per round |
| 10 | No unlabeled-mask access | ✅ | firewall (133 tests) |
| 11 | No split leakage | ✅ | case-disjoint split verified (48/6/6 cases) |
| 12 | SAM-H cache works under run | ✅ | P7/P8 ran with vit_h; cache reused |
| 13 | Runtime acceptable | ⚠️ | ~72 s/cell, dominated by per-cell volume RELOAD (NFS) → the B6 throughput issue; fine for smoke, must fix before Stage 1 scale |

## Notes / carry-forward

- The ~72 s/cell is almost entirely MMWHS volume re-loading from NFS (each cell
  rebuilds `_IndexedSubset` from scratch). This is exactly the **B6** bottleneck and
  the dominant cost at Stage 1 scale — fix before the 1,600+-cell matrix.
- Sample-ID format confirmed: `{Case}_{slice:03d}` (e.g. `Case1012_117`) — encodes
  case_id + slice_index per the 3D-as-slice requirement; dataset name is the
  trajectory `dataset` field.
- Visual overlay confirms remap + axial orientation are anatomically correct.
- **Formal Stage 0b** (adaptive bench_res=512) is still pending **B2**; rerun there
  before declaring Stage 0b fully closed for Stage 1.
