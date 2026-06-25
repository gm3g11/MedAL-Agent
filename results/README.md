# Benchmark results — `frozen_v5`, 19 datasets

The complete results of the 19-dataset benchmark: **570 cells = 19 datasets × 10 methods (P0–P9) × 3
seeds (1000 / 2000 / 3000)**, each an active-learning trajectory over ~7 budget rounds.

## Files
- `frozen_v5/<dataset>__<P#>__s<seed>.jsonl` — one cell's full trajectory; **one JSON line per budget
  round**, recording the labeled budget and the achieved accuracy plus rich selection metadata.
- `summary_19set.csv` — a tidy flat table (3,900 rows): `dataset, policy_id, method, seed, round,
  labeled_count, labeled_ratio, mean_dsc_fg`. Open directly in pandas / Excel.

## Key fields in each JSONL line
| field | meaning |
|---|---|
| `labeled_ratio` / `labeled_count` | the AL budget at this round (curve x-axis) |
| `metrics.mean_dsc_fg` | per-case macro foreground Dice (primary y-axis metric) |
| `selected_ids` / `selected_scores` | which pool samples this method acquired this round, and their scores |
| `selection_diagnostics` | per-method internal diagnostics (e.g. cluster counts, AP val-correlation) |
| `policy_id` / `policy_name` / `seed` / `round` | the cell coordinates |

(Cluster-specific `*_path` fields point at the original run filesystem and can be ignored.)

## Reproduce the analysis (no GPU needed)
From the repo root:
```bash
python scripts/analyze_benchmark.py       # per-dataset rankings, AUBC, gain-over-Random, Friedman
python scripts/analyze_skill_ceiling.py   # oracle vs fixed-best ceiling
python scripts/budget_sensitivity.py      # per-budget method ranking (does PAAL catch up?)
```
All three read `results/frozen_v5/` directly.

## Not included
The heavy per-round prediction masks (`predictions/*.npz`) are intentionally **excluded** (large, and not
needed for the curve/ranking analysis). Regenerate them with `run_one --save-predictions` if required.
