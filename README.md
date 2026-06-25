# MedAL-Bench — Active Learning for 2D Medical Image Segmentation

A reproducible benchmark of **10 active-learning (AL) acquisition methods** for 2D medical image
segmentation. Every method is implemented behind one interface, run through one fixed training/evaluation
pipeline, and scored by one metric (AUBC over the budget→Dice curve), so the *only* thing that differs
between runs is the acquisition function.

- **10 methods (P0–P9):** Random, Entropy, BALD, CoreSet, BADGE, Entropy+CoreSet, Selective Uncertainty,
  SAM-CoreSet, TypiClust, PAAL — see [`METHODS.md`](METHODS.md) for the algorithm, paper, and code mapping
  of each; citations in [`references.bib`](references.bib).
- **Protocol:** from-scratch U-Net retraining each AL round, low-budget regime (~1–20% of the pool),
  patient/case-disjoint splits, per-case foreground Dice, 3 seeds.

> **Honest headline result (3 seeds, 19 datasets):** no method *significantly* beats Random at this
> low-label budget; the robust signal is a **floor** — PAAL/SelUnc/Entropy/CoreSet are reliably *worse*
> than Random, while BADGE/Ent+Core/TypiClust sit in a top cluster that is statistically *tied* with
> Random. This is consistent with the cold-start AL literature (Hacohen 2022, Gaillochet 2023). See
> [Results](#results).

---

## The 10 methods

| id | Method | Family | Paper |
|----|--------|--------|-------|
| **P0** | Random | baseline | Settles 2009 (survey) |
| **P1** | Normalized Entropy | uncertainty | Shannon 1948; Settles 2009 |
| **P2** | BALD (MC-dropout) | uncertainty | Houlsby 2011; Gal 2017 |
| **P3** | CoreSet (greedy k-center) | diversity | Sener & Savarese, ICLR 2018 |
| **P4** | BADGE | hybrid (uncertainty+diversity) | Ash et al., ICLR 2020 |
| **P5** | Entropy → CoreSet | hybrid | Sener & Savarese 2018 (two-stage) |
| **P6** | Selective Uncertainty | uncertainty | Ma et al., ICASSP 2024 |
| **P7** | SAM-CoreSet | diversity (foundation feat.) | Sener 2018 + Kirillov 2023 (SAM) |
| **P8** | TypiClust (on SAM features) | diversity (foundation feat.) | Hacohen et al., ICML 2022 |
| **P9** | PAAL | predictor-based | Shi et al., IJCAI 2024 |

Build any method programmatically: `from medal_bench.policies import build; policy = build("P4")`.

---

## Installation

```bash
git clone <your-repo-url> && cd medal-bench   # this directory
conda create -n medal-bench python=3.10 && conda activate medal-bench

# 1) install PyTorch matching YOUR CUDA first (we used CUDA 12.1; torch 2.4.1 still supports V100/sm_70):
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
# 2) the rest:
pip install -r requirements.txt
# 3) (optional) make the package importable from anywhere:
pip install -e .
```

All commands below are run from this directory; if you skip `pip install -e .`, prefix with
`PYTHONPATH=$(pwd)`.

**For P7/P8 (foundation features)** download the SAM ViT-H checkpoint once:
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

---

## Data

Datasets are **public** but **not bundled** (licensing + size). Each dataset has an adapter under
`medal_bench/data/adapters/`; the adapter docstring documents the **exact on-disk layout** it expects and
the **download source**. Arrange downloaded data under a single `--data-root`:

```
<data-root>/
├── 2d/<dataset>/...    # native-2D sets (dermoscopy, fundus, endoscopy, ultrasound, histology, X-ray, OCT…)
└── 3d/<dataset>/...    # volumetric sets, sliced to 2D on load (CT/MRI; MSD, BTCV, MMWHS, …)
```

The registry `medal_bench/data/adapters/DATASET_REGISTRY` maps a dataset name → its adapter + path. To add
a dataset, drop in an adapter (subclass `MedALDataset`, return `Sample(sample_id, image (C,H,W) float32,
mask (H,W) int64, patient_id)`) and one registry line. Confirm a dataset loads:

```bash
python -c "from medal_bench.data.adapters import DATASET_REGISTRY as R; \
d=R['busi']('<data-root>'); print(len(d), d.num_classes, d[0].image.shape)"
```

---

## Quickstart — run one (method, dataset, seed) cell

```bash
python -m medal_bench.runner.run_one \
  --policy P4 --dataset busi --seed 1000 \
  --profile bench512_v5 --out-dir runs/demo \
  --data-root <data-root> --device cuda:0
```

This runs the full AL loop (init set → train U-Net from scratch → evaluate → acquire `k` samples → repeat
over the budget grid) and writes a JSONL trajectory to `runs/demo/busi__P4__s1000.jsonl` (one line per
budget round, with `metrics.mean_dsc_fg` = per-case macro foreground Dice). It is **idempotent** (skips a
complete cell; `--force` to rerun).

P7/P8 additionally need SAM features — add `--foundation sam --sam-model-type vit_h --sam-checkpoint
sam_vit_h_4b8939.pth`. Pre-warm them once per dataset to avoid recompute:
`python -m medal_bench.runner.precompute_sam --datasets busi --seeds 1000 --profile bench512_v5 --sam-model-type vit_h`.

---

## Run the full benchmark

The matrix is **datasets × {P0…P9} × {seed 1000, 2000, 3000}**. `dispatch.py` claims and runs cells from a
shared queue (resumable, lease-based, idempotent — safe to run many workers in parallel):

```bash
python -m medal_bench.runner.dispatch \
  --datasets busi,isic2018,glas2015,refuge,kvasir_seg \
  --methods P0,P1,P2,P3,P4,P5,P6,P7,P8,P9 \
  --seeds 1000,2000,3000 --profile bench512_v5 \
  --out-dir runs/frozen_v5 --foundation sam --sam-model-type vit_h --prefer heavy
```

Example SGE/SLURM worker scripts (one worker per GPU; re-submit to add workers) are in `submit/`.

---

## Analyze results

**The full 19-dataset results are bundled** in [`results/`](results/) — 570 cells (19 datasets × 10
methods × 3 seeds) of budget→Dice trajectories, plus a tidy [`results/summary_19set.csv`](results/summary_19set.csv).
So the analysis runs out-of-box (no GPU, no re-running the benchmark):

```python
from medal_bench.analysis.derived import load_curves, aubc, average_rank, win_rate
curves = load_curves("results/frozen_v5")        # {(dataset, method, seed): [(budget_frac, dsc), ...]}
score  = aubc(*zip(*curves[("busi","P4",1000)])) # area under the budget→Dice curve
```

Reproduce the headline analysis with the included scripts (they read `results/frozen_v5/` directly):
`scripts/analyze_benchmark.py` (per-dataset rankings, AUBC, gain-over-Random, differentiation),
`scripts/analyze_skill_ceiling.py` (oracle vs fixed-best ceiling), and `scripts/budget_sensitivity.py`
(per-budget method ranking). See [`results/README.md`](results/README.md) for the file format.

---

## Reproducibility

The canonical configuration (profile **`bench512_v5`**, in `medal_bench/profiles/__init__.py`):

| knob | value |
|---|---|
| image size | 512×512, letterbox (aspect-preserving) pad |
| training | from-scratch nnU-Net-style 2D U-Net, AdamW lr 1e-3, batch 12, **adaptive train-to-plateau** (1000–5000 iters) per round |
| budget grid | pool-dependent, ~1–20% of the AL pool (low-budget regime) |
| metric | `mean_dsc_fg_case_macro` — per-case macro foreground Dice, restricted to the valid (un-padded) region |
| seeds | 1000, 2000, 3000 (the init set is shared across methods within a seed) |
| determinism | deterministic cuDNN + a deterministic CE reduction (`trainer.py`) |

**GPU-arch / TF32 note (important for cross-machine reproducibility).** TF32 exists on Ampere/Hopper
(A40/H100) but not Volta (V100), so the *same* code gives slightly different numbers across GPU
architectures. We therefore **pin each dataset's full method×seed matrix to a single GPU architecture**.
Within one architecture results are deterministic; do not mix archs for a single dataset.

---

## Results

On the 19-dataset, 3-seed run (570 cells), per-method average rank (1 = best): **BADGE 3.42**, Ent+Core
3.74, TypiClust 4.47, **Random 4.68**, BALD/SAM-CoreSet 5.42, Entropy 6.11, CoreSet 6.53, SelUnc 7.47,
**PAAL 7.74**. Statistically: methods differ overall (Friedman p=6.5e-6), but this is a **floor, not a
ceiling** — the top cluster (BADGE/Ent+Core/TypiClust/Random) is inseparable (BADGE-vs-Random Wilcoxon
p=0.073), while PAAL/SelUnc/Entropy/CoreSet are reliably *below* Random. This matches the cold-start AL
literature: uncertainty/predictor-based acquisition is detrimental at low budget, and only diversity/hybrid
methods are non-detrimental. PAAL's last-place rank is a faithful *low-budget* result — it is designed for
higher budgets / 3D / joint predictor-segmentation training, which this benchmark deliberately does not use.

---

## Repository structure

```
medal_bench/
├── policies/      # the 10 AL methods P0–P9 (+ helpers, registry); see METHODS.md
├── data/          # MedALDataset interface + per-dataset adapters + label remapping
├── models/        # 2D U-Net (nnU-Net-style)
├── features/      # SAM-H (vit_h) foundation-feature extraction (for P7/P8)
├── runner/        # run_one (one cell), dispatch (the matrix), al_loop, trainer, eval, splits
├── profiles/      # run configs (bench512_v5 = canonical) + budget grids
├── analysis/      # AUBC, rankings, win-rate, regret (derived.py)
└── tests/         # unit tests (pytest medal_bench/tests)
results/           # bundled 19-dataset results: 570 cell JSONLs + summary_19set.csv (see results/README.md)
scripts/           # headline-analysis scripts (analyze_benchmark, analyze_skill_ceiling, budget_sensitivity)
submit/            # example cluster (SGE/SLURM) worker + launch scripts
METHODS.md  references.bib  requirements.txt  pyproject.toml
```

---

## Citing

If you use this benchmark, please cite the methods you run (BibTeX in [`references.bib`](references.bib))
and this repository. The benchmark deliberately re-implements each acquisition function faithfully to its
original paper — see [`METHODS.md`](METHODS.md) for the paper→code mapping.

## License

MIT (see `pyproject.toml`). Add a `LICENSE` file before publishing if you want a different license. Dataset
licenses are governed by each dataset's original source (see the adapter docstrings).
