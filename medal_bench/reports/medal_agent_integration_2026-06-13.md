# medal_agent ↔ medal_bench integration (bridge) — 2026-06-13

## What changed

You built `/groups/echambe2/datasets/medal_agent/` — an audited 21-dataset loader
(pre-sliced 2D PNGs, case-disjoint split manifests, strict dense remaps, Check A/B +
loader-smoke + 168 overlays all green). It was **not connected** to the `medal_bench`
P0–P9 AL runner, which consumes a different interface (`MedALDataset → Sample`,
builds its own `make_split`).

I added a **bridge** so the runner can consume all 21 audited datasets:
- `medal_bench/data/adapters/medal_agent_bridge.py` — `MedalAgentBridge(MedALDataset)`
  wrapping `medal_agent.SlicedDataset`; `register_medal_agent_datasets()` adds all 21
  to `DATASET_REGISTRY` (no-op if medal_agent is unimportable).
- `data/adapters/__init__.py` — calls the registrar; `BRIDGED_DATASETS` lists the ids.

**Registry is now 32 datasets** = 11 medal_bench-native (busi, kvasir_seg, isic2018,
cvc_clinicdb, glas2015, hyperkvasir_seg, origa, promise12, msd07_pancreas + my interim
mmwhs_ct/mmwhs_mr) + 21 bridged medal_agent datasets.

Verified: bridge yields the correct `Sample` contract (image `(C,H,W)` float32 [0,1],
mask `(H,W)` int64 ≤ C−1) and **case-disjoint splits** via the runner's `make_split`
for mmwhs (C=8), ext_brats2020 (C=4), msd_task09_spleen (C=2), btcv_synapse (C=14).

## Two decisions for you

### F1 — splits: re-split vs honor medal_agent's native splits
The runner ALWAYS re-splits via `make_split(adapter, seed)`. The bridge exposes a
medal_agent split (default `train`) as the universe with `patient_id = case_id`, so
`make_split` re-carves **case-disjoint** train/val/test (leakage-free). This is correct
and sufficient for the pre-Stage-1 smoke, BUT it **discards medal_agent's curated
native splits** (e.g. btcv's native 27/3 train/val, the `test_has_labels` test sets).

For **formal Stage 1** you likely want to honor medal_agent's native splits + the
`test_has_labels` gate (eval on the real held-out test where labels exist). That needs
a small runner hook: let `run_al` accept an externally-provided split instead of always
calling `make_split`. **Recommend I add that hook before formal Stage 1.** (Low risk:
optional param; default behavior unchanged.)

### F2 — duplication: my interim MMWHS work is superseded
My Stage −1 B1 (`data/adapters/mmwhs.py` + `data/remap.py`, ids `mmwhs_ct`/`mmwhs_mr`,
reading raw NIfTI from `data/3d/mmwhs/extracted`) **duplicates** medal_agent's audited
mmwhs (pre-sliced + remapped). The bridged id `mmwhs` (medal_agent) is the source of
truth. **Recommend: deprecate `mmwhs_ct`/`mmwhs_mr` + `data/remap.py` + their tests in
favor of the bridge.** I left them in place (harmless, tests still pass) rather than
delete without your OK. Note: **B3 (budget), B5 (derived metrics), B8 (frozen v2) are
NOT duplicated** by medal_agent and remain in use.

## Modality/mask defaults chosen for multi-view datasets
`mmwhs→ct`, `ext_brats2020→t1ce`, `care_leftatrium_2026→mask_subdir=atrium`; all others
use medal_agent's first modality/mask. Other views (mmwhs mr, brats t1/t2/flair, myops
c0/lge/t2, care_la scar) can be added as extra ids when wanted.

## On "run everything before Stage 1"
Pre-Stage-1 = a loop-smoke across all 21 bridged datasets (legacy smoke config:
img128, pool_cap=32) to prove every dataset loads + trains + the C-class path runs
end-to-end through the runner. Fast because medal_agent serves PNGs (no per-cell NIfTI
reload — unlike my interim MMWHS adapter). Results → `reports/stage_pre1_*` (in progress).
This is NOT a scientific benchmark; it's the integration gate before the formal,
adaptive-resolution, pool-dependent-budget Stage 1.
