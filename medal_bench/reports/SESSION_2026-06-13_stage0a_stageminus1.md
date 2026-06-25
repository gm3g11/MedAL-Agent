# Session summary — 2026-06-13: Stage 0a + Stage −1 build (B1/B3/B8)

For the returning reviewer. What I ran, what I built, what's verified, what's next.
All work is on the local filesystem (no commits made — repo is not a git repo).

## TL;DR

- **Stage 0a: PASS (with documented warnings).** Loop/logging/determinism/SAM-H
  path/firewall all validated on wired datasets. → `reports/stage0a_smoke_2026-06-13.md`
- **Stage −1 plan written** (B1–B8, ordered, with risks/tests). → `reports/stage_minus1_plan_2026-06-13.md`
- **B1 (keystone) DONE + verified:** remap infrastructure + MMWHS adapter
  (`mmwhs_ct` 60 cases / `mmwhs_mr` 46 cases) wired, remapped, orientation-correct,
  case-disjoint, with a visual overlay. 14 new tests pass.
- **B3 (pool-dependent budgets) DONE:** `profiles/budget.py` + 8 tests.
- **B8 (frozen v2) DRAFT:** `profiles/frozen_v2.py` + 6 tests.
- **BTCV decision note written** (code 16 ambiguity). → `reports/btcv_remap_decision_2026-06-13.md`
- **Test suite: 124 passing** (102 prior + 22 new). No regressions.
- **Stage 0b interim PASS** (MMWHS-CT multiclass validation): 15/15 cells OK; the
  8-class remap path runs end-to-end (loss + 8-class DSC/HD95 + query, no
  class-index error). → `reports/stage0b_interim_2026-06-13.md`

## Deliverables status (vs your §15 list)

| # | Deliverable | Status |
|---|---|---|
| A | Run Stage 0a + report | ✅ `reports/stage0a_smoke_2026-06-13.md` |
| B | Stage −1 implementation plan | ✅ `reports/stage_minus1_plan_2026-06-13.md` |
| C | Implement B1–B8 (safe subset) | 🔨 B1 ✅, B3 ✅, B8 draft ✅; B2/B4/B5/B6/B7 planned (need your sign-off — see below) |
| D | Wire MMWHS (adapter+remap+tests+smoke+overlay) | ✅ adapter+remap+tests+overlay; loader smoke = Stage 0b interim (running) |
| E | BTCV remap decision note | ✅ `reports/btcv_remap_decision_2026-06-13.md` (needs your decision) |
| F | Frozen config v2 draft | ✅ `profiles/frozen_v2.py` (DRAFT; resolution not final-frozen) |
| G | Stage 0b on MMWHS + report | ✅ interim PASS (15/15 cells, `reports/stage0b_interim_2026-06-13.md`); formal adaptive-512 run waits on B2 |

## New/changed files

- `data/remap.py` — `LabelRemapper` (vectorized LUT, hard-errors on unknown codes)
  + `MMWHS_REMAP`, `BTCV_REMAP`, `MYOPS_REMAP`, `BRATS_REMAP`.
- `data/adapters/mmwhs.py` — `MMWHSAdapter(root, modality)`; axial-axis-from-affine
  slicing (handles MR axial-on-axis-1 + 2 CT degenerate affines); CT HU window /
  MR percentile norm; applies MMWHS_REMAP.
- `data/adapters/__init__.py` — registered `mmwhs_ct`, `mmwhs_mr` (registry now 11).
- `profiles/budget.py` — `budget_grid(N, num_classes)` (Cases A–D + initial floor/cap).
- `profiles/frozen_v2.py` — frozen v2 draft + `FROZEN_V2_HASH`.
- `tests/`: `test_remap.py` (8), `test_mmwhs_adapter.py` (6), `test_budget.py` (8),
  `test_frozen_v2.py` (6).
- Reports + run artifacts under `reports/` and `runs/stage0a`, `runs/stage0b`.

## Key verified facts

- MMWHS on disk: 106 cases (CT 60, MR 46); native label codes EXACTLY
  {0,205,420,421,500,550,600,820,850} (421 only in Case3010) → dense {0..7}. ✓
- MMWHS gotchas found + handled: MR's axial axis is axis 1 (not 2); CT Case2009 &
  Case2017 have all-zero affines (axcodes None) → fall back to last axis.
- Overlay (`runs/stage0b/overlays/mmwhs_ct_overlay.png`) shows an anatomically
  correct axial cardiac slice with all 8 classes on the contrast blood pools.
- mmwhs_ct case-disjoint split: 48/6/6 cases (disjoint), 10976/1437/1260 slices →
  Case-C budget grid [28,55,110,220,550,1100].
- SAM-B vs SAM-H caches are collision-free (distinct encoder_id keys); 3 vit_h
  caches built during Stage 0a.
- BTCV: 30 paired cases, codes {0..12,16}; 16→13 remap is training-safe but code
  16's organ identity is unconfirmed (Tier-C caveat).

## What I deliberately did NOT build (need your input)

These are planned in the Stage −1 doc but I held off — they're either medium/high
risk or depend on a decision only you should make:

- **B2 adaptive resolution** — touches the hot resize/collate path AND changes the
  frozen preprocessing. Provisional default 512 is recorded in frozen v2 but the
  final freeze waits on your resolution decision + the Stage 1 512/640/768
  sensitivity. The formal adaptive-512 Stage 0b run depends on this.
- **B6 throughput fix** — highest-risk (central `_IndexedSubset` path). The slow
  Stage 0b you'll see in the log is exactly this problem (each cell reloads
  volumes). I want to confirm the disk-cache approach with you before touching it.
- **B4 full-sup runner / B5 derived metrics / B7 SAM precompute** — safe, additive;
  I can knock these out next session; they're needed for the Stage 1 *report*, not
  Stage 0b.

## Recommended next steps (in order)

1. Approve the BTCV note (register now with `16→13`, or hold).
2. Confirm bench resolution 512 + green-light B2 (adaptive resolution) so the
   formal Stage 0b + Stage 1 can use it.
3. Approve the B6 disk-cache approach (cache resized arrays keyed by
   dataset+preprocess+remap+resolution) before I touch the loader.
4. Then: B4 + B5 + B7, re-freeze v2, formal Stage 0b, Stage 1.
