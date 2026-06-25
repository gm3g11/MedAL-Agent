# BTCV / Synapse — Remap Decision Note (Stage −1 deliverable E)

Date 2026-06-13. Decision needed before BTCV is registered + headline-reported.

## On-disk facts (verified by full-volume scan, 30 cases)

- Root: `/groups/echambe2/datasets/data/3d/btcv_synapse/extracted/btcv_ct/`
- Layout: `imagesTr`(27) + `labelsTr`(27), `imagesVa`(3) + `labelsVa`(3) =
  **30 labeled cases, all image/label paired. No image-only test set.**
- Format: NIfTI, image int16, **label float32 (integer-valued), 512×512×Z, slice 3.0 mm.**
- **Native label codes (union over all 30): {0,1,2,3,4,5,6,7,8,9,10,11,12,16}.**
  - 13, 14, 15 are **absent everywhere**; **16 is present in every volume**, organ-sized
    (Case0001: code 16 = 33,699 voxels, comparable to IVC/portal-vein).

## The ambiguity

Standard BTCV (Synapse syn3193805) labels its **13 organs as codes 1–13**:
1 spleen · 2 R-kidney · 3 L-kidney · 4 gallbladder · 5 esophagus · 6 liver ·
7 stomach · 8 aorta · 9 IVC · 10 portal&splenic vein · 11 pancreas ·
12 R-adrenal · 13 L-adrenal.

**This copy uses 16 instead of 13.** The `BTCV_LABELS_GUIDE.txt` on disk explains
how the labels were re-downloaded but does **not** give the code legend, so the
**semantic identity of code 16 cannot be confirmed from on-disk artifacts alone.**
Most likely it is the 13th organ (left adrenal gland) carried under a non-standard
code, but this is **unconfirmed**.

## Decision

**Two-tier decision that unblocks training now without over-claiming:**

1. **Training & aggregate metrics — SAFE NOW.** Dense remap
   `BTCV_REMAP = {0→0, 1→1, …, 12→12, 16→13}` (14 classes; already in
   `data/remap.py`, test `test_btcv_dense_remap_after_semantic_confirmation`
   passing). Correctness of the loss, mean/macro/foreground DSC, and HD95 does
   **not** depend on knowing the *name* of class 13 — only that code 16 is a
   distinct valid foreground class, which the scan confirms. The remapper
   hard-errors if any case ever shows 13/14/15 or another code.

2. **Per-organ reporting — BLOCKED pending confirmation.** Do **not** print a
   per-organ DSC label for dense-class 13 (native 16) until its identity is
   confirmed against the original Synapse `dataset.json` (Abdomen/RawData). Until
   then report it as `organ_16 (unconfirmed)` and mark BTCV **Tier C (provenance
   caveat)** in the inclusion table.

## Recommended action

- **Register BTCV** as `btcv_synapse` (CT, 14 classes) for Stage 1 using the dense
  remap above — it is training-correct today. (Adapter mirrors the MMWHS pattern:
  nibabel integer read, axial-axis slicing, HU window, `BTCV_REMAP`. Not yet
  written/registered — pending your go-ahead on this note.)
- **Before Stage 2 headline use**, confirm code-16 identity from Synapse
  `dataset.json`; update the per-organ legend. If confirmation is impossible,
  keep BTCV as a supplementary (Tier C) multiclass dataset, not a headline organ
  benchmark.
- **Split note:** the 30 labeled cases are pooled; the existing case-disjoint
  splitter handles train/val/test from the 30. The on-disk `imagesVa` (3) is just
  a folder split, not a sacred test set — fold all 30 into the case-disjoint
  splitter for consistency with other datasets.

## Open question for you
Do you accept option (1)+(2) — register BTCV now with `16→13` for training, label
class 13 as "unconfirmed" until verified — or hold BTCV entirely until code 16 is
confirmed against Synapse?
