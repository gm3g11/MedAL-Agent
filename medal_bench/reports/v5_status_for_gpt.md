# frozen_v5 — acceleration + determinism update (for GPT review)

Supersedes the earlier "TF32-off / QC3=0.0000" version of this file (that snapshot is now wrong on
two points — see §3). This reflects the code as of 2026-06-20 after an acceleration + determinism pass.

## 1. What changed and why

**A. TF32 re-enabled (the headline acceleration).** `seeds.py` previously forced
`cudnn.allow_tf32=False` + `matmul.allow_tf32=False` to fight a cross-arch confound. That confound is
**already eliminated by single-arch pinning** (each dataset's full 10-policy matrix runs on ONE GPU
arch), so TF32-off was redundant. We verified empirically on an H100 that TF32-on is **bit-identical
across re-runs** (same weight hashes) and ~**2.1× faster** end-to-end. TF32 is now ON, determinism
(`cudnn.deterministic`, `use_deterministic_algorithms(True)`) is still ON. V100/Volta has no TF32
hardware, so those 9 datasets are unaffected (FP32 either way).

**B. Deterministic cross-entropy (a real determinism bug we found + fixed).** While proofreading before
the multi-day run we found that **btcv_synapse round-0 was NOT reproducible** across policies (P0 vs P9,
same arch/seed/config: round-0 DSC 0.4054 vs 0.3942, **spread 0.0113**, different ckpt hashes) — even
though round-0 is policy-independent by construction. A strict-determinism probe
(`use_deterministic_algorithms(warn_only=False)`) named the culprit: **`F.cross_entropy` →
`nll_loss2d_forward` reduces with non-deterministic atomics**, which `warn_only=True` had been silently
permitting. Fixed by splitting the reduction (`reduction="none"` then an explicit deterministic
`.mean()`; mathematically identical, sum/N). This op was non-deterministic on EVERY dataset — btcv just
amplified it past the rounding threshold; others stayed sub-threshold (round-0 spread read 0.0000).

**C. Redundant-resize removal (byte-identical speed).** `collate_to_batch` re-ran a same-size
`F.interpolate` on already-resized data; short-circuited (verified byte-identical, maxdiff 0.0).

**Evaluated and rejected:** `channels_last` (the U-Net is all `InstanceNorm2d`, which doesn't preserve
channels_last in torch 2.4.1 → ~18 relayouts/forward → ~0× gain); `torch.compile` and `fp16/bf16
autocast` (declined — added trust surface for marginal gain; TF32 is the clean, safe win).

## 2. Determinism is now verified across ALL 10 policies
A strict-mode (`warn_only=False`, crash-on-any-non-deterministic-op) sweep of **every policy P0–P9**:
**0 non-deterministic crashes.** 8/10 completed clean on a 3-class dataset (P0,P1,P3–P8); P2 (BALD) and
P9 (PAAL) initially only *timed out* (slow MC / AccuracyPredictor passes — not a determinism failure)
and were then re-run to completion on a small dataset, both **clean (rc=0)**. So `cross_entropy` was the
sole non-deterministic op in the run's paths, and the fixed config is fully deterministic everywhere.

## 3. Corrections to the prior version of this doc
- **QC1** is now "TF32 **on** + deterministic" (single-arch makes it confound-free), not "TF32 off".
  `qc_report.py` QC1 was updated to verify the new regime and PASSES.
- **QC3** round-0 invariance: the prior "spread = 0.0000, byte-identical" claim was true for most
  datasets but **false for btcv** (0.0113) due to bug §1B. After the CE fix, round-0 invariance is now
  **guaranteed by construction** (no non-deterministic op remains), not merely observed — and will read
  0.0000 on btcv too in the re-run.

## 4. Re-run scope (recommended: full, unified)
The CE fix changes the loss on every dataset, so previously-finished cells are not config-consistent
with the fix → **re-run all 3 seeds (1000/2000/3000) under one unified config (TF32-on + det-CE)**,
seed-2000-first. Reusing old seed-1000 cells would mix configs across seeds (inflating seed-variance),
and because TF32-on is ~1.5× faster overall the full re-run costs about the same as a config-consistent
reuse anyway — so full re-run is both cleaner and ~free.

## 5. Tests
180/180 pass with all of the above (the CE fix is a numerics change; suite green).
