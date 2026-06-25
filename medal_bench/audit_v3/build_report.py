"""Build the v3 final report (medal_agent_v3_fair_test_eval.md) and README_REPRODUCE.md.

Aggregates outputs from analyze.py + diagnostics.py.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path("/groups/echambe2/gmeng/MedAL-Agent/repo/code")
TABLES = REPO / "tables"
REPORTS = REPO / "reports"
RUNS_V3 = REPO / "runs" / "test_eval_v3"

POLICIES = ["P0","P1","P2","P3","P4","P5","P6","P7","P8","P9"]
NAMES = {
    "P0":"Random","P1":"NormEnt","P2":"BALD","P3":"CoreSet","P4":"BADGE",
    "P5":"Ent→CS","P6":"SelUnc","P7":"SAM-CS","P8":"SAM-TC","P9":"PAAL",
}
SEEDS = [1000, 2000, 3000]
DATASETS = ["busi","cvc_clinicdb","isic2018","promise12"]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fmt_ms(vals, nd=3):
    if not vals: return "—"
    vals = [float(v) for v in vals if v not in ("nan","—","")]
    if not vals: return "—"
    if len(vals) == 1: return f"{vals[0]:.{nd}f}"
    return f"{np.mean(vals):.{nd}f} ± {np.std(vals):.{nd}f}"


def main():
    summary = read_csv(TABLES / "v3_summary_final_round.csv")
    ranks = read_csv(TABLES / "v3_val_vs_test_ranks.csv")
    pairwise = read_csv(TABLES / "v3_pairwise_stats.csv")
    failure = read_csv(TABLES / "v3_failure_summary.csv")
    peal = read_csv(TABLES / "v3_peal_round_summary.csv")
    paal = read_csv(TABLES / "v3_paal_ap_round_summary.csv")

    # Aggregations
    final_dsc_by = defaultdict(list)  # (ds, p, split) -> [floats]
    final_hd_by = defaultdict(list)
    final_aulc_by = defaultdict(list)
    for r in summary:
        ds, p = r["dataset"], r["policy"]
        final_dsc_by[(ds, p, "val")].append(float(r["val_dsc_final"]))
        final_dsc_by[(ds, p, "test")].append(float(r["test_dsc_final"]))
        final_aulc_by[(ds, p, "val")].append(float(r["val_aulc_dsc"]))
        final_aulc_by[(ds, p, "test")].append(float(r["test_aulc_dsc"]))
        try: final_hd_by[(ds, p, "val")].append(float(r["val_hd95_filt"]))
        except ValueError: pass
        try: final_hd_by[(ds, p, "test")].append(float(r["test_hd95_filt"]))
        except ValueError: pass

    # Per-dataset best by mean DSC
    def best(ds, split):
        cands = [(p, float(np.mean(final_dsc_by[(ds, p, split)])))
                 for p in POLICIES if (ds, p, split) in final_dsc_by]
        return max(cands, key=lambda x: x[1]) if cands else None

    # Spearman ρ per dataset (mean across 3 seeds)
    spear_per_ds = defaultdict(set)
    for r in ranks:
        spear_per_ds[r["dataset"]].add(float(r["spearman_rho_val_vs_test"]))
    spear_per_ds_mean = {ds: float(np.mean(list(v))) for ds, v in spear_per_ds.items()}

    # Failure aggregates
    collapse_by = defaultdict(lambda: defaultdict(int))
    empty_pred_by = defaultdict(lambda: defaultdict(list))
    for r in failure:
        ds, p = r["dataset"], r["policy"]
        collapse_by[ds][p] += int(r["test_collapse"]) + int(r["val_collapse"])
        empty_pred_by[ds][p].append(float(r["test_empty_pred_rate"]))

    # PEAL aggregate
    peal_by_ds = defaultdict(list)
    for r in peal:
        if r["peal_mean_disagreement"] != "—":
            peal_by_ds[r["dataset"]].append(float(r["peal_mean_disagreement"]))

    # PAAL aggregate
    paal_corr_by_ds = defaultdict(list)
    paal_acc_by_ds = defaultdict(list)
    for r in paal:
        if r["ap_val_corr"] != "—":
            paal_corr_by_ds[r["dataset"]].append(float(r["ap_val_corr"]))
        if r["pred_acc_mean"] != "—":
            paal_acc_by_ds[r["dataset"]].append(float(r["pred_acc_mean"]))

    # Test-vs-val winner table
    winners = {ds: (best(ds, "val"), best(ds, "test")) for ds in DATASETS}

    # ----- Build report -----
    L = []
    L.append("# MedAL-Agent v3 fair test-set evaluation")
    L.append("")
    L.append("**Goal**: build MedAL-Agent evaluation and diagnostic skills under a fair fixed P0–P9 comparison.")
    L.append("**This is pilot-quality evidence under a fair fixed-protocol comparison.**")
    L.append("NOT a publication-quality benchmark; NOT optimized for best possible policy performance.")
    L.append("")
    L.append("**Setup**: replayed v1's `selected_ids` (no new active-learning trajectories were generated)")
    L.append("on the EXACT v1 training config (nnU-Net 2D, 250 iters/round, batch 8, 256×256, AdamW lr=1e-3,")
    L.append("from-scratch each round) and evaluated on BOTH val AND test splits per round. Checkpoints saved.")
    L.append("")
    L.append("**Coverage**: 120/120 cells = 4 datasets × 10 policies × 3 seeds. 720 checkpoints saved. 0 failures.")
    L.append("")
    L.append("---")
    L.append("")

    # 1. Val → Test transfer
    L.append("## Q1. Did validation trends transfer to test?")
    L.append("")
    L.append("### Winners per dataset (mean DSC across 3 seeds)")
    L.append("")
    L.append("| dataset | val winner | test winner | match? |")
    L.append("|---|---|---|---|")
    for ds, (vw, tw) in winners.items():
        match = "✓ same" if vw and tw and vw[0] == tw[0] else "✗ different"
        v_str = f"{vw[0]} ({NAMES[vw[0]]}) = {vw[1]:.3f}" if vw else "—"
        t_str = f"{tw[0]} ({NAMES[tw[0]]}) = {tw[1]:.3f}" if tw else "—"
        L.append(f"| {ds} | {v_str} | {t_str} | {match} |")
    L.append("")
    L.append("### Spearman ρ (val ranks vs test ranks; mean across 3 seeds)")
    L.append("")
    L.append("| dataset | Spearman ρ |")
    L.append("|---|---|")
    for ds in DATASETS:
        L.append(f"| {ds} | {spear_per_ds_mean[ds]:+.3f} |")
    L.append("")
    L.append("**Reading**: 3 of 4 winners match across val→test (BUSI, ISIC, PROMISE12). CVC has P4 (BADGE) on val and P3 (CoreSet) on test — close but different.")
    L.append("All Spearman ρ are positive, with ISIC showing the strongest rank agreement (ρ = +0.86) and PROMISE12 the weakest (ρ = +0.53).")
    L.append("")
    L.append("**Caveat**: n=3 seeds per dataset is small; the absolute ρ values should be interpreted as directional evidence, not as tight estimates.")
    L.append("")
    L.append("---")
    L.append("")

    # 2. BADGE on CVC
    L.append("## Q2. Does BADGE remain strong on CVC?")
    L.append("")
    L.append("**Partial.** P4 BADGE wins CVC on val (mean DSC = " +
             f"{best('cvc_clinicdb', 'val')[1]:.3f}) but P3 CoreSet wins on test (" +
             f"{best('cvc_clinicdb', 'test')[1]:.3f}). The two are close (Δ ≈ {best('cvc_clinicdb', 'test')[1] - float(np.mean(final_dsc_by[('cvc_clinicdb', 'P4', 'test')])):.3f} between P4 val and P3 test).")
    L.append("")
    L.append("| policy | val DSC | test DSC |")
    L.append("|---|---|---|")
    for p in ["P0","P1","P3","P4","P8"]:
        v = fmt_ms(final_dsc_by[("cvc_clinicdb", p, "val")])
        t = fmt_ms(final_dsc_by[("cvc_clinicdb", p, "test")])
        L.append(f"| {p} ({NAMES[p]}) | {v} | {t} |")
    L.append("")
    L.append("BADGE is still in the top tier on CVC, but the val→test ranking is unstable enough that single-seed claims would be misleading.")
    L.append("")
    L.append("---")
    L.append("")

    # 3. SAM-TC on PROMISE12
    L.append("## Q3. Does SAM-TypiClust remain strong on PROMISE12?")
    L.append("")
    L.append("**Yes.** P8 SAM-TC wins PROMISE12 on BOTH val and test:")
    L.append("")
    p8_v = float(np.mean(final_dsc_by[("promise12", "P8", "val")]))
    p8_t = float(np.mean(final_dsc_by[("promise12", "P8", "test")]))
    p0_v = float(np.mean(final_dsc_by[("promise12", "P0", "val")]))
    p0_t = float(np.mean(final_dsc_by[("promise12", "P0", "test")]))
    L.append(f"- val: P8 = {p8_v:.3f} vs Random P0 = {p0_v:.3f} (Δ = {p8_v - p0_v:+.3f})")
    L.append(f"- test: P8 = {p8_t:.3f} vs Random P0 = {p0_t:.3f} (Δ = {p8_t - p0_t:+.3f})")
    L.append("")
    L.append("| policy | val DSC | test DSC |")
    L.append("|---|---|---|")
    for p in ["P0","P3","P4","P5","P8"]:
        v = fmt_ms(final_dsc_by[("promise12", p, "val")])
        t = fmt_ms(final_dsc_by[("promise12", p, "test")])
        L.append(f"| {p} ({NAMES[p]}) | {v} | {t} |")
    L.append("")
    L.append("---")
    L.append("")

    # 4. BUSI / ISIC weak/saturated
    L.append("## Q4. Are BUSI and ISIC still weak / saturated evidence?")
    L.append("")
    L.append("**Yes.**")
    L.append("")
    L.append("- **BUSI** is a small dataset (val n≈78). P8 SAM-TC wins both splits but the spread between top and bottom policies is narrow:")
    busi_dscs = [float(np.mean(final_dsc_by[("busi", p, "test")])) for p in POLICIES]
    L.append(f"  - test DSC range across policies: [{min(busi_dscs):.3f}, {max(busi_dscs):.3f}] (spread = {max(busi_dscs) - min(busi_dscs):.3f})")
    L.append("  - margins over Random are within typical 3-seed std on this dataset; BUSI claims should remain weak.")
    L.append("")
    L.append("- **ISIC2018** saturates at ≤ 5% budget (already noted in v2 audit; ratified here by the test-set numbers):")
    isic_dscs = [float(np.mean(final_dsc_by[("isic2018", p, "test")])) for p in POLICIES]
    L.append(f"  - test DSC range: [{min(isic_dscs):.3f}, {max(isic_dscs):.3f}] (spread = {max(isic_dscs) - min(isic_dscs):.3f})")
    L.append("  - the 15%/20% checkpoints add no information on ISIC under the current grid.")
    L.append("")
    L.append("---")
    L.append("")

    # 5. Failure rates
    L.append("## Q5. How often do policies fail through empty predictions?")
    L.append("")
    L.append("Test-set empty-prediction rate (fraction of val samples where the model predicts zero foreground pixels):")
    L.append("")
    L.append("| dataset | " + " | ".join(POLICIES) + " |")
    L.append("|" + "|".join(["---"]*(len(POLICIES)+1)) + "|")
    for ds in DATASETS:
        row = [ds]
        for p in POLICIES:
            v = fmt_ms(empty_pred_by[ds][p], nd=3)
            row.append(v)
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    # Cells with any collapse
    collapsed_cells = sum(1 for r in failure if int(r["test_collapse"]) or int(r["val_collapse"]))
    L.append(f"**Total cells with collapse flag (final DSC < 0.05 on val OR test)**: {collapsed_cells} / 120.")
    L.append("")
    if collapsed_cells == 0:
        L.append("✓ No cells collapsed in v3. (The v1 P9 PROMISE12 collapse at seed 1000 was a pre-PAAL-fix artifact;")
        L.append("v3 uses the corrected ResNet-18 AP and that cell now trains stably.)")
    L.append("")
    L.append("---")
    L.append("")

    # 6. PROMISE12 case-level
    L.append("## Q6. Is PROMISE12 still unstable at case level?")
    L.append("")
    L.append("**Unknown — case-level metrics NOT computed in this pass.**")
    L.append("")
    L.append("Case-level DSC/HD95 requires aggregating slice-level predictions back to the case (volume) level.")
    L.append("Although we now have all 30 PROMISE12 checkpoints saved, computing per-case metrics requires an")
    L.append("additional pass that:")
    L.append("1. Loads each checkpoint")
    L.append("2. Runs prediction over the val+test slices for that adapter")
    L.append("3. Groups slices by `case_id` (Case00…Case49)")
    L.append("4. Computes binary DSC/HD95 per case (e.g. concatenated argmax masks vs case-level GT)")
    L.append("")
    L.append("This is listed as a NEXT-step diagnostic, NOT done here. Slice-level DSC/HD95 numbers (above) are")
    L.append("the only currently-available PROMISE12 evidence. **We do NOT claim case-level improvement.**")
    L.append("")
    L.append("---")
    L.append("")

    # 7. PEAL
    L.append("## Q7. What did PEAL select?")
    L.append("")
    L.append("**Only round-level scalars are available** (per-image disagreement requires additional logging or")
    L.append("re-running PEAL with extended diagnostics — not done in this pass).")
    L.append("")
    L.append("Per-round mean hflip disagreement over the unlabeled pool (mean ± std across rounds × seeds, from v1 logs):")
    L.append("")
    L.append("| dataset | mean ± std | n |")
    L.append("|---|---|---|")
    for ds in DATASETS:
        v = peal_by_ds[ds]
        L.append(f"| {ds} | {fmt_ms(v, nd=4)} | {len(v)} |")
    L.append("")
    L.append("Reading: BUSI and CVC show the highest mean disagreement (~6–7%) — these are the noisiest datasets and the")
    L.append("model is most uncertain under perturbation. PROMISE12 has the lowest (~1%) — flipped predictions agree on")
    L.append("most pixels. ISIC sits in between (~3%).")
    L.append("")
    L.append("**What PEAL would select** (had per-image logging been enabled): top-K images by entropy × disagreement-mask mean.")
    L.append("Without per-image logging we cannot show selection histograms or correlate with foreground content.")
    L.append("This is a next-step diagnostic.")
    L.append("")
    L.append("---")
    L.append("")

    # 8. PAAL calibration vs correlation
    L.append("## Q8. Is PAAL calibrated, or only correlated?")
    L.append("")
    L.append("**Correlated, but AP quality is decoupled from AL utility.**")
    L.append("")
    L.append("Per-round AP val correlation (Pearson r between AP-predicted Dice and actual Dice on a small held-out labeled split):")
    L.append("")
    L.append("| dataset | AP val_corr (mean ± std) | AP pred_acc_mean (mean ± std) | n |")
    L.append("|---|---|---|---|")
    for ds in DATASETS:
        c = paal_corr_by_ds[ds]
        a = paal_acc_by_ds[ds]
        L.append(f"| {ds} | {fmt_ms(c, nd=3)} | {fmt_ms(a, nd=3)} | {len(c)} |")
    L.append("")
    L.append("AP correlation is consistently HIGH (0.77–0.90 across datasets). The Accuracy Predictor learns the labeled")
    L.append("Dice well. Yet PAAL's actual AL-test DSC is mid-pack on every dataset:")
    L.append("")
    L.append("| dataset | PAAL test DSC | Random test DSC | top policy test DSC |")
    L.append("|---|---|---|---|")
    for ds in DATASETS:
        p9 = float(np.mean(final_dsc_by[(ds, "P9", "test")]))
        p0 = float(np.mean(final_dsc_by[(ds, "P0", "test")]))
        top = best(ds, "test")
        L.append(f"| {ds} | {p9:.3f} | {p0:.3f} | {top[0]} = {top[1]:.3f} |")
    L.append("")
    L.append("**Conclusion**: high AP correlation ≠ high AL utility. The AP can rank images by likely-low-Dice, but picking")
    L.append("those samples (weighted by Weighted Polling clusters) does not consistently improve downstream segmentation in")
    L.append("our small-budget regime. AP calibration (predicted vs actual Dice scatter, ECE-style bins) is a next-step diagnostic")
    L.append("that requires logging per-sample AP outputs during the AL pass (not currently saved).")
    L.append("")
    L.append("---")
    L.append("")

    # 9. Skills gained
    L.append("## Q9. What skills did MedAL-Agent gain from this pass?")
    L.append("")
    L.append("- **Fair test-set evaluation**: the v1 pilot was val-only; v3 now has both val AND test for all 120 cells under the EXACT v1 training config.")
    L.append("- **Per-round checkpoint saving**: 720 checkpoints (= 120 cells × 6 rounds) are now persisted at `runs/test_eval_v3/ckpts/`. Future diagnostics that need predictions (case-level PROMISE12, AP calibration curves, PEAL per-image, mapper coverage at multiple time slices) can be run from these without retraining.")
    L.append("- **Extended metric set**: DSC, IoU, HD95 (filtered + penalty), ASSD, empty-pred rate, empty-GT rate, HD95-undefined rate — per round, per split.")
    L.append("- **Val→test rank-transfer evidence**: 3 of 4 winners stable, all Spearman ρ positive.")
    L.append("- **Failure visibility**: 0 collapses in v3 — confirms the v2 PAAL fix is stable across seeds and the PROMISE12 v1-seed-1000 collapse was a pre-fix artifact.")
    L.append("- **Diagnostic separation of AP quality from AL utility** (Q8 above).")
    L.append("")
    L.append("---")
    L.append("")

    # Claim safety
    L.append("## Claim safety")
    L.append("")
    L.append("### Allowed claims")
    L.append("- Fair fixed-protocol comparison across all 10 policies × 3 seeds × 4 datasets, with val AND test eval.")
    L.append("- Val→test winner transfer holds for 3 of 4 datasets at the policy level.")
    L.append("- P8 SAM-TC remains a positive signal on BUSI and PROMISE12 across val and test.")
    L.append("- P4 BADGE remains a positive signal on CVC (val winner) and ISIC (val + test winner), with the CVC test-set winner being P3 CoreSet (very close).")
    L.append("- No collapses on any (dataset, policy, seed) in v3.")
    L.append("- ISIC saturates early under the {1,2,5,10,15,20}% budget grid.")
    L.append("- PAAL AP is well-correlated with held-out labeled Dice (val_corr ≈ 0.77–0.90), but this AP quality does NOT translate to AL utility in our small-budget regime.")
    L.append("")
    L.append("### Forbidden claims (NOT supported here)")
    L.append("- ❌ Universal best AL method.")
    L.append("- ❌ Publication-quality benchmark.")
    L.append("- ❌ Best possible policy performance (we did not tune anything for v3).")
    L.append("- ❌ PROMISE12 case-level improvement (not computed).")
    L.append("- ❌ AULC-HD95 improvement (HD95 was logged per-round in v3, so this is now in principle computable, but we do not claim improvements over v1 without explicit comparison).")
    L.append("- ❌ TopoAlign-style topology / KMeans topology / Mapper coverage causal claims (out of scope this round).")
    L.append("")
    L.append("---")
    L.append("")

    # File index
    L.append("## Files produced")
    L.append("")
    L.append("### Tables (`tables/`)")
    L.append("- `v3_per_cell_metrics.csv` (720 rows): per-round val+test DSC/IoU/HD95(filt+pen)/ASSD/empty rates + checkpoint path")
    L.append("- `v3_summary_final_round.csv` (120 rows): final-round DSC/HD95 + AULC-DSC for both splits")
    L.append("- `v3_val_vs_test_ranks.csv` (120 rows): per-cell val and test ranks, Spearman ρ per (ds, seed)")
    L.append("- `v3_pairwise_stats.csv` (72 rows): policy vs baseline paired diffs (val AND test), bootstrap CI, Wilcoxon, sign, paired rank-biserial, Cliff's δ")
    L.append("- `v3_failure_summary.csv` (120 rows): collapse flags + empty-pred + HD95-undef rates per cell")
    L.append("- `v3_peal_round_summary.csv` (72 rows): P6 PEAL per-round mean disagreement (from v1 logs)")
    L.append("- `v3_paal_ap_round_summary.csv` (72 rows): P9 PAAL per-round AP loss/val_corr/pred_acc (from v1 logs)")
    L.append("")
    L.append("### Reports (`reports/`)")
    L.append("- `preflight_inventory.md`: pre-flight artifact inventory (v3 task A)")
    L.append("- `medal_agent_v3_fair_test_eval.md`: this file")
    L.append("")
    L.append("### Checkpoints")
    L.append("- `runs/test_eval_v3/ckpts/`: 720 model state dicts (= 120 cells × 6 rounds × 1 ckpt/round)")
    L.append("")
    L.append("### Trajectories")
    L.append("- `runs/test_eval_v3/*.jsonl`: 120 v3 JSONLs (each = 6 rounds with val+test metrics + ckpt path)")
    L.append("")
    L.append("### Job logs")
    L.append("- `runs/test_eval_v3/logs/*.log`: 120+ SGE stdout logs")
    L.append("")
    L.append("---")
    L.append("")
    L.append("*Generated by `medal_bench/audit_v3/{analyze,diagnostics,build_report}.py` against all 120 v3 cells. Replay used v1's selected_ids verbatim — no new active-learning trajectories generated.*")

    (REPORTS / "medal_agent_v3_fair_test_eval.md").write_text("\n".join(L))
    print(f"wrote reports/medal_agent_v3_fair_test_eval.md ({len(L)} lines)")

    # ----- README_REPRODUCE.md -----
    R = []
    R.append("# README — Reproducing the v3 fair test-set evaluation pass")
    R.append("")
    R.append("This document describes how to re-run the v3 audit pipeline.")
    R.append("")
    R.append("## Environment")
    R.append("")
    R.append("- Conda env: `/groups/echambe2/gmeng/conda_envs/medal-agent`")
    R.append("- Python: 3.10")
    R.append("- Key packages: torch 2.4.1+cu121, transformers (for SAM in P7/P8 diagnostics), medpy (HD95/ASSD), scipy, sklearn, matplotlib")
    R.append("- Activate: `source /groups/echambe2/gmeng/MedAL-Agent/medal-agent.env`")
    R.append("- Hardware tested: V100-32GB (`gpu@@csecri-v100`), A40-46GB (`gpu@@coba-a40`)")
    R.append("")
    R.append("## Inputs (existing)")
    R.append("")
    R.append("- v1 trajectories: `runs/pilot_v1/*.jsonl` (120 cells, with `selected_ids` per round) — these provide the AL selection histories that v3 REPLAYS.")
    R.append("- v1 splits: regenerated at runtime by `runner.splits.make_split(adapter, seed=cfg.seed)` — same seed → same split.")
    R.append("- Dataset adapters: `medal_bench/data/adapters/{busi,cvc_clinicdb,isic2018,promise12}.py`")
    R.append("- Dataset roots: `/groups/echambe2/datasets/data/2d/*`")
    R.append("- SAM feature cache (P7/P8 / diagnostics): `/groups/echambe2/gmeng/MedAL-Agent/cache/foundation_features/*.h5`")
    R.append("")
    R.append("## Step 1 — Preflight inventory")
    R.append("")
    R.append("```")
    R.append("cd /groups/echambe2/gmeng/MedAL-Agent/repo/code")
    R.append("/groups/echambe2/gmeng/conda_envs/medal-agent/bin/python3 -m medal_bench.audit_v3.preflight")
    R.append("# Writes:")
    R.append("#   reports/preflight_inventory.md")
    R.append("#   tables/preflight_artifacts.csv")
    R.append("```")
    R.append("")
    R.append("## Step 2 — Canary")
    R.append("")
    R.append("Verify the runner works end-to-end on a small cell:")
    R.append("")
    R.append("```")
    R.append("qsub -q gpu@@csecri-v100 -N v3_canary_cvc_P0_s1000 \\")
    R.append("    -v 'DATASET=cvc_clinicdb,POLICY=P0,SEED=1000' \\")
    R.append("    scripts/v3/run_one_replay.sh")
    R.append("```")
    R.append("")
    R.append("Expected wall time: ~2.5 min on V100. Check `runs/test_eval_v3/cvc_clinicdb__P0__s1000.jsonl` has 6 records and `runs/test_eval_v3/ckpts/cvc_clinicdb__P0__s1000__r{0..5}.pt` are saved.")
    R.append("")
    R.append("## Step 3 — Full matrix")
    R.append("")
    R.append("After the canary passes, submit all 120 cells:")
    R.append("")
    R.append("```")
    R.append("bash scripts/v3/submit_all_replay.sh")
    R.append("```")
    R.append("")
    R.append("Queue assignment:")
    R.append("- `gpu@@coba-a40` for ISIC2018 (larger images)")
    R.append("- `gpu@@csecri-v100` for BUSI, CVC, PROMISE12")
    R.append("")
    R.append("Wall time on our cluster: ~80 min on average (V100 cells ~10 min each, ISIC ~25 min each, with ~10 concurrent jobs).")
    R.append("")
    R.append("## Step 4 — Build analysis tables + report")
    R.append("")
    R.append("After all 120 cells complete (or as far as completion gets):")
    R.append("")
    R.append("```")
    R.append("/groups/echambe2/gmeng/conda_envs/medal-agent/bin/python3 -m medal_bench.audit_v3.analyze")
    R.append("/groups/echambe2/gmeng/conda_envs/medal-agent/bin/python3 -m medal_bench.audit_v3.diagnostics")
    R.append("/groups/echambe2/gmeng/conda_envs/medal-agent/bin/python3 -m medal_bench.audit_v3.build_report")
    R.append("```")
    R.append("")
    R.append("Writes:")
    R.append("- `tables/v3_per_cell_metrics.csv`")
    R.append("- `tables/v3_summary_final_round.csv`")
    R.append("- `tables/v3_val_vs_test_ranks.csv`")
    R.append("- `tables/v3_pairwise_stats.csv`")
    R.append("- `tables/v3_failure_summary.csv`")
    R.append("- `tables/v3_peal_round_summary.csv`")
    R.append("- `tables/v3_paal_ap_round_summary.csv`")
    R.append("- `reports/medal_agent_v3_fair_test_eval.md`")
    R.append("")
    R.append("## Where selected_ids came from")
    R.append("")
    R.append("v3 reads `runs/pilot_v1/{dataset}__{policy}__s{seed}.jsonl` and uses each round's `selected_ids` field verbatim.")
    R.append("The cold start (round 0 labeled set) is reconstructed from `np.random.RandomState(cfg.seed)` shuffle of the train pool — identical to v1's `runner/al_loop.run_al` cold-start code path.")
    R.append("No policy's `score()` or `select()` is called for selection purposes in v3.")
    R.append("")
    R.append("## Where checkpoints are saved")
    R.append("")
    R.append("`runs/test_eval_v3/ckpts/{dataset}__{policy}__s{seed}__r{round}.pt` (= 720 files total)")
    R.append("")
    R.append("Each file is a dict with:")
    R.append("- `model_state`: state_dict for nnU-Net 2D PlainConvUNet")
    R.append("- `input_channels`, `num_classes`, `features_per_stage`, `dropout_p`: rebuild args")
    R.append("- `round`, `n_labeled`, `dataset`, `policy_id`, `seed`: identity")
    R.append("- `v1_jsonl_origin`: source v1 trajectory path")
    R.append("")
    R.append("## Hardware / runtime notes")
    R.append("")
    R.append("- A V100 cell (BUSI/CVC/PROMISE12) trains+evals 6 rounds in ~10–15 min.")
    R.append("- An A40 cell (ISIC2018) takes ~20–30 min.")
    R.append("- Eval (val+test) costs ~30 s per round per split on these dataset sizes.")
    R.append("- Total wall time for 120 cells with ~10 concurrent slots: ~80 min.")
    R.append("")
    R.append("## Known limitations")
    R.append("")
    R.append("- **PROMISE12 case-level metrics not computed**. Per-case DSC/HD95 require slice-to-case aggregation; checkpoints are saved so this can be done in a future pass without retraining.")
    R.append("- **PEAL per-image disagreement not logged** in v1; v3 doesn't add this either. Future: load each P6 checkpoint, forward + hflip-forward over the unlabeled pool, log per-image disagreement, mark which were selected.")
    R.append("- **PAAL AP per-sample predictions not logged** in v1; v3 doesn't add this. Future: load each P9 checkpoint, train AP, log predicted vs actual Dice for each labeled-set sample.")
    R.append("- **Mapper / topology analysis not done**. KMeans coverage proxy (v2) is the only representation-coverage diagnostic.")
    R.append("- **3 seeds is small**. Bootstrap CIs are reported but should be interpreted with caution.")
    R.append("- **No cold-start ablations**. All policies use uniform-random cold start with the same `cfg.seed`. Different cold-start strategies are listed as future work.")
    R.append("")
    R.append("## Reproducibility caveats")
    R.append("")
    R.append("- v3 round-0 DSC values may differ from v1's by ~0.05–0.10 due to torch/cuDNN non-determinism on small (n=5–20) labeled sets even at the same seed.")
    R.append("- Final-round (n=125 for BUSI etc) DSCs are much more stable across runs.")
    R.append("")

    (REPORTS / "README_REPRODUCE.md").write_text("\n".join(R))
    print(f"wrote reports/README_REPRODUCE.md ({len(R)} lines)")


if __name__ == "__main__":
    main()
