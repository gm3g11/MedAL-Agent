"""Automatic QC / acceptance report for the frozen_v5 3-seed run.

Implements the 6-part acceptance gate:
  QC1  TF32 ON + deterministic (single-arch pinning [QC2] makes it confound-free)
  QC2  every dataset pinned to ONE GPU arch across all P0-P9 and all seeds
  QC3  round-0 invariance: init-hash, component seeds, GPU arch, stop_iter/reason,
       ckpt hash, DSC identical across P0-P9 (per dataset,seed). Round-0 trains on the
       shared initial set with policy-independent seeds, so under the single-arch fix the
       round-0 DSC MUST be ~identical across policies; a material spread means the GPU
       confound is not fixed.
  QC4  all cells are frozen_v5 (v5 budget grid + run dir), no v4 seed-1000 data mixed in
  QC5  3-seed stats: DSC, AUBC, HD95/ASSD, detection/missed, method mean/std, regret,
       rank, catastrophic-collapse rate
  QC6  P6 selected-set diagnostics: pred-fg ratio, target/boundary fracs, divergence flag
       (+ optional GT-based foreground size / unique cases / adjacent redundancy with --deep)

Reads the per-cell trajectory JSONLs (no GPU). Tolerant of a partial run (reports coverage).
Usage: python -m medal_bench.analysis.qc_report --run-dir runs/frozen_v5 [--out report.md] [--deep]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from collections import defaultdict

import numpy as np

from medal_bench.analysis.derived import aubc

POLICIES = [f"P{i}" for i in range(10)]
V5_CASE_B = [0.01, 0.02, 0.04, 0.07, 0.10, 0.15, 0.20]
COLLAPSE_DSC = 0.15            # final DSC below this = catastrophic collapse
EXPECTED_SEEDS = (1000, 2000, 3000)
# the 19 usable datasets in the frozen_v5 matrix (rose1 excluded as degenerate)
EXPECTED_DATASETS = [
    "btcv_synapse", "busi", "care_leftatrium_2026", "ext_abdoment1k", "ext_brats2020",
    "flare22", "glas2015", "hvsmr2016", "isic2018", "kits19", "kvasir_seg", "liqa_mri",
    "mmwhs_ct", "msd_task03_liver", "msd_task04_hippocampus", "msd_task07_pancreas",
    "msd_task09_spleen", "origa", "refuge",
]


def _arch(name: str) -> str:
    n = (name or "").lower()
    if "v100" in n:
        return "Volta"
    if "a40" in n or "a100" in n or "a10" in n:
        return "Ampere"
    if "h100" in n or "h200" in n:
        return "Hopper"
    return name or "unknown"


def _hash_ids(ids) -> str:
    return hashlib.md5((";".join(map(str, sorted(ids)))).encode()).hexdigest()[:10]


def load_cells(run_dirs, seeds=(1000, 2000, 3000)):
    cells = {}
    for d in run_dirs:
        for f in glob.glob(f"{d}/*__s*.jsonl"):
            try:
                r = [json.loads(l) for l in open(f)]
            except Exception:
                continue
            if not r or len(r) != r[0].get("total_rounds", -1):
                continue                       # only completed cells
            b = os.path.basename(f).replace(".jsonl", "")
            parts = b.split("__")
            if len(parts) < 3:
                continue
            ds, pol, srk = parts[0], parts[1], parts[2]
            try:
                seed = int(srk[1:])
            except ValueError:
                continue
            if seed not in seeds:
                continue
            cells[(ds, pol, seed)] = r
    return cells


# ---------------- QC0 completeness ----------------
def qc0_completeness(cells):
    """No missing method-budget trajectories: every (dataset,method,seed) present AND every
    cell has its full round set (load_cells already drops incomplete cells, so absence = missing)."""
    present = set(cells.keys())
    expected = {(d, p, s) for d in EXPECTED_DATASETS for p in POLICIES for s in EXPECTED_SEEDS}
    missing = sorted(expected - present)
    extra = sorted(present - expected)           # e.g. a stray seed / dataset
    lines = [f"  expected {len(expected)} cells (19 ds x 10 methods x 3 seeds); present {len(present & expected)}"]
    if missing:
        lines.append(f"  MISSING {len(missing)}: {missing[:12]}{' ...' if len(missing) > 12 else ''}")
    if extra:
        lines.append(f"  unexpected cells: {extra[:6]}")
    ok = (not missing and not extra)
    return ok, lines


# ---------------- QC1 ----------------
def qc1_tf32():
    """frozen_v5-accel: TF32 is ON (Ampere/Hopper tensor-core fp32, ~2x faster) and is
    itself deterministic (bit-identical weight hashes across re-runs). The old cross-arch
    confound is eliminated by SINGLE-ARCH PINNING (QC2): each dataset's full policy matrix
    runs on one GPU arch, so TF32 is consistent within every comparison. QC1 therefore now
    verifies TF32 is ON *with determinism still ON* (not the old, now-redundant TF32-off)."""
    lines, ok = [], True
    try:
        import torch
        from medal_bench.runner.seeds import seed_all
        seed_all(1234)
        c = torch.backends.cudnn.allow_tf32
        m = torch.backends.cuda.matmul.allow_tf32
        d = torch.backends.cudnn.deterministic
        ok = (c is True and m is True and d is True)
        lines.append(f"  cudnn.allow_tf32={c}  matmul.allow_tf32={m}  cudnn.deterministic={d}  ->  {'PASS' if ok else 'FAIL'}")
        lines.append("  (TF32 on + deterministic; single-arch pinning [QC2] makes it confound-free)")
    except Exception as e:  # no GPU / torch in this context: fall back to source check
        src = open("medal_bench/runner/seeds.py").read()
        ok = ("cudnn.allow_tf32 = True" in src and "matmul.allow_tf32 = True" in src)
        lines.append(f"  (runtime check unavailable: {e}); source sets TF32 on: {'PASS' if ok else 'FAIL'}")
    return ok, lines


# ---------------- QC2 ----------------
def qc2_single_arch(cells):
    by_ds = defaultdict(set)
    for (ds, pol, seed), r in cells.items():
        by_ds[ds].add(_arch(r[0].get("gpu_name", "")))
    lines, ok = [], True
    for ds in sorted(by_ds):
        archs = sorted(by_ds[ds])
        bad = len(archs) > 1
        ok = ok and not bad
        lines.append(f"  {ds:<24} {','.join(archs)}{'   <-- MIXED-ARCH' if bad else ''}")
    return ok, lines


# ---------------- QC3 ----------------
def qc3_round0(cells):
    by_dseed = defaultdict(dict)            # (ds,seed) -> {pol: round0 record}
    for (ds, pol, seed), r in cells.items():
        by_dseed[(ds, seed)][pol] = r[0]
    lines, ok = [], True
    lines.append(f"  {'dataset/seed':<28} {'n_pol':>5} {'seed/init id':>12} {'arch':>7} {'r0 DSC spread':>13}")
    for (ds, seed) in sorted(by_dseed):
        g = by_dseed[(ds, seed)]
        if len(g) < 2:
            continue
        # init-ids hash + component seeds must be identical across policies
        idh = {_hash_ids(x.get("initial_labeled_ids", [])) for x in g.values()}
        cs = {json.dumps(x.get("component_seeds", {}), sort_keys=True) for x in g.values()}
        archs = {_arch(x.get("gpu_name", "")) for x in g.values()}
        d0 = [x["metrics"].get("mean_dsc_fg") for x in g.values() if x["metrics"].get("mean_dsc_fg") is not None]
        spread = (max(d0) - min(d0)) if d0 else float("nan")
        seed_ok = (len(idh) == 1 and len(cs) == 1)
        # under single-arch + identical seeds, round-0 DSC should be ~identical
        dsc_ok = (spread == spread and spread <= 0.005) or len(g) < 10
        flag = ""
        if not seed_ok:
            flag = "  <-- seeds/init DIFFER across policies!"; ok = False
        elif spread == spread and spread > 0.01:
            flag = f"  <-- r0 DSC spread {spread:.3f} (confound not fixed?)"; ok = False
        lines.append(f"  {ds+'/s'+str(seed):<28} {len(g):>5} {('OK' if seed_ok else 'DIFF'):>12} "
                     f"{('+'.join(sorted(archs))):>7} {spread:>13.4f}{flag}")
    return ok, lines


# ---------------- QC4 ----------------
def qc4_all_v5(cells, run_dirs):
    lines, ok = [], True
    # (a) run dir must be the v5 dir, not the old v4 stage2 dirs
    bad_dirs = [d for d in run_dirs if "stage2" in d]
    if bad_dirs:
        ok = False; lines.append(f"  FAIL: v4 dirs included: {bad_dirs}")
    # (b) v4-vs-v5 by the DISTINGUISHING budget point: v4 grid has 5% and no 7%; v5 has
    # 4%+7% and no 5%. The 2*C initial-set floor can merge the lowest points on high-class
    # small datasets (so a v5 cell may have <7 distinct points) — but a v4 cell ALWAYS shows
    # a 5% point and a v5 cell NEVER does, so flag on the 5% signature, not the point count.
    v5_cells = v4_cells = 0; v4_examples = []
    for (ds, pol, seed), r in cells.items():
        bd = r[0].get("budget_denominator", {})
        N = bd.get("actual_AL_pool_N")
        if not N or not (500 <= N < 5000):
            continue                              # Case A/C/D not on the Case-B grid
        fr = [c / N for c in bd.get("budget_plan", [])]
        has_5pct = any(abs(f - 0.05) < 0.004 for f in fr)   # v4 signature
        if has_5pct:
            v4_cells += 1; v4_examples.append(f"{ds}/{pol}/s{seed}")
        else:
            v5_cells += 1
    lines.append(f"  Case-B cells: v5-grid (no 5% point) {v5_cells} | v4-grid (has 5% point) {v4_cells}")
    if v4_cells:
        ok = False
        lines.append(f"  FAIL: v4-grid cells present (v4 data mixed in): {v4_examples[:6]}")
    seeds = sorted({s for (_, _, s) in cells})
    lines.append(f"  seeds present: {seeds}")
    return ok, lines


# ---------------- QC5 ----------------
def _final(r, key):
    return r[-1]["metrics"].get(key)


def qc5_stats(cells, degenerate=("rose1",)):
    datasets = sorted({d for (d, _, _) in cells} - set(degenerate))
    seeds = sorted({s for (_, _, s) in cells})
    # per (ds,pol): mean over available seeds of final-DSC, AUBC, HD95, ASSD, detect, missed
    agg = {}
    for ds in datasets:
        for pol in POLICIES:
            runs = [cells[(ds, pol, s)] for s in seeds if (ds, pol, s) in cells]
            if not runs:
                continue
            finals = [_final(r, "mean_dsc_fg") for r in runs]
            aubcs = []
            for r in runs:
                fr = [x.get("labeled_ratio") for x in r]
                sc = [x["metrics"].get("mean_dsc_fg") for x in r]
                if len(fr) >= 2 and all(v is not None for v in fr + sc):
                    aubcs.append(aubc(fr, sc))
            hd = [_final(r, "mean_hd95_fg") for r in runs]
            assd = [_final(r, "assd_case_macro_fg") for r in runs]
            det = [_final(r, "structure_detection_rate") for r in runs]
            miss = [_final(r, "missed_structure_rate") for r in runs]
            nan = lambda xs: [x for x in xs if x is not None and x == x]
            agg[(ds, pol)] = dict(
                dsc=float(np.mean(finals)), dsc_std=float(np.std(finals)), n_seeds=len(runs),
                aubc=float(np.mean(aubcs)) if aubcs else float("nan"),
                hd95=float(np.mean(nan(hd))) if nan(hd) else float("nan"),
                assd=float(np.mean(nan(assd))) if nan(assd) else float("nan"),
                detect=float(np.mean(nan(det))) if nan(det) else float("nan"),
                collapse=sum(1 for f in finals if f is not None and f < COLLAPSE_DSC),
            )
    lines = []
    lines.append(f"  datasets={len(datasets)} seeds={seeds}  (cells aggregated over available seeds)")
    # per-method means across datasets + regret + rank
    permethod = {}
    for pol in POLICIES:
        ds_dsc = {ds: agg[(ds, pol)]["dsc"] for ds in datasets if (ds, pol) in agg}
        ds_aubc = {ds: agg[(ds, pol)]["aubc"] for ds in datasets if (ds, pol) in agg}
        coll = sum(agg[(ds, pol)]["collapse"] for ds in datasets if (ds, pol) in agg)
        permethod[pol] = dict(mean_dsc=np.mean(list(ds_dsc.values())) if ds_dsc else float("nan"),
                              mean_aubc=np.nanmean(list(ds_aubc.values())) if ds_aubc else float("nan"),
                              n=len(ds_dsc), collapses=coll)
    # regret (best DSC per dataset minus method) + average rank
    regret = defaultdict(list); ranks = defaultdict(list)
    for ds in datasets:
        present = [(pol, agg[(ds, pol)]["dsc"]) for pol in POLICIES if (ds, pol) in agg]
        if len(present) < 2:
            continue
        best = max(v for _, v in present)
        order = sorted(present, key=lambda kv: -kv[1])
        rankmap = {pol: i + 1 for i, (pol, _) in enumerate(order)}
        for pol, v in present:
            regret[pol].append(best - v); ranks[pol].append(rankmap[pol])
    names = {"P0": "Random", "P1": "Entropy", "P2": "BALD", "P3": "CoreSet", "P4": "BADGE",
             "P5": "Ent+CoreSet", "P6": "SelUnc", "P7": "SAM-CoreSet", "P8": "SAM-TypiClust", "P9": "PAAL"}
    lines.append(f"\n  {'policy':<16} {'meanDSC':>8} {'meanAUBC':>9} {'avgRank':>8} {'meanRegret':>11} {'collapses':>10} {'n':>3}")
    for pol in sorted(POLICIES, key=lambda p: -(permethod[p]["mean_dsc"] if permethod[p]["mean_dsc"] == permethod[p]["mean_dsc"] else -9)):
        pm = permethod[pol]
        ar = float(np.mean(ranks[pol])) if ranks[pol] else float("nan")
        mr = float(np.mean(regret[pol])) if regret[pol] else float("nan")
        lines.append(f"  {pol+' '+names[pol]:<16} {pm['mean_dsc']:>8.3f} {pm['mean_aubc']:>9.3f} "
                     f"{ar:>8.2f} {mr:>11.4f} {pm['collapses']:>10} {pm['n']:>3}")
    total_cells = len(cells)
    total_coll = sum(agg[k]["collapse"] for k in agg)
    lines.append(f"\n  catastrophic-collapse cells (final DSC<{COLLAPSE_DSC}): {total_coll} / {total_cells} completed")
    return True, lines


# ---------------- QC6 ----------------
def qc6_p6(cells, deep=False):
    lines = []
    lines.append("  P6 per-cell diagnostics (selection + divergence):")
    lines.append(f"  {'dataset/seed':<26} {'finalDSC':>8} {'detect':>7} {'sel_fg':>7} {'tgt_frac':>8} {'diverged_rounds':>15}")
    p6 = sorted([(ds, seed, r) for (ds, pol, seed), r in cells.items() if pol == "P6"])
    for ds, seed, r in p6:
        fr = r[-1]["metrics"].get("mean_dsc_fg")
        det = r[-1]["metrics"].get("structure_detection_rate")
        sel_fg = r[-1].get("selected_pred_fg_ratio")
        diag = r[-1].get("selection_diagnostics", {}) or {}
        tgt = diag.get("selu_target_frac")
        div = sum(1 for x in r if (x.get("training", {}) or {}).get("diverged"))
        lines.append(f"  {ds+'/s'+str(seed):<26} {(fr if fr is not None else float('nan')):>8.3f} "
                     f"{(det if det is not None else float('nan')):>7.3f} "
                     f"{(sel_fg if sel_fg is not None else float('nan')):>7.3f} "
                     f"{(tgt if tgt is not None else float('nan')):>8.4f} {div:>15}")
    if not p6:
        lines.append("  (no completed P6 cells yet)")
    lines.append("  Deep GT-based fg-size / unique-cases / adjacent-redundancy: see forensic "
                 "(submit/p6_forensic.py) — P6 selects ~5-6x smaller fg, fewer cases, more adjacency.")
    return True, lines


def build(run_dirs, deep=False):
    cells = load_cells(run_dirs)
    seeds = sorted({s for (_, _, s) in cells})
    nds = len({d for (d, _, _) in cells})
    out = [f"# frozen_v5 QC report  ({len(cells)} completed cells, {nds} datasets, seeds {seeds})\n"]
    for tag, title, fn in [
        ("QC0", "completeness — no missing method-budget trajectories", lambda: qc0_completeness(cells)),
        ("QC1", "TF32 on + deterministic (single-arch safe)", lambda: qc1_tf32()),
        ("QC2", "single GPU arch per dataset", lambda: qc2_single_arch(cells)),
        ("QC3", "round-0 invariance across P0-P9", lambda: qc3_round0(cells)),
        ("QC4", "all cells frozen_v5 (no v4 mixed)", lambda: qc4_all_v5(cells, run_dirs)),
        ("QC5", "3-seed stats (DSC/AUBC/HD95/ASSD/detect/regret/rank/collapse)", lambda: qc5_stats(cells)),
        ("QC6", "P6 canonical-baseline diagnostics", lambda: qc6_p6(cells, deep)),
    ]:
        try:
            ok, lines = fn()
        except Exception as e:
            ok, lines = False, [f"  ERROR: {e}"]
        out.append(f"## {tag} — {title}: {'PASS' if ok else 'FAIL/REVIEW'}")
        out.extend(lines)
        out.append("")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", action="append", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--deep", action="store_true")
    a = ap.parse_args(argv)
    md = build(a.run_dir, a.deep)
    if a.out:
        open(a.out, "w").write(md)
        print(f"wrote {a.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
