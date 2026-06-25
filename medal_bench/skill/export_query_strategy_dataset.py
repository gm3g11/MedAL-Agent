"""Stage S1 -- export the versioned Query-Strategy-Skill dataset from frozen_v5.

Two tables (grouped by DATASET, the only valid split key):
  * cells_raw.csv  : one row per (dataset, method, seed)   -- 570 rows (audit grain)
  * skill_rows.csv : one row per (dataset, method)         -- 190 rows (LODO modeling grain)

Features are decision-time only:
  Block A static descriptors  (dataset_features.csv, adapter-derived)
  Block B shared round-0 state (consensus over P1-P9, never P0)
  Block D method descriptors   (schema.METHOD_DESC)
Targets (regret PRIMARY) are derived from the post-acquisition trajectory.
Collapse flags come from OBSERVED outcomes only.

Run:  python -m medal_bench.skill.export_query_strategy_dataset
"""
from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np

from medal_bench.skill import schema as S


# ---------------------------------------------------------------------------
def _load_cells(runs_dir: str) -> dict:
    """{(ds,m,seed): record-list} for the 19-set, dropping .partial files."""
    out = {}
    for path in sorted(glob.glob(os.path.join(runs_dir, "*.jsonl"))):
        b = os.path.basename(path)
        if not b.endswith(".jsonl") or ".partial" in b:
            continue
        name = b[:-6]
        try:
            ds, rest = name.split("__P", 1)
            m = "P" + rest.split("__s")[0]
            seed = int(rest.split("__s")[1])
        except (ValueError, IndexError):
            continue
        if ds not in S.DS19 or m not in S.METHODS or seed not in S.SEEDS:
            continue
        rows = [json.loads(l) for l in open(path)]
        if rows:
            out[(ds, m, seed)] = rows
    return out


def _aubc(rows) -> float:
    fr = np.array([r["labeled_ratio"] for r in rows], float)
    sc = np.array([r["metrics"]["mean_dsc_fg_case_macro"] for r in rows], float)
    order = np.argsort(fr)
    fr, sc = fr[order], sc[order]
    span = fr[-1] - fr[0]
    return float(np.trapz(sc, fr) / span) if span > 0 else float(sc.mean())


def _runtime(r) -> dict:
    v = r.get("runtime_sec", 0) or 0
    if isinstance(v, dict):
        return {k: float(v.get(k, 0) or 0) for k in ("train", "eval", "select")}
    return {"train": float(v), "eval": 0.0, "select": 0.0}


def _collapse_flags(rows, random_final: float | None) -> dict:
    dsc = [r["metrics"]["mean_dsc_fg_case_macro"] for r in rows]
    final = dsc[-1]
    # running-peak-to-later max drop (mid-trajectory instability)
    peak = dsc[0]
    max_drop = 0.0
    for y in dsc[1:]:
        peak = max(peak, y)
        max_drop = max(max_drop, peak - y)
    abs_c = int(final < S.COLLAPSE_ABS)
    rel_c = int(random_final is not None and final < random_final - S.COLLAPSE_REL_GAP)
    inst_c = int(max_drop > S.COLLAPSE_DROP)
    return dict(c_abs=abs_c, c_rel=rel_c, c_instab=inst_c,
                max_mid_drop=round(float(max_drop), 5),
                is_collapse=int(abs_c or rel_c or inst_c))


def _round0_consensus(cells, ds, seed) -> dict:
    """Shared round-0 state from the P1-P9 consensus (median across them)."""
    recs = []
    for m in S.ROUND0_CONSENSUS_METHODS:
        r = cells.get((ds, m, seed))
        if r:
            recs.append(r[0])
    if not recs:
        return {}
    def med(fn):
        vals = [fn(r) for r in recs if fn(r) is not None]
        return float(np.median(vals)) if vals else None
    def fg_classes(r):
        dpc = (r.get("metrics") or {}).get("dsc_per_class") or []
        return dpc[1:] if len(dpc) >= 2 else dpc
    return dict(
        r0_dsc=round(med(lambda r: r["metrics"]["mean_dsc_fg_case_macro"]) or 0.0, 6),
        r0_detection_rate=round(med(lambda r: r["metrics"].get("structure_detection_rate")) or 0.0, 6),
        r0_missed_rate=round(med(lambda r: r["metrics"].get("missed_structure_rate")) or 0.0, 6),
        r0_dsc_class_min=round(med(lambda r: float(np.min(fg_classes(r))) if fg_classes(r) else None) or 0.0, 6),
        r0_dsc_class_mean=round(med(lambda r: float(np.mean(fg_classes(r))) if fg_classes(r) else None) or 0.0, 6),
        r0_labeled_frac=round(med(lambda r: r.get("labeled_ratio")) or 0.0, 6),
    )


# ---------------------------------------------------------------------------
def export(runs_dir: str = S.RUNS_DIR, out_dir: str = S.SKILL_DIR) -> tuple[list, list]:
    feat_path = os.path.join(out_dir, "dataset_features.csv")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(
            f"{feat_path} missing -- run `python -m medal_bench.skill.dataset_features` first")
    static = {r["dataset"]: r for r in csv.DictReader(open(feat_path))}

    cells = _load_cells(runs_dir)

    # per (ds,seed) Random final DSC for the relative-collapse reference
    rand_final = {}
    for (ds, m, s), rows in cells.items():
        if m == "P0":
            rand_final[(ds, s)] = rows[-1]["metrics"]["mean_dsc_fg_case_macro"]

    # ---- cells_raw (570) ----
    raw = []
    for (ds, m, seed), rows in sorted(cells.items()):
        last = rows[-1]
        rt = [_runtime(r) for r in rows]
        train_cost = sum(x["train"] for x in rt)
        query_cost = sum(x["select"] for x in rt)
        r0 = _round0_consensus(cells, ds, seed)
        col = _collapse_flags(rows, rand_final.get((ds, seed)))
        st = static[ds]
        fam, *_flags = S.METHOD_DESC[m]
        raw.append(dict(
            dataset=ds, method=m, seed=seed,
            modality=st["modality"], object_family=st["object_family"],
            n_classes=int(st["n_classes"]), is_multiclass=int(st["is_multiclass"]),
            is_3d=int(last.get("dim") == "3d"),
            pool_N=int((last.get("budget_denominator") or {}).get("actual_AL_pool_N", 0)),
            full_train_N=int((last.get("budget_denominator") or {}).get("full_train_N", 0)),
            n_rounds=len(rows),
            aubc=round(_aubc(rows), 6),
            dsc_final=round(last["metrics"]["mean_dsc_fg_case_macro"], 6),
            detection_final=round(last["metrics"].get("structure_detection_rate", float("nan")), 6),
            missed_final=round(last["metrics"].get("missed_structure_rate", float("nan")), 6),
            train_cost=round(train_cost, 1), query_cost=round(query_cost, 2),
            arch=("Hopper" if "H100" in (last.get("gpu_name") or "") else
                  "Ampere" if "A40" in (last.get("gpu_name") or "") or "A100" in (last.get("gpu_name") or "") else
                  "Volta" if "V100" in (last.get("gpu_name") or "") else "?"),
            **col, **r0,
        ))

    # ---- skill_rows (190): seed-aggregate ----
    by_dm = {}
    for r in raw:
        by_dm.setdefault((r["dataset"], r["method"]), []).append(r)
    # per (ds): best aubc_mean over methods, and per-seed best aubc (for p_within_eps)
    aubc_mean = {(ds, m): float(np.mean([x["aubc"] for x in v]))
                 for (ds, m), v in by_dm.items()}
    best_mean = {ds: max(aubc_mean[(ds, m)] for m in S.METHODS) for ds in S.DS19}
    per_seed_aubc = {(r["dataset"], r["method"], r["seed"]): r["aubc"] for r in raw}
    per_seed_best = {(ds, s): max(per_seed_aubc[(ds, m, s)] for m in S.METHODS)
                     for ds in S.DS19 for s in S.SEEDS}
    # method-level expected query cost z-score (compute descriptor)
    m_qcost = {m: float(np.mean([r["query_cost"] for r in raw if r["method"] == m]))
               for m in S.METHODS}
    qc_vals = np.array(list(m_qcost.values()))
    qc_mu, qc_sd = qc_vals.mean(), qc_vals.std() + 1e-9
    qcost_z = {m: round((m_qcost[m] - qc_mu) / qc_sd, 4) for m in S.METHODS}

    rows_out = []
    for ds in S.DS19:
        order = sorted(S.METHODS, key=lambda m: -aubc_mean[(ds, m)])
        rank_of = {m: i + 1 for i, m in enumerate(order)}
        for m in S.METHODS:
            v = by_dm[(ds, m)]
            st = static[ds]
            fam, *flags = S.METHOD_DESC[m]
            am = aubc_mean[(ds, m)]
            regret = best_mean[ds] - am
            p_within = float(np.mean([
                int(per_seed_best[(ds, s)] - per_seed_aubc[(ds, m, s)] <= S.EPS_AUBC)
                for s in S.SEEDS]))
            # seed-mean round-0 features
            r0keys = [k for k in v[0] if k.startswith("r0_")]
            r0mean = {k: round(float(np.mean([x[k] for x in v])), 6) for k in r0keys}
            rows_out.append(dict(
                dataset=ds, method=m,
                # block A static
                modality=st["modality"], object_family=st["object_family"],
                n_classes=int(st["n_classes"]), is_multiclass=int(st["is_multiclass"]),
                is_3d=v[0]["is_3d"], pool_N=v[0]["pool_N"], full_train_N=v[0]["full_train_N"],
                n_groups=int(st["n_groups"]), slices_per_case=float(st["slices_per_case"]),
                fg_frac_mean=float(st["fg_frac_mean"]), fg_frac_median=float(st["fg_frac_median"]),
                rarest_class_frac=float(st["rarest_class_frac"]),
                class_imbalance=float(st["class_imbalance"]),
                img_h=int(st["img_h"]), img_w=int(st["img_w"]), aspect_ratio=float(st["aspect_ratio"]),
                # block D method
                family=fam, m_unc=int(flags[0]), m_div=int(flags[1]), m_hyb=int(flags[2]),
                m_found=int(flags[3]), m_pred=int(flags[4]), m_stoch=int(flags[5]),
                in_blocklist=int(m in S.BLOCKLIST), exp_query_cost_z=qcost_z[m],
                # targets
                aubc_mean=round(am, 6), aubc_std=round(float(np.std([x["aubc"] for x in v])), 6),
                dsc_final_mean=round(float(np.mean([x["dsc_final"] for x in v])), 6),
                regret=round(regret, 6), rank=rank_of[m],
                within_eps=int(regret <= S.EPS_AUBC), p_within_eps=round(p_within, 4),
                collapse_prob=round(float(np.mean([x["is_collapse"] for x in v])), 4),
                train_cost_mean=round(float(np.mean([x["train_cost"] for x in v])), 1),
                query_cost_mean=round(float(np.mean([x["query_cost"] for x in v])), 2),
                **r0mean,
            ))

    os.makedirs(out_dir, exist_ok=True)
    _write(os.path.join(out_dir, "cells_raw.csv"), raw)
    _write(os.path.join(out_dir, "skill_rows.csv"), rows_out)
    meta = dict(version=S.SKILL_DATASET_VERSION, n_cells=len(raw), n_skill_rows=len(rows_out),
                datasets=len(S.DS19), methods=len(S.METHODS), seeds=S.SEEDS,
                eps_aubc=S.EPS_AUBC, blocklist=S.BLOCKLIST,
                collapse=dict(abs=S.COLLAPSE_ABS, rel_gap=S.COLLAPSE_REL_GAP, drop=S.COLLAPSE_DROP))
    json.dump(meta, open(os.path.join(out_dir, "skill_dataset_meta.json"), "w"), indent=2)
    print(f"wrote cells_raw.csv ({len(raw)})  skill_rows.csv ({len(rows_out)})  "
          f"meta.json  -> {out_dir}")
    return raw, rows_out


def _write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    export()
