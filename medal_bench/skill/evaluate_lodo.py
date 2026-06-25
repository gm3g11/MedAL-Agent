"""Stage S2 -- leave-one-dataset-out evaluation of Query-Strategy selectors.

Every selector maps a held-out dataset's 10 candidate method-rows (FEATURES ONLY)
to a chosen method; it is realized against that method's true seed-aggregated AUBC.
Models are fit on the 18 training datasets only (encoder + scaler fit on train),
so the held-out dataset is never seen. The headline question: does any learned
router beat always-BADGE (realized AUBC 0.6795) by more than the ~0.016 cross-seed
noise? Acceptance for THIS project is do-no-harm (blocklist) + collapse avoidance.

Run:  python -m medal_bench.skill.evaluate_lodo
Writes: reports/query_skill_lodo_results.md , runs/frozen_v5/skill/lodo_results.json
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd

from medal_bench.skill import schema as S
from medal_bench.skill import splits

warnings.filterwarnings("ignore")
RNG = np.random.RandomState(0)

NUM = [c for c in S.STATIC_NUM_COLS] + S.ROUND0_COLS + S.METHOD_FLAG_COLS + \
      ["in_blocklist", "exp_query_cost_z"]
CAT = S.STATIC_CAT_COLS + ["family"]


# --------------------------------------------------------------------------- #
#  feature matrix (encoder/scaler fit on TRAIN only -- no held-out leakage)
# --------------------------------------------------------------------------- #
def _design(train_df, test_df):
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc.fit(train_df[CAT])
    sc = StandardScaler().fit(train_df[NUM].astype(float))

    def tx(df):
        return np.hstack([sc.transform(df[NUM].astype(float)), enc.transform(df[CAT])])
    return tx(train_df), tx(test_df)


# --------------------------------------------------------------------------- #
#  learned utility regressors  (predict aubc_mean; pick argmax over candidates)
# --------------------------------------------------------------------------- #
def _make_models():
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    from sklearn.linear_model import ElasticNet, Ridge
    models = {
        "ridge": Ridge(alpha=1.0, random_state=0),
        "elasticnet": ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=0),
        "rf": RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1),
        "extratrees": ExtraTreesRegressor(n_estimators=300, random_state=0, n_jobs=-1),
        "gp": GaussianProcessRegressor(kernel=RBF() + WhiteKernel(), alpha=1e-3,
                                       normalize_y=True, random_state=0),
    }
    try:
        from catboost import CatBoostRegressor
        models["catboost"] = CatBoostRegressor(iterations=400, depth=4, learning_rate=0.05,
                                               loss_function="RMSE", random_seed=0, verbose=False)
    except Exception:
        pass
    return models


def _run_learned(df, model_factory, target="aubc_mean"):
    """LODO: per fold fit on train, predict utility for held-out's 10 rows, pick argmax.
    Returns picks {dataset: method}."""
    picks = {}
    for train_ds, held in splits.lodo_folds():
        tr = df[df.dataset.isin(train_ds)]
        te = df[df.dataset == held]
        Xtr, Xte = _design(tr, te)
        model = model_factory()
        model.fit(Xtr, tr[target].astype(float).values)
        pred = model.predict(Xte)
        picks[held] = te.iloc[int(np.argmax(pred))].method
    return picks


def _run_thompson(df, n_samples=200, target="aubc_mean"):
    """Bootstrap-ensemble (RF) -> per-candidate (mu, sigma); Thompson pick averaged
    over samples. Returns (picks, calib_records)."""
    from sklearn.ensemble import RandomForestRegressor
    picks_count = {}
    calib = []
    rs = np.random.RandomState(0)
    for train_ds, held in splits.lodo_folds():
        tr = df[df.dataset.isin(train_ds)]
        te = df[df.dataset == held].reset_index(drop=True)
        Xtr, Xte = _design(tr, te)
        y = tr[target].astype(float).values
        rf = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1).fit(Xtr, y)
        per_tree = np.stack([t.predict(Xte) for t in rf.estimators_])  # (T, 10)
        mu, sigma = per_tree.mean(0), per_tree.std(0) + 1e-6
        draws = rs.normal(mu, sigma, size=(n_samples, len(te)))   # (T, 10)
        votes = np.bincount(draws.argmax(1), minlength=len(te))
        picks_count[held] = te.iloc[int(np.argmax(votes))].method
        # predicted P(method within eps of the best) from the same ensemble draws
        within = (draws.max(1, keepdims=True) - draws) <= S.EPS_AUBC   # (T, 10)
        p_within = within.mean(0)
        for i in range(len(te)):
            calib.append((float(p_within[i]), int(te.iloc[i].regret <= S.EPS_AUBC)))
    return picks_count, calib


# --------------------------------------------------------------------------- #
#  baselines / heuristics  (also LODO where they use training statistics)
# --------------------------------------------------------------------------- #
def _baselines(df):
    picks = {}  # name -> {dataset: method}
    am = {(r.dataset, r.method): r.aubc_mean for r in df.itertuples()}

    def best_by(train_ds, sub=S.METHODS):
        means = {m: np.mean([am[(d, m)] for d in train_ds]) for m in sub}
        return max(means, key=means.get)

    fam_of = {r.method: r.family for r in df.itertuples()}
    picks["always_BADGE"] = {d: "P4" for d in S.DS19}
    picks["random_method"] = {d: None for d in S.DS19}  # handled specially (expectation)

    gb, gbf, pmod, pobj, hand, block, near = ({} for _ in range(7))
    feat_static = df.drop_duplicates("dataset").set_index("dataset")
    for train_ds, held in splits.lodo_folds():
        gb[held] = best_by(train_ds)
        # best family then its best member
        fam_mean = {}
        for f in set(fam_of.values()):
            ms = [m for m in S.METHODS if fam_of[m] == f]
            fam_mean[f] = np.mean([am[(d, m)] for d in train_ds for m in ms])
        bestfam = max(fam_mean, key=fam_mean.get)
        gbf[held] = best_by(train_ds, [m for m in S.METHODS if fam_of[m] == bestfam])
        block[held] = best_by(train_ds, S.ALLOWED)
        # per-modality / per-object best (fallback to global best if unseen)
        held_mod = feat_static.loc[held, "modality"]
        held_obj = feat_static.loc[held, "object_family"]
        same_mod = [d for d in train_ds if feat_static.loc[d, "modality"] == held_mod]
        same_obj = [d for d in train_ds if feat_static.loc[d, "object_family"] == held_obj]
        pmod[held] = best_by(same_mod) if same_mod else best_by(train_ds)
        pobj[held] = best_by(same_obj) if same_obj else best_by(train_ds)
        # hand phase rule (from descriptive analysis), never blocklisted
        mc = feat_static.loc[held, "is_multiclass"]
        hand[held] = "P5" if mc else "P8"
        # nearest training dataset by standardized static features -> its own best method
        cols = ["n_classes", "is_3d", "fg_frac_mean", "rarest_class_frac",
                "class_imbalance", "pool_N", "r0_dsc"]
        Z = feat_static[cols].astype(float)
        z = (Z - Z.mean()) / (Z.std() + 1e-9)
        dists = {d: float(np.linalg.norm(z.loc[held] - z.loc[d])) for d in train_ds}
        nn = min(dists, key=dists.get)
        near[held] = max(S.METHODS, key=lambda m: am[(nn, m)])
    picks.update(global_best_fixed=gb, global_best_family=gbf, per_modality=pmod,
                 per_object_family=pobj, hand_phase_rule=hand,
                 blocklist_global_best=block, nearest_dataset=near)
    return picks


# --------------------------------------------------------------------------- #
def _score(df, picks_by_name):
    """Realized metrics per selector over the 19 LODO folds."""
    am = {(r.dataset, r.method): r.aubc_mean for r in df.itertuples()}
    reg = {(r.dataset, r.method): r.regret for r in df.itertuples()}
    coll = {(r.dataset, r.method): r.collapse_prob for r in df.itertuples()}
    best_m = {d: min((m for m in S.METHODS), key=lambda m: reg[(d, m)]) for d in S.DS19}
    out = {}
    for name, picks in picks_by_name.items():
        if name == "random_method":
            realized = [np.mean([am[(d, m)] for m in S.METHODS]) for d in S.DS19]
            regs = [np.mean([reg[(d, m)] for m in S.METHODS]) for d in S.DS19]
            colls = [np.mean([coll[(d, m)] for m in S.METHODS]) for d in S.DS19]
            top1 = np.nan
            blk = len(S.BLOCKLIST) / len(S.METHODS)
        else:
            realized = [am[(d, picks[d])] for d in S.DS19]
            regs = [reg[(d, picks[d])] for d in S.DS19]
            colls = [coll[(d, picks[d])] for d in S.DS19]
            top1 = np.mean([int(picks[d] == best_m[d]) for d in S.DS19])
            blk = np.mean([int(picks[d] in S.BLOCKLIST) for d in S.DS19])
        out[name] = dict(
            realized_aubc=float(np.mean(realized)),
            mean_regret=float(np.mean(regs)), median_regret=float(np.median(regs)),
            worst_regret=float(np.max(regs)),
            within_eps_rate=float(np.mean([r <= S.EPS_AUBC for r in regs])),
            top1_acc=float(top1), collapse_selected=float(np.mean(colls)),
            blocklist_selected=float(blk),
            _regrets=regs,
        )
    return out


def _strata(df):
    """name -> set(datasets). Report separately; never report only the global mean."""
    st = df.drop_duplicates("dataset").set_index("dataset")
    mri = {"mri", "cardiac_mri", "lge_mri", "t1ce", "multi_parametric_mri"}
    photo = {"dermoscopy", "fundus", "endoscopy", "ultrasound", "histology"}
    am = {(r.dataset, r.method): r.aubc_mean for r in df.itertuples()}
    oracle = {d: max(am[(d, m)] for m in S.METHODS) for d in S.DS19}
    g = lambda cond: {d for d in S.DS19 if cond(d)}
    return {
        "2D (is_3d=0)": g(lambda d: st.loc[d, "is_3d"] == 0),
        "3D-as-slice": g(lambda d: st.loc[d, "is_3d"] == 1),
        "binary": g(lambda d: st.loc[d, "is_multiclass"] == 0),
        "multiclass": g(lambda d: st.loc[d, "is_multiclass"] == 1),
        "CT": g(lambda d: st.loc[d, "modality"] == "ct"),
        "MRI-family": g(lambda d: st.loc[d, "modality"] in mri),
        "photographic": g(lambda d: st.loc[d, "modality"] in photo),
        "small-pool(<500)": g(lambda d: st.loc[d, "pool_N"] < 500),
        "rare-fg(<3%)": g(lambda d: st.loc[d, "fg_frac_mean"] < 0.03),
        "easy(oracle>0.8)": g(lambda d: oracle[d] > 0.8),
        "hard(oracle<0.5)": g(lambda d: oracle[d] < 0.5),
    }


def _stratified(df, picks_by_name, selectors):
    am = {(r.dataset, r.method): r.aubc_mean for r in df.itertuples()}
    oracle = {d: max(am[(d, m)] for m in S.METHODS) for d in S.DS19}
    strata = _strata(df)
    lines = [f"{'stratum':20s} {'n':>2s} " + " ".join(f"{s[:11]:>11s}" for s in selectors) + "   oracleΔ"]
    for sname, dss in strata.items():
        if not dss:
            continue
        cells = []
        badge = np.mean([am[(d, "P4")] for d in dss])
        for sel in selectors:
            picks = picks_by_name[sel]
            val = np.mean([am[(d, picks[d])] for d in dss])
            cells.append(f"{val-badge:+11.4f}" if sel != "always_BADGE" else f"{badge:11.4f}")
        orc = np.mean([oracle[d] for d in dss]) - badge
        lines.append(f"{sname:20s} {len(dss):2d} " + " ".join(cells) + f"  +{orc:.4f}")
    return "\n".join(lines)


def _oracles(df, cells_path):
    """Cheat ceilings: in-sample per-dataset oracle and the HONEST out-of-seed oracle
    (pick on 2 seeds, score on the 3rd)."""
    am = {(r.dataset, r.method): r.aubc_mean for r in df.itertuples()}
    per_ds = float(np.mean([max(am[(d, m)] for m in S.METHODS) for d in S.DS19]))
    cells = pd.read_csv(cells_path)
    aubc_s = {(r.dataset, r.method, int(r.seed)): r.aubc for r in cells.itertuples()}
    oos = []
    for d in S.DS19:
        for tr_s, te_s in splits.out_of_seed_folds():
            pick = max(S.METHODS, key=lambda m: np.mean([aubc_s[(d, m, s)] for s in tr_s]))
            oos.append(aubc_s[(d, pick, te_s)])
    return dict(oracle_per_dataset_insample=per_ds, oracle_out_of_seed=float(np.mean(oos)))


def evaluate(skill_dir: str = S.SKILL_DIR):
    df = pd.read_csv(os.path.join(skill_dir, "skill_rows.csv"))
    picks = _baselines(df)
    for name, mdl in _make_models().items():
        picks[f"learned_{name}"] = _run_learned(df, (lambda m=name, M=_make_models: M()[m]))
    th_picks, calib = _run_thompson(df)
    picks["learned_thompson"] = th_picks

    scores = _score(df, picks)
    oracles = _oracles(df, os.path.join(skill_dir, "cells_raw.csv"))
    badge = scores["always_BADGE"]

    # paired Wilcoxon: each learned/heuristic selector's per-fold regret vs BADGE
    from scipy.stats import wilcoxon
    for name, s in scores.items():
        if name in ("always_BADGE", "random_method"):
            s["wilcoxon_vs_badge_p"] = np.nan
            continue
        try:
            d = np.array(s["_regrets"]) - np.array(badge["_regrets"])
            s["wilcoxon_vs_badge_p"] = float(wilcoxon(d)[1]) if np.any(d) else 1.0
        except Exception:
            s["wilcoxon_vs_badge_p"] = np.nan

    # calibration (Brier) of the Thompson within-eps probabilities
    from medal_bench.skill.calibration import brier, ece
    cp = np.array([p for p, _ in calib]); cy = np.array([y for _, y in calib])
    calib_metrics = dict(brier=brier(cp, cy), ece=ece(cp, cy), n=len(calib))

    strat_selectors = ["always_BADGE", "hand_phase_rule", "blocklist_global_best",
                       "learned_catboost", "learned_rf"]
    strat_selectors = [s for s in strat_selectors if s in picks]
    strat_txt = _stratified(df, picks, strat_selectors)
    _report(scores, oracles, calib_metrics, skill_dir, strat_txt)
    return scores, oracles, calib_metrics


def _report(scores, oracles, calib, skill_dir, strat_txt=""):
    order = ["random_method", "always_BADGE", "global_best_fixed", "global_best_family",
             "per_modality", "per_object_family", "hand_phase_rule", "nearest_dataset",
             "blocklist_global_best", "learned_ridge", "learned_elasticnet", "learned_rf",
             "learned_extratrees", "learned_gp", "learned_catboost", "learned_thompson"]
    order = [o for o in order if o in scores]
    badge = scores["always_BADGE"]["realized_aubc"]
    hdr = f"{'selector':22s} {'realAUBC':>9s} {'dAUBCvsBADGE':>12s} {'meanReg':>8s} {'worstReg':>8s} {'top1':>5s} {'blklst%':>7s} {'collapse':>8s} {'wilcoxP':>8s}"
    lines = [hdr, "-" * len(hdr)]
    for name in order:
        s = scores[name]
        lines.append(f"{name:22s} {s['realized_aubc']:9.4f} {s['realized_aubc']-badge:+12.4f} "
                     f"{s['mean_regret']:8.4f} {s['worst_regret']:8.4f} "
                     f"{(s['top1_acc'] if not np.isnan(s['top1_acc']) else 0):5.2f} "
                     f"{s['blocklist_selected']*100:6.0f}% {s['collapse_selected']:8.4f} "
                     f"{(s['wilcoxon_vs_badge_p'] if not np.isnan(s['wilcoxon_vs_badge_p']) else 1):8.3f}")
    lines += ["", f"CEILINGS  per-dataset oracle (in-sample): {oracles['oracle_per_dataset_insample']:.4f}"
                  f"  | honest out-of-seed oracle: {oracles['oracle_out_of_seed']:.4f}"
                  f"  (+{oracles['oracle_out_of_seed']-badge:.4f} vs BADGE)",
              f"CALIBRATION (Thompson within-eps): Brier={calib['brier']:.4f} ECE={calib['ece']:.4f} n={calib['n']}",
              f"NOISE FLOOR cross-seed AUBC std ~ 0.016 -> any dAUBC below this is not real.",
              "", "## Stratified realized-AUBC delta vs always-BADGE (no stratum should hide a win)",
              strat_txt]
    txt = "\n".join(lines)
    print(txt)
    os.makedirs("reports", exist_ok=True)
    with open("reports/query_skill_lodo_results.md", "w") as fh:
        fh.write("# Query Strategy Skill -- LODO results (frozen_v5 19-set)\n\n```\n" + txt + "\n```\n")
    clean = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in scores.items()}
    json.dump(dict(scores=clean, oracles=oracles, calibration=calib),
              open(os.path.join(skill_dir, "lodo_results.json"), "w"), indent=2)
    print(f"\nwrote reports/query_skill_lodo_results.md + {skill_dir}/lodo_results.json")


if __name__ == "__main__":
    evaluate()
