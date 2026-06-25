"""Stage S2 -- the agent-facing Query Strategy Skill.

`recommend_query_strategy(state, ...)` returns a ranked recommendation WITH
predictive uncertainty, within-eps probability, collapse risk, expected costs,
human-readable evidence, and a fallback. It is deterministic.

Design follows the benchmark verdict: no learned router reliably beats the safe
default, so the skill is a calibrated *portfolio* selector --
  * blocklist the durably-bad methods (P1/P3/P6/P9) from recommendation;
  * default to BADGE (or Entropy->CoreSet on multi-class) and only deviate when a
    learned utility model is confident BEYOND the cross-seed noise (it rarely is);
  * surface collapse risk and out-of-distribution uncertainty honestly.
The LLM agent consumes this object; it does not invent rankings.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from medal_bench.skill import schema as S

NUM = list(S.STATIC_NUM_COLS) + S.ROUND0_COLS + S.METHOD_FLAG_COLS + ["in_blocklist", "exp_query_cost_z"]
CAT = S.STATIC_CAT_COLS + ["family"]


class QueryStrategySkill:
    def __init__(self, skill_dir: str = S.SKILL_DIR):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        self.df = pd.read_csv(os.path.join(skill_dir, "skill_rows.csv"))
        self.enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(self.df[CAT])
        self.sc = StandardScaler().fit(self.df[NUM].astype(float))
        X = np.hstack([self.sc.transform(self.df[NUM].astype(float)), self.enc.transform(self.df[CAT])])
        self.rf = RandomForestRegressor(n_estimators=400, random_state=0, n_jobs=-1)
        self.rf.fit(X, self.df["aubc_mean"].astype(float).values)
        # per-method training priors
        g = self.df.groupby("method")
        self.collapse_prior = g["collapse_prob"].mean().to_dict()
        self.train_cost = g["train_cost_mean"].mean().to_dict()
        self.query_cost = g["query_cost_mean"].mean().to_dict()
        self.qcost_z = g["exp_query_cost_z"].first().to_dict()
        self.known_modalities = set(self.df["modality"])
        self.known_objects = set(self.df["object_family"])

    # -- per-method (mu, sigma) for a dataset state -------------------------- #
    def _predict(self, state: dict):
        rows = []
        for m in S.METHODS:
            fam, *flags = S.METHOD_DESC[m]
            r = dict(state)
            r.update(family=fam, m_unc=flags[0], m_div=flags[1], m_hyb=flags[2],
                     m_found=flags[3], m_pred=flags[4], m_stoch=flags[5],
                     in_blocklist=int(m in S.BLOCKLIST), exp_query_cost_z=self.qcost_z[m])
            rows.append(r)
        cand = pd.DataFrame(rows)
        X = np.hstack([self.sc.transform(cand[NUM].astype(float)), self.enc.transform(cand[CAT])])
        per_tree = np.stack([t.predict(X) for t in self.rf.estimators_])  # (T, 10)
        mu, sigma = per_tree.mean(0), per_tree.std(0) + 1e-6
        return {m: (float(mu[i]), float(sigma[i])) for i, m in enumerate(S.METHODS)}

    def _collapse_risk(self, m: str, state: dict) -> float:
        risk = float(self.collapse_prior.get(m, 0.0))
        # known P6/P9 failure profile: few cases + tiny foreground (forensic finding)
        if m in ("P6", "P9") and state.get("n_groups", 1e9) < 60 and state.get("fg_frac_mean", 1) < 0.05:
            risk = max(risk, 0.5)
        return round(risk, 4)

    def recommend(self, state: dict, target_budget=None, target_metric: str = "dsc",
                  compute_constraint: str | None = None, risk_tolerance: str = "balanced") -> dict:
        mu_sig = self._predict(state)
        default = "P5" if int(state.get("is_multiclass", 0)) else "P4"  # safe defaults
        ood = (state.get("modality") not in self.known_modalities or
               state.get("object_family") not in self.known_objects)

        # score allowed methods; risk-averse subtracts collapse risk
        def util(m):
            mu, _ = mu_sig[m]
            return mu - (0.1 * self._collapse_risk(m, state) if risk_tolerance == "averse" else 0.0)

        allowed = [m for m in S.ALLOWED]
        if risk_tolerance == "averse":
            allowed = [m for m in allowed if self._collapse_risk(m, state) < 0.34]
        ranked = sorted(allowed, key=util, reverse=True)
        top = ranked[0]

        # do-no-harm clamp: deviate from the default only if the model's edge over the
        # default exceeds the cross-seed noise floor (it rarely does) AND the candidate is
        # no less collapse-safe than the default (so a deviation can never raise collapse
        # risk -- e.g. it blocks the P8-on-busi case where P8 collapses on some seeds).
        edge = mu_sig[top][0] - mu_sig[default][0]
        safe = self._collapse_risk(top, state) <= self._collapse_risk(default, state) + 1e-9
        recommended = top if (edge > S.EPS_AUBC and not ood and safe) else default
        if recommended not in ranked:
            ranked = [recommended] + ranked

        # compute-aware: among methods within eps of the recommended mu, pick cheapest
        if compute_constraint == "low":
            tie = [m for m in ranked if mu_sig[recommended][0] - mu_sig[m][0] <= S.EPS_AUBC]
            recommended = min(tie, key=lambda m: self.query_cost[m])

        mu_b, sig_b = mu_sig[recommended]
        # P(within eps of the best candidate) analytic-ish from ensemble spread
        best_mu = max(mu for mu, _ in mu_sig.values())
        p_within = {m: round(float(_phi((mu_sig[m][0] - (best_mu - S.EPS_AUBC)) / mu_sig[m][1])), 4)
                    for m in S.METHODS}

        evidence = [
            f"default={default} ({'multi-class->Ent+Core' if default=='P5' else 'BADGE'}); "
            f"model top allowed={top} (edge {edge:+.4f} vs default, noise floor {S.EPS_AUBC}).",
            f"blocklisted (never recommended): {S.BLOCKLIST} -- durably worse than Random.",
            ("OUT-OF-DISTRIBUTION: modality/object unseen in the 19-set training pool; "
             "recommendation held at the safe default, treat as low-confidence."
             if ood else "in-distribution relative to the 19-set."),
        ]
        return dict(
            recommended=recommended, fallback_method="P4",
            ranked_methods=ranked,
            expected_utility={m: round(mu_sig[m][0], 4) for m in S.METHODS},
            uncertainty_interval=[round(mu_b - sig_b, 4), round(mu_b + sig_b, 4)],
            probability_within_epsilon=p_within,
            collapse_risk={m: self._collapse_risk(m, state) for m in S.METHODS},
            expected_training_cost={m: round(self.train_cost[m], 1) for m in S.METHODS},
            expected_query_cost={m: round(self.query_cost[m], 2) for m in S.METHODS},
            blocklisted=list(S.BLOCKLIST),
            out_of_distribution=bool(ood),
            evidence=evidence,
        )


def _phi(z):
    from math import erf, sqrt
    return 0.5 * (1 + erf(z / sqrt(2)))


def recommend_query_strategy(state, target_budget=None, target_metric="dsc",
                             compute_constraint=None, risk_tolerance="balanced",
                             skill_dir: str = S.SKILL_DIR):
    """Convenience wrapper that builds the skill and returns one recommendation."""
    return QueryStrategySkill(skill_dir).recommend(
        state, target_budget, target_metric, compute_constraint, risk_tolerance)
