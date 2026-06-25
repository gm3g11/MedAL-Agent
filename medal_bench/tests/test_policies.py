"""Policy unit tests: registry round-trip, valid output, determinism,
no-leakage (labeled + unlabeled-pool mask firewall), BALD components,
Selective Uncertainty target/boundary scoring, PAAL AP+WPS."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from medal_bench.policies import all_ids, build, PolicyContext


ALL_POLICIES = ["P0","P1","P2","P3","P4","P5","P6","P7","P8","P9"]
ABLATION_POLICIES = ["P4b", "P8b", "P8c"]
CORE_PLUS_ABLATION = ALL_POLICIES + ABLATION_POLICIES


# ---- 1. registry round-trip ----

def test_registry_has_core_and_ablations():
    ids = set(all_ids())
    for pid in ALL_POLICIES:
        assert pid in ids, f"missing core {pid} in registry: {sorted(ids)}"
    assert ids == set(CORE_PLUS_ABLATION), \
        f"registry mismatch: {ids ^ set(CORE_PLUS_ABLATION)}"


# ---- 2. valid output shape ----

@pytest.mark.parametrize("pid", CORE_PLUS_ABLATION)
def test_select_returns_k_distinct_indices(pid, make_ctx):
    ctx = make_ctx(seed=1000, round_idx=0, want_features=True, want_foundation=True)
    policy = build(pid)
    k = 3
    scores = policy.score(ctx)
    out = policy.select(ctx, scores, k)
    assert len(out) == k, f"{pid}: expected {k} indices, got {len(out)}"
    assert len(set(out)) == k, f"{pid}: duplicates in {out}"
    n_pool = len(ctx.pool)
    for i in out:
        assert 0 <= i < n_pool, f"{pid}: out-of-range index {i} (pool={n_pool})"


# ---- 3. determinism on same seed ----

@pytest.mark.parametrize("pid", CORE_PLUS_ABLATION)
def test_determinism_same_seed(pid, make_ctx):
    ctx1 = make_ctx(seed=42, round_idx=0, want_features=True, want_foundation=True)
    ctx2 = make_ctx(seed=42, round_idx=0, want_features=True, want_foundation=True)
    p1 = build(pid)
    p2 = build(pid)
    s1 = p1.score(ctx1); s2 = p2.score(ctx2)
    out1 = p1.select(ctx1, s1, k=3)
    out2 = p2.select(ctx2, s2, k=3)
    assert out1 == out2, f"{pid} non-deterministic: {out1} vs {out2}"


# ---- 4. no val/test label access ----
# All policies receive only pool + labeled metadata, never val/test datasets.
# Smoke test: ctx.features does not contain anything val/test-derived;
# ctx.labeled has masks but most policies must not read them.
#
# Exception: P9 (canonical PAAL) intentionally consumes labeled (image, mask)
# pairs to train its Accuracy Predictor — that is the algorithm's design.
# Reading labeled masks is permitted; the firewall protects val/test only.

_POLICIES_NO_LABELED_MASK = [p for p in CORE_PLUS_ABLATION if p != "P9"]

@pytest.mark.parametrize("pid", _POLICIES_NO_LABELED_MASK)
def test_firewall_no_labeled_mask_access(pid, make_ctx, monkeypatch):
    ctx = make_ctx(seed=1000, round_idx=0, want_features=True, want_foundation=True)
    # patch Sample to raise if .mask is read on labeled samples
    orig_getitem = type(ctx.labeled).__getitem__
    def raising_getitem(self, i):
        s = orig_getitem(self, i)
        bad = type(s)(
            sample_id=s.sample_id, image=s.image,
            mask=_RaisingMask("labeled_mask_access"),
            meta=s.meta, patient_id=s.patient_id, slice_index=s.slice_index,
        )
        return bad
    monkeypatch.setattr(type(ctx.labeled), "__getitem__", raising_getitem)
    policy = build(pid)
    s = policy.score(ctx)
    policy.select(ctx, s, k=3)


class _RaisingMask:
    def __init__(self, msg): self._msg = msg
    def __getattr__(self, n):
        raise AssertionError(f"firewall violation: {self._msg}")


# ---- 5. BALD components (T=1 + dropout disabled) ----

def test_bald_components_T1_no_dropout(tiny_model, make_ctx):
    # disable all dropout
    for m in tiny_model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.p = 0.0
    ctx = make_ctx(seed=1000, round_idx=0)
    bald = build("P2", T=1)
    _ = bald.score(ctx)
    pred_ent = ctx.diagnostics_out["bald_predictive_entropy"]
    mean_pp_ent = ctx.diagnostics_out["bald_mean_per_pass_entropy"]
    # at T=1 + dropout disabled, the two components must equal
    assert torch.allclose(pred_ent, mean_pp_ent, atol=1e-5), \
        "BALD components T=1+no-dropout: predictive_entropy must == mean_per_pass_entropy"
    # and their difference (the MI term) must be zero
    mi = pred_ent - mean_pp_ent
    assert torch.allclose(mi, torch.zeros_like(mi), atol=1e-6), \
        f"MI term not zero: max|.| = {mi.abs().max()}"


# ---- 6. Selective Uncertainty (P6): target/boundary diagnostics + behavior ----

def test_p6_selective_uncertainty_diagnostics(make_ctx):
    ctx = make_ctx(seed=1000, round_idx=0)
    pol = build("P6")
    _ = pol.score(ctx)
    out = ctx.diagnostics_out
    for key in ("selu_target_frac", "selu_boundary_frac",
                "selu_score_target_mean", "selu_score_boundary_mean"):
        assert key in out, f"Selective Uncertainty missing diagnostic {key}"
    assert 0.0 <= out["selu_target_frac"] <= 1.0
    assert 0.0 <= out["selu_boundary_frac"] <= 1.0


def test_p6_selective_differs_from_naive_full_entropy():
    """Selective Uncertainty must rank a small-foreground-uncertain image ABOVE
    a diffusely-mild image, whereas naive mean-over-all-pixels entropy ranks
    them the other way. (arXiv:2401.16298 core motivation.)"""
    from medal_bench.runner.prediction_cache import PredictionCache
    from medal_bench.policies._helpers import score_per_pixel, aggregate

    H = W = 8
    # Image A: 2x2 max-entropy foreground block, confident background elsewhere.
    pA = torch.full((2, H, W), 0.0)
    pA[0] = 0.99; pA[1] = 0.01                      # confident background
    pA[0, :2, :2] = 0.5; pA[1, :2, :2] = 0.5        # uncertain fg block
    # Image B: uniform mild foreground probability everywhere (no boundary).
    pB = torch.empty((2, H, W)); pB[0] = 0.85; pB[1] = 0.15
    probs = torch.stack([pA, pB], dim=0)            # (2, C=2, H, W)
    argmax = torch.argmax(probs, dim=1)
    cache = PredictionCache(probs=probs, argmax=argmax, fnames=["A", "B"])
    ctx = PolicyContext(seed=0, round_idx=0, pred_cache=cache, num_classes=2)

    pol = build("P6")
    sel = pol.score(ctx).cpu().numpy()
    naive = aggregate(score_per_pixel(probs, "entropy"), argmax, "full").cpu().numpy()

    assert sel[0] > sel[1], f"selective should prefer A (small uncertain fg): {sel}"
    assert naive[1] > naive[0], f"naive full-mean entropy should prefer B: {naive}"
    # and the selection order must agree with the selective score (A before B)
    assert pol.select(ctx, None, k=2)[0] == 0


# ---- 6b. Unlabeled-pool mask firewall (no unlabeled GT at query time) ----
# This is the leakage-critical guard: a policy must NEVER read ground-truth
# masks of UNLABELED pool samples. Covers required tests #9 (P6) and #12 (P9).

class _PoolMaskFirewall:
    """Wrap the unlabeled pool so any read of an unlabeled sample's .mask
    raises. Wraps only the pool instance (NOT labeled), so P9 can still read
    LABELED masks for its Accuracy Predictor while unlabeled GT stays sealed."""
    def __init__(self, base): self._b = base
    def __len__(self): return len(self._b)
    def sample_ids(self): return self._b.sample_ids()
    def patient_ids(self):
        return self._b.patient_ids() if hasattr(self._b, "patient_ids") else None
    def __getitem__(self, i):
        s = self._b[i]
        return type(s)(
            sample_id=s.sample_id, image=s.image,
            mask=_RaisingMask("unlabeled_pool_mask_access"),
            meta=s.meta, patient_id=s.patient_id, slice_index=s.slice_index,
        )


@pytest.mark.parametrize("pid", CORE_PLUS_ABLATION)
def test_firewall_no_unlabeled_pool_mask_access(pid, make_ctx):
    import dataclasses
    ctx = make_ctx(seed=1000, round_idx=0, want_features=True, want_foundation=True)
    ctx = dataclasses.replace(ctx, pool=_PoolMaskFirewall(ctx.pool))
    policy = build(pid, ap_epochs=2, ap_batch_size=2) if pid == "P9" else build(pid)
    s = policy.score(ctx)
    policy.select(ctx, s, k=3)


# ---- 7. PAAL (P9) AP trains and emits AP / WPS diagnostics ----

def test_p9_paal_diagnostics(make_ctx):
    ctx = make_ctx(seed=1000, round_idx=0, want_features=True, want_foundation=False)
    # keep ap_epochs tiny so the test stays fast
    pol = build("P9", ap_epochs=2, ap_batch_size=2)
    scores = pol.score(ctx)
    out = ctx.diagnostics_out
    for key in ("paal_ap_epochs", "paal_ap_loss_mean", "paal_ap_loss_last",
                "paal_pred_acc_mean", "paal_pred_acc_std",
                "paal_score_mean", "paal_score_std"):
        assert key in out, f"PAAL missing diagnostic {key}"
    assert 0.0 <= out["paal_pred_acc_mean"] <= 1.0, \
        f"predicted accuracy outside [0,1]: {out['paal_pred_acc_mean']}"
    # exercise WPS too
    sel = pol.select(ctx, scores, k=3)
    assert "paal_n_clusters" in out, "WPS must record paal_n_clusters"
    assert "paal_cluster_sizes" in out
    assert "paal_selected_clusters" in out
    assert len(sel) == 3 and len(set(sel)) == 3
