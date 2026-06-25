"""Section 7/9/10/11 targeted tests: BALD, CoreSet distance-update, PAAL
priority/WPS, SAM model-type config, and no-NaN/Inf scores."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from medal_bench.policies import build, PolicyContext


# ---- P2 BALD --------------------------------------------------------------

def test_bald_dropout_active_bn_eval(tiny_model):
    from medal_bench.models.nnunet import enable_mc_dropout
    n = enable_mc_dropout(tiny_model)
    assert n > 0, "model should have dropout modules"
    for m in tiny_model.modules():
        cls = m.__class__.__name__
        if cls.startswith("Dropout"):
            assert m.training is True, "dropout must be active during MC inference"
        if "Norm" in cls:
            assert m.training is False, "norm layers must stay in eval (running stats frozen)"


def test_bald_positive_when_mc_predictions_disagree(tiny_model, make_ctx):
    for m in tiny_model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.p = 0.9
    ctx = make_ctx(seed=1, round_idx=0)
    s = build("P2", T=10).score(ctx)
    assert s.min() >= -1e-4, f"BALD must be >= 0: min={s.min()}"
    assert float(s.mean()) > 1e-4, "high dropout should produce positive BALD"
    assert ctx.diagnostics_out.get("bald_T") == 10


# ---- P3 CoreSet -----------------------------------------------------------

def test_coreset_distance_updates_after_selection():
    from medal_bench.policies._helpers import kcenter_greedy
    from sklearn.metrics import pairwise_distances
    feats = np.array([[0.0], [1.0], [2.0], [10.0]])
    dm = pairwise_distances(feats)
    # labeled={0}; first pick farthest=10 (idx3); after update, next farthest=2 (idx2)
    assert kcenter_greedy(dm, [0], 2) == [3, 2]


# ---- P9 PAAL --------------------------------------------------------------

def test_paal_wps_cluster_diverse_and_priority():
    # P9 WPS L2-normalizes features, so separate the clusters by ANGLE.
    feats = np.array([[1, 0], [1, 0.05], [1, -0.05],
                      [0, 1], [0.05, 1], [-0.05, 1]], np.float32)
    # higher score == lower predicted accuracy == higher priority
    scores = torch.tensor([0.1, 0.9, 0.2, 0.3, 0.2, 0.95])
    ctx = PolicyContext(seed=0, round_idx=0, num_classes=2,
                        features={"task_unet_pool": feats})
    sel = build("P9", cluster_rule="fixed:2").select(ctx, scores, k=2)
    assert len(sel) == 2 and len(set(sel)) == 2
    # one per cluster (diverse), each the highest-score (lowest-acc) member
    assert set(sel) == {1, 5}, sel
    assert ctx.diagnostics_out.get("paal_n_clusters") == 2


# ---- SAM model-type config ------------------------------------------------

def test_sam_config_allows_vit_b_vit_l_vit_h(tmp_path):
    from medal_bench.features.sam import resolve_sam_spec
    assert resolve_sam_spec("vit_b").model_type == "vit_b"
    ck = tmp_path / "fake.pth"; ck.write_bytes(b"0")
    for mt in ("vit_l", "vit_h"):
        spec = resolve_sam_spec(mt, checkpoint=str(ck))
        assert spec.model_type == mt and spec.backend == "original"
    with pytest.raises(ValueError):
        resolve_sam_spec("vit_x")


def test_sam_grayscale_rgb_conversion_deterministic():
    from medal_bench.features.sam import _to_sam_input, SamPreprocessConfig
    cfg = SamPreprocessConfig()
    img = np.random.RandomState(0).rand(1, 40, 50).astype(np.float32)
    a, b = _to_sam_input(img, cfg), _to_sam_input(img, cfg)
    assert a.shape == (1, 3, 1024, 1024)
    assert torch.allclose(a, b)


# ---- no NaN/Inf scores ----------------------------------------------------

@pytest.mark.parametrize("pid", ["P1", "P2", "P5", "P6", "P9"])
def test_no_nan_inf_scores(pid, make_ctx):
    ctx = make_ctx(seed=3, round_idx=0, want_features=True, want_foundation=True)
    pol = build(pid, ap_epochs=2, ap_batch_size=2) if pid == "P9" else build(pid)
    s = pol.score(ctx)
    assert s is not None
    arr = s.detach().cpu().numpy() if hasattr(s, "detach") else np.asarray(s)
    assert np.isfinite(arr).all(), f"{pid} produced non-finite scores"
