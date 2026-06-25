"""Acquisition memory-bounding equivalence gate.

The streaming/chunked acquisition path (P1/P5/P6 via stream_pool_reduce, P2 BALD
via per-batch accumulation) must select the SAME samples as the old
full-materialization path that built the entire (N, C, H, W) probs on CPU.

For a small MULTI-CLASS synthetic case (C=4, N=40, H=W=64) with non-trivial
letterbox valid_bboxes and a tiny real build_unet_2d in eval mode, we run BOTH
paths and assert the actual select(...) output is IDENTICAL, and scores match to
1e-6 where applicable. "Old" is reconstructed inline (whole-pool reduction).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from medal_bench.models.nnunet import build_unet_2d
from medal_bench.policies import build, PolicyContext
from medal_bench.runner.prediction_cache import (
    build_prediction_cache, stream_pool_reduce, PredictionCache,
)

C = 4
N = 40
HW = 64
BS = 7          # deliberately not a divisor of N, to exercise ragged last batch
SEED = 1234


def _model():
    torch.manual_seed(0)
    return build_unet_2d(
        input_channels=1, num_classes=C,
        features_per_stage=(4, 8, 16),
        dropout_p=0.1,
    ).eval()


def _images():
    rng = np.random.RandomState(SEED)
    return [torch.from_numpy(rng.randn(1, HW, HW).astype(np.float32)) for _ in range(N)]


def _valid_bboxes():
    """Non-trivial letterbox rects: a mix of full-canvas and padded samples."""
    rng = np.random.RandomState(7)
    bb = []
    for i in range(N):
        if i % 3 == 0:
            bb.append((0, 0, HW, HW))                         # full canvas
        else:
            y0 = int(rng.randint(0, 8)); x0 = int(rng.randint(0, 8))
            h = HW - y0 - int(rng.randint(0, 8))
            w = HW - x0 - int(rng.randint(0, 8))
            bb.append((y0, x0, h, w))
    return np.asarray(bb, dtype=np.int64)


def _iter(images):
    for i, x in enumerate(images):
        yield f"s_{i:03d}", x


class _Pool:
    """Minimal pool exposing .image / .sample_id (P2 reads ctx.pool)."""
    def __init__(self, images):
        self._imgs = images
    def __len__(self):
        return len(self._imgs)
    def __getitem__(self, i):
        from types import SimpleNamespace
        return SimpleNamespace(image=self._imgs[i], sample_id=f"s_{i:03d}")


def _old_full_cache(model, images):
    """OLD path: materialize the full (N, C, H, W) probs + (N, H, W) argmax."""
    return build_prediction_cache(model, _iter(images), device="cpu", batch_size=BS)


# --------------------------------------------------------------------------
# P1 / P5 / P6: streamed reduction must equal whole-pool reduction
# --------------------------------------------------------------------------

def _ctx_old(model, images, vb, pol):
    cache = _old_full_cache(model, images)
    return PolicyContext(
        seed=SEED, round_idx=0, model=model, pred_cache=cache,
        pool=_Pool(images), num_classes=C, valid_bboxes=vb,
        streamed_reduce=None,
    )


def _ctx_new(model, images, vb, pol):
    reduced, argmax, fnames = stream_pool_reduce(
        model, _iter(images), per_batch_fn=pol.per_batch_reduce,
        device="cpu", batch_size=BS, valid_bboxes=vb,
    )
    cache = PredictionCache(probs=None, argmax=argmax, fnames=fnames)
    return PolicyContext(
        seed=SEED, round_idx=0, model=model, pred_cache=cache,
        pool=_Pool(images), num_classes=C, valid_bboxes=vb,
        streamed_reduce=reduced,
    )


def _task_features(images):
    feats = np.stack([im.mean(dim=(-2, -1)).flatten().numpy() for im in images], axis=0)
    feats = np.tile(feats, (1, 8)).astype(np.float32)
    lbl = feats[:4].copy()
    return {"task_unet_pool": feats, "task_unet_label": lbl}


def test_p1_streamed_equals_full():
    model = _model(); images = _images(); vb = _valid_bboxes()
    k = 9
    p_old = build("P1"); p_new = build("P1")
    c_old = _ctx_old(model, images, vb, p_old)
    c_new = _ctx_new(model, images, vb, p_new)
    s_old = p_old.score(c_old); s_new = p_new.score(c_new)
    assert torch.allclose(s_old, s_new, atol=1e-6), (s_old - s_new).abs().max()
    assert p_old.select(c_old, s_old, k) == p_new.select(c_new, s_new, k)


def test_p5_streamed_equals_full():
    model = _model(); images = _images(); vb = _valid_bboxes()
    k = 6
    feats = _task_features(images)
    p_old = build("P5"); p_new = build("P5")
    c_old = _ctx_old(model, images, vb, p_old); c_old.features = feats
    c_new = _ctx_new(model, images, vb, p_new); c_new.features = feats
    s_old = p_old.score(c_old); s_new = p_new.score(c_new)
    assert torch.allclose(s_old, s_new, atol=1e-6), (s_old - s_new).abs().max()
    assert p_old.select(c_old, s_old, k) == p_new.select(c_new, s_new, k)


def test_p6_streamed_equals_full():
    model = _model(); images = _images(); vb = _valid_bboxes()
    k = 11
    p_old = build("P6"); p_new = build("P6")
    c_old = _ctx_old(model, images, vb, p_old)
    c_new = _ctx_new(model, images, vb, p_new)
    s_old = p_old.score(c_old); s_new = p_new.score(c_new)
    assert torch.allclose(s_old, s_new, atol=1e-6), (s_old - s_new).abs().max()
    # the component rankings (which drive selection) must match exactly
    assert np.array_equal(p_old._s_target, p_new._s_target)
    assert np.array_equal(p_old._s_boundary, p_new._s_boundary)
    assert p_old.select(c_old, s_old, k) == p_new.select(c_new, s_new, k)
    # diagnostics recombine exactly
    for key in ("selu_target_frac", "selu_boundary_frac",
                "selu_score_target_mean", "selu_score_boundary_mean"):
        assert abs(c_old.diagnostics_out[key] - c_new.diagnostics_out[key]) < 1e-6


# --------------------------------------------------------------------------
# P2 BALD: chunked per-batch accumulation must be byte-identical to the
# original (running-sum built from per-pass torch.cat of full (N,C,H,W)).
# --------------------------------------------------------------------------

def _bald_old_score(model, images, vb, T, bs):
    """Original P2 accumulation: per-pass full (N,C,H,W) cat, then sum.
    Same forward order (pass-outer, batch-inner) and same single seed, so the
    NEW in-place per-batch accumulation must reproduce this exactly."""
    from medal_bench.models.nnunet import enable_mc_dropout
    from medal_bench.policies._helpers import aggregate, topk_indices

    device = "cpu"
    torch.manual_seed(SEED + 0)
    enable_mc_dropout(model)
    imgs = list(images)

    running_sum_probs = None
    sum_per_pass_entropy = None
    with torch.no_grad():
        for _ in range(T):
            pass_probs = []
            pass_entropy = []
            for start in range(0, len(imgs), bs):
                xb = torch.stack(imgs[start:start + bs], dim=0).to(device, dtype=torch.float32)
                p = F.softmax(model(xb), dim=1)
                e = torch.mean(-p * torch.log2(p + 1e-12), dim=1)
                pass_probs.append(p.detach().cpu())
                pass_entropy.append(e.detach().cpu())
            P = torch.cat(pass_probs, dim=0)
            E = torch.cat(pass_entropy, dim=0)
            if running_sum_probs is None:
                running_sum_probs = P.clone(); sum_per_pass_entropy = E.clone()
            else:
                running_sum_probs += P; sum_per_pass_entropy += E
    model.eval()
    mean_probs = running_sum_probs / T
    pred_ent = torch.mean(-mean_probs * torch.log2(mean_probs + 1e-12), dim=1)
    mean_pp_ent = sum_per_pass_entropy / T
    bald = pred_ent - mean_pp_ent
    score = aggregate(bald, torch.argmax(mean_probs, dim=1), "valid", valid_bboxes=vb)
    return score, topk_indices(score, 9)


def test_p2_bald_chunked_equals_original():
    vb = _valid_bboxes()
    T = 4
    # OLD: reconstruct the original accumulation on a fresh model.
    model_old = _model(); images = _images()
    s_old, sel_old = _bald_old_score(model_old, images, vb, T=T, bs=BS)

    # NEW: run the refactored P2 policy (in-place per-batch accumulation).
    model_new = _model()
    pol = build("P2", T=T, mc_batch_size=BS)
    ctx = PolicyContext(
        seed=SEED, round_idx=0, model=model_new, pool=_Pool(images),
        num_classes=C, valid_bboxes=vb,
    )
    s_new = pol.score(ctx)
    sel_new = pol.select(ctx, s_new, 9)

    assert torch.allclose(s_old, s_new, atol=1e-6), (s_old - s_new).abs().max()
    assert sel_old == sel_new
