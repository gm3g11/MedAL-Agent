"""P4 canonical BADGE tests (Section 4): the analytic CE gradient embedding must
match autograd, have dimension C*D, reflect current weights (no stale cache), and
be distinct from the P4b CE+Dice ablation."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from medal_bench.policies import build
from medal_bench.policies._badge_grad import (
    canonical_ce_grad_embedding, ce_dice_grad_embedding, _find_seg_head, _to_input,
)

NC = 4  # tiny_model num_classes


def test_badge_canonical_gradient_matches_autograd_toy(tiny_model, pool):
    """Analytic embedding == autograd gradient of the CE pseudo-label loss w.r.t.
    the 1x1 seg-head weight (CE reduction='mean' => the mean-pooled BADGE form)."""
    g = canonical_ce_grad_embedding(tiny_model, pool[0], NC).reshape(NC, -1)
    head = _find_seg_head(tiny_model, NC)
    x = _to_input(pool[0].image, next(tiny_model.parameters()).device)
    tiny_model.zero_grad()
    logits = tiny_model(x)
    pseudo = torch.argmax(logits, dim=1)
    F.cross_entropy(logits, pseudo).backward()
    autograd = head.weight.grad.detach().reshape(NC, -1).cpu()
    tiny_model.zero_grad()
    assert g.shape == autograd.shape
    assert torch.allclose(g, autograd, atol=1e-4), (g[0, :3], autograd[0, :3])


def test_badge_embedding_dimension(tiny_model, pool):
    D = _find_seg_head(tiny_model, NC).in_channels
    g = canonical_ce_grad_embedding(tiny_model, pool[0], NC)
    assert g.numel() == NC * D, (g.numel(), NC, D)


def test_badge_embedding_reflects_weight_change(tiny_model, pool):
    """No stale cache: same model object, changed weights => different embedding
    (covers the old id(model)-cache staleness)."""
    g1 = canonical_ce_grad_embedding(tiny_model, pool[0], NC)
    with torch.no_grad():
        for p in tiny_model.parameters():
            p.add_(0.5 * torch.randn_like(p))
    g2 = canonical_ce_grad_embedding(tiny_model, pool[0], NC)
    assert not torch.allclose(g1, g2)


def test_badge_canonical_and_ce_dice_embeddings_differ(tiny_model, pool):
    g_can = canonical_ce_grad_embedding(tiny_model, pool[0], NC)
    g_cd = ce_dice_grad_embedding(tiny_model, pool[0], NC)
    assert not torch.allclose(g_can, g_cd)


def test_badge_p4_and_p4b_distinct_ids():
    p4, p4b = build("P4"), build("P4b")
    assert p4.id == "P4" and p4b.id == "P4b"
    assert type(p4).__name__ != type(p4b).__name__
    assert "ce-dice" in p4b.name.lower()


def test_badge_kmeanspp_unique(make_ctx):
    ctx = make_ctx(seed=7, round_idx=0)
    out = build("P4").select(ctx, None, k=3)
    assert len(out) == 3 and len(set(out)) == 3
