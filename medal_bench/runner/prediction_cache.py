"""Per-round prediction cache built once and shared across components.

For policies that need probs/argmax over the unlabeled pool, the runner builds
this once at the start of each AL round and passes it through PolicyContext.
P2 BALD does its own K-pass MC dropout (separate from this cache); it still
reuses the deterministic ``argmax`` for fg/boundary masks so the fallback
triggers are consistent across BALD and ``foreground_bald``-style policies.

Memory: the full ``probs`` (N, C, H, W) float32 is hundreds of GB for large
multi-class pools (e.g. mmwhs C=8, N~10k -> ~171 GB) and OOM-kills the runner.
Policies whose per-sample score is a per-sample-independent reduction of probs
(P1/P5/P6) instead use ``stream_pool_reduce`` below, which runs the batched
forward and accumulates only the small per-sample ``(N,)`` reductions, never
holding the full ``(N, C, H, W)``. ``build_prediction_cache`` then materializes
only the small ``argmax`` (uint8) for those policies. Policies that genuinely
need the full probs (P9, which concatenates probs with the image as a network
input) still request ``materialize_probs=True``.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F


@dataclass
class PredictionCache:
    probs: Optional[torch.Tensor]   # (N, C, H, W) float32 on CPU; softmaxed. None when not materialized.
    argmax: torch.Tensor            # (N, H, W) uint8 on CPU (num_classes <= 255, exact)
    fnames: List                    # row-aligned sample ids


def _batched_forward(model, images_iter, device: str, batch_size: int):
    """Yield (probs_batch (B,C,H,W) float32 cpu, argmax_batch (B,H,W) uint8 cpu,
    ids_batch) over ``images_iter`` yielding (sample_id, (C,H,W) tensor).

    Numerically identical to one-at-a-time inference: the net is in eval mode and
    uses InstanceNorm (per-sample), so there is no cross-batch interaction.
    """
    batch_ids: List = []
    batch_x: List = []

    def _run():
        xb = torch.stack(batch_x, dim=0).to(device, dtype=torch.float32)  # (B, C, H, W)
        p = F.softmax(model(xb), dim=1)
        a = torch.argmax(p, dim=1).to(torch.uint8)
        out = (p.detach().cpu(), a.detach().cpu(), list(batch_ids))
        batch_x.clear()
        batch_ids.clear()
        return out

    for sample_id, x in images_iter:
        if x.dim() == 4:
            x = x.squeeze(0)                 # tolerate (1, C, H, W)
        batch_x.append(x)
        batch_ids.append(sample_id)
        if len(batch_x) >= batch_size:
            yield _run()
    if batch_x:
        yield _run()


@torch.no_grad()
def build_prediction_cache(model, images_iter, device: str = "cuda:0",
                           batch_size: int = 32,
                           materialize_probs: bool = True) -> PredictionCache:
    """Build the cache from an iterator yielding (sample_id, image_tensor).

    ``image_tensor`` is (C, H, W), all at the runner's image_size (uniform within
    a run). Images are batched for throughput. When ``materialize_probs`` is False
    only the small ``argmax`` (uint8) is kept (for save_preds / fg diagnostics);
    the full probs is never concatenated, bounding RAM to O(batch * C * H * W).
    """
    was_training = model.training
    model.eval()

    probs_chunks: List = [] if materialize_probs else None
    argmax_chunks: List = []
    fnames: List = []

    for p, a, ids in _batched_forward(model, images_iter, device, batch_size):
        if materialize_probs:
            probs_chunks.append(p)
        argmax_chunks.append(a)
        fnames.extend(ids)

    if was_training:
        model.train()

    probs = torch.cat(probs_chunks, dim=0) if materialize_probs else None
    argmax = torch.cat(argmax_chunks, dim=0)
    assert argmax.shape[0] == len(fnames)
    if probs is not None:
        assert probs.shape[0] == argmax.shape[0]
    return PredictionCache(probs=probs, argmax=argmax, fnames=fnames)


@torch.no_grad()
def stream_pool_reduce(
    model,
    images_iter,
    per_batch_fn: Callable[[torch.Tensor, torch.Tensor, Optional["np.ndarray"], int], Dict[str, torch.Tensor]],
    device: str = "cuda:0",
    batch_size: int = 32,
    valid_bboxes=None,
    build_argmax: bool = True,
):
    """Stream the pool through ``model`` in batches, apply ``per_batch_fn`` to each
    batch's (B,C,H,W) probs + (B,H,W) argmax, and concatenate the per-batch (B,)
    reductions into whole-pool (N,) tensors. Never holds the full (N,C,H,W) probs.

    ``per_batch_fn(probs_b, argmax_b, valid_b, offset) -> dict[str, (B,) tensor]``
    where ``valid_b`` is the slice of ``valid_bboxes`` for this batch (or None) and
    ``offset`` is the running global index of the first sample in the batch.

    Returns ``(reduced, argmax, fnames)`` where ``reduced`` is a dict mapping each
    key to a concatenated (N,) cpu tensor, ``argmax`` is the (N,H,W) uint8 cache
    (or None when ``build_argmax`` is False), and ``fnames`` are the row-aligned ids.

    Because the net is eval-mode + InstanceNorm (per-sample), batching is numerically
    identical to one-at-a-time; the concatenation of per-batch (B,) results equals the
    whole-pool (N,) reduction exactly.
    """
    was_training = model.training
    model.eval()

    accum: Dict[str, List[torch.Tensor]] = {}
    argmax_chunks: List = [] if build_argmax else None
    fnames: List = []
    offset = 0
    for p, a, ids in _batched_forward(model, images_iter, device, batch_size):
        valid_b = valid_bboxes[offset:offset + p.shape[0]] if valid_bboxes is not None else None
        out = per_batch_fn(p, a, valid_b, offset)
        for key, val in out.items():
            accum.setdefault(key, []).append(val.detach().cpu())
        if build_argmax:
            argmax_chunks.append(a)
        fnames.extend(ids)
        offset += p.shape[0]

    if was_training:
        model.train()

    reduced = {key: torch.cat(chunks, dim=0) for key, chunks in accum.items()}
    argmax = torch.cat(argmax_chunks, dim=0) if build_argmax else None
    return reduced, argmax, fnames
