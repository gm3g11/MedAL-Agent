"""Smoke-gate-only foundation feature stub.

P8/P9 expect foundation-model features (DINOv2-ViT-S/14 by default). For the
v1 smoke matrix we need a deterministic feature vector per sample to verify
the selection plumbing — not feature quality. We emit a seeded-random 384-D
vector keyed on (sample_id, seed) so:

  - the same (sample_id, seed) always produces the same vector;
  - the smoke matrix is fully reproducible;
  - the policy logic (k-center, k-means++ on these vectors) gets exercised.

The trajectory record's ``foundation`` block must log
``encoder_id="stub_seeded_random"`` and ``cache_version="smoke_v0"`` so it
is obvious this is NOT a real DINOv2 run (constraint #8).
"""
from __future__ import annotations

import hashlib

import numpy as np

from medal_bench.data.base import MedALDataset


STUB_DIM = 384  # matches DINOv2-ViT-S/14 dim so swap-in is trivial later
STUB_ENCODER_ID = "stub_seeded_random"
STUB_CACHE_VERSION = "smoke_v0"


def _sample_seed(sample_id: str, seed: int) -> int:
    h = hashlib.sha256(f"{sample_id}|{seed}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")


def extract_foundation_features_stub(ds: MedALDataset, seed: int) -> np.ndarray:
    """Return (N, STUB_DIM) seeded-random foundation features."""
    sids = ds.sample_ids()
    out = np.empty((len(sids), STUB_DIM), dtype=np.float32)
    for i, sid in enumerate(sids):
        rng = np.random.RandomState(_sample_seed(sid, seed))
        out[i] = rng.standard_normal(STUB_DIM).astype(np.float32)
    return out


def foundation_stub_meta() -> dict:
    return {
        "encoder_id": STUB_ENCODER_ID,
        "checkpoint": None,
        "layer": None,
        "pooling_rule": "deterministic_seeded_random",
        "cache_version": STUB_CACHE_VERSION,
        "dim": STUB_DIM,
    }
