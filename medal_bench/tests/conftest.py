"""Pytest fixtures: tiny model, synthetic pool/labeled datasets,
PolicyContext factory, and a no-op `enable_mc_dropout` patch on CPU."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import List

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from medal_bench.data import MedALDataset, Sample
from medal_bench.policies import Policy, PolicyContext
from medal_bench.runner.prediction_cache import PredictionCache


@pytest.fixture(autouse=True)
def _deterministic_cudnn():
    """Match production: same-seed runs must be reproducible (see runner/seeds.py).
    Without this, GPU cuDNN conv backward makes P9's AccuracyPredictor training
    flaky (test_determinism_same_seed[P9])."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    yield


# ---- in-memory synthetic dataset ----

class TinySegDataset(MedALDataset):
    name = "synthetic"
    modality = "synthetic"
    target = "synthetic_fg"
    dim = "2d"
    query_unit = "image"
    num_classes = 4

    def __init__(self, n: int, h: int, w: int, num_classes: int, seed: int, tag: str):
        self.num_classes = num_classes
        rng = np.random.RandomState(seed)
        self._images = [torch.from_numpy(rng.randn(1, h, w).astype(np.float32)) for _ in range(n)]
        self._masks  = [torch.from_numpy(rng.randint(0, num_classes, (h, w)).astype(np.int64)) for _ in range(n)]
        self._ids = [f"{tag}_{i:03d}" for i in range(n)]

    def __len__(self): return len(self._ids)
    def sample_ids(self): return list(self._ids)
    def __getitem__(self, i: int) -> Sample:
        return Sample(sample_id=self._ids[i], image=self._images[i], mask=self._masks[i])


# ---- tiny dropout-compatible nnU-Net ----

@pytest.fixture(scope="session")
def cuda_or_cpu():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def tiny_model(cuda_or_cpu):
    """Tiny dropout-compatible nnU-Net (3 stages, dropout=0.1)."""
    from medal_bench.models.nnunet import build_unet_2d
    torch.manual_seed(0)
    net = build_unet_2d(
        input_channels=1, num_classes=4,
        features_per_stage=(4, 8, 16),
        dropout_p=0.1,
    ).to(cuda_or_cpu).eval()
    return net


# ---- synthetic active set ----

@pytest.fixture
def pool():
    return TinySegDataset(n=16, h=32, w=32, num_classes=4, seed=1, tag="pool")


@pytest.fixture
def labeled():
    return TinySegDataset(n=4, h=32, w=32, num_classes=4, seed=2, tag="lbl")


# ---- PredictionCache built from the tiny model + pool ----

@pytest.fixture
def pred_cache(tiny_model, pool, cuda_or_cpu):
    from medal_bench.runner.prediction_cache import build_prediction_cache
    def iterator():
        for i in range(len(pool)):
            s = pool[i]
            yield s.sample_id, s.image
    return build_prediction_cache(tiny_model, iterator(), device=cuda_or_cpu)


# ---- feature matrices (NxD) for P5/P7/P8/P9 ----

@pytest.fixture
def task_features(tiny_model, pool, labeled, cuda_or_cpu):
    """Task encoder features: take the encoder bottleneck via a tiny hack -
    run forward and grab the last encoder's output. For the test we just use a
    pooled flat representation of the image itself - good enough to exercise
    the policy logic."""
    def _feat(ds):
        feats = []
        for i in range(len(ds)):
            x = ds[i].image
            feats.append(x.mean(dim=(-2, -1)).flatten().cpu().numpy())  # (1,)-D x N
        # repeat to give policies something to discriminate
        F = np.stack(feats, axis=0)
        F = np.tile(F, (1, 8))
        return F.astype(np.float32)
    return {"task_unet_pool": _feat(pool), "task_unet_label": _feat(labeled)}


@pytest.fixture
def foundation_features(pool, labeled):
    """Fake foundation features (random but seeded), 384-D like ViT-S/14."""
    rng_p = np.random.RandomState(101)
    rng_l = np.random.RandomState(202)
    return {
        "foundation_pool":  rng_p.randn(len(pool),    384).astype(np.float32),
        "foundation_label": rng_l.randn(len(labeled), 384).astype(np.float32),
    }


# ---- context factory ----

@pytest.fixture
def make_ctx(tiny_model, pool, labeled, pred_cache, task_features, foundation_features):
    def _make(seed=1000, round_idx=0, want_features=False, want_foundation=False):
        feats = {}
        if want_features:
            feats.update(task_features)
        if want_foundation:
            feats.update(foundation_features)
        return PolicyContext(
            seed=seed, round_idx=round_idx, model=tiny_model,
            pred_cache=pred_cache, pool=pool, labeled=labeled,
            features=feats, num_classes=4,
        )
    return _make
