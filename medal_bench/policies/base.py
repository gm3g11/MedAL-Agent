"""Policy interface and shared types.

A policy has two responsibilities at AL round t:

  1. score(model, pool)         -> np.ndarray[N] of per-unit scores
                                   (may be None for purely random policies)
  2. select(scores, features, k) -> list[int] of length k, indices into the
                                    candidate pool

The runner provides a per-round PredictionCache and any feature matrices the
policy needs. The runner is also responsible for the firewall: pool data and
the trained model are passed in; val/test labels are NEVER passed.

Determinism contract: every policy must be seedable. Calls into stochastic
samplers (np.random, sklearn) must be derived from a single seed at
__init__ time so two runs with the same seed produce identical selections.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np


@dataclass
class PolicyContext:
    """Inputs the runner provides to a policy each round.

    Held as a dataclass so policies can read what they need without coupling
    to runner internals.
    """
    seed: int
    round_idx: int
    # the model is whatever the runner's trainer returns; policies that don't
    # need a model (P0) can ignore it.
    model: Any = None
    # per-round prediction cache; built once and shared across policies that
    # need probs / argmax. None for P0; available for P1-P4 and P7.
    pred_cache: Any = None
    # pool dataset (read-only). Length = N; indexable.
    pool: Any = None
    # labeled dataset (for representativeness-based policies like P5/P7's
    # k-center initial seeds). Read-only metadata only; the policy must NOT
    # use labels.
    labeled: Any = None
    # feature matrices (NxD) when applicable, indexed by feature name
    # ('task_unet' | 'foundation'). The runner pre-computes these or returns
    # None when a policy doesn't need them.
    features: Mapping[str, np.ndarray] = field(default_factory=dict)
    # number of classes for the segmentation task (incl. background = 0)
    num_classes: int = 0
    # per-pool-unit valid (un-padded) rectangle, aligned to ``pool`` order:
    # int array (N, 4) = (y0, x0, h, w). None when there is no padding (square
    # resize) — aggregation then falls back to the full canvas. Used by P1/P2/P5/P6
    # to avoid letterbox-pad-driven query scores. A bbox (not an N,H,W mask) keeps
    # this O(N) instead of O(N·H·W).
    valid_bboxes: Optional[np.ndarray] = None
    # component query seed (= numpy SeedSequence(seed+round_idx) stream); policies
    # that draw RNG may prefer this. Falls back to seed+round_idx when 0/unset.
    query_seed: int = 0
    # for streaming-reduction policies (P1/P5/P6, which declare
    # needs_pred_cache_probs=False): the runner pre-accumulates the per-batch (N,)
    # reductions here so the full (N,C,H,W) probs is never materialized. dict of
    # {key: (N,) cpu tensor}. None -> the policy falls back to reducing
    # pred_cache.probs directly (used by unit tests that populate probs).
    streamed_reduce: Optional[Mapping[str, Any]] = None
    # opaque dict the policy can write diagnostics into; runner logs them
    # to the JSONL trajectory.
    diagnostics_out: dict = field(default_factory=dict)


class Policy(abc.ABC):
    """Abstract policy.

    Subclasses set the class-level ``name`` (id is owned by the registry) and
    implement ``score`` and/or ``select``. Most policies are score+top-k; a few
    (P0 Random, P3 CoreSet, P4 BADGE, P7 SAM-CoreSet, P8 SAM-TypiClust)
    do not override ``score()`` — their selection is feature-clustering only.
    """
    id: str = ""               # e.g., "P3"
    name: str = ""             # e.g., "ROI-aware Entropy"
    version: str = "1.0"       # method-implementation version (bump on algorithm change)
    is_ablation: bool = False  # True for non-core ablation methods (P4b, P8b)
    needs_pred_cache: bool = False
    # whether the policy needs the FULL probs (N,C,H,W) materialized. Streaming
    # policies (P1/P5/P6) set this False: the runner streams their per_batch_reduce
    # and only materializes the small argmax. P9 keeps it True (it feeds probs into
    # a network). Only consulted when needs_pred_cache is True.
    needs_pred_cache_probs: bool = True
    needs_features: tuple[str, ...] = ()   # which feature matrices the runner must precompute

    def __init__(self, **config: Any) -> None:
        self.config: dict[str, Any] = dict(config)

    # ------------------------------------------------------------------
    # scoring (per-unit float; larger == "more selectable")
    # ------------------------------------------------------------------
    def score(self, ctx: PolicyContext) -> Optional[np.ndarray]:
        """Return per-unit scores (N,) float32, or None if the policy is purely
        sampling-based (e.g., P0 random, P5 k-center on its own).
        """
        return None

    # ------------------------------------------------------------------
    # selection
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def select(self, ctx: PolicyContext, scores: Optional[np.ndarray], k: int) -> list[int]:
        """Return ``k`` indices into the pool (no duplicates). ``scores`` is
        whatever ``score`` returned for this round (or None).
        """
        ...

    # ------------------------------------------------------------------
    # utility
    # ------------------------------------------------------------------
    def describe(self) -> Mapping[str, Any]:
        """Stable description for the trajectory log."""
        return {"policy_id": self.id, "policy_name": self.name,
                "method_version": self.version, "is_ablation": self.is_ablation,
                "policy_config": dict(self.config)}
