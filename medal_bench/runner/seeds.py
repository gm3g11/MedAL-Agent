"""Deterministic seeding helpers.

Call ``seed_all(seed)`` ONCE per (run_id, round) BEFORE any stochastic
operation. The policy classes are responsible for using ``rng_for(seed)``
when they need an isolated RNG (matters for foundation-cache builds, k-means
init, etc.).

Notes:
- DETERMINISTIC BY DEFAULT (2026-06-12). cuDNN is set deterministic + benchmark
  off so that same-seed runs reproduce identical trained weights — and therefore
  identical AL selections — across all score-based policies (P1/P2/P5/P6/P9).
  Previously cudnn.deterministic=False made trajectories non-reproducible (P9's
  AccuracyPredictor training was the most visible symptom: flaky same-seed test).
- Opt back into the faster non-deterministic mode with env MEDAL_NONDETERMINISTIC=1
  (only for throughput experiments where reproducibility is not required).
"""
from __future__ import annotations
import os
import random
import numpy as np


def deterministic_enabled() -> bool:
    return os.environ.get("MEDAL_NONDETERMINISTIC", "0") != "1"


def seed_all(seed: int) -> None:
    """Seed every RNG that affects selection or training reproducibility, and
    set the cuDNN determinism mode (deterministic unless MEDAL_NONDETERMINISTIC=1)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        det = deterministic_enabled()
        torch.backends.cudnn.deterministic = det
        torch.backends.cudnn.benchmark = not det
        if det:
            # cudnn.deterministic alone does NOT make all CUDA ops deterministic at
            # larger resolutions (e.g. conv-transpose backward uses atomics). Force
            # deterministic algorithms everywhere; warn_only so unsupported ops don't
            # crash. Requires the cuBLAS workspace env for matmul determinism.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            # TF32 (Ampere/Hopper tensor-core fp32) is ~2x faster on conv/matmul and is
            # ITSELF deterministic — bit-identical weight hashes across re-runs (verified).
            # The original cross-arch confound (TF32-on A40/H100 vs no-TF32 V100 -> ~0.005
            # DSC drift) is eliminated by SINGLE-ARCH PINNING: each dataset's full policy
            # matrix runs on ONE GPU arch, so TF32 is consistent within every comparison.
            # So we keep TF32 ON for speed without reintroducing the confound. (V100/Volta
            # has no TF32 -> those datasets stay fp32 regardless.)
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            # warn_only=True (default) lets ops lacking a deterministic impl run
            # NON-deterministically instead of crashing. Set MEDAL_STRICT_DETERMINISM=1
            # to flip warn_only=False so torch RAISES and names the offending op (debug).
            try:
                _strict = os.environ.get("MEDAL_STRICT_DETERMINISM") == "1"
                torch.use_deterministic_algorithms(True, warn_only=not _strict)
            except Exception:
                pass
    except ImportError:
        pass


def rng_for(seed: int) -> np.random.RandomState:
    """Isolated RandomState. Useful for sklearn calls that take random_state."""
    return np.random.RandomState(seed)


# Component-level seeds (frozen_v3): a single round seed (seed+r) is split into
# four independent component streams so model weight init, the batch loader, the
# query/selection RNG, and training dropout are each reproducible on their own and
# do not depend on the byte-exact op order of prior rounds. Derived with
# numpy.SeedSequence so the four uint32 streams are well-separated and deterministic.
_COMPONENTS = ("model_init_seed", "loader_seed", "query_seed", "dropout_seed")


def component_seeds(round_seed: int) -> dict:
    """Return the four component seeds derived from ``round_seed`` (= cfg.seed + r)."""
    state = np.random.SeedSequence(int(round_seed)).generate_state(len(_COMPONENTS))
    return {name: int(v) for name, v in zip(_COMPONENTS, state)}


def seed_torch(seed: int) -> None:
    """Seed ONLY the torch global RNG (CPU + CUDA). Used to anchor a single torch
    component (weight init or dropout) without re-running the full seed_all setup."""
    try:
        import torch
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except ImportError:
        pass
