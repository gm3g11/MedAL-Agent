"""Policy registry: name -> Policy class.

Policies register themselves at import time via @register("P3").
The runner loads the registry once (medal_bench.policies) and looks up by id.
"""
from __future__ import annotations
from typing import Callable, Dict, Type

from medal_bench.policies.base import Policy


_REGISTRY: Dict[str, Type[Policy]] = {}


def register(policy_id: str) -> Callable[[Type[Policy]], Type[Policy]]:
    """Decorator: register a Policy subclass under its short id (e.g., "P3")."""
    def deco(cls: Type[Policy]) -> Type[Policy]:
        if policy_id in _REGISTRY:
            raise KeyError(f"policy id {policy_id!r} already registered: {_REGISTRY[policy_id]}")
        cls.id = policy_id
        _REGISTRY[policy_id] = cls
        return cls
    return deco


def get(policy_id: str) -> Type[Policy]:
    if policy_id not in _REGISTRY:
        raise KeyError(
            f"unknown policy {policy_id!r}. Registered: {sorted(_REGISTRY)}. "
            "Did you import medal_bench.policies?"
        )
    return _REGISTRY[policy_id]


def build(policy_id: str, **config) -> Policy:
    return get(policy_id)(**config)


def all_ids() -> list[str]:
    return sorted(_REGISTRY)
