"""
Signal registry.

A strategy YAML names a signal by string:

    signal:
      name: volar
      params:
        lookback_months: 12
        skip_months: 1

...and this registry turns that into an object. Adding a new signal (value,
quality, low-volatility) means writing one class and registering it here --
the pipeline does not change. That is the whole design (SPEC.md §3).
"""

from __future__ import annotations

from typing import Any, Callable

from engine.signals.base import Signal, SignalResult
from engine.signals.momentum import (
    CompositeSignal,
    MomentumSignal,
    SharpeSignal,
    VolarSignal,
)

_REGISTRY: dict[str, Callable[..., Signal]] = {}


def register_signal(name: str, factory: Callable[..., Signal]) -> None:
    _REGISTRY[name.lower()] = factory


def get_signal(name: str, **params: Any) -> Signal:
    """Build a signal by the name used in strategy config."""
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown signal '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[key](**params)


def registered_signals() -> list[str]:
    return sorted(_REGISTRY)


register_signal("momentum", MomentumSignal)
register_signal("volar", VolarSignal)
register_signal("sharpe", SharpeSignal)
register_signal("composite", CompositeSignal)

__all__ = [
    "Signal",
    "SignalResult",
    "MomentumSignal",
    "VolarSignal",
    "SharpeSignal",
    "CompositeSignal",
    "get_signal",
    "register_signal",
    "registered_signals",
]
