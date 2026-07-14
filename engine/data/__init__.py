"""
Data adapter registry.

The market config says WHICH vendor to use (`data_adapter: yfinance`).
This factory turns that string into an object. Adding a new vendor means
registering it here and naming it in a YAML file -- no engine changes.
"""

from __future__ import annotations

from typing import Callable

from engine.data.base import PriceAdapter, PriceData
from engine.markets.market import Market

_REGISTRY: dict[str, Callable[..., PriceAdapter]] = {}


def register_adapter(name: str, factory: Callable[..., PriceAdapter]) -> None:
    """Make an adapter available to market configs under `name`."""
    _REGISTRY[name.lower()] = factory


def get_adapter(market: Market, **kwargs) -> PriceAdapter:
    """
    Build the adapter this market's config asks for.

        market = load_market("india")
        adapter = get_adapter(market)      # -> whatever india.yaml names
    """
    name = market.data_adapter.lower()
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown data_adapter '{market.data_adapter}' in market "
            f"'{market.market_id}'. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](market, **kwargs)


def registered_adapters() -> list[str]:
    return sorted(_REGISTRY)


# --- built-in registrations ------------------------------------------------

def _make_yfinance(market: Market, **kwargs) -> PriceAdapter:
    from engine.data.yfinance_adapter import YFinanceAdapter

    return YFinanceAdapter(market, **kwargs)


register_adapter("yfinance", _make_yfinance)

__all__ = [
    "PriceAdapter",
    "PriceData",
    "get_adapter",
    "register_adapter",
    "registered_adapters",
]
