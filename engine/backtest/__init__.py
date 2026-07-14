"""Backtest: walk history forward, with no look-ahead and no free trades."""

from engine.backtest.calendar import (
    periods_per_year,
    rebalance_dates,
    trading_days,
)
from engine.backtest.engine import BacktestResult, run_backtest
from engine.backtest.overlay import (
    AbsoluteMomentum,
    AlwaysOn,
    Overlay,
    OverlayDecision,
    TrendFilter,
    get_overlay,
    registered_overlays,
)

__all__ = [
    "run_backtest",
    "BacktestResult",
    "rebalance_dates",
    "trading_days",
    "periods_per_year",
    "Overlay",
    "OverlayDecision",
    "AlwaysOn",
    "TrendFilter",
    "AbsoluteMomentum",
    "get_overlay",
    "registered_overlays",
]
