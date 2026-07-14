"""
Weight drift: what you actually hold, versus what you thought you held.

You buy 20 names at 5% each. A month later the winners have grown to 7% and the
losers shrunk to 3%. YOU DID NOTHING, but your portfolio changed.

This matters for two reasons:

1. TURNOVER. The trades needed at the next rebalance are measured against these
   DRIFTED weights, not against the original 5% targets. A backtest that skips
   drift computes the wrong trades and therefore the wrong costs.

2. RETURNS. The portfolio's return over the holding period is the weighted sum of
   its constituents' returns -- which requires knowing the weights at the START of
   the period.

Momentum has a pleasant quirk here: drift is partly self-reinforcing. Winners grow
their own weight, which is a mild free "let your profits run" effect between
rebalances.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.data.base import PriceData
from engine.portfolio.portfolio import Portfolio


def drift_weights(
    portfolio: Portfolio,
    prices: PriceData,
    to_date: pd.Timestamp,
) -> tuple[pd.Series, float]:
    """
    Carry a portfolio forward from its own date to `to_date` with no trading.

    Returns:
        (drifted_weights, period_return)

    Cash earns nothing here. That is a deliberate simplification for V0 -- it makes
    a cash position slightly PESSIMISTIC rather than flattering, which is the right
    direction to be wrong in.
    """
    start = pd.Timestamp(portfolio.as_of)
    end = pd.Timestamp(to_date)

    if end < start:
        raise ValueError(f"Cannot drift backwards: {start.date()} -> {end.date()}")

    if portfolio.n_positions == 0:
        return pd.Series(dtype=float), 0.0        # all cash: no drift, no return

    close = prices.close

    returns: dict[str, float] = {}
    for position in portfolio.positions:
        if position.symbol not in close.columns:
            returns[position.symbol] = 0.0        # no data -> assume flat, never NaN
            continue

        series = close[position.symbol].dropna()
        if series.empty:
            returns[position.symbol] = 0.0
            continue

        p0 = series.asof(start)
        p1 = series.asof(end)

        if not (pd.notna(p0) and pd.notna(p1)) or p0 <= 0:
            returns[position.symbol] = 0.0
            continue

        returns[position.symbol] = float(p1 / p0) - 1.0

    weights = portfolio.weights
    rets = pd.Series(returns, dtype=float).reindex(weights.index).fillna(0.0)

    # Growth factor of each holding, and of the portfolio as a whole.
    grown = weights * (1.0 + rets)

    # Cash is part of the portfolio and does not grow.
    total_value = float(grown.sum()) + portfolio.cash_weight

    if total_value <= 0:
        return weights, -1.0

    period_return = total_value - 1.0

    # Re-express as weights of the NEW, larger (or smaller) portfolio.
    drifted = grown / total_value

    return drifted, period_return


def drifted_cash_weight(portfolio: Portfolio, drifted: pd.Series) -> float:
    """Whatever is not in positions after drift is cash."""
    return max(0.0, 1.0 - float(drifted.sum()))
