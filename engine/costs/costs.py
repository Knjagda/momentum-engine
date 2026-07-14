"""
Costs and trades. Where paper returns quietly leak away.

SPEC.md §4.3: EVERY simulated trade pays. No exceptions, no cost-free fills.

This module is small and unglamorous, and it is the difference between a backtest
and a fantasy. The pitch deck's Indian research showed ~32% MONTHLY turnover; at
India's ~49bps round-trip that is roughly 0.32 x 49bps x 12 ≈ 1.9% a year burnt
before the strategy earns a rupee. A momentum strategy that beats its benchmark by
2% gross is, after costs, a worse way to lose money than an index fund.

Three subtleties that most homemade backtests get wrong:

1. WEIGHTS DRIFT. You bought 20 names at 5% each; a month later the winners are 7%
   and the losers 3%. The trades you must place are measured against those DRIFTED
   weights, not against your original targets. Ignore this and turnover is wrong.

2. COSTS ARE ASYMMETRIC. In India, selling triggers STT; buying triggers stamp duty.
   Selling genuinely costs more than buying. The engine reads both from the market
   config rather than assuming a single number.

3. TINY TRADES ARE NOT WORTH MAKING. Rebalancing a position by 0.03% burns more in
   friction than it corrects. A minimum trade size prevents death by a thousand cuts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from engine.markets.market import Market

Side = Literal["BUY", "SELL"]

BPS = 10_000.0


@dataclass(frozen=True)
class Trade:
    """One instruction. Sizes are PORTFOLIO WEIGHTS, not share counts."""

    symbol: str
    side: Side
    weight_delta: float        # always positive; `side` carries the direction
    cost_bps: float            # cost rate applied to this trade
    cost_weight: float         # cost as a fraction of total portfolio value

    def __repr__(self) -> str:
        return f"<{self.side} {self.symbol} {self.weight_delta:.2%} @{self.cost_bps:.1f}bps>"


@dataclass(frozen=True)
class TradeList:
    """Everything that must happen on one rebalance date, and what it costs."""

    market_id: str
    as_of: pd.Timestamp
    trades: list[Trade] = field(default_factory=list)

    @property
    def buys(self) -> list[Trade]:
        return [t for t in self.trades if t.side == "BUY"]

    @property
    def sells(self) -> list[Trade]:
        return [t for t in self.trades if t.side == "SELL"]

    @property
    def gross_traded(self) -> float:
        """Total weight changing hands: sum of |Δw| across both sides."""
        return float(sum(t.weight_delta for t in self.trades))

    @property
    def turnover(self) -> float:
        """
        One-way turnover -- the standard convention: gross traded / 2.

        A full replacement of the portfolio (sell everything, buy 20 new names)
        moves 200% of weight, which is 100% turnover.
        """
        return self.gross_traded / 2.0

    @property
    def total_cost(self) -> float:
        """
        Total cost of this rebalance, as a fraction of portfolio value.

        Subtract this directly from the period's return. This is the number that
        turns a gross backtest into a net one.
        """
        return float(sum(t.cost_weight for t in self.trades))

    @property
    def total_cost_bps(self) -> float:
        return self.total_cost * BPS

    def to_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(columns=["symbol", "side", "weight_delta", "cost_bps", "cost_weight"])
        return pd.DataFrame(
            [
                {
                    "symbol": t.symbol,
                    "side": t.side,
                    "weight_delta": t.weight_delta,
                    "cost_bps": t.cost_bps,
                    "cost_weight": t.cost_weight,
                }
                for t in self.trades
            ]
        )

    def __len__(self) -> int:
        return len(self.trades)

    def __repr__(self) -> str:
        return (
            f"<TradeList {self.market_id} @{self.as_of.date()}: "
            f"{len(self.buys)} buys, {len(self.sells)} sells, "
            f"turnover={self.turnover:.1%}, cost={self.total_cost_bps:.1f}bps>"
        )


def compute_trades(
    current_weights: pd.Series,
    target_weights: pd.Series,
    market: Market,
    as_of: pd.Timestamp,
    min_trade_weight: float = 0.0005,     # 5bps of the portfolio; below this, don't bother
) -> TradeList:
    """
    Work out what to trade to get from `current_weights` to `target_weights`.

    `current_weights` should be the DRIFTED weights (what you actually hold now
    after prices moved), not the weights you originally targeted.

    Costs come from the market config, and BUYS AND SELLS ARE PRICED SEPARATELY --
    because in India they genuinely differ (STT on sells, stamp duty on buys).
    """
    current = current_weights.astype(float)
    target = target_weights.astype(float)

    universe = sorted(set(current.index) | set(target.index))
    current = current.reindex(universe).fillna(0.0)
    target = target.reindex(universe).fillna(0.0)

    buy_bps = market.costs.buy_cost_bps()
    sell_bps = market.costs.sell_cost_bps()

    trades: list[Trade] = []

    for symbol in universe:
        delta = float(target[symbol] - current[symbol])

        # Don't trade noise. The friction exceeds the benefit.
        if abs(delta) < min_trade_weight:
            continue

        side: Side = "BUY" if delta > 0 else "SELL"
        rate = buy_bps if side == "BUY" else sell_bps
        size = abs(delta)

        trades.append(
            Trade(
                symbol=symbol,
                side=side,
                weight_delta=size,
                cost_bps=rate,
                cost_weight=size * rate / BPS,
            )
        )

    trades.sort(key=lambda t: (t.side, -t.weight_delta))

    return TradeList(market_id=market.market_id, as_of=pd.Timestamp(as_of), trades=trades)


def annual_cost_drag(turnover_per_period: float, periods_per_year: int, market: Market) -> float:
    """
    Rough annual performance drag from trading, as a decimal (0.019 = 1.9%/yr).

    A back-of-envelope sanity check, useful BEFORE running a full backtest:
    if the drag exceeds the edge you hope to capture, the strategy is dead on
    arrival and no amount of backtesting will save it.
    """
    round_trip = market.costs.round_trip_bps() / BPS
    return turnover_per_period * round_trip * periods_per_year
