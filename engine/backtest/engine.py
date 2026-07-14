"""
The backtest loop.

At each rebalance date D, in this exact order:

    1. Ask the overlay: should we be invested at all? (uses data < D)
    2. Work out who is eligible in the universe.        (uses data < D)
    3. Score them, rank, select top N, weight.          (uses data < D)
    4. Compare the target against what we ACTUALLY hold now (drifted weights).
    5. Trade the difference. PAY THE COSTS.
    6. Hold until the next rebalance date. Let prices move.

Everything in steps 1-3 is decided using only information that existed BEFORE D.
That is the whole ballgame. A backtest that peeks -- even by one day, even by
accident -- produces beautiful numbers that will never be repeated with real money.

Costs are charged at step 5, before the holding period return is earned. You pay to
get in, and only then do you find out whether it was worth it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from engine.backtest.calendar import periods_per_year
from engine.backtest.overlay import AlwaysOn, Overlay, OverlayDecision
from engine.costs import TradeList, compute_trades
from engine.data.base import PriceData
from engine.markets.market import Market
from engine.portfolio import (
    Portfolio,
    build_portfolio,
    cash_portfolio,
    drift_weights,
    select_with_buffer,
)
from engine.signals.base import Signal
from engine.universe.universe import Membership, eligible_universe


@dataclass
class BacktestResult:
    """Everything that happened, and enough detail to prove it."""

    market_id: str
    currency: str
    frequency: str

    dates: pd.DatetimeIndex
    equity: pd.Series                 # net of all costs. THE headline curve.
    gross_equity: pd.Series           # what it would have been with free trading
    period_returns: pd.Series         # net return each period
    costs: pd.Series                  # cost paid at each rebalance (fraction)
    turnover: pd.Series               # one-way turnover at each rebalance
    cash_periods: pd.Series           # bool: were we in cash this period?

    benchmark_equity: pd.Series | None = None
    benchmark_returns: pd.Series | None = None

    portfolios: list[Portfolio] = field(default_factory=list)
    trades: list[TradeList] = field(default_factory=list)
    decisions: list[OverlayDecision] = field(default_factory=list)

    disclaimers: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def periods_per_year(self) -> int:
        return periods_per_year(self.frequency)

    @property
    def total_cost_paid(self) -> float:
        """Cumulative drag from trading, as a fraction of starting capital."""
        return float(self.costs.sum())

    @property
    def cost_of_trading(self) -> float:
        """How much of the gross return trading friction ate."""
        if self.gross_equity.empty or self.equity.empty:
            return 0.0
        return float(self.gross_equity.iloc[-1] - self.equity.iloc[-1])

    def __repr__(self) -> str:
        if self.equity.empty:
            return "<BacktestResult empty>"
        return (
            f"<BacktestResult {self.market_id} {self.frequency}: "
            f"{len(self.dates)} periods, "
            f"equity {self.equity.iloc[0]:.2f} → {self.equity.iloc[-1]:.2f}>"
        )


def run_backtest(
    market: Market,
    membership: Membership,
    prices: PriceData,
    signal: Signal,
    rebalance_dates: pd.DatetimeIndex,
    top_n: int,
    frequency: str = "monthly",
    weighting: str = "equal",
    max_position_weight: float | None = None,
    max_sector_weight: float | None = None,
    overlay: Overlay | None = None,
    benchmark: pd.Series | None = None,
    min_trade_weight: float = 0.0005,
    exit_rank: int | None = None,
    initial_capital: float = 1.0,
) -> BacktestResult:
    """
    Walk history forward. Make decisions with yesterday's information only.
    """
    if len(rebalance_dates) < 2:
        raise ValueError(
            f"Need at least 2 rebalance dates to measure a return, got {len(rebalance_dates)}"
        )

    overlay = overlay or AlwaysOn()

    equity = initial_capital
    gross_equity = initial_capital

    current_weights = pd.Series(dtype=float)
    current_cash = 1.0

    eq_curve, gross_curve, rets, cost_series, turn_series, cash_flags = [], [], [], [], [], []
    portfolios: list[Portfolio] = []
    trade_lists: list[TradeList] = []
    decisions: list[OverlayDecision] = []

    dates = pd.DatetimeIndex(rebalance_dates).sort_values()

    for i in range(len(dates) - 1):
        d = dates[i]
        next_d = dates[i + 1]

        # ---- 1. Should we be invested at all? (data < d only) --------------
        decision = (
            overlay.decide(benchmark, d)
            if benchmark is not None
            else OverlayDecision(risk_on=True, reason="no_benchmark")
        )
        decisions.append(decision)

        # ---- 2-3. Build the target portfolio (data < d only) ---------------
        if not decision.risk_on:
            target = cash_portfolio(
                market.market_id, market.currency, d, reason=decision.reason
            )
        else:
            snapshot = eligible_universe(
                prices=prices,
                membership=membership,
                as_of=d,
                min_history_days=signal.required_history_days,
            )

            if not snapshot.eligible:
                target = cash_portfolio(
                    market.market_id, market.currency, d, reason="no_eligible_securities"
                )
            else:
                scores = signal.compute(prices, d, symbols=snapshot.eligible)

                if len(scores.valid) == 0:
                    target = cash_portfolio(
                        market.market_id, market.currency, d, reason="no_scoreable_securities"
                    )
                else:
                    # THE BUFFER: keep names that are still good enough, rather than
                    # selling anything that slipped one rank. Cuts turnover hard.
                    preselected = (
                        select_with_buffer(
                            signal_result=scores,
                            top_n=top_n,
                            current_symbols=list(current_weights.index),
                            exit_rank=exit_rank,
                        )
                        if exit_rank is not None
                        else None
                    )

                    target = build_portfolio(
                        signal_result=scores,
                        market=market,
                        top_n=top_n,
                        weighting=weighting,
                        membership=membership,
                        prices=prices,
                        max_position_weight=max_position_weight,
                        max_sector_weight=max_sector_weight,
                        preselected_symbols=preselected,
                    )

        portfolios.append(target)

        # ---- 4-5. Trade the difference, and PAY (SPEC §4.3) -----------------
        trades = compute_trades(
            current_weights=current_weights,
            target_weights=target.weights,
            market=market,
            as_of=d,
            min_trade_weight=min_trade_weight,
        )
        trade_lists.append(trades)

        cost = trades.total_cost
        equity *= (1.0 - cost)          # costs come out before the period is earned

        # ---- 6. Hold to the next rebalance date ----------------------------
        drifted, period_return = drift_weights(target, prices, next_d)

        equity *= (1.0 + period_return)
        gross_equity *= (1.0 + period_return)   # the same ride, but with free trading

        # What we actually hold going into the next decision.
        current_weights = drifted
        current_cash = max(0.0, 1.0 - float(drifted.sum())) if len(drifted) else 1.0

        eq_curve.append(equity)
        gross_curve.append(gross_equity)
        rets.append(period_return - cost)
        cost_series.append(cost)
        turn_series.append(trades.turnover)
        cash_flags.append(target.n_positions == 0)

    index = dates[1:]

    result = BacktestResult(
        market_id=market.market_id,
        currency=market.currency,
        frequency=frequency,
        dates=index,
        equity=pd.Series(eq_curve, index=index, dtype=float),
        gross_equity=pd.Series(gross_curve, index=index, dtype=float),
        period_returns=pd.Series(rets, index=index, dtype=float),
        costs=pd.Series(cost_series, index=index, dtype=float),
        turnover=pd.Series(turn_series, index=index, dtype=float),
        cash_periods=pd.Series(cash_flags, index=index, dtype=bool),
        portfolios=portfolios,
        trades=trade_lists,
        decisions=decisions,
        config={
            "signal": signal.name,
            "signal_params": dict(signal.params),
            "top_n": top_n,
            "weighting": weighting,
            "frequency": frequency,
            "overlay": overlay.name,
            "max_position_weight": max_position_weight,
            "max_sector_weight": max_sector_weight,
            "exit_rank": exit_rank,
            "universe": membership.universe_key,
            "universe_size": len(membership),
        },
        disclaimers=[
            d for d in [membership.disclaimer] if d
        ] + [
            "⚠️ Backtests are simulations, not predictions. Past performance does "
            "not indicate future results. Costs are modelled estimates, not actual fills."
        ],
    )

    if benchmark is not None:
        result.benchmark_equity = _benchmark_curve(benchmark, dates, initial_capital)
        if result.benchmark_equity is not None:
            result.benchmark_returns = result.benchmark_equity.pct_change().fillna(
                result.benchmark_equity.iloc[0] / initial_capital - 1.0
            )

    return result


def _benchmark_curve(
    benchmark: pd.Series, dates: pd.DatetimeIndex, initial_capital: float
) -> pd.Series | None:
    """Buy-and-hold the index over the same dates, for an apples-to-apples comparison."""
    clean = benchmark.dropna()
    if clean.empty:
        return None

    values = [clean.asof(d) for d in dates]
    series = pd.Series(values, index=dates, dtype=float).dropna()

    if series.empty or series.iloc[0] <= 0:
        return None

    normalized = series / series.iloc[0] * initial_capital
    return normalized.iloc[1:]      # align with the strategy curve
