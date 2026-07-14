"""
Performance metrics.

The numbers people actually look at -- and a few they should look at but usually
do not.

A note on honesty. It is trivially easy to report a headline CAGR and let people
assume it is achievable. It is not: a 20% CAGR with a -55% drawdown is a strategy
almost nobody can actually hold, because they will capitulate at the bottom. That
is why max drawdown sits beside CAGR here rather than in a footnote, and why we
report gross AND net returns so trading costs cannot hide.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def total_return(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0]) - 1.0


def cagr(equity: pd.Series, periods_per_year: int) -> float:
    """Compound annual growth rate. The number everyone quotes."""
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0

    years = len(equity) / periods_per_year
    if years <= 0:
        return 0.0

    growth = float(equity.iloc[-1] / equity.iloc[0])
    if growth <= 0:
        return -1.0

    return growth ** (1.0 / years) - 1.0


def max_drawdown(equity: pd.Series) -> float:
    """
    Worst peak-to-trough fall, as a negative number.

    THE most important number in the report, and the one investors feel. A strategy
    with a -55% drawdown is one most people will abandon at exactly the wrong moment,
    which turns a paper gain into a realised loss.
    """
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    if equity.empty:
        return equity
    return equity / equity.cummax() - 1.0


def volatility(returns: pd.Series, periods_per_year: int) -> float:
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    return float(clean.std(ddof=1)) * np.sqrt(periods_per_year)


def sharpe_ratio(returns: pd.Series, periods_per_year: int, risk_free: float = 0.0) -> float:
    """Excess return per unit of TOTAL volatility (upside and downside alike)."""
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0

    excess = clean - (risk_free / periods_per_year)
    sd = float(excess.std(ddof=1))
    if sd == 0:
        return 0.0

    return float(excess.mean()) / sd * np.sqrt(periods_per_year)


def sortino_ratio(returns: pd.Series, periods_per_year: int, risk_free: float = 0.0) -> float:
    """
    Like Sharpe, but only penalises DOWNSIDE volatility.

    Sharpe punishes a strategy for going up violently, which is a strange thing to
    punish. Sortino does not.
    """
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0

    excess = clean - (risk_free / periods_per_year)
    downside = excess[excess < 0]

    if len(downside) < 2:
        return float("inf") if excess.mean() > 0 else 0.0

    dd = float(downside.std(ddof=1))
    if dd == 0:
        return 0.0

    return float(excess.mean()) / dd * np.sqrt(periods_per_year)


def calmar_ratio(equity: pd.Series, periods_per_year: int) -> float:
    """CAGR divided by max drawdown. Return per unit of PAIN."""
    dd = abs(max_drawdown(equity))
    if dd == 0:
        return 0.0
    return cagr(equity, periods_per_year) / dd


def win_rate(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    return float((clean > 0).mean())


# ---------------------------------------------------------------------------
# Benchmark-relative
# ---------------------------------------------------------------------------


def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0

    var = float(aligned.iloc[:, 1].var(ddof=1))
    if var == 0:
        return 0.0

    cov = float(aligned.cov().iloc[0, 1])
    return cov / var


def alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int,
    risk_free: float = 0.0,
) -> float:
    """Annualized excess return after adjusting for market exposure."""
    b = beta(returns, benchmark_returns)
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if aligned.empty:
        return 0.0

    rf_period = risk_free / periods_per_year
    strat = float(aligned.iloc[:, 0].mean()) - rf_period
    bench = float(aligned.iloc[:, 1].mean()) - rf_period

    return (strat - b * bench) * periods_per_year


def tracking_error(returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int) -> float:
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0
    diff = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return float(diff.std(ddof=1)) * np.sqrt(periods_per_year)


def information_ratio(
    returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int
) -> float:
    te = tracking_error(returns, benchmark_returns, periods_per_year)
    if te == 0:
        return 0.0

    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    diff = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return float(diff.mean()) * periods_per_year / te


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


@dataclass
class Metrics:
    """A full performance summary. Includes the numbers that do not flatter."""

    total_return: float
    cagr: float
    volatility: float
    max_drawdown: float
    sharpe: float
    sortino: float
    calmar: float
    win_rate: float

    avg_turnover: float = 0.0
    total_costs: float = 0.0
    cost_drag: float = 0.0          # gross CAGR minus net CAGR
    gross_cagr: float = 0.0
    periods_in_cash: float = 0.0

    benchmark_cagr: float | None = None
    benchmark_max_drawdown: float | None = None
    benchmark_sharpe: float | None = None
    excess_cagr: float | None = None
    alpha: float | None = None
    beta: float | None = None
    information_ratio: float | None = None

    n_periods: int = 0
    years: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def compute_metrics(result) -> Metrics:
    """Build the full metric set from a BacktestResult."""
    ppy = result.periods_per_year
    equity = result.equity
    returns = result.period_returns

    m = Metrics(
        total_return=total_return(equity),
        cagr=cagr(equity, ppy),
        volatility=volatility(returns, ppy),
        max_drawdown=max_drawdown(equity),
        sharpe=sharpe_ratio(returns, ppy),
        sortino=sortino_ratio(returns, ppy),
        calmar=calmar_ratio(equity, ppy),
        win_rate=win_rate(returns),
        avg_turnover=float(result.turnover.mean()) if len(result.turnover) else 0.0,
        total_costs=float(result.costs.sum()) if len(result.costs) else 0.0,
        gross_cagr=cagr(result.gross_equity, ppy),
        periods_in_cash=float(result.cash_periods.mean()) if len(result.cash_periods) else 0.0,
        n_periods=len(equity),
        years=len(equity) / ppy if ppy else 0.0,
    )

    m.cost_drag = m.gross_cagr - m.cagr

    if result.benchmark_equity is not None and result.benchmark_returns is not None:
        bench_eq = result.benchmark_equity
        bench_ret = result.benchmark_returns

        m.benchmark_cagr = cagr(bench_eq, ppy)
        m.benchmark_max_drawdown = max_drawdown(bench_eq)
        m.benchmark_sharpe = sharpe_ratio(bench_ret, ppy)
        m.excess_cagr = m.cagr - m.benchmark_cagr
        m.alpha = alpha(returns, bench_ret, ppy)
        m.beta = beta(returns, bench_ret)
        m.information_ratio = information_ratio(returns, bench_ret, ppy)

    return m
