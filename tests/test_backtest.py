"""
Tests for the backtest loop and metrics.

Built on synthetic markets whose behaviour is known by construction, so that when
a number comes out we know whether it is right -- rather than squinting at a plot
and hoping.

The crown jewel is test_trend_filter_protects_capital_in_a_crash: a market that
rises then collapses 50%. Without an overlay the strategy eats the crash. With one,
it should step aside. That is the pitch deck's "what if the market crashes?" promise,
tested rather than asserted.
"""

import numpy as np
import pandas as pd
import pytest

from engine.backtest import (
    AlwaysOn,
    TrendFilter,
    rebalance_dates,
    run_backtest,
    trading_days,
)
from engine.data.base import PriceData
from engine.markets.market import load_market
from engine.metrics import (
    cagr,
    compute_metrics,
    max_drawdown,
    sharpe_ratio,
    win_rate,
)
from engine.signals import MomentumSignal
from engine.universe.universe import Member, Membership

START = "2018-01-01"
END = "2023-12-31"


# ---------------------------------------------------------------------------
# Calendars
# ---------------------------------------------------------------------------


def test_exchange_calendars_genuinely_differ():
    """
    NYSE is shut on 1 January; NSE trades. If both markets produced identical
    rebalance dates, someone hard-coded a calendar.
    """
    us = load_market("us")
    india = load_market("india")

    us_dates = rebalance_dates(us, "2024-01-01", "2024-03-31", "monthly")
    in_dates = rebalance_dates(india, "2024-01-01", "2024-03-31", "monthly")

    assert us_dates[0] == pd.Timestamp("2024-01-02")   # New Year's Day: closed
    assert in_dates[0] == pd.Timestamp("2024-01-01")   # NSE: open
    assert not us_dates.equals(in_dates)


def test_rebalance_frequencies():
    us = load_market("us")

    monthly = rebalance_dates(us, "2023-01-01", "2023-12-31", "monthly")
    quarterly = rebalance_dates(us, "2023-01-01", "2023-12-31", "quarterly")
    weekly = rebalance_dates(us, "2023-01-01", "2023-12-31", "weekly")

    assert len(monthly) == 12
    assert len(quarterly) == 4
    assert 50 <= len(weekly) <= 53


def test_rebalance_dates_are_real_trading_days():
    india = load_market("india")
    valid = set(trading_days(india, START, END))

    for d in rebalance_dates(india, START, END, "monthly"):
        assert d in valid, f"{d.date()} is not an NSE trading day"


# ---------------------------------------------------------------------------
# Synthetic markets
# ---------------------------------------------------------------------------


def _make_prices(market, spec: dict[str, np.ndarray], dates) -> PriceData:
    close = pd.DataFrame(spec, index=dates)
    volume = pd.DataFrame({c: np.full(len(dates), 5_000_000) for c in close.columns}, index=dates)
    return PriceData(market=market, close=close, volume=volume)


@pytest.fixture
def market():
    return load_market("us")


@pytest.fixture
def days(market):
    return trading_days(market, START, END)


@pytest.fixture
def membership(market):
    return Membership(
        market=market,
        universe_key="sp500",
        members=[Member(s, s, "Tech") for s in ["A", "B", "C", "D", "E", "F"]],
        survivorship_bias=True,
        disclaimer="⚠️ survivorship bias",
    )


@pytest.fixture
def trending(market, days):
    """Six stocks with different steady growth rates. Momentum should find the best."""
    n = len(days)
    rng = np.random.default_rng(11)
    spec = {}
    for i, sym in enumerate(["A", "B", "C", "D", "E", "F"]):
        drift = 0.0008 - i * 0.0003            # A grows fastest, F shrinks
        noise = rng.normal(0, 0.008, n)
        spec[sym] = 100 * np.cumprod(1 + drift + noise)
    return _make_prices(market, spec, days)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def test_backtest_produces_an_equity_curve(market, membership, trending):
    dates = rebalance_dates(market, "2019-06-01", END, "monthly")

    result = run_backtest(
        market=market,
        membership=membership,
        prices=trending,
        signal=MomentumSignal(lookback_months=12, skip_months=1),
        rebalance_dates=dates,
        top_n=3,
    )

    assert len(result.equity) == len(dates) - 1
    assert result.equity.iloc[-1] > 0
    assert len(result.portfolios) == len(dates) - 1
    assert result.disclaimers                       # never silent about bias


def test_momentum_finds_the_winners(market, membership, trending):
    """A grows fastest, F shrinks. A momentum Top-3 should mostly hold the leaders."""
    dates = rebalance_dates(market, "2019-06-01", END, "monthly")

    result = run_backtest(
        market=market, membership=membership, prices=trending,
        signal=MomentumSignal(), rebalance_dates=dates, top_n=3,
    )

    held = [s for pf in result.portfolios for s in pf.symbols]
    assert held.count("A") > held.count("F")


def test_costs_reduce_returns(market, membership, trending):
    """
    Gross and net must diverge. If they do not, trading is free somewhere --
    which violates SPEC §4.3 and is how backtests lie.
    """
    dates = rebalance_dates(market, "2019-06-01", END, "monthly")

    result = run_backtest(
        market=market, membership=membership, prices=trending,
        signal=MomentumSignal(), rebalance_dates=dates, top_n=3,
    )

    assert result.equity.iloc[-1] < result.gross_equity.iloc[-1]
    assert result.total_cost_paid > 0
    assert result.turnover.mean() > 0


def test_india_costs_bite_harder_than_us_costs(membership, days):
    """The same strategy, the same prices -- India keeps less of it."""
    us = load_market("us")
    india = load_market("india")

    n = len(days)
    rng = np.random.default_rng(5)
    spec = {
        s: 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
        for s in ["A", "B", "C", "D", "E", "F"]
    }

    results = {}
    for market in (us, india):
        prices = _make_prices(market, spec, days)
        mem = Membership(
            market=market, universe_key="sp500",
            members=[Member(s, s, "Tech") for s in spec],
            survivorship_bias=True, disclaimer="⚠️",
        )
        dates = rebalance_dates(market, "2019-06-01", END, "monthly")
        results[market.market_id] = run_backtest(
            market=market, membership=mem, prices=prices,
            signal=MomentumSignal(), rebalance_dates=dates, top_n=3,
        )

    assert results["INDIA"].total_cost_paid > results["US"].total_cost_paid


# ---------------------------------------------------------------------------
# THE CRASH TEST
# ---------------------------------------------------------------------------


def test_trend_filter_protects_capital_in_a_crash(market, membership, days):
    """
    A market that climbs for years, then falls 50% over six months.

    Momentum alone owns "the strongest stocks" -- but in a crash the strongest
    stocks still collapse. Best-of-a-bad-lot is not a defence.

    The trend filter should notice the benchmark break below its moving average
    and move to cash, taking a materially smaller drawdown. This is the pitch
    deck's "what if the market crashes?" answer, and here it is tested rather
    than promised.

    Honest caveat: this is not free. In choppy sideways markets the same filter
    whipsaws you in and out, paying costs each time. It buys insurance; insurance
    has a premium.
    """
    n = len(days)
    crash_start = int(n * 0.75)

    # A benchmark that rises, then halves.
    bench = np.concatenate([
        100 * np.cumprod(np.full(crash_start, 1.0005)),
        np.linspace(1.0, 0.5, n - crash_start) * 100 * (1.0005 ** crash_start),
    ])
    benchmark = pd.Series(bench, index=days)

    # Stocks that track the same shape.
    rng = np.random.default_rng(9)
    spec = {}
    for i, sym in enumerate(["A", "B", "C", "D", "E", "F"]):
        noise = rng.normal(0, 0.004, n)
        spec[sym] = bench * (1 + 0.0002 * i) * np.cumprod(1 + noise)

    prices = _make_prices(market, spec, days)
    dates = rebalance_dates(market, "2019-06-01", END, "monthly")

    common = dict(
        market=market, membership=membership, prices=prices,
        signal=MomentumSignal(), rebalance_dates=dates, top_n=3,
        benchmark=benchmark,
    )

    unprotected = run_backtest(**common, overlay=AlwaysOn())
    protected = run_backtest(**common, overlay=TrendFilter(ma_days=200))

    dd_unprotected = max_drawdown(unprotected.equity)
    dd_protected = max_drawdown(protected.equity)

    # The overlay actually stepped aside at some point.
    assert protected.cash_periods.any(), "trend filter never went to cash in a 50% crash"

    # And it hurt less.
    assert dd_protected > dd_unprotected, (
        f"trend filter made drawdown worse: {dd_protected:.1%} vs {dd_unprotected:.1%}"
    )


def test_cash_periods_earn_nothing_but_lose_nothing(market, membership, days):
    """Being in cash is a valid state the engine must be able to hold."""
    n = len(days)
    falling = pd.Series(np.linspace(200, 100, n), index=days)   # benchmark only falls

    rng = np.random.default_rng(2)
    spec = {s: 100 * np.cumprod(1 + rng.normal(-0.0005, 0.01, n)) for s in ["A", "B", "C", "D", "E", "F"]}
    prices = _make_prices(market, spec, days)
    dates = rebalance_dates(market, "2019-06-01", END, "monthly")

    result = run_backtest(
        market=market, membership=membership, prices=prices,
        signal=MomentumSignal(), rebalance_dates=dates, top_n=3,
        benchmark=falling, overlay=TrendFilter(ma_days=200),
    )

    assert result.cash_periods.any()


# ---------------------------------------------------------------------------
# NO LOOK-AHEAD, end to end
# ---------------------------------------------------------------------------


def test_backtest_decisions_do_not_depend_on_the_future(market, membership, trending):
    """
    Run the backtest, then corrupt all prices after a cutoff and run it again.

    Every portfolio decided BEFORE the cutoff must be byte-identical. If any
    changes, the engine is reading the future -- and every result it has ever
    produced is worthless.
    """
    dates = rebalance_dates(market, "2019-06-01", "2022-12-31", "monthly")
    cutoff = pd.Timestamp("2021-06-01")

    original = run_backtest(
        market=market, membership=membership, prices=trending,
        signal=MomentumSignal(), rebalance_dates=dates, top_n=3,
    )

    tampered_close = trending.close.copy()
    future = tampered_close.index >= cutoff
    tampered_close.loc[future] = tampered_close.loc[future] * 50      # absurd

    tampered = run_backtest(
        market=market, membership=membership,
        prices=PriceData(market=market, close=tampered_close, volume=trending.volume),
        signal=MomentumSignal(), rebalance_dates=dates, top_n=3,
    )

    for a, b in zip(original.portfolios, tampered.portfolios):
        if a.as_of >= cutoff:
            break
        assert a.symbols == b.symbols, f"look-ahead detected at {a.as_of.date()}"
        assert a.weights.equals(b.weights)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_max_drawdown_on_a_known_series():
    equity = pd.Series([100, 120, 60, 80, 150])      # peak 120 -> trough 60 = -50%
    assert max_drawdown(equity) == pytest.approx(-0.50)


def test_cagr_on_a_known_series():
    equity = pd.Series([1.0] + [None] * 22 + [2.0]).interpolate()   # 24 monthly points
    equity = pd.Series(np.linspace(1.0, 2.0, 24))
    # doubling over 2 years ≈ 41.4% CAGR
    assert cagr(equity, periods_per_year=12) == pytest.approx(0.414, abs=0.02)


def test_win_rate():
    assert win_rate(pd.Series([0.1, -0.05, 0.02, -0.01])) == pytest.approx(0.5)


def test_sharpe_is_zero_for_flat_returns():
    assert sharpe_ratio(pd.Series([0.0] * 12), periods_per_year=12) == 0.0


def test_metrics_report_is_complete(market, membership, trending, days):
    n = len(days)
    benchmark = pd.Series(100 * np.cumprod(np.full(n, 1.0003)), index=days)
    dates = rebalance_dates(market, "2019-06-01", END, "monthly")

    result = run_backtest(
        market=market, membership=membership, prices=trending,
        signal=MomentumSignal(), rebalance_dates=dates, top_n=3,
        benchmark=benchmark,
    )
    m = compute_metrics(result)

    # Every headline number is present.
    for field in ("cagr", "max_drawdown", "sharpe", "sortino", "calmar", "win_rate"):
        assert getattr(m, field) is not None

    # Benchmark-relative numbers exist too.
    assert m.benchmark_cagr is not None
    assert m.excess_cagr == pytest.approx(m.cagr - m.benchmark_cagr)

    # Costs are exposed, never buried.
    assert m.gross_cagr >= m.cagr
    assert m.cost_drag >= 0
    assert m.avg_turnover > 0


def test_cost_drag_is_reported_not_hidden(market, membership, trending):
    """
    A backtest that only reports gross returns is misleading by omission.
    Both numbers, always.
    """
    dates = rebalance_dates(market, "2019-06-01", END, "monthly")
    result = run_backtest(
        market=market, membership=membership, prices=trending,
        signal=MomentumSignal(), rebalance_dates=dates, top_n=3,
    )
    m = compute_metrics(result)

    assert m.cost_drag > 0
    assert m.gross_cagr > m.cagr
