"""
Tests for costs, trades, and weight drift.

The headline test is test_india_cost_drag_is_brutal_at_high_turnover. It is not
really a unit test -- it is a warning, encoded. It asserts that the Indian
momentum research's ~32% monthly turnover costs well over 1% a year in India,
which is a large chunk of any edge you hope to earn.
"""

import numpy as np
import pandas as pd
import pytest

from engine.costs import TradeList, annual_cost_drag, compute_trades
from engine.data.base import PriceData
from engine.markets.market import load_market
from engine.portfolio import (
    Portfolio,
    Position,
    build_portfolio,
    cash_portfolio,
    drift_weights,
)
from engine.signals.base import SignalResult

AS_OF = pd.Timestamp("2024-07-01")


@pytest.fixture
def us():
    return load_market("us")


@pytest.fixture
def india():
    return load_market("india")


# ---------------------------------------------------------------------------
# Trade generation
# ---------------------------------------------------------------------------


def test_no_trades_when_nothing_changes(us):
    weights = pd.Series({"AAA": 0.5, "BBB": 0.5})
    trades = compute_trades(weights, weights, us, AS_OF)

    assert len(trades) == 0
    assert trades.turnover == 0.0
    assert trades.total_cost == 0.0


def test_full_replacement_is_one_hundred_percent_turnover(us):
    """
    Sell everything, buy an entirely new book. 200% of weight moves, which by the
    standard one-way convention is 100% turnover.
    """
    old = pd.Series({"AAA": 0.5, "BBB": 0.5})
    new = pd.Series({"CCC": 0.5, "DDD": 0.5})

    trades = compute_trades(old, new, us, AS_OF)

    assert trades.turnover == pytest.approx(1.0)
    assert trades.gross_traded == pytest.approx(2.0)
    assert len(trades.buys) == 2
    assert len(trades.sells) == 2


def test_partial_rotation(us):
    """Swap one name out of four: 25% of the book changes hands one-way."""
    old = pd.Series({"AAA": 0.25, "BBB": 0.25, "CCC": 0.25, "DDD": 0.25})
    new = pd.Series({"AAA": 0.25, "BBB": 0.25, "CCC": 0.25, "EEE": 0.25})

    trades = compute_trades(old, new, us, AS_OF)

    assert trades.turnover == pytest.approx(0.25)
    assert {t.symbol for t in trades.sells} == {"DDD"}
    assert {t.symbol for t in trades.buys} == {"EEE"}


def test_tiny_trades_are_skipped(us):
    """
    Rebalancing a holding by 0.02% costs more in friction than it fixes.
    Death by a thousand cuts is a real way to lose money.
    """
    old = pd.Series({"AAA": 0.5000, "BBB": 0.5000})
    new = pd.Series({"AAA": 0.5002, "BBB": 0.4998})

    trades = compute_trades(old, new, us, AS_OF, min_trade_weight=0.0005)
    assert len(trades) == 0

    # ...but a real trade still goes through.
    bigger = pd.Series({"AAA": 0.55, "BBB": 0.45})
    trades2 = compute_trades(old, bigger, us, AS_OF, min_trade_weight=0.0005)
    assert len(trades2) == 2


# ---------------------------------------------------------------------------
# EVERY TRADE PAYS (SPEC §4.3)
# ---------------------------------------------------------------------------


def test_every_trade_costs_something(us, india):
    old = pd.Series({"AAA": 1.0})
    new = pd.Series({"BBB": 1.0})

    for market in (us, india):
        trades = compute_trades(old, new, market, AS_OF)
        assert trades.total_cost > 0
        for t in trades.trades:
            assert t.cost_weight > 0, f"free trade in {market.market_id}"


def test_india_sells_cost_more_than_buys(india):
    """
    STT is levied on the SELL side in India. Stamp duty on the buy side is smaller.
    So selling genuinely costs more -- and the engine must reflect that rather than
    assuming one symmetric number.
    """
    old = pd.Series({"AAA": 1.0})
    new = pd.Series({"BBB": 1.0})

    trades = compute_trades(old, new, india, AS_OF)

    sell = trades.sells[0]
    buy = trades.buys[0]

    assert sell.cost_bps > buy.cost_bps
    assert sell.cost_weight > buy.cost_weight


def test_us_costs_are_symmetric(us):
    old = pd.Series({"AAA": 1.0})
    new = pd.Series({"BBB": 1.0})

    trades = compute_trades(old, new, us, AS_OF)
    assert trades.sells[0].cost_bps == pytest.approx(trades.buys[0].cost_bps)


def test_india_rebalance_costs_far_more_than_the_us(us, india):
    """Same trades, different country. The cost gap is roughly 5x."""
    old = pd.Series({"AAA": 0.5, "BBB": 0.5})
    new = pd.Series({"CCC": 0.5, "DDD": 0.5})

    us_cost = compute_trades(old, new, us, AS_OF).total_cost
    in_cost = compute_trades(old, new, india, AS_OF).total_cost

    assert in_cost > us_cost * 3


# ---------------------------------------------------------------------------
# THE WARNING, ENCODED
# ---------------------------------------------------------------------------


def test_india_cost_drag_is_brutal_at_high_turnover(india, us):
    """
    Raju & Chandrasekaran (2020) found a long-only NIFTY100 momentum strategy beat
    the index by ~10.7%/yr -- with ~32% MONTHLY turnover.

    At India's round-trip cost, that turnover burns well over 1% a year before the
    strategy earns a rupee. That does not kill a 10% edge, but it would comfortably
    kill a 2% one -- and it is exactly why honest cost modelling is not optional
    here. The same turnover in the US costs a fraction of that.
    """
    india_drag = annual_cost_drag(turnover_per_period=0.32, periods_per_year=12, market=india)
    us_drag = annual_cost_drag(turnover_per_period=0.32, periods_per_year=12, market=us)

    assert india_drag > 0.015, "India cost drag should exceed 1.5%/yr at 32% monthly turnover"
    assert india_drag > us_drag * 3

    # And weekly rebalancing would be self-harm.
    weekly = annual_cost_drag(turnover_per_period=0.32, periods_per_year=52, market=india)
    assert weekly > india_drag * 4


# ---------------------------------------------------------------------------
# Weight drift
# ---------------------------------------------------------------------------


@pytest.fixture
def prices(us):
    dates = pd.bdate_range("2024-01-01", "2024-12-31")
    n = len(dates)

    close = pd.DataFrame(
        {
            "WINNER": np.linspace(100, 200, n),     # doubles
            "LOSER": np.linspace(100, 50, n),       # halves
            "FLAT": np.full(n, 100.0),
        },
        index=dates,
    )
    volume = pd.DataFrame({c: np.full(n, 1e6) for c in close.columns}, index=dates)
    return PriceData(market=us, close=close, volume=volume)


def test_weights_drift_with_prices(prices):
    """
    Start equal-weighted. WINNER doubles, LOSER halves, FLAT does nothing.
    You traded nothing -- but you no longer hold what you thought you held.
    """
    pf = Portfolio(
        market_id="US",
        currency="USD",
        as_of=pd.Timestamp("2024-01-01"),
        positions=[
            Position("WINNER", 1 / 3),
            Position("LOSER", 1 / 3),
            Position("FLAT", 1 / 3),
        ],
    )

    drifted, period_return = drift_weights(pf, prices, pd.Timestamp("2024-12-31"))

    assert drifted["WINNER"] > 1 / 3
    assert drifted["LOSER"] < 1 / 3
    assert drifted.sum() == pytest.approx(1.0)

    # (2 + 0.5 + 1) / 3 - 1 ≈ 16.7%
    assert period_return == pytest.approx(0.1667, abs=0.01)


def test_drift_matters_for_turnover(prices, us):
    """
    THE POINT OF THIS MODULE.

    Rebalancing back to equal weight after drift requires REAL trades. A backtest
    that compares its new targets against the OLD targets (5%, 5%, ...) instead of
    the drifted reality sees no trades at all -- and reports zero cost for a
    rebalance that genuinely costs money.
    """
    pf = Portfolio(
        market_id="US",
        currency="USD",
        as_of=pd.Timestamp("2024-01-01"),
        positions=[
            Position("WINNER", 1 / 3),
            Position("LOSER", 1 / 3),
            Position("FLAT", 1 / 3),
        ],
    )

    drifted, _ = drift_weights(pf, prices, pd.Timestamp("2024-12-31"))
    target = pd.Series({"WINNER": 1 / 3, "LOSER": 1 / 3, "FLAT": 1 / 3})

    # Against the drifted reality: real trades, real cost.
    real = compute_trades(drifted, target, us, pd.Timestamp("2024-12-31"))
    assert real.turnover > 0.10
    assert real.total_cost > 0

    # Against the stale targets: the backtest would see nothing to do. Wrong.
    naive = compute_trades(pf.weights, target, us, pd.Timestamp("2024-12-31"))
    assert len(naive) == 0


def test_cash_portfolio_does_not_drift(prices, us):
    pf = cash_portfolio("US", "USD", pd.Timestamp("2024-01-01"))
    drifted, period_return = drift_weights(pf, prices, pd.Timestamp("2024-12-31"))

    assert drifted.empty
    assert period_return == 0.0


def test_drift_cannot_go_backwards(prices):
    pf = Portfolio(
        market_id="US",
        currency="USD",
        as_of=pd.Timestamp("2024-06-01"),
        positions=[Position("FLAT", 1.0)],
    )
    with pytest.raises(ValueError, match="backwards"):
        drift_weights(pf, prices, pd.Timestamp("2024-01-01"))


# ---------------------------------------------------------------------------
# End to end: portfolio -> trades
# ---------------------------------------------------------------------------


def test_first_rebalance_from_empty_is_full_investment(us):
    scores = SignalResult(
        name="volar",
        as_of=AS_OF,
        scores=pd.Series({"AAA": 3.0, "BBB": 2.0, "CCC": 1.0}),
    )
    pf = build_portfolio(scores, us, top_n=3)

    trades = compute_trades(pd.Series(dtype=float), pf.weights, us, AS_OF)

    assert len(trades.buys) == 3
    assert len(trades.sells) == 0
    assert trades.gross_traded == pytest.approx(1.0)
    assert trades.turnover == pytest.approx(0.5)


def test_tradelist_reports_itself(us):
    trades = compute_trades(
        pd.Series({"AAA": 1.0}), pd.Series({"BBB": 1.0}), us, AS_OF
    )
    frame = trades.to_frame()

    assert set(frame["side"]) == {"BUY", "SELL"}
    assert trades.total_cost_bps > 0
    assert "turnover" in repr(trades)
