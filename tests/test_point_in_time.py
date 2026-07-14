"""
Tests for POINT-IN-TIME membership.

The bug these prevent is subtle, invisible, and devastating.

If you backtest 2013 using TODAY's S&P 500 list, your universe contains NVDA, PLTR,
APP, VST, CVNA -- companies that joined the index precisely BECAUSE they went up
enormously. Momentum's entire job is to find the biggest risers, and you have
guaranteed they are all sitting in the room before it starts looking.

That is not survivorship bias (missing losers). It is INCLUSION bias (guaranteed
winners), and for a momentum strategy it is far more dangerous, because it is aimed
squarely at what the strategy hunts.
"""

import numpy as np
import pandas as pd
import pytest

from engine.data.base import PriceData
from engine.markets.market import load_market
from engine.signals import MomentumSignal
from engine.universe.universe import (
    Member,
    Membership,
    eligible_universe,
)

DATES = pd.bdate_range("2010-01-01", "2024-12-31")


@pytest.fixture
def market():
    return load_market("us")


@pytest.fixture
def membership(market):
    """
    Three companies:
      OLDTIMER  in the index the whole time
      LATECOMER joined in 2020 -- AFTER it had already gone up 10x
      FALLEN    was a member until 2015, then removed
    """
    return Membership(
        market=market,
        universe_key="sp500_pit",
        members=[
            Member("OLDTIMER", "Old Timer Co", "Industrials",
                   added=None, removed=None),
            Member("LATECOMER", "Latecomer Inc", "Information Technology",
                   added=pd.Timestamp("2020-06-01"), removed=None),
            Member("FALLEN", "Fallen Corp", "Energy",
                   added=None, removed=pd.Timestamp("2015-03-01")),
        ],
        survivorship_bias=False,
        disclaimer=None,
    )


@pytest.fixture
def prices(market):
    """LATECOMER is the rocket -- exactly the stock momentum would want to own."""
    n = len(DATES)
    close = pd.DataFrame(
        {
            "OLDTIMER": np.linspace(100, 200, n),          # steady, doubles
            "LATECOMER": np.linspace(10, 1000, n),         # 100x. A momentum magnet.
            "FALLEN": np.linspace(100, 120, n),
        },
        index=DATES,
    )
    volume = pd.DataFrame({c: np.full(n, 5_000_000) for c in close.columns}, index=DATES)
    return PriceData(market=market, close=close, volume=volume)


# ---------------------------------------------------------------------------
# THE INCLUSION-BIAS TEST
# ---------------------------------------------------------------------------


def test_cannot_buy_a_company_before_it_joined_the_index(prices, membership):
    """
    THE WHOLE POINT.

    In 2013, LATECOMER was rising fast -- but it was NOT in the S&P 500. A 2013
    investor could not have held it in an S&P 500 strategy, because it was not
    there to hold.

    Without point-in-time membership, the backtest cheerfully buys it, earns its
    100x, and reports a spectacular CAGR that no human being could have achieved.
    """
    snap = eligible_universe(
        prices=prices,
        membership=membership,
        as_of=pd.Timestamp("2013-01-02"),
        min_history_days=252,
    )

    assert "LATECOMER" not in snap.eligible, "bought a stock before it joined the index"
    assert "OLDTIMER" in snap.eligible


def test_can_buy_it_once_it_actually_joined(prices, membership):
    """After June 2020 it is genuinely a member, and genuinely investable."""
    snap = eligible_universe(
        prices=prices,
        membership=membership,
        as_of=pd.Timestamp("2021-01-04"),
        min_history_days=252,
    )
    assert "LATECOMER" in snap.eligible


def test_removed_company_is_investable_before_removal_only(prices, membership):
    """
    FALLEN was a real member until 2015. A 2013 backtest SHOULD be able to hold it --
    and should take the consequences.

    This is the other half of honesty: not only must we avoid buying tomorrow's
    winners early, we must also be exposed to yesterday's losers.
    """
    before = eligible_universe(
        prices, membership, pd.Timestamp("2013-01-02"), min_history_days=252
    )
    after = eligible_universe(
        prices, membership, pd.Timestamp("2016-01-04"), min_history_days=252
    )

    assert "FALLEN" in before.eligible
    assert "FALLEN" not in after.eligible


def test_membership_as_of_shrinks_the_universe_in_the_past(membership):
    early = membership.as_of("2013-01-02")
    late = membership.as_of("2024-01-02")

    assert "LATECOMER" not in early.symbols
    assert "LATECOMER" in late.symbols
    assert "FALLEN" in early.symbols
    assert "FALLEN" not in late.symbols


def test_point_in_time_is_detected(membership, market):
    assert membership.is_point_in_time is True

    flat = Membership(
        market=market,
        universe_key="sp500",
        members=[Member("AAA"), Member("BBB")],
        survivorship_bias=True,
        disclaimer="⚠️",
    )
    assert flat.is_point_in_time is False

    # A flat snapshot cannot be time-travelled -- as_of returns it unchanged,
    # and the bias stands. We do not pretend otherwise.
    assert flat.as_of("2013-01-01").symbols == flat.symbols


# ---------------------------------------------------------------------------
# The bias, measured
# ---------------------------------------------------------------------------


def test_inclusion_bias_inflates_the_signal(prices, membership, market):
    """
    Run the same momentum signal in 2013 with and without point-in-time membership.

    WITH today's list: the top pick is LATECOMER, the 100x rocket -- a stock the
    strategy could not possibly have owned.

    WITH point-in-time: it is not even in the universe.

    That single difference is worth many percent a year of fictional CAGR.
    """
    as_of = pd.Timestamp("2013-01-02")
    signal = MomentumSignal(lookback_months=12, skip_months=1)

    # Naive: today's membership, no dates. This is what we shipped in Commit 4.
    naive = Membership(
        market=market,
        universe_key="sp500",
        members=[Member(m.symbol, m.name, m.sector) for m in membership.members],
        survivorship_bias=True,
        disclaimer="⚠️",
    )

    naive_snap = eligible_universe(prices, naive, as_of, min_history_days=252)
    pit_snap = eligible_universe(prices, membership, as_of, min_history_days=252)

    naive_scores = signal.compute(prices, as_of, symbols=naive_snap.eligible)
    pit_scores = signal.compute(prices, as_of, symbols=pit_snap.eligible)

    # The naive universe hands momentum the rocket.
    assert naive_scores.valid.idxmax() == "LATECOMER"

    # Point-in-time does not.
    assert "LATECOMER" not in pit_scores.valid.index
    assert pit_scores.valid.idxmax() != "LATECOMER"
