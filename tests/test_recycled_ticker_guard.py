"""
Regression tests: a recycled ticker must never produce a fabricated return.

THE HAZARD. A ticker is a slot, not an identity. Dean Foods was DF until its 2020
bankruptcy; another company holds DF now. A vendor that returns "whatever owns this
symbol" splices two companies into one series. If the engine ranks across that
handoff, a bankrupt $0.50 stock "becoming" a $30 stock books a 6,000% gain that
never happened.

THE PROTECTION, in layers:
  1. Point-in-time membership -- a symbol is only held inside its membership
     interval, and a reissued symbol's bars fall outside it. (covered elsewhere)
  2. min_history_days -- a stub of a few bars cannot be ranked.
  3. max_history_gap_days -- enough TOTAL bars is not enough; the recent window the
     signal reads must be CONTIGUOUS. This is what catches a splice that sneaks past
     the bar count.
  4. max_staleness_days -- history that stopped long ago cannot be traded on today.

Layers 3 and 4 are what these tests lock down. Without them, a spliced series with
285 old bars plus 4 new ones clears a 252-bar threshold and gets ranked across a
five-year dormancy.
"""

import numpy as np
import pandas as pd
import pytest

from engine.data.base import PriceData
from engine.markets.market import load_market
from engine.universe.universe import Member, Membership, eligible_universe

DECISION_DATE = pd.Timestamp("2018-06-01")
MIN_HISTORY = 252


@pytest.fixture
def market():
    return load_market("us")


def _bars(start: str, periods: int, price: float) -> pd.Series:
    idx = pd.bdate_range(start, periods=periods)
    return pd.Series(np.full(periods, price), index=idx)


@pytest.fixture
def prices(market) -> PriceData:
    """
    CLEAN   -- continuous bars right up to the decision date.
    SPLICE  -- 285 bars in 2012-13 at $50, then a 5-year dormancy, then 4 bars in
               2018 at $500 (a different company holding the recycled symbol).
               Total bars (289) clears MIN_HISTORY, so only a gap check stops it.
    STALE   -- 400 continuous bars that stop in 2013, long before the decision date.
    """
    clean = _bars("2016-01-04", 620, 100.0)

    old_era = _bars("2012-08-29", 285, 50.0)          # company A
    new_era = _bars("2018-05-25", 4, 500.0)           # company B, recycled symbol
    splice = pd.concat([old_era, new_era])

    stale = _bars("2011-06-01", 400, 75.0)            # ends mid-2013

    idx = clean.index.union(splice.index).union(stale.index)
    close = pd.DataFrame(
        {
            "CLEAN": clean.reindex(idx),
            "SPLICE": splice.reindex(idx),
            "STALE": stale.reindex(idx),
        },
        index=idx,
    )
    volume = pd.DataFrame(
        {c: np.where(close[c].notna(), 5_000_000, np.nan) for c in close.columns},
        index=idx,
    )
    return PriceData(market=market, close=close, volume=volume)


@pytest.fixture
def membership(market) -> Membership:
    # All three are members across the whole span -- so ONLY the price-side guards
    # can save us. This is the pessimistic case: imperfect membership metadata.
    return Membership(
        market=market,
        universe_key="test",
        members=[
            Member("CLEAN", "Clean Co", "Tech"),
            Member("SPLICE", "Recycled Ticker Co", "Tech"),
            Member("STALE", "Long Dead Co", "Energy"),
        ],
        survivorship_bias=False,
        disclaimer=None,
    )


def _snapshot(prices, membership, **kw):
    return eligible_universe(
        prices=prices,
        membership=membership,
        as_of=DECISION_DATE,
        min_history_days=MIN_HISTORY,
        **kw,
    )


def test_clean_symbol_is_eligible(prices, membership):
    snap = _snapshot(prices, membership)
    assert "CLEAN" in snap.eligible


def test_spliced_ticker_is_rejected(prices, membership):
    """The core protection: a recycled-ticker splice must not be rankable."""
    snap = _snapshot(prices, membership)
    assert "SPLICE" not in snap.eligible, (
        "a series with a multi-year dormancy was ranked -- a return computed across "
        "that gap would be fabricated"
    )
    assert snap.dropped.get("SPLICE") in {"history_gap", "insufficient_history"}


def test_stale_prices_are_rejected(prices, membership):
    """Plenty of bars, but they stopped years ago -- not tradeable today."""
    snap = _snapshot(prices, membership)
    assert "STALE" not in snap.eligible
    assert snap.dropped.get("STALE") == "stale_prices"


def test_splice_would_pass_a_naive_bar_count(prices):
    """
    Documents WHY the gap check is needed: the spliced series has more than
    min_history_days bars, so a count-only filter lets it through.
    """
    closes = prices.close["SPLICE"].dropna()
    assert len(closes) >= MIN_HISTORY, "fixture no longer exercises the hazard"


def test_disabling_the_gap_check_lets_the_splice_through(prices, membership):
    """
    The guard is doing the work -- turn it off and the hazard returns. This is what
    makes the test meaningful rather than incidental.
    """
    snap = _snapshot(
        prices, membership, max_history_gap_days=0, max_staleness_days=0
    )
    assert "SPLICE" in snap.eligible, (
        "with the guard disabled the splice should be eligible; if it is not, this "
        "test is passing for the wrong reason"
    )
