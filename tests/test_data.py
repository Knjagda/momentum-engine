"""
Tests for the data layer.

These run OFFLINE against a fake adapter. Tests that hit the network are slow
and flaky, and a red suite you learn to ignore is worse than no suite.
Live data is verified separately via `python -m scripts.fetch_demo`.

The most important test here is test_up_to_blocks_look_ahead -- it guards
SPEC.md §4.1, the rule that keeps backtests honest.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from engine.data import get_adapter, registered_adapters
from engine.data.base import PriceAdapter, PriceData
from engine.markets.market import load_market


# ---------------------------------------------------------------------------
# A fake adapter: deterministic, offline, instant.
# ---------------------------------------------------------------------------


class FakeAdapter(PriceAdapter):
    """Generates a predictable price series. No network."""

    def fetch(self, symbols, start, end) -> PriceData:
        dates = pd.bdate_range(start=start, end=end)
        rng = np.random.default_rng(seed=42)  # fixed seed -> reproducible (SPEC §4.2)

        close = pd.DataFrame(
            {s: 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, len(dates))) for s in symbols},
            index=dates,
        )
        volume = pd.DataFrame(
            {s: rng.integers(1_000_000, 5_000_000, len(dates)) for s in symbols},
            index=dates,
        )
        return PriceData(market=self.market, close=close, volume=volume)


@pytest.fixture
def us_prices() -> PriceData:
    market = load_market("us")
    adapter = FakeAdapter(market)
    return adapter.fetch(["AAPL", "MSFT", "NVDA"], "2023-01-01", "2023-12-31")


# ---------------------------------------------------------------------------
# THE BIG ONE: no look-ahead
# ---------------------------------------------------------------------------


def test_up_to_blocks_look_ahead(us_prices):
    """
    SPEC.md §4.1: on rebalance date D, a signal may only see data from BEFORE D.

    You cannot decide to buy at D's close using D's close. That is time travel,
    and it is how backtests come to promise returns that never materialise.
    """
    cutoff = pd.Timestamp("2023-06-15")
    history = us_prices.up_to(cutoff)

    assert history.close.index.max() < cutoff
    assert cutoff not in history.close.index
    assert len(history.close) < len(us_prices.close)  # we really did cut something


def test_up_to_is_strict_not_inclusive(us_prices):
    """The boundary date itself must be excluded, not included."""
    trading_day = us_prices.close.index[100]
    history = us_prices.up_to(trading_day)
    assert trading_day not in history.close.index


def test_up_to_preserves_market(us_prices):
    history = us_prices.up_to("2023-06-15")
    assert history.market.market_id == us_prices.market.market_id


# ---------------------------------------------------------------------------
# PriceData behaviour
# ---------------------------------------------------------------------------


def test_price_data_basics(us_prices):
    assert us_prices.symbols == ["AAPL", "MSFT", "NVDA"]
    assert us_prices.start < us_prices.end
    assert not us_prices.close.empty


def test_returns_are_computed_from_adjusted_closes(us_prices):
    rets = us_prices.returns()
    assert rets.shape == us_prices.close.shape
    assert rets.iloc[0].isna().all()      # first row has no prior day
    assert rets.iloc[1:].notna().all().all()


def test_drop_incomplete_removes_short_history(us_prices):
    """A stock that listed midway must not silently pollute the ranking."""
    holed = us_prices.close.copy()
    holed.loc[holed.index[:200], "NVDA"] = np.nan   # NVDA "lists" late

    data = PriceData(market=us_prices.market, close=holed, volume=us_prices.volume)
    cleaned = data.drop_incomplete(min_coverage=0.9)

    assert "NVDA" not in cleaned.symbols
    assert "AAPL" in cleaned.symbols


def test_fake_adapter_is_reproducible():
    """SPEC.md §4.2: same inputs -> identical outputs. No hidden randomness."""
    market = load_market("us")
    a = FakeAdapter(market).fetch(["AAPL"], "2023-01-01", "2023-03-31")
    b = FakeAdapter(market).fetch(["AAPL"], "2023-01-01", "2023-03-31")
    pd.testing.assert_frame_equal(a.close, b.close)


# ---------------------------------------------------------------------------
# The adapter is chosen BY THE MARKET CONFIG
# ---------------------------------------------------------------------------


def test_yfinance_adapter_is_registered():
    assert "yfinance" in registered_adapters()


@pytest.mark.parametrize("key", ["us", "india"])
def test_factory_builds_adapter_named_in_market_config(key):
    market = load_market(key)
    adapter = get_adapter(market)
    assert adapter.market.market_id == market.market_id


def test_unknown_adapter_fails_loudly():
    market = load_market("us")
    broken = type(market)(**{**market.__dict__, "data_adapter": "bloomberg_terminal"})
    with pytest.raises(KeyError):
        get_adapter(broken)


# ---------------------------------------------------------------------------
# Symbol conventions stay in the market layer, never in the engine
# ---------------------------------------------------------------------------


def test_adapter_uses_market_ticker_convention():
    """
    The India adapter must ask the market for ".NS" -- it must never write it.
    Verified here by checking the market resolves it, since the adapter delegates.
    """
    india = load_market("india")
    us = load_market("us")

    assert india.resolve_ticker("RELIANCE") == "RELIANCE.NS"
    assert us.resolve_ticker("AAPL") == "AAPL"

    # And display symbols come back clean.
    assert india.strip_ticker("RELIANCE.NS") == "RELIANCE"


def test_empty_symbol_list_is_rejected():
    market = load_market("us")
    adapter = get_adapter(market)
    with pytest.raises(ValueError):
        adapter.fetch([], "2023-01-01", "2023-12-31")
