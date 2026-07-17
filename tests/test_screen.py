"""
Tests for fundamental metrics and screens.

The two dangers under test:

1. TIME MACHINE ON RATIOS. Price-to-book mixes a price (known daily) with book value
   (known only from a filed 10-K). The ratio must use the last PUBLISHED book value,
   never a period-end figure that had not been filed yet.

2. SILENT UNIVERSE SHRINKAGE. A screen that quietly drops every name it lacks data
   for turns "S&P 500" into "S&P 500 companies with clean EDGAR data" -- a hidden
   bias. The screen must REPORT coverage, not hide it.
"""

import numpy as np
import pandas as pd
import pytest

from engine.data.base import PriceData
from engine.data.fundamentals import FundamentalData
from engine.markets.market import load_market
from engine.signals.fundamentals_metrics import compute_fundamental_metrics
from engine.signals.screen import (
    Screen,
    cheaper_than_median,
    get_screen,
    positive_earnings,
)

DATES = pd.bdate_range("2020-01-01", "2024-12-31")


def facts(rows) -> FundamentalData:
    return FundamentalData(
        facts=pd.DataFrame(
            rows,
            columns=["symbol", "concept", "period_end", "filed", "value", "form", "fiscal_year"],
        ),
        source="test",
    )


@pytest.fixture
def market():
    return load_market("us")


def make_prices(market, price_map: dict[str, float]) -> PriceData:
    n = len(DATES)
    close = pd.DataFrame({s: np.full(n, p) for s, p in price_map.items()}, index=DATES)
    volume = pd.DataFrame({s: np.full(n, 5_000_000) for s in price_map}, index=DATES)
    return PriceData(market=market, close=close, volume=volume)


# ---------------------------------------------------------------------------
# Point-in-time ratios
# ---------------------------------------------------------------------------


def test_price_to_book_uses_published_not_period_end(market):
    """
    ACME year ends 31 Dec 2022, book equity 100, shares 10 → BVPS 10.
    The 10-K is filed 20 Feb 2023. Price is 20 → P/B should be 2.0.

    On 15 Jan 2023, that book value HAD NOT BEEN FILED. So P/B must be NaN (or use
    an earlier filing), never computed from the not-yet-public 2022 equity.
    """
    prices = make_prices(market, {"ACME": 20.0})
    fund = facts([
        ["ACME", "equity", "2022-12-31", "2023-02-20", 100.0, "10-K", 2022],
        ["ACME", "shares", "2022-12-31", "2023-02-20", 10.0, "10-K", 2022],
        ["ACME", "net_income", "2022-12-31", "2023-02-20", 5.0, "10-K", 2022],
    ])

    # Before the filing: nothing published → no P/B.
    early = compute_fundamental_metrics(prices, fund, ["ACME"], "2023-01-15")
    assert pd.isna(early.loc["ACME", "price_to_book"]), "used un-filed book value"

    # After the filing: P/B = 20 / (100/10) = 2.0
    later = compute_fundamental_metrics(prices, fund, ["ACME"], "2023-03-01")
    assert later.loc["ACME", "price_to_book"] == pytest.approx(2.0)


def test_negative_book_value_gives_nan_not_a_bogus_ratio(market):
    """A company with negative equity has no meaningful P/B. NaN, not a negative."""
    prices = make_prices(market, {"ACME": 20.0})
    fund = facts([
        ["ACME", "equity", "2022-12-31", "2023-02-20", -50.0, "10-K", 2022],
        ["ACME", "shares", "2022-12-31", "2023-02-20", 10.0, "10-K", 2022],
    ])
    m = compute_fundamental_metrics(prices, fund, ["ACME"], "2023-03-01")
    assert pd.isna(m.loc["ACME", "price_to_book"])


def test_market_cap_is_price_times_shares(market):
    prices = make_prices(market, {"ACME": 50.0})
    fund = facts([
        ["ACME", "shares", "2022-12-31", "2023-02-20", 1_000_000.0, "10-K", 2022],
        ["ACME", "equity", "2022-12-31", "2023-02-20", 100.0, "10-K", 2022],
    ])
    m = compute_fundamental_metrics(prices, fund, ["ACME"], "2023-03-01")
    assert m.loc["ACME", "market_cap"] == pytest.approx(50_000_000.0)


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------


def test_positive_earnings_gate(market):
    prices = make_prices(market, {"WIN": 10.0, "LOSE": 10.0})
    fund = facts([
        ["WIN", "net_income", "2022-12-31", "2023-02-20", 5.0, "10-K", 2022],
        ["WIN", "equity", "2022-12-31", "2023-02-20", 50.0, "10-K", 2022],
        ["WIN", "shares", "2022-12-31", "2023-02-20", 10.0, "10-K", 2022],
        ["LOSE", "net_income", "2022-12-31", "2023-02-20", -8.0, "10-K", 2022],
        ["LOSE", "equity", "2022-12-31", "2023-02-20", 50.0, "10-K", 2022],
        ["LOSE", "shares", "2022-12-31", "2023-02-20", 10.0, "10-K", 2022],
    ])
    m = compute_fundamental_metrics(prices, fund, ["WIN", "LOSE"], "2023-03-01")
    screen = Screen("earnings", [positive_earnings()])
    result = screen.apply(m, pd.Timestamp("2023-03-01"))

    assert "WIN" in result.passed
    assert "LOSE" in result.failed


def test_cheaper_than_median_keeps_the_cheap_half(market):
    prices = make_prices(market, {"CHEAP": 10.0, "MID": 20.0, "RICH": 40.0})
    # Same book value per share (10) → P/B = 1, 2, 4. Median = 2.
    fund = facts([
        r for s, p in [("CHEAP", 10.0), ("MID", 20.0), ("RICH", 40.0)]
        for r in [
            [s, "equity", "2022-12-31", "2023-02-20", 100.0, "10-K", 2022],
            [s, "shares", "2022-12-31", "2023-02-20", 10.0, "10-K", 2022],
            [s, "net_income", "2022-12-31", "2023-02-20", 5.0, "10-K", 2022],
        ]
    ])
    m = compute_fundamental_metrics(prices, fund, ["CHEAP", "MID", "RICH"], "2023-03-01")
    screen = Screen("value", [cheaper_than_median("price_to_book")])
    result = screen.apply(m, pd.Timestamp("2023-03-01"))

    assert "CHEAP" in result.passed
    assert "MID" in result.passed        # at the median, inclusive
    assert "RICH" in result.failed


def test_screen_reports_coverage_honestly(market):
    """
    Two names have fundamentals, one does not. The screen must put the third in
    no_data and report coverage < 100% -- never silently drop it.
    """
    prices = make_prices(market, {"A": 10.0, "B": 10.0, "GHOST": 10.0})
    fund = facts([
        ["A", "equity", "2022-12-31", "2023-02-20", 100.0, "10-K", 2022],
        ["A", "shares", "2022-12-31", "2023-02-20", 10.0, "10-K", 2022],
        ["A", "net_income", "2022-12-31", "2023-02-20", 5.0, "10-K", 2022],
        ["B", "equity", "2022-12-31", "2023-02-20", 100.0, "10-K", 2022],
        ["B", "shares", "2022-12-31", "2023-02-20", 10.0, "10-K", 2022],
        ["B", "net_income", "2022-12-31", "2023-02-20", 5.0, "10-K", 2022],
        # GHOST: no fundamentals at all
    ])
    m = compute_fundamental_metrics(prices, fund, ["A", "B", "GHOST"], "2023-03-01")
    result = get_screen("value").apply(m, pd.Timestamp("2023-03-01"))

    assert "GHOST" in result.no_data
    assert result.coverage == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# THE TRAP: SVB / First Republic
# ---------------------------------------------------------------------------


def test_documents_the_value_trap_on_distressed_banks(market):
    """
    This does not test a fix -- there is none without better data. It DOCUMENTS the
    trap so no one forgets it.

    In early 2023 SVB traded at a low price-to-book. A value screen WOULD have bought
    it. Weeks later it was worth zero. Our price data cannot even represent that
    collapse (yfinance drops delisted names), so a value backtest silently skips the
    loss and overstates the strategy.

    Here: SIVB looks like a screaming value (P/B 0.5) and the screen dutifully passes
    it -- exactly the wrong call, made on data that looked fine at the time.
    """
    prices = make_prices(market, {"SIVB": 50.0, "SAFE": 50.0})
    fund = facts([
        # SIVB: huge book value, low P/B → looks cheap. (It was about to vanish.)
        ["SIVB", "equity", "2022-12-31", "2023-02-24", 1000.0, "10-K", 2022],
        ["SIVB", "shares", "2022-12-31", "2023-02-24", 10.0, "10-K", 2022],
        ["SIVB", "net_income", "2022-12-31", "2023-02-24", 15.0, "10-K", 2022],
        # SAFE: higher P/B
        ["SAFE", "equity", "2022-12-31", "2023-02-24", 250.0, "10-K", 2022],
        ["SAFE", "shares", "2022-12-31", "2023-02-24", 10.0, "10-K", 2022],
        ["SAFE", "net_income", "2022-12-31", "2023-02-24", 15.0, "10-K", 2022],
    ])
    m = compute_fundamental_metrics(prices, fund, ["SIVB", "SAFE"], "2023-03-01")
    result = get_screen("value").apply(m, pd.Timestamp("2023-03-01"))

    # The screen passes SIVB as "cheap and profitable" -- which is the trap.
    assert "SIVB" in result.passed
    # P/B really is lower (cheaper) for the doomed bank.
    assert m.loc["SIVB", "price_to_book"] < m.loc["SAFE", "price_to_book"]
