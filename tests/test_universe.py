"""
Tests for the universe layer.

The subtle bug this file exists to prevent: an eligibility filter that decides
who was tradable in 2015 using data from 2016. It looks harmless and it silently
inflates every backtest, because you end up only ever holding companies that
turned out to be liquid and long-lived.
"""

import csv

import numpy as np
import pandas as pd
import pytest

from engine.data.base import PriceData
from engine.markets.market import load_market
from engine.universe.universe import (
    Member,
    Membership,
    eligible_universe,
    load_membership,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic prices we fully control
# ---------------------------------------------------------------------------

DATES = pd.bdate_range("2020-01-01", "2023-12-31")


def _series(n: int, start: float = 100.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return start * np.cumprod(1 + rng.normal(0.0004, 0.01, n))


@pytest.fixture
def market():
    return load_market("us")


@pytest.fixture
def prices(market) -> PriceData:
    n = len(DATES)

    close = pd.DataFrame(
        {
            "OLD": _series(n, seed=1),        # full history, liquid
            "ALSO_OLD": _series(n, seed=2),   # full history, liquid
            "NEW": _series(n, seed=3),        # listed recently
            "THIN": _series(n, seed=4),       # full history, but barely trades
        },
        index=DATES,
    )
    # NEW only starts trading in mid-2023
    close.loc[close.index < pd.Timestamp("2023-07-01"), "NEW"] = np.nan

    volume = pd.DataFrame(
        {
            "OLD": np.full(n, 2_000_000),
            "ALSO_OLD": np.full(n, 2_000_000),
            "NEW": np.full(n, 2_000_000),
            "THIN": np.full(n, 10),           # essentially untradable
        },
        index=DATES,
    )

    return PriceData(market=market, close=close, volume=volume)


@pytest.fixture
def membership(market) -> Membership:
    return Membership(
        market=market,
        universe_key="sp500",
        members=[
            Member("OLD", "Old Corp", "Tech"),
            Member("ALSO_OLD", "Also Old Inc", "Health"),
            Member("NEW", "Newly Listed Co", "Tech"),
            Member("THIN", "Thinly Traded Ltd", "Energy"),
            Member("MISSING", "No Data Co", "Utilities"),
        ],
        survivorship_bias=True,
        disclaimer="⚠️ survivorship bias",
    )


# ---------------------------------------------------------------------------
# Eligibility filters
# ---------------------------------------------------------------------------


def test_liquid_long_lived_names_are_eligible(prices, membership):
    snap = eligible_universe(prices, membership, "2023-01-03", min_history_days=252)

    assert "OLD" in snap.eligible
    assert "ALSO_OLD" in snap.eligible


def test_recently_listed_stock_is_excluded(prices, membership):
    """
    A stock with 3 months of history cannot have a 12-month momentum score.
    Including it would either crash the ranking or silently fabricate a number.
    """
    snap = eligible_universe(prices, membership, "2023-01-03", min_history_days=252)

    assert "NEW" not in snap.eligible
    assert snap.dropped["NEW"] == "insufficient_history"


def test_illiquid_stock_is_excluded(prices, membership):
    """A stock we cannot actually buy is not an investment opportunity."""
    snap = eligible_universe(prices, membership, "2023-01-03", min_history_days=252)

    assert "THIN" not in snap.eligible
    assert snap.dropped["THIN"] == "illiquid"


def test_symbol_with_no_data_is_excluded(prices, membership):
    snap = eligible_universe(prices, membership, "2023-01-03", min_history_days=252)

    assert "MISSING" not in snap.eligible
    assert snap.dropped["MISSING"] == "no_data"


def test_drop_reasons_are_reported(prices, membership):
    """We must be able to explain WHY the universe shrank -- silent drops hide bugs."""
    snap = eligible_universe(prices, membership, "2023-01-03", min_history_days=252)
    reasons = snap.drop_reasons()

    assert reasons["insufficient_history"] == 1
    assert reasons["illiquid"] == 1
    assert reasons["no_data"] == 1


# ---------------------------------------------------------------------------
# THE IMPORTANT ONE: eligibility must not see the future
# ---------------------------------------------------------------------------


def test_eligibility_does_not_peek_at_the_future(prices, membership):
    """
    NEW lists in July 2023.

    Asked in January 2023, the engine must NOT know that. If a future version of
    this filter looks at the whole price frame instead of history-up-to-date, NEW
    would appear eligible in January -- and the backtest would be buying a company
    before it was listed.
    """
    early = eligible_universe(prices, membership, "2023-01-03", min_history_days=60)
    assert "NEW" not in early.eligible, "look-ahead: saw a stock before it listed"

    # By late 2023 it has enough history and legitimately becomes eligible.
    later = eligible_universe(prices, membership, "2023-12-01", min_history_days=60)
    assert "NEW" in later.eligible


def test_universe_shrinks_as_we_go_back_in_time(prices, membership):
    """Sanity: earlier dates have fewer qualifying names, never more."""
    early = eligible_universe(prices, membership, "2021-01-04", min_history_days=252)
    late = eligible_universe(prices, membership, "2023-12-01", min_history_days=252)

    assert early.n_eligible <= late.n_eligible


# ---------------------------------------------------------------------------
# Membership loading
# ---------------------------------------------------------------------------


def test_membership_reads_symbol_name_sector(tmp_path, market, monkeypatch):
    csv_path = tmp_path / "data" / "universes" / "us_sp500.csv"
    csv_path.parent.mkdir(parents=True)

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["symbol", "name", "sector"])
        w.writerow(["AAPL", "Apple Inc.", "Information Technology"])
        w.writerow(["XOM", "Exxon Mobil", "Energy"])

    import engine.universe.universe as uni

    monkeypatch.setattr(uni, "REPO_ROOT", tmp_path)

    m = load_membership(market, "sp500")
    assert m.symbols == ["AAPL", "XOM"]
    assert m.sector_of("XOM") == "Energy"
    assert len(m) == 2


def test_missing_universe_file_fails_with_a_useful_message(market, monkeypatch, tmp_path):
    import engine.universe.universe as uni

    monkeypatch.setattr(uni, "REPO_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError, match="build_universes"):
        load_membership(market, "sp500")


def test_membership_carries_the_survivorship_disclaimer(membership):
    """SPEC.md §4.4 -- the bias must travel with the data, not be forgotten."""
    assert membership.survivorship_bias is True
    assert membership.disclaimer is not None
