"""
Tests for point-in-time fundamentals.

Two bugs are prevented here, and both are silent killers.

THE THREE-MONTH TIME MACHINE. A December balance sheet is not public in December.
It is public in February or March. A screen that uses it on 2 January is reading a
document that does not exist. With prices, look-ahead is a one-day sin; with
fundamentals it is a NINETY-day one.

THE RESTATEMENT LAUNDRY. EDGAR holds the same period many times over, because
every 10-K repeats the prior two years -- and companies restate. If you take the
LATEST version you are trading on a number that was corrected after the fact. The
investor standing there at the time saw the ORIGINAL.
"""

import pandas as pd
import pytest

from engine.data.fundamentals import FundamentalData


def facts(rows) -> FundamentalData:
    return FundamentalData(
        facts=pd.DataFrame(
            rows,
            columns=["symbol", "concept", "period_end", "filed", "value", "form", "fiscal_year"],
        ),
        source="test",
    )


# ---------------------------------------------------------------------------
# THE TIME MACHINE
# ---------------------------------------------------------------------------


def test_cannot_see_a_figure_before_it_was_filed():
    """
    ACME's year ends 31 Dec 2023. The 10-K is filed 20 Feb 2024.

    On 2 January 2024 that number DID NOT EXIST. Anyone claiming to have screened
    on it is claiming to have read a document from the future.
    """
    data = facts([
        ["ACME", "revenue", "2022-12-31", "2023-02-20", 100.0, "10-K", 2022],
        ["ACME", "revenue", "2023-12-31", "2024-02-20", 150.0, "10-K", 2023],
    ])

    # 2 Jan 2024: the FY2023 figure has not been filed. We must see FY2022's.
    early = data.as_of("2024-01-02")
    assert early.loc["ACME", "revenue"] == 100.0, "read a filing from the future"

    # 1 March 2024: it is out. Now we may use it.
    later = data.as_of("2024-03-01")
    assert later.loc["ACME", "revenue"] == 150.0


def test_a_filing_is_invisible_on_the_very_day_it_is_filed():
    """Strictly before. We do not get to trade on this morning's 10-K at yesterday's close."""
    data = facts([
        ["ACME", "revenue", "2022-12-31", "2023-02-20", 100.0, "10-K", 2022],
        ["ACME", "revenue", "2023-12-31", "2024-02-20", 150.0, "10-K", 2023],
    ])

    on_the_day = data.as_of("2024-02-20")
    assert on_the_day.loc["ACME", "revenue"] == 100.0


def test_nothing_is_visible_before_the_first_filing():
    data = facts([
        ["ACME", "revenue", "2023-12-31", "2024-02-20", 150.0, "10-K", 2023],
    ])
    assert data.as_of("2020-01-01").empty


# ---------------------------------------------------------------------------
# THE RESTATEMENT LAUNDRY
# ---------------------------------------------------------------------------


def test_we_use_the_number_as_first_reported_not_the_restatement():
    """
    ACME reported FY2023 revenue of 150 in Feb 2024.
    In Feb 2025, the FY2024 10-K restates FY2023 down to 120 (an accounting error).

    An investor in March 2024 SAW 150. That is the number our backtest must use.
    Using 120 means our strategy knew about an error that had not been found yet.
    """
    data = facts([
        ["ACME", "revenue", "2023-12-31", "2024-02-20", 150.0, "10-K", 2023],   # original
        ["ACME", "revenue", "2023-12-31", "2025-02-20", 120.0, "10-K", 2024],   # restated
    ])

    # Only ONE fact survives -- the original.
    assert len(data.facts) == 1
    assert float(data.facts.iloc[0]["value"]) == 150.0

    # And it stays 150 even long after the restatement.
    assert data.as_of("2026-01-01").loc["ACME", "revenue"] == 150.0


def test_comparatives_do_not_corrupt_the_filing_lag():
    """
    The bug that made our spike report a 405-day median filing lag.

    Every 10-K repeats the prior years. If you naively measure filed-minus-period
    across ALL rows, the comparatives make the lag look like a year. Keeping only
    the first filing gives the TRUE lag, which is 30-90 days.
    """
    data = facts([
        ["ACME", "revenue", "2023-12-31", "2024-02-20", 150.0, "10-K", 2023],   # 51 days
        ["ACME", "revenue", "2023-12-31", "2025-02-20", 150.0, "10-K", 2024],   # comparative
        ["ACME", "revenue", "2023-12-31", "2026-02-20", 150.0, "10-K", 2025],   # comparative
    ])

    lag = data.filing_lag()
    assert len(lag) == 1
    assert lag.iloc[0] == 51           # NOT 417, NOT 782


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


def test_most_recent_available_period_is_used():
    """We want the freshest figure that was actually public -- not the oldest."""
    data = facts([
        ["ACME", "revenue", "2021-12-31", "2022-02-20", 90.0, "10-K", 2021],
        ["ACME", "revenue", "2022-12-31", "2023-02-20", 100.0, "10-K", 2022],
        ["ACME", "revenue", "2023-12-31", "2024-02-20", 150.0, "10-K", 2023],
    ])
    assert data.as_of("2023-06-01").loc["ACME", "revenue"] == 100.0


def test_multiple_symbols_and_concepts():
    data = facts([
        ["ACME", "revenue", "2023-12-31", "2024-02-20", 150.0, "10-K", 2023],
        ["ACME", "assets", "2023-12-31", "2024-02-20", 500.0, "10-K", 2023],
        ["BETA", "revenue", "2023-12-31", "2024-03-15", 80.0, "10-K", 2023],
    ])

    snapshot = data.as_of("2024-04-01")
    assert set(snapshot.index) == {"ACME", "BETA"}
    assert snapshot.loc["ACME", "assets"] == 500.0
    assert snapshot.loc["BETA", "revenue"] == 80.0

    # BETA filed later -- in early March, it is still invisible.
    partial = data.as_of("2024-03-01")
    assert "BETA" not in partial.index
    assert "ACME" in partial.index


def test_staleness_flags_a_company_that_stopped_filing():
    """A company that has not filed in over a year is a red flag, not a data point."""
    data = facts([
        ["ALIVE", "revenue", "2024-12-31", "2025-02-20", 10.0, "10-K", 2024],
        ["ZOMBIE", "revenue", "2021-12-31", "2022-02-20", 10.0, "10-K", 2021],
    ])

    stale = data.staleness("2025-06-01")
    assert stale["ALIVE"] < 150
    assert stale["ZOMBIE"] > 1000


def test_coverage_reports_gaps_honestly():
    data = facts([
        ["ACME", "revenue", "2023-12-31", "2024-02-20", 150.0, "10-K", 2023],
        ["ACME", "equity", "2023-12-31", "2024-02-20", 300.0, "10-K", 2023],
        ["BETA", "revenue", "2023-12-31", "2024-02-20", 80.0, "10-K", 2023],
    ])

    assert data.coverage("revenue") == 1.0
    assert data.coverage("equity") == 0.5        # BETA has none
    assert data.coverage("nonexistent") == 0.0


def test_missing_columns_are_rejected_loudly():
    with pytest.raises(ValueError, match="missing columns"):
        FundamentalData(facts=pd.DataFrame({"symbol": ["A"]}))
