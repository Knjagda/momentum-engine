"""
Tests for the EDGAR adapter's deterministic tag-priority dedup.

The bulk parser and the EDGAR adapter were silently disagreeing on equity for ~100
S&P 500 names, because when a company reports equity under two tags for the same
period -- StockholdersEquity (parent-only) and
StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest (total) --
neither had a deterministic rule for which wins. The bulk parser was fixed first;
this locks the SAME rule into EDGAR so the two sources agree.

We test _extract() directly with a synthetic SEC companyfacts payload -- no network.
"""

import pandas as pd
import pytest

from engine.data.edgar_adapter import EdgarAdapter
from engine.markets.market import load_market


def _payload(entries_by_tag: dict) -> dict:
    """Build a minimal SEC companyfacts-shaped payload from {tag: [entries]}."""
    return {
        "facts": {
            "us-gaap": {
                tag: {"units": {"USD": entries}}
                for tag, entries in entries_by_tag.items()
            }
        }
    }


@pytest.fixture
def adapter(tmp_path):
    # cache_dir in tmp so we never touch real cache; no network is used by _extract.
    return EdgarAdapter(
        market=load_market("us"),
        user_agent="momentum-engine test@realdomain.org",
        cache_dir=tmp_path,
    )


def test_parent_only_equity_wins_same_period(adapter):
    """
    Both equity tags report the SAME period. Parent-only StockholdersEquity (83.9B)
    must win over ...IncludingPortion... (90.3B).
    """
    payload = _payload({
        "StockholdersEquity": [
            {"end": "2024-01-31", "filed": "2024-03-01", "val": 83_900_000_000,
             "form": "10-Q", "fy": 2024},
        ],
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": [
            {"end": "2024-01-31", "filed": "2024-03-01", "val": 90_300_000_000,
             "form": "10-Q", "fy": 2024},
        ],
    })
    out = adapter._extract("WMT", payload, ["equity"])
    eq = out[out["concept"] == "equity"]
    assert len(eq) == 1
    assert eq.iloc[0]["value"] == 83_900_000_000


def test_lower_priority_tag_still_fills_missing_periods(adapter):
    """
    TRAP 2 preserved: if the preferred tag is ABSENT for a period but the secondary
    tag reports it, we still keep the secondary -- history stays unbroken.
    """
    payload = _payload({
        "StockholdersEquity": [
            {"end": "2024-01-31", "filed": "2024-03-01", "val": 83_900_000_000,
             "form": "10-Q", "fy": 2024},
        ],
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": [
            # a DIFFERENT, earlier period the primary tag never reported
            {"end": "2023-01-31", "filed": "2023-03-01", "val": 80_000_000_000,
             "form": "10-Q", "fy": 2023},
        ],
    })
    out = adapter._extract("WMT", payload, ["equity"])
    eq = out[out["concept"] == "equity"].sort_values("period_end")
    assert len(eq) == 2, "secondary tag should fill the period the primary lacks"
    assert set(eq["value"]) == {83_900_000_000, 80_000_000_000}


def test_first_filed_wins_on_restatement_same_tag(adapter):
    """Same tag, same period, original + restatement -> keep first-filed (no look-ahead)."""
    payload = _payload({
        "StockholdersEquity": [
            {"end": "2022-12-31", "filed": "2023-02-01", "val": 50_000_000_000,
             "form": "10-K", "fy": 2022},
            {"end": "2022-12-31", "filed": "2023-11-15", "val": 55_000_000_000,
             "form": "10-K", "fy": 2022},
        ],
    })
    out = adapter._extract("XYZ", payload, ["equity"])
    eq = out[out["concept"] == "equity"]
    assert len(eq) == 1
    assert eq.iloc[0]["value"] == 50_000_000_000, "should keep first-reported"


def test_empty_payload_returns_empty(adapter):
    out = adapter._extract("NADA", {"facts": {}}, ["equity"])
    assert out.empty
    assert list(out.columns) == ["symbol", "concept", "period_end", "filed", "value", "form", "fiscal_year"]
