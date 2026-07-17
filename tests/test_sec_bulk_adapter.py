"""
Tests for the SEC bulk adapter (piece 3), offline.

We pre-seed the cache with synthetic quarter ZIPs and a fake CIK map, then check the
adapter combines them correctly behind the standard interface:
  - returns FundamentalData the engine accepts,
  - keeps first-reported values ACROSS quarters (not just within one),
  - filters to requested symbols,
  - inverts the ticker->CIK map correctly,
  - never needs the network when quarters are already cached.
"""

import json
import zipfile

import pytest

from engine.data import get_fundamental_adapter
from engine.data.fundamentals import FundamentalData
from engine.data.sec_bulk_adapter import SecBulkAdapter


def _write_quarter(cache_dir, quarter, rows):
    """rows: list of (adsh, cik, form, period, filed, fy, tag, ddate, value)."""
    sub_cols = "adsh\tcik\tform\tperiod\tfiled\tfy"
    num_cols = "adsh\ttag\tversion\tddate\tqtrs\tuom\tsegments\tcoreg\tvalue"
    subs, nums, seen = [], [], set()
    for adsh, cik, form, period, filed, fy, tag, ddate, value in rows:
        if adsh not in seen:
            subs.append(f"{adsh}\t{cik}\t{form}\t{period}\t{filed}\t{fy}")
            seen.add(adsh)
        nums.append(f"{adsh}\t{tag}\tus-gaap\t{ddate}\t0\tUSD\t\t\t{value}")
    p = cache_dir / f"{quarter}.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("sub.txt", sub_cols + "\n" + "\n".join(subs))
        zf.writestr("num.txt", num_cols + "\n" + "\n".join(nums))
    return p


@pytest.fixture
def seeded(tmp_path):
    """Two cached quarters and a CIK map, all offline."""
    cache = tmp_path / "raw"
    cache.mkdir()
    cik_map = tmp_path / "cik_map.json"
    cik_map.write_text(json.dumps({"AAPL": 320193, "MSFT": 789019}))

    # 2024q1: AAPL equity 50000 for period 2023-12-30, filed 2024-02-01
    _write_quarter(cache, "2024q1", [
        ("a1", 320193, "10-K", "20231230", "20240201", "2023",
         "StockholdersEquity", "20231230", "50000"),
        ("m1", 789019, "10-Q", "20231231", "20240125", "2024",
         "StockholdersEquity", "20231231", "60000"),
    ])
    # 2024q2: a RESTATEMENT of AAPL's same 2023-12-30 equity, filed later. Must NOT win.
    _write_quarter(cache, "2024q2", [
        ("a2", 320193, "10-K/A", "20231230", "20240510", "2023",
         "StockholdersEquity", "20231230", "55000"),
    ])
    return cache, cik_map


def _adapter(seeded):
    cache, cik_map = seeded
    return SecBulkAdapter(
        email="me@example.com",
        start_quarter="2024q1",
        end_quarter="2024q2",
        cache_dir=cache,
        cik_map_path=cik_map,
    )


def test_returns_fundamental_data(seeded):
    fd = _adapter(seeded).fetch(["AAPL", "MSFT"])
    assert isinstance(fd, FundamentalData)
    assert set(fd.symbols) == {"AAPL", "MSFT"}


def test_first_reported_wins_across_quarters(seeded):
    """AAPL equity restated in 2024q2 must NOT override the original from 2024q1."""
    fd = _adapter(seeded).fetch(["AAPL"])
    # as_of well after both filings -- should still see the ORIGINAL 50000
    snap = fd.as_of("2024-07-01", concepts=["equity"])
    assert snap.loc["AAPL", "equity"] == 50000


def test_symbol_filter(seeded):
    fd = _adapter(seeded).fetch(["MSFT"])
    assert set(fd.symbols) == {"MSFT"}
    assert "AAPL" not in fd.symbols


def test_point_in_time_respected(seeded):
    """Before AAPL's filing date, its equity is invisible."""
    fd = _adapter(seeded).fetch(["AAPL"])
    # 2024-01-15 is before the 2024-02-01 filing -> nothing visible yet
    snap = fd.as_of("2024-01-15", concepts=["equity"])
    assert "AAPL" not in snap.index or snap.empty


def test_no_network_when_cached(seeded):
    """Cached quarters -> fetch succeeds with no network (this test has none)."""
    fd = _adapter(seeded).fetch(["AAPL", "MSFT"])
    assert len(fd.symbols) == 2


def test_factory_registration(seeded):
    """get_fundamental_adapter('sec_bulk') returns the right adapter."""
    cache, cik_map = seeded
    adapter = get_fundamental_adapter(
        "sec_bulk", email="me@example.com",
        start_quarter="2024q1", end_quarter="2024q1",
        cache_dir=cache, cik_map_path=cik_map,
    )
    assert adapter.name == "sec_bulk"


def test_rejects_bad_email():
    with pytest.raises(ValueError, match="email"):
        SecBulkAdapter(email="not-an-email")
