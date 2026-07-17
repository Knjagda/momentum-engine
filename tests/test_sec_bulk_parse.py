"""
Tests for the SEC bulk parser (piece 2), all offline with synthetic ZIPs.

We forge tiny sub.txt / num.txt files with known contents and assert the parser:
  - keeps only our concepts (drops the 99% of tags we don't use),
  - maps CIK -> ticker and drops (but counts) the unmappable,
  - drops per-segment slices, keeping company-wide totals (Dec-2024 format),
  - keeps the FIRST-reported value when a period is restated,
  - produces exactly our FACT_COLUMNS shape.
"""

import io
import zipfile

import pandas as pd
import pytest

from engine.data.fundamentals import FACT_COLUMNS, FundamentalData
from engine.data.sec_bulk_parse import parse_quarter, parse_quarter_report

# CIK -> ticker for the test companies
CIK_MAP = {320193: "AAPL", 789019: "MSFT"}


def _zip(sub_rows: str, num_rows: str, sub_cols: str, num_cols: str) -> io.BytesIO:
    """Build an in-memory ZIP with tab-separated sub.txt and num.txt."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("sub.txt", sub_cols + "\n" + sub_rows)
        zf.writestr("num.txt", num_cols + "\n" + num_rows)
    buf.seek(0)
    # parse_quarter takes a path; write to a temp and return its bytes via a path shim
    return buf


@pytest.fixture
def sample_zip(tmp_path):
    """
    A realistic-ish quarter:
      - AAPL 10-K: equity 50000, revenue 100000 (company-wide), plus a per-SEGMENT
        revenue row that must be DROPPED.
      - MSFT 10-Q: equity 60000.
      - CIK 999999 (unmapped): equity 1 -- must be dropped but counted.
      - A junk tag (LeaseCost) that must be filtered out.
    """
    sub_cols = "adsh\tcik\tform\tperiod\tfiled\tfy"
    sub_rows = "\n".join([
        "0001-AAPL\t320193\t10-K\t20231230\t20240201\t2023",
        "0002-MSFT\t789019\t10-Q\t20231231\t20240125\t2024",
        "0003-XXXX\t999999\t10-K\t20231231\t20240115\t2023",  # unmapped CIK
    ])

    num_cols = "adsh\ttag\tversion\tddate\tqtrs\tuom\tsegments\tcoreg\tvalue"
    num_rows = "\n".join([
        # AAPL company-wide (segments empty) -- KEEP
        "0001-AAPL\tStockholdersEquity\tus-gaap\t20231230\t0\tUSD\t\t\t50000",
        "0001-AAPL\tRevenues\tus-gaap\t20231230\t4\tUSD\t\t\t100000",
        # AAPL per-segment revenue (segments non-empty) -- DROP
        "0001-AAPL\tRevenues\tus-gaap\t20231230\t4\tUSD\tProductAxis=iPhone\t\t60000",
        # AAPL junk tag -- DROP
        "0001-AAPL\tLeaseCost\tus-gaap\t20231230\t4\tUSD\t\t\t999",
        # MSFT equity -- KEEP
        "0002-MSFT\tStockholdersEquity\tus-gaap\t20231231\t0\tUSD\t\t\t60000",
        # Unmapped company equity -- DROP (but count)
        "0003-XXXX\tStockholdersEquity\tus-gaap\t20231231\t0\tUSD\t\t\t1",
    ])

    p = tmp_path / "2024q1.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("sub.txt", sub_cols + "\n" + sub_rows)
        zf.writestr("num.txt", num_cols + "\n" + num_rows)
    return p


def test_produces_fact_columns_shape(sample_zip):
    facts = parse_quarter(sample_zip, CIK_MAP)
    assert list(facts.columns) == FACT_COLUMNS
    # And FundamentalData accepts it without complaint.
    fd = FundamentalData(facts=facts, source="sec_bulk_test")
    assert fd is not None


def test_keeps_only_known_concepts(sample_zip):
    facts = parse_quarter(sample_zip, CIK_MAP)
    concepts = set(facts["concept"])
    assert concepts <= {"equity", "revenue"}          # not LeaseCost
    assert "equity" in concepts


def test_drops_and_counts_unmapped_ciks(sample_zip):
    report = parse_quarter_report(sample_zip, CIK_MAP)
    assert report["total_filers"] == 3
    assert report["mappable_filers"] == 2
    assert report["unmapped_filers"] == 1
    # the unmapped company's data is absent
    assert set(report["facts"]["symbol"]) == {"AAPL", "MSFT"}


def test_drops_per_segment_rows(sample_zip):
    """AAPL revenue must be the 100000 company-wide total, not the 60000 segment."""
    facts = parse_quarter(sample_zip, CIK_MAP)
    aapl_rev = facts[(facts["symbol"] == "AAPL") & (facts["concept"] == "revenue")]
    assert len(aapl_rev) == 1
    assert aapl_rev.iloc[0]["value"] == 100000


def test_maps_tags_to_concepts(sample_zip):
    facts = parse_quarter(sample_zip, CIK_MAP)
    aapl_eq = facts[(facts["symbol"] == "AAPL") & (facts["concept"] == "equity")]
    assert aapl_eq.iloc[0]["value"] == 50000


def test_first_reported_wins_on_restatement(tmp_path):
    """
    Same company, same period, same concept, reported twice: an original filing and
    a later restatement with a different number. The parser must keep the ORIGINAL
    (earliest filed) -- that is what an honest backtest would have seen at the time.
    """
    sub_cols = "adsh\tcik\tform\tperiod\tfiled\tfy"
    sub_rows = "\n".join([
        "0001-ORIG\t320193\t10-K\t20221231\t20230201\t2022",   # original, filed first
        "0002-REST\t320193\t10-K\t20221231\t20231115\t2022",   # restatement, later
    ])
    num_cols = "adsh\ttag\tversion\tddate\tqtrs\tuom\tsegments\tcoreg\tvalue"
    num_rows = "\n".join([
        "0001-ORIG\tStockholdersEquity\tus-gaap\t20221231\t0\tUSD\t\t\t50000",
        "0002-REST\tStockholdersEquity\tus-gaap\t20221231\t0\tUSD\t\t\t55000",  # restated up
    ])
    p = tmp_path / "2023q1.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("sub.txt", sub_cols + "\n" + sub_rows)
        zf.writestr("num.txt", num_cols + "\n" + num_rows)

    facts = parse_quarter(p, CIK_MAP)
    eq = facts[facts["concept"] == "equity"]
    assert len(eq) == 1
    assert eq.iloc[0]["value"] == 50000, "should keep first-reported, not restated"


def test_handles_old_format_without_segments_column(tmp_path):
    """Pre-Dec-2024 files have no 'segments' column -- parser must still work."""
    sub_cols = "adsh\tcik\tform\tperiod\tfiled\tfy"
    sub_rows = "0001-AAPL\t320193\t10-K\t20221231\t20230201\t2022"
    num_cols = "adsh\ttag\tversion\tddate\tqtrs\tuom\tvalue"     # no segments/coreg
    num_rows = "0001-AAPL\tStockholdersEquity\tus-gaap\t20221231\t0\tUSD\t50000"
    p = tmp_path / "2020q1.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("sub.txt", sub_cols + "\n" + sub_rows)
        zf.writestr("num.txt", num_cols + "\n" + num_rows)

    facts = parse_quarter(p, CIK_MAP)
    assert len(facts) == 1
    assert facts.iloc[0]["value"] == 50000


def test_filed_date_is_preserved_for_point_in_time(sample_zip):
    """The filed date -- our anti-look-ahead key -- must survive parsing intact."""
    facts = parse_quarter(sample_zip, CIK_MAP)
    aapl_eq = facts[(facts["symbol"] == "AAPL") & (facts["concept"] == "equity")]
    assert pd.Timestamp(aapl_eq.iloc[0]["filed"]) == pd.Timestamp("2024-02-01")


# ---------------------------------------------------------------------------
# Regression tests for the WMT equity bug (real-data format)
# ---------------------------------------------------------------------------
#
# The original tests used empty-string segments. REAL SEC data writes an empty
# segment as the string 'nan' (pandas NaN round-tripped through TSV), and puts
# equity-component breakdowns (RetainedEarnings, NoncontrollingInterest, ...) in the
# segments field. The parser kept those breakdown rows, and equity came out wrong for
# ~100 S&P 500 names. These tests reproduce the real format so the bug can't return.


def _zip_real_format(tmp_path, num_rows, name="2024q4.zip"):
    """Build a ZIP whose num.txt matches REAL SEC formatting (segments='nan' etc.)."""
    sub_cols = "adsh\tcik\tform\tperiod\tfiled\tfy"
    sub_rows = "0001-WMT\t104169\t10-Q\t20240131\t20240301\t2024"
    num_cols = "adsh\ttag\tversion\tddate\tqtrs\tuom\tsegments\tcoreg\tvalue"
    p = tmp_path / name
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("sub.txt", sub_cols + "\n" + sub_rows)
        zf.writestr("num.txt", num_cols + "\n" + "\n".join(num_rows))
    return p


WMT_CIK = {104169: "WMT"}


def test_nan_string_segment_is_treated_as_company_wide(tmp_path):
    """A total row whose segments field is the string 'nan' must be KEPT."""
    rows = [
        "0001-WMT\tStockholdersEquity\tus-gaap\t20240131\t0\tUSD\tnan\tnan\t83900",
    ]
    facts = parse_quarter(_zip_real_format(tmp_path, rows), WMT_CIK)
    eq = facts[facts["concept"] == "equity"]
    assert len(eq) == 1
    assert eq.iloc[0]["value"] == 83900


def test_equity_component_breakdown_rows_are_dropped(tmp_path):
    """
    The exact WMT bug: the real total (83900, segments 'nan') plus a pile of
    EquityComponents=... breakdown rows. Only the total must survive.
    """
    rows = [
        "0001-WMT\tStockholdersEquity\tus-gaap\t20240131\t0\tUSD\tnan\tnan\t83900",
        "0001-WMT\tStockholdersEquity\tus-gaap\t20240131\t0\tUSD\tEquityComponents=RetainedEarnings;\tnan\t89800",
        "0001-WMT\tStockholdersEquity\tus-gaap\t20240131\t0\tUSD\tEquityComponents=NoncontrollingInterest;\tnan\t6500",
        "0001-WMT\tStockholdersEquity\tus-gaap\t20240131\t0\tUSD\tEquityComponents=CommonStock;\tnan\t800",
    ]
    facts = parse_quarter(_zip_real_format(tmp_path, rows), WMT_CIK)
    eq = facts[facts["concept"] == "equity"]
    assert len(eq) == 1, "breakdown rows leaked in"
    assert eq.iloc[0]["value"] == 83900, "picked a breakdown component, not the total"


def test_parent_only_equity_wins_over_including_minority(tmp_path):
    """
    When both equity tags appear as company-wide totals (segments 'nan'), the
    parent-only StockholdersEquity (83900) must beat the higher
    ...IncludingPortionAttributableToNoncontrolling... (90300).
    """
    rows = [
        "0001-WMT\tStockholdersEquityIncludingPortionAttributableToNoncontrollingInterest\tus-gaap\t20240131\t0\tUSD\tnan\tnan\t90300",
        "0001-WMT\tStockholdersEquity\tus-gaap\t20240131\t0\tUSD\tnan\tnan\t83900",
    ]
    facts = parse_quarter(_zip_real_format(tmp_path, rows), WMT_CIK)
    eq = facts[facts["concept"] == "equity"]
    assert len(eq) == 1
    assert eq.iloc[0]["value"] == 83900, "should prefer parent-only equity"
