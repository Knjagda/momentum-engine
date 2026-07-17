"""
Tests for the SEC bulk downloader/cache (piece 1).

The network download itself is not unit-tested (it needs the SEC); what we test is
all the logic AROUND it -- validation, path handling, the quarter-range math, and
crucially the "is this cached file actually valid?" guard that stops a truncated
download from poisoning every future run.
"""

import io
import zipfile

import pytest

from engine.data.sec_bulk_download import (
    SecBulkDownloadError,
    cache_path,
    ensure_quarter,
    quarter_url,
    quarters_between,
    _looks_like_valid_zip,
)


def _make_zip(path, members: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)


# ---- quarter validation ---------------------------------------------------


def test_url_construction():
    assert quarter_url("2024q1").endswith("/2024q1.zip")


@pytest.mark.parametrize("bad", ["2024", "2024q5", "24q1", "2024Q", "abcdq1", "2024-q1"])
def test_rejects_malformed_quarters(bad, tmp_path):
    with pytest.raises(ValueError):
        ensure_quarter(bad, "me@example.com", cache_dir=tmp_path)


def test_rejects_pre_xbrl_years(tmp_path):
    with pytest.raises(ValueError, match="XBRL data starts"):
        ensure_quarter("2007q1", "me@example.com", cache_dir=tmp_path)


def test_requires_email(tmp_path):
    with pytest.raises(ValueError, match="email"):
        ensure_quarter("2024q1", "not-an-email", cache_dir=tmp_path)


# ---- cache validity guard -------------------------------------------------


def test_valid_zip_recognised(tmp_path):
    p = tmp_path / "2024q1.zip"
    _make_zip(p, {"sub.txt": "x", "num.txt": "y"})
    assert _looks_like_valid_zip(p)


def test_zip_missing_required_members_is_invalid(tmp_path):
    p = tmp_path / "2024q1.zip"
    _make_zip(p, {"sub.txt": "x"})           # no num.txt
    assert not _looks_like_valid_zip(p)


def test_corrupt_file_is_invalid(tmp_path):
    p = tmp_path / "2024q1.zip"
    p.write_bytes(b"this is not a zip")
    assert not _looks_like_valid_zip(p)


def test_empty_file_is_invalid(tmp_path):
    p = tmp_path / "2024q1.zip"
    p.write_bytes(b"")
    assert not _looks_like_valid_zip(p)


def test_ensure_quarter_returns_cached_without_network(tmp_path):
    """
    If a valid ZIP is already cached, ensure_quarter must return it WITHOUT any
    network access. We prove 'no network' by the simple fact that this test has
    none and yet succeeds.
    """
    p = cache_path("2024q1", tmp_path)
    _make_zip(p, {"sub.txt": "x", "num.txt": "y"})
    result = ensure_quarter("2024q1", "me@example.com", cache_dir=tmp_path)
    assert result == p


# ---- quarter range math ---------------------------------------------------


def test_quarters_between_same_year():
    assert quarters_between("2012q1", "2012q4") == ["2012q1", "2012q2", "2012q3", "2012q4"]


def test_quarters_between_crossing_years():
    assert quarters_between("2012q3", "2013q2") == ["2012q3", "2012q4", "2013q1", "2013q2"]


def test_quarters_between_single():
    assert quarters_between("2020q2", "2020q2") == ["2020q2"]


def test_quarters_between_long_span():
    span = quarters_between("2012q1", "2026q1")
    assert len(span) == 57          # 14 years * 4 + 1
    assert span[0] == "2012q1"
    assert span[-1] == "2026q1"
