"""
SEC bulk financial-statement data sets: the downloader/cache (piece 1 of the adapter).

ONE JOB, done boringly well: given a quarter like "2024q1", make sure its raw ZIP is
on disk -- downloading it once if missing, skipping instantly if already there.

Why a separate piece: these files are ~100 MB each and a full history is ~5 GB. We
download each quarter EXACTLY ONCE, ever. Everything downstream (parsing, filtering,
building the fundamentals store) reads from this local cache, so a rebuild never
re-hits the SEC. This is the "pull once, keep locally" pattern, made concrete.

The SEC's rules we respect:
  - A descriptive User-Agent WITH a contact email is required, or they 403.
  - Be gentle: one request per quarter, cached forever. No hammering.

This module does NOT parse anything. It hands back a path to a verified ZIP. Parsing
is the next piece. Keeping them separate means a parsing bug never forces a re-download.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = "https://www.sec.gov/files/dera/data/financial-statement-data-sets"

# Raw ZIPs live here. Gitignored (data/ is), like every other cache.
RAW_CACHE_DIR = Path("data/fundamentals/sec_bulk_raw")

# XBRL data only exists from 2009 Q1 onward -- earlier quarters simply do not exist.
EARLIEST_YEAR = 2009


class SecBulkDownloadError(RuntimeError):
    """Raised when a quarter cannot be fetched, with a human-readable reason."""


def quarter_url(quarter: str) -> str:
    return f"{BASE_URL}/{quarter}.zip"


def _validate_quarter(quarter: str) -> None:
    """Cheap sanity check on the 'YYYYqN' format before we hit the network."""
    q = quarter.lower().strip()
    if len(q) != 6 or q[4] != "q" or not q[:4].isdigit() or q[5] not in "1234":
        raise ValueError(f"Bad quarter '{quarter}'. Expected e.g. '2024q1'.")
    year = int(q[:4])
    if year < EARLIEST_YEAR:
        raise ValueError(
            f"{quarter}: SEC XBRL data starts {EARLIEST_YEAR}Q1. Nothing exists earlier."
        )


def cache_path(quarter: str, cache_dir: Path | str = RAW_CACHE_DIR) -> Path:
    return Path(cache_dir) / f"{quarter.lower()}.zip"


def _looks_like_valid_zip(path: Path) -> bool:
    """A cached file is only trustworthy if it opens AND contains the expected members."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            return {"sub.txt", "num.txt"}.issubset(names)
    except zipfile.BadZipFile:
        return False


def ensure_quarter(
    quarter: str,
    email: str,
    cache_dir: Path | str = RAW_CACHE_DIR,
    timeout: int = 180,
    force: bool = False,
) -> Path:
    """
    Guarantee the quarter's ZIP is on disk and valid; return its path.

    Downloads only if missing/corrupt (or force=True). A previously-downloaded,
    still-valid file is returned instantly with no network call.
    """
    if "@" not in email:
        raise ValueError("SEC requires a real contact email in the User-Agent.")
    _validate_quarter(quarter)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(quarter, cache_dir)

    # Fast path: already have a good copy.
    if not force and _looks_like_valid_zip(path):
        return path

    # A corrupt/partial file from a prior failed run -- remove before retrying.
    if path.exists():
        path.unlink()

    url = quarter_url(quarter)
    req = Request(url, headers={"User-Agent": f"momentum-engine research {email}"})

    try:
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    except HTTPError as e:
        if e.code == 403:
            raise SecBulkDownloadError(
                f"{quarter}: SEC returned 403. The User-Agent/email was rejected."
            ) from e
        if e.code == 404:
            raise SecBulkDownloadError(
                f"{quarter}: not found (404). Future/nonexistent quarter?"
            ) from e
        raise SecBulkDownloadError(f"{quarter}: HTTP {e.code} fetching {url}.") from e
    except URLError as e:
        raise SecBulkDownloadError(f"{quarter}: network error: {e.reason}") from e

    # Write, then verify what we wrote. A truncated download must not masquerade
    # as a good cache entry on the next run.
    path.write_bytes(data)
    if not _looks_like_valid_zip(path):
        path.unlink(missing_ok=True)
        raise SecBulkDownloadError(
            f"{quarter}: downloaded {len(data)/1e6:.1f} MB but it is not a valid "
            "financial-statement ZIP (missing sub.txt/num.txt or corrupt)."
        )
    return path


def quarters_between(start: str, end: str) -> list[str]:
    """
    All quarter strings from start..end inclusive, e.g. ('2012q1','2013q2') ->
    ['2012q1','2012q2','2012q3','2012q4','2013q1','2013q2']. For bulk backfills.
    """
    _validate_quarter(start)
    _validate_quarter(end)
    sy, sq = int(start[:4]), int(start[5])
    ey, eq = int(end[:4]), int(end[5])
    out: list[str] = []
    y, q = sy, sq
    while (y, q) <= (ey, eq):
        out.append(f"{y}q{q}")
        q += 1
        if q > 4:
            q = 1
            y += 1
    return out
