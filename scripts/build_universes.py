"""
Build the universe membership CSVs.

Run:  python -m scripts.build_universes

Fetches CURRENT index constituents and writes them to data/universes/.

⚠️ SURVIVORSHIP BIAS. These are today's members. Companies that were dropped
from an index in the past are absent, so a backtest over this list is optimistic:
it only ever holds companies that survived. We disclose this on every backtest
rather than pretending otherwise. The real fix is point-in-time membership data,
which is paid and fiddly -- a deliberate Later item, not an oversight.

This is a one-off utility, not engine code. It is allowed to know that Wikipedia
and NSE exist; the engine is not.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "universes"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}


# ---------------------------------------------------------------------------
# US
# ---------------------------------------------------------------------------


def _read_html_tables(url: str) -> list[pd.DataFrame]:
    """
    Fetch HTML with `requests`, then parse.

    We do NOT hand the URL straight to pandas.read_html: that path uses urllib,
    which trusts the machine's certificate store. On corporate Windows machines
    that store is often stale or behind an SSL-inspecting proxy, producing
    CERTIFICATE_VERIFY_FAILED. `requests` ships its own current CA bundle
    (certifi) and sidesteps the whole problem.
    """
    import io

    import requests

    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return pd.read_html(io.StringIO(resp.text))


def build_sp500() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = _read_html_tables(url)

    df = next(t for t in tables if "Symbol" in t.columns)
    out = pd.DataFrame({
        "symbol": df["Symbol"].astype(str).str.strip(),
        "name": df.get("Security", pd.Series(dtype=str)).astype(str).str.strip(),
        "sector": df.get("GICS Sector", pd.Series(dtype=str)).astype(str).str.strip(),
    })
    # Yahoo writes class shares with a hyphen: BRK.B -> BRK-B
    out["symbol"] = out["symbol"].str.replace(".", "-", regex=False)
    return out


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Wikipedia tables sometimes come back with MultiIndex headers. Flatten them."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join(str(p) for p in tup).strip() for tup in df.columns]
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Find a column whose name contains any of `candidates` (case-insensitive)."""
    for col in df.columns:
        low = col.lower()
        for want in candidates:
            if want in low:
                return col
    return None


def build_nasdaq100() -> pd.DataFrame:
    """
    Wikipedia reshuffles this page periodically, so match columns by CONTENT
    rather than by an exact header string. If it still cannot be found, say so
    clearly instead of dying with a bare StopIteration.
    """
    # Wikipedia moved the constituents OFF the main Nasdaq-100 article onto its
    # own list page. Try the list page first, fall back to the main article.
    urls = [
        "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies",
        "https://en.wikipedia.org/wiki/Nasdaq-100",
    ]

    tables: list[pd.DataFrame] = []
    url = urls[0]
    for candidate in urls:
        try:
            tables = [_flatten_columns(t) for t in _read_html_tables(candidate)]
            url = candidate
            if any(_pick_column(t, ("ticker", "symbol")) for t in tables):
                break
        except Exception:
            continue

    best = None
    for df in tables:
        ticker_col = _pick_column(df, ("ticker", "symbol"))
        name_col = _pick_column(df, ("company", "security", "name"))
        # The constituents table is the big one with both a ticker and a name.
        if ticker_col and name_col and len(df) >= 50:
            best = (df, ticker_col, name_col)
            break

    if best is None:
        seen = [list(t.columns)[:6] for t in tables]
        raise LookupError(
            f"No constituents table found on {url}. Tables seen: {seen}"
        )

    df, ticker_col, name_col = best
    sector_col = _pick_column(df, ("sector", "industry"))

    out = pd.DataFrame({
        "symbol": df[ticker_col].astype(str).str.strip(),
        "name": df[name_col].astype(str).str.strip(),
        "sector": (
            df[sector_col].astype(str).str.strip()
            if sector_col else pd.Series([""] * len(df))
        ),
    })
    out["symbol"] = out["symbol"].str.replace(".", "-", regex=False)
    return out


# ---------------------------------------------------------------------------
# India -- NSE publishes official constituent CSVs
# ---------------------------------------------------------------------------


def build_nifty(list_name: str) -> pd.DataFrame:
    """list_name e.g. 'ind_nifty50list' / 'ind_nifty200list' / 'ind_nifty500list'."""
    import io

    import requests

    url = f"https://nsearchives.nseindia.com/content/indices/{list_name}.csv"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = [c.strip() for c in df.columns]

    return pd.DataFrame({
        # Plain NSE symbols. The ".NS" suffix is applied by the MARKET, not here.
        "symbol": df["Symbol"].astype(str).str.strip(),
        "name": df["Company Name"].astype(str).str.strip(),
        "sector": df.get("Industry", pd.Series([""] * len(df))).astype(str).str.strip(),
    })


# ---------------------------------------------------------------------------

BUILDERS = {
    "us_sp500":     lambda: build_sp500(),
    "us_nasdaq100": lambda: build_nasdaq100(),
    "in_nifty50":   lambda: build_nifty("ind_nifty50list"),
    "in_nifty200":  lambda: build_nifty("ind_nifty200list"),
    "in_nifty500":  lambda: build_nifty("ind_nifty500list"),
}


def write_csv(name: str, df: pd.DataFrame) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.csv"

    df = df.dropna(subset=["symbol"])
    df = df[df["symbol"].str.len() > 0]
    df = df.drop_duplicates(subset=["symbol"]).sort_values("symbol")

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["symbol", "name", "sector"])
        for row in df.itertuples(index=False):
            writer.writerow([row.symbol, row.name, row.sector])

    return path


def main() -> None:
    print()
    print("=" * 70)
    print("  BUILDING UNIVERSE MEMBERSHIP LISTS")
    print("=" * 70)
    print("  ⚠️  These are CURRENT members -> survivorship bias.")
    print("      Disclosed on every backtest. Fixed later with point-in-time data.")
    print()

    failures = []

    for name, builder in BUILDERS.items():
        try:
            df = builder()
            path = write_csv(name, df)
            print(f"  ✅ {name:<14} {len(df):>4} symbols  →  {path.relative_to(OUT_DIR.parents[1])}")
        except Exception as exc:
            failures.append(name)
            print(f"  ❌ {name:<14} FAILED: {type(exc).__name__}: {exc}")

    print()
    if failures:
        print(f"  {len(failures)} source(s) failed: {', '.join(failures)}")
        print("  Public data sources move. If a fetch broke, we can point it")
        print("  elsewhere or drop the CSV in by hand -- the engine only reads")
        print("  data/universes/*.csv with columns: symbol,name,sector")
        sys.exit(1)

    print("  All universes built.")
    print()


if __name__ == "__main__":
    main()
