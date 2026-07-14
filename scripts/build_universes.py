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
# POINT-IN-TIME MEMBERSHIP  (the fix for inclusion bias)
# ---------------------------------------------------------------------------


def build_sp400_point_in_time() -> pd.DataFrame:
    """
    S&P 400 MidCap, point-in-time.

    Mid caps are the sweet spot for a first fundamental build: big enough that
    they rarely go to zero and yfinance actually has their prices, small enough
    that they are not all owned by every index fund on earth.
    """
    return _build_index_point_in_time(
        url="https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
        index_name="S&P 400 MidCap",
        min_current=300,
    )


def build_sp500_point_in_time() -> pd.DataFrame:
    return _build_index_point_in_time(
        url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        index_name="S&P 500",
        min_current=400,
    )


def build_sp900_point_in_time() -> pd.DataFrame:
    """
    S&P 500 + S&P 400 = the investable large/mid-cap universe, ~900 names.

    Deliberately EXCLUDES micro caps. Not because they are uninteresting -- AAII's
    best screen (Tiny Titans) lives there -- but because we cannot yet test them
    honestly. yfinance has no prices for the many micro caps that delisted, so a
    micro-cap backtest would be a survivorship fantasy. We will earn the right to
    that universe by buying proper data, not by pretending.
    """
    large = build_sp500_point_in_time()
    mid = build_sp400_point_in_time()

    combined = pd.concat([large, mid], ignore_index=True)

    # A company can be promoted from the S&P 400 to the S&P 500 -- it will then
    # appear in both, with different intervals. That is correct, and we keep both.
    return combined.sort_values(["symbol", "start_date"]).drop_duplicates(
        subset=["symbol", "start_date", "end_date"]
    )


def _build_index_point_in_time(
    url: str, index_name: str, min_current: int
) -> pd.DataFrame:
    """
    Reconstruct WHO WAS IN the S&P 500 ON EACH DATE, from the index's own
    add/remove history.

    WHY THIS MATTERS -- and it is not the bias people usually name.

    Everyone worries about survivorship bias: dead companies missing from the list.
    Real, but for the S&P 500 it is not the dominant problem. The bigger one is
    INCLUSION BIAS.

    Ask how a company GETS INTO the S&P 500: by growing enormously. So if you
    backtest 2013 using the 2026 membership list, you have handed the strategy a
    universe pre-selected for the next decade's biggest winners -- NVDA, PLTR, APP,
    VST, CVNA. Momentum's entire job is to find the strongest risers, and you have
    quietly guaranteed they are all in the room before it starts looking.

    That is not "we deleted the losers". It is "we guaranteed the future winners are
    present", which is worse, because it is aimed at exactly what the strategy hunts.

    This function fixes it: a stock becomes eligible only from the date it ACTUALLY
    JOINED the index.

    Output: one row per membership INTERVAL (a company can join, leave, and rejoin).
        symbol, name, sector, start_date, end_date
        end_date empty = still a member.
    """
    tables = [_flatten_columns(t) for t in _read_html_tables(url)]

    # Table 1: current constituents.
    current_tbl = next(
        t for t in tables if _pick_column(t, ("symbol",)) and len(t) > min_current
    )
    sym_col = _pick_column(current_tbl, ("symbol",))
    name_col = _pick_column(current_tbl, ("security",)) or _pick_column(current_tbl, ("name",))
    sec_col = _pick_column(current_tbl, ("gics sector", "sector"))

    current: dict[str, dict] = {}
    for r in current_tbl.itertuples(index=False):
        d = dict(zip(current_tbl.columns, r))
        sym = str(d[sym_col]).strip().replace(".", "-")
        current[sym] = {
            "name": str(d.get(name_col, "")).strip(),
            "sector": str(d.get(sec_col, "")).strip() if sec_col else "",
        }

    # Table 2: the change log (Date / Added Ticker / Removed Ticker).
    changes = None
    for t in tables:
        cols = " ".join(t.columns).lower()
        if "added" in cols and "removed" in cols and "date" in cols:
            changes = t
            break
    if changes is None:
        # Degrade HONESTLY: no change log means no point-in-time. Say so loudly
        # rather than silently shipping a biased universe dressed up as a clean one.
        print(f"     ⚠️  No add/remove change log found for {index_name}.")
        print("         Falling back to a CURRENT snapshot — INCLUSION BIAS REMAINS.")
        out = pd.DataFrame({
            "symbol": list(current),
            "name": [v["name"] for v in current.values()],
            "sector": [v["sector"] for v in current.values()],
            "start_date": "",
            "end_date": "",
        })
        return out.sort_values("symbol")

    date_col = _pick_column(changes, ("date",))
    add_col = next((c for c in changes.columns if "added" in c.lower() and "ticker" in c.lower()), None)
    rem_col = next((c for c in changes.columns if "removed" in c.lower() and "ticker" in c.lower()), None)
    add_name = next((c for c in changes.columns if "added" in c.lower() and "security" in c.lower()), None)
    rem_name = next((c for c in changes.columns if "removed" in c.lower() and "security" in c.lower()), None)

    if not (date_col and add_col and rem_col):
        raise LookupError(f"Change log columns not recognised: {list(changes.columns)}")

    events = []
    for r in changes.itertuples(index=False):
        d = dict(zip(changes.columns, r))
        when = pd.to_datetime(str(d[date_col]), errors="coerce")
        if pd.isna(when):
            continue

        added = str(d.get(add_col, "")).strip().replace(".", "-")
        removed = str(d.get(rem_col, "")).strip().replace(".", "-")
        added = "" if added.lower() in ("nan", "") else added
        removed = "" if removed.lower() in ("nan", "") else removed

        events.append({
            "date": when,
            "added": added,
            "removed": removed,
            "added_name": str(d.get(add_name, "")).strip() if add_name else "",
            "removed_name": str(d.get(rem_name, "")).strip() if rem_name else "",
        })

    events.sort(key=lambda e: e["date"])

    # --- Step 1: walk BACKWARDS from today to find the membership at the start.
    active = set(current)
    for e in reversed(events):
        if e["added"] and e["added"] in active:
            active.discard(e["added"])       # it was not a member before it was added
        if e["removed"]:
            active.add(e["removed"])         # it WAS a member before it was removed

    earliest = events[0]["date"] if events else pd.Timestamp("1990-01-01")

    # --- Step 2: walk FORWARDS, opening and closing membership intervals.
    names: dict[str, str] = {s: v["name"] for s, v in current.items()}
    open_since: dict[str, pd.Timestamp] = {s: earliest for s in active}
    intervals: list[dict] = []

    for e in events:
        if e["removed"]:
            sym = e["removed"]
            names.setdefault(sym, e["removed_name"])
            start = open_since.pop(sym, earliest)
            intervals.append({"symbol": sym, "start": start, "end": e["date"]})

        if e["added"]:
            sym = e["added"]
            names.setdefault(sym, e["added_name"])
            if sym not in open_since:
                open_since[sym] = e["date"]

    for sym, start in open_since.items():
        intervals.append({"symbol": sym, "start": start, "end": pd.NaT})

    # --- Sanity check: our reconstruction should end at today's actual membership.
    reconstructed_now = {i["symbol"] for i in intervals if pd.isna(i["end"])}
    missing = set(current) - reconstructed_now
    extra = reconstructed_now - set(current)
    if missing or extra:
        print(f"     note: {index_name} drift — {len(missing)} missing, {len(extra)} extra")
        print("     (Wikipedia's change log is incomplete for older years. Expected.)")

    out = pd.DataFrame(intervals)
    out["name"] = out["symbol"].map(lambda s: names.get(s, ""))
    out["sector"] = out["symbol"].map(lambda s: current.get(s, {}).get("sector", ""))
    out["start_date"] = out["start"].dt.strftime("%Y-%m-%d")
    out["end_date"] = out["end"].dt.strftime("%Y-%m-%d").fillna("")

    return out[["symbol", "name", "sector", "start_date", "end_date"]].sort_values(
        ["symbol", "start_date"]
    )


# ---------------------------------------------------------------------------

BUILDERS = {
    "us_sp500":     lambda: build_sp500(),
    "us_sp500_pit": lambda: build_sp500_point_in_time(),
    "us_sp400_pit": lambda: build_sp400_point_in_time(),
    "us_sp900_pit": lambda: build_sp900_point_in_time(),
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

    # Point-in-time files have MULTIPLE rows per symbol (join, leave, rejoin),
    # so only de-duplicate the flat snapshot files.
    if "start_date" not in df.columns:
        df = df.drop_duplicates(subset=["symbol"])

    df = df.sort_values("symbol")

    df.to_csv(path, index=False, encoding="utf-8")
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
