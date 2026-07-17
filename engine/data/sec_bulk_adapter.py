"""
SEC bulk adapter (piece 3 of 3): the store, behind the standard interface.

This is where the three pieces meet the engine. It:

  1. ensures the needed quarters are cached          (piece 1: sec_bulk_download)
  2. parses each into clean point-in-time facts       (piece 2: sec_bulk_parse)
  3. combines them, keeping first-reported values, into one FundamentalData

and exposes it as a FundamentalAdapter with the SAME .fetch(symbols) signature the
engine already uses. Swapping the per-company EDGAR fetch for this is a ONE-WORD
change at the call site -- get_fundamental_adapter("sec_bulk") -- and the screen,
metrics, and backtest never notice. That is the Market-as-config rule applied to
data vendors: swap the source, keep the engine.

WHY THIS IS BETTER than the per-company EdgarAdapter it replaces:
  - Coverage: bulk quarters carry ~5,600 filers each; the per-company fetch managed
    ~1,100 across the whole universe, with silent CIK-map gaps.
  - Speed: a handful of local file reads vs thousands of throttled HTTP calls.
  - Same honesty: point-in-time is native (the 'filed' date rides along), and we keep
    FIRST-reported values, so no look-ahead.

HONEST LIMITATION carried from piece 2: the CIK->ticker map is SEC's CURRENT ticker
list. A company that delisted years ago may have fundamentals in the ZIPs but no
current ticker to map onto -- so its data goes unused. This is a mild survivorship
flavour on the MAPPING side. It does not corrupt anything (we cannot price those dead
names anyway), but it means "coverage" here is coverage of STILL-LISTED companies.
Only paid data with historical ticker mappings fixes that.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from engine.data.fundamentals import FundamentalAdapter, FundamentalData
from engine.data.sec_bulk_download import (
    RAW_CACHE_DIR,
    ensure_quarter,
    quarters_between,
)
from engine.data.sec_bulk_parse import parse_quarter

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
DEFAULT_START = "2012q1"      # sensible default: post-XBRL-wall, momentum+value era


class SecBulkAdapter(FundamentalAdapter):
    """Fundamentals from the SEC's bulk quarterly data sets. Free, complete, fast."""

    name = "sec_bulk"

    def __init__(
        self,
        email: str,
        start_quarter: str = DEFAULT_START,
        end_quarter: str | None = None,
        cache_dir: Path | str = RAW_CACHE_DIR,
        cik_map_path: Path | str | None = None,
    ) -> None:
        if "@" not in email:
            raise ValueError("SEC requires a real contact email.")
        self.email = email
        self.start_quarter = start_quarter
        self.end_quarter = end_quarter or _current_quarter()
        self.cache_dir = Path(cache_dir)
        # Reuse the EDGAR adapter's cached CIK map if present, so both share one map.
        self.cik_map_path = (
            Path(cik_map_path) if cik_map_path
            else Path("data/fundamentals") / "cik_map.json"
        )
        self._cik_to_ticker: dict[int, str] | None = None

    # -- CIK -> ticker (inverse of the EDGAR adapter's ticker -> CIK) ----------

    def _load_cik_to_ticker(self) -> dict[int, str]:
        """
        Build the CIK->ticker map. Prefer the cached file the EDGAR adapter already
        wrote (ticker->CIK); invert it. If absent, fetch company_tickers.json.
        """
        if self._cik_to_ticker is not None:
            return self._cik_to_ticker

        ticker_to_cik: dict[str, int] = {}
        if self.cik_map_path.exists():
            with self.cik_map_path.open("r", encoding="utf-8") as fh:
                ticker_to_cik = {k: int(v) for k, v in json.load(fh).items()}
        else:
            # Fall back to a live fetch (same source the EDGAR adapter uses).
            import requests
            resp = requests.get(
                TICKERS_URL,
                headers={"User-Agent": f"momentum-engine research {self.email}"},
                timeout=30,
            )
            resp.raise_for_status()
            raw = resp.json()
            ticker_to_cik = {v["ticker"].upper(): int(v["cik_str"]) for v in raw.values()}
            self.cik_map_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cik_map_path.open("w", encoding="utf-8") as fh:
                json.dump(ticker_to_cik, fh)

        # Invert. If two tickers share a CIK (rare: share classes), last wins -- fine,
        # since our share-class dedup handles the portfolio side separately.
        self._cik_to_ticker = {cik: tk for tk, cik in ticker_to_cik.items()}
        return self._cik_to_ticker

    # -- the fetch the engine calls -------------------------------------------

    def fetch(
        self,
        symbols: list[str],
        concepts: list[str] | None = None,
        verbose: bool = False,
    ) -> FundamentalData:
        """
        Return point-in-time fundamentals for `symbols`, assembled from the bulk
        quarterly data sets between start_quarter and end_quarter.

        Downloads any missing quarters once (then cached forever), parses each, and
        combines. `symbols` filters the result; `concepts` optionally narrows columns.
        """
        cik_to_ticker = self._load_cik_to_ticker()
        wanted = {s.upper() for s in symbols}

        # Restrict the CIK map to the tickers we actually want -- smaller + faster parse.
        cik_subset = {cik: tk for cik, tk in cik_to_ticker.items() if tk in wanted}

        quarters = quarters_between(self.start_quarter, self.end_quarter)
        if verbose:
            print(f"  SEC bulk: {len(quarters)} quarters, "
                  f"{len(cik_subset)} of {len(wanted)} tickers mappable.")

        frames: list[pd.DataFrame] = []
        for q in quarters:
            path = ensure_quarter(q, self.email, cache_dir=self.cache_dir)
            facts = parse_quarter(path, cik_subset)
            if not facts.empty:
                frames.append(facts)
            if verbose and facts is not None:
                print(f"    {q}: {len(facts):,} facts")

        if not frames:
            empty = pd.DataFrame(
                columns=["symbol", "concept", "period_end", "filed", "value", "form", "fiscal_year"]
            )
            return FundamentalData(facts=empty, source="sec_bulk")

        combined = pd.concat(frames, ignore_index=True)

        # A period can appear across multiple quarterly files (a 10-K in 2024q1 also
        # restates 2022 numbers). Keep the earliest-filed value per (symbol, concept,
        # period_end) -- first reported, no look-ahead -- exactly as within a quarter.
        combined = (
            combined
            .dropna(subset=["symbol", "concept", "period_end", "filed"])
            .sort_values("filed")
            .drop_duplicates(subset=["symbol", "concept", "period_end"], keep="first")
            .reset_index(drop=True)
        )

        if concepts:
            combined = combined[combined["concept"].isin(concepts)].reset_index(drop=True)

        return FundamentalData(facts=combined, source="sec_bulk")


def _current_quarter() -> str:
    today = pd.Timestamp.today()
    return f"{today.year}q{(today.month - 1) // 3 + 1}"
