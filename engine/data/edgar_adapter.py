"""
SEC EDGAR: free, official, and the primary source.

Bloomberg and FactSet build on this same data. It costs nothing, needs no API key,
and every fact carries the date it was filed -- which is the only thing we truly
need, because it lets us enforce "you may not know what you could not have known".

THREE REAL-WORLD TRAPS, all found the hard way by running the spike:

1. THE TICKER→CIK MAP LIES.
   company_tickers.json mapped XOM to CIK 2115436 -- a recently registered shell.
   Exxon's real CIK is 34088. Had we trusted it, Exxon would have silently vanished
   from every backtest. No error. No warning. Just a company quietly missing.
   So: we VALIDATE, and we SHOUT when a company comes back empty.

2. COMPANIES CHANGE TAGS OVER TIME.
   ASC 606 (2018) moved nearly everyone from `SalesRevenueNet` to
   `RevenueFromContractWithCustomerExcludingAssessedTax`. A single hard-coded tag
   would show a company's revenue mysteriously vanishing in 2018. So each concept
   is a PRIORITY CHAIN, and we take facts from whichever tag the company was using
   at the time.

3. THE SAME PERIOD IS FILED MANY TIMES.
   Handled in FundamentalData: we keep the earliest filing (as-first-reported).

Rate limit: SEC allows 10 requests/second. We stay well under. Be a good citizen --
they will block you otherwise, and they would be right to.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from engine.data.fundamentals import FACT_COLUMNS, FundamentalAdapter, FundamentalData
from engine.markets.market import Market

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

CACHE_DIR = Path("data/fundamentals")

# Concept → candidate XBRL tags, BEST FIRST.
# We take facts from every tag in the chain, then keep the earliest filing for each
# period. That way a company that switched tags in 2018 keeps an unbroken history.
CONCEPT_TAGS: dict[str, list[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",   # post-ASC 606
        "Revenues",
        "SalesRevenueNet",                                        # pre-ASC 606
        "SalesRevenueGoodsNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "shares": [
        "CommonStockSharesOutstanding",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
}

ACCEPTED_FORMS = {"10-K", "10-Q", "20-F", "40-F"}


class EdgarAdapter(FundamentalAdapter):
    """Free fundamentals, straight from the source."""

    name = "edgar"

    def __init__(
        self,
        market: Market,
        user_agent: str,
        cache_dir: Path | str = CACHE_DIR,
    ) -> None:
        if not user_agent or "example.com" in user_agent or "@" not in user_agent:
            raise ValueError(
                "SEC requires a real contact in the User-Agent, e.g. "
                "'momentum-engine you@yourdomain.com'. They will 403 you otherwise, "
                "and they are entitled to."
            )

        self.market = market
        self.user_agent = user_agent
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # The currency comes from the MARKET, never from a literal in this file.
        # (Our own AST contract test caught me hard-coding it. It was right.)
        #
        # It also makes a real limitation explicit rather than hidden: point this
        # adapter at India and it will filter for INR, find nothing, and say so.
        # EDGAR is the U.S. Securities and Exchange Commission. It does not know
        # about the NSE, and it never will.
        currency = market.currency
        self._units = {currency, "shares", f"{currency}/shares"}

        self._headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        self._cik_map: dict[str, int] | None = None

    # -- http ---------------------------------------------------------------

    def _get(self, url: str) -> dict:
        time.sleep(0.12)                          # < 10 req/sec. Be polite.
        response = requests.get(url, headers=self._headers, timeout=30)
        response.raise_for_status()
        return response.json()

    # -- ticker → CIK, with validation --------------------------------------

    def cik_map(self, refresh: bool = False) -> dict[str, int]:
        cache = self.cache_dir / "cik_map.json"

        if cache.exists() and not refresh:
            with cache.open("r", encoding="utf-8") as fh:
                return {k: int(v) for k, v in json.load(fh).items()}

        raw = self._get(TICKERS_URL)
        mapping = {v["ticker"].upper(): int(v["cik_str"]) for v in raw.values()}

        with cache.open("w", encoding="utf-8") as fh:
            json.dump(mapping, fh)

        return mapping

    def resolve_cik(self, symbol: str) -> int | None:
        if self._cik_map is None:
            self._cik_map = self.cik_map()
        return self._cik_map.get(symbol.upper())

    # -- the fetch ----------------------------------------------------------

    def fetch(
        self,
        symbols: list[str],
        concepts: list[str] | None = None,
        verbose: bool = True,
    ) -> FundamentalData:
        wanted = concepts or list(CONCEPT_TAGS)
        unknown = set(wanted) - set(CONCEPT_TAGS)
        if unknown:
            raise KeyError(f"Unknown concepts {sorted(unknown)}. Known: {sorted(CONCEPT_TAGS)}")

        rows: list[dict] = []
        unresolved: list[str] = []
        empty: list[str] = []

        for i, symbol in enumerate(symbols, 1):
            cached = self._load_cache(symbol)
            if cached is not None:
                rows.extend(cached.to_dict("records"))
                continue

            cik = self.resolve_cik(symbol)
            if cik is None:
                unresolved.append(symbol)
                continue

            try:
                facts = self._get(FACTS_URL.format(cik=cik))
            except Exception:
                empty.append(symbol)
                continue

            extracted = self._extract(symbol, facts, wanted)

            # TRAP 1. A company that returns NOTHING is not a shrug -- it usually
            # means the ticker→CIK map sent us to the wrong entity (this is exactly
            # what happened with XOM). Never let it pass silently.
            if extracted.empty:
                empty.append(symbol)
                continue

            self._save_cache(symbol, extracted)
            rows.extend(extracted.to_dict("records"))

            if verbose and i % 50 == 0:
                print(f"    ...{i}/{len(symbols)}")

        if verbose and (unresolved or empty):
            print()
            if unresolved:
                print(f"  ⚠️  {len(unresolved)} tickers not in SEC's map: "
                      f"{', '.join(unresolved[:12])}{'...' if len(unresolved) > 12 else ''}")
            if empty:
                print(f"  ⚠️  {len(empty)} tickers returned NO facts — likely a bad CIK "
                      f"mapping, NOT an absent company:")
                print(f"      {', '.join(empty[:12])}{'...' if len(empty) > 12 else ''}")
                print("      These are silently missing from any backtest. Investigate them.")
            print()

        frame = (
            pd.DataFrame(rows, columns=FACT_COLUMNS)
            if rows else pd.DataFrame(columns=FACT_COLUMNS)
        )
        return FundamentalData(facts=frame, source="SEC EDGAR")

    # -- parsing ------------------------------------------------------------

    def _extract(self, symbol: str, payload: dict, wanted: list[str]) -> pd.DataFrame:
        facts = payload.get("facts", {})
        gaap = facts.get("us-gaap", {})
        dei = facts.get("dei", {})

        rows: list[dict] = []

        for concept in wanted:
            for tag in CONCEPT_TAGS[concept]:
                # TRAP 2: walk the WHOLE chain, not just the first hit. A company
                # that switched tags in 2018 has its history split across two of them.
                source = gaap.get(tag) or dei.get(tag)
                if not source:
                    continue

                for unit, entries in source.get("units", {}).items():
                    if unit not in self._units:
                        continue

                    for entry in entries:
                        if entry.get("form") not in ACCEPTED_FORMS:
                            continue
                        if "filed" not in entry or "end" not in entry:
                            continue
                        if entry.get("val") is None:
                            continue

                        rows.append({
                            "symbol": symbol,
                            "concept": concept,
                            "period_end": entry["end"],
                            "filed": entry["filed"],
                            "value": float(entry["val"]),
                            "form": entry["form"],
                            "fiscal_year": entry.get("fy"),
                        })

        return pd.DataFrame(rows, columns=FACT_COLUMNS)

    # -- cache --------------------------------------------------------------

    def _cache_path(self, symbol: str) -> Path:
        return self.cache_dir / f"{symbol.upper()}.parquet"

    def _load_cache(self, symbol: str) -> pd.DataFrame | None:
        path = self._cache_path(symbol)
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception:
            return None

    def _save_cache(self, symbol: str, frame: pd.DataFrame) -> None:
        try:
            frame.to_parquet(self._cache_path(symbol), index=False)
        except Exception:
            pass          # a cache miss is an inconvenience, not a failure
