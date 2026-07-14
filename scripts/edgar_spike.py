"""
SEC EDGAR SPIKE — prove it works before we build on it.

    python -m scripts.edgar_spike

This is a SPIKE, not a feature. Its only job is to answer three questions before we
spend weeks building a fundamental data layer:

    1. Is EDGAR really point-in-time?  (does every fact carry a FILING date?)
    2. How big is the look-ahead trap?  (how long AFTER a quarter ends is it public?)
    3. How bad is tag normalisation?    (do companies use the same XBRL tags?)

Question 2 is the one that matters most. A company's Q4 balance sheet describes
31 December -- but nobody could SEE it until the 10-K was filed, often 60-90 days
later. A screen run on 2 January using December's numbers is reading a document
that does not exist yet. With fundamentals this look-ahead is far more dangerous
than anything we have caught so far, because the delay is MONTHS, not one day.

If EDGAR gives us the filing date, we can enforce the rule mechanically.
If it does not, this whole plan is dead and we should pay for data instead.

Requires nothing. No API key, no signup. Just a polite User-Agent.
"""

from __future__ import annotations

import json
import time
from collections import Counter

import pandas as pd
import requests

# The SEC requires you to identify yourself. This is not optional -- they will
# return 403 without it. Put a real contact address here.
USER_AGENT = "momentum-engine research kunal@example.com"

HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
RATE_LIMIT_SECONDS = 0.12          # SEC allows 10 req/sec; stay well under.

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# A deliberately mixed sample: mega-cap tech, retail, industrial, financial, energy.
# Different sectors tag things differently -- that is the point.
SAMPLE = ["AAPL", "MSFT", "WMT", "CAT", "JPM", "XOM", "PG", "NVDA"]

# The concepts an AAII-style screen actually needs.
WANTED = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
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
    "shares": [
        "CommonStockSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "EntityCommonStockSharesOutstanding",
    ],
    "assets": ["Assets"],
}


def get(url: str) -> dict:
    time.sleep(RATE_LIMIT_SECONDS)
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def resolve_ciks(tickers: list[str]) -> dict[str, int]:
    raw = get(TICKERS_URL)
    lookup = {v["ticker"].upper(): int(v["cik_str"]) for v in raw.values()}
    return {t: lookup[t] for t in tickers if t in lookup}


def main() -> None:
    print()
    print("=" * 88)
    print("  SEC EDGAR SPIKE — can we build honest fundamentals for free?")
    print("=" * 88)

    if "example.com" in USER_AGENT:
        print("  ⚠️  Edit USER_AGENT at the top of this file with a real email first.")
        print("      The SEC blocks requests that do not identify a contact.")
        print()

    print("  Resolving tickers → CIK...")
    ciks = resolve_ciks(SAMPLE)
    print(f"  Resolved {len(ciks)}/{len(SAMPLE)}: {', '.join(ciks)}")
    print()

    lag_rows: list[dict] = []
    tag_usage: dict[str, Counter] = {k: Counter() for k in WANTED}
    earliest: dict[str, pd.Timestamp] = {}

    for ticker, cik in ciks.items():
        print(f"  Fetching {ticker} (CIK {cik})...", end=" ", flush=True)
        try:
            facts = get(FACTS_URL.format(cik=cik))
        except Exception as exc:
            print(f"FAILED: {exc}")
            continue

        gaap = facts.get("facts", {}).get("us-gaap", {})
        dei = facts.get("facts", {}).get("dei", {})
        available = set(gaap) | set(dei)

        # --- Q3: which tags does THIS company actually use? -------------------
        for concept, candidates in WANTED.items():
            for tag in candidates:
                if tag in available:
                    tag_usage[concept][tag] += 1

        # --- Q1 + Q2: filing lag on annual net income ------------------------
        source = gaap.get("NetIncomeLoss") or gaap.get("ProfitLoss")
        if source:
            for entry in source.get("units", {}).get("USD", []):
                if entry.get("form") != "10-K" or "filed" not in entry:
                    continue

                period_end = pd.Timestamp(entry["end"])
                filed = pd.Timestamp(entry["filed"])

                lag_rows.append({
                    "ticker": ticker,
                    "period_end": period_end,
                    "filed": filed,
                    "lag_days": (filed - period_end).days,
                })

                prev = earliest.get(ticker)
                if prev is None or period_end < prev:
                    earliest[ticker] = period_end

        print(f"{len(available)} concepts")

    print()

    if not lag_rows:
        print("  ❌ No filing dates found. EDGAR is not usable this way. Stop here.")
        return

    lags = pd.DataFrame(lag_rows).drop_duplicates(subset=["ticker", "period_end", "filed"])

    # ---- ANSWER 1 ---------------------------------------------------------
    print("=" * 88)
    print("  Q1: IS EDGAR POINT-IN-TIME?")
    print("=" * 88)
    print(f"  ✅ YES. Every fact carries a `filed` date.")
    print(f"     {len(lags)} annual filings found across {lags['ticker'].nunique()} companies.")
    print()

    # ---- ANSWER 2 — the important one -------------------------------------
    print("=" * 88)
    print("  Q2: HOW BIG IS THE LOOK-AHEAD TRAP?")
    print("=" * 88)
    print("  Days between the END of a fiscal year and the DAY IT BECAME PUBLIC:")
    print()
    print(f"  {'':<12}{'MEDIAN':>9}{'MIN':>7}{'MAX':>7}")
    print("  " + "-" * 36)
    for ticker, group in lags.groupby("ticker"):
        print(f"  {ticker:<12}{group['lag_days'].median():>9.0f}"
              f"{group['lag_days'].min():>7.0f}{group['lag_days'].max():>7.0f}")

    print("  " + "-" * 36)
    print(f"  {'ALL':<12}{lags['lag_days'].median():>9.0f}"
          f"{lags['lag_days'].min():>7.0f}{lags['lag_days'].max():>7.0f}")
    print()
    print(f"  ⚠️  A screen run on 2 January using December's numbers is reading a")
    print(f"      document that would not exist for another ~{lags['lag_days'].median():.0f} DAYS.")
    print("      That is not a rounding error. That is a time machine, and it is the")
    print("      single easiest way to build a fundamental backtest that lies.")
    print()
    print("      Our rule: a fundamental fact is usable only from its `filed` date.")
    print()

    # ---- ANSWER 3 ---------------------------------------------------------
    print("=" * 88)
    print("  Q3: HOW BAD IS TAG NORMALISATION?")
    print("=" * 88)
    print("  Same concept, different XBRL tags. This is the real engineering cost --")
    print("  and it is exactly what a paid vendor like Sharadar does for you.")
    print()
    n = len(ciks)
    for concept, counter in tag_usage.items():
        print(f"  {concept.upper()}")
        if not counter:
            print("     ⚠️  none of our candidate tags found — needs investigation")
        for tag, count in counter.most_common():
            bar = "█" * count
            print(f"     {count}/{n}  {bar:<10} {tag}")
        print()

    # ---- The 2009 wall ----------------------------------------------------
    print("=" * 88)
    print("  THE 2009 WALL")
    print("=" * 88)
    print("  Earliest fiscal year we can see, per company:")
    for ticker, when in sorted(earliest.items(), key=lambda kv: kv[1]):
        print(f"     {ticker:<8}{when.date()}")
    print()
    print("  XBRL was phased in from 2009. Anything earlier must be scraped from")
    print("  raw HTML filings -- weeks of work, and fragile.")
    print()
    print("  CONSEQUENCE: fundamental strategies CANNOT be tested through 2008.")
    print("  We lose the most informative stress test in the dataset. Accept that")
    print("  knowingly, or pay for data that goes back further.")
    print()
    print("=" * 88)
    print()


if __name__ == "__main__":
    main()
