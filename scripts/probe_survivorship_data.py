"""
Do free data sources actually have prices for DEAD companies? A concrete test.

    python -m scripts.probe_survivorship_data                 # Stooq only (no key)
    python -m scripts.probe_survivorship_data TIINGO_KEY      # Stooq + Tiingo

THE QUESTION. yfinance cannot price the ~410 delisted names in our universe (LEH,
SIVB, FRC, ...). Paid data (Sharadar/Norgate) fixes it for ~$50. Before spending,
we check whether a FREE source already has these prices. This probe tries a handful
of known-dead tickers against Stooq (free, no key) and Tiingo (free key, 50 sym/hr)
and reports, per source, whether real price history came back.

WHAT "PASS" MEANS. The source returns multiple years of daily bars for a company we
KNOW is dead -- i.e. it retained the ticker after delisting instead of dropping it
like yfinance does. If a source passes on most of these, it's worth building an
adapter around. If it fails, we've earned the conclusion that paid data is needed.

We test six names with known death dates:
  LEH   Lehman Brothers        bankrupt Sep 2008
  SIVB  Silicon Valley Bank    failed Mar 2023
  FRC   First Republic Bank    failed May 2023
  YHOO  Yahoo                  acquired 2017
  TWTR  Twitter                taken private 2022
  CELG  Celgene                acquired by BMS 2019
"""

from __future__ import annotations

import io
import sys
import time
from urllib.request import Request, urlopen

import pandas as pd

# (display name, [symbol variants to try], approx death year)
DEAD = [
    ("Lehman Brothers", ["leh.us", "lehmq.us"], 2008),
    ("Silicon Valley Bank", ["sivb.us", "sivbq.us"], 2023),
    ("First Republic Bank", ["frc.us", "frcb.us"], 2023),
    ("Yahoo", ["yhoo.us"], 2017),
    ("Twitter", ["twtr.us"], 2022),
    ("Celgene", ["celg.us"], 2019),
]


def _get(url: str, headers: dict | None = None, timeout: int = 30) -> bytes:
    req = Request(url, headers=headers or {"User-Agent": "Mozilla/5.0 research"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


# ----------------------------------------------------------------------------
# Stooq: direct CSV endpoint, no key. Dead tickers use their delisted symbol.
#   https://stooq.com/q/d/l/?s=SYMBOL&i=d   -> CSV of daily OHLCV
# ----------------------------------------------------------------------------

def probe_stooq(symbols: list[str]) -> tuple[bool, str]:
    for sym in symbols:
        url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
        try:
            raw = _get(url).decode("utf-8", errors="replace")
        except Exception as e:
            return False, f"error: {e}"
        # Stooq returns literally "No data" (or a tiny body) when it has nothing.
        if "Date" not in raw.splitlines()[0] if raw.strip() else True:
            continue
        try:
            df = pd.read_csv(io.StringIO(raw))
        except Exception:
            continue
        if len(df) > 250:  # more than ~1 trading year of bars = real history
            span = f"{df['Date'].iloc[0]} → {df['Date'].iloc[-1]}"
            return True, f"{sym}: {len(df):,} bars ({span})"
    return False, "no data for any symbol variant"


# ----------------------------------------------------------------------------
# Tiingo: real API, free key. 30+ yrs history claimed. Uses plain ticker.
#   https://api.tiingo.com/tiingo/daily/<ticker>/prices?startDate=...
# ----------------------------------------------------------------------------

def probe_tiingo(symbols: list[str], key: str) -> tuple[bool, str]:
    # Tiingo uses the plain ticker (no .us); derive it from the first variant.
    for sym in symbols:
        ticker = sym.split(".")[0].upper()
        url = (
            f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
            f"?startDate=2005-01-01&token={key}"
        )
        try:
            raw = _get(url, headers={"Content-Type": "application/json"}).decode()
        except Exception as e:
            # 404 = ticker not found; keep trying variants
            if "404" in str(e):
                continue
            return False, f"error: {e}"
        try:
            df = pd.read_json(io.StringIO(raw))
        except Exception:
            continue
        if len(df) > 250:
            span = f"{df['date'].iloc[0]} → {df['date'].iloc[-1]}"
            return True, f"{ticker}: {len(df):,} bars ({str(span)[:40]})"
        time.sleep(1.5)  # be polite to the free tier
    return False, "no data for any symbol variant"


def main() -> None:
    tiingo_key = sys.argv[1] if len(sys.argv) > 1 else None

    print()
    print("=" * 78)
    print("  SURVIVORSHIP DATA PROBE — do free sources have prices for DEAD tickers?")
    print("=" * 78)
    print("  A source PASSES if it returns >1 year of daily bars for a known-dead name.")
    print("  yfinance fails ALL of these -- that's the gap we're trying to fill free.")
    print()

    stooq_pass = 0
    tiingo_pass = 0

    header = f"  {'COMPANY':<22}{'STOOQ':<34}"
    if tiingo_key:
        header += "TIINGO"
    print(header)
    print("  " + "-" * 74)

    for name, symbols, _year in DEAD:
        ok_s, msg_s = probe_stooq(symbols)
        stooq_pass += ok_s
        mark_s = "✅ " if ok_s else "❌ "
        line = f"  {name:<22}{mark_s + msg_s:<34}"

        if tiingo_key:
            ok_t, msg_t = probe_tiingo(symbols, tiingo_key)
            tiingo_pass += ok_t
            line += ("✅ " if ok_t else "❌ ") + msg_t

        print(line)
        time.sleep(0.5)

    print()
    print("=" * 78)
    print("  RESULT")
    print("=" * 78)
    print(f"  Stooq : {stooq_pass}/{len(DEAD)} dead tickers had real price history")
    if tiingo_key:
        print(f"  Tiingo: {tiingo_pass}/{len(DEAD)} dead tickers had real price history")
    else:
        print("  Tiingo: not tested (pass a free API key to include it)")
    print()
    print("  If a source got most/all of these, it retains delisted tickers -- worth")
    print("  building an adapter around, and the survivorship hole closes for free.")
    print("  If both mostly failed, paid data (Sharadar ~$50 once) is the honest fix.")
    print()


if __name__ == "__main__":
    main()
