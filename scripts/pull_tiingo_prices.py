"""
Pull DELISTED names' prices through Tiingo, once, into the cache (Option C).

    python -m scripts.pull_tiingo_prices YOUR_TIINGO_KEY

WHY ONLY DEAD NAMES. Tiingo's free tier caps at 500 UNIQUE SYMBOLS/MONTH, so we can't
pull the whole ~1600 universe free. But we don't need to: yfinance already prices the
~1200 living names free. We spend Tiingo's quota ONLY on the ~410 delisted names
yfinance can't price -- the ones causing survivorship bias. That fits under 500/month
and stays free. Run scripts.find_dead_names first to produce the target list.

RESUMABLE BY DESIGN. Free tier is ~50 symbols/hour, so ~410 names take ~9 hourly runs.
This script:
  - fetches one ticker at a time and caches it immediately (CSV, no parquet needed),
  - SKIPS any ticker already cached (instant), so re-running resumes where it left off,
  - stops cleanly when rate-limited and tells you to re-run.

Run it repeatedly until it reports 0 remaining. Then the honest backtest merges
yfinance (living) + Tiingo (dead) for a complete, survivorship-free price set.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from engine.data.tiingo_adapter import TiingoAdapter
from engine.data.tiingo_core import TiingoError
from engine.markets.market import load_market
from engine.universe.universe import load_membership

UNIVERSE = "sp900_pit"
START = "2004-01-01"      # deep history so momentum lookbacks are covered
END = "2026-12-31"
DEAD_FILE = Path("data/dead_names.txt")
# Tickers Tiingo has already told us it does not have. A miss costs an API call and
# caches NOTHING, so without this record we re-ask about the same ~52 names every
# round and burn the entire 50/hour quota on questions already answered.
MISS_FILE = Path("data/tiingo_misses.txt")


def _load_misses() -> set[str]:
    if MISS_FILE.exists():
        return {ln.strip() for ln in MISS_FILE.read_text().splitlines() if ln.strip()}
    return set()


def _record_miss(ticker: str) -> None:
    MISS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with MISS_FILE.open("a", encoding="utf-8") as fh:
        fh.write(ticker.upper() + "\n")


def _load_targets(market) -> list[str]:
    """Prefer the dead-names file; fall back to the full universe if absent."""
    if DEAD_FILE.exists():
        names = [ln.strip() for ln in DEAD_FILE.read_text().splitlines() if ln.strip()]
        print(f"  Targeting {len(names)} DEAD names from {DEAD_FILE}")
        return sorted(set(names))
    membership = load_membership(market, UNIVERSE)
    print(f"  {DEAD_FILE} not found -- targeting full universe "
          f"({len(membership.symbols)}). Run find_dead_names first for Option C.")
    return sorted(set(membership.symbols))


def main() -> None:
    if len(sys.argv) < 2:
        print("\n  python -m scripts.pull_tiingo_prices YOUR_TIINGO_KEY\n")
        return
    key = sys.argv[1]

    market = load_market("us")
    symbols = _load_targets(market)
    adapter = TiingoAdapter(market, api_key=key)

    print()
    print("=" * 72)
    print(f"  TIINGO PULL - {len(symbols)} target symbols")
    print("=" * 72)
    print("  Cached tickers are skipped instantly. Re-run anytime to resume.")
    print("  Free tier ~50/hr and 500 unique/month, so pace across a few days.\n")

    known_misses = _load_misses()
    if known_misses:
        print(f"  Skipping {len(known_misses)} tickers already known absent "
              f"(see {MISS_FILE}).\n")

    done, fetched, missing, failed = 0, 0, 0, 0
    t0 = time.time()

    for i, sym in enumerate(symbols, 1):
        ticker = market.resolve_ticker(sym)
        cache = adapter._cache_path(ticker)

        if cache.exists():
            done += 1
            continue

        # A previously-recorded miss costs nothing to skip -- and skipping it is the
        # whole point: re-asking burns an API call from a 50/hour budget.
        if ticker.upper() in known_misses:
            missing += 1
            continue

        try:
            adj, vol = adapter._fetch_one(ticker, START, END)
            if adj.empty:
                missing += 1
                _record_miss(ticker)
                known_misses.add(ticker.upper())
                status = "- no data (unknown to Tiingo) [recorded, won't retry]"
            else:
                fetched += 1
                status = f"+ {len(adj):,} bars, {adj.index[0].date()}->{adj.index[-1].date()}"
        except TiingoError as e:
            failed += 1
            print(f"  [{i}/{len(symbols)}] {sym:<8} x {e}")
            print("\n  Stopping (likely hourly limit). Re-run the same command to resume.")
            break
        except Exception as e:  # noqa: BLE001
            failed += 1
            status = f"x {type(e).__name__}: {e}"

        print(f"  [{i}/{len(symbols)}] {sym:<8} {status}")

    remaining = len(symbols) - done - fetched - missing
    print()
    print("=" * 72)
    print(f"  Already cached : {done}")
    print(f"  Newly fetched  : {fetched}")
    print(f"  Unknown/dead   : {missing}  (not in Tiingo -- expected for some names)")
    print(f"  Remaining      : {max(0, remaining)}")
    print("=" * 72)
    if remaining > 0:
        print("  Not finished -- re-run the same command to fetch the rest.")
    else:
        print("  OK All target names cached. The honest backtest can now run.")
    print()


if __name__ == "__main__":
    main()
