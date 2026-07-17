"""
Confirm the placeholder-trim works on REAL Tiingo data.

    python -m scripts.verify_tiingo_delisting YOUR_TIINGO_KEY

Piece 1 trims only the FROZEN zero-volume placeholder tail Tiingo pads onto delisted
tickers, keeping all real trading (including a failed stock's penny afterlife, which
our liquidity filters reject from ever being bought). This shows, per ticker: raw bar
count, cleaned bar count, how many placeholder bars were trimmed, and the last kept
date + price.
"""
from __future__ import annotations
import sys
from engine.data.tiingo_core import fetch_raw, clean_series

CASES = [
    ("SIVB", "failed 2023-03, penny afterlife to ~2024-11, then frozen"),
    ("FRCB", "failed 2023-05, trades at ~$0 with volume to today (no trim)"),
    ("YHOO", "acquired 2017-06 (short frozen tail trimmed)"),
    ("CELG", "acquired 2019-11"),
    ("AAPL", "alive (nothing trimmed)"),
]


def main() -> None:
    if len(sys.argv) < 2:
        print("\n  python -m scripts.verify_tiingo_delisting YOUR_TIINGO_KEY\n"); return
    key = sys.argv[1]

    print()
    print("=" * 88)
    print("  TIINGO PLACEHOLDER TRIM — real data")
    print("=" * 88)
    print(f"  {'TICKER':<7}{'RAW':>7}{'CLEAN':>7}{'TRIMMED':>9}{'LAST KEPT':>26}   NOTE")
    print("  " + "-" * 84)
    for ticker, note in CASES:
        raw = fetch_raw(ticker, key)
        if raw.empty:
            print(f"  {ticker:<7}{'(no data)':>7}")
            continue
        adj, vol = clean_series(raw)
        trimmed = len(raw) - len(adj)
        if not adj.empty:
            last = f"{adj.index[-1].date()} @ ${adj.iloc[-1]:,.2f}"
        else:
            last = "-"
        print(f"  {ticker:<7}{len(raw):>7,}{len(adj):>7,}{trimmed:>9,}{last:>26}   {note}")
    print()
    print("  ✅ if failed/acquired names had their frozen tails trimmed while keeping")
    print("  the real collapse, and AAPL trimmed 0. The penny afterlife staying is")
    print("  fine -- liquidity filters refuse to buy sub-dollar illiquid names.")
    print()


if __name__ == "__main__":
    main()
