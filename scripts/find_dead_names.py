"""
Which universe names can yfinance NOT price? Those are the ones Tiingo must supply.

    python -m scripts.find_dead_names

Option C plan: use yfinance for the ~1,200 living names (free, fast, already works)
and spend Tiingo's tight free quota (500 unique symbols/month) ONLY on the dead names
yfinance drops. This script finds that dead set and writes it to a file the Tiingo
puller then targets -- so we never waste Tiingo quota on names yfinance already has.

A name is "dead to yfinance" if yfinance returns no usable price history for it over
the full window. Those are the delisted companies (LEH, SIVB, FRC, ...) whose absence
causes the survivorship bias.

Output: data/dead_names.txt  (one ticker per line)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from engine.data import get_adapter
from engine.markets.market import load_market
from engine.universe.universe import load_membership

UNIVERSE = "sp900_pit"
PRICE_START = "2008-06-01"
OUT = Path("data/dead_names.txt")


def main() -> None:
    market = load_market("us")
    membership = load_membership(market, UNIVERSE)
    symbols = sorted(set(membership.symbols))
    adapter = get_adapter(market)

    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    print()
    print("=" * 72)
    print(f"  FINDING DEAD NAMES — {UNIVERSE} ({len(symbols)} symbols)")
    print("=" * 72)
    print("  Fetching all via yfinance; whatever it can't price is 'dead'.\n")

    priced = adapter.fetch(symbols, PRICE_START, today)

    # Names that came back with at least SOME real data are "alive to yfinance".
    alive = set()
    for s in priced.symbols:
        col = priced.close[s] if s in priced.close.columns else None
        if col is not None and col.notna().sum() > 20:   # >20 real bars = usable
            alive.add(s)

    dead = [s for s in symbols if s not in alive]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(dead) + "\n")

    print("=" * 72)
    print(f"  Alive (yfinance can price) : {len(alive)}")
    print(f"  Dead  (need Tiingo)        : {len(dead)}")
    print("=" * 72)
    print(f"  Wrote {len(dead)} dead names to {OUT}")
    if len(dead) <= 500:
        print(f"  ✅ {len(dead)} ≤ 500 -- fits Tiingo's free monthly unique-symbol cap.")
    else:
        print(f"  ⚠️  {len(dead)} > 500 -- exceeds Tiingo free cap; will need 2 months")
        print("     or a one-month paid tier. We'll batch it.")
    print()
    print("  Next: python -m scripts.pull_tiingo_prices <KEY>  (now targets dead names)")
    print()


if __name__ == "__main__":
    main()
