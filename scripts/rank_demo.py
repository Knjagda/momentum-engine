"""
Rank real stocks. The first time the engine has an opinion.

Run:  python -m scripts.rank_demo

This wires together everything built so far:

    market config -> data adapter -> universe -> eligibility -> signal -> rank

Note what is NOT in this file: no country logic, no ".NS", no currency symbols,
no 252. It loops over markets and asks each one how to do its job.
"""

from __future__ import annotations

import pandas as pd

from engine.data import get_adapter
from engine.markets.market import load_market
from engine.signals import get_signal
from engine.universe.universe import eligible_universe, load_membership

# Which universe to rank in each market.
RUNS = [
    ("us", "sp500"),
    ("india", "nifty200"),
]

SIGNAL = "volar"
SIGNAL_PARAMS = {"lookback_months": 12, "skip_months": 1, "vol_window_days": 126}

AS_OF = pd.Timestamp("2025-01-02")     # the "rebalance date"
HISTORY_START = "2022-06-01"           # enough runway for a 12-month lookback
TOP_N = 15


def main() -> None:
    signal = get_signal(SIGNAL, **SIGNAL_PARAMS)

    for market_key, universe_key in RUNS:
        market = load_market(market_key)
        membership = load_membership(market, universe_key)
        adapter = get_adapter(market)

        print()
        print("=" * 78)
        print(f"  {market.name}  ·  {membership.universe_key}  ·  signal = {signal.name}")
        print("=" * 78)
        print(f"  As of      : {AS_OF.date()}  (rebalance date)")
        print(f"  Universe   : {len(membership)} members")
        print(f"  Signal     : {signal}")
        print(f"  Needs      : {signal.required_history_days} trading days of history")
        print()
        print("  Fetching prices... (first run is slow; afterwards it is cached)")

        data = adapter.fetch(membership.symbols, HISTORY_START, AS_OF)

        # Who is actually investable on this date? (as-of, no peeking ahead)
        snapshot = eligible_universe(
            prices=data,
            membership=membership,
            as_of=AS_OF,
            min_history_days=signal.required_history_days,
        )

        print(f"  Eligible   : {snapshot.n_eligible} of {len(membership)}")
        for reason, count in sorted(snapshot.drop_reasons().items()):
            print(f"      dropped {count:>3}  ({reason})")
        print()

        result = signal.compute(data, AS_OF, symbols=snapshot.eligible)
        top = result.top(TOP_N)

        print(f"  TOP {TOP_N} BY {signal.name.upper()}")
        print(f"  {'#':<4}{'SYMBOL':<14}{'SCORE':>9}   {'SECTOR':<28}")
        print(f"  {'-' * 70}")
        for i, (symbol, score) in enumerate(top.items(), start=1):
            sector = membership.sector_of(symbol)[:27]
            print(f"  {i:<4}{symbol:<14}{score:>9.2f}   {sector:<28}")
        print()

        if membership.disclaimer:
            print(f"  {membership.disclaimer}")
        print()

    print("=" * 78)
    print("  Two countries, one pipeline. Rankings are rules, not opinions.")
    print("  ⚠️  Backtests and rankings are NOT predictions.")
    print("=" * 78)
    print()


if __name__ == "__main__":
    main()
