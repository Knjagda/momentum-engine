"""
Live data check: fetch real prices for BOTH markets through the same code path.

Run:  python -m scripts.fetch_demo

This is the first time we touch the internet. Note what this script does NOT
contain: no ".NS", no "USD", no country branching. It loops over markets and
asks each one how to do its job.
"""

from engine.data import get_adapter
from engine.markets.market import load_market

# A few liquid names per market, just to prove the pipe works.
SAMPLES = {
    "us": ["AAPL", "MSFT", "NVDA"],
    "india": ["RELIANCE", "TCS", "INFY"],
}

START = "2024-01-01"
END = "2024-12-31"


def main() -> None:
    for market_key, symbols in SAMPLES.items():
        market = load_market(market_key)
        adapter = get_adapter(market)

        print()
        print("=" * 72)
        print(f"  {market.name}")
        print("=" * 72)
        print(f"  Requesting : {symbols}")
        print(f"  Vendor sees: {[market.resolve_ticker(s) for s in symbols]}")
        print(f"  Adapter    : {adapter}")
        print()

        try:
            data = adapter.fetch(symbols, START, END)
        except Exception as exc:
            print(f"  ❌ Fetch failed: {exc}")
            continue

        data = data.drop_incomplete()

        print(f"  ✅ {data}")
        print()
        print("  Last 3 adjusted closes:")
        tail = data.close.tail(3).round(2)
        for dt, row in tail.iterrows():
            values = "  ".join(
                f"{sym}={market.format_money(val)}" for sym, val in row.items()
            )
            print(f"    {dt.date()}   {values}")
        print()

        # Demonstrate the anti-look-ahead gate.
        cutoff = "2024-07-01"
        history = data.up_to(cutoff)
        print(f"  Look-ahead guard: as of {cutoff}, the engine can see")
        print(f"    {len(history.close)} days, ending {history.end.date()} "
              f"(strictly before {cutoff}) ✅")

    print()
    print("=" * 72)
    print("  Two countries. One code path. Zero country-specific logic above.")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()
