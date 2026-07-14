"""
Show every market the engine can run, side by side.

Run:  python -m scripts.show_markets

This is the "toggle" made visible. Notice that this script contains no
country-specific logic at all -- it just loops over whatever config files
exist and asks each Market object about itself.
"""

from engine.markets.market import available_markets, load_market


def main() -> None:
    keys = available_markets()

    print()
    print("=" * 78)
    print("  MARKETS THE ENGINE CAN CURRENTLY RUN")
    print("=" * 78)
    print(f"  Found {len(keys)} market config(s): {', '.join(keys)}")
    print("  (Adding another market = adding a YAML file. No code changes.)")
    print()

    for key in keys:
        m = load_market(key)

        print("-" * 78)
        print(f"  {m.name}   [load_market('{key}')]")
        print("-" * 78)
        print(f"  Currency          : {m.currency} ({m.currency_symbol})")
        print(f"  Trading calendar  : {m.calendar}")
        print(f"  Benchmark         : {m.benchmark.name}  ({m.benchmark.ticker})")
        print(f"  Data adapter      : {m.data_adapter}")
        print(f"  Ticker convention : {m.resolve_ticker('EXAMPLE')}")
        print()

        print(f"  Universes:")
        for uni in m.universes.values():
            flag = " ⚠️ survivorship bias" if uni.survivorship_bias else ""
            print(f"    - {uni.key:<12} {uni.name}{flag}")
        print()

        print(f"  Trading costs (every simulated trade pays these):")
        print(f"    Buy         : {m.costs.buy_cost_bps():>6.2f} bps")
        print(f"    Sell        : {m.costs.sell_cost_bps():>6.2f} bps")
        print(f"    Round trip  : {m.costs.round_trip_bps():>6.2f} bps")
        print()

        print(f"  Example: {m.format_money(100000)} portfolio, one full rebalance")
        cost = 100_000 * m.costs.round_trip_bps() / 10_000
        print(f"    costs roughly {m.format_money(cost)} in fees/taxes/slippage.")
        print()

    print("=" * 78)
    print("  Same engine. Same code path. The only difference is the config file.")
    print("=" * 78)
    print()


if __name__ == "__main__":
    main()
