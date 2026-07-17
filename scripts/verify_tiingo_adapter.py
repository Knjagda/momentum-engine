"""
End-to-end check of the TiingoAdapter on real data.

    python -m scripts.verify_tiingo_adapter YOUR_TIINGO_KEY

Pulls a few live + dead tickers THROUGH the adapter (not the raw core) and shows the
assembled PriceData: which symbols came back, the date span, and that a dead name
(SIVB) carries its collapse. Proves the adapter is ready for a real backtest.
"""
from __future__ import annotations
import sys
from engine.data import get_adapter
from engine.markets.market import load_market


def main() -> None:
    if len(sys.argv) < 2:
        print("\n  python -m scripts.verify_tiingo_adapter YOUR_TIINGO_KEY\n"); return
    key = sys.argv[1]

    market = load_market("us")
    from engine.data.tiingo_adapter import TiingoAdapter
    adapter = TiingoAdapter(market, api_key=key)

    # mix of alive (AAPL, MSFT) and dead (SIVB, CELG) names
    symbols = ["AAPL", "MSFT", "SIVB", "CELG"]
    print(f"\n  Fetching {symbols} via TiingoAdapter (2015-2024)...")
    data = adapter.fetch(symbols, "2015-01-01", "2024-12-31")

    print(f"\n  Symbols returned: {data.symbols}")
    print(f"  Date span: {data.start.date()} → {data.end.date()}")
    print(f"  Shape: {data.close.shape[0]} days × {data.close.shape[1]} symbols\n")

    for s in data.symbols:
        col = data.close[s].dropna()
        if col.empty:
            continue
        print(f"  {s:<6} {col.index[0].date()} → {col.index[-1].date()}  "
              f"first ${col.iloc[0]:,.2f}  last ${col.iloc[-1]:,.2f}  "
              f"peak ${col.max():,.2f}")

    print("\n  ✅ If SIVB shows a high peak and a low last price (its collapse), and")
    print("  alive names run to 2024, the adapter is delivering survivorship-free data.")
    print("  Second run should be instant (per-ticker cache).\n")


if __name__ == "__main__":
    main()
