"""
Show the REAL shape of delisted tickers' tails, so we detect death correctly.

    python -m scripts.inspect_tiingo_tails YOUR_TIINGO_KEY

My first detection logic (trailing zero-volume run) failed on real data: SIVB's
post-collapse OTC afterlife has sporadic tiny volume, and acquired names (YHOO,
CELG) just end cleanly with no tail at all. Before rewriting detection, LOOK at the
actual bars: where does real trading stop, and what does the placeholder look like?
"""
from __future__ import annotations
import sys
from engine.data.tiingo_core import fetch_raw

def main() -> None:
    if len(sys.argv) < 2:
        print("\n  python -m scripts.inspect_tiingo_tails YOUR_TIINGO_KEY\n"); return
    key = sys.argv[1]

    for ticker, note in [("SIVB", "failed 2023-03"), ("FRCB", "failed 2023-05"),
                         ("YHOO", "acquired 2017-06")]:
        df = fetch_raw(ticker, key)
        print("\n" + "=" * 70)
        print(f"  {ticker}  ({note}) — {len(df):,} bars, "
              f"{df.index[0].date()} → {df.index[-1].date()}")
        print("=" * 70)
        if df.empty:
            print("  no data"); continue

        # Show the transition: find last bar with volume > some real threshold,
        # then print a window around it plus the very end.
        vol = df["adjVolume"] if "adjVolume" in df.columns else df["volume"]
        real = vol[vol > 1000]      # bars with meaningful volume
        if not real.empty:
            last_real = real.index[-1]
            pos = df.index.get_loc(last_real)
            print(f"  Last bar with volume>1000: {last_real.date()} "
                  f"(bar {pos} of {len(df)-1})")
            print(f"\n  --- 5 bars around last real trading ---")
            lo = max(0, pos - 2); hi = min(len(df), pos + 4)
            _dump(df.iloc[lo:hi], vol)
        print(f"\n  --- final 6 bars (the placeholder tail) ---")
        _dump(df.iloc[-6:], vol)


def _dump(sub, vol):
    print(f"  {'DATE':<12}{'CLOSE':>10}{'ADJCLOSE':>11}{'VOLUME':>14}")
    for dt, row in sub.iterrows():
        c = row.get("close", float('nan'))
        a = row.get("adjClose", float('nan'))
        v = vol.get(dt, 0)
        print(f"  {str(dt.date()):<12}{c:>10.2f}{a:>11.2f}{v:>14,.0f}")


if __name__ == "__main__":
    main()
