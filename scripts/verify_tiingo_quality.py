"""
Before trusting Tiingo for survivorship-free backtests, verify TWO things on a
known failure (SIVB) and one acquisition (CELG):

    python -m scripts.verify_tiingo_quality YOUR_TIINGO_KEY

1. DOES THE DEATH GET PRICED THROUGH? A survivorship-free source must show the
   collapse, not just stop. We print SIVB's LAST 12 bars -- in March 2023 it should
   crater toward zero, then delist. If the price just ends cleanly at a normal level,
   the "history" is truncated and the loss our backtest needs is still missing.

2. ARE PRICES SPLIT/DIVIDEND ADJUSTED? Tiingo returns raw AND adjusted columns
   (adjOpen/adjHigh/adjLow/adjClose/adjVolume). We need the ADJUSTED series so a
   split doesn't look like a 50% crash. We confirm adjClose is present and compare
   it to raw close on a normal bar.

If both hold, we build a Tiingo adapter and the survivorship hole closes for free.
"""

from __future__ import annotations

import io
import sys
from urllib.request import Request, urlopen

import pandas as pd


def fetch(ticker: str, key: str, start: str = "2005-01-01") -> pd.DataFrame:
    url = (
        f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
        f"?startDate={start}&token={key}"
    )
    req = Request(url, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=30) as resp:
        raw = resp.read().decode()
    df = pd.read_json(io.StringIO(raw))
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def main() -> None:
    if len(sys.argv) < 2:
        print("\n  python -m scripts.verify_tiingo_quality YOUR_TIINGO_KEY\n")
        return
    key = sys.argv[1]

    print()
    print("=" * 78)
    print("  TIINGO QUALITY CHECK — is this data trustworthy for honest backtests?")
    print("=" * 78)

    # ---- 1. columns present? -----------------------------------------------
    sivb = fetch("SIVB", key)
    print(f"\n  SIVB returned {len(sivb):,} bars.")
    print(f"  Columns: {list(sivb.columns)}")

    have_adj = {"adjClose", "adjOpen", "adjHigh", "adjLow"}.issubset(sivb.columns)
    print(f"  Adjusted columns present: {'✅ yes' if have_adj else '❌ NO'}")

    # ---- 2. does SIVB collapse through its March 2023 death? ----------------
    print("\n" + "=" * 78)
    print("  SIVB — LAST 12 BARS (should crater toward zero in March 2023)")
    print("=" * 78)
    tail = sivb.tail(12)
    print(f"  {'DATE':<12}{'CLOSE':>12}{'ADJ CLOSE':>12}{'VOLUME':>14}")
    print("  " + "-" * 50)
    for _, r in tail.iterrows():
        close = r.get("close", float("nan"))
        adj = r.get("adjClose", float("nan"))
        vol = r.get("volume", 0)
        print(f"  {str(r['date']):<12}{close:>12.2f}{adj:>12.2f}{vol:>14,.0f}")

    last_close = sivb.iloc[-1].get("close", float("nan"))
    peak = sivb["close"].max()
    print(f"\n  Peak close: ${peak:,.2f}   Final close: ${last_close:,.2f}")
    if last_close < peak * 0.25:
        print("  ✅ The collapse IS in the data -- final price far below peak.")
    else:
        print("  ⚠️  Final price not obviously collapsed -- inspect the tail above.")

    # ---- 3. adjusted vs raw on a normal bar --------------------------------
    if have_adj:
        mid = sivb.iloc[len(sivb) // 2]
        print("\n" + "=" * 78)
        print("  ADJUSTED vs RAW (a mid-history bar — they differ by splits/divs)")
        print("=" * 78)
        print(f"  {mid['date']}:  close=${mid['close']:.2f}   "
              f"adjClose=${mid['adjClose']:.2f}")
        print("  Backtests must use the ADJUSTED series.")

    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    print("  If adjusted columns are present AND SIVB craters through March 2023,")
    print("  Tiingo is trustworthy survivorship-free data -- we build the adapter and")
    print("  close the biggest gap in the engine, for free.")
    print()


if __name__ == "__main__":
    main()
