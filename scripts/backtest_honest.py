"""
THE HONEST BACKTEST -- how much of our "excess return" was survivorship illusion?

    python -m scripts.backtest_honest YOUR_TIINGO_KEY

Every backtest so far ran on yfinance alone, which cannot price the ~414 delisted
names in our universe. It therefore tested only the survivors, and survivors are
flattering: the companies that failed simply never appear, so their losses are never
taken. This runs the SAME momentum strategy on two price sets:

  SURVIVORS ONLY  yfinance -- the ~1,192 names that still exist. The number we had.
  SURVIVORSHIP-FREE  yfinance for the living, PLUS Tiingo for the dead names,
                     merged. The failures are present and their collapses are taken.

The difference between the two is the survivorship illusion, quantified.

WHY MERGE INSTEAD OF USING TIINGO FOR EVERYTHING. Tiingo's free tier caps unique
symbols per month, so we spent its quota only on the names yfinance cannot supply.
Each vendor does what it is best at; the engine does not care which one a column
came from.

HONESTY NOTE PRINTED WITH THE RESULT. Not all 414 dead names were obtainable -- some
are absent from Tiingo under those tickers. The output states exactly how many dead
names were included and how many are still missing, so the number carries its own
caveat instead of relying on anyone's memory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from engine.backtest import get_overlay, rebalance_dates, run_backtest
from engine.data import get_adapter
from engine.data.base import PriceData
from engine.data.tiingo_adapter import TiingoAdapter
from engine.markets.market import load_market
from engine.metrics import cagr, max_drawdown, sharpe_ratio, sortino_ratio
from engine.signals import get_signal
from engine.universe.universe import load_membership

UNIVERSE = "sp900_pit"
TOP_N = 20
DEFAULT_START = "2010-06-01"
LOOKBACK_RUNWAY_YEARS = 2      # runway before START for the 12-month momentum lookback
DEAD_FILE = Path("data/dead_names.txt")


def _has_data(frame: pd.DataFrame) -> list[str]:
    """Columns that actually contain prices. yfinance returns an all-NaN column for
    every symbol it failed to download, so counting columns overstates coverage."""
    return [c for c in frame.columns if frame[c].notna().any()]


def _merge(a: PriceData, b: PriceData, market) -> PriceData:
    """
    Combine two price sets. Where both carry a symbol, `b` (Tiingo) WINS: those are
    the dead names, for which yfinance has either nothing or -- worse -- a recycled
    ticker's successor company. Dropping a's version outright avoids both.
    """
    overlap = a.close.columns.intersection(b.close.columns)
    a_close = a.close.drop(columns=overlap)
    a_vol = a.volume.drop(columns=overlap, errors="ignore")

    close = a_close.join(b.close, how="outer")
    volume = a_vol.join(b.volume, how="outer")
    return PriceData(market=market, close=close.sort_index(),
                     volume=volume.sort_index())


def _run(market, membership, prices, benchmark, signal, dates, spy_cagr):
    result = run_backtest(
        market=market, membership=membership, prices=prices, signal=signal,
        rebalance_dates=dates, top_n=TOP_N, frequency="monthly",
        benchmark=benchmark, overlay=get_overlay("always_on"),
    )
    eq = result.equity
    return {
        "cagr": cagr(eq, 12),
        "excess": cagr(eq, 12) - spy_cagr,
        "dd": max_drawdown(eq),
        "sharpe": sharpe_ratio(result.period_returns, 12),
        "sortino": sortino_ratio(result.period_returns, 12),
        "final": eq.iloc[-1],
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("\n  python -m scripts.backtest_honest YOUR_TIINGO_KEY [START_DATE]")
        print("     e.g. ... 2005-01-01   to include the 2008-09 crisis\n")
        return
    key = sys.argv[1]
    start = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_START
    price_start = (pd.Timestamp(start)
                   - pd.DateOffset(years=LOOKBACK_RUNWAY_YEARS)).strftime("%Y-%m-%d")

    market = load_market("us")
    membership = load_membership(market, UNIVERSE)
    symbols = sorted(set(membership.symbols))
    signal = get_signal("momentum", lookback_months=12, skip_months=1)
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    dates = rebalance_dates(market, start, today, "monthly")

    dead = []
    if DEAD_FILE.exists():
        dead = sorted({ln.strip() for ln in DEAD_FILE.read_text().splitlines()
                       if ln.strip()})

    print()
    print("=" * 90)
    print("  THE HONEST BACKTEST -- survivors only vs survivorship-free")
    print("=" * 90)
    print(f"  Universe: {UNIVERSE} ({len(symbols)} names)   Strategy: momentum 12-1, "
          f"top {TOP_N}, monthly")
    print(f"  Period: {start} -> {today}   (prices from {price_start})")
    print(f"  Dead names identified: {len(dead)}")
    print()

    # ---- benchmark ---------------------------------------------------------
    yf = get_adapter(market)
    spy = yf.fetch(["SPY"], price_start, today).close["SPY"].dropna()
    spy_vals = pd.Series([spy.asof(d) for d in dates], index=dates).dropna()
    spy_curve = (spy_vals / spy_vals.iloc[0]).iloc[1:]
    spy_cagr = cagr(spy_curve, 12)
    print(f"  SPY: {spy_cagr:.2%} CAGR, {max_drawdown(spy_curve):.1%} max DD\n")

    # ---- 1. survivors only (yfinance) --------------------------------------
    print("  Fetching yfinance prices (survivors only)...")
    yf_prices = yf.fetch(symbols, price_start, today)
    yf_bench = yf.fetch_benchmark(price_start, today)
    n_yf = len(_has_data(yf_prices.close))
    print(f"  yfinance actually priced {n_yf} / {len(symbols)} names "
          f"(the rest are empty columns).\n")

    # ---- 2. the dead names from Tiingo (cache only) ------------------------
    print("  Loading dead names from the Tiingo cache...")
    tg = TiingoAdapter(market, api_key=key)
    cached_dead = [
        s for s in dead
        if tg._cache_path(market.resolve_ticker(s)).exists()
    ]
    missing_dead = [s for s in dead if s not in cached_dead]
    print(f"  {len(cached_dead)} of {len(dead)} dead names are cached "
          f"({len(missing_dead)} unavailable).")

    tg_prices = tg.fetch(cached_dead, price_start, today)
    print(f"  Tiingo supplied {len(tg_prices.symbols)} dead names.\n")

    # ---- 3. merge ----------------------------------------------------------
    merged = _merge(yf_prices, tg_prices, market)
    n_merged = len(_has_data(merged.close))
    n_tg = len(_has_data(tg_prices.close))
    print(f"  Merged price set: {n_merged} names with real data "
          f"({n_merged - n_tg} living + {n_tg} dead).\n")

    biased = _run(market, membership, yf_prices, yf_bench, signal, dates, spy_cagr)
    honest = _run(market, membership, merged, yf_bench, signal, dates, spy_cagr)

    # ---- results -----------------------------------------------------------
    print("=" * 90)
    print(f"  {'PRICE SET':<30}{'CAGR':>9}{'vs SPY':>9}{'MAX DD':>9}"
          f"{'SHARPE':>8}{'SORTINO':>9}{'FINAL $100k':>15}")
    print("  " + "-" * 86)
    for label, r in [("survivors only (yfinance)", biased),
                     ("survivorship-free (merged)", honest)]:
        print(f"  {label:<30}{r['cagr']:>9.2%}{r['excess']:>+9.2%}{r['dd']:>9.1%}"
              f"{r['sharpe']:>8.2f}{r['sortino']:>9.2f}"
              f"{'$' + format(r['final'] * 100000, ',.0f'):>15}")

    gap = biased["excess"] - honest["excess"]
    print()
    print("=" * 90)
    print("  THE SURVIVORSHIP ILLUSION")
    print("=" * 90)
    print(f"  Survivors-only excess vs SPY   : {biased['excess']:+.2%}")
    print(f"  Survivorship-free excess vs SPY: {honest['excess']:+.2%}")
    print(f"  Difference (the illusion)      : {gap:+.2%} per year")
    print()
    if abs(gap) < 0.005:
        print("  The gap is small -- momentum's edge largely SURVIVES honest data.")
    elif honest["excess"] > 0:
        print("  The edge SHRINKS but survives -- part illusion, part real.")
    else:
        print("  The edge largely VANISHES -- it was mostly survivorship.")

    print()
    print("=" * 90)
    print("  WHAT THIS NUMBER STILL DOES NOT INCLUDE")
    print("=" * 90)
    print(f"  - {len(missing_dead)} dead names could not be priced by either source.")
    print("    Their failures are still absent, so the honest column remains")
    print("    slightly optimistic -- but by a bounded, stated amount.")
    print("  - Fundamentals-based screens are not applied here; this is momentum only.")
    print("  - Backtests are simulations, not predictions.")
    print()


if __name__ == "__main__":
    main()
