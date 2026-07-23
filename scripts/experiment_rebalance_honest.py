"""
REBALANCE FREQUENCY -- is monthly worth its turnover, or should we trade quarterly?

    python -m scripts.experiment_rebalance_honest YOUR_TIINGO_KEY [START_DATE]

THE QUESTION WE HAVE NEVER ASKED. Everything so far has been rebalanced monthly, by
default rather than by decision. Measured turnover is 37% PER MONTH -- roughly 440%
a year, meaning the portfolio is replaced more than four times over. AAII rebalances
its 30-year Shadow Stock portfolio QUARTERLY, explicitly to keep real-world costs
down.

Two forces pull in opposite directions:
  MONTHLY catches momentum sooner. The signal decays, so acting on it late means
  holding fading winners -- which is exactly why the no-trade buffer FAILED when we
  tested it.
  QUARTERLY trades roughly a third as often. Costs are charged on every trade here,
  so less churn shows up directly as return.

The buffer test suggests monthly should win, because it showed that delaying exits
costs more than it saves. But a buffer and a slower rebalance are not the same thing:
a buffer holds losers longer while still buying winners promptly; quarterly delays
BOTH sides equally. So the result is genuinely not predictable from what we know.

THE TAX POINT, WHICH THE BACKTEST CANNOT SEE. These numbers charge commission and
slippage but NO tax. At 440% annual turnover essentially every gain in a taxable
account is short-term, taxed as ordinary income rather than at the long-term rate.
Quarterly does not fix that either (still under a year), but it materially reduces
the number of taxable events. This script reports turnover per year so the size of
that unmodelled cost is at least visible.
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
TOP_N = 25
SECTOR_CAP = 0.25
DEFAULT_START = "2005-01-01"
LOOKBACK_RUNWAY_YEARS = 2
DEAD_FILE = Path("data/dead_names.txt")

# (label, frequency string, periods per year)
FREQUENCIES = [
    ("monthly", "monthly", 12),
    ("quarterly", "quarterly", 4),
]


def _merge(a: PriceData, b: PriceData, market) -> PriceData:
    overlap = a.close.columns.intersection(b.close.columns)
    close = a.close.drop(columns=overlap).join(b.close, how="outer")
    volume = a.volume.drop(columns=overlap, errors="ignore").join(b.volume, how="outer")
    return PriceData(market=market, close=close.sort_index(),
                     volume=volume.sort_index())


def _turnover(result) -> float | None:
    """Average fraction of the portfolio replaced per rebalance."""
    try:
        prev, changes = None, []
        for p in result.portfolios:
            cur = set(p.symbols)
            if prev is not None and cur:
                changes.append(len(cur - prev) / max(len(cur), 1))
            prev = cur
        return float(pd.Series(changes).mean()) if changes else None
    except Exception:
        return None


def main() -> None:
    if len(sys.argv) < 2:
        print("\n  python -m scripts.experiment_rebalance_honest KEY [START_DATE]\n")
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

    print()
    print("=" * 92)
    print("  REBALANCE FREQUENCY -- monthly vs quarterly")
    print("=" * 92)
    print(f"  {UNIVERSE}, momentum 12-1, top {TOP_N}, {SECTOR_CAP:.0%} sector cap,")
    print(f"  equal weight, trend-filtered, survivorship-free.")
    print(f"  Period: {start} -> {today}\n")

    yf = get_adapter(market)
    yf_prices = yf.fetch(symbols, price_start, today)
    yf_bench = yf.fetch_benchmark(price_start, today)
    dead = sorted({ln.strip() for ln in DEAD_FILE.read_text().splitlines()
                   if ln.strip()}) if DEAD_FILE.exists() else []
    tg = TiingoAdapter(market, api_key=key)
    cached_dead = [s for s in dead
                   if tg._cache_path(market.resolve_ticker(s)).exists()]
    tg_prices = tg.fetch(cached_dead, price_start, today)
    prices = _merge(yf_prices, tg_prices, market)

    rows = []
    for label, freq, ppy in FREQUENCIES:
        try:
            dates = rebalance_dates(market, start, today, freq)
        except Exception as e:
            print(f"  {label}: could not build rebalance dates -- {e}")
            continue

        spy = yf_bench.reindex(dates, method="ffill")
        spy_cagr = cagr((spy / spy.iloc[0]).dropna().iloc[1:], ppy)

        try:
            r = run_backtest(
                market=market, membership=membership, prices=prices, signal=signal,
                rebalance_dates=dates, top_n=TOP_N, frequency=freq,
                benchmark=yf_bench, overlay=get_overlay("trend_filter"),
                weighting="equal", max_sector_weight=SECTOR_CAP,
            )
        except Exception as e:
            print(f"  {label}: run failed -- {e}")
            continue

        eq = r.equity
        tpr = _turnover(r)
        rows.append({
            "label": label, "ppy": ppy, "n_rebal": len(dates),
            "cagr": cagr(eq, ppy),
            "spy_cagr": spy_cagr,
            "excess": cagr(eq, ppy) - spy_cagr,
            "dd": max_drawdown(eq),
            "sharpe": sharpe_ratio(r.period_returns, ppy),
            "sortino": sortino_ratio(r.period_returns, ppy),
            "turn_per": tpr,
            "turn_yr": (tpr * ppy) if tpr is not None else None,
        })
        print(f"  {label:<11} {len(dates):>4} rebalances   excess {rows[-1]['excess']:+.2%}   "
              f"DD {rows[-1]['dd']:.1%}   Sharpe {rows[-1]['sharpe']:.2f}"
              + (f"   turnover {tpr:.0%}/period, {tpr*ppy:.0%}/yr"
                 if tpr is not None else ""))

    if len(rows) < 2:
        print("\n  Need both frequencies to compare. Check the 'quarterly' option "
              "is supported by rebalance_dates and run_backtest.\n")
        return

    print()
    print("=" * 92)
    print(f"  {'FREQUENCY':<12}{'CAGR':>9}{'vs SPY':>9}{'MAX DD':>9}{'SHARPE':>8}"
          f"{'SORTINO':>9}{'TURNOVER/YR':>14}{'TRADES/YR':>11}")
    print("  " + "-" * 88)
    for r in rows:
        ty = f"{r['turn_yr']:.0%}" if r["turn_yr"] is not None else "n/a"
        trades = (f"{r['turn_yr'] * TOP_N:.0f}" if r["turn_yr"] is not None else "n/a")
        print(f"  {r['label']:<12}{r['cagr']:>9.2%}{r['excess']:>+9.2%}{r['dd']:>9.1%}"
              f"{r['sharpe']:>8.2f}{r['sortino']:>9.2f}{ty:>14}{trades:>11}")

    m = next(r for r in rows if r["label"] == "monthly")
    q = next(r for r in rows if r["label"] == "quarterly")
    d_exc = (q["excess"] - m["excess"]) * 100
    d_dd = (q["dd"] - m["dd"]) * 100

    print()
    print("=" * 92)
    print("  QUARTERLY vs MONTHLY")
    print("=" * 92)
    print(f"  Excess return   {d_exc:+.2f}pt")
    print(f"  Drawdown        {abs(d_dd):.1f}pt "
          f"{'shallower' if d_dd > 0 else 'deeper'}")
    print(f"  Sharpe          {q['sharpe'] - m['sharpe']:+.2f}")
    print(f"  Sortino         {q['sortino'] - m['sortino']:+.2f}")
    if m["turn_yr"] and q["turn_yr"]:
        print(f"  Turnover        {m['turn_yr']:.0%}/yr -> {q['turn_yr']:.0%}/yr "
              f"({(1 - q['turn_yr']/m['turn_yr'])*100:.0f}% fewer trades)")

    print()
    print("=" * 92)
    print("  WHAT THE BACKTEST CANNOT SEE")
    print("=" * 92)
    print("  These numbers include commission and slippage but NO TAX. In a taxable")
    print("  account, gains held under a year are taxed as ordinary income rather")
    print("  than at the long-term rate. Neither frequency holds long enough to")
    print("  qualify, so quarterly does not fix that -- but it does cut the number of")
    print("  taxable events sharply.")
    print()
    print("  For a family-and-friends product this matters twice over: fewer trades")
    print("  also means less to execute by hand each period, and less opportunity to")
    print("  make a mistake. If quarterly costs little, it may be the better product")
    print("  even where it is the slightly worse backtest.")
    print()
    print("  Backtests are simulations, not predictions.")
    print()


if __name__ == "__main__":
    main()
