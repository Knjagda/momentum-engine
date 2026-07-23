"""
WEIGHTING AND BUFFER -- how should the 25 names be sized, and how often traded?

    python -m scripts.experiment_weighting_honest YOUR_TIINGO_KEY [START_DATE]

TWO QUESTIONS WE HAVE NEVER TESTED, both on the configuration we just chose
(25 holdings, 25% sector cap, trend-filtered, survivorship-free).

1. WEIGHTING. We have always used equal weight. The engine also supports
   inverse-volatility (naive risk parity -- smaller positions in wilder stocks) and
   capped. The research prior is that equal weight WINS for momentum specifically,
   because momentum is inherently a high-volatility strategy and inverse-vol fights
   the signal by underweighting the strongest movers. That is a prediction, so we
   test it rather than assert it.

2. THE NO-TRADE BUFFER. AAII's Shadow Stock portfolio buys at price-to-book <= 0.90
   but holds until it exceeds 1.00 -- a deliberate gap between "good enough to buy"
   and "bad enough to sell". We built the same mechanism (exit_rank) and have never
   switched it on. Without it, a name that drifts from rank 25 to rank 26 gets sold
   and rebought, handing the difference to the broker. We test exit ranks of 1.25x,
   1.5x and 2x the holding count.

WHY THIS MATTERS MORE THAN IT SOUNDS. Costs are charged on every trade in these
backtests, so a buffer that cuts turnover shows up directly as return. This is one
of the few changes that can improve the number without taking more risk.
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
TOP_N = 25                      # the count the holdings x cap grid selected
SECTOR_CAP = 0.25               # the global standard, now feasible at 25 names
DEFAULT_START = "2005-01-01"
LOOKBACK_RUNWAY_YEARS = 2
DEAD_FILE = Path("data/dead_names.txt")

WEIGHTINGS = ["equal", "inverse_vol"]
# exit_rank None = no buffer (what we ship). Others = hold until rank drifts past X.
EXIT_RANKS = [None, int(TOP_N * 1.25), int(TOP_N * 1.5), TOP_N * 2]


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
        print("\n  python -m scripts.experiment_weighting_honest KEY [START_DATE]\n")
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

    print()
    print("=" * 92)
    print("  WEIGHTING AND NO-TRADE BUFFER")
    print("=" * 92)
    print(f"  {UNIVERSE}, momentum 12-1, top {TOP_N}, {SECTOR_CAP:.0%} sector cap,")
    print(f"  trend-filtered, survivorship-free.  Period: {start} -> {today}\n")

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

    spy = yf_bench.reindex(dates, method="ffill")
    spy_cagr = cagr((spy / spy.iloc[0]).dropna().iloc[1:], 12)
    print(f"  SPY: {spy_cagr:.2%} CAGR\n")

    rows = []
    n, total = 0, len(WEIGHTINGS) * len(EXIT_RANKS)
    for wt in WEIGHTINGS:
        for ex in EXIT_RANKS:
            n += 1
            try:
                r = run_backtest(
                    market=market, membership=membership, prices=prices,
                    signal=signal, rebalance_dates=dates, top_n=TOP_N,
                    frequency="monthly", benchmark=yf_bench,
                    overlay=get_overlay("trend_filter"),
                    weighting=wt, max_sector_weight=SECTOR_CAP,
                    exit_rank=ex,
                )
            except (ValueError, TypeError) as e:
                print(f"  [{n}/{total}] {wt:<12} exit={str(ex):<5} SKIPPED -- {e}")
                continue
            eq = r.equity
            rows.append({
                "wt": wt, "exit": ex,
                "cagr": cagr(eq, 12),
                "excess": cagr(eq, 12) - spy_cagr,
                "dd": max_drawdown(eq),
                "sharpe": sharpe_ratio(r.period_returns, 12),
                "sortino": sortino_ratio(r.period_returns, 12),
                "turnover": _turnover(r),
            })
            t = rows[-1]["turnover"]
            print(f"  [{n}/{total}] {wt:<12} exit={str(ex):<5} "
                  f"excess {rows[-1]['excess']:+.2%}  DD {rows[-1]['dd']:.1%}  "
                  f"Sharpe {rows[-1]['sharpe']:.2f}"
                  + (f"  turnover {t:.0%}" if t is not None else ""))

    if not rows:
        print("\n  Nothing ran. Check that run_backtest accepts exit_rank.\n")
        return

    print()
    print("=" * 92)
    print(f"  {'WEIGHTING':<14}{'BUFFER':<10}{'CAGR':>9}{'vs SPY':>9}{'MAX DD':>9}"
          f"{'SHARPE':>8}{'SORTINO':>9}{'TURNOVER':>10}")
    print("  " + "-" * 88)
    for r in rows:
        ex = "none" if r["exit"] is None else f"rank {r['exit']}"
        tv = f"{r['turnover']:.0%}" if r["turnover"] is not None else "n/a"
        print(f"  {r['wt']:<14}{ex:<10}{r['cagr']:>9.2%}{r['excess']:>+9.2%}"
              f"{r['dd']:>9.1%}{r['sharpe']:>8.2f}{r['sortino']:>9.2f}{tv:>10}")

    base = next((r for r in rows if r["wt"] == "equal" and r["exit"] is None), None)
    if base:
        print()
        print("=" * 92)
        print("  VS EQUAL WEIGHT, NO BUFFER (what we ship today)")
        print("=" * 92)
        for r in rows:
            if r is base:
                continue
            ex = "none" if r["exit"] is None else f"rank {r['exit']}"
            d_dd = (r["dd"] - base["dd"]) * 100
            print(f"  {r['wt']:<14}{ex:<10} "
                  f"excess {(r['excess']-base['excess'])*100:+.2f}pt   "
                  f"drawdown {abs(d_dd):.1f}pt "
                  f"{'shallower' if d_dd > 0 else 'deeper'}   "
                  f"Sharpe {r['sharpe']-base['sharpe']:+.2f}")

    print()
    print("=" * 92)
    print("  HOW TO READ THIS")
    print("=" * 92)
    print("  WEIGHTING: if inverse-vol loses, that confirms the prior -- momentum is a")
    print("  high-volatility strategy and shrinking the wildest positions fights the")
    print("  signal. If it wins on Sharpe while costing return, it is the same trade")
    print("  the sector cap offered, and should be judged the same way.")
    print()
    print("  BUFFER: a wider exit rank means fewer trades. Because costs are charged")
    print("  here, less churn shows up directly as return. If a buffer improves the")
    print("  number, it is close to free -- the rare change that takes no extra risk.")
    print("  AAII has used exactly this asymmetry for 30 years (buy at P/B <= 0.90,")
    print("  hold until it exceeds 1.00).")
    print()
    print("  Backtests are simulations, not predictions.")
    print()


if __name__ == "__main__":
    main()
