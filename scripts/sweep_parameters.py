"""
PARAMETER SENSITIVITY -- is 12-1 / top-20 special, or just one point on a plateau?

    python -m scripts.sweep_parameters YOUR_TIINGO_KEY [START_DATE]

THE LAST OVERFITTING QUESTION. Out-of-sample validation showed the edge holds across
time. This asks a different question: does the edge hold across NEARBY PARAMETER
CHOICES? We picked 12-month lookback, 1-month skip, top 20 -- but if only those exact
numbers work and 11-1 or 13-2 or top-25 collapse, the edge is a fragile artifact of a
lucky parameter pick, not a property of momentum.

CRITICAL DISCIPLINE -- this is NOT an optimiser. The temptation with a sweep is to run
the grid, pick the best cell, and report it. That is the very overfitting we are
testing for. This script does the opposite: it reports the WHOLE distribution and
asks whether our chosen (12, 1, 20) is TYPICAL among its neighbours. We never select
a winner. A broad plateau where most cells beat SPY is the good outcome; a single hot
cell surrounded by cold ones is the bad one.

The survivorship-free price set is built ONCE and reused for every cell, so the grid
is fast.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from engine.backtest import get_overlay, rebalance_dates, run_backtest
from engine.data import get_adapter
from engine.data.base import PriceData
from engine.data.tiingo_adapter import TiingoAdapter
from engine.markets.market import load_market
from engine.metrics import cagr, max_drawdown, sharpe_ratio
from engine.signals import get_signal
from engine.universe.universe import load_membership

UNIVERSE = "sp900_pit"
DEFAULT_START = "2005-01-01"
LOOKBACK_RUNWAY_YEARS = 2
DEAD_FILE = Path("data/dead_names.txt")

# The grid. Centred on our chosen (12, 1, 20), spanning sensible neighbours.
LOOKBACKS = [6, 9, 11, 12, 13, 15]
SKIPS = [0, 1, 2]
TOP_NS = [10, 15, 20, 25, 30]

CHOSEN = (12, 1, 20)


def _merge(a: PriceData, b: PriceData, market) -> PriceData:
    overlap = a.close.columns.intersection(b.close.columns)
    close = a.close.drop(columns=overlap).join(b.close, how="outer")
    volume = a.volume.drop(columns=overlap, errors="ignore").join(b.volume, how="outer")
    return PriceData(market=market, close=close.sort_index(),
                     volume=volume.sort_index())


def main() -> None:
    if len(sys.argv) < 2:
        print("\n  python -m scripts.sweep_parameters YOUR_TIINGO_KEY [START_DATE]\n")
        return
    key = sys.argv[1]
    start = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_START
    price_start = (pd.Timestamp(start)
                   - pd.DateOffset(years=LOOKBACK_RUNWAY_YEARS)).strftime("%Y-%m-%d")

    market = load_market("us")
    membership = load_membership(market, UNIVERSE)
    symbols = sorted(set(membership.symbols))
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    dates = rebalance_dates(market, start, today, "monthly")

    print()
    print("=" * 82)
    print("  PARAMETER SENSITIVITY -- robust plateau or lonely spike?")
    print("=" * 82)
    print(f"  {UNIVERSE}, survivorship-free, trend-filtered, {start} -> {today}")
    print(f"  Grid: lookback {LOOKBACKS} x skip {SKIPS} x topN {TOP_NS} "
          f"= {len(LOOKBACKS)*len(SKIPS)*len(TOP_NS)} cells")
    print(f"  Our chosen cell: lookback={CHOSEN[0]}, skip={CHOSEN[1]}, topN={CHOSEN[2]}\n")

    # ---- build the price set ONCE -----------------------------------------
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
    spy_curve = (spy / spy.iloc[0]).dropna()
    spy_cagr = cagr(spy_curve.iloc[1:], 12)
    print(f"  SPY over window: {spy_cagr:.2%} CAGR\n")

    # ---- SPEED: memoize eligible_universe across cells ---------------------
    # The eligible set depends only on (as_of, min_history_days) -- NOT on skip or
    # top_n. Across the grid there are only a handful of distinct min_history_days
    # values (one per lookback) and the rebalance dates are identical every cell.
    # Without caching, the expensive liquidity/history filter reruns ~250 dates x 84
    # cells = 21,000 times. With caching it runs ~250 x (distinct lookbacks) times.
    import engine.backtest.engine as _engine
    _orig_eligible = _engine.eligible_universe
    _cache: dict = {}

    def _memo_eligible(*, prices, membership, as_of, min_history_days, **kw):
        k = (pd.Timestamp(as_of), int(min_history_days))
        if k not in _cache:
            _cache[k] = _orig_eligible(
                prices=prices, membership=membership, as_of=as_of,
                min_history_days=min_history_days, **kw
            )
        return _cache[k]

    _engine.eligible_universe = _memo_eligible
    print("  Eligibility memoized across cells (skip/topN reuse the same universe).\n")
    print("  Running grid...\n")

    combos = [(lb, sk, tn) for lb in LOOKBACKS for sk in SKIPS
              for tn in TOP_NS if sk < lb]
    total = len(combos)

    results = []
    for n, (lb, sk, tn) in enumerate(combos, 1):
        signal = get_signal("momentum", lookback_months=lb, skip_months=sk)
        r = run_backtest(
            market=market, membership=membership, prices=prices,
            signal=signal, rebalance_dates=dates, top_n=tn,
            frequency="monthly", benchmark=yf_bench,
            overlay=get_overlay("trend_filter"),
        )
        eq = r.equity
        results.append({
            "lb": lb, "sk": sk, "tn": tn,
            "cagr": cagr(eq, 12),
            "excess": cagr(eq, 12) - spy_cagr,
            "sharpe": sharpe_ratio(r.period_returns, 12),
            "dd": max_drawdown(eq),
        })
        print(f"  [{n:>2}/{total}] lb={lb:>2} sk={sk} topN={tn:>2}  "
              f"excess {cagr(eq, 12) - spy_cagr:+.2%}")

    # restore the original (be a good citizen even though the script now exits)
    _engine.eligible_universe = _orig_eligible
    print()

    df = pd.DataFrame(results)

    # ---- distribution ------------------------------------------------------
    beat = (df["excess"] > 0).mean()
    print("=" * 82)
    print("  DISTRIBUTION ACROSS ALL CELLS")
    print("=" * 82)
    print(f"  Cells tested            : {len(df)}")
    print(f"  Beat SPY                : {(df['excess']>0).sum()}/{len(df)} ({beat:.0%})")
    print(f"  Excess vs SPY  median   : {df['excess'].median():+.2%}")
    print(f"                 min..max : {df['excess'].min():+.2%} .. {df['excess'].max():+.2%}")
    print(f"  Sharpe         median   : {df['sharpe'].median():.2f}")
    print(f"                 min..max : {df['sharpe'].min():.2f} .. {df['sharpe'].max():.2f}")

    # ---- where does our chosen cell sit? ----------------------------------
    chosen = df[(df.lb == CHOSEN[0]) & (df.sk == CHOSEN[1]) & (df.tn == CHOSEN[2])]
    if not chosen.empty:
        c = chosen.iloc[0]
        pct = (df["excess"] < c["excess"]).mean()
        print()
        print("=" * 82)
        print("  OUR CHOSEN CELL (12, 1, 20) IN CONTEXT")
        print("=" * 82)
        print(f"  Excess {c['excess']:+.2%}, Sharpe {c['sharpe']:.2f}, DD {c['dd']:.1%}")
        print(f"  It sits at the {pct:.0%} percentile of all cells by excess return.")
        if 0.30 <= pct <= 0.85:
            print("  -> TYPICAL, not exceptional. Good: the edge is not a lucky pick.")
        elif pct > 0.85:
            print("  -> Near the TOP. Suspicious -- did we (even implicitly) pick a peak?")
        else:
            print("  -> Below median. We did NOT cherry-pick a hot cell (reassuring).")

    # ---- lookback robustness (skip=1, topN=20 slice) ----------------------
    print()
    print("-" * 82)
    print("  SLICE: excess vs SPY by lookback (skip=1, topN=20)")
    print("-" * 82)
    sl = df[(df.sk == 1) & (df.tn == 20)].sort_values("lb")
    print(f"  {'lookback':<10}" + "".join(f"{lb:>8}" for lb in sl['lb']))
    print(f"  {'excess':<10}" + "".join(f"{e*100:>7.1f}%" for e in sl['excess']))

    # ---- topN robustness (lb=12, skip=1 slice) ----------------------------
    print()
    print("-" * 82)
    print("  SLICE: excess vs SPY by topN (lookback=12, skip=1)")
    print("-" * 82)
    sl = df[(df.lb == 12) & (df.sk == 1)].sort_values("tn")
    print(f"  {'topN':<10}" + "".join(f"{tn:>8}" for tn in sl['tn']))
    print(f"  {'excess':<10}" + "".join(f"{e*100:>7.1f}%" for e in sl['excess']))

    print()
    print("=" * 82)
    print("  READING THIS. A broad plateau -- most cells beating SPY, our choice")
    print("  TYPICAL rather than peak, neighbours similar -- means the edge is a")
    print("  property of momentum, not of a lucky parameter. A lonely spike (only our")
    print("  cell wins, neighbours collapse) would mean overfitting. We do NOT pick the")
    print("  best cell; being unremarkable among winners is exactly what we want.")
    print()


if __name__ == "__main__":
    main()
