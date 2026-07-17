"""
THE HONEST BACKTEST — how much of our "excess return" was survivorship illusion?

    python -m scripts.backtest_honest YOUR_TIINGO_KEY

Every backtest so far ran on yfinance, which cannot price the ~410 delisted names in
the universe -- so it silently tested only survivors, inflating returns. We now have
Tiingo, which prices the dead names through their collapse. This runs the SAME
momentum strategy on BOTH sources, side by side:

    yfinance  (survivorship-BIASED)  -- survivors only, the number we had
    Tiingo    (survivorship-FREE)    -- includes the failures, the honest number

The DIFFERENCE between them is the survivorship illusion, quantified. If our +8.74%
excess shrinks a lot, much of it was never real. If it mostly holds, momentum is
sturdier than we feared. Either way it's the first honest number the engine produces.

PREREQUISITE: run scripts.pull_tiingo_prices first until the universe is cached, so
this reads from disk and is fast. Uncached tickers would make this crawl.
"""

from __future__ import annotations

import sys

import pandas as pd

from engine.backtest import get_overlay, rebalance_dates, run_backtest
from engine.data import get_adapter
from engine.data.tiingo_adapter import TiingoAdapter
from engine.markets.market import load_market
from engine.metrics import cagr, max_drawdown, sharpe_ratio, sortino_ratio
from engine.signals import get_signal
from engine.universe.universe import load_membership

UNIVERSE = "sp900_pit"
TOP_N = 20
START = "2010-06-01"
PRICE_START = "2008-06-01"     # runway for 12-month momentum lookback


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
        print("\n  python -m scripts.backtest_honest YOUR_TIINGO_KEY\n")
        return
    key = sys.argv[1]

    market = load_market("us")
    membership = load_membership(market, UNIVERSE)
    symbols = membership.symbols
    signal = get_signal("momentum", lookback_months=12, skip_months=1)
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    dates = rebalance_dates(market, START, today, "monthly")

    print()
    print("=" * 88)
    print("  THE HONEST BACKTEST — survivorship-biased (yfinance) vs free (Tiingo)")
    print("=" * 88)
    print(f"  Universe: {UNIVERSE} ({len(symbols)} names)   Strategy: momentum 12-1, "
          f"top {TOP_N}, monthly")
    print(f"  Period: {START} → {today}")
    print()

    # ---- SPY benchmark (same for both) -------------------------------------
    yf = get_adapter(market)      # yfinance
    spy = yf.fetch(["SPY"], PRICE_START, today).close["SPY"].dropna()
    spy_vals = pd.Series([spy.asof(d) for d in dates], index=dates).dropna()
    spy_curve = (spy_vals / spy_vals.iloc[0]).iloc[1:]
    spy_cagr = cagr(spy_curve, 12)
    spy_dd = max_drawdown(spy_curve)
    print(f"  SPY: {spy_cagr:.2%} CAGR, {spy_dd:.1%} max DD\n")

    # ---- 1. yfinance (survivorship-biased) ---------------------------------
    print("  Fetching yfinance prices (survivors only)...")
    yf_prices = yf.fetch(symbols, PRICE_START, today)
    yf_bench = yf.fetch_benchmark(PRICE_START, today)
    print(f"  yfinance priced {len(yf_prices.symbols)} / {len(symbols)} names.")

    # ---- 2. Tiingo (survivorship-free) -------------------------------------
    print("  Fetching Tiingo prices (includes delisted -- from cache)...")
    tg = TiingoAdapter(market, api_key=key)
    tg_prices = tg.fetch(symbols, PRICE_START, today)
    tg_bench = tg.fetch_benchmark(PRICE_START, today)
    print(f"  Tiingo priced {len(tg_prices.symbols)} / {len(symbols)} names.")
    print()

    yf_res = _run(market, membership, yf_prices, yf_bench, signal, dates, spy_cagr)
    tg_res = _run(market, membership, tg_prices, tg_bench, signal, dates, spy_cagr)

    # ---- results side by side ----------------------------------------------
    print("=" * 88)
    print(f"  {'SOURCE':<28}{'CAGR':>9}{'vs SPY':>9}{'MAX DD':>9}"
          f"{'SHARPE':>8}{'SORTINO':>9}{'FINAL $100k':>14}")
    print("  " + "-" * 84)
    for label, r in [("yfinance (survivor-biased)", yf_res),
                     ("Tiingo (survivorship-free)", tg_res)]:
        print(f"  {label:<28}{r['cagr']:>9.2%}{r['excess']:>+9.2%}{r['dd']:>9.1%}"
              f"{r['sharpe']:>8.2f}{r['sortino']:>9.2f}"
              f"{'$' + format(r['final']*100000, ',.0f'):>14}")

    # ---- the survivorship gap ----------------------------------------------
    gap = yf_res["excess"] - tg_res["excess"]
    print()
    print("=" * 88)
    print("  THE SURVIVORSHIP ILLUSION")
    print("=" * 88)
    print(f"  yfinance excess vs SPY : {yf_res['excess']:+.2%}")
    print(f"  Tiingo   excess vs SPY : {tg_res['excess']:+.2%}")
    print(f"  Difference (illusion)  : {gap:+.2%} per year")
    print()
    if abs(gap) < 0.01:
        print("  The gap is small -- momentum's edge mostly SURVIVES honest data.")
    elif tg_res["excess"] > 0:
        print("  The edge SHRINKS but survives -- part illusion, part real.")
    else:
        print("  The edge largely VANISHES on honest data -- it was mostly survivorship.")
    print()
    print("  Coverage note: Tiingo priced more names than yfinance; the extra ones are")
    print("  the delisted companies whose failures yfinance silently skipped. That")
    print("  difference in the universe is exactly what makes this number honest.")
    print()
    print("  ⚠️  Backtests are simulations, not predictions.")
    print()


if __name__ == "__main__":
    main()
