"""
Do risk overlays earn their keep once the drawdown is measured HONESTLY?

    python -m scripts.experiment_overlays_honest YOUR_TIINGO_KEY [START_DATE]

WHY RE-RUN THIS. We evaluated overlays in an earlier session and concluded they were
"insurance with a premium" -- they cut drawdown but cost return, so none was adopted.
That judgement was made against a max drawdown of about -61%, measured on
SURVIVORS-ONLY prices.

We now know that figure was understated. On survivorship-free data the same strategy
draws down -64.9%, and the gap widens precisely in crises: including failures worsened
drawdown by 0.9pt in the calm decade but 3.9pt across 2008-09. In other words, we
priced the insurance against an under-measured risk.

So the honest question is: with the real drawdown known, is the trend filter's premium
worth paying after all? A -65% drawdown ends most strategies in practice, because
almost nobody holds through it. An overlay that costs some CAGR but keeps the trough
survivable may be worth far more than the earlier comparison suggested.

This runs every registered overlay on the SURVIVORSHIP-FREE merged price set and
reports return AND risk side by side, so the trade is visible rather than assumed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from engine.backtest import get_overlay, rebalance_dates, run_backtest
from engine.backtest.overlay import registered_overlays
from engine.data import get_adapter
from engine.data.base import PriceData
from engine.data.tiingo_adapter import TiingoAdapter
from engine.markets.market import load_market
from engine.metrics import cagr, max_drawdown, sharpe_ratio, sortino_ratio
from engine.signals import get_signal
from engine.universe.universe import load_membership

UNIVERSE = "sp900_pit"
TOP_N = 20
DEFAULT_START = "2005-01-01"        # include the crisis by default -- that is the point
LOOKBACK_RUNWAY_YEARS = 2
DEAD_FILE = Path("data/dead_names.txt")


def _merge(a: PriceData, b: PriceData, market) -> PriceData:
    """Tiingo (b) wins on overlap -- those are the dead names."""
    overlap = a.close.columns.intersection(b.close.columns)
    close = a.close.drop(columns=overlap).join(b.close, how="outer")
    volume = a.volume.drop(columns=overlap, errors="ignore").join(b.volume, how="outer")
    return PriceData(market=market, close=close.sort_index(),
                     volume=volume.sort_index())


def main() -> None:
    if len(sys.argv) < 2:
        print("\n  python -m scripts.experiment_overlays_honest KEY [START_DATE]\n")
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
    print("  RISK OVERLAYS ON SURVIVORSHIP-FREE DATA -- is the premium worth it now?")
    print("=" * 92)
    print(f"  {UNIVERSE}, momentum 12-1, top {TOP_N}, monthly")
    print(f"  Period: {start} -> {today}\n")

    yf = get_adapter(market)
    spy = yf.fetch(["SPY"], price_start, today).close["SPY"].dropna()
    spy_vals = pd.Series([spy.asof(d) for d in dates], index=dates).dropna()
    spy_curve = (spy_vals / spy_vals.iloc[0]).iloc[1:]
    spy_cagr = cagr(spy_curve, 12)
    print(f"  SPY: {spy_cagr:.2%} CAGR, {max_drawdown(spy_curve):.1%} max DD\n")

    print("  Building survivorship-free price set...")
    yf_prices = yf.fetch(symbols, price_start, today)
    yf_bench = yf.fetch_benchmark(price_start, today)

    dead = sorted({ln.strip() for ln in DEAD_FILE.read_text().splitlines()
                   if ln.strip()}) if DEAD_FILE.exists() else []
    tg = TiingoAdapter(market, api_key=key)
    cached_dead = [s for s in dead
                   if tg._cache_path(market.resolve_ticker(s)).exists()]
    tg_prices = tg.fetch(cached_dead, price_start, today)
    prices = _merge(yf_prices, tg_prices, market)
    print(f"  {len(cached_dead)} dead names included.\n")

    rows = []
    for name in registered_overlays():
        result = run_backtest(
            market=market, membership=membership, prices=prices, signal=signal,
            rebalance_dates=dates, top_n=TOP_N, frequency="monthly",
            benchmark=yf_bench, overlay=get_overlay(name),
        )
        eq = result.equity
        rows.append({
            "overlay": name,
            "cagr": cagr(eq, 12),
            "excess": cagr(eq, 12) - spy_cagr,
            "dd": max_drawdown(eq),
            "sharpe": sharpe_ratio(result.period_returns, 12),
            "sortino": sortino_ratio(result.period_returns, 12),
            "final": eq.iloc[-1],
        })

    print("=" * 92)
    print(f"  {'OVERLAY':<22}{'CAGR':>9}{'vs SPY':>9}{'MAX DD':>9}"
          f"{'SHARPE':>8}{'SORTINO':>9}{'FINAL $100k':>15}")
    print("  " + "-" * 88)
    for r in rows:
        print(f"  {r['overlay']:<22}{r['cagr']:>9.2%}{r['excess']:>+9.2%}"
              f"{r['dd']:>9.1%}{r['sharpe']:>8.2f}{r['sortino']:>9.2f}"
              f"{'$' + format(r['final'] * 100000, ',.0f'):>15}")

    base = next((r for r in rows if r["overlay"] == "always_on"), None)
    if base:
        print()
        print("=" * 92)
        print("  THE TRADE, STATED PLAINLY (vs always_on)")
        print("=" * 92)
        for r in rows:
            if r["overlay"] == "always_on":
                continue
            d_cagr = r["cagr"] - base["cagr"]
            d_dd = r["dd"] - base["dd"]          # less negative = shallower
            print(f"  {r['overlay']:<22} CAGR {d_cagr:+.2%}   "
                  f"drawdown {abs(d_dd):.1f}pt "
                  f"{'shallower' if d_dd > 0 else 'deeper'}")
        print()
        print("  Earlier we judged overlays against an UNDERSTATED -61% drawdown and")
        print("  declined them. Judge the same trade against the honest figure above:")
        print("  a drawdown most investors would not survive is a different kind of")
        print("  risk from one they merely dislike.")

    print()
    print("  Backtests are simulations, not predictions.")
    print()


if __name__ == "__main__":
    main()
