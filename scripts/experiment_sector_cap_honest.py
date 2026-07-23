"""
SECTOR CAP -- what does limiting concentration actually cost, on honest data?

    python -m scripts.experiment_sector_cap_honest YOUR_TIINGO_KEY [START_DATE]

THE OPEN QUESTION. Today's live signal is 85% Information Technology -- seventeen of
twenty holdings are semiconductor and hardware companies. That is the strategy working
as designed: momentum buys what is winning, and right now that is one industry. The
report flags it, but nothing stops it.

We have never decided whether to cap it. An earlier sector-cap experiment existed but
its conclusion was never recorded, and the config sets no `max_sector_weight`. So the
product currently ships an unmanaged concentration bet by default rather than by
decision. This run closes that.

WHAT IT MEASURES. The same survivorship-free, trend-filtered momentum strategy at
several sector ceilings. For each, the return AND the risk AND how concentrated the
portfolio actually ended up -- because a cap that never binds is theatre, and a cap
that binds constantly is a different strategy wearing momentum's name.

HOW TO READ IT. A cap is worth adopting if it meaningfully reduces the worst drawdown
or improves risk-adjusted return for a return cost you would accept. It is NOT worth
adopting merely because concentration feels uncomfortable -- we measured the trend
filter the same way and it earned its place; the sector cap has to earn its own.
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
DEFAULT_START = "2005-01-01"
LOOKBACK_RUNWAY_YEARS = 2
DEAD_FILE = Path("data/dead_names.txt")

# None = no cap (today's behaviour). Then progressively tighter ceilings.
CAPS = [None, 0.50, 0.40, 0.35, 0.30, 0.25]


def _merge(a: PriceData, b: PriceData, market) -> PriceData:
    """Tiingo (b) wins on overlap -- those are the dead names."""
    overlap = a.close.columns.intersection(b.close.columns)
    close = a.close.drop(columns=overlap).join(b.close, how="outer")
    volume = a.volume.drop(columns=overlap, errors="ignore").join(b.volume, how="outer")
    return PriceData(market=market, close=close.sort_index(),
                     volume=volume.sort_index())


def main() -> None:
    if len(sys.argv) < 2:
        print("\n  python -m scripts.experiment_sector_cap_honest KEY [START_DATE]\n")
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
    print("=" * 84)
    print("  SECTOR CAP ON SURVIVORSHIP-FREE DATA -- is concentration worth limiting?")
    print("=" * 84)
    print(f"  {UNIVERSE}, momentum 12-1, top {TOP_N}, monthly, trend-filtered")
    print(f"  Period: {start} -> {today}\n")

    # ---- build the honest price set once ----------------------------------
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
    print(f"  Survivorship-free price set built ({len(cached_dead)} dead names).\n")

    spy = yf_bench.reindex(dates, method="ffill")
    spy_curve = (spy / spy.iloc[0]).dropna()
    spy_cagr = cagr(spy_curve.iloc[1:], 12)
    print(f"  SPY: {spy_cagr:.2%} CAGR, {max_drawdown(spy_curve.iloc[1:]):.1%} max DD\n")
    print("  Running caps...\n")

    rows = []
    for cap in CAPS:
        result = run_backtest(
            market=market, membership=membership, prices=prices, signal=signal,
            rebalance_dates=dates, top_n=TOP_N, frequency="monthly",
            benchmark=yf_bench, overlay=get_overlay("trend_filter"),
            weighting="equal", max_sector_weight=cap,
        )
        eq = result.equity
        # How concentrated did it ACTUALLY get? A cap that never binds changes nothing.
        worst_sector = None
        try:
            concentrations = [
                float(p.sector_weights().max())
                for p in result.portfolios if len(p.symbols)
            ]
            worst_sector = max(concentrations) if concentrations else None
            typical_sector = (pd.Series(concentrations).median()
                              if concentrations else None)
        except Exception:
            typical_sector = None

        rows.append({
            "cap": cap,
            "cagr": cagr(eq, 12),
            "excess": cagr(eq, 12) - spy_cagr,
            "dd": max_drawdown(eq),
            "sharpe": sharpe_ratio(result.period_returns, 12),
            "sortino": sortino_ratio(result.period_returns, 12),
            "worst_sector": worst_sector,
            "typical_sector": typical_sector,
        })
        label = "none" if cap is None else f"{cap:.0%}"
        print(f"  cap={label:<6} excess {rows[-1]['excess']:+.2%}   "
              f"DD {rows[-1]['dd']:.1%}   Sharpe {rows[-1]['sharpe']:.2f}")

    print()
    print("=" * 84)
    print(f"  {'SECTOR CAP':<12}{'CAGR':>9}{'vs SPY':>9}{'MAX DD':>9}{'SHARPE':>8}"
          f"{'SORTINO':>9}{'WORST SECTOR':>14}{'TYPICAL':>9}")
    print("  " + "-" * 80)
    for r in rows:
        label = "none" if r["cap"] is None else f"{r['cap']:.0%}"
        ws = f"{r['worst_sector']:.0%}" if r["worst_sector"] is not None else "n/a"
        ts = f"{r['typical_sector']:.0%}" if r["typical_sector"] is not None else "n/a"
        print(f"  {label:<12}{r['cagr']:>9.2%}{r['excess']:>+9.2%}{r['dd']:>9.1%}"
              f"{r['sharpe']:>8.2f}{r['sortino']:>9.2f}{ws:>14}{ts:>9}")

    base = rows[0]
    print()
    print("=" * 84)
    print("  THE TRADE, STATED PLAINLY (vs no cap)")
    print("=" * 84)
    for r in rows[1:]:
        d_cagr = r["cagr"] - base["cagr"]
        d_dd = r["dd"] - base["dd"]            # less negative = shallower
        d_sharpe = r["sharpe"] - base["sharpe"]
        print(f"  cap {r['cap']:.0%}:  CAGR {d_cagr:+.2%}   "
              f"drawdown {abs(d_dd):.1f}pt {'shallower' if d_dd > 0 else 'deeper'}   "
              f"Sharpe {d_sharpe:+.2f}")

    print()
    print("  DECIDE ON THE EVIDENCE, NOT THE DISCOMFORT. Adopt a cap only if it buys")
    print("  drawdown or risk-adjusted return worth its cost in CAGR -- the same test")
    print("  the trend filter passed. If every cap costs return without improving")
    print("  Sharpe or drawdown, the honest answer is to stay uncapped and DISCLOSE")
    print("  the concentration prominently, which the signal report already does.")
    print()
    print("  Backtests are simulations, not predictions.")
    print()


if __name__ == "__main__":
    main()
