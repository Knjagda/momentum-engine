"""
Run a real backtest from a strategy config file.

    python -m scripts.backtest                                  # the US default
    python -m scripts.backtest india_nifty200_top20_momentum       # the same engine, India

The strategy is a YAML FILE, not code. That is the thing that eventually becomes
the no-code strategy builder in the pitch deck: this script is already the engine
behind it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

from engine.backtest import get_overlay, rebalance_dates, run_backtest
from engine.data import get_adapter
from engine.markets.market import load_market
from engine.metrics import compute_metrics
from engine.signals import get_signal
from engine.universe.universe import load_membership

REPO_ROOT = Path(__file__).resolve().parents[1]
STRATEGY_DIR = REPO_ROOT / "config" / "strategies"

DEFAULT_STRATEGY = "us_sp500_top20_momentum"


def resolve_date(value, default_today: bool = False) -> str:
    """
    Allow `today` (or an omitted end date) in strategy configs.

    A hard-coded end date silently goes stale: you keep backtesting to 2024 while
    the calendar says 2026, quietly discarding the most recent -- and most
    out-of-sample -- data you have.
    """
    if value is None and default_today:
        return pd.Timestamp.today().strftime("%Y-%m-%d")
    if isinstance(value, str) and value.strip().lower() in {"today", "now"}:
        return pd.Timestamp.today().strftime("%Y-%m-%d")
    return str(value)


def load_strategy(name: str) -> dict:
    path = STRATEGY_DIR / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in STRATEGY_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"No strategy '{name}'. Available: {available}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def pct(x: float | None) -> str:
    return "  —   " if x is None else f"{x:>7.2%}"


def num(x: float | None) -> str:
    return "  —   " if x is None else f"{x:>7.2f}"


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_STRATEGY
    cfg = load_strategy(name)

    market = load_market(cfg["market"])
    membership = load_membership(market, cfg["universe"])
    adapter = get_adapter(market)

    signal = get_signal(cfg["signal"]["name"], **cfg["signal"].get("params", {}))
    top_n = cfg["selection"]["top_n"]
    weighting = cfg["weighting"]["method"]
    max_pos = cfg["weighting"].get("max_position_weight")
    max_sector = cfg["weighting"].get("max_sector_weight")   # was silently ignored before
    frequency = cfg["rebalance"]["frequency"]

    start = resolve_date(cfg["backtest"]["start"])
    end = resolve_date(cfg["backtest"].get("end"), default_today=True)

    trend_cfg = cfg.get("overlay", {}).get("trend_filter", {})
    if trend_cfg.get("enabled"):
        overlay = get_overlay("trend_filter", ma_days=trend_cfg.get("benchmark_ma_days", 200))
    else:
        overlay = get_overlay("always_on")

    print()
    print("=" * 78)
    print(f"  {cfg['name']}")
    print("=" * 78)
    print(f"  Market      : {market.name}   ({market.currency}, {market.calendar})")
    print(f"  Universe    : {membership.universe_key}  ({len(membership)} members)")
    print(f"  Signal      : {signal}")
    print(f"  Portfolio   : top {top_n}, {weighting} weight")
    print(f"  Rebalance   : {frequency}")
    print(f"  Overlay     : {overlay.name}")
    print(f"  Period      : {start} → {end}")
    print()

    # Fetch with a runway so the first rebalance has full lookback history.
    fetch_start = (pd.Timestamp(start) - pd.DateOffset(months=18)).strftime("%Y-%m-%d")

    print("  Fetching prices... (first run is slow; cached afterwards)")
    prices = adapter.fetch(membership.symbols, fetch_start, end)
    benchmark = adapter.fetch_benchmark(fetch_start, end)
    print(f"  Got {len(prices.symbols)} symbols, {len(prices.close)} days")
    print()

    dates = rebalance_dates(market, start, end, frequency)
    print(f"  Running {len(dates) - 1} rebalances on the {market.calendar} calendar...")

    result = run_backtest(
        market=market,
        membership=membership,
        prices=prices,
        signal=signal,
        rebalance_dates=dates,
        top_n=top_n,
        frequency=frequency,
        weighting=weighting,
        max_position_weight=max_pos,
        max_sector_weight=max_sector,
        overlay=overlay,
        benchmark=benchmark,
    )
    m = compute_metrics(result)

    sym = market.currency_symbol
    growth = 100_000 * result.equity.iloc[-1]
    bench_growth = (
        100_000 * result.benchmark_equity.iloc[-1]
        if result.benchmark_equity is not None else None
    )

    print()
    print("-" * 78)
    print(f"  RESULTS   ({m.years:.1f} years, {m.n_periods} periods)")
    print("-" * 78)
    print(f"  {'':<24}{'STRATEGY':>12}{'BENCHMARK':>14}")
    print(f"  {'-' * 52}")
    print(f"  {'CAGR':<24}{pct(m.cagr)}{pct(m.benchmark_cagr):>14}")
    print(f"  {'Max drawdown':<24}{pct(m.max_drawdown)}{pct(m.benchmark_max_drawdown):>14}")
    print(f"  {'Sharpe ratio':<24}{num(m.sharpe)}{num(m.benchmark_sharpe):>14}")
    print(f"  {'Volatility':<24}{pct(m.volatility)}")
    print(f"  {'Sortino ratio':<24}{num(m.sortino)}")
    print(f"  {'Calmar ratio':<24}{num(m.calmar)}")
    print(f"  {'Win rate':<24}{pct(m.win_rate)}")
    print()
    print(f"  {'Excess CAGR vs bench':<24}{pct(m.excess_cagr)}")
    print(f"  {'Alpha (annualized)':<24}{pct(m.alpha)}")
    print(f"  {'Beta':<24}{num(m.beta)}")
    print(f"  {'Information ratio':<24}{num(m.information_ratio)}")
    print()
    print("  THE COST OF TRADING (what most backtests hide)")
    print(f"  {'Gross CAGR':<24}{pct(m.gross_cagr)}")
    print(f"  {'Net CAGR':<24}{pct(m.cagr)}")
    print(f"  {'Cost drag':<24}{pct(m.cost_drag)}   ← trading friction, per year")
    print(f"  {'Avg turnover':<24}{pct(m.avg_turnover)}   per rebalance")
    print(f"  {'Periods in cash':<24}{pct(m.periods_in_cash)}")
    print()
    print(f"  {sym}100,000 → {sym}{growth:,.0f}", end="")
    if bench_growth:
        print(f"      (benchmark: {sym}{bench_growth:,.0f})")
    else:
        print()
    print()

    latest = result.portfolios[-1]
    print("-" * 78)
    print(f"  CURRENT HOLDINGS  (as of {latest.as_of.date()})")
    print("-" * 78)
    if latest.n_positions == 0:
        print(f"  100% CASH — overlay says stand aside ({latest.metadata.get('reason')})")
    else:
        frame = latest.to_frame()
        for row in frame.itertuples():
            print(f"  {row.rank:<4}{row.symbol:<14}{row.weight:>7.2%}   {row.sector[:28]}")
        print()
        print("  Sector exposure:")
        for sector, w in latest.sector_weights().items():
            flag = "  ⚠️ concentrated" if w > 0.35 else ""
            print(f"    {sector[:30]:<32}{w:>7.1%}{flag}")
    print()

    print("=" * 78)
    for d in result.disclaimers:
        print(f"  {d}")
    print("=" * 78)
    print()


if __name__ == "__main__":
    main()
