"""
DOES ADDING A VALUE GATE IMPROVE MOMENTUM?

    python -m scripts.experiment_value your@email.com

Every internal lever we tried -- turnover, vol targeting, trend filter, sector cap --
either did nothing or bought crisis-insurance at a return cost. None found a NEW edge,
because none changed WHAT we hold, only HOW.

This changes what we hold. Momentum and value are the two most-documented factors in
finance, and they tend to win in DIFFERENT regimes: momentum rides trends, value
catches the reversals that hurt momentum most. Combining them MIGHT give a steadier
edge -- "cheap AND rising" -- or the value gate might just throw away momentum's best
names (the expensive rockets) for nothing. We find out.

Fundamental data is EDGAR, so this can only run 2010+ (the XBRL wall). We compare:

    momentum only                 the honest baseline
    momentum + value gate         positive earnings AND price-to-book below median

THREE THINGS TO WATCH:
  1. Does the value gate raise risk-adjusted return, or just cut it?
  2. COVERAGE -- how many names did the screen actually have data for? If EDGAR
     coverage is thin, the screened universe is small and the result is fragile.
  3. THE SIVB TRAP -- in 2023 the screen will have bought distressed banks at low
     price-to-book that later went to zero. We cannot even price that collapse, so
     the value result is OPTIMISTIC in a way we must state, not hide.

⚠️ Only 2010+, EDGAR-only, US large/mid cap. This is a FIRST LOOK, not a verdict.
"""

from __future__ import annotations

import sys

import pandas as pd

from engine.backtest import get_overlay, rebalance_dates, run_backtest
from engine.data import get_adapter, get_fundamental_adapter
from engine.markets.market import load_market
from engine.metrics import cagr, max_drawdown, sharpe_ratio, sortino_ratio
from engine.signals import get_signal
from engine.signals.screen import get_screen
from engine.universe.universe import load_membership

UNIVERSE = "sp900_pit"       # large + mid cap, where EDGAR coverage is best
TOP_N = 20
START = "2010-06-01"         # after the XBRL wall + a runway for fundamentals


def main() -> None:
    if len(sys.argv) < 2 or "@" not in sys.argv[1]:
        print("\n  Pass your email (SEC requires it):")
        print("      python -m scripts.experiment_value you@yourdomain.com\n")
        return

    email = sys.argv[1]
    market = load_market("us")
    membership = load_membership(market, UNIVERSE)
    adapter = get_adapter(market)
    edgar = get_fundamental_adapter("edgar", market=market,
                                    user_agent=f"momentum-engine research {email}")
    signal = get_signal("momentum", lookback_months=12, skip_months=1)

    today = pd.Timestamp.today().strftime("%Y-%m-%d")

    print()
    print("=" * 92)
    print("  MOMENTUM + VALUE — does a value gate improve momentum?")
    print("=" * 92)
    print(f"  Universe : {UNIVERSE}  ({len(membership.symbols)} symbols, large/mid cap)")
    print(f"  Period   : {START} → {today}   (EDGAR = 2010+ only)")
    print(f"  Screen   : positive earnings AND price-to-book ≤ universe median")
    print()
    print("  Fetching prices + fundamentals (fundamentals cached from earlier run)...")

    prices = adapter.fetch(membership.symbols, "2008-06-01", today)
    benchmark = adapter.fetch_benchmark("2008-06-01", today)
    spy = adapter.fetch(["SPY"], "2008-06-01", today).close["SPY"].dropna()
    fundamentals = edgar.fetch(membership.symbols, verbose=False)
    print(f"  {len(prices.symbols)} priced, {len(fundamentals.symbols)} with fundamentals.")
    print()

    dates = rebalance_dates(market, START, today, "monthly")
    spy_vals = pd.Series([spy.asof(d) for d in dates], index=dates).dropna()
    spy_curve = (spy_vals / spy_vals.iloc[0]).iloc[1:]
    spy_cagr = cagr(spy_curve, 12)
    spy_dd = max_drawdown(spy_curve)

    print("=" * 92)
    print(f"  RESULTS   (SPY: {spy_cagr:.2%} CAGR, {spy_dd:.1%} max DD)")
    print("=" * 92)
    print(f"  {'CONFIG':<26}{'CAGR':>9}{'vs SPY':>9}{'MAX DD':>9}"
          f"{'SHARPE':>8}{'SORTINO':>9}{'AVG HELD':>9}")
    print("  " + "-" * 88)

    configs = [
        ("momentum only", None),
        ("momentum + value gate", get_screen("value")),
    ]

    results = {}
    for label, screen in configs:
        result = run_backtest(
            market=market, membership=membership, prices=prices, signal=signal,
            rebalance_dates=dates, top_n=TOP_N, frequency="monthly",
            benchmark=benchmark, overlay=get_overlay("always_on"),
            screen=screen, fundamentals=fundamentals if screen else None,
        )
        results[label] = result

        eq = result.equity
        c = cagr(eq, 12)
        avg_held = sum(p.n_positions for p in result.portfolios) / len(result.portfolios)

        print(
            f"  {label:<26}{c:>9.2%}{c - spy_cagr:>+9.2%}{max_drawdown(eq):>9.1%}"
            f"{sharpe_ratio(result.period_returns, 12):>8.2f}"
            f"{sortino_ratio(result.period_returns, 12):>9.2f}{avg_held:>9.1f}"
        )

    print()

    # ---- how often did the screen leave us short of names or in cash? ------
    screened = results["momentum + value gate"]
    cash_periods = sum(1 for p in screened.portfolios if p.n_positions == 0)
    short_periods = sum(1 for p in screened.portfolios if 0 < p.n_positions < TOP_N)

    print("=" * 92)
    print("  SCREEN DIAGNOSTICS")
    print("=" * 92)
    print(f"  Rebalances fully in cash (nobody passed): {cash_periods}/{len(screened.portfolios)}")
    print(f"  Rebalances short of {TOP_N} names:            {short_periods}/{len(screened.portfolios)}")
    print("  If these are high, EDGAR coverage was too thin to build a full book and")
    print("  the value result rests on a handful of names -- treat it as fragile.")
    print()

    print("=" * 92)
    print("  ⚠️  THE VALUE TRAP — READ BEFORE BELIEVING THE VALUE COLUMN")
    print("=" * 92)
    print("  A value gate buys cheap stocks. In early 2023 the cheapest banks by")
    print("  price-to-book were SVB and First Republic -- right before they hit zero.")
    print("  Our price data cannot represent that collapse (yfinance drops delisted")
    print("  names), so this backtest NEVER TAKES THAT LOSS. The value line above is")
    print("  therefore optimistic by an unknown amount. Only survivorship-free price")
    print("  data (Sharadar/Norgate) can close that gap. Until then: a first look.")
    print()
    print("  ⚠️  Backtests are simulations, not predictions.")
    print()


if __name__ == "__main__":
    main()
