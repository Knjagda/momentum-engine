"""
DOES CAPPING SECTOR WEIGHT HELP, OR JUST HANDCUFF THE STRATEGY?

    python -m scripts.experiment_sector_cap

Our honest baseline holds 70% Information Technology right now -- SNDK, WDC, MU,
INTC, AMD, LRCX, AMAT... a semiconductor fund in a momentum costume. That is
momentum's defining failure mode: it buys whatever has been winning, and what has
been winning is usually one theme. Nobody DECIDED to make a 70% chip bet; the
ranking did it silently.

A sector cap says "no sector above X%". But there is a real cost, and we must be
honest about it: momentum's edge IS concentration. Forcing the portfolio out of the
hottest sector means holding names momentum ranked LOWER. We might be trading return
for safety -- or we might be cutting exactly the crowded, crash-prone bets that blow
up. This experiment finds out which.

IMPORTANT NUANCE: our cap REDISTRIBUTES within the 20 names we already hold. If 14
are Tech, capping Tech at 35% pushes weight onto the 6 non-Tech names we happen to
own -- it does NOT reach further down the ranking for fresh names. So this tests
"hold the same names, weighted more evenly across sectors", not "replace Tech names
with others". That is a genuine strategy choice, not a bug, but worth naming.

Tested across BOTH eras. A cap that only helps in one is a period bet.

⚠️ We try four cap levels. The multiple-testing warning is mild here, but read the
SHAPE (does tighter = safer? at what return cost?), not the single best cell.
"""

from __future__ import annotations

import pandas as pd

from engine.backtest import get_overlay, rebalance_dates, run_backtest
from engine.data import get_adapter
from engine.markets.market import load_market
from engine.metrics import cagr, max_drawdown, sharpe_ratio, sortino_ratio
from engine.signals import get_signal
from engine.universe.universe import load_membership

UNIVERSE = "sp500_pit"
TOP_N = 20

ERAS = {
    "FULL 2005-today": "2005-01-01",
    "MODERN 2016-today": "2016-01-01",
}

CAPS = [None, 0.50, 0.40, 0.30]      # None = uncapped baseline


def main() -> None:
    market = load_market("us")
    membership = load_membership(market, UNIVERSE)
    adapter = get_adapter(market)
    signal = get_signal("momentum", lookback_months=12, skip_months=1)

    today = pd.Timestamp.today().strftime("%Y-%m-%d")

    print()
    print("=" * 94)
    print("  SECTOR CAP — does forcing diversification help or handcuff momentum?")
    print("=" * 94)
    print(f"  Strategy : momentum / top {TOP_N} / equal, always invested (no overlay)")
    print(f"  Universe : {UNIVERSE}")
    print("  The cap redistributes WITHIN the 20 held names -- it does not fetch new ones.")
    print()
    print("  Fetching prices...")

    prices = adapter.fetch(membership.symbols, "2003-06-01", today)
    benchmark = adapter.fetch_benchmark("2003-06-01", today)
    spy = adapter.fetch(["SPY"], "2003-06-01", today).close["SPY"].dropna()
    print(f"  {len(prices.symbols)} symbols.")
    print()

    for era_name, start in ERAS.items():
        dates = rebalance_dates(market, start, today, "monthly")
        spy_vals = pd.Series([spy.asof(d) for d in dates], index=dates).dropna()
        spy_curve = (spy_vals / spy_vals.iloc[0]).iloc[1:]
        spy_cagr = cagr(spy_curve, 12)

        print("=" * 94)
        print(f"  {era_name}      (SPY: {spy_cagr:.2%} CAGR)")
        print("=" * 94)
        print(f"  {'SECTOR CAP':<16}{'CAGR':>9}{'vs SPY':>9}{'MAX DD':>9}"
              f"{'SHARPE':>8}{'SORTINO':>9}{'CALMAR':>8}{'AVG TOP SECTOR':>16}")
        print("  " + "-" * 92)

        for cap in CAPS:
            result = run_backtest(
                market=market, membership=membership, prices=prices, signal=signal,
                rebalance_dates=dates, top_n=TOP_N, frequency="monthly",
                benchmark=benchmark, overlay=get_overlay("always_on"),
                max_sector_weight=cap,
            )

            eq = result.equity
            c = cagr(eq, 12)
            dd = max_drawdown(eq)
            calmar = c / abs(dd) if dd < 0 else 0.0

            # Average weight of the single largest sector each rebalance -- shows how
            # concentrated the book actually was.
            top_sector_weights = []
            for pf in result.portfolios:
                sw = pf.sector_weights()
                if len(sw):
                    top_sector_weights.append(float(sw.max()))
            avg_top = sum(top_sector_weights) / len(top_sector_weights) if top_sector_weights else 0.0

            label = "uncapped" if cap is None else f"{cap:.0%}"
            print(
                f"  {label:<16}{c:>9.2%}{c - spy_cagr:>+9.2%}{dd:>9.1%}"
                f"{sharpe_ratio(result.period_returns, 12):>8.2f}"
                f"{sortino_ratio(result.period_returns, 12):>9.2f}"
                f"{calmar:>8.2f}{avg_top:>15.0%}"
            )
        print()

    print("=" * 94)
    print("  READING IT")
    print("=" * 94)
    print("  HELPS:  tighter caps raise Sharpe/Calmar or cut drawdown at little return cost,")
    print("          in BOTH eras. Concentration was uncompensated risk -- cut it.")
    print()
    print("  HANDCUFFS: tighter caps steadily bleed return with no risk benefit. Momentum's")
    print("          edge IS its concentration; forcing it out just holds worse names.")
    print()
    print("  Watch AVG TOP SECTOR: it shows how concentrated the book really was. If even")
    print("  'uncapped' is ~40%, the 70% we saw today is a recent extreme, not the norm.")
    print()
    print("  ⚠️  Backtests are simulations, not predictions.")
    print()


if __name__ == "__main__":
    main()
