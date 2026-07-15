"""
DOES VOLATILITY TARGETING FIX OUR -28% DRAWDOWN?

    python -m scripts.experiment_vol_target

The pre-registered hypothesis (parked days ago): our honest momentum baseline
(+4.37% vs SPY, but -28% drawdown) crashes in the same high-volatility regimes
Barroso & Santa-Clara identified. Scaling exposure DOWN when the strategy's own
recent returns turn violent should cut the worst drawdowns without surrendering
most of the return.

This is the HONEST version of the volatility idea. Volar -- scoring each stock by
its own choppiness -- was our weakest signal. This is different: it sizes the whole
book by the STRATEGY's realised risk, using only past returns.

We test the winning config from the signal grid (momentum / top 20 / equal), always
invested, on the point-in-time universe, across BOTH eras. We compare:

    raw                     no risk scaling (the baseline)
    vol-target @ 10/12/15%  three target volatilities

WHAT WOULD MAKE THIS REAL vs A CURVE-FIT:
  - It should help in BOTH eras, not one.
  - The MECHANISM should be visible: lower drawdown, exposure falling in crises.
  - We are testing ONE idea with a few sensible parameters, not sweeping hundreds.
    The multiple-testing risk is low here -- but target_vol is still a knob, so we
    look for robustness across the three values, not the single best one.

Barroso & Santa-Clara reported Sharpe 0.53 -> 0.97. We will not get that (they had
better data and no trend filter interaction); we are looking for the SHAPE of the
result -- drawdown down, Sharpe up, return mostly intact.
"""

from __future__ import annotations

import pandas as pd

from engine.backtest import (
    VolatilityTarget,
    get_overlay,
    rebalance_dates,
    run_backtest,
)
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

TARGETS = [None, 0.10, 0.12, 0.15]      # None = raw baseline


def main() -> None:
    market = load_market("us")
    membership = load_membership(market, UNIVERSE)
    adapter = get_adapter(market)
    signal = get_signal("momentum", lookback_months=12, skip_months=1)

    today = pd.Timestamp.today().strftime("%Y-%m-%d")

    print()
    print("=" * 92)
    print("  VOLATILITY TARGETING — does it fix the -28% drawdown?")
    print("=" * 92)
    print(f"  Strategy : momentum / top {TOP_N} / equal, always invested (no trend filter)")
    print(f"  Universe : {UNIVERSE}")
    print("  Isolating vol-targeting: trend filter OFF so we see THIS effect alone.")
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
        spy_dd = max_drawdown(spy_curve)

        print("=" * 92)
        print(f"  {era_name}      (SPY: {spy_cagr:.2%} CAGR, {spy_dd:.1%} max DD)")
        print("=" * 92)
        print(f"  {'CONFIG':<22}{'CAGR':>9}{'vs SPY':>9}{'MAX DD':>9}"
              f"{'SHARPE':>8}{'SORTINO':>9}{'CALMAR':>8}{'AVG EXP':>9}")
        print("  " + "-" * 88)

        baseline_dd = None

        for target in TARGETS:
            vt = (
                VolatilityTarget(target_vol=target, lookback_periods=6, periods_per_year=12)
                if target is not None else None
            )

            result = run_backtest(
                market=market, membership=membership, prices=prices, signal=signal,
                rebalance_dates=dates, top_n=TOP_N, frequency="monthly",
                benchmark=benchmark, overlay=get_overlay("always_on"),
                vol_target=vt,
            )

            eq = result.equity
            c = cagr(eq, 12)
            dd = max_drawdown(eq)
            sh = sharpe_ratio(result.period_returns, 12)
            so = sortino_ratio(result.period_returns, 12)
            calmar = c / abs(dd) if dd < 0 else 0.0

            # average exposure actually deployed
            if result.vol_decisions:
                avg_exp = sum(d.exposure for d in result.vol_decisions) / len(result.vol_decisions)
            else:
                avg_exp = 1.0

            label = "raw (baseline)" if target is None else f"vol-target @ {target:.0%}"
            if target is None:
                baseline_dd = dd

            print(
                f"  {label:<22}{c:>9.2%}{c - spy_cagr:>+9.2%}{dd:>9.1%}"
                f"{sh:>8.2f}{so:>9.2f}{calmar:>8.2f}{avg_exp:>9.0%}"
            )

        print()

    print("=" * 92)
    print("  HOW TO READ THIS")
    print("=" * 92)
    print("  SUCCESS looks like: max drawdown shrinks meaningfully (toward -15/-20%),")
    print("  Sharpe and Calmar RISE, and CAGR gives up only a little -- in BOTH eras.")
    print()
    print("  FAILURE looks like: drawdown barely moves, or CAGR collapses for no")
    print("  drawdown benefit, or it helps in one era and hurts in the other.")
    print()
    print("  AVG EXP is how much of the book was deployed on average. If it is ~95%,")
    print("  the targeting almost never engaged and the test is inconclusive -- the")
    print("  strategy was rarely volatile enough to trigger it.")
    print()
    print("  ⚠️  Vol targeting delevers in danger; it never levers up. It reduces the")
    print("      WORST outcomes, it does not raise the average one. Backtests remain")
    print("      simulations, not predictions.")
    print()


if __name__ == "__main__":
    main()
