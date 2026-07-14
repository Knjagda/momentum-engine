"""
THE TURNOVER EXPERIMENT.

    python -m scripts.experiment_turnover

Our honest backtest (point-in-time universe) beat the S&P 500 by roughly nothing,
while SPMO -- a real, live, concentrated S&P 500 momentum fund -- beat it by ~5%/yr.
The most obvious suspect is TURNOVER: we churn ~35% of the portfolio every month.
SPMO's index reconstitutes semi-annually.

Every one of those trades pays a spread. Worse, most of them are pointless: selling
a stock because it slipped from rank 19 to rank 21 is not a decision, it is a twitch.

This grid tests the two cheapest fixes:

    NO-TRADE BUFFER    hold a name until it falls below `exit_rank`, not rank 21
    LOWER FREQUENCY    rebalance quarterly instead of monthly

Run against the POINT-IN-TIME universe only. Running experiments on the biased
universe would be tuning a strategy to exploit a bug in our own data -- which is
how people talk themselves into strategies that lose money.

⚠️ A WARNING ABOUT THIS SCRIPT. We are about to try ~8 configurations and pick the
best. That is the multiple-testing problem (Harvey, Liu & Zhu 2016): try enough
variants and one WILL look good by luck alone. Treat the winner as a hypothesis to
be tested out-of-sample, NOT as a discovery. We are looking for a broad pattern --
"less trading helps" -- not for the single magic number.
"""

from __future__ import annotations

import pandas as pd

from engine.backtest import get_overlay, rebalance_dates, run_backtest
from engine.data import get_adapter
from engine.markets.market import load_market
from engine.metrics import cagr, max_drawdown, sharpe_ratio
from engine.signals import get_signal
from engine.universe.universe import load_membership

UNIVERSE = "sp500_pit"          # the honest one. Never tune on the biased list.
TOP_N = 20
START = "2005-01-01"

# (label, frequency, exit_rank)
GRID = [
    ("Monthly, no buffer  (current)", "monthly",   None),
    ("Monthly, exit@25",              "monthly",   25),
    ("Monthly, exit@30",              "monthly",   30),
    ("Monthly, exit@40",              "monthly",   40),
    ("Quarterly, no buffer",          "quarterly", None),
    ("Quarterly, exit@30",            "quarterly", 30),
    ("Quarterly, exit@40",            "quarterly", 40),
]


def main() -> None:
    market = load_market("us")
    membership = load_membership(market, UNIVERSE)
    adapter = get_adapter(market)
    signal = get_signal("volar", lookback_months=12, skip_months=1, vol_window_days=126)

    today = pd.Timestamp.today().strftime("%Y-%m-%d")

    print()
    print("=" * 92)
    print("  TURNOVER EXPERIMENT — does trading less make us more?")
    print("=" * 92)
    print(f"  Universe : {UNIVERSE}  "
          f"({'POINT-IN-TIME ✅' if membership.is_point_in_time else '⚠️ BIASED'})")
    print(f"  Strategy : top {TOP_N} by volar, equal weight, always invested")
    print(f"  Period   : {START} → {today}")
    print()
    print("  Fetching prices (this is the slow part)...")

    prices = adapter.fetch(membership.symbols, "2003-06-01", today)
    benchmark = adapter.fetch_benchmark("2003-06-01", today)
    spy = adapter.fetch(["SPY"], "2003-06-01", today)
    print(f"  {len(prices.symbols)} symbols loaded.")
    print()

    # SPY baseline over the same window.
    spy_series = spy.close["SPY"].dropna()

    print("=" * 92)
    print(f"  {'CONFIG':<32}{'TURNOVER':>10}{'GROSS':>9}{'NET':>9}"
          f"{'DRAG':>8}{'MAX DD':>9}{'SHARPE':>8}{'vs SPY':>9}")
    print(f"  {'(per rebalance)':<32}{'':>10}{'CAGR':>9}{'CAGR':>9}{'/yr':>8}")
    print("  " + "-" * 88)

    results = []

    for label, freq, exit_rank in GRID:
        dates = rebalance_dates(market, START, today, freq)

        result = run_backtest(
            market=market,
            membership=membership,
            prices=prices,
            signal=signal,
            rebalance_dates=dates,
            top_n=TOP_N,
            frequency=freq,
            benchmark=benchmark,
            overlay=get_overlay("always_on"),      # isolate turnover; no overlay noise
            exit_rank=exit_rank,
        )

        ppy = result.periods_per_year
        equity = result.equity
        rets = result.period_returns

        # SPY over exactly these dates, so the comparison is fair.
        spy_vals = pd.Series([spy_series.asof(d) for d in dates], index=dates).dropna()
        spy_curve = (spy_vals / spy_vals.iloc[0]).iloc[1:]
        spy_cagr = cagr(spy_curve, ppy)

        net = cagr(equity, ppy)
        gross = cagr(result.gross_equity, ppy)

        row = {
            "label": label,
            "turnover": float(result.turnover.mean()),
            "gross": gross,
            "net": net,
            "drag": gross - net,
            "maxdd": max_drawdown(equity),
            "sharpe": sharpe_ratio(rets, ppy),
            "excess": net - spy_cagr,
        }
        results.append(row)

        print(
            f"  {label:<32}{row['turnover']:>9.1%}{row['gross']:>9.2%}{row['net']:>9.2%}"
            f"{row['drag']:>8.2%}{row['maxdd']:>9.1%}{row['sharpe']:>8.2f}"
            f"{row['excess']:>+9.2%}"
        )

    print("  " + "-" * 88)
    print(f"  {'SPY (the market)':<32}{'—':>9}{'—':>9}{spy_cagr:>9.2%}"
          f"{'—':>8}{max_drawdown(spy_curve):>9.1%}")
    print(f"  {'SPMO (real momentum fund)':<32}{'semi-annual':>9}"
          f"{'':>9}{'~20.5%':>9}{'':>8}{'-20.4%':>9}{'1.17':>8}{'+4.9%':>9}")
    print("=" * 92)
    print()

    # --- read the pattern, not the winner -----------------------------------
    baseline = results[0]
    best = max(results, key=lambda r: r["net"])

    print("  WHAT THIS SHOWS")
    print("  " + "-" * 88)
    print(f"  Baseline turnover : {baseline['turnover']:.1%} per rebalance "
          f"→ {baseline['drag']:.2%}/yr in costs")
    print(f"  Lowest turnover   : {min(r['turnover'] for r in results):.1%}")
    print()

    if best["label"] == baseline["label"]:
        print("  ⚠️  Trading less did NOT help. Turnover is not the leak.")
        print("      The gap versus SPMO must come from somewhere else -- signal design,")
        print("      weighting, or concentration. Worth looking there next.")
    else:
        gain = best["net"] - baseline["net"]
        cost_saved = baseline["drag"] - best["drag"]
        alpha_change = gain - cost_saved
        print(f"  Best config: {best['label']}  ({gain:+.2%}/yr vs baseline)")
        print(f"    of which cost saved : {cost_saved:+.2%}/yr")
        print(f"    of which signal      : {alpha_change:+.2%}/yr")
        print()
        if cost_saved > abs(alpha_change):
            print("  ✅ The gain is mostly SAVED COSTS -- exactly the mechanism we predicted.")
            print("     That is a robust reason to believe it, not a curve-fit.")
        else:
            print("  ⚠️  The gain is mostly from a CHANGED SIGNAL, not saved costs.")
            print("      Be suspicious: that is the kind of result that does not repeat.")

    print()
    print("  ⚠️  We just tried 7 configurations. At least one will look good by luck")
    print("      (Harvey, Liu & Zhu 2016). Treat the winner as a HYPOTHESIS, not a finding.")
    print("      It must survive an out-of-sample test before it earns real money.")
    print()


if __name__ == "__main__":
    main()
