"""
THE SIGNAL GRID: is our signal allergic to the winners?

    python -m scripts.experiment_signal        (slow: ~48 backtests)

Turnover was not the leak (0.55%/yr -- we tested it, the hypothesis died). The next
suspect is the signal itself.

VOLAR DIVIDES BY VOLATILITY. The stocks that made 2016-2026 -- NVDA, TSLA, AMD,
PLTR -- were violently volatile. We may have built a signal that systematically
penalises exactly the stocks that won. Meanwhile SPMO risk-adjusts over a much
gentler 3-year window and then multiplies by MARKET CAP, so it holds NVDA huge.
We hold it at 5%, if at all.

Three levers, tested together because they interact:

    SIGNAL         momentum (raw)  |  volar (÷ vol)  |  sharpe (÷ vol, risk-free)
    CONCENTRATION  top 10 / 20 / 50 / 100
    WEIGHTING      equal  |  inverse_vol

THE RULE OF THIS EXPERIMENT: we are looking for a config that works in BOTH eras.
Not the best cell. THE BEST CELL IS ALMOST CERTAINLY LUCK -- 24 configs × 2 eras is
48 chances for randomness to hand us a beautiful number (Harvey, Liu & Zhu 2016).

A signal that only wins in 2016-2026 is not an edge. It is a bet on the last decade
repeating, which is the single most expensive assumption in finance.
"""

from __future__ import annotations

import itertools

import pandas as pd

from engine.backtest import get_overlay, rebalance_dates, run_backtest
from engine.data import get_adapter
from engine.markets.market import load_market
from engine.metrics import cagr, max_drawdown, sharpe_ratio
from engine.signals import get_signal
from engine.universe.universe import load_membership

UNIVERSE = "sp500_pit"          # the honest one. Never tune on the biased list.

SIGNALS = ["momentum", "volar", "sharpe"]
TOP_NS = [10, 20, 50, 100]
WEIGHTINGS = ["equal", "inverse_vol"]

ERAS = {
    "FULL 2005-today": "2005-01-01",
    "MODERN 2016-today": "2016-01-01",
}


def main() -> None:
    market = load_market("us")
    membership = load_membership(market, UNIVERSE)
    adapter = get_adapter(market)

    today = pd.Timestamp.today().strftime("%Y-%m-%d")

    print()
    print("=" * 96)
    print("  SIGNAL GRID — is our volatility penalty costing us the winners?")
    print("=" * 96)
    print(f"  Universe: {UNIVERSE} "
          f"({'POINT-IN-TIME ✅' if membership.is_point_in_time else '⚠️ BIASED'})")
    print(f"  {len(SIGNALS)} signals × {len(TOP_NS)} sizes × {len(WEIGHTINGS)} weightings "
          f"× {len(ERAS)} eras = {len(SIGNALS)*len(TOP_NS)*len(WEIGHTINGS)*len(ERAS)} backtests")
    print("  This takes a few minutes. Go make tea.")
    print()

    prices = adapter.fetch(membership.symbols, "2003-06-01", today)
    benchmark = adapter.fetch_benchmark("2003-06-01", today)
    spy_series = adapter.fetch(["SPY"], "2003-06-01", today).close["SPY"].dropna()
    print(f"  {len(prices.symbols)} symbols loaded.")
    print()

    all_results: dict[str, list[dict]] = {}

    for era_name, start in ERAS.items():
        dates = rebalance_dates(market, start, today, "monthly")

        spy_vals = pd.Series([spy_series.asof(d) for d in dates], index=dates).dropna()
        spy_curve = (spy_vals / spy_vals.iloc[0]).iloc[1:]
        spy_cagr = cagr(spy_curve, 12)

        rows = []
        combos = list(itertools.product(SIGNALS, TOP_NS, WEIGHTINGS))

        print(f"  Running {era_name} ({len(combos)} configs)", end="", flush=True)

        for sig_name, top_n, weighting in combos:
            signal = get_signal(
                sig_name, lookback_months=12, skip_months=1
            ) if sig_name == "momentum" else get_signal(
                sig_name, lookback_months=12, skip_months=1, vol_window_days=126
            )

            result = run_backtest(
                market=market,
                membership=membership,
                prices=prices,
                signal=signal,
                rebalance_dates=dates,
                top_n=top_n,
                frequency="monthly",
                weighting=weighting,
                benchmark=benchmark,
                overlay=get_overlay("always_on"),   # isolate the signal
            )

            net = cagr(result.equity, 12)
            rows.append({
                "signal": sig_name,
                "top_n": top_n,
                "weighting": weighting,
                "key": f"{sig_name}/{top_n}/{weighting}",
                "net": net,
                "excess": net - spy_cagr,
                "maxdd": max_drawdown(result.equity),
                "sharpe": sharpe_ratio(result.period_returns, 12),
                "turnover": float(result.turnover.mean()),
            })
            print(".", end="", flush=True)

        print(" done.")

        rows.sort(key=lambda r: r["excess"], reverse=True)
        for i, r in enumerate(rows):
            r["rank"] = i + 1
        all_results[era_name] = rows
        all_results[f"{era_name}__spy"] = spy_cagr

    # ---- per-era tables ---------------------------------------------------
    for era_name in ERAS:
        rows = all_results[era_name]
        spy_cagr = all_results[f"{era_name}__spy"]

        print()
        print("=" * 96)
        print(f"  {era_name}      (SPY = {spy_cagr:.2%})")
        print("=" * 96)
        print(f"  {'RANK':<6}{'SIGNAL':<11}{'TOP N':>6}{'WEIGHTING':>14}"
              f"{'NET CAGR':>10}{'vs SPY':>9}{'MAX DD':>9}{'SHARPE':>8}{'TURN':>7}")
        print("  " + "-" * 92)
        for r in rows:
            print(
                f"  {r['rank']:<6}{r['signal']:<11}{r['top_n']:>6}{r['weighting']:>14}"
                f"{r['net']:>10.2%}{r['excess']:>+9.2%}{r['maxdd']:>9.1%}"
                f"{r['sharpe']:>8.2f}{r['turnover']:>7.0%}"
            )

    # ---- THE ONLY TABLE THAT MATTERS: consistency across eras --------------
    full = {r["key"]: r for r in all_results["FULL 2005-today"]}
    modern = {r["key"]: r for r in all_results["MODERN 2016-today"]}

    combined = []
    for key in full:
        f, m = full[key], modern[key]
        combined.append({
            "key": key,
            "full_rank": f["rank"],
            "modern_rank": m["rank"],
            "worst_rank": max(f["rank"], m["rank"]),     # robustness, not peak
            "full_excess": f["excess"],
            "modern_excess": m["excess"],
        })

    combined.sort(key=lambda r: r["worst_rank"])

    print()
    print("=" * 96)
    print("  CONSISTENCY — ranked by WORST era, not best")
    print("=" * 96)
    print("  A config that tops one era and flops the other is a period bet, not an edge.")
    print("  Sorting by the WORST of its two ranks finds what survives BOTH.")
    print()
    print(f"  {'CONFIG':<28}{'FULL RANK':>11}{'MODERN RANK':>13}"
          f"{'FULL vs SPY':>13}{'MODERN vs SPY':>15}")
    print("  " + "-" * 92)
    for r in combined[:10]:
        print(
            f"  {r['key']:<28}{r['full_rank']:>11}{r['modern_rank']:>13}"
            f"{r['full_excess']:>+13.2%}{r['modern_excess']:>+15.2%}"
        )

    print()
    print("  " + "-" * 92)
    print("  WORST PERFORMERS (what to avoid):")
    for r in combined[-3:]:
        print(
            f"  {r['key']:<28}{r['full_rank']:>11}{r['modern_rank']:>13}"
            f"{r['full_excess']:>+13.2%}{r['modern_excess']:>+15.2%}"
        )

    # ---- did the volatility penalty hurt us? ------------------------------
    print()
    print("=" * 96)
    print("  THE HYPOTHESIS UNDER TEST: does dividing by volatility cost us the winners?")
    print("=" * 96)

    for era_name in ERAS:
        rows = all_results[era_name]
        by_signal = {}
        for sig in SIGNALS:
            hits = [r["excess"] for r in rows if r["signal"] == sig]
            by_signal[sig] = sum(hits) / len(hits)

        print(f"  {era_name:<22}", end="")
        for sig in SIGNALS:
            print(f"{sig}: {by_signal[sig]:>+7.2%}   ", end="")
        print()

    print()
    print("  (average excess CAGR across all sizes and weightings, per signal)")
    print()
    print("  If RAW MOMENTUM beats VOLAR in the modern era but loses in the full era,")
    print("  the volatility penalty is not wrong -- it just missed one unusual decade.")
    print("  If raw momentum wins in BOTH, our signal has a real problem.")
    print()
    print("  ⚠️  48 backtests. At least one number here is luck. Look for the PATTERN.")
    print()


if __name__ == "__main__":
    main()
