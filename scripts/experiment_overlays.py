"""
DO TWO RISK OVERLAYS STACK, OR JUST CHARGE THE PREMIUM TWICE?

    python -m scripts.experiment_overlays

We now have two ways to cut risk, and both came back "insurance with a premium":

    TREND FILTER   go to cash when the benchmark is below its 200d MA.
                   Reacts to the MARKET's trend.
    VOL TARGET     scale exposure down when the STRATEGY's own returns turn violent.
                   Reacts to the STRATEGY's realised risk.

They watch different things -- so stacking them could be COMPLEMENTARY (each catches
crises the other misses) or REDUNDANT (both fire in the same 2008-type panic, and we
pay the whipsaw premium twice for one benefit).

We do not know. This 2x2 finds out:

                    no vol-target        vol-target @ 12%
    no overlay      RAW BASELINE         vol only
    trend filter    trend only           BOTH STACKED

Across both eras. The question is whether BOTH beats the better of the two singles --
if it does not, stacking is just double taxation and we should never ship it on by
default.

⚠️ This is 8 backtests (4 configs x 2 eras). Low multiple-testing risk, but the
target_vol and MA-length are knobs; we read the SHAPE, not the single best cell.
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
TARGET_VOL = 0.12
MA_DAYS = 200

ERAS = {
    "FULL 2005-today": "2005-01-01",
    "MODERN 2016-today": "2016-01-01",
}

# (label, use_trend_filter, use_vol_target)
CONFIGS = [
    ("raw (baseline)",     False, False),
    ("trend filter only",  True,  False),
    ("vol-target only",    False, True),
    ("BOTH stacked",       True,  True),
]


def main() -> None:
    market = load_market("us")
    membership = load_membership(market, UNIVERSE)
    adapter = get_adapter(market)
    signal = get_signal("momentum", lookback_months=12, skip_months=1)

    today = pd.Timestamp.today().strftime("%Y-%m-%d")

    print()
    print("=" * 94)
    print("  OVERLAY INTERACTION — do trend filter + vol targeting stack or clash?")
    print("=" * 94)
    print(f"  Strategy : momentum / top {TOP_N} / equal")
    print(f"  Overlays : trend filter (200d MA)  x  vol-target ({TARGET_VOL:.0%})")
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
        print(f"  {'CONFIG':<22}{'CAGR':>9}{'vs SPY':>9}{'MAX DD':>9}"
              f"{'SHARPE':>8}{'SORTINO':>9}{'CALMAR':>8}{'CASH%':>7}")
        print("  " + "-" * 90)

        rows = {}
        for label, use_trend, use_vol in CONFIGS:
            overlay = get_overlay("trend_filter", ma_days=MA_DAYS) if use_trend else get_overlay("always_on")
            vt = VolatilityTarget(target_vol=TARGET_VOL, lookback_periods=6, periods_per_year=12) if use_vol else None

            result = run_backtest(
                market=market, membership=membership, prices=prices, signal=signal,
                rebalance_dates=dates, top_n=TOP_N, frequency="monthly",
                benchmark=benchmark, overlay=overlay, vol_target=vt,
            )

            eq = result.equity
            c = cagr(eq, 12)
            dd = max_drawdown(eq)
            calmar = c / abs(dd) if dd < 0 else 0.0
            cash_pct = float(result.cash_periods.mean()) if len(result.cash_periods) else 0.0

            rows[label] = {
                "cagr": c, "excess": c - spy_cagr, "dd": dd,
                "sharpe": sharpe_ratio(result.period_returns, 12),
                "sortino": sortino_ratio(result.period_returns, 12),
                "calmar": calmar, "cash": cash_pct,
            }

            print(
                f"  {label:<22}{c:>9.2%}{c - spy_cagr:>+9.2%}{dd:>9.1%}"
                f"{rows[label]['sharpe']:>8.2f}{rows[label]['sortino']:>9.2f}"
                f"{calmar:>8.2f}{cash_pct:>7.0%}"
            )

        # ---- the verdict for this era --------------------------------------
        print()
        both = rows["BOTH stacked"]
        trend = rows["trend filter only"]
        vol = rows["vol-target only"]
        best_single_dd = max(trend["dd"], vol["dd"])          # less negative = better
        best_single_sharpe = max(trend["sharpe"], vol["sharpe"])

        print(f"  Best single overlay: DD {best_single_dd:.1%}, Sharpe {best_single_sharpe:.2f}")
        print(f"  BOTH stacked:        DD {both['dd']:.1%}, Sharpe {both['sharpe']:.2f}")

        dd_gain = best_single_dd - both["dd"]     # positive = stacking cut DD further
        if both["dd"] > best_single_dd + 0.02 and both["sharpe"] >= best_single_sharpe - 0.03:
            print("  → COMPLEMENTARY: stacking cut drawdown further without wrecking Sharpe.")
        elif both["sharpe"] < best_single_sharpe - 0.05:
            print("  → REDUNDANT / DOUBLE-TAXED: stacking hurt Sharpe for little extra protection.")
        else:
            print("  → MARGINAL: stacking ≈ the better single overlay. Not worth the complexity.")
        print()

    print("=" * 94)
    print("  READING IT")
    print("=" * 94)
    print("  COMPLEMENTARY (ship-worthy): BOTH cuts drawdown meaningfully below the best")
    print("  single overlay, in BOTH eras, without gutting Sharpe.")
    print()
    print("  DOUBLE-TAXED (do not ship together): BOTH bleeds more return than either")
    print("  single, for a drawdown barely better than one alone. Two premiums, one benefit.")
    print()
    print("  Look at CASH% too: if BOTH parks far more in cash than either single, the")
    print("  two overlays are firing on the SAME crises -- redundant, not additive.")
    print()
    print("  ⚠️  Backtests are simulations, not predictions.")
    print()


if __name__ == "__main__":
    main()
