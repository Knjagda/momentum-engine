"""
HOLDINGS x SECTOR CAP -- does holding more names make a 25% cap viable?

    python -m scripts.experiment_holdings_x_cap YOUR_TIINGO_KEY [START_DATE]

THE QUESTION. 25% is the recognised global threshold for "concentrated": the SEC
treats >25% in one industry as concentration, and SEBI uses 25% for sector exposure.
But on our data a 25% cap at 20 holdings made things WORSE -- it cost return AND
deepened the drawdown versus no cap at all.

The suspicion is that the binding constraint is not the cap, it is the HOLDING COUNT.
At 20 names a 25% ceiling allows only five per sector, so the portfolio must reach
into weak sectors to fill the quota. Buying bad stocks to satisfy a rule is not
diversification, it is dilution. With more names the same ceiling allows seven or
eight per sector, which may be satisfiable without reaching.

So we test them TOGETHER rather than one at a time.

WHAT TO LOOK FOR. If the 25% column improves as holdings rise -- and especially if
some (holdings, 25%) combination beats our current 20-name portfolio on drawdown
without giving up much return -- then we can meet the global standard by evidence
rather than by assertion. If 25% hurts at every holding count, the honest conclusion
is that a momentum strategy cannot be squeezed to 25% and we should follow SPMO's
approach: no hard cap, prominent disclosure.

Costs are charged on every trade, so the extra turnover from holding more names is
already paid for in these numbers.
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
DEFAULT_START = "2005-01-01"
LOOKBACK_RUNWAY_YEARS = 2
DEAD_FILE = Path("data/dead_names.txt")

HOLDINGS = [15, 20, 25, 30, 35, 40]
CAPS = [None, 0.35, 0.25]          # uncapped reference, interim, global standard

BASELINE = (20, None)               # what we ship today


def _merge(a: PriceData, b: PriceData, market) -> PriceData:
    overlap = a.close.columns.intersection(b.close.columns)
    close = a.close.drop(columns=overlap).join(b.close, how="outer")
    volume = a.volume.drop(columns=overlap, errors="ignore").join(b.volume, how="outer")
    return PriceData(market=market, close=close.sort_index(),
                     volume=volume.sort_index())


def _label(cap) -> str:
    return "none" if cap is None else f"{cap:.0%}"


def main() -> None:
    if len(sys.argv) < 2:
        print("\n  python -m scripts.experiment_holdings_x_cap KEY [START_DATE]\n")
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
    print("  HOLDINGS x SECTOR CAP -- can more names make a 25% cap work?")
    print("=" * 92)
    print(f"  {UNIVERSE}, momentum 12-1, monthly, trend-filtered, survivorship-free")
    print(f"  Period: {start} -> {today}")
    print(f"  Grid: holdings {HOLDINGS} x caps {[_label(c) for c in CAPS]} "
          f"= {len(HOLDINGS) * len(CAPS)} runs\n")

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

    spy = yf_bench.reindex(dates, method="ffill")
    spy_curve = (spy / spy.iloc[0]).dropna()
    spy_cagr = cagr(spy_curve.iloc[1:], 12)
    print(f"  SPY: {spy_cagr:.2%} CAGR\n")
    print("  Running grid...\n")

    results = {}
    n = 0
    total = len(HOLDINGS) * len(CAPS)
    for tn in HOLDINGS:
        for cap in CAPS:
            n += 1
            try:
                r = run_backtest(
                    market=market, membership=membership, prices=prices, signal=signal,
                    rebalance_dates=dates, top_n=tn, frequency="monthly",
                    benchmark=yf_bench, overlay=get_overlay("trend_filter"),
                    weighting="equal", max_sector_weight=cap,
                )
            except ValueError as e:
                # The engine refuses caps that cannot be satisfied: capping at X%
                # needs at least ceil(1/X) sectors, so 25% needs 4+ and 35% needs 3+.
                # A momentum portfolio this small simply does not span enough sectors.
                # That is a FINDING, not a failure -- record it and continue.
                results[(tn, cap)] = {"infeasible": True, "why": str(e)}
                print(f"  [{n:>2}/{total}] {tn:>2} names, cap {_label(cap):<5} "
                      f"INFEASIBLE -- portfolio does not span enough sectors")
                continue
            eq = r.equity
            # what concentration did it actually reach?
            worst = None
            med_sectors = min_sectors = None
            try:
                cons = [float(p.sector_weights().max())
                        for p in r.portfolios if len(p.symbols)]
                worst = max(cons) if cons else None
                # How many DISTINCT sectors did the portfolio span? This is what
                # decides whether a cap is even arithmetically possible.
                spans = [int(len(p.sector_weights())) for p in r.portfolios
                         if len(p.symbols)]
                if spans:
                    med_sectors = float(pd.Series(spans).median())
                    min_sectors = int(min(spans))
            except Exception:
                pass
            results[(tn, cap)] = {
                "cagr": cagr(eq, 12),
                "excess": cagr(eq, 12) - spy_cagr,
                "dd": max_drawdown(eq),
                "sharpe": sharpe_ratio(r.period_returns, 12),
                "sortino": sortino_ratio(r.period_returns, 12),
                "worst": worst,
                "med_sectors": med_sectors,
                "min_sectors": min_sectors,
            }
            print(f"  [{n:>2}/{total}] {tn:>2} names, cap {_label(cap):<5} "
                  f"excess {results[(tn, cap)]['excess']:+.2%}  "
                  f"DD {results[(tn, cap)]['dd']:.1%}  "
                  f"Sharpe {results[(tn, cap)]['sharpe']:.2f}")

    # ---- how many sectors does momentum actually span? --------------------
    print()
    print("=" * 92)
    print("  SECTOR BREADTH -- the variable that decides whether a cap is possible")
    print("=" * 92)
    print("  Capping at X% needs at least ceil(1/X) sectors: 25% needs 4+, 35% needs 3+.")
    print()
    print(f"  {'NAMES':<8}{'MEDIAN SECTORS':>16}{'FEWEST EVER':>14}"
          f"{'25% POSSIBLE?':>16}{'35% POSSIBLE?':>16}")
    print("  " + "-" * 88)
    for tn in HOLDINGS:
        u = results[(tn, None)]
        med = u.get("med_sectors")
        mn = u.get("min_sectors")
        if med is None:
            print(f"  {tn:<8}{'n/a':>16}")
            continue
        ok25 = "always" if (mn or 0) >= 4 else "not always"
        ok35 = "always" if (mn or 0) >= 3 else "not always"
        print(f"  {tn:<8}{med:>16.1f}{mn:>14}{ok25:>16}{ok35:>16}")

    # ---- the 25% question, front and centre -------------------------------
    print()
    print("=" * 92)
    print("  DOES A 25% CAP GET BETTER WITH MORE NAMES?")
    print("=" * 92)
    print(f"  {'NAMES':<8}{'UNCAPPED EXC':>14}{'25% EXC':>10}{'COST':>9}"
          f"{'UNCAPPED DD':>13}{'25% DD':>9}{'DD CHANGE':>12}")
    print("  " + "-" * 88)
    for tn in HOLDINGS:
        u = results[(tn, None)]
        c = results[(tn, 0.25)]
        if c.get("infeasible"):
            print(f"  {tn:<8}{u['excess']:>+14.2%}{'--':>10}{'--':>9}"
                  f"{u['dd']:>13.1%}{'--':>9}"
                  f"{'not enough sectors':>12}")
            continue
        d_dd = (c["dd"] - u["dd"]) * 100     # +ve = shallower
        print(f"  {tn:<8}{u['excess']:>+14.2%}{c['excess']:>+10.2%}"
              f"{(c['excess']-u['excess'])*100:>+8.2f}p"
              f"{u['dd']:>13.1%}{c['dd']:>9.1%}"
              f"{d_dd:>+10.1f}pt")

    # ---- full grid --------------------------------------------------------
    print()
    print("=" * 92)
    print("  FULL GRID -- excess vs SPY / max drawdown / Sharpe")
    print("=" * 92)
    print(f"  {'NAMES':<8}" + "".join(f"{'cap ' + _label(c):>26}" for c in CAPS))
    print("  " + "-" * 88)
    for tn in HOLDINGS:
        cells = ""
        for cap in CAPS:
            r = results[(tn, cap)]
            if r.get("infeasible"):
                cells += f"{'infeasible':>23}   "
                continue
            cells += f"{r['excess']:>+8.2%} {r['dd']:>7.1%} {r['sharpe']:>6.2f}   "
        print(f"  {tn:<8}{cells}")

    # ---- everything vs what we ship today ---------------------------------
    base = results[BASELINE]
    print()
    print("=" * 92)
    print(f"  VS WHAT WE SHIP TODAY ({BASELINE[0]} names, cap {_label(BASELINE[1])}: "
          f"{base['excess']:+.2%} excess, {base['dd']:.1%} DD, "
          f"Sharpe {base['sharpe']:.2f})")
    print("=" * 92)
    better = []
    for (tn, cap), r in results.items():
        if (tn, cap) == BASELINE or r.get("infeasible"):
            continue
        d_dd = (r["dd"] - base["dd"]) * 100
        d_exc = (r["excess"] - base["excess"]) * 100
        # worth a look if drawdown improves without giving up much return
        if d_dd > 0.5 and d_exc > -1.5:
            better.append((tn, cap, d_exc, d_dd, r))
    if better:
        print("  Shallower drawdown for less than 1.5pt of excess return:\n")
        for tn, cap, d_exc, d_dd, r in sorted(better, key=lambda x: -x[3]):
            print(f"    {tn:>2} names, cap {_label(cap):<5} "
                  f"excess {d_exc:+.2f}pt, drawdown {d_dd:+.1f}pt shallower, "
                  f"Sharpe {r['sharpe']:.2f}, Sortino {r['sortino']:.2f}")
    else:
        print("  Nothing improves drawdown meaningfully without costing real return.")

    print()
    print("=" * 92)
    print("  HOW TO READ THIS")
    print("=" * 92)
    print("  If the 25% cap's drawdown improves as holdings rise, the cap was never")
    print("  the problem -- 20 names was. Adopt the smallest holding count where 25%")
    print("  stops hurting, and we meet the global standard on evidence.")
    print()
    print("  If 25% hurts at EVERY holding count, momentum cannot be squeezed that")
    print("  far. The honest answer is then SPMO's: no hard cap, prominent disclosure")
    print("  -- which is what a $11.5bn institutional momentum fund actually does.")
    print()
    print("  More names also means more trades. Costs are already charged here, so")
    print("  the numbers above are net of that.")
    print()
    print("  Backtests are simulations, not predictions.")
    print()


if __name__ == "__main__":
    main()
