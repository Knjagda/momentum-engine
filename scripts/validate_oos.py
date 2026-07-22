"""
OUT-OF-SAMPLE VALIDATION -- does the edge hold on data we did not design against?

    python -m scripts.validate_oos YOUR_TIINGO_KEY [START_DATE]

THE LAST HIDDEN ADVANTAGE. Every result so far -- the +4.7% edge, the trend-filter
adoption -- was measured on the SAME history we looked at while building the strategy.
Even without explicit curve-fitting, choices accrete: we picked 12-1 momentum, top 20,
monthly, this universe, partly because they behave well on this data. A strategy can
look good purely because it was selected on the sample it is judged on.

The honest question: does it hold on periods it was not chosen for?

This does not TUNE anything (tuning on the test set is the very sin we are checking
for). It takes the FIXED strategy and asks whether its performance is CONSISTENT
across independent sub-periods:

  1. SPLIT-SAMPLE. First half vs second half. If the edge appears in one half and
     vanishes in the other, the single-number summary is a regime artifact, not a
     durable property.

  2. PER-YEAR. Excess return every calendar year. A real edge shows up in many years,
     not one or two monsters. We report the hit rate (fraction of years beating SPY)
     and whether dropping the best single year destroys the edge.

  3. ROLLING 3-YEAR. Excess return over rolling 3-year windows -- how an investor who
     started at a bad time would actually have experienced it.

Consistency is the evidence. A strategy that beats SPY in both halves, in most years,
and across most rolling windows is far likelier to be real than one whose whole edge
lives in a single lucky stretch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from engine.backtest import get_overlay, rebalance_dates, run_backtest
from engine.data import get_adapter
from engine.data.base import PriceData
from engine.data.tiingo_adapter import TiingoAdapter
from engine.markets.market import load_market
from engine.metrics import cagr, max_drawdown, sharpe_ratio
from engine.signals import get_signal
from engine.universe.universe import load_membership

UNIVERSE = "sp900_pit"
TOP_N = 20
DEFAULT_START = "2005-01-01"
LOOKBACK_RUNWAY_YEARS = 2
DEAD_FILE = Path("data/dead_names.txt")


def _merge(a: PriceData, b: PriceData, market) -> PriceData:
    overlap = a.close.columns.intersection(b.close.columns)
    close = a.close.drop(columns=overlap).join(b.close, how="outer")
    volume = a.volume.drop(columns=overlap, errors="ignore").join(b.volume, how="outer")
    return PriceData(market=market, close=close.sort_index(),
                     volume=volume.sort_index())


def _period_stats(strat_ret: pd.Series, spy_ret: pd.Series, ppy: int) -> dict:
    """CAGR/excess/Sharpe for an aligned slice of strategy and SPY period returns."""
    strat_ret, spy_ret = strat_ret.align(spy_ret, join="inner")
    if len(strat_ret) < 2:
        return {}
    strat_curve = (1 + strat_ret).cumprod()
    spy_curve = (1 + spy_ret).cumprod()
    return {
        "n": len(strat_ret),
        "cagr": cagr(strat_curve, ppy),
        "spy_cagr": cagr(spy_curve, ppy),
        "excess": cagr(strat_curve, ppy) - cagr(spy_curve, ppy),
        "sharpe": sharpe_ratio(strat_ret, ppy),
        "dd": max_drawdown(strat_curve),
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("\n  python -m scripts.validate_oos YOUR_TIINGO_KEY [START_DATE]\n")
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
    print("=" * 88)
    print("  OUT-OF-SAMPLE VALIDATION -- is the edge consistent, or one lucky stretch?")
    print("=" * 88)
    print(f"  {UNIVERSE}, momentum 12-1, top {TOP_N}, monthly, trend-filtered")
    print(f"  Period: {start} -> {today}   (FIXED strategy -- nothing is tuned here)\n")

    # ---- survivorship-free prices -----------------------------------------
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

    result = run_backtest(
        market=market, membership=membership, prices=prices, signal=signal,
        rebalance_dates=dates, top_n=TOP_N, frequency="monthly",
        benchmark=yf_bench, overlay=get_overlay("trend_filter"),
    )
    ppy = result.periods_per_year
    strat = result.period_returns.dropna()

    # SPY period returns aligned to the same rebalance dates.
    spy_px = yf_bench.reindex(strat.index.union(yf_bench.index)).ffill()
    spy_ret = spy_px.reindex(strat.index).pct_change().dropna()
    strat, spy_ret = strat.align(spy_ret, join="inner")

    # ---- 1. SPLIT-SAMPLE ---------------------------------------------------
    mid = strat.index[len(strat) // 2]
    first = _period_stats(strat.loc[:mid], spy_ret.loc[:mid], ppy)
    second = _period_stats(strat.loc[mid:], spy_ret.loc[mid:], ppy)

    print("-" * 88)
    print("  1. SPLIT-SAMPLE -- first half vs second half (independent periods)")
    print("-" * 88)
    print(f"  {'HALF':<10}{'FROM':<12}{'TO':<12}{'CAGR':>8}{'SPY':>8}"
          f"{'EXCESS':>9}{'SHARPE':>8}{'MAX DD':>9}")
    for label, s, lo, hi in [
        ("first", first, strat.index[0], mid),
        ("second", second, mid, strat.index[-1]),
    ]:
        if s:
            print(f"  {label:<10}{str(lo.date()):<12}{str(hi.date()):<12}"
                  f"{s['cagr']:>8.1%}{s['spy_cagr']:>8.1%}{s['excess']:>+9.2%}"
                  f"{s['sharpe']:>8.2f}{s['dd']:>9.1%}")
    verdict = "EDGE IN BOTH HALVES" if (first.get("excess", -1) > 0 and
                                        second.get("excess", -1) > 0) else \
              "EDGE IN ONLY ONE HALF -- regime-dependent, treat with caution"
    print(f"\n  -> {verdict}\n")

    # ---- 2. PER-YEAR -------------------------------------------------------
    print("-" * 88)
    print("  2. PER-CALENDAR-YEAR excess vs SPY (a real edge shows up in many years)")
    print("-" * 88)
    yearly = []
    for yr, idx in strat.groupby(strat.index.year).groups.items():
        s_r = strat.loc[idx]
        p_r = spy_ret.loc[idx]
        s_tot = (1 + s_r).prod() - 1
        p_tot = (1 + p_r).prod() - 1
        yearly.append((yr, s_tot, p_tot, s_tot - p_tot))
    line = "  "
    for yr, s_tot, p_tot, exc in yearly:
        mark = "+" if exc > 0 else "-"
        line += f"{yr}:{mark}{abs(exc)*100:>4.0f}%  "
        if len(line) > 78:
            print(line); line = "  "
    if line.strip():
        print(line)
    wins = sum(1 for _y, _s, _p, e in yearly if e > 0)
    hit = wins / len(yearly) if yearly else 0
    # edge without the single best year?
    excesses = sorted((e for *_x, e in yearly), reverse=True)
    total_excess = sum(excesses)
    ex_best = total_excess - excesses[0] if excesses else 0
    print(f"\n  Beat SPY in {wins}/{len(yearly)} years ({hit:.0%} hit rate).")
    print(f"  Cumulative annual excess: {total_excess*100:+.0f}pt; "
          f"without the single best year: {ex_best*100:+.0f}pt "
          f"({'still positive' if ex_best > 0 else 'turns NEGATIVE -- edge is one year'}).\n")

    # ---- 3. ROLLING 3-YEAR -------------------------------------------------
    print("-" * 88)
    print("  3. ROLLING 3-YEAR excess (how a badly-timed start would have felt)")
    print("-" * 88)
    win = ppy * 3
    roll_ex = []
    if len(strat) >= win:
        for i in range(len(strat) - win + 1):
            s_sl = strat.iloc[i:i + win]
            p_sl = spy_ret.iloc[i:i + win]
            roll_ex.append((1 + s_sl).prod() ** (ppy / win) - 1
                           - ((1 + p_sl).prod() ** (ppy / win) - 1))
        roll = pd.Series(roll_ex)
        pos = (roll > 0).mean()
        print(f"  {len(roll)} rolling 3-year windows.")
        print(f"  Beat SPY in {pos:.0%} of them.")
        print(f"  Worst 3-yr excess: {roll.min()*100:+.1f}%/yr   "
              f"median: {roll.median()*100:+.1f}%/yr   "
              f"best: {roll.max()*100:+.1f}%/yr")
    else:
        print("  Not enough history for 3-year windows.")

    print()
    print("=" * 88)
    print("  READING THIS. Consistency across halves, years, and windows is the")
    print("  evidence of a durable edge. An edge that lives in one half, one year, or")
    print("  a few windows is likely overfit to this sample. This test cannot PROVE an")
    print("  edge is real -- but it can expose one that is not.")
    print()
    print("  Backtests are simulations, not predictions.")
    print()


if __name__ == "__main__":
    main()
