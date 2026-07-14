"""
REALITY CHECK: validate the engine against real momentum funds.

    python -m scripts.validate

A backtest can claim anything. Real momentum ETFs cannot -- they trade with real
money, real costs, and no survivorship bias, and their results are public.

So we run OUR engine over the exact period those funds have been live, on the same
market, and compare. Three outcomes:

    Our edge ≈ theirs          -> the engine is probably measuring reality.
    Our edge >> theirs         -> we are measuring our own bias.
    Our edge collapses in the
    recent era but was huge
    in the old era             -> survivorship bias, precisely located.

The last one is the diagnosis we most expect. Companies that DIED are missing from
our universe -- and momentum in 2007 would have loaded up on financials and
homebuilders, some of which went to zero. We did not just delete some losers; we
deleted exactly the losers this strategy would have bought.

Benchmarks used:
    SPY   S&P 500                          (the market)
    MTUM  iShares MSCI USA Momentum        (live Apr 2013; ~125 names, cap-weighted)
    SPMO  Invesco S&P 500 Momentum         (live Oct 2015; concentrated -- closest to us)

SPMO is the fairest comparison: like us, it takes the strongest names within the
S&P 500. MTUM is broader and more diluted.
"""

from __future__ import annotations

import pandas as pd

from engine.backtest import get_overlay, rebalance_dates, run_backtest
from engine.data import get_adapter
from engine.markets.market import load_market
from engine.metrics import cagr, max_drawdown, sharpe_ratio
from engine.signals import get_signal
from engine.universe.universe import load_membership

ETFS = {
    "SPY": "S&P 500 (the market)",
    "MTUM": "iShares Momentum (live 2013)",
    "SPMO": "Invesco S&P500 Momentum (live 2015)",
}

# Eras. The split is the point: MTUM went live in 2013.
ERAS = {
    "2005-2012 (pre-MTUM)": ("2005-01-01", "2012-12-31"),
    "2013-today (MTUM era)": ("2013-06-01", None),
    # SPMO only launched in late 2015. Comparing our 2013-start curve against
    # SPMO's 2015-start curve was MY BUG -- different periods, different markets.
    # This era covers SPMO fully, so the comparison is finally apples-to-apples.
    "2016-today (SPMO era, clean)": ("2016-01-01", None),
}

SIGNAL = "volar"
SIGNAL_PARAMS = {"lookback_months": 12, "skip_months": 1, "vol_window_days": 126}
TOP_N = 20
FREQ = "monthly"


def buy_and_hold_curve(prices: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Equity curve of simply holding something, sampled on our rebalance dates."""
    clean = prices.dropna()
    if clean.empty:
        return pd.Series(dtype=float)

    values = pd.Series([clean.asof(d) for d in dates], index=dates, dtype=float).dropna()
    if values.empty or values.iloc[0] <= 0:
        return pd.Series(dtype=float)

    return (values / values.iloc[0]).iloc[1:]


def stats(equity: pd.Series) -> dict:
    if equity.empty or len(equity) < 3:
        return {}
    rets = equity.pct_change().dropna()
    return {
        "cagr": cagr(equity, 12),
        "maxdd": max_drawdown(equity),
        "sharpe": sharpe_ratio(rets, 12),
        "final": float(equity.iloc[-1]),
    }


def row(label: str, s: dict, spy_cagr: float | None = None) -> str:
    if not s:
        return f"  {label:<34}{'— not enough data —':>36}"
    excess = ""
    if spy_cagr is not None:
        excess = f"{s['cagr'] - spy_cagr:>+9.2%}"
    return (
        f"  {label:<34}{s['cagr']:>8.2%}{s['maxdd']:>9.2%}"
        f"{s['sharpe']:>8.2f}{excess:>10}"
    )


def main() -> None:
    import sys

    universe = sys.argv[1] if len(sys.argv) > 1 else "sp500"

    market = load_market("us")
    membership = load_membership(market, universe)
    adapter = get_adapter(market)
    signal = get_signal(SIGNAL, **SIGNAL_PARAMS)

    today = pd.Timestamp.today().strftime("%Y-%m-%d")

    print()
    print("=" * 84)
    print("  REALITY CHECK — our engine vs real, live momentum funds")
    print("=" * 84)
    print("  Real funds trade real money with real costs and NO survivorship bias.")
    print("  If our edge is far larger than theirs, we are measuring our own bias.")
    print()
    print(f"  Universe: {membership.universe_key}  "
          f"({'POINT-IN-TIME ✅' if membership.is_point_in_time else 'today\'s list ⚠️ inclusion bias'})")
    print()

    print("  Fetching S&P 500 prices and ETF benchmarks...")
    prices = adapter.fetch(membership.symbols, "2003-06-01", today)
    etf_data = adapter.fetch(list(ETFS), "2003-06-01", today)
    print(f"  Got {len(prices.symbols)} stocks and {len(etf_data.symbols)} ETFs.")
    print()

    for era_name, (start, end) in ERAS.items():
        end = end or today
        dates = rebalance_dates(market, start, end, FREQ)

        if len(dates) < 6:
            continue

        print("=" * 84)
        print(f"  {era_name}      {start} → {end}")
        print("=" * 84)
        print(f"  {'':<34}{'CAGR':>8}{'MAX DD':>9}{'SHARPE':>8}{'vs SPY':>10}")
        print(f"  {'-' * 78}")

        # SPY first -- everything is measured against it.
        spy_curve = buy_and_hold_curve(etf_data.close.get("SPY", pd.Series(dtype=float)), dates)
        spy_stats = stats(spy_curve)
        spy_cagr = spy_stats.get("cagr")

        # --- OUR ENGINE, exactly as configured (with the trend filter) --------
        benchmark = adapter.fetch_benchmark("2003-06-01", end)

        ours_filtered = run_backtest(
            market=market, membership=membership, prices=prices, signal=signal,
            rebalance_dates=dates, top_n=TOP_N, frequency=FREQ,
            benchmark=benchmark, overlay=get_overlay("trend_filter", ma_days=200),
        )

        # --- OUR ENGINE, always invested --------------------------------------
        # The ETFs never go to cash, so THIS is the apples-to-apples comparison.
        # Comparing our cash-capable strategy to an always-invested fund would
        # flatter us on drawdown for a reason that has nothing to do with stock picking.
        ours_always = run_backtest(
            market=market, membership=membership, prices=prices, signal=signal,
            rebalance_dates=dates, top_n=TOP_N, frequency=FREQ,
            benchmark=benchmark, overlay=get_overlay("always_on"),
        )

        print(row("OUR ENGINE (always invested)", stats(ours_always.equity), spy_cagr))
        print(row("OUR ENGINE (+ trend filter)", stats(ours_filtered.equity), spy_cagr))
        print(f"  {'-' * 78}")

        for ticker, desc in ETFS.items():
            if ticker not in etf_data.close.columns:
                continue
            curve = buy_and_hold_curve(etf_data.close[ticker], dates)
            s = stats(curve)
            label = f"{ticker}  {desc}"
            print(row(label, s, spy_cagr if ticker != "SPY" else None))

        print()

        # --- the verdict -------------------------------------------------------
        our_stats = stats(ours_always.equity)
        spmo_curve = buy_and_hold_curve(etf_data.close.get("SPMO", pd.Series(dtype=float)), dates)
        spmo_stats = stats(spmo_curve)

        if our_stats and spy_cagr is not None:
            our_excess = our_stats["cagr"] - spy_cagr
            print(f"  Our excess CAGR over SPY (always invested): {our_excess:+.2%}")

            if spmo_stats:
                spmo_excess = spmo_stats["cagr"] - spy_cagr
                print(f"  SPMO's excess CAGR over SPY (real fund):    {spmo_excess:+.2%}")
                gap = our_excess - spmo_excess
                print()
                # A check that only catches OVERclaiming is half a check. Losing to a
                # real fund by 5%/yr is not "in the ballpark" -- it is a different failure,
                # and it needs to be called out just as loudly.
                if gap > 0.04:
                    print(f"  ⚠️  We claim {gap:.1%}/yr MORE than a real concentrated momentum fund.")
                    print("      That gap is our bias, not our skill.")
                elif gap > 0.015:
                    print(f"  ⚠️  We claim {gap:.1%}/yr more than SPMO. Some of this is likely bias.")
                elif gap < -0.02:
                    print(f"  ⚠️  We UNDERPERFORM a real momentum fund by {abs(gap):.1%}/yr.")
                    print("      Our data is now honest, but our strategy is losing to the")
                    print("      professionals. Suspects: turnover (we churn ~35%/month vs")
                    print("      SPMO's semi-annual), the overlay, or the signal itself.")
                else:
                    print("  ✅ Our edge is genuinely comparable to a real fund.")
            print()

    print("=" * 84)
    print("  HOW TO READ THIS")
    print("=" * 84)
    print("  If our advantage is far bigger in 2005-2012 than in 2013-today, that is the")
    print("  fingerprint of SURVIVORSHIP BIAS: the earlier era contains more companies that")
    print("  died and were removed from the index -- and momentum would have OWNED some of")
    print("  them into 2008. Our universe cannot buy them, so we never take those losses.")
    print()
    print("  ⚠️  Backtests are simulations, not predictions.")
    print("=" * 84)
    print()


if __name__ == "__main__":
    main()
