"""
TODAY'S SIGNALS -- what to hold right now, and what to trade to get there.

    python -m scripts.signals_today                 # uses the default strategy config
    python -m scripts.signals_today us_sp500_top20_momentum

This is the PRODUCT, not a backtest. A backtest answers "would this have worked?";
this answers "what do I do on Monday?" Those need different outputs: a user does not
want an equity curve, they want a list of trades.

WHY THIS NEEDS NO TIINGO KEY. Survivorship-free data exists to measure the PAST
honestly -- dead companies must be present so their failures are counted. But today's
index members are, by definition, all alive, and yfinance prices living companies
fine. So live signals need no delisted history, no API key, and run in seconds.

WHAT IT PRODUCES
  1. The overlay verdict -- invested, or standing aside in cash.
  2. Today's target portfolio (names + weights + sectors).
  3. THE TRADE LIST -- buy / sell / hold versus the last saved portfolio. This is
     the part a user actually acts on.
  4. A saved snapshot in data/portfolios/, so next month can diff against it.

The no-trade buffer is applied exactly as in the backtest: a holding that slips
below top_n is KEPT until it falls out of the wider exit band. Without this the
live signal would churn on noise and hand the difference to the broker.

IMPORTANT: this reads the same strategy config the backtest reads, so what you trade
is what was validated. If they drift apart, the validation stops meaning anything.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from engine.backtest import get_overlay
from engine.data import get_adapter
from engine.markets.market import load_market
from engine.portfolio.construction import build_portfolio, select_with_buffer
from engine.signals import get_signal
from engine.universe.universe import eligible_universe, load_membership

DEFAULT_STRATEGY = "us_sp500_top20_momentum"
REPO_ROOT = Path(__file__).resolve().parents[1]
STRATEGY_DIR = REPO_ROOT / "config" / "strategies"
SAVE_DIR = Path("data/portfolios")
# Enough runway for a 12-month lookback plus the 200-day trend MA, with slack.
HISTORY_MONTHS = 24


def load_strategy(name: str) -> dict:
    """Load a strategy config. (Duplicated from scripts/backtest.py; when the app
    layer lands this belongs in the engine so both the CLI and the app share it.)"""
    path = STRATEGY_DIR / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in STRATEGY_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"No strategy '{name}'. Available: {available}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_previous(strategy_id: str) -> dict | None:
    """Most recent saved snapshot for this strategy, if any."""
    if not SAVE_DIR.exists():
        return None
    files = sorted(SAVE_DIR.glob(f"{strategy_id}_*.json"))
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text())
    except Exception:
        return None


def _save(strategy_id: str, payload: dict) -> Path:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    path = SAVE_DIR / f"{strategy_id}_{payload['as_of']}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


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
    max_sector = cfg["weighting"].get("max_sector_weight")
    # THE NO-TRADE BUFFER IS OFF UNLESS THE CONFIG ASKS FOR IT.
    # This matters for integrity: every validation run (backtest, honest backtest,
    # out-of-sample, parameter sweep) used NO buffer, because run_backtest defaults
    # exit_rank=None and no script passes it. If the live signal quietly applied a
    # buffer, users would be trading a strategy we never tested. The buffer is very
    # likely WORTH adopting -- it cuts turnover and costs -- but adopting it means
    # re-validating first, exactly as we did when reversing the overlay decision.
    exit_rank = cfg.get("selection", {}).get("exit_rank")

    trend_cfg = cfg.get("overlay", {}).get("trend_filter", {})
    if trend_cfg.get("enabled"):
        overlay = get_overlay("trend_filter",
                              ma_days=trend_cfg.get("benchmark_ma_days", 200))
    else:
        overlay = get_overlay("always_on")

    today = pd.Timestamp.today().normalize()
    fetch_start = (today - pd.DateOffset(months=HISTORY_MONTHS)).strftime("%Y-%m-%d")

    print()
    print("=" * 78)
    print(f"  TODAY'S SIGNALS -- {cfg['name']}")
    print("=" * 78)
    print(f"  Date        : {today.date()}")
    print(f"  Universe    : {membership.universe_key} ({len(membership)} members)")
    print(f"  Strategy    : {signal}, top {top_n}, {weighting} weight")
    print(f"  Overlay     : {overlay.name}")
    print(f"  Buffer      : " + (f"exit_rank={exit_rank}" if exit_rank
                                 else "off (matches how the strategy was validated)"))
    print()
    print("  Fetching current prices...")

    prices = adapter.fetch(membership.symbols, fetch_start, today.strftime("%Y-%m-%d"))
    benchmark = adapter.fetch_benchmark(fetch_start, today.strftime("%Y-%m-%d"))
    print(f"  Got {len(prices.symbols)} symbols, through {prices.close.index[-1].date()}.\n")

    # ---- 1. overlay: are we invested at all? -------------------------------
    decision = overlay.decide(benchmark, today)
    print("-" * 78)
    print("  MARKET FILTER")
    print("-" * 78)
    if decision.risk_on:
        d = decision.detail
        extra = ""
        if "price" in d and "ma" in d:
            extra = f"  (benchmark {d['price']:,.0f} vs {d['ma_days']}d avg {d['ma']:,.0f})"
        print(f"  INVESTED -- {decision.reason}{extra}\n")
    else:
        d = decision.detail
        print(f"  STAND ASIDE -- {decision.reason}")
        if "price" in d and "ma" in d:
            print(f"  Benchmark {d['price']:,.0f} is below its {d['ma_days']}-day "
                  f"average of {d['ma']:,.0f}.")
        print("\n  TARGET: 100% CASH. Sell all holdings; hold nothing until the")
        print("  benchmark recovers above its trend. This is the rule that cut the")
        print("  worst drawdown from -65% to -34%.\n")

    # ---- 2. rank and build -------------------------------------------------
    previous = _load_previous(cfg["strategy_id"])
    prev_symbols = list(previous["symbols"]) if previous else []

    snapshot = eligible_universe(
        prices=prices,
        membership=membership,
        as_of=today,
        min_history_days=signal.required_history_days,
    )
    scores = signal.compute(prices, as_of=today, symbols=snapshot.eligible)

    if decision.risk_on:
        selected = (
            select_with_buffer(
                signal_result=scores, top_n=top_n,
                current_symbols=prev_symbols, exit_rank=exit_rank,
            )
            if exit_rank is not None
            else None
        )
        portfolio = build_portfolio(
            signal_result=scores, market=market, top_n=top_n,
            weighting=weighting, membership=membership, prices=prices,
            max_position_weight=max_pos, max_sector_weight=max_sector,
            preselected_symbols=selected,
        )
        target = portfolio.symbols
        weights = portfolio.weights
    else:
        target, weights = [], pd.Series(dtype=float)

    print("-" * 78)
    print(f"  UNIVERSE: {snapshot.n_eligible} of {len(membership)} names are rankable")
    print("-" * 78)
    for reason, n in sorted(snapshot.drop_reasons().items(), key=lambda kv: -kv[1]):
        print(f"    {reason:<24}{n:>5}")
    print()

    # ---- 3. today's holdings ----------------------------------------------
    if target:
        print("-" * 78)
        print(f"  TARGET PORTFOLIO ({len(target)} names)")
        print("-" * 78)
        frame = portfolio.to_frame()
        for row in frame.itertuples():
            print(f"  {row.rank:<4}{row.symbol:<10}{row.weight:>7.2%}   "
                  f"{str(row.sector)[:30]}")
        print()
        sect = portfolio.sector_weights()
        if not sect.empty:
            print("  Sector exposure:")
            for s, w in sorted(sect.items(), key=lambda kv: -kv[1]):
                flag = "   <-- concentrated" if w > 0.35 else ""
                print(f"    {str(s)[:30]:<32}{w:>7.1%}{flag}")
        print()

    # ---- 4. THE TRADE LIST -------------------------------------------------
    print("=" * 78)
    print("  WHAT TO TRADE")
    print("=" * 78)
    if previous is None:
        print(f"  No previous portfolio saved for '{cfg['strategy_id']}'.")
        print("  This is your STARTING portfolio -- buy everything listed above.")
        print(f"  Next run will compare against today and show only the changes.\n")
    else:
        prev_date = previous.get("as_of", "?")
        buys = [s for s in target if s not in prev_symbols]
        sells = [s for s in prev_symbols if s not in target]
        holds = [s for s in target if s in prev_symbols]

        print(f"  Compared with the portfolio saved on {prev_date}:\n")
        if not buys and not sells:
            print("  NO TRADES. The portfolio is unchanged -- the buffer absorbed")
            print("  this month's rank shuffling. Do nothing.\n")
        else:
            if sells:
                print(f"  SELL ({len(sells)}):")
                for s in sells:
                    print(f"    - {s}")
                print()
            if buys:
                print(f"  BUY ({len(buys)}):")
                for s in buys:
                    w = weights.get(s, 0)
                    print(f"    + {s:<10}{w:>7.2%}")
                print()
            print(f"  HOLD ({len(holds)}): " + ", ".join(holds[:12])
                  + (" ..." if len(holds) > 12 else ""))
            print()
            turnover = (len(buys) + len(sells)) / max(len(target), 1)
            print(f"  Turnover this rebalance: ~{turnover:.0%} of the portfolio.")
            print()

    # ---- 5. save -----------------------------------------------------------
    payload = {
        "strategy_id": cfg["strategy_id"],
        "as_of": str(today.date()),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "risk_on": bool(decision.risk_on),
        "overlay_reason": decision.reason,
        "symbols": list(target),
        "weights": {s: float(weights.get(s, 0)) for s in target},
        "universe": membership.universe_key,
        "n_eligible": snapshot.n_eligible,
    }
    path = _save(cfg["strategy_id"], payload)
    print("-" * 78)
    print(f"  Saved to {path}")
    print("  Next month's run will diff against this file to produce the trade list.")
    print("-" * 78)
    print()
    print("  Momentum is a rules-based strategy with real risk. Historical testing")
    print("  showed ~-34% worst drawdown and multi-year stretches behind the index.")
    print("  This is not investment advice.")
    print()


if __name__ == "__main__":
    main()
