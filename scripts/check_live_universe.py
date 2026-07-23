"""
CAN TODAY'S SIGNAL BE TRUSTED? -- audit the live universe before anyone trades it.

    python -m scripts.check_live_universe [strategy_name]

A backtest can be wrong and only mislead us. A LIVE signal that is wrong loses real
money, so it deserves its own audit. Three things can silently corrupt it:

  1. STALE MEMBERSHIP. A company that left the index but whose record has no removal
     date still looks like a current member. We would rank and hold a name that is
     not in the index -- and if its ticker was reissued, we would be pricing a
     completely different company. (We already know some membership records carry
     placeholder dates, so this is a live risk, not a hypothetical.)

  2. RECYCLED / DEAD TICKERS AMONG "CURRENT" MEMBERS. If the membership says a name
     is current but no vendor can price it, the record is wrong.

  3. THIN OR BROKEN HISTORY on a name we are about to buy. The eligibility filter
     catches most of this, but for a live trade we want to SEE it, not trust it.

This is read-only and prints an audit, not a portfolio.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

from engine.data import get_adapter
from engine.markets.market import load_market
from engine.universe.universe import load_membership

DEFAULT_STRATEGY = "us_sp500_top20_momentum"
REPO_ROOT = Path(__file__).resolve().parents[1]
STRATEGY_DIR = REPO_ROOT / "config" / "strategies"
HISTORY_MONTHS = 24

# A real index has a known size. Wildly more "current" members than this means the
# membership file is carrying names it should have retired.
EXPECTED_SIZE = {"sp500_pit": 503, "sp500": 503, "sp900_pit": 900, "nifty200": 200}


def load_strategy(name: str) -> dict:
    path = STRATEGY_DIR / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in STRATEGY_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"No strategy '{name}'. Available: {available}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_STRATEGY
    cfg = load_strategy(name)
    market = load_market(cfg["market"])
    membership = load_membership(market, cfg["universe"])
    adapter = get_adapter(market)

    today = pd.Timestamp.today().normalize()
    current = membership.as_of(today)
    current_symbols = sorted(set(current.symbols))
    all_symbols = sorted(set(membership.symbols))

    print()
    print("=" * 78)
    print("  LIVE UNIVERSE AUDIT -- can today's signal be trusted?")
    print("=" * 78)
    print(f"  Universe file : {membership.universe_key}")
    print(f"  Total records : {len(all_symbols)}  (every name that was EVER a member)")
    print(f"  'Current' now : {len(current_symbols)}")

    expected = EXPECTED_SIZE.get(membership.universe_key)
    if expected:
        diff = len(current_symbols) - expected
        print(f"  Real index is : ~{expected}")
        if abs(diff) <= 25:
            print(f"  -> OK, within {abs(diff)} of the real index size.")
        else:
            print(f"  -> WARNING: {diff:+d} vs the real index. The membership file is")
            print("     probably carrying names it should have retired (missing")
            print("     removal dates), which would put non-members in the portfolio.")
    print()

    # ---- can every 'current' member actually be priced? --------------------
    print("  Checking whether every 'current' member can be priced...")
    fetch_start = (today - pd.DateOffset(months=HISTORY_MONTHS)).strftime("%Y-%m-%d")
    prices = adapter.fetch(current_symbols, fetch_start, today.strftime("%Y-%m-%d"))
    close = prices.close

    unpriced, thin, stale_px, ok = [], [], [], []
    for s in current_symbols:
        if s not in close.columns:
            unpriced.append(s)
            continue
        col = close[s].dropna()
        if col.empty:
            unpriced.append(s)
        elif len(col) < 250:
            thin.append((s, len(col)))
        elif (today - col.index[-1]).days > 10:
            stale_px.append((s, col.index[-1].date()))
        else:
            ok.append(s)

    print()
    print("-" * 78)
    print("  RESULT")
    print("-" * 78)
    print(f"  Priced and healthy       : {len(ok)}")
    print(f"  NOT PRICEABLE AT ALL     : {len(unpriced)}")
    print(f"  Thin history (<250 bars) : {len(thin)}")
    print(f"  Stale (no recent bars)   : {len(stale_px)}")
    print()

    if unpriced:
        print("  *** NOT PRICEABLE but marked CURRENT -- these records are wrong.")
        print("      Either the company left the index (missing removal date) or the")
        print("      ticker no longer trades. Either way they should not be rankable.")
        for s in unpriced[:40]:
            print(f"        {s}")
        if len(unpriced) > 40:
            print(f"        ... and {len(unpriced) - 40} more")
        print()

    if stale_px:
        print("  *** STALE PRICES -- last bar is old. Likely delisted or renamed:")
        for s, d in stale_px[:25]:
            print(f"        {s:<10} last bar {d}")
        if len(stale_px) > 25:
            print(f"        ... and {len(stale_px) - 25} more")
        print()

    if thin:
        print("  Thin history (recent listing/spin-off -- usually legitimate, but")
        print("  they cannot be momentum-ranked and will be filtered out):")
        for s, n in thin[:20]:
            print(f"        {s:<10}{n:>5} bars")
        if len(thin) > 20:
            print(f"        ... and {len(thin) - 20} more")
        print()

    # ---- verdict -----------------------------------------------------------
    print("=" * 78)
    print("  VERDICT")
    print("=" * 78)
    bad = len(unpriced) + len(stale_px)
    if bad == 0:
        print("  Every current member prices cleanly. The live universe is trustworthy.")
    else:
        pctbad = bad / max(len(current_symbols), 1)
        print(f"  {bad} of {len(current_symbols)} 'current' members ({pctbad:.1%}) cannot be")
        print("  priced or are stale. Until these records are fixed, the live signal is")
        print("  ranking against a universe that does not match the real index.")
        print()
        print("  This does NOT necessarily corrupt the chosen names -- unpriceable ones")
        print("  are dropped by the eligibility filter -- but it means the universe is")
        print("  not what the config claims, and a RECYCLED ticker among them could be")
        print("  priced (wrongly) and selected.")
    print()


if __name__ == "__main__":
    main()
