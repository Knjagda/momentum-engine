"""
Prove the bulk adapter: same universe, old source vs new, coverage side by side.

    python -m scripts.compare_fundamentals your@email.com

The whole reason we built the bulk pipeline was coverage: the per-company EDGAR
fetch silently missed ~30% of the universe because of ticker->CIK gaps. This script
fetches the SAME tickers both ways and shows the difference in black and white.

It also spot-checks that the two sources AGREE on a number they both have -- if bulk
says Apple's equity is X, per-company EDGAR should say the same X. Same data, same
answer; only the delivery differs.
"""

from __future__ import annotations

import sys

import pandas as pd

from engine.data import get_adapter, get_fundamental_adapter
from engine.markets.market import load_market
from engine.universe.universe import load_membership

UNIVERSE = "sp500_pit"
AS_OF = "2024-06-01"


def main() -> None:
    if len(sys.argv) < 2 or "@" not in sys.argv[1]:
        print("\n  python -m scripts.compare_fundamentals you@email.com\n")
        return
    email = sys.argv[1]

    market = load_market("us")
    membership = load_membership(market, UNIVERSE)
    symbols = membership.symbols

    print()
    print("=" * 80)
    print("  FUNDAMENTALS COVERAGE — per-company EDGAR vs SEC bulk")
    print("=" * 80)
    print(f"  Universe: {UNIVERSE}  ({len(symbols)} tickers)")
    print()

    # --- new: bulk ---------------------------------------------------------
    print("  Fetching via SEC BULK (new)...")
    bulk = get_fundamental_adapter(
        "sec_bulk", email=email, start_quarter="2022q1", end_quarter="2024q4",
    )
    bulk_fd = bulk.fetch(symbols, verbose=True)
    bulk_syms = set(bulk_fd.symbols)
    print(f"  → bulk covers {len(bulk_syms)} / {len(symbols)} tickers "
          f"({len(bulk_syms)/len(symbols):.0%})")
    print()

    # --- old: per-company --------------------------------------------------
    print("  Fetching via per-company EDGAR (old)... (slower)")
    edgar = get_fundamental_adapter("edgar", market=market,
                                    user_agent=f"momentum-engine research {email}")
    edgar_fd = edgar.fetch(symbols, verbose=False)
    edgar_syms = set(edgar_fd.symbols)
    print(f"  → per-company covers {len(edgar_syms)} / {len(symbols)} tickers "
          f"({len(edgar_syms)/len(symbols):.0%})")
    print()

    # --- the difference ----------------------------------------------------
    only_bulk = bulk_syms - edgar_syms
    only_edgar = edgar_syms - bulk_syms
    print("=" * 80)
    print(f"  Covered by BULK but not per-company: {len(only_bulk)}")
    if only_bulk:
        print(f"     e.g. {sorted(only_bulk)[:15]}")
    print(f"  Covered by per-company but not BULK: {len(only_edgar)}")
    if only_edgar:
        print(f"     e.g. {sorted(only_edgar)[:15]}")
    print()

    # --- agreement spot-check ---------------------------------------------
    both = sorted(bulk_syms & edgar_syms)
    if both:
        print("  AGREEMENT CHECK (equity, as_of {}):".format(AS_OF))
        b_snap = bulk_fd.as_of(AS_OF, concepts=["equity"])
        e_snap = edgar_fd.as_of(AS_OF, concepts=["equity"])
        checked = 0
        for sym in both:
            if sym in b_snap.index and sym in e_snap.index:
                bv, ev = b_snap.loc[sym, "equity"], e_snap.loc[sym, "equity"]
                if pd.notna(bv) and pd.notna(ev):
                    match = "✓" if abs(bv - ev) < 1 else "✗ DIFFER"
                    if checked < 8 or match.startswith("✗"):
                        print(f"     {sym:<6} bulk={bv/1e9:>8.1f}B  edgar={ev/1e9:>8.1f}B  {match}")
                    checked += 1
        print(f"  Checked {checked} companies both sources have.")
    print()
    print("  If bulk coverage >> per-company, piece 3 did its job. If the numbers")
    print("  agree where both have them, the bulk pipeline is not just bigger -- it's")
    print("  correct. The remaining uncovered tickers are mostly delisted names whose")
    print("  CURRENT ticker the SEC map no longer lists (the known mapping limitation).")
    print()


if __name__ == "__main__":
    main()
