"""
Find RECYCLED TICKERS: one symbol, two different companies, spliced into one series.

    python -m scripts.detect_recycled_tickers

THE PROBLEM. A ticker is a slot, not an identity. Dean Foods was DF until its 2020
bankruptcy; some other company holds DF now. Tiingo faithfully returns whoever owns
the symbol, so a naive fetch splices two companies' price histories together. Any
calculation spanning the handoff invents a return -- a bankrupt $0.50 stock
"becoming" a $30 stock is a 6,000% gain the backtest would happily book. That is
worse than missing data: it is WRONG data wearing a real name.

WHAT THIS DOES. Measures the scale before we build any treatment:
  1. Splits each cached price series into ERAS -- contiguous runs of trading
     separated by a long gap (a dead ticker sitting unused, then reissued).
  2. Compares those eras against the symbol's point-in-time MEMBERSHIP intervals
     (which encode when the symbol referred to an index member, and how often).
  3. Flags anything suspicious: multiple eras, bars starting long after the symbol
     left the index, or a tiny recent stub tacked onto a long-dead history.

Read-only. Changes nothing. Tells us how many tickers are affected and how badly,
so the fix is sized to the real problem rather than my imagination.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from engine.data.tiingo_adapter import CACHE_DIR
from engine.markets.market import load_market
from engine.universe.universe import load_membership

UNIVERSE = "sp900_pit"

# A pause this long between trades means the symbol went dormant -- the signature of
# delisting followed by reissue. Real trading has holidays and halts, not half-year
# silences.
GAP_DAYS = 180

# An era this small, sitting after a long-dead history, is almost certainly a new
# company that just picked up the symbol.
STUB_BARS = 60


def split_eras(idx: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Split a date index into (start, end, n_bars) eras separated by long gaps."""
    if len(idx) == 0:
        return []
    gaps = idx.to_series().diff().dt.days
    breaks = [0] + [i for i, g in enumerate(gaps) if g and g > GAP_DAYS] + [len(idx)]
    eras = []
    for a, b in zip(breaks[:-1], breaks[1:]):
        chunk = idx[a:b]
        if len(chunk):
            eras.append((chunk[0], chunk[-1], len(chunk)))
    return eras


def main() -> None:
    market = load_market("us")
    membership = load_membership(market, UNIVERSE)

    # symbol -> list of (added, removed) membership intervals
    intervals: dict[str, list[tuple[pd.Timestamp | None, pd.Timestamp | None]]] = {}
    for m in membership.members:
        intervals.setdefault(m.symbol.upper(), []).append((m.added, m.removed))

    folder = Path(CACHE_DIR) / market.market_id.lower()
    files = sorted(folder.glob("*.csv"))
    if not files:
        print(f"\n  No cached Tiingo data in {folder}. Run the puller first.\n")
        return

    print()
    print("=" * 86)
    print("  RECYCLED TICKER SCAN -- cached Tiingo series vs membership intervals")
    print("=" * 86)
    print(f"  Scanned {len(files)} cached tickers. Gap threshold: {GAP_DAYS} days.\n")

    multi_era, late_start, clean = [], [], 0

    for f in files:
        sym = f.stem.upper()
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
        except Exception:
            continue
        close = df["close"].dropna() if "close" in df.columns else pd.Series(dtype=float)
        if close.empty:
            continue

        eras = split_eras(pd.DatetimeIndex(close.index))
        mem = intervals.get(sym, [])
        # Latest date the symbol was an index member (None == still a member).
        last_removed = None
        if mem:
            removals = [r for _a, r in mem if r is not None]
            if removals and len(removals) == len(mem):
                last_removed = max(removals)

        if len(eras) > 1:
            multi_era.append((sym, eras, mem))
        elif last_removed is not None and eras and eras[0][0] > last_removed:
            # Entire history begins AFTER the symbol left the index -- so these bars
            # cannot be the company we care about.
            late_start.append((sym, eras[0], last_removed))
        else:
            clean += 1

    # ---- multi-era: the classic recycled ticker ---------------------------
    print("-" * 86)
    print(f"  MULTI-ERA SERIES (a dormant gap, then trading resumes): {len(multi_era)}")
    print("-" * 86)
    if multi_era:
        print(f"  {'SYM':<7}{'ERA':<5}{'FROM':<12}{'TO':<12}{'BARS':>7}   VERDICT")
        for sym, eras, mem in multi_era[:40]:
            for n, (a, b, cnt) in enumerate(eras, 1):
                verdict = ""
                if n > 1:
                    verdict = "likely NEW company" if cnt <= STUB_BARS else "second era -- check"
                print(f"  {sym:<7}{n:<5}{str(a.date()):<12}{str(b.date()):<12}"
                      f"{cnt:>7}   {verdict}")
            spans = ", ".join(
                f"{(a.date() if a is not None else '?')}..{(r.date() if r is not None else 'now')}"
                for a, r in mem
            ) or "(no membership record)"
            print(f"  {'':7}membership: {spans}\n")
        if len(multi_era) > 40:
            print(f"  ... and {len(multi_era) - 40} more\n")
    else:
        print("  none\n")

    # ---- history entirely after the symbol left the index ------------------
    print("-" * 86)
    print(f"  HISTORY STARTS AFTER THE SYMBOL LEFT THE INDEX: {len(late_start)}")
    print("-" * 86)
    if late_start:
        print(f"  {'SYM':<7}{'BARS FROM':<12}{'TO':<12}{'BARS':>7}   LEFT INDEX")
        for sym, era, removed in late_start[:40]:
            a, b, cnt = era
            print(f"  {sym:<7}{str(a.date()):<12}{str(b.date()):<12}{cnt:>7}   "
                  f"{removed.date()}")
        if len(late_start) > 40:
            print(f"  ... and {len(late_start) - 40} more")
    else:
        print("  none")

    print()
    print("=" * 86)
    print(f"  Clean single-era series : {clean}")
    print(f"  Multi-era (recycled?)   : {len(multi_era)}")
    print(f"  Wrong-company history   : {len(late_start)}")
    print("=" * 86)
    print("  Multi-era and wrong-company series must NOT be used as-is: a return")
    print("  computed across the handoff is fabricated. Next step is to slice each")
    print("  series to the era matching its membership interval, and forbid any")
    print("  calculation from crossing a gap.")
    print()


if __name__ == "__main__":
    main()
