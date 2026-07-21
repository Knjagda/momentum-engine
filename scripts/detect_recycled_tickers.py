"""
Find price data that can actually CONTAMINATE the backtest (v2, corrected logic).

    python -m scripts.detect_recycled_tickers

WHY V2. The first version asked "do the bars start after the symbol left the index?"
That over-flagged badly: W.R. Grace left the S&P in 2000 but kept trading publicly
until its 2021 acquisition, and Apollo Education traded for 15 months after leaving.
Their data is perfectly good. Leaving an index is not dying.

THE CORRECT QUESTION. Our engine only ever holds a symbol while it is a member (the
rebalance loop calls eligible_universe(as_of=d) every period). So price data OUTSIDE
a symbol's membership intervals is never selected, never priced, never booked -- it
cannot contaminate anything, no matter whose company it belongs to. What matters is
only this:

    Is there suspicious price data INSIDE a membership interval we actually trade?

Three ways that can happen, and this script reports each:

  RISK 1  SPLICE INSIDE A HELD WINDOW. A long dormant gap falls inside a membership
          interval, so one column holds two companies during a period we trade.
          A return computed across that handoff is fabricated.

  RISK 2  LATE START INSIDE A HELD WINDOW. Price data begins well after the interval
          starts, i.e. we are missing the company we should be holding and may be
          looking at a successor.

  RISK 3  NO DATA AT ALL for a traded interval -- harmless (the name is simply
          skipped) but worth counting, since it is residual survivorship.

Everything else is reported as OUT-OF-WINDOW ONLY: real recycling, but structurally
harmless because the impostor bars sit outside every interval we trade.

Read-only. Changes nothing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from engine.data.tiingo_adapter import CACHE_DIR
from engine.markets.market import load_market
from engine.universe.universe import load_membership

UNIVERSE = "sp900_pit"

# The backtest window. Membership intervals ending before this are never traded, so
# their price data -- right or wrong -- is irrelevant.
BACKTEST_START = pd.Timestamp("2010-06-01")

# A dormant pause this long means the symbol stopped trading and was later reissued.
GAP_DAYS = 180

# Price data starting this long after a traded interval begins is suspicious: we are
# missing the company we should have been holding.
LATE_START_DAYS = 365


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


def _overlaps(a_start, a_end, b_start, b_end) -> bool:
    return a_start <= b_end and b_start <= a_end


def main() -> None:
    market = load_market("us")
    membership = load_membership(market, UNIVERSE)
    today = pd.Timestamp.today()

    intervals: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for m in membership.members:
        start = m.added if m.added is not None else pd.Timestamp("1900-01-01")
        end = m.removed if m.removed is not None else today
        intervals.setdefault(m.symbol.upper(), []).append((start, end))

    folder = Path(CACHE_DIR) / market.market_id.lower()
    files = sorted(folder.glob("*.csv"))
    if not files:
        print(f"\n  No cached Tiingo data in {folder}. Run the puller first.\n")
        return

    print()
    print("=" * 88)
    print("  CONTAMINATION SCAN (v2) -- only data inside TRADED windows can hurt us")
    print("=" * 88)
    print(f"  Cached tickers: {len(files)}   Backtest window: {BACKTEST_START.date()} ->")
    print(f"  Gap threshold: {GAP_DAYS} days   Late-start threshold: {LATE_START_DAYS} days\n")

    risk_splice, risk_late, risk_nodata, harmless = [], [], [], []
    clean = 0

    for f in files:
        sym = f.stem.upper()
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
        except Exception:
            continue
        close = df["close"].dropna() if "close" in df.columns else pd.Series(dtype=float)
        eras = split_eras(pd.DatetimeIndex(close.index)) if not close.empty else []

        # Only intervals we actually trade matter.
        traded = [(s, e) for s, e in intervals.get(sym, []) if e >= BACKTEST_START]
        if not traded:
            continue  # never held in our window -- data cannot be used at all

        flagged = False
        for s, e in traded:
            s_eff = max(s, BACKTEST_START)
            covering = [er for er in eras if _overlaps(er[0], er[1], s_eff, e)]

            if not covering:
                risk_nodata.append((sym, s_eff, e))
                flagged = True
                continue

            # RISK 1: a gap between two eras falls inside the traded window.
            if len(covering) > 1:
                risk_splice.append((sym, s_eff, e, covering))
                flagged = True
                continue

            # RISK 2: the covering era starts well after the window opens.
            era_start = covering[0][0]
            if (era_start - s_eff).days > LATE_START_DAYS:
                risk_late.append((sym, s_eff, e, era_start, covering[0][2]))
                flagged = True

        if not flagged:
            outside = [
                er for er in eras
                if not any(_overlaps(er[0], er[1], max(s, BACKTEST_START), e)
                           for s, e in traded)
            ]
            if outside:
                harmless.append((sym, outside))
            else:
                clean += 1

    def _hdr(title: str, n: int) -> None:
        print("-" * 88)
        print(f"  {title}: {n}")
        print("-" * 88)

    _hdr("RISK 1 - SPLICE INSIDE A TRADED WINDOW (fabricated returns possible)",
         len(risk_splice))
    for sym, s, e, covering in risk_splice[:30]:
        spans = " | ".join(f"{a.date()}..{b.date()} ({n})" for a, b, n in covering)
        print(f"  {sym:<7} traded {s.date()}..{e.date()}   eras: {spans}")
    if not risk_splice:
        print("  none")
    print()

    _hdr("RISK 2 - DATA STARTS LATE INSIDE A TRADED WINDOW", len(risk_late))
    for sym, s, e, era_start, n in risk_late[:30]:
        print(f"  {sym:<7} traded {s.date()}..{e.date()}   data starts "
              f"{era_start.date()} ({n} bars) -- {(era_start - s).days} days late")
    if not risk_late:
        print("  none")
    print()

    _hdr("RISK 3 - NO DATA FOR A TRADED WINDOW (residual survivorship)",
         len(risk_nodata))
    for sym, s, e in risk_nodata[:30]:
        print(f"  {sym:<7} traded {s.date()}..{e.date()}   no cached bars")
    if len(risk_nodata) > 30:
        print(f"  ... and {len(risk_nodata) - 30} more")
    if not risk_nodata:
        print("  none")
    print()

    _hdr("OUT-OF-WINDOW ONLY - real recycling, structurally harmless", len(harmless))
    for sym, outside in harmless[:20]:
        spans = " | ".join(f"{a.date()}..{b.date()} ({n})" for a, b, n in outside)
        print(f"  {sym:<7} unused eras: {spans}")
    if len(harmless) > 20:
        print(f"  ... and {len(harmless) - 20} more")
    if not harmless:
        print("  none")

    print()
    print("=" * 88)
    print(f"  Clean                    : {clean}")
    print(f"  Harmless recycling       : {len(harmless)}")
    print(f"  RISK 1 splice-in-window  : {len(risk_splice)}")
    print(f"  RISK 2 late-start        : {len(risk_late)}")
    print(f"  RISK 3 no-data           : {len(risk_nodata)}")
    print("=" * 88)
    print("  Only RISK 1 and RISK 2 can produce WRONG numbers. RISK 3 is absence,")
    print("  which we disclose. Everything else is already handled by the fact that")
    print("  the engine only holds a symbol inside its membership intervals.")
    print()


if __name__ == "__main__":
    main()
