"""
The universe: who is eligible to be ranked on a given date.

Two jobs:

1. MEMBERSHIP -- which securities belong to the index (S&P 500, Nifty 200, ...).
   V0 uses a CURRENT membership list, which introduces survivorship bias. This
   is disclosed loudly rather than hidden. (SPEC.md §4.4)

2. ELIGIBILITY -- of those members, which are actually tradable on date D:
     - enough price history for the signal to be computed
     - enough liquidity that we could really buy them

Both filters are computed AS OF a date, using only prior data. A universe that
peeks at the future is a look-ahead bug wearing a disguise.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import numpy as np

from engine.data.base import PriceData
from engine.markets.market import Market

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Membership
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Member:
    """
    One constituent of an index, over one membership INTERVAL.

    A company can join, be removed, and rejoin -- so one symbol may have several
    Member rows with different date ranges.

    added   = when it entered the index   (None = member since before our records)
    removed = when it left                (None = still a member)
    """

    symbol: str
    name: str = ""
    sector: str = ""
    added: pd.Timestamp | None = None
    removed: pd.Timestamp | None = None

    def was_member_on(self, when: pd.Timestamp) -> bool:
        if self.added is not None and when < self.added:
            return False
        if self.removed is not None and when >= self.removed:
            return False
        return True


@dataclass(frozen=True)
class Membership:
    """The index constituents, as loaded from config."""

    market: Market
    universe_key: str
    members: list[Member]
    survivorship_bias: bool
    disclaimer: str | None

    @property
    def symbols(self) -> list[str]:
        # De-duplicate: a symbol may appear in several membership intervals.
        seen: list[str] = []
        for m in self.members:
            if m.symbol not in seen:
                seen.append(m.symbol)
        return seen

    @property
    def is_point_in_time(self) -> bool:
        """True if we know WHEN each company was in the index."""
        return any(m.added is not None or m.removed is not None for m in self.members)

    def as_of(self, when: date | datetime | str) -> "Membership":
        """
        The index as it ACTUALLY WAS on `when`.

        This is the fix for inclusion bias. Using today's membership list to backtest
        2013 hands the strategy a universe pre-selected for the next decade's winners
        -- and momentum's whole job is to find winners. Here, a company is only
        available from the date it genuinely joined.
        """
        if not self.is_point_in_time:
            return self          # flat snapshot: nothing we can do, bias stands

        cutoff = pd.Timestamp(when)
        return Membership(
            market=self.market,
            universe_key=self.universe_key,
            members=[m for m in self.members if m.was_member_on(cutoff)],
            survivorship_bias=self.survivorship_bias,
            disclaimer=self.disclaimer,
        )

    def sector_of(self, symbol: str) -> str:
        for m in self.members:
            if m.symbol == symbol:
                return m.sector
        return ""

    def __len__(self) -> int:
        return len(self.members)

    def __repr__(self) -> str:
        return (
            f"<Membership {self.market.market_id}/{self.universe_key}: "
            f"{len(self.members)} members>"
        )


def load_membership(market: Market, universe_key: str) -> Membership:
    """
    Load index constituents for a market/universe pair.

    The FILE PATH comes from the market config -- the engine does not know or
    care that one list is American and another Indian.
    """
    universe = market.get_universe(universe_key)

    path = REPO_ROOT / universe.path
    if not path.exists():
        raise FileNotFoundError(
            f"Universe file missing: {path}\n"
            f"Build it first:  python -m scripts.build_universes"
        )

    def _date(value: str | None) -> pd.Timestamp | None:
        if not value or not str(value).strip():
            return None
        parsed = pd.to_datetime(str(value).strip(), errors="coerce")
        return None if pd.isna(parsed) else parsed

    members: list[Member] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            symbol = (row.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            members.append(
                Member(
                    symbol=symbol,
                    name=(row.get("name") or "").strip(),
                    sector=(row.get("sector") or "").strip(),
                    added=_date(row.get("start_date")),
                    removed=_date(row.get("end_date")),
                )
            )

    if not members:
        raise ValueError(f"Universe file is empty: {path}")

    return Membership(
        market=market,
        universe_key=universe_key,
        members=members,
        survivorship_bias=universe.survivorship_bias,
        disclaimer=universe.disclaimer,
    )


# ---------------------------------------------------------------------------
# Eligibility (as of a date)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UniverseSnapshot:
    """Who is investable on a specific date, and who got dropped and why."""

    market: Market
    universe_key: str
    as_of: pd.Timestamp
    eligible: list[str]
    dropped: dict[str, str] = field(default_factory=dict)

    @property
    def n_eligible(self) -> int:
        return len(self.eligible)

    def drop_reasons(self) -> dict[str, int]:
        """Tally of why securities were excluded -- useful for sanity-checking."""
        counts: dict[str, int] = {}
        for reason in self.dropped.values():
            counts[reason] = counts.get(reason, 0) + 1
        return counts

    def __repr__(self) -> str:
        return (
            f"<UniverseSnapshot {self.market.market_id}/{self.universe_key} "
            f"@{self.as_of.date()}: {len(self.eligible)} eligible, "
            f"{len(self.dropped)} dropped>"
        )


def eligible_universe(
    prices: PriceData,
    membership: Membership,
    as_of: date | datetime | str,
    min_history_days: int,
    max_staleness_days: int = 45,
    max_history_gap_days: int = 90,
) -> UniverseSnapshot:
    """
    Determine who can be ranked on `as_of`, using ONLY data from before it.

    Args:
        prices:            price history (any range -- it is cut here)
        membership:        index constituents
        as_of:             the rebalance date
        min_history_days:  trading days of history the signal needs
                           (e.g. 12-month momentum needs ~252)
        max_staleness_days:  newest bar must be at least this fresh, else the
                           security is not really tradeable on this date
        max_history_gap_days:  no gap this long inside the recent window, else the
                           series splices two different companies (recycled ticker)

    Dropped for:
        no_data           -- security never appears in the price data
        insufficient_history -- listed too recently to score
        stale_prices      -- has history, but it stopped long before this date
        history_gap       -- history is not contiguous (recycled ticker splice)
        illiquid          -- fails the market's min average daily traded value
    """
    market = membership.market
    cutoff = pd.Timestamp(as_of)

    # STEP 0: the index AS IT WAS on this date -- not as it is today.
    # Without this, a 2013 backtest can buy companies that only joined in 2024
    # BECAUSE they went up 50x. (See Membership.as_of)
    membership = membership.as_of(cutoff)

    # THE GUARD: only history strictly before the decision date. (SPEC.md §4.1)
    history = prices.up_to(cutoff)

    eligible: list[str] = []
    dropped: dict[str, str] = {}

    liq_window = market.liquidity.lookback_days
    min_value = market.liquidity.min_avg_daily_value

    members = list(membership.symbols)

    # VECTORISED eligibility. Same rules, same precedence, same drop reasons as the
    # per-symbol loop this replaced -- just computed as whole-DataFrame operations
    # instead of ~1,600 individual pandas calls per rebalance (which made a single
    # backtest take ~150s). The 214-test suite, incl. the recycled-ticker guards,
    # pins the behaviour: if those pass, this is doing exactly what the loop did.
    #
    # Precedence matters: a symbol's REASON is the FIRST check it fails, in order
    # no_data -> insufficient_history -> stale_prices -> history_gap -> illiquid.
    # We assign reasons in that order and never overwrite an earlier one.

    close = history.close
    volume = history.volume

    present = [s for s in members if s in close.columns]
    absent = [s for s in members if s not in close.columns]

    for s in absent:
        dropped[s] = "no_data"

    if not present:
        return UniverseSnapshot(
            market=market, universe_key=membership.universe_key, as_of=cutoff,
            eligible=[], dropped=dropped,
            survivorship_bias=membership.survivorship_bias,
            disclaimer=membership.disclaimer,
        )

    sub = close[present]
    valid = sub.notna()
    counts = valid.sum(axis=0)                       # bars per symbol

    # insufficient_history
    hist_ok = counts >= min_history_days
    for s in counts.index[~hist_ok]:
        dropped[s] = "insufficient_history"

    survivors = [s for s in present if hist_ok[s]]
    if not survivors:
        return UniverseSnapshot(
            market=market, universe_key=membership.universe_key, as_of=cutoff,
            eligible=[], dropped=dropped,
            survivorship_bias=membership.survivorship_bias,
            disclaimer=membership.disclaimer,
        )

    surv = sub[survivors]
    surv_valid = valid[survivors]

    # stale_prices (vectorised): last valid index per column. Trick: multiply the row
    # position by validity, take the max -> last valid row per symbol.
    row_pos = pd.Series(range(len(surv)), index=surv.index)
    last_pos = surv_valid.mul(row_pos, axis=0).where(surv_valid).max(axis=0)
    last_dates = surv.index[last_pos.astype(int).to_numpy()]
    stale_ok = pd.Series(True, index=survivors)
    if max_staleness_days > 0:
        age = (cutoff - pd.DatetimeIndex(last_dates)).days
        stale_ok = pd.Series(age <= max_staleness_days, index=survivors)
    for s in survivors:
        if not stale_ok[s]:
            dropped[s] = "stale_prices"

    after_stale = [s for s in survivors if stale_ok[s]]

    # liquidity (vectorised): traded value over the last liq_window rows.
    liq_ok = pd.Series(True, index=after_stale)
    if min_value > 0 and after_stale:
        tail_close = surv[after_stale].tail(liq_window)
        tail_vol = volume.reindex(columns=after_stale).reindex(tail_close.index).tail(liq_window)
        traded = (tail_close * tail_vol)
        liq_mean = traded.mean(axis=0)               # NaN-skipping mean per symbol
        liq_ok = (liq_mean >= min_value) & liq_mean.notna()

    # history_gap (vectorised where it counts): the largest spacing between
    # consecutive VALID bars inside each symbol's recent required-length window must
    # not exceed the limit. We must look at the tail of each symbol's NON-NULL bars
    # (matching the original col.dropna().tail(min_history_days)), because a sparse or
    # spliced series' recent real bars can span a long dormant gap.
    gap_bad: set[str] = set()
    if max_history_gap_days > 0 and after_stale:
        for s in after_stale:
            valid_idx = surv[s].dropna().index
            if len(valid_idx) >= 2:
                tail_idx = valid_idx[-min_history_days:]
                # Gaps in calendar days between consecutive valid bars. Use numpy
                # timedelta64[D] rather than hand-rolled epoch math (asi8 units vary
                # by index resolution and silently corrupt a manual day conversion).
                gaps_days = np.diff(tail_idx.values).astype("timedelta64[D]").astype(int)
                if gaps_days.size and int(gaps_days.max()) > max_history_gap_days:
                    gap_bad.add(s)

    for s in after_stale:
        if s in gap_bad:
            dropped[s] = "history_gap"
            continue
        if not liq_ok.get(s, False):
            dropped[s] = "illiquid"
            continue
        eligible.append(s)

    return UniverseSnapshot(
        market=market,
        universe_key=membership.universe_key,
        as_of=cutoff,
        eligible=eligible,
        dropped=dropped,
    )
