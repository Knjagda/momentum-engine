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
) -> UniverseSnapshot:
    """
    Determine who can be ranked on `as_of`, using ONLY data from before it.

    Args:
        prices:            price history (any range -- it is cut here)
        membership:        index constituents
        as_of:             the rebalance date
        min_history_days:  trading days of history the signal needs
                           (e.g. 12-month momentum needs ~252)

    Dropped for:
        no_data           -- security never appears in the price data
        insufficient_history -- listed too recently to score
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

    for symbol in membership.symbols:
        if symbol not in history.close.columns:
            dropped[symbol] = "no_data"
            continue

        closes = history.close[symbol].dropna()

        if len(closes) < min_history_days:
            dropped[symbol] = "insufficient_history"
            continue

        # Liquidity: average daily traded VALUE (price x volume), in market currency.
        # Volume alone is meaningless across price levels and currencies.
        if min_value > 0 and symbol in history.volume.columns:
            recent_close = history.close[symbol].tail(liq_window)
            recent_vol = history.volume[symbol].tail(liq_window)
            traded_value = (recent_close * recent_vol).dropna()

            if traded_value.empty or traded_value.mean() < min_value:
                dropped[symbol] = "illiquid"
                continue

        eligible.append(symbol)

    return UniverseSnapshot(
        market=market,
        universe_key=membership.universe_key,
        as_of=cutoff,
        eligible=eligible,
        dropped=dropped,
    )
