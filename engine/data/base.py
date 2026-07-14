"""
The data layer contract.

Every price source (yfinance today; Tiingo/Polygon/an India provider tomorrow)
implements PriceAdapter. The engine talks to this interface and never to a
specific vendor -- so swapping providers is a config change, not a rewrite.

SPEC.md §2: the adapter is named in the market config (`data_adapter:`), so
even the choice of vendor is per-market configuration.

CRITICAL (SPEC.md §4.1 -- no look-ahead):
    PriceData.up_to(date) is the ONLY sanctioned way for a signal to see history.
    It returns strictly-before-date rows. A signal that reaches around this and
    touches the raw frame can see the future, and its backtest is a lie.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from engine.markets.market import Market


@dataclass(frozen=True)
class PriceData:
    """
    Adjusted daily prices for a set of securities in one market.

    Frames are indexed by date; columns are DISPLAY symbols (no vendor suffix),
    so downstream code never sees provider-specific tickers.

    `close` is SPLIT- AND DIVIDEND-ADJUSTED. Using raw close prices silently
    corrupts every return calculation, so the adapter always adjusts.
    """

    market: Market
    close: pd.DataFrame
    volume: pd.DataFrame

    # -- the anti-look-ahead gate -------------------------------------------

    def up_to(self, as_of: date | datetime | str) -> "PriceData":
        """
        History strictly BEFORE `as_of`.

        This is the guardrail that makes backtests honest. On rebalance date D,
        a signal may only use data that existed before D -- you cannot trade on
        D's close using D's close.
        """
        cutoff = pd.Timestamp(as_of)
        return PriceData(
            market=self.market,
            close=self.close.loc[self.close.index < cutoff],
            volume=self.volume.loc[self.volume.index < cutoff],
        )

    # -- convenience --------------------------------------------------------

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    @property
    def start(self) -> pd.Timestamp | None:
        return self.close.index.min() if len(self.close) else None

    @property
    def end(self) -> pd.Timestamp | None:
        return self.close.index.max() if len(self.close) else None

    def returns(self) -> pd.DataFrame:
        """Daily simple returns from adjusted closes."""
        return self.close.pct_change()

    def drop_incomplete(self, min_coverage: float = 0.9) -> "PriceData":
        """
        Remove securities with too much missing history.

        A stock that listed halfway through the test period will have NaNs, and
        a naive ranking would either crash or silently favour it. Drop it instead.
        """
        if self.close.empty:
            return self
        coverage = self.close.notna().mean()
        keep = coverage[coverage >= min_coverage].index
        return PriceData(
            market=self.market,
            close=self.close[keep],
            volume=self.volume[keep] if not self.volume.empty else self.volume,
        )

    def __repr__(self) -> str:
        n = len(self.symbols)
        rows = len(self.close)
        span = f"{self.start.date()} → {self.end.date()}" if rows else "empty"
        return f"<PriceData {self.market.market_id}: {n} symbols, {rows} days, {span}>"


class PriceAdapter(ABC):
    """Base class for every price data source."""

    def __init__(self, market: Market) -> None:
        self.market = market

    @abstractmethod
    def fetch(
        self,
        symbols: list[str],
        start: date | datetime | str,
        end: date | datetime | str,
    ) -> PriceData:
        """
        Fetch adjusted daily prices for `symbols` between `start` and `end`.

        Implementations MUST:
          1. Convert display symbols to vendor tickers via market.resolve_ticker()
          2. Return SPLIT- AND DIVIDEND-ADJUSTED closes
          3. Return columns as DISPLAY symbols (vendor suffix stripped)
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__} market={self.market.market_id}>"
