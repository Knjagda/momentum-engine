"""
Signals: how a security gets a score.

THE PLUG-IN CONTRACT (SPEC.md §3).

Every strategy in this engine -- momentum today, value or quality tomorrow --
is the same machine with a different Signal plugged in:

    score every security -> rank -> take top N -> weight -> hold -> rebalance

So a Signal has exactly one job: given price history and a date, return a score
per symbol. It does not know about portfolios, costs, weights, or countries.

Two rules every Signal must obey:

1. NO LOOK-AHEAD (SPEC §4.1). compute() cuts history to strictly-before as_of
   itself. Even if a caller hands it the full frame, it cannot cheat.

2. NO HARD-CODED MARKET WIRING (SPEC §2). Annualization, currency, calendar --
   all come from prices.market. A signal never assumes a country.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from engine.data.base import PriceData


@dataclass(frozen=True)
class SignalResult:
    """Scores for one signal, on one date. Higher score = more attractive."""

    name: str
    as_of: pd.Timestamp
    scores: pd.Series           # index = symbol, value = score (NaN if unscoreable)
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> pd.Series:
        """Only symbols that actually got a score. NaNs are not opinions."""
        return self.scores.dropna()

    def rank(self, ascending: bool = False) -> pd.Series:
        """Rank 1 = best. Unscoreable symbols are excluded, not ranked last."""
        return self.valid.rank(ascending=ascending, method="first").astype(int)

    def top(self, n: int) -> pd.Series:
        """The n highest-scoring symbols, best first."""
        return self.valid.sort_values(ascending=False).head(n)

    def zscore(self) -> pd.Series:
        """
        Standardize scores so different signals can be combined.

        Raw scores are not comparable across signals -- a momentum return of 0.4
        and a volatility ratio of 1.8 live on different scales. Z-scoring puts
        them on a common footing. This is the machinery that makes composite
        (multi-signal) strategies possible.
        """
        valid = self.valid
        if len(valid) < 2 or valid.std(ddof=0) == 0:
            return pd.Series(0.0, index=valid.index)
        return (valid - valid.mean()) / valid.std(ddof=0)

    def __len__(self) -> int:
        return len(self.valid)

    def __repr__(self) -> str:
        return (
            f"<SignalResult {self.name} @{self.as_of.date()}: "
            f"{len(self.valid)}/{len(self.scores)} scored>"
        )


class Signal(ABC):
    """Base class for every scoring method."""

    name: str = "signal"

    def __init__(self, **params: Any) -> None:
        self.params = params

    # -- the contract -------------------------------------------------------

    @property
    @abstractmethod
    def required_history_days(self) -> int:
        """
        Trading days of history this signal needs to produce a score.

        The universe filter uses this so that a stock which listed three months
        ago is never handed to a signal that needs twelve.
        """
        raise NotImplementedError

    @abstractmethod
    def _score(self, history: PriceData, as_of: pd.Timestamp, symbols: list[str]) -> pd.Series:
        """Score symbols using `history`, which is ALREADY cut to before as_of."""
        raise NotImplementedError

    # -- public entry point (this is where look-ahead is prevented) ----------

    def compute(
        self,
        prices: PriceData,
        as_of: date | datetime | str,
        symbols: list[str] | None = None,
    ) -> SignalResult:
        cutoff = pd.Timestamp(as_of)

        # THE GUARD. The signal never sees data from as_of or later, no matter
        # what the caller passed in.
        history = prices.up_to(cutoff)

        if symbols is None:
            symbols = list(history.close.columns)

        scores = self._score(history, cutoff, symbols)

        return SignalResult(
            name=self.name,
            as_of=cutoff,
            scores=scores.reindex(symbols),
            params=dict(self.params),
        )

    # -- shared helpers -----------------------------------------------------

    @staticmethod
    def _price_asof(series: pd.Series, when: pd.Timestamp) -> float:
        """
        Last traded price on or before `when`.

        Uses asof() rather than exact lookup because `when` is a CALENDAR date
        and markets close on weekends and holidays -- and those holidays differ
        between countries, which is exactly why we never assume a fixed offset.
        """
        clean = series.dropna()
        if clean.empty:
            return float("nan")
        try:
            value = clean.asof(when)
        except (KeyError, ValueError):
            return float("nan")
        return float(value) if pd.notna(value) else float("nan")

    @staticmethod
    def _annualized_vol(returns: pd.Series, trading_days_per_year: int) -> float:
        """
        Annualized standard deviation of daily returns.

        The annualization factor comes from the MARKET (252 US, 250 India) --
        it is not a constant in this file.
        """
        clean = returns.dropna()
        if len(clean) < 2:
            return float("nan")
        vol = float(clean.std(ddof=1)) * np.sqrt(trading_days_per_year)
        return vol if vol > 0 else float("nan")

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"<{type(self).__name__} {args}>"
