"""
The Portfolio object.

THE SEAM (SPEC.md §5). This is the engine's output and the boundary of the
deterministic core. Everything downstream consumes this and nothing reaches back
inside: metrics, reporting, the agentic explainer, and later the HEES portfolio
X-ray that will diagnose momentum's crowding problem.

A Portfolio is a decision, frozen in time: on THIS date, in THIS market, hold
THESE names at THESE weights, with this much in cash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Position:
    """One holding. Carries WHY it is held, not just how much."""

    symbol: str
    weight: float
    score: float = float("nan")
    rank: int = 0
    sector: str = ""

    def __repr__(self) -> str:
        return f"<{self.symbol} {self.weight:.1%} rank={self.rank}>"


@dataclass(frozen=True)
class Portfolio:
    """What to hold, on a given date, in a given market."""

    market_id: str
    currency: str
    as_of: pd.Timestamp
    positions: list[Position]
    cash_weight: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- views --------------------------------------------------------------

    @property
    def symbols(self) -> list[str]:
        return [p.symbol for p in self.positions]

    @property
    def weights(self) -> pd.Series:
        return pd.Series({p.symbol: p.weight for p in self.positions}, dtype=float)

    @property
    def n_positions(self) -> int:
        return len(self.positions)

    @property
    def invested_weight(self) -> float:
        return float(sum(p.weight for p in self.positions))

    def sector_weights(self) -> pd.Series:
        """
        Exposure by sector.

        A momentum Top-20 can silently become one giant sector bet -- this is the
        crudest possible check on that. HEES will do the real job later, because
        sectors are only the surface: five different "sectors" can all be riding
        the same underlying driver.
        """
        totals: dict[str, float] = {}
        for p in self.positions:
            key = p.sector or "Unknown"
            totals[key] = totals.get(key, 0.0) + p.weight
        return pd.Series(totals, dtype=float).sort_values(ascending=False)

    def to_frame(self) -> pd.DataFrame:
        """Tabular view -- for reporting, export, and eyeballing."""
        return pd.DataFrame(
            [
                {
                    "rank": p.rank,
                    "symbol": p.symbol,
                    "weight": p.weight,
                    "score": p.score,
                    "sector": p.sector,
                }
                for p in self.positions
            ]
        ).sort_values("rank").reset_index(drop=True)

    # -- integrity ----------------------------------------------------------

    def validate(self, tolerance: float = 1e-6) -> None:
        """
        Weights must sum to 1 (including cash), and none may be negative.

        A portfolio whose weights do not sum to 1 is not a portfolio -- it is a
        bug that will quietly distort every return figure downstream.
        """
        total = self.invested_weight + self.cash_weight

        if abs(total - 1.0) > tolerance:
            raise ValueError(
                f"Weights sum to {total:.6f}, not 1.0 "
                f"(invested={self.invested_weight:.6f}, cash={self.cash_weight:.6f})"
            )

        for p in self.positions:
            if p.weight < -tolerance:
                raise ValueError(f"Negative weight for {p.symbol}: {p.weight}")

        if self.cash_weight < -tolerance:
            raise ValueError(f"Negative cash weight: {self.cash_weight}")

    def __len__(self) -> int:
        return len(self.positions)

    def __repr__(self) -> str:
        return (
            f"<Portfolio {self.market_id} @{self.as_of.date()}: "
            f"{self.n_positions} positions, {self.cash_weight:.0%} cash>"
        )


def cash_portfolio(market_id: str, currency: str, as_of: pd.Timestamp, **metadata) -> Portfolio:
    """
    100% cash. What a risk overlay produces when it says "get out".

    This is a real, valid portfolio -- not an error state. Being in cash is a
    position, and the backtest must be able to hold it.
    """
    return Portfolio(
        market_id=market_id,
        currency=currency,
        as_of=pd.Timestamp(as_of),
        positions=[],
        cash_weight=1.0,
        metadata=metadata,
    )
