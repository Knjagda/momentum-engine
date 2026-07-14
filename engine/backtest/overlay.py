"""
Risk overlays: the "what if the market crashes?" answer.

The strategy picks the strongest stocks in the universe. But in 2008 the strongest
stocks still fell 40% -- "best of a bad lot" is not a defence. An overlay sits ABOVE
the stock selection and asks a different question: should we be invested AT ALL?

Two implemented:

  TrendFilter       benchmark above its long moving average? invest : cash.
                    Research: Faber (2007). Crude, cheap, and surprisingly effective.

  AbsoluteMomentum  benchmark's own trailing return positive? invest : cash.
                    Research: Moskowitz, Ooi & Pedersen (2012); Antonacci's dual
                    momentum pairs this with relative momentum.

This is what the pitch deck means by "if your portfolio drops >15%, shift to cash",
except done properly -- reacting to the MARKET's trend rather than to your own
losses after the fact.

Honest limits: a trend filter WILL whipsaw you in choppy sideways markets, getting
you out at the bottom and back in at the top, repeatedly, paying costs each time.
It buys crash protection and it pays for it. There is no free lunch here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class OverlayDecision:
    """Invest, or stand aside -- and why."""

    risk_on: bool
    reason: str
    detail: dict | None = None

    def __bool__(self) -> bool:
        return self.risk_on


class Overlay(ABC):
    """Decides whether to be invested at all on a given date."""

    name = "overlay"

    @abstractmethod
    def decide(self, benchmark: pd.Series, as_of: pd.Timestamp) -> OverlayDecision:
        """`benchmark` is the index price series. Cut to before as_of internally."""
        raise NotImplementedError


class AlwaysOn(Overlay):
    """No overlay. Always fully invested. The control case."""

    name = "always_on"

    def decide(self, benchmark: pd.Series, as_of: pd.Timestamp) -> OverlayDecision:
        return OverlayDecision(risk_on=True, reason="no_overlay")


class TrendFilter(Overlay):
    """
    Invest only while the benchmark trades above its own moving average.

    Research: Faber (2007), A Quantitative Approach to Tactical Asset Allocation.
    """

    name = "trend_filter"

    def __init__(self, ma_days: int = 200) -> None:
        self.ma_days = ma_days

    def decide(self, benchmark: pd.Series, as_of: pd.Timestamp) -> OverlayDecision:
        # THE GUARD: only data strictly before the decision date (SPEC §4.1).
        history = benchmark.loc[benchmark.index < pd.Timestamp(as_of)].dropna()

        if len(history) < self.ma_days:
            # Not enough history to judge. Default to invested rather than
            # inventing a signal we cannot support.
            return OverlayDecision(
                risk_on=True,
                reason="insufficient_history",
                detail={"have": len(history), "need": self.ma_days},
            )

        price = float(history.iloc[-1])
        ma = float(history.tail(self.ma_days).mean())
        risk_on = price > ma

        return OverlayDecision(
            risk_on=risk_on,
            reason="above_ma" if risk_on else "below_ma",
            detail={"price": price, "ma": ma, "ma_days": self.ma_days},
        )


class AbsoluteMomentum(Overlay):
    """
    Invest only while the benchmark's own trailing return is positive.

    Research: Moskowitz, Ooi & Pedersen (2012), Time Series Momentum.
    """

    name = "absolute_momentum"

    def __init__(self, lookback_months: int = 12) -> None:
        self.lookback_months = lookback_months

    def decide(self, benchmark: pd.Series, as_of: pd.Timestamp) -> OverlayDecision:
        from dateutil.relativedelta import relativedelta

        cutoff = pd.Timestamp(as_of)
        history = benchmark.loc[benchmark.index < cutoff].dropna()

        if history.empty:
            return OverlayDecision(risk_on=True, reason="insufficient_history")

        start = cutoff - relativedelta(months=self.lookback_months)
        if history.index.min() > start:
            return OverlayDecision(risk_on=True, reason="insufficient_history")

        p0 = float(history.asof(start))
        p1 = float(history.iloc[-1])

        if not (p0 > 0):
            return OverlayDecision(risk_on=True, reason="insufficient_history")

        trailing = (p1 / p0) - 1.0
        risk_on = trailing > 0

        return OverlayDecision(
            risk_on=risk_on,
            reason="positive_trend" if risk_on else "negative_trend",
            detail={"trailing_return": trailing, "lookback_months": self.lookback_months},
        )


_OVERLAYS = {
    "always_on": AlwaysOn,
    "trend_filter": TrendFilter,
    "absolute_momentum": AbsoluteMomentum,
}


def get_overlay(name: str, **params) -> Overlay:
    key = name.lower()
    if key not in _OVERLAYS:
        raise KeyError(f"Unknown overlay '{name}'. Available: {sorted(_OVERLAYS)}")
    return _OVERLAYS[key](**params)


def registered_overlays() -> list[str]:
    return sorted(_OVERLAYS)
