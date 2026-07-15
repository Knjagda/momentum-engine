"""
Volatility targeting: hold LESS when the strategy itself is getting dangerous.

THE DISTINCTION THAT MATTERS -- and that we got wrong once already.

Volar (which our own grid showed is our WEAKEST signal) divides each STOCK's return
by that STOCK's volatility. It is a cross-sectional, stock-picking penalty. It asks
"which stocks are choppy?" and it did not help.

This is a different animal entirely. It asks "is the STRATEGY dangerous right now?"
and scales the WHOLE portfolio's exposure up or down in response. When momentum's
own recent returns have been violent, we hold less of everything and park the rest
in cash. When calm, we hold the full amount.

    exposure = target_volatility / recent_realised_volatility     (capped at 1.0)

Research: Barroso & Santa-Clara (2015), "Momentum has its moments." Scaling momentum
by its own ~6-month realised volatility to a constant target took the strategy's
Sharpe from 0.53 to 0.97 and -- crucially for us -- roughly HALVED the worst
drawdowns. Momentum crashes happen in predictable high-volatility regimes (Daniel &
Moskowitz 2016); this steps out of them mechanically.

WHY IT SHOULD HELP OUR -28% PROBLEM: momentum's ugliest losses cluster in panicky,
high-vol markets (2008-09, 2020, the 2009 rebound where beaten-down junk rocketed).
In exactly those windows, trailing vol spikes -- so this dials exposure down BEFORE
the worst of it, using only past data.

Two honest limits, stated up front:
  1. Capped at 1.0 -- we DELEVER in danger but never LEVER up in calm. Leverage on
     a family account is a different risk conversation we are not having yet.
  2. It is not free. In a calm market that suddenly craters (no vol warning), it
     cannot help. And it will sometimes cut exposure right before a rebound, missing
     upside. It reduces the WORST outcomes; it does not raise every outcome.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VolTargetDecision:
    """How much of the portfolio to actually hold this period, and why."""

    exposure: float                 # 0.0 (all cash) .. 1.0 (fully invested)
    realised_vol: float | None      # annualised, from trailing strategy returns
    target_vol: float
    reason: str

    def __repr__(self) -> str:
        rv = "n/a" if self.realised_vol is None else f"{self.realised_vol:.1%}"
        return f"<VolTarget exposure={self.exposure:.2f} realised={rv} target={self.target_vol:.1%}>"


class VolatilityTarget:
    """
    Scales exposure so the strategy's forward risk sits near a constant target.

    Feed it the strategy's OWN period returns (net, as actually earned) up to but
    not including the decision date. It returns an exposure multiplier in [floor, 1].
    """

    def __init__(
        self,
        target_vol: float = 0.12,       # ~12% annualised, per Barroso & Santa-Clara
        lookback_periods: int = 6,      # ~6 months when rebalancing monthly
        periods_per_year: int = 12,
        min_exposure: float = 0.0,      # allow full de-risking to cash
        max_exposure: float = 1.0,      # NEVER lever up (family accounts)
    ) -> None:
        if not (0.0 <= min_exposure <= max_exposure):
            raise ValueError("need 0 <= min_exposure <= max_exposure")
        if target_vol <= 0:
            raise ValueError("target_vol must be positive")
        if lookback_periods < 2:
            raise ValueError("need at least 2 periods to estimate volatility")

        self.target_vol = target_vol
        self.lookback_periods = lookback_periods
        self.periods_per_year = periods_per_year
        self.min_exposure = min_exposure
        self.max_exposure = max_exposure

    def decide(self, strategy_returns: pd.Series) -> VolTargetDecision:
        """
        `strategy_returns` = the strategy's own realised period returns SO FAR,
        newest last. Uses only the trailing window -- no look-ahead.
        """
        clean = strategy_returns.dropna()

        # Not enough history yet: invest fully rather than invent a signal.
        if len(clean) < self.lookback_periods:
            return VolTargetDecision(
                exposure=self.max_exposure,
                realised_vol=None,
                target_vol=self.target_vol,
                reason="insufficient_history",
            )

        window = clean.iloc[-self.lookback_periods:]
        period_std = float(window.std(ddof=1))
        realised_vol = period_std * np.sqrt(self.periods_per_year)

        # A dead-calm window would divide by ~zero and demand infinite exposure;
        # the cap handles it, but guard explicitly for clarity.
        if realised_vol <= 1e-9:
            return VolTargetDecision(
                exposure=self.max_exposure,
                realised_vol=realised_vol,
                target_vol=self.target_vol,
                reason="near_zero_vol",
            )

        raw = self.target_vol / realised_vol
        exposure = float(np.clip(raw, self.min_exposure, self.max_exposure))

        if exposure >= self.max_exposure:
            reason = "calm_full_exposure"
        elif exposure <= self.min_exposure:
            reason = "extreme_vol_min_exposure"
        else:
            reason = "scaled_down"

        return VolTargetDecision(
            exposure=exposure,
            realised_vol=realised_vol,
            target_vol=self.target_vol,
            reason=reason,
        )


def apply_exposure(weights: pd.Series, exposure: float) -> pd.Series:
    """
    Scale portfolio weights by an exposure multiplier. The remainder is cash.

    exposure=0.6 on a fully-invested book means 60% in the same names at the same
    RELATIVE proportions, 40% in cash. We never change WHICH names we hold or their
    relative sizes -- only how much of the book is deployed. Stock selection and
    risk sizing stay cleanly separate.
    """
    if weights.empty:
        return weights
    return weights * float(exposure)
