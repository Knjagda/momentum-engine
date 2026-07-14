"""
The momentum signal family.

Four signals, all plugging into the same pipeline:

    momentum   raw trailing return, skipping the most recent month
    volar      that return divided by volatility  (Definedge's method)
    sharpe     risk-adjusted return of the PATH, not just the endpoints
    composite  a z-score blend of several signals

See STRATEGIES_AND_RESEARCH.md for the research behind each.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from engine.data.base import PriceData
from engine.signals.base import Signal, SignalResult


class MomentumSignal(Signal):
    """
    Cross-sectional price momentum. Buy what has been going up.

    Research: Jegadeesh & Titman (1993) -- the most replicated anomaly in finance.

    THE SKIP MONTH is the part people get wrong. The academic standard is "12-2"
    (or 12-1): measure the return from 12 months ago up to ONE MONTH ago, and
    ignore the most recent month entirely.

    Why deliberately throw away the freshest data? Because at short horizons
    stocks REVERSE rather than continue -- last month's biggest winner tends to
    give some back. Including that month injects reversal noise into a trend
    signal, and measurably weakens it. Skipping it is not an oversight; it is
    the whole point.

        skip_months = 1  ->  the "12-1" convention
        skip_months = 2  ->  the "12-2" convention (Calluzzo et al. 2025)
        skip_months = 0  ->  naive momentum. Available, but you have been warned.
    """

    name = "momentum"

    def __init__(self, lookback_months: int = 12, skip_months: int = 1, **kw) -> None:
        super().__init__(lookback_months=lookback_months, skip_months=skip_months, **kw)
        self.lookback_months = lookback_months
        self.skip_months = skip_months

    @property
    def required_history_days(self) -> int:
        # ~21 trading days per month, plus a small buffer for holidays.
        return int(self.lookback_months * 21 * 1.05) + 5

    def _window(self, as_of: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
        """The (start, end) of the measurement window. End is BEFORE as_of."""
        end = as_of - relativedelta(months=self.skip_months)
        start = as_of - relativedelta(months=self.lookback_months)
        return start, end

    def _score(self, history: PriceData, as_of: pd.Timestamp, symbols: list[str]) -> pd.Series:
        start, end = self._window(as_of)
        out: dict[str, float] = {}

        for symbol in symbols:
            if symbol not in history.close.columns:
                out[symbol] = np.nan
                continue

            series = history.close[symbol]
            p_start = self._price_asof(series, start)
            p_end = self._price_asof(series, end)

            if not (np.isfinite(p_start) and np.isfinite(p_end)) or p_start <= 0:
                out[symbol] = np.nan
                continue

            out[symbol] = (p_end / p_start) - 1.0

        return pd.Series(out, dtype=float)


class VolarSignal(MomentumSignal):
    """
    Volatility-adjusted momentum -- "Volar". Definedge's signature ranking method.

        score = momentum return / annualized volatility

    Two stocks both up 40%: one climbed steadily, the other lurched up and down
    and happened to end high. Raw momentum cannot tell them apart. Volar prefers
    the steady climber -- higher return PER UNIT OF RISK.

    This matters beyond elegance: momentum's ugliest feature is its crash risk,
    and volatile momentum names are the ones that crash hardest. Penalising
    volatility at the ranking stage is a first line of defence.

    Related research: Barroso & Santa-Clara (2015) scale the whole strategy by
    its trailing volatility (~6-month window, ~12% annualized target) and roughly
    DOUBLE the Sharpe ratio while nearly eliminating crashes. Volar applies the
    same intuition per-stock instead of per-portfolio.
    """

    name = "volar"

    def __init__(
        self,
        lookback_months: int = 12,
        skip_months: int = 1,
        vol_window_days: int = 126,     # ~6 months, per Barroso & Santa-Clara
        **kw,
    ) -> None:
        super().__init__(lookback_months=lookback_months, skip_months=skip_months, **kw)
        self.params["vol_window_days"] = vol_window_days
        self.vol_window_days = vol_window_days

    @property
    def required_history_days(self) -> int:
        return max(super().required_history_days, self.vol_window_days + 5)

    def _score(self, history: PriceData, as_of: pd.Timestamp, symbols: list[str]) -> pd.Series:
        momentum = super()._score(history, as_of, symbols)
        _, end = self._window(as_of)

        # Annualization comes from the market, not from a constant in this file.
        ann = history.market.trading_days_per_year

        out: dict[str, float] = {}
        for symbol in symbols:
            mom = momentum.get(symbol, np.nan)
            if not np.isfinite(mom) or symbol not in history.close.columns:
                out[symbol] = np.nan
                continue

            # Volatility measured over the same window that ENDS at the skip date,
            # so the signal and its risk adjustment see the same slice of time.
            window = history.close[symbol].loc[:end].tail(self.vol_window_days + 1)
            vol = self._annualized_vol(window.pct_change(), ann)

            out[symbol] = mom / vol if np.isfinite(vol) and vol > 0 else np.nan

        return pd.Series(out, dtype=float)


class SharpeSignal(VolarSignal):
    """
    Risk-adjusted momentum computed from the PATH rather than the endpoints.

    Volar takes a start-to-end return and divides by volatility. Sharpe instead
    annualizes the MEAN DAILY RETURN over the window and divides by volatility.

    The difference is subtle but real: two stocks can share the same start and
    end price via wildly different journeys. Volar sees identical numerators;
    Sharpe does not. Related to the "frog in the pan" finding -- momentum built
    from many small steady steps continues better than momentum built from a few
    violent jumps (Da, Gutierrez & Warachka, 2014).
    """

    name = "sharpe"

    def _score(self, history: PriceData, as_of: pd.Timestamp, symbols: list[str]) -> pd.Series:
        start, end = self._window(as_of)
        ann = history.market.trading_days_per_year

        out: dict[str, float] = {}
        for symbol in symbols:
            if symbol not in history.close.columns:
                out[symbol] = np.nan
                continue

            window = history.close[symbol].loc[start:end].dropna()
            if len(window) < 2:
                out[symbol] = np.nan
                continue

            daily = window.pct_change().dropna()
            vol = self._annualized_vol(daily, ann)
            if not np.isfinite(vol) or vol <= 0:
                out[symbol] = np.nan
                continue

            out[symbol] = (float(daily.mean()) * ann) / vol

        return pd.Series(out, dtype=float)


class CompositeSignal(Signal):
    """
    Blend several signals into one score, via z-scores.

    Why this exists NOW even though V0 ships with single signals: the research is
    clear that composites beat any single signal. Baltussen et al. (2025) combine
    price momentum with ten alternative momentum signals across 150 years and 46
    countries and find better returns AND better risk-adjusted performance than
    price momentum alone.

    Raw scores cannot simply be added -- a 0.4 return and a 1.8 volatility ratio
    are different units. Each signal is z-scored first (mean 0, std 1), then
    combined with weights. This is the machinery that lets value, quality, and
    low-volatility slot in later without touching the pipeline.

        CompositeSignal(signals=[MomentumSignal(), VolarSignal()], weights=[0.5, 0.5])
    """

    name = "composite"

    def __init__(self, signals: list[Signal], weights: list[float] | None = None, **kw) -> None:
        if not signals:
            raise ValueError("CompositeSignal needs at least one signal")

        if weights is None:
            weights = [1.0 / len(signals)] * len(signals)
        if len(weights) != len(signals):
            raise ValueError(
                f"Got {len(signals)} signals but {len(weights)} weights -- they must match"
            )

        total = sum(weights)
        if total <= 0:
            raise ValueError("Signal weights must sum to a positive number")

        super().__init__(
            components=[s.name for s in signals],
            weights=[w / total for w in weights],
            **kw,
        )
        self.signals = signals
        self.weights = [w / total for w in weights]

    @property
    def required_history_days(self) -> int:
        return max(s.required_history_days for s in self.signals)

    def _score(self, history: PriceData, as_of: pd.Timestamp, symbols: list[str]) -> pd.Series:
        blended = pd.Series(0.0, index=symbols, dtype=float)
        contributed = pd.Series(0.0, index=symbols, dtype=float)

        for signal, weight in zip(self.signals, self.weights):
            result = SignalResult(
                name=signal.name,
                as_of=as_of,
                scores=signal._score(history, as_of, symbols).reindex(symbols),
            )
            z = result.zscore().reindex(symbols)

            # A symbol only counts where the sub-signal actually scored it.
            mask = z.notna()
            blended[mask] += z[mask] * weight
            contributed[mask] += weight

        # If no sub-signal could score a symbol, it has no composite score.
        blended[contributed == 0] = np.nan

        # Re-scale for symbols where some sub-signals were missing, so a stock
        # is not penalised merely for having partial coverage.
        partial = (contributed > 0) & (contributed < 1)
        blended[partial] = blended[partial] / contributed[partial]

        return blended
