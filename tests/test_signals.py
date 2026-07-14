"""
Tests for the signal layer.

Rather than checking arithmetic on random data, these build SYNTHETIC STOCKS
whose behaviour is known by construction, then assert the signal reacts the way
the theory says it should. If a signal is broken, these say WHY.

The cast:
    STEADY    rises smoothly all year          -> should score well everywhere
    JUMPY     same total return, wild path     -> volar/sharpe must penalise it
    FADER     rose all year, CRASHED last month-> the skip-month test
    LOSER     falls all year                   -> should score badly
    NEWBIE    listed three months ago          -> unscoreable, must be NaN
"""

import numpy as np
import pandas as pd
import pytest

from engine.data.base import PriceData
from engine.markets.market import load_market
from engine.signals import (
    CompositeSignal,
    MomentumSignal,
    SharpeSignal,
    VolarSignal,
    get_signal,
    registered_signals,
)

DATES = pd.bdate_range("2022-01-03", "2024-06-28")
AS_OF = pd.Timestamp("2024-07-01")

# The 12-2 measurement window implied by AS_OF (12 months back, skipping 1).
WIN_START = pd.Timestamp("2023-07-01")
WIN_END = pd.Timestamp("2024-06-01")


def _path(vol: float, window_return: float, seed: int) -> np.ndarray:
    """
    A random price path with a given daily volatility, whose return OVER THE
    MEASUREMENT WINDOW is exactly `window_return`.

    Controlling the window return precisely is what lets us isolate one variable
    at a time: two stocks with a known return gap but very different volatility.
    """
    rng = np.random.default_rng(seed)
    n = len(DATES)

    series = pd.Series(100 * np.cumprod(1 + rng.normal(0, vol, n)), index=DATES)

    i0 = series.index.get_indexer([WIN_START], method="ffill")[0]
    i1 = series.index.get_indexer([WIN_END], method="ffill")[0]

    # Solve for the drift that forces the window return to the target.
    realized = series.iloc[i1] / series.iloc[i0]
    g = np.log((1 + window_return) / realized) / (i1 - i0)

    return (series * np.exp(g * np.arange(n))).values


def _steady(n: int, total: float) -> np.ndarray:
    """Low-volatility path with a controlled window return."""
    return _path(vol=0.008, window_return=total, seed=1)


def _jumpy(n: int, total: float, seed: int = 2) -> np.ndarray:
    """High-volatility path with a controlled window return."""
    return _path(vol=0.040, window_return=total, seed=seed)


@pytest.fixture
def market():
    return load_market("us")


@pytest.fixture
def prices(market) -> PriceData:
    n = len(DATES)

    close = pd.DataFrame(
        {
            "STEADY": _steady(n, 0.50),   # calm, +50% over the window
            "JUMPY": _jumpy(n, 0.70),     # wild, +70% -- a BETTER raw return
            "FADER": _steady(n, 0.80),
            "LOSER": _steady(n, -0.30),
            "NEWBIE": _steady(n, 0.50),
        },
        index=DATES,
    )

    # FADER climbed all year, then collapsed 35% in the final month.
    last_month = close.index >= pd.Timestamp("2024-06-01")
    crash = np.linspace(1.0, 0.65, last_month.sum())
    close.loc[last_month, "FADER"] = close.loc[last_month, "FADER"].values * crash

    # NEWBIE only listed in April 2024.
    close.loc[close.index < pd.Timestamp("2024-04-01"), "NEWBIE"] = np.nan

    volume = pd.DataFrame(
        {c: np.full(n, 2_000_000) for c in close.columns}, index=DATES
    )
    return PriceData(market=market, close=close, volume=volume)


# ---------------------------------------------------------------------------
# THE SKIP MONTH -- the thing everyone gets wrong
# ---------------------------------------------------------------------------


def test_skip_month_ignores_the_recent_crash(prices):
    """
    FADER rose 80% over the year, then crashed 35% in the final month.

    A 12-month momentum signal WITH a skip month should still rank it highly --
    it deliberately does not look at the last month, because short-horizon moves
    are reversals, not trend.
    """
    with_skip = MomentumSignal(lookback_months=12, skip_months=1).compute(prices, AS_OF)
    scores = with_skip.valid

    assert scores["FADER"] > scores["STEADY"], "skip-month signal wrongly saw the crash"
    assert scores["FADER"] > 0


def test_without_skip_month_the_crash_is_visible(prices):
    """
    The control experiment. With skip_months=0 the signal DOES see the crash,
    and FADER's score collapses.

    Together with the test above, this proves the skip month is actually doing
    something -- it is not decorative.
    """
    no_skip = MomentumSignal(lookback_months=12, skip_months=0).compute(prices, AS_OF)
    with_skip = MomentumSignal(lookback_months=12, skip_months=1).compute(prices, AS_OF)

    assert no_skip.valid["FADER"] < with_skip.valid["FADER"]


# ---------------------------------------------------------------------------
# VOLAR -- volatility must be punished
# ---------------------------------------------------------------------------


def test_volar_flips_the_ranking_that_raw_momentum_gets_wrong(prices):
    """
    THE CENTRAL TEST FOR VOLAR.

    JUMPY earned MORE than STEADY over the window (+70% vs +50%) -- but it did so
    via a violent, high-volatility path. STEADY climbed calmly.

        raw momentum  -> prefers JUMPY   (it only sees the bigger number)
        volar         -> prefers STEADY  (return PER UNIT OF RISK)

    The flip is the whole point of Definedge's method. Momentum's ugliest feature
    is its crash risk, and the high-volatility winners are the ones that crash
    hardest -- so penalising volatility at the ranking stage is a first line of
    defence, not a cosmetic tweak.
    """
    momentum = MomentumSignal().compute(prices, AS_OF).valid
    volar = VolarSignal().compute(prices, AS_OF).valid

    # Raw momentum is seduced by the bigger return.
    assert momentum["JUMPY"] > momentum["STEADY"]

    # Volar is not.
    assert volar["STEADY"] > volar["JUMPY"]


def test_sharpe_also_flips_the_ranking(prices):
    """Sharpe reaches the same conclusion as Volar by a slightly different route."""
    sharpe = SharpeSignal().compute(prices, AS_OF).valid
    assert sharpe["STEADY"] > sharpe["JUMPY"]


def test_losers_score_below_winners(prices):
    for signal in (MomentumSignal(), VolarSignal(), SharpeSignal()):
        scores = signal.compute(prices, AS_OF).valid
        assert scores["LOSER"] < scores["STEADY"], f"{signal.name} failed to punish LOSER"


# ---------------------------------------------------------------------------
# NO LOOK-AHEAD
# ---------------------------------------------------------------------------


def test_signal_cannot_see_the_future(prices, market):
    """
    Corrupt every price AFTER the decision date. Scores must not budge.

    If this fails, the signal is reading the future -- and every backtest built
    on it is fiction. (SPEC.md §4.1)
    """
    before = VolarSignal().compute(prices, pd.Timestamp("2024-01-02")).valid

    tampered = prices.close.copy()
    future = tampered.index >= pd.Timestamp("2024-01-02")
    tampered.loc[future] = tampered.loc[future] * 100     # absurd future prices

    after = VolarSignal().compute(
        PriceData(market=market, close=tampered, volume=prices.volume),
        pd.Timestamp("2024-01-02"),
    ).valid

    pd.testing.assert_series_equal(before, after)


def test_unscoreable_symbols_are_nan_not_zero(prices):
    """
    NEWBIE listed 3 months ago and cannot have a 12-month score.

    It must be NaN -- an admission of ignorance. A zero would be an OPINION,
    and would quietly place it mid-pack in the rankings.
    """
    result = MomentumSignal(lookback_months=12).compute(prices, AS_OF)

    assert pd.isna(result.scores["NEWBIE"])
    assert "NEWBIE" not in result.valid.index
    assert "NEWBIE" not in result.rank().index


# ---------------------------------------------------------------------------
# SignalResult behaviour
# ---------------------------------------------------------------------------


def test_top_n_returns_best_first(prices):
    result = VolarSignal().compute(prices, AS_OF)
    top = result.top(2)

    assert len(top) == 2
    assert top.iloc[0] >= top.iloc[1]
    assert "LOSER" not in top.index


def test_rank_one_is_the_best(prices):
    result = MomentumSignal().compute(prices, AS_OF)
    ranks = result.rank()
    best = result.valid.idxmax()

    assert ranks[best] == 1


def test_zscore_standardizes(prices):
    z = VolarSignal().compute(prices, AS_OF).zscore()
    assert abs(z.mean()) < 1e-9
    assert abs(z.std(ddof=0) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Composite blending
# ---------------------------------------------------------------------------


def test_composite_blends_two_signals(prices):
    composite = CompositeSignal(
        signals=[MomentumSignal(), VolarSignal()],
        weights=[0.5, 0.5],
    )
    result = composite.compute(prices, AS_OF)

    assert len(result.valid) >= 4
    assert result.valid["STEADY"] > result.valid["LOSER"]


def test_composite_weights_are_normalized():
    composite = CompositeSignal(
        signals=[MomentumSignal(), VolarSignal()],
        weights=[3.0, 1.0],
    )
    assert composite.weights == pytest.approx([0.75, 0.25])


def test_composite_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        CompositeSignal(signals=[MomentumSignal(), VolarSignal()], weights=[1.0])


def test_composite_requires_the_longest_history():
    composite = CompositeSignal(
        signals=[MomentumSignal(lookback_months=3), MomentumSignal(lookback_months=12)]
    )
    assert composite.required_history_days == MomentumSignal(
        lookback_months=12
    ).required_history_days


# ---------------------------------------------------------------------------
# Registry + market-driven annualization
# ---------------------------------------------------------------------------


def test_all_signals_are_registered():
    for name in ("momentum", "volar", "sharpe", "composite"):
        assert name in registered_signals()


def test_get_signal_builds_from_config_style_params():
    signal = get_signal("volar", lookback_months=6, skip_months=1, vol_window_days=63)
    assert isinstance(signal, VolarSignal)
    assert signal.lookback_months == 6
    assert signal.vol_window_days == 63


def test_unknown_signal_fails_loudly():
    with pytest.raises(KeyError):
        get_signal("crystal_ball")


def test_annualization_comes_from_the_market_not_a_constant():
    """
    The US annualizes by 252 trading days, India by 250. Same prices, different
    markets -> different volatility -> different Volar score.

    If these came out identical, someone hard-coded 252 in a signal.
    """
    us, india = load_market("us"), load_market("india")
    n = len(DATES)

    close = pd.DataFrame({"X": _jumpy(n, 0.5)}, index=DATES)
    volume = pd.DataFrame({"X": np.full(n, 1_000_000)}, index=DATES)

    us_score = VolarSignal().compute(
        PriceData(market=us, close=close, volume=volume), AS_OF
    ).valid["X"]
    in_score = VolarSignal().compute(
        PriceData(market=india, close=close, volume=volume), AS_OF
    ).valid["X"]

    assert us_score != in_score
    assert us.trading_days_per_year != india.trading_days_per_year


def test_signals_declare_their_history_needs():
    """The universe filter relies on this to avoid handing a 3-month stock to a 12-month signal."""
    assert MomentumSignal(lookback_months=12).required_history_days > 240
    assert MomentumSignal(lookback_months=3).required_history_days < 100
    assert VolarSignal(lookback_months=3, vol_window_days=200).required_history_days > 200
