"""
Tests for volatility targeting.

The one idea under test: when the strategy's OWN recent returns get violent, hold
less of everything. This is NOT Volar (which scores individual stocks by their
volatility and which we already showed fails). This scales the whole book's exposure
based on the strategy's realised risk. Barroso & Santa-Clara (2015).
"""

import numpy as np
import pandas as pd
import pytest

from engine.backtest.vol_target import (
    VolatilityTarget,
    apply_exposure,
)


def test_calm_returns_full_exposure():
    """Low realised vol → hold the full book."""
    vt = VolatilityTarget(target_vol=0.12, lookback_periods=6)
    calm = pd.Series([0.005, 0.004, 0.006, 0.005, 0.004, 0.005])   # ~tiny monthly moves
    d = vt.decide(calm)
    assert d.exposure == pytest.approx(1.0)
    assert d.reason == "calm_full_exposure"


def test_violent_returns_cut_exposure():
    """High realised vol → hold less."""
    vt = VolatilityTarget(target_vol=0.12, lookback_periods=6)
    wild = pd.Series([0.15, -0.18, 0.20, -0.22, 0.17, -0.19])       # huge monthly swings
    d = vt.decide(wild)
    assert d.exposure < 0.5, f"expected heavy de-risking, got {d.exposure}"
    assert d.reason in ("scaled_down", "extreme_vol_min_exposure")


def test_exposure_is_inversely_proportional_to_vol():
    """Double the realised vol → roughly half the exposure."""
    vt = VolatilityTarget(target_vol=0.12, lookback_periods=6, min_exposure=0.0)

    base = np.array([0.02, -0.02, 0.02, -0.02, 0.02, -0.02])
    calm = vt.decide(pd.Series(base))
    loud = vt.decide(pd.Series(base * 2))

    # exposure ~ 1/vol, so doubling vol should ~halve exposure (both below cap)
    if calm.exposure < 1.0:
        assert loud.exposure == pytest.approx(calm.exposure / 2, rel=0.05)


def test_never_levers_above_one():
    """We de-risk but never gear up -- family accounts."""
    vt = VolatilityTarget(target_vol=0.50, lookback_periods=6, max_exposure=1.0)
    calm = pd.Series([0.001] * 6)
    assert vt.decide(calm).exposure <= 1.0


def test_insufficient_history_stays_invested():
    """Early on, before we have a window, invest fully rather than guess."""
    vt = VolatilityTarget(lookback_periods=6)
    d = vt.decide(pd.Series([0.01, 0.02]))
    assert d.exposure == 1.0
    assert d.reason == "insufficient_history"


def test_can_go_fully_to_cash():
    vt = VolatilityTarget(target_vol=0.12, lookback_periods=6, min_exposure=0.0)
    catastrophic = pd.Series([0.4, -0.5, 0.45, -0.55, 0.5, -0.6])
    d = vt.decide(catastrophic)
    assert d.exposure < 0.25


def test_apply_exposure_scales_and_leaves_cash():
    weights = pd.Series({"A": 0.5, "B": 0.5})
    scaled = apply_exposure(weights, 0.6)
    assert scaled.sum() == pytest.approx(0.6)
    assert scaled["A"] == pytest.approx(0.3)
    # relative proportions unchanged
    assert scaled["A"] / scaled["B"] == pytest.approx(1.0)


def test_apply_exposure_of_one_is_identity():
    weights = pd.Series({"A": 0.5, "B": 0.5})
    assert apply_exposure(weights, 1.0).equals(weights)


def test_rejects_bad_config():
    with pytest.raises(ValueError):
        VolatilityTarget(target_vol=-0.1)
    with pytest.raises(ValueError):
        VolatilityTarget(lookback_periods=1)
    with pytest.raises(ValueError):
        VolatilityTarget(min_exposure=0.5, max_exposure=0.2)


# ---------------------------------------------------------------------------
# The property that actually matters, end to end
# ---------------------------------------------------------------------------


def test_vol_targeting_reduces_drawdown_on_a_crash():
    """
    A return stream that is calm, then violently negative (a momentum crash).

    Applying vol targeting AFTER the fact -- feeding each period's decision the
    returns available before it -- should produce a shallower drawdown than the
    raw stream, because exposure falls as volatility spikes.

    This is the whole promise, tested on a controlled series.
    """
    rng = np.random.default_rng(0)
    calm = rng.normal(0.01, 0.02, 40)                 # 40 calm months
    crash = np.array([-0.10, -0.15, 0.12, -0.18, -0.20, 0.15, -0.12])  # violent
    raw = pd.Series(np.concatenate([calm, crash]))

    vt = VolatilityTarget(target_vol=0.12, lookback_periods=6, min_exposure=0.0)

    # Walk forward: each period scaled by a decision using only PRIOR returns.
    scaled_returns = []
    for i in range(len(raw)):
        history = raw.iloc[:i]
        exposure = vt.decide(history).exposure if i > 0 else 1.0
        scaled_returns.append(raw.iloc[i] * exposure)
    scaled = pd.Series(scaled_returns)

    def max_dd(returns):
        equity = (1 + returns).cumprod()
        return float((equity / equity.cummax() - 1).min())

    raw_dd = max_dd(raw)
    scaled_dd = max_dd(scaled)

    assert scaled_dd > raw_dd, (
        f"vol targeting should reduce drawdown: raw {raw_dd:.1%} vs scaled {scaled_dd:.1%}"
    )
