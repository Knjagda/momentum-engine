"""
Tests for portfolio construction.

The one that matters most is test_position_cap_is_iterative_not_single_pass --
a naive cap silently leaves violations behind, and you would never notice until
a single name was 22% of a portfolio you believed was capped at 10%.
"""

import numpy as np
import pandas as pd
import pytest

from engine.data.base import PriceData
from engine.markets.market import load_market
from engine.portfolio import (
    Portfolio,
    Position,
    build_portfolio,
    cap_position_weights,
    cap_sector_weights,
    cash_portfolio,
    equal_weight,
    inverse_vol_weight,
    registered_weightings,
)
from engine.signals.base import SignalResult
from engine.universe.universe import Member, Membership

AS_OF = pd.Timestamp("2024-07-01")


@pytest.fixture
def market():
    return load_market("us")


@pytest.fixture
def membership(market):
    return Membership(
        market=market,
        universe_key="sp500",
        members=[
            Member("AAA", "Alpha", "Tech"),
            Member("BBB", "Bravo", "Tech"),
            Member("CCC", "Charlie", "Tech"),
            Member("DDD", "Delta", "Energy"),
            Member("EEE", "Echo", "Health"),
        ],
        survivorship_bias=True,
        disclaimer="⚠️ test",
    )


@pytest.fixture
def scores():
    return SignalResult(
        name="volar",
        as_of=AS_OF,
        scores=pd.Series({"AAA": 5.0, "BBB": 4.0, "CCC": 3.0, "DDD": 2.0, "EEE": 1.0}),
        params={"lookback_months": 12},
    )


# ---------------------------------------------------------------------------
# Selection and weighting
# ---------------------------------------------------------------------------


def test_holds_the_top_n_by_score(scores, market, membership):
    pf = build_portfolio(scores, market, top_n=3, membership=membership)

    assert pf.n_positions == 3
    assert set(pf.symbols) == {"AAA", "BBB", "CCC"}
    assert "EEE" not in pf.symbols


def test_equal_weight_splits_evenly(scores, market, membership):
    pf = build_portfolio(scores, market, top_n=4, weighting="equal", membership=membership)

    for p in pf.positions:
        assert p.weight == pytest.approx(0.25)
    assert pf.invested_weight == pytest.approx(1.0)


def test_weights_always_sum_to_one(scores, market, membership):
    for n in (1, 2, 3, 5):
        pf = build_portfolio(scores, market, top_n=n, membership=membership)
        pf.validate()      # raises if weights + cash != 1


def test_asking_for_more_names_than_exist_holds_what_is_available(scores, market, membership):
    """
    Better an honest portfolio of 5 than a padded one of 20 containing guesses.
    """
    pf = build_portfolio(scores, market, top_n=20, membership=membership)

    assert pf.n_positions == 5
    assert pf.metadata["top_n_requested"] == 20
    assert pf.metadata["top_n_held"] == 5
    pf.validate()


def test_positions_carry_rank_score_and_sector(scores, market, membership):
    pf = build_portfolio(scores, market, top_n=3, membership=membership)
    top = pf.positions[0]

    assert top.symbol == "AAA"
    assert top.rank == 1
    assert top.score == 5.0
    assert top.sector == "Tech"


# ---------------------------------------------------------------------------
# THE CAPPING TESTS
# ---------------------------------------------------------------------------


def test_position_cap_is_iterative_not_single_pass():
    """
    Start: one name at 50%, four at 12.5%. Cap = 20%.

    A NAIVE cap trims the 50% to 20% and dumps the 30% excess onto the others,
    pushing them to ~20%+ -- and stops, leaving fresh violations behind. The cap
    must iterate until nothing breaches.
    """
    weights = pd.Series({"BIG": 0.50, "A": 0.125, "B": 0.125, "C": 0.125, "D": 0.125})

    capped = cap_position_weights(weights, max_weight=0.20)

    assert capped.max() <= 0.20 + 1e-9, "cap left a violation behind"
    assert capped.sum() == pytest.approx(1.0)
    assert capped["BIG"] == pytest.approx(0.20)


def test_position_cap_leaves_compliant_weights_alone():
    weights = pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
    capped = cap_position_weights(weights, max_weight=0.50)

    pd.testing.assert_series_equal(capped, weights)


def test_impossible_position_cap_fails_loudly():
    """5 names cannot be capped at 10% each -- that is only 50% of a portfolio."""
    weights = equal_weight(["A", "B", "C", "D", "E"])

    with pytest.raises(ValueError, match="Cannot cap"):
        cap_position_weights(weights, max_weight=0.10)


def test_sector_cap_breaks_up_a_concentrated_bet():
    """
    Momentum's occupational hazard: the Top-N quietly becomes one big theme bet.

    Here 3 of 5 names are Tech (60%). A 40% sector cap must cut Tech down and
    push the difference into the others.
    """
    weights = equal_weight(["AAA", "BBB", "CCC", "DDD", "EEE"])
    sectors = {
        "AAA": "Tech", "BBB": "Tech", "CCC": "Tech",
        "DDD": "Energy", "EEE": "Health",
    }

    capped = cap_sector_weights(weights, sectors, max_sector_weight=0.40)
    by_sector = capped.groupby(pd.Series(sectors)).sum()

    assert by_sector["Tech"] <= 0.40 + 1e-9
    assert capped.sum() == pytest.approx(1.0)
    assert by_sector["Energy"] > 0.20      # the freed weight went somewhere


def test_sector_cap_applied_through_construction(scores, market, membership):
    """3 of the 5 fixture names are Tech. Cap it at 40% end-to-end."""
    pf = build_portfolio(
        scores, market, top_n=5, membership=membership, max_sector_weight=0.40
    )

    assert pf.sector_weights()["Tech"] <= 0.40 + 1e-6
    pf.validate()


def test_both_caps_together(scores, market, membership):
    pf = build_portfolio(
        scores,
        market,
        top_n=5,
        membership=membership,
        max_position_weight=0.30,
        max_sector_weight=0.50,
    )

    assert pf.weights.max() <= 0.30 + 1e-6
    pf.validate()


# ---------------------------------------------------------------------------
# Inverse-vol weighting
# ---------------------------------------------------------------------------


def test_inverse_vol_gives_the_calm_stock_more_weight(market):
    dates = pd.bdate_range("2023-01-02", "2024-06-28")
    n = len(dates)
    rng = np.random.default_rng(3)

    close = pd.DataFrame(
        {
            "CALM": 100 * np.cumprod(1 + rng.normal(0.0004, 0.005, n)),
            "WILD": 100 * np.cumprod(1 + rng.normal(0.0004, 0.040, n)),
        },
        index=dates,
    )
    volume = pd.DataFrame({c: np.full(n, 1e6) for c in close.columns}, index=dates)
    prices = PriceData(market=market, close=close, volume=volume)

    w = inverse_vol_weight(["CALM", "WILD"], prices=prices, as_of=AS_OF)

    assert w["CALM"] > w["WILD"]
    assert w.sum() == pytest.approx(1.0)


def test_weighting_schemes_are_registered():
    assert "equal" in registered_weightings()
    assert "inverse_vol" in registered_weightings()


# ---------------------------------------------------------------------------
# Portfolio object
# ---------------------------------------------------------------------------


def test_cash_portfolio_is_valid():
    """Being 100% in cash is a decision, not an error. The backtest must hold it."""
    pf = cash_portfolio("US", "USD", AS_OF, reason="trend_filter_off")

    pf.validate()
    assert pf.cash_weight == 1.0
    assert pf.n_positions == 0
    assert pf.metadata["reason"] == "trend_filter_off"


def test_validate_rejects_weights_that_do_not_sum_to_one():
    bad = Portfolio(
        market_id="US",
        currency="USD",
        as_of=AS_OF,
        positions=[Position("AAA", 0.5), Position("BBB", 0.2)],
        cash_weight=0.0,
    )
    with pytest.raises(ValueError, match="Weights sum to"):
        bad.validate()


def test_validate_rejects_negative_weights():
    bad = Portfolio(
        market_id="US",
        currency="USD",
        as_of=AS_OF,
        positions=[Position("AAA", 1.2), Position("BBB", -0.2)],
        cash_weight=0.0,
    )
    with pytest.raises(ValueError, match="Negative weight"):
        bad.validate()


def test_sector_weights_expose_concentration(scores, market, membership):
    pf = build_portfolio(scores, market, top_n=5, membership=membership)
    sw = pf.sector_weights()

    assert sw["Tech"] == pytest.approx(0.6)     # 3 of 5 names
    assert sw.sum() == pytest.approx(1.0)


def test_portfolio_records_how_it_was_built(scores, market, membership):
    """Auditability: a portfolio must be able to explain itself."""
    pf = build_portfolio(scores, market, top_n=3, weighting="equal", membership=membership)

    assert pf.metadata["signal"] == "volar"
    assert pf.metadata["weighting"] == "equal"
    assert pf.currency == "USD"
    assert pf.market_id == "US"
