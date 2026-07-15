"""
Tests for share-class collapsing.

Alphabet is in the S&P 500 twice: GOOGL (voting) and GOOG (non-voting). They track
the same company. Holding both at 5% each is not a 5% position twice -- it is a 10%
bet on one company wearing two hats. Left alone, our top-20 equal-weight portfolio
was really a top-19 with a double weight on Alphabet.
"""

import pandas as pd
import pytest

from engine.markets.market import load_market
from engine.portfolio import build_portfolio
from engine.portfolio.construction import collapse_share_classes
from engine.signals.base import SignalResult

AS_OF = pd.Timestamp("2024-07-01")


def result(**scores) -> SignalResult:
    return SignalResult(name="momentum", as_of=AS_OF, scores=pd.Series(scores, dtype=float))


def test_secondary_class_dropped_when_primary_present():
    """GOOG drops out; GOOGL stays and absorbs the position."""
    out = collapse_share_classes(["AAPL", "GOOGL", "MSFT", "GOOG", "NVDA"])
    assert out == ["AAPL", "GOOGL", "MSFT", "NVDA"]
    assert "GOOG" not in out


def test_order_is_preserved():
    out = collapse_share_classes(["GOOG", "GOOGL", "AAPL"])
    # GOOG appears first but is the secondary; it is dropped, GOOGL keeps its slot.
    assert out == ["GOOGL", "AAPL"]


def test_secondary_kept_if_primary_absent():
    """
    If only GOOG made the cut and GOOGL did not, we still hold GOOG. Refusing to
    own a company because we happened to rank its B-shares would be worse than the
    problem we are solving.
    """
    out = collapse_share_classes(["AAPL", "GOOG", "MSFT"])
    assert out == ["AAPL", "GOOG", "MSFT"]


def test_multiple_pairs():
    out = collapse_share_classes(["FOXA", "FOX", "NWSA", "NWS", "AAPL"])
    assert out == ["FOXA", "NWSA", "AAPL"]


def test_no_pairs_is_identity():
    names = ["AAPL", "MSFT", "NVDA", "AMZN"]
    assert collapse_share_classes(names) == names


def test_portfolio_does_not_hold_both_alphabet_classes():
    """End to end: both classes rank highly, but the portfolio holds one."""
    market = load_market("us")
    scores = result(GOOGL=10.0, GOOG=9.9, AAPL=9.0, MSFT=8.0, NVDA=7.0)

    pf = build_portfolio(scores, market, top_n=3, weighting="equal")

    assert "GOOG" not in pf.symbols
    assert "GOOGL" in pf.symbols
    # And we still hold a full 3 names -- GOOG's slot went to the next best (NVDA),
    # not left empty.
    assert pf.n_positions == 3
    assert set(pf.symbols) == {"GOOGL", "AAPL", "MSFT"}
    pf.validate()


def test_alphabet_gets_single_weight_not_double():
    """The actual bug: Alphabet must not carry 2x the intended weight."""
    market = load_market("us")
    scores = result(GOOGL=10.0, GOOG=9.9, AAPL=9.0, MSFT=8.0)

    pf = build_portfolio(scores, market, top_n=3, weighting="equal")
    weights = dict(zip(pf.symbols, [p.weight for p in pf.positions]))

    # One Alphabet position at ~1/3, not two at ~1/3 each.
    alphabet_weight = weights.get("GOOGL", 0) + weights.get("GOOG", 0)
    assert alphabet_weight == pytest.approx(1 / 3, abs=0.02)
