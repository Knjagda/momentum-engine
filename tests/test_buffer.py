"""
Tests for the no-trade buffer.

The behaviour under test is deliberately "lazy": a holding that slips a few ranks
is KEPT, not sold. That laziness is the point -- selling a stock because it moved
from rank 19 to rank 21 is not a decision, it is a twitch, and every twitch pays a
spread.
"""

import pandas as pd
import pytest

from engine.markets.market import load_market
from engine.portfolio import build_portfolio, select_with_buffer
from engine.signals.base import SignalResult

AS_OF = pd.Timestamp("2024-07-01")


def scores(**kw) -> SignalResult:
    return SignalResult(name="volar", as_of=AS_OF, scores=pd.Series(kw, dtype=float))


@pytest.fixture
def market():
    return load_market("us")


def test_a_slipping_holding_is_kept_not_sold():
    """
    We hold C. It has slipped to rank 4 -- outside the top 3, but still inside the
    hold band of 5. The naive rule sells it and buys D. The buffer keeps it.
    """
    s = scores(A=10.0, B=9.0, C=8.0, D=8.5, E=7.0)
    # ranks: A=1, B=2, D=3, C=4, E=5

    selected = select_with_buffer(s, top_n=3, current_symbols=["A", "B", "C"], exit_rank=5)

    assert "C" in selected, "sold a holding that was still good enough to keep"
    assert set(selected) == {"A", "B", "C"}
    assert "D" not in selected      # D is better, but there is no room and C is fine


def test_a_collapsing_holding_is_sold():
    """The buffer is lazy, not blind. Fall far enough and you are out."""
    s = scores(A=10.0, B=9.0, C=1.0, D=8.5, E=7.0)
    # C is now rank 5, outside the hold band of 4

    selected = select_with_buffer(s, top_n=3, current_symbols=["A", "B", "C"], exit_rank=4)

    assert "C" not in selected
    assert "D" in selected          # its slot goes to the best available name


def test_empty_slots_are_filled_with_the_best_names():
    s = scores(A=10.0, B=9.0, C=8.0, D=7.0)
    selected = select_with_buffer(s, top_n=3, current_symbols=[], exit_rank=5)

    assert selected == ["A", "B", "C"]


def test_new_entrants_must_clear_the_HIGHER_bar():
    """
    Entry is strict (rank <= top_n); holding is lenient (rank <= exit_rank).
    A stock at rank 4 cannot ENTER a top-3 portfolio -- it can only be KEPT.
    """
    s = scores(A=10.0, B=9.0, C=8.0, D=7.5, E=7.0)
    selected = select_with_buffer(s, top_n=3, current_symbols=[], exit_rank=5)

    assert "D" not in selected      # rank 4: good enough to hold, not to buy


def test_buffer_reduces_turnover_versus_naive_selection():
    """The whole point, measured."""
    s = scores(A=10.0, B=9.0, C=8.0, D=8.2, E=8.1, F=1.0)
    holdings = ["A", "B", "C"]

    naive = list(s.top(3).index)                                    # ranks 1,2,3
    buffered = select_with_buffer(s, 3, holdings, exit_rank=6)

    naive_trades = len(set(holdings) ^ set(naive))
    buffered_trades = len(set(holdings) ^ set(buffered))

    assert buffered_trades < naive_trades


def test_exit_rank_below_top_n_is_rejected():
    """A hold band narrower than the entry band is incoherent."""
    s = scores(A=1.0, B=2.0)
    with pytest.raises(ValueError, match="exit_rank"):
        select_with_buffer(s, top_n=20, current_symbols=[], exit_rank=10)


def test_buffer_flows_through_portfolio_construction(market):
    s = scores(A=10.0, B=9.0, C=8.0, D=8.5)
    preselected = select_with_buffer(s, top_n=3, current_symbols=["A", "B", "C"], exit_rank=5)

    pf = build_portfolio(s, market, top_n=3, preselected_symbols=preselected)

    assert set(pf.symbols) == {"A", "B", "C"}
    pf.validate()


def test_unscoreable_holdings_are_dropped():
    """A stock the signal can no longer score cannot be 'kept' on faith."""
    s = SignalResult(
        name="volar",
        as_of=AS_OF,
        scores=pd.Series({"A": 10.0, "B": 9.0, "C": float("nan"), "D": 8.0}),
    )
    selected = select_with_buffer(s, top_n=3, current_symbols=["A", "B", "C"], exit_rank=5)

    assert "C" not in selected
    assert "D" in selected
