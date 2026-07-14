"""
Tests for the market config loader.

The last test in this file is the important one: it MECHANICALLY ENFORCES
the "no hard-coded market wiring" rule from SPEC.md §2. If someone (including
a future me) types "NYSE" into engine logic, the test suite goes red.
"""

import ast
from pathlib import Path

import pytest

from engine.markets.market import (
    CostModel,
    Market,
    available_markets,
    load_market,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = REPO_ROOT / "engine"


# ---------------------------------------------------------------------------
# Both markets load
# ---------------------------------------------------------------------------


def test_both_markets_are_available():
    markets = available_markets()
    assert "us" in markets
    assert "india" in markets


@pytest.mark.parametrize("key", ["us", "india"])
def test_market_loads(key):
    market = load_market(key)
    assert isinstance(market, Market)
    assert market.currency
    assert market.calendar
    assert market.benchmark.ticker
    assert market.universes


def test_unknown_market_fails_loudly():
    with pytest.raises(FileNotFoundError):
        load_market("atlantis")


# ---------------------------------------------------------------------------
# The two markets genuinely differ (this is the whole point)
# ---------------------------------------------------------------------------


def test_markets_differ_on_every_axis_that_matters():
    us = load_market("us")
    india = load_market("india")

    assert us.currency != india.currency
    assert us.calendar != india.calendar
    assert us.benchmark.ticker != india.benchmark.ticker
    assert us.ticker_suffix != india.ticker_suffix


def test_ticker_convention_is_market_driven():
    us = load_market("us")
    india = load_market("india")

    # US needs no suffix; India needs the exchange suffix.
    assert us.resolve_ticker("aapl") == "AAPL"
    assert india.resolve_ticker("reliance") == "RELIANCE.NS"

    # Applying twice must not double up.
    assert india.resolve_ticker(india.resolve_ticker("TCS")) == "TCS.NS"

    # And we can get back to the display symbol.
    assert india.strip_ticker("TCS.NS") == "TCS"


def test_india_is_more_expensive_to_trade_than_the_us():
    """
    Indian momentum research shows ~32% MONTHLY turnover. Transaction taxes
    (STT, stamp duty, GST) make honest cost modelling essential -- a strategy
    that looks great gross can bleed out after costs.
    """
    us = load_market("us")
    india = load_market("india")

    assert india.costs.round_trip_bps() > us.costs.round_trip_bps()

    # STT is a sell-side tax: selling must cost more than buying in India.
    assert india.costs.sell_cost_bps() > india.costs.buy_cost_bps()


def test_money_is_never_assumed_to_be_dollars():
    us = load_market("us")
    india = load_market("india")

    assert us.format_money(1000) == "$1,000.00"
    assert india.format_money(1000) == "₹1,000.00"


# ---------------------------------------------------------------------------
# Universes
# ---------------------------------------------------------------------------


def test_universe_lookup_and_disclaimer():
    us = load_market("us")
    sp500 = us.get_universe("sp500")

    assert sp500.survivorship_bias is True
    assert sp500.disclaimer is not None
    assert "survivorship bias" in sp500.disclaimer.lower()


def test_unknown_universe_fails_loudly():
    us = load_market("us")
    with pytest.raises(KeyError):
        us.get_universe("nifty50")  # exists in India, not in the US


# ---------------------------------------------------------------------------
# Cost maths
# ---------------------------------------------------------------------------


def test_gst_applies_to_charges_not_to_trade_value():
    costs = CostModel(
        commission_bps=10.0,
        exchange_txn_bps=0.0,
        slippage_bps=0.0,
        gst_on_charges_pct=0.18,
    )
    # 10 bps commission + 18% GST on that commission = 11.8 bps
    assert costs.buy_cost_bps() == pytest.approx(11.8)


def test_every_trade_pays_something():
    """SPEC.md §4.3 -- no cost-free simulated fills. Ever."""
    for key in available_markets():
        market = load_market(key)
        assert market.costs.buy_cost_bps() > 0, f"{key} has free buys"
        assert market.costs.sell_cost_bps() > 0, f"{key} has free sells"


# ---------------------------------------------------------------------------
# THE CONTRACT TEST -- no hard-coded market wiring anywhere in engine/
# ---------------------------------------------------------------------------

# Exact market-specific values that must never be baked into engine logic.
FORBIDDEN_EXACT = {"USD", "INR", "NYSE", "NSE", ".NS", ".BO", "^GSPC", "^NSEI"}

# Substrings that give away a hard-coded market.
FORBIDDEN_SUBSTRINGS = ["S&P 500", "Nifty", "Sensex", "Nasdaq"]


def _docstring_node_ids(tree: ast.AST) -> set[int]:
    """Identify docstrings so we can exclude them -- documentation is not logic."""
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr):
                value = body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    ids.add(id(value))
    return ids


def test_engine_contains_no_hard_coded_market_wiring():
    """
    SPEC.md §2: a Market is a config object. The engine must never assume a country.

    This inspects real string literals via the AST -- comments and docstrings are
    exempt, because explaining the rule is not breaking it.

    If this test fails, the fix is NOT to edit this test. It is to move the offending
    value into config/markets/*.yaml and read it from the Market object.
    """
    offences = []

    for py_file in ENGINE_DIR.rglob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))
        skip = _docstring_node_ids(tree)
        rel = py_file.relative_to(REPO_ROOT)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if id(node) in skip:
                continue

            literal = node.value
            if literal in FORBIDDEN_EXACT:
                offences.append(f"{rel}:{node.lineno} hard-codes {literal!r}")
            for banned in FORBIDDEN_SUBSTRINGS:
                if banned in literal:
                    offences.append(f"{rel}:{node.lineno} hard-codes {banned!r}")

    assert not offences, (
        "Hard-coded market wiring found in engine/ (violates SPEC.md §2):\n  "
        + "\n  ".join(offences)
        + "\n\nMove these into config/markets/*.yaml and read them from the Market object."
    )
