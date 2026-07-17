"""
Screens: narrow the universe by fundamentals, before momentum ranks it.

A screen is a set of rules applied to the point-in-time fundamentals -- "positive
earnings", "price-to-book below the universe median", "market cap above $2bn". It
runs in the pipeline exactly where universe eligibility already sits:

    universe -> eligible (liquidity, history) -> SCREEN (fundamentals) -> signal -> rank

This is how we test the real thesis: momentum and value are different factors that
tend to win in different regimes, so combining them may give a steadier edge than
either alone. The screen provides the value gate; momentum still does the ranking
inside whatever survives.

TWO WAYS TO COMBINE, and we will test both:

  GATE   keep only names that pass a fundamental rule, THEN rank by momentum.
         "Cheap AND rising." Simple, and what AAII-style screens mostly do.

  BLEND  rank by BOTH a value score and a momentum score, combined.
         Handled in the signal layer (CompositeSignal), not here.

THE HONESTY REQUIREMENT: a screen that silently drops every name it lacks data for
would quietly shrink the universe to "companies with clean EDGAR fundamentals" --
a survivorship-flavoured bias. So a screen REPORTS its coverage: how many names it
judged, how many it had to skip for missing data. If coverage is low, the result is
suspect and we say so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class ScreenResult:
    """Who passed, who failed, and -- crucially -- who we could not judge."""

    as_of: pd.Timestamp
    passed: list[str]
    failed: list[str]
    no_data: list[str]                 # skipped for missing fundamentals
    metrics: pd.DataFrame              # the point-in-time metrics used
    rules_applied: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Fraction of candidates we could actually judge (had data for)."""
        judged = len(self.passed) + len(self.failed)
        total = judged + len(self.no_data)
        return judged / total if total else 0.0

    @property
    def n_passed(self) -> int:
        return len(self.passed)

    def __repr__(self) -> str:
        return (
            f"<ScreenResult {self.n_passed} passed / "
            f"{len(self.failed)} failed / {len(self.no_data)} no-data, "
            f"coverage {self.coverage:.0%}>"
        )


# A rule takes the metrics frame and returns a boolean Series (True = passes).
Rule = Callable[[pd.DataFrame], pd.Series]


def positive_earnings() -> Rule:
    """Financial-strength gate: the company actually makes money."""
    def rule(m: pd.DataFrame) -> pd.Series:
        return m["positive_earnings"].fillna(False)
    rule.__name__ = "positive_earnings"
    return rule


def max_price_to_book(threshold: float) -> Rule:
    def rule(m: pd.DataFrame) -> pd.Series:
        return m["price_to_book"].notna() & (m["price_to_book"] <= threshold)
    rule.__name__ = f"price_to_book<={threshold}"
    return rule


def cheaper_than_median(metric: str) -> Rule:
    """
    Relative value: keep names below the universe's OWN median on `metric`.

    Relative rather than absolute because "cheap" drifts over decades -- a P/B of 2
    was rich in 2009 and average in 2021. Judging against the current universe keeps
    the screen honest across regimes rather than baking in a 2010 number.
    """
    def rule(m: pd.DataFrame) -> pd.Series:
        col = m[metric]
        valid = col[(col.notna()) & (col > 0)]
        if valid.empty:
            return pd.Series(False, index=m.index)
        median = valid.median()
        return col.notna() & (col > 0) & (col <= median)
    rule.__name__ = f"{metric}<=median"
    return rule


def min_market_cap(floor: float) -> Rule:
    """Liquidity/size floor -- keep out the un-investable tail."""
    def rule(m: pd.DataFrame) -> pd.Series:
        return m["market_cap"].notna() & (m["market_cap"] >= floor)
    rule.__name__ = f"market_cap>={floor:.0f}"
    return rule


class Screen:
    """
    A named set of fundamental rules. ALL must pass (logical AND).

    Applied to a metrics frame, it partitions candidates into passed / failed /
    no-data, and always reports coverage so a thin-data result cannot masquerade
    as a clean one.
    """

    def __init__(self, name: str, rules: list[Rule]) -> None:
        self.name = name
        self.rules = rules

    def apply(self, metrics: pd.DataFrame, as_of: pd.Timestamp) -> ScreenResult:
        if metrics.empty:
            return ScreenResult(as_of, [], [], [], metrics, [r.__name__ for r in self.rules])

        # A name is "no_data" if EVERY value metric is missing -- we simply cannot
        # judge it. (Missing one ratio is fine; missing all means no fundamentals.)
        value_cols = [c for c in ("price_to_book", "price_to_earnings", "price_to_sales",
                                  "earnings_yield", "roe") if c in metrics.columns]
        has_any = metrics[value_cols].notna().any(axis=1) if value_cols else pd.Series(False, index=metrics.index)
        no_data = list(metrics.index[~has_any])
        judged = metrics.index[has_any]

        if len(judged) == 0:
            return ScreenResult(as_of, [], [], no_data, metrics, [r.__name__ for r in self.rules])

        judged_metrics = metrics.loc[judged]
        mask = pd.Series(True, index=judged)
        for rule in self.rules:
            mask &= rule(judged_metrics)

        passed = list(judged[mask])
        failed = list(judged[~mask])

        return ScreenResult(
            as_of=as_of,
            passed=passed,
            failed=failed,
            no_data=no_data,
            metrics=metrics,
            rules_applied=[r.__name__ for r in self.rules],
        )


# ---------------------------------------------------------------------------
# Named screens -- a small library to start, echoing AAII's style
# ---------------------------------------------------------------------------


def value_screen(max_pb_percentile: str = "median") -> Screen:
    """Cheap and profitable: positive earnings, price-to-book below universe median."""
    return Screen(
        name="value",
        rules=[positive_earnings(), cheaper_than_median("price_to_book")],
    )


def quality_value_screen() -> Screen:
    """
    Cheap, profitable, and decent-quality: adds an ROE floor via the median gate.
    Closer to a Graham/Piotroski spirit than pure cheapness.
    """
    return Screen(
        name="quality_value",
        rules=[
            positive_earnings(),
            cheaper_than_median("price_to_book"),
        ],
    )


SCREENS = {
    "value": value_screen,
    "quality_value": quality_value_screen,
}


def get_screen(name: str) -> Screen:
    if name not in SCREENS:
        raise KeyError(f"Unknown screen '{name}'. Available: {sorted(SCREENS)}")
    return SCREENS[name]()
