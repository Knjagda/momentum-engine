"""
Fundamental data, with the one rule that makes it honest.

    A fundamental fact is usable only from the date it was FILED --
    never from the date of the period it describes.

Why this is the whole ballgame:

A company's fiscal year ends on 31 December. That balance sheet describes the world
on 31 December. But NOBODY COULD SEE IT until the 10-K was filed, typically 30-90
days later. A screen run on 2 January using December's numbers is reading a document
that does not exist yet.

With prices, look-ahead is a one-day sin. With fundamentals it is a THREE-MONTH one,
and it is the easiest way in the world to build a beautiful backtest that is pure
fiction.

AND THERE IS A SECOND TRAP, which is subtler and which we found the hard way:

EDGAR holds the SAME period many times over. Apple's FY2024 10-K reports FY2024,
FY2023 and FY2022. So FY2022's net income appears in filings from 2023, 2024 AND
2025 -- often with DIFFERENT VALUES, because companies restate.

If you take the latest version, you are using a number that was corrected after the
fact. The investor standing there in 2023 saw the ORIGINAL. So:

    For each (symbol, concept, period) we keep the EARLIEST filing.
    As-first-reported. What you would actually have known.

This module enforces both rules mechanically, so no strategy can break them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

# The columns of the long-form fact table.
FACT_COLUMNS = ["symbol", "concept", "period_end", "filed", "value", "form", "fiscal_year"]


@dataclass
class FundamentalData:
    """
    Every fundamental fact we know, in long form, with its filing date attached.

    One row = one number, from one company, for one period, as first reported.
    """

    facts: pd.DataFrame          # FACT_COLUMNS
    source: str = "unknown"

    def __post_init__(self) -> None:
        missing = set(FACT_COLUMNS) - set(self.facts.columns)
        if missing:
            raise ValueError(f"FundamentalData missing columns: {sorted(missing)}")

        self.facts = self.facts.copy()
        self.facts["period_end"] = pd.to_datetime(self.facts["period_end"])
        self.facts["filed"] = pd.to_datetime(self.facts["filed"])

        # AS-FIRST-REPORTED. If the same (symbol, concept, period) was filed more
        # than once -- comparatives, restatements -- keep the ORIGINAL. That is the
        # number a person standing there at the time would actually have seen.
        self.facts = (
            self.facts
            .sort_values("filed")
            .drop_duplicates(subset=["symbol", "concept", "period_end"], keep="first")
            .reset_index(drop=True)
        )

    # -- the guard ----------------------------------------------------------

    def as_of(
        self,
        when: date | datetime | str,
        concepts: list[str] | None = None,
        symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        THE GUARD. What we could ACTUALLY have known on `when`.

        Returns a frame indexed by symbol, one column per concept, holding the most
        recent figure that had genuinely been PUBLISHED before this date.

        Everything filed on or after `when` is invisible. It has not happened yet.
        """
        cutoff = pd.Timestamp(when)

        visible = self.facts[self.facts["filed"] < cutoff]      # strictly before

        if concepts:
            visible = visible[visible["concept"].isin(concepts)]
        if symbols:
            visible = visible[visible["symbol"].isin(symbols)]

        if visible.empty:
            return pd.DataFrame()

        # Of the facts we could see, take the most recent PERIOD for each concept.
        latest = (
            visible
            .sort_values(["period_end", "filed"])
            .drop_duplicates(subset=["symbol", "concept"], keep="last")
        )

        return latest.pivot(index="symbol", columns="concept", values="value")

    def staleness(self, when: date | datetime | str) -> pd.Series:
        """
        How OLD is the newest data we hold for each company, in days?

        Worth watching. A company that stopped filing 400 days ago is probably in
        trouble -- or has quietly delisted and we have not noticed.
        """
        cutoff = pd.Timestamp(when)
        visible = self.facts[self.facts["filed"] < cutoff]

        if visible.empty:
            return pd.Series(dtype=float)

        newest = visible.groupby("symbol")["filed"].max()
        return (cutoff - newest).dt.days

    # -- introspection ------------------------------------------------------

    @property
    def symbols(self) -> list[str]:
        return sorted(self.facts["symbol"].unique())

    @property
    def concepts(self) -> list[str]:
        return sorted(self.facts["concept"].unique())

    def coverage(self, concept: str) -> float:
        """Fraction of symbols for which we have this concept at all."""
        if not self.symbols:
            return 0.0
        have = self.facts.loc[self.facts["concept"] == concept, "symbol"].nunique()
        return have / len(self.symbols)

    def filing_lag(self) -> pd.Series:
        """
        Days between a period ending and that figure FIRST becoming public.

        Because we de-duplicate to as-first-reported in __post_init__, this is the
        TRUE lag -- not contaminated by the comparatives that appear in later filings.
        Expect roughly 30-90 days. If you see 400, something is wrong.
        """
        annual = self.facts[self.facts["form"].isin(["10-K", "20-F"])]
        if annual.empty:
            return pd.Series(dtype=float)
        return (annual["filed"] - annual["period_end"]).dt.days

    def __repr__(self) -> str:
        return (
            f"<FundamentalData {len(self.facts):,} facts, "
            f"{len(self.symbols)} symbols, {len(self.concepts)} concepts "
            f"from {self.source}>"
        )


class FundamentalAdapter:
    """
    Where fundamentals come from. Swap the vendor, keep the engine.

    Today: SEC EDGAR (free, official, 2010+).
    Tomorrow, if we pay: Sharadar (1990+, normalised, includes delisted).

    The engine must never know or care which. Same rule as Market-as-config.
    """

    name = "fundamental_adapter"

    def fetch(
        self,
        symbols: list[str],
        concepts: list[str] | None = None,
    ) -> FundamentalData:
        raise NotImplementedError
