"""
Fundamental metrics: value and quality ratios, computed HONESTLY.

A ratio like price-to-book needs two things measured at the same instant: a PRICE
(known every day) and BOOK EQUITY (known only from the last filing that had been
PUBLISHED by that date). Getting the timing wrong here is the classic fundamental
backtest lie -- using December's book value on 2 January, months before the 10-K
that reported it existed.

So every metric here takes an `as_of` date and pulls:
  - the price AS OF that date              (PriceData.up_to)
  - the fundamentals PUBLISHED before it   (FundamentalData.as_of, filed-date gated)

and refuses to mix a today price with a not-yet-filed balance sheet.

THE TRAP WE KNOW IS COMING: value screens buy cheap stocks. In early 2023, the
cheapest financials by price-to-book were SVB and First Republic -- weeks before
they went to zero. Our price data cannot even hold them (yfinance has no post-
collapse prices), so a value backtest will never take that loss. This module cannot
fix that -- only better data can -- but the screen layer above it says so loudly.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

from engine.data.base import PriceData
from engine.data.fundamentals import FundamentalData


def _price_asof(prices: PriceData, symbols: list[str], as_of: pd.Timestamp) -> pd.Series:
    """Last close strictly before as_of, per symbol. Uses the anti-look-ahead gate."""
    history = prices.up_to(as_of)          # SPEC §4.1 -- nothing at/after as_of
    if history.close.empty:
        return pd.Series(dtype=float)
    last = history.close.ffill().iloc[-1]
    return last.reindex(symbols)


def compute_fundamental_metrics(
    prices: PriceData,
    fundamentals: FundamentalData,
    symbols: list[str],
    as_of: date | datetime | str,
) -> pd.DataFrame:
    """
    A point-in-time snapshot of value/quality metrics for each symbol.

    Returns a DataFrame indexed by symbol with columns:
        price_to_book, price_to_earnings, price_to_sales,
        earnings_yield, roe, positive_earnings, market_cap

    Any metric we cannot compute (missing fundamentals, non-positive denominator)
    is NaN -- never guessed. A screen treats NaN as "does not qualify", which is the
    safe direction: we would rather miss a name than buy one on invented data.
    """
    cutoff = pd.Timestamp(as_of)

    price = _price_asof(prices, symbols, cutoff)

    # Fundamentals PUBLISHED before as_of (filed-date gated inside as_of()).
    facts = fundamentals.as_of(
        cutoff,
        concepts=["equity", "net_income", "revenue", "shares", "assets"],
        symbols=symbols,
    )

    out = pd.DataFrame(index=symbols)
    out["price"] = price

    if facts.empty:
        # No fundamentals visible yet -- everything NaN, nothing qualifies.
        for col in ("price_to_book", "price_to_earnings", "price_to_sales",
                    "earnings_yield", "roe", "market_cap"):
            out[col] = np.nan
        out["positive_earnings"] = False
        return out

    for col in ("equity", "net_income", "revenue", "shares", "assets"):
        out[col] = facts[col].reindex(symbols) if col in facts.columns else np.nan

    # Market cap = price x shares outstanding.
    out["market_cap"] = out["price"] * out["shares"]

    # Book value per share = equity / shares. Price-to-book = price / bvps.
    bvps = out["equity"] / out["shares"]
    out["price_to_book"] = np.where(bvps > 0, out["price"] / bvps, np.nan)

    # Earnings per share = net_income / shares. P/E only meaningful if earnings > 0.
    eps = out["net_income"] / out["shares"]
    out["price_to_earnings"] = np.where(eps > 0, out["price"] / eps, np.nan)

    # Earnings yield = E/P. Defined even when we'd rather rank cheap-to-expensive;
    # unlike P/E it behaves sensibly for low (still positive) earnings.
    out["earnings_yield"] = np.where(
        (out["market_cap"] > 0) & (out["net_income"].notna()),
        out["net_income"] / out["market_cap"],
        np.nan,
    )

    # Price-to-sales = market cap / revenue.
    out["price_to_sales"] = np.where(
        out["revenue"] > 0, out["market_cap"] / out["revenue"], np.nan
    )

    # Return on equity = net income / equity. A quality measure.
    out["roe"] = np.where(out["equity"] > 0, out["net_income"] / out["equity"], np.nan)

    # A hard financial-strength gate used by many AAII-style screens.
    out["positive_earnings"] = out["net_income"] > 0

    return out[[
        "price", "market_cap",
        "price_to_book", "price_to_earnings", "price_to_sales",
        "earnings_yield", "roe", "positive_earnings",
    ]]
