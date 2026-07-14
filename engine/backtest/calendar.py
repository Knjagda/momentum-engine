"""
Rebalance dates, on the RIGHT exchange calendar.

You cannot rebalance on the 1st of the month, because the 1st is often a Saturday,
or Christmas, or Diwali. Rebalance dates must land on actual trading days -- and
which days those are depends on the country.

NYSE and NSE have different holidays, and India's move with the lunar calendar.
So the calendar name comes from the market config (SPEC §2), and we ask a proper
exchange-calendar library rather than assuming "weekdays minus a few".

Getting this wrong is not cosmetic: rebalancing on a date with no prices means the
engine silently uses the previous close, quietly shifting every decision by a day.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

import pandas as pd

from engine.markets.market import Market

Frequency = Literal["weekly", "monthly", "quarterly"]
DayChoice = Literal["first_trading_day", "last_trading_day"]

_PERIOD_CODE = {
    "weekly": "W",
    "monthly": "M",
    "quarterly": "Q",
}

PERIODS_PER_YEAR = {
    "weekly": 52,
    "monthly": 12,
    "quarterly": 4,
}


def trading_days(
    market: Market,
    start: date | datetime | str,
    end: date | datetime | str,
) -> pd.DatetimeIndex:
    """Every real trading day for this market's exchange, between start and end."""
    import pandas_market_calendars as mcal

    calendar = mcal.get_calendar(market.calendar)      # from config: NYSE / NSE / ...
    days = calendar.valid_days(start_date=pd.Timestamp(start), end_date=pd.Timestamp(end))

    index = pd.DatetimeIndex(days)
    if index.tz is not None:
        index = index.tz_convert(None) if index.tz else index
        index = index.tz_localize(None) if index.tz else index

    return pd.DatetimeIndex(index).normalize()


def rebalance_dates(
    market: Market,
    start: date | datetime | str,
    end: date | datetime | str,
    frequency: Frequency = "monthly",
    day: DayChoice = "first_trading_day",
) -> pd.DatetimeIndex:
    """
    The dates on which the strategy re-ranks and trades.

    Monthly is the default for good reason: it captures momentum cycles without
    the cost bloodbath of weekly trading. Our own cost model says 32% turnover
    rebalanced weekly burns >8%/yr in India -- an edge-destroying number.
    """
    if frequency not in _PERIOD_CODE:
        raise ValueError(
            f"Unknown frequency '{frequency}'. Available: {sorted(_PERIOD_CODE)}"
        )

    days = trading_days(market, start, end)
    if len(days) == 0:
        return pd.DatetimeIndex([])

    periods = days.to_period(_PERIOD_CODE[frequency])
    frame = pd.DataFrame({"day": days}, index=periods)

    if day == "first_trading_day":
        picked = frame.groupby(level=0)["day"].min()
    elif day == "last_trading_day":
        picked = frame.groupby(level=0)["day"].max()
    else:
        raise ValueError(f"Unknown day choice '{day}'")

    return pd.DatetimeIndex(picked.values).sort_values()


def periods_per_year(frequency: Frequency) -> int:
    """Used to annualize returns, volatility and Sharpe from period data."""
    if frequency not in PERIODS_PER_YEAR:
        raise ValueError(f"Unknown frequency '{frequency}'")
    return PERIODS_PER_YEAR[frequency]
