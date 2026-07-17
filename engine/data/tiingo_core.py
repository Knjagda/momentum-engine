"""
Tiingo price data: the fetch + delisting-detection core (piece 1 of the adapter).

WHY TIINGO. yfinance drops a company the moment it delists, so ~410 dead names in
our universe (LEH, SIVB, FRC, ...) simply cannot be priced -- and every backtest
silently runs on survivors only, inflating returns. Tiingo's free tier keeps the
FULL history of delisted names, including the collapse. That closes the single
biggest gap in the engine, for free (verified: SIVB peak $755 -> $0.01, adjusted
columns present).

THE QUIRK THIS MODULE HANDLES. Tiingo does not END a delisted ticker's series at the
death date. Instead it carries the ticker forward to TODAY at a frozen placeholder
(SIVB sits at $0.01, ZERO volume, for years after March 2023). If we handed that to
the backtest as-is, the engine would think SIVB is a live, tradeable $0.01 micro-cap
-- a NEW survivorship-flavoured bias. So we must DETECT THE REAL DEATH: the last bar
with genuine trading, and treat everything after it as delisted (NaN), not as price.

This module does the vendor-specific work (HTTP, JSON, adjusted-close selection,
death detection) and returns clean per-symbol frames. The adapter wrapper that
conforms to PriceAdapter is the next piece; keeping them separate makes the tricky
death-detection logic testable offline with synthetic series.
"""

from __future__ import annotations

import io
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

TIINGO_BASE = "https://api.tiingo.com/tiingo/daily"

# A delisted ticker's series often ends with a PLACEHOLDER tail: bars carried forward
# to today at a FROZEN price with EXACTLY zero volume (no trades, no price movement).
# That is fake -- it is not trading, just Tiingo padding the series to the present.
# We trim ONLY that tail. Everything with real volume OR real price movement stays,
# including a failed stock's messy penny-trading afterlife (SIVB at $0.01 for months)
# -- our liquidity/min-price filters already refuse to BUY such names, so keeping the
# bars is harmless and avoids us guessing where "death" was.
_MIN_PLACEHOLDER_RUN = 3    # a short frozen-zero-volume tail is enough to be padding


class TiingoError(RuntimeError):
    pass


def fetch_raw(ticker: str, key: str, start: str = "2004-01-01",
              end: str | None = None, timeout: int = 30) -> pd.DataFrame:
    """
    Raw Tiingo daily bars for one ticker. Returns a DataFrame with a DatetimeIndex
    and the columns Tiingo provides (close, adjClose, volume, adjVolume, ...).
    Empty DataFrame if the ticker is unknown (404).
    """
    url = f"{TIINGO_BASE}/{ticker}/prices?startDate={start}"
    if end:
        url += f"&endDate={end}"
    url += f"&token={key}"
    req = Request(url, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
    except HTTPError as e:
        if e.code == 404:
            return pd.DataFrame()
        if e.code == 429:
            raise TiingoError(f"{ticker}: rate limited (50/hr on free tier).") from e
        raise TiingoError(f"{ticker}: HTTP {e.code}.") from e

    df = pd.read_json(io.StringIO(raw))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.set_index("date").sort_index()


def placeholder_cutoff(close: pd.Series, volume: pd.Series) -> pd.Timestamp | None:
    """
    Find where the FROZEN zero-volume placeholder tail begins, so we can drop it.

    Returns the date of the last REAL bar (keep up to and including it), or None if
    there is no placeholder tail (the series is real all the way to the end).

    Placeholder definition: a trailing run of bars, each with volume == 0 AND price
    unchanged from the prior bar. This is padding to today's date -- not trading.
    A single flat day is not enough; we require a short run to avoid trimming a
    legitimate quiet bar.
    """
    if close.empty:
        return None
    vol = volume.reindex(close.index).fillna(0).to_numpy()
    px = close.to_numpy()

    # Walk back over the trailing run of (zero-volume AND price-frozen) bars.
    run = 0
    i = len(px) - 1
    while i > 0 and vol[i] <= 0 and px[i] == px[i - 1]:
        run += 1
        i -= 1

    if run < _MIN_PLACEHOLDER_RUN:
        return None
    # i now points at the last bar of real trading (the frozen tail started at i+1).
    return close.index[i]


def clean_series(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Turn raw Tiingo bars into (adjusted_close, volume), with the FROZEN zero-volume
    placeholder tail removed. The real trading history -- including a failed stock's
    collapse and penny-trading afterlife -- is kept; only the fake padding is dropped.

    Returns two date-indexed Series: adjusted close and volume.
    """
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # ADJUSTED close is mandatory (splits/divs). Fall back to close only if adj is
    # entirely absent -- but Tiingo always provides it, so that path is defensive.
    adj = df["adjClose"] if "adjClose" in df.columns else df["close"]
    vol = df["adjVolume"] if "adjVolume" in df.columns else df.get("volume", pd.Series(index=df.index, dtype=float))

    cutoff = placeholder_cutoff(adj, vol)
    if cutoff is not None:
        adj = adj.loc[:cutoff]
        vol = vol.loc[:cutoff]

    return adj.astype(float), vol.astype(float)


def fetch_clean(ticker: str, key: str, start: str = "2004-01-01",
                end: str | None = None) -> tuple[pd.Series, pd.Series, pd.Timestamp | None]:
    """
    Convenience: fetch + clean in one call.
    Returns (adjusted_close, volume, last_real_bar_or_None).
    `last_real_bar` is the final genuine trading date (== series end after trimming),
    or None if there was no data.
    """
    raw = fetch_raw(ticker, key, start=start, end=end)
    if raw.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), None
    adj, vol = clean_series(raw)
    last_real = adj.index[-1] if not adj.empty else None
    return adj, vol, last_real
