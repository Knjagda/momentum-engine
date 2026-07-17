"""
Tiingo price adapter (piece 2 of 3): survivorship-free prices behind PriceAdapter.

This wraps the piece-1 core (fetch + placeholder-trim) in the standard PriceAdapter
interface, so the engine can pull prices from Tiingo exactly as it does from yfinance
-- same fetch() signature, same PriceData return, display-symbol columns. Swap the
vendor, keep the engine.

WHY IT EXISTS. yfinance cannot price delisted names; Tiingo can. Pointing the engine
at Tiingo makes every backtest survivorship-honest.

HOW IT DIFFERS FROM yfinance (and why the code looks different):
  - ONE TICKER PER REQUEST. Tiingo's daily endpoint is per-ticker, so we loop. That
    is fine because we PER-TICKER CACHE: a dead company's history never changes, so
    we fetch it once and keep it forever. Re-runs read parquet, never the network.
  - THROTTLED. The free tier allows ~50 symbols/hour. We pace only ACTUAL network
    fetches (cached tickers are free) and surface a clear message if throttled.
  - ADJUSTED + TRIMMED. The core already returns split/dividend-adjusted closes with
    the frozen placeholder tail removed; we just assemble them into a frame.

The API key is passed in (never hardcoded). Get a free one at tiingo.com.
"""

from __future__ import annotations

import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from engine.data.base import PriceAdapter, PriceData
from engine.data.tiingo_core import TiingoError, clean_series, fetch_raw
from engine.markets.market import Market

CACHE_DIR = Path("data/prices/tiingo")

# Free tier ~50 symbols/hour. We pace fetches conservatively: ~1.3s between network
# calls keeps us well under any burst limit; the hourly cap is the real ceiling, and
# we pull the dead-ticker set once (then cached), so a slow first run is acceptable.
_SECONDS_BETWEEN_FETCHES = 1.3


class TiingoAdapter(PriceAdapter):
    """Survivorship-free daily prices from Tiingo, behind the standard interface."""

    def __init__(self, market: Market, api_key: str, use_cache: bool = True) -> None:
        super().__init__(market)
        if not api_key:
            raise ValueError("TiingoAdapter needs an API key (free at tiingo.com).")
        self.api_key = api_key
        self.use_cache = use_cache
        self._last_fetch_ts = 0.0

    # -- the interface the engine calls ---------------------------------------

    def fetch(
        self,
        symbols: list[str],
        start: date | datetime | str,
        end: date | datetime | str,
    ) -> PriceData:
        if not symbols:
            raise ValueError("fetch() called with no symbols")

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        start_s = start_ts.strftime("%Y-%m-%d")
        end_s = end_ts.strftime("%Y-%m-%d")

        close_cols: dict[str, pd.Series] = {}
        volume_cols: dict[str, pd.Series] = {}

        for display in symbols:
            ticker = self.market.resolve_ticker(display)
            adj, vol = self._fetch_one(ticker, start_s, end_s)
            if adj.empty:
                continue                       # unknown/dead-to-Tiingo: just omit
            # window to the requested range
            adj = adj.loc[(adj.index >= start_ts) & (adj.index <= end_ts)]
            vol = vol.loc[(vol.index >= start_ts) & (vol.index <= end_ts)]
            close_cols[display] = adj          # DISPLAY symbol, not vendor ticker
            volume_cols[display] = vol

        if not close_cols:
            raise RuntimeError(
                f"Tiingo returned no data for any of {len(symbols)} symbols "
                f"in {self.market.market_id}. Check tickers and key."
            )

        close = pd.DataFrame(close_cols).sort_index()
        volume = pd.DataFrame(volume_cols).sort_index()
        close.index = pd.to_datetime(close.index)
        volume.index = pd.to_datetime(volume.index)

        return PriceData(market=self.market, close=close, volume=volume)

    def fetch_benchmark(
        self,
        start: date | datetime | str,
        end: date | datetime | str,
    ) -> pd.Series:
        """Benchmark index. Tiingo carries SPY etc.; use the market's benchmark ticker."""
        ticker = self.market.benchmark.ticker.lstrip("^")   # Tiingo uses plain symbols
        adj, _vol = self._fetch_one(
            ticker,
            pd.Timestamp(start).strftime("%Y-%m-%d"),
            pd.Timestamp(end).strftime("%Y-%m-%d"),
        )
        if adj.empty:
            raise RuntimeError(f"No Tiingo benchmark data for {ticker}")
        adj = adj.loc[(adj.index >= pd.Timestamp(start)) & (adj.index <= pd.Timestamp(end))]
        adj.name = ticker
        return adj.sort_index()

    # -- per-ticker fetch with cache + throttle -------------------------------

    def _fetch_one(self, ticker: str, start: str, end: str) -> tuple[pd.Series, pd.Series]:
        """Fetch one ticker's cleaned (adjClose, volume), using the per-ticker cache."""
        cache = self._cache_path(ticker)
        if self.use_cache and cache.exists():
            try:
                combined = pd.read_parquet(cache)
                return combined["close"].dropna(), combined["volume"].dropna()
            except Exception:
                pass  # fall through to refetch if the cache file is unreadable

        self._throttle()
        try:
            raw = fetch_raw(ticker, self.api_key, start=start)
        except TiingoError:
            raise
        adj, vol = clean_series(raw)

        if self.use_cache and not adj.empty:
            self._write_cache(cache, adj, vol)
        return adj, vol

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_fetch_ts
        if elapsed < _SECONDS_BETWEEN_FETCHES:
            time.sleep(_SECONDS_BETWEEN_FETCHES - elapsed)
        self._last_fetch_ts = time.time()

    # -- cache ----------------------------------------------------------------

    def _cache_path(self, ticker: str) -> Path:
        folder = CACHE_DIR / self.market.market_id.lower()
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{ticker.upper()}.parquet"

    @staticmethod
    def _write_cache(path: Path, close: pd.Series, volume: pd.Series) -> None:
        combined = pd.DataFrame({"close": close, "volume": volume})
        try:
            combined.to_parquet(path)
        except Exception:  # parquet engine missing -- caching is a nicety
            pass
