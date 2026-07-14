"""
yfinance price adapter.

V0 data source: free, good enough to build and validate the engine.
Production will swap in a paid provider (Tiingo / Polygon / an India vendor) --
which is why the vendor is named in market config and hidden behind PriceAdapter.

Caching: downloads are cached to disk (data/cache/) because backtests re-run
constantly and hammering a free API is both slow and rude. Delete the cache
folder to force a refresh.
"""

from __future__ import annotations

import hashlib
import warnings
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from engine.data.base import PriceAdapter, PriceData
from engine.markets.market import Market

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


class YFinanceAdapter(PriceAdapter):
    """Fetch adjusted daily prices from Yahoo Finance."""

    def __init__(self, market: Market, use_cache: bool = True) -> None:
        super().__init__(market)
        self.use_cache = use_cache

    # -- public -------------------------------------------------------------

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

        cache_path = self._cache_path(symbols, start_ts, end_ts)
        if self.use_cache and cache_path.exists():
            close, volume = self._read_cache(cache_path)
        else:
            close, volume = self._download(symbols, start_ts, end_ts)
            if self.use_cache:
                self._write_cache(cache_path, close, volume)

        return PriceData(market=self.market, close=close, volume=volume)

    def fetch_benchmark(
        self,
        start: date | datetime | str,
        end: date | datetime | str,
    ) -> pd.Series:
        """The benchmark index. No ticker suffix -- indices are not equities."""
        import yfinance as yf

        ticker = self.market.benchmark.ticker      # raw, NOT resolve_ticker()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                tickers=[ticker],
                start=pd.Timestamp(start),
                end=pd.Timestamp(end),
                auto_adjust=True,
                progress=False,
                group_by="column",
            )

        if raw is None or raw.empty:
            raise RuntimeError(f"No benchmark data for {ticker}")

        close = self._extract(raw, "Close", [ticker])
        series = close.iloc[:, 0]
        series.index = pd.to_datetime(series.index)
        series.name = ticker
        return series.sort_index().dropna()

    # -- download -----------------------------------------------------------

    def _download(
        self, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        import yfinance as yf  # imported lazily so tests can run without network

        # The engine never writes vendor ticker conventions itself. It asks the market.
        vendor_tickers = [self.market.resolve_ticker(s) for s in symbols]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                tickers=vendor_tickers,
                start=start,
                end=end,
                auto_adjust=True,      # splits AND dividends handled. Non-negotiable.
                progress=False,
                group_by="column",
                threads=True,
            )

        if raw is None or raw.empty:
            raise RuntimeError(
                f"No data returned for {len(vendor_tickers)} symbols in "
                f"{self.market.market_id}. Check tickers and date range."
            )

        close = self._extract(raw, "Close", vendor_tickers)
        volume = self._extract(raw, "Volume", vendor_tickers)

        # Hand back DISPLAY symbols -- downstream code never sees ".NS" et al.
        close.columns = [self.market.strip_ticker(c) for c in close.columns]
        volume.columns = [self.market.strip_ticker(c) for c in volume.columns]

        close.index = pd.to_datetime(close.index)
        volume.index = pd.to_datetime(volume.index)

        return close.sort_index(), volume.sort_index()

    @staticmethod
    def _extract(raw: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
        """
        Pull one field out of yfinance's response.

        yfinance returns MultiIndex columns for several tickers but flat columns
        for a single ticker -- handle both rather than assuming.
        """
        if isinstance(raw.columns, pd.MultiIndex):
            if field not in raw.columns.get_level_values(0):
                raise KeyError(f"'{field}' not in response. Got: {set(raw.columns.get_level_values(0))}")
            frame = raw[field].copy()
        else:
            if field not in raw.columns:
                raise KeyError(f"'{field}' not in response. Got: {list(raw.columns)}")
            frame = raw[[field]].copy()
            frame.columns = [tickers[0]]

        return frame

    # -- cache --------------------------------------------------------------

    def _cache_path(self, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp) -> Path:
        key = "|".join([
            self.market.market_id,
            ",".join(sorted(symbols)),
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
        ])
        digest = hashlib.md5(key.encode()).hexdigest()[:16]
        folder = CACHE_DIR / self.market.market_id.lower()
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{digest}.parquet"

    @staticmethod
    def _write_cache(path: Path, close: pd.DataFrame, volume: pd.DataFrame) -> None:
        combined = pd.concat({"close": close, "volume": volume}, axis=1)
        try:
            combined.to_parquet(path)
        except Exception:  # parquet engine missing -- caching is a nicety, not a requirement
            pass

    @staticmethod
    def _read_cache(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        combined = pd.read_parquet(path)
        return combined["close"].copy(), combined["volume"].copy()
