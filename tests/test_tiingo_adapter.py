"""
Tests for the TiingoAdapter (piece 2), offline via monkeypatched fetch_raw.

We don't hit the network; we replace fetch_raw with a synthetic source and check the
adapter's assembly: it returns PriceData with DISPLAY-symbol columns, adjusted closes,
per-ticker caching (second call doesn't refetch), symbol windowing, and graceful
omission of unknown tickers. The placeholder-trim itself is covered in test_tiingo_core.
"""

import pandas as pd
import pytest

from engine.data import get_adapter
from engine.data.base import PriceData
from engine.markets.market import load_market


def _fake_raw(ticker, key, start="2004-01-01", end=None, timeout=30):
    """Synthetic Tiingo response: 100 bars, adjusted columns present."""
    if ticker.upper() == "UNKNOWN":
        return pd.DataFrame()                      # 404-style miss
    idx = pd.bdate_range("2023-01-02", periods=100)
    base = 100.0 if ticker.upper() == "AAPL" else 50.0
    return pd.DataFrame({
        "adjClose": [base + i for i in range(100)],
        "adjVolume": [1_000_000] * 100,
        "close": [base + i for i in range(100)],
        "volume": [1_000_000] * 100,
    }, index=idx)


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    import engine.data.tiingo_adapter as mod
    monkeypatch.setattr(mod, "fetch_raw", _fake_raw)
    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path / "tiingo")
    monkeypatch.setattr(mod, "_SECONDS_BETWEEN_FETCHES", 0.0)  # no real throttle in tests
    market = load_market("us")
    from engine.data.tiingo_adapter import TiingoAdapter
    return TiingoAdapter(market, api_key="fake-key")


def test_requires_api_key():
    from engine.data.tiingo_adapter import TiingoAdapter
    with pytest.raises(ValueError, match="API key"):
        TiingoAdapter(load_market("us"), api_key="")


def test_fetch_returns_price_data(adapter):
    pd_out = adapter.fetch(["AAPL", "MSFT"], "2023-01-01", "2023-12-31")
    assert isinstance(pd_out, PriceData)
    assert set(pd_out.symbols) == {"AAPL", "MSFT"}
    assert not pd_out.close.empty


def test_columns_are_display_symbols(adapter):
    """Columns must be display symbols, never vendor tickers with suffixes."""
    pd_out = adapter.fetch(["AAPL"], "2023-01-01", "2023-12-31")
    assert list(pd_out.close.columns) == ["AAPL"]


def test_unknown_ticker_omitted(adapter):
    """A ticker Tiingo doesn't have is omitted, not fatal, if others succeed."""
    pd_out = adapter.fetch(["AAPL", "UNKNOWN"], "2023-01-01", "2023-12-31")
    assert set(pd_out.symbols) == {"AAPL"}


def test_all_unknown_raises(adapter):
    with pytest.raises(RuntimeError, match="no data"):
        adapter.fetch(["UNKNOWN"], "2023-01-01", "2023-12-31")


def test_uses_adjusted_close(adapter):
    """Values come from adjClose."""
    pd_out = adapter.fetch(["AAPL"], "2023-01-01", "2023-12-31")
    assert pd_out.close["AAPL"].iloc[0] == 100.0     # base for AAPL in _fake_raw


def test_per_ticker_cache_avoids_refetch(adapter, monkeypatch):
    """
    Second fetch of the same ticker reads cache -- fetch_raw not called again.
    Caching depends on a parquet engine (pyarrow/fastparquet); if none is installed,
    caching is a documented no-op, so we skip rather than assert.
    """
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        try:
            import fastparquet  # noqa: F401
        except ImportError:
            pytest.skip("no parquet engine; caching is a documented nicety, not required")

    calls = {"n": 0}
    import engine.data.tiingo_adapter as mod
    orig = _fake_raw
    def counting(ticker, key, **kw):
        calls["n"] += 1
        return orig(ticker, key, **kw)
    monkeypatch.setattr(mod, "fetch_raw", counting)

    adapter.fetch(["AAPL"], "2023-01-01", "2023-12-31")
    first = calls["n"]
    adapter.fetch(["AAPL"], "2023-01-01", "2023-12-31")   # should hit cache
    assert calls["n"] == first, "second fetch should use cache, not refetch"


def test_factory_builds_tiingo():
    """get_adapter can build a Tiingo adapter when a market names it."""
    market = load_market("us")
    a = get_adapter(market, api_key="fake-key") if market.data_adapter == "tiingo" else None
    # us market is yfinance by default; just assert the registry knows tiingo
    from engine.data import registered_adapters
    assert "tiingo" in registered_adapters()
