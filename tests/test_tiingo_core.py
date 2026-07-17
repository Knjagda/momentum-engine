"""
Tests for the Tiingo fetch + placeholder-trim core (piece 1).

Rewritten to match the REAL delisted-tail shapes we observed on live data, which
differ by how the company died:

  YHOO (acquired): real trading ends at the deal price ($52.58), then a SHORT frozen
    tail (same price, volume==0) padded to the delist date. Trim the frozen tail.
  SIVB (failed):   collapses, then trades as a penny ($0.01) WITH real volume for
    ~18 months (OTC afterlife), THEN a frozen zero-volume tail to today. Keep the
    penny-with-volume history (liquidity filters reject buying it); trim only the
    frozen zero tail.
  FRCB (failed):   trades at ~$0.00 WITH volume all the way to today -- no frozen
    tail at all. Nothing to trim.

The rule under test: trim only a trailing run of bars that are BOTH zero-volume AND
price-frozen. Real volume OR real price movement means the bar stays.
"""

import numpy as np
import pandas as pd

from engine.data.tiingo_core import (
    clean_series,
    placeholder_cutoff,
    _MIN_PLACEHOLDER_RUN,
)


def _df(prices, volumes, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(prices))
    return pd.DataFrame({"adjClose": prices, "adjVolume": volumes,
                         "close": prices, "volume": volumes}, index=idx)


# ---- placeholder_cutoff ---------------------------------------------------


def test_live_series_no_cutoff():
    df = _df([100 + i for i in range(50)], [1000] * 50)
    assert placeholder_cutoff(df["adjClose"], df["adjVolume"]) is None


def test_yhoo_pattern_frozen_tail_trimmed():
    """Acquisition: real trading, then a short frozen zero-volume tail at deal price."""
    prices = [50, 51, 52.58] + [52.58] * 6      # last real bar 52.58, then frozen
    volumes = [1e6, 1e6, 2.5e8] + [0] * 6
    df = _df(prices, volumes)
    cutoff = placeholder_cutoff(df["adjClose"], df["adjVolume"])
    assert cutoff == df.index[2]                 # keep up to the real 52.58 bar
    adj, vol = clean_series(df)
    assert len(adj) == 3
    assert adj.iloc[-1] == 52.58


def test_sivb_pattern_penny_with_volume_kept():
    """
    Failed stock: collapse, then penny trading WITH volume (kept), then frozen zeros.
    Only the frozen-zero tail is trimmed; the penny-with-volume afterlife stays.
    """
    collapse = list(np.linspace(300, 0.01, 40))
    penny_traded = [0.01] * 30                    # $0.01 but WITH volume -> real
    frozen = [0.01] * 10                          # $0.01, zero volume -> placeholder
    prices = collapse + penny_traded + frozen
    volumes = [1e6] * 40 + [5000] * 30 + [0] * 10
    df = _df(prices, volumes)

    adj, vol = clean_series(df)
    # frozen tail (10) removed; collapse + penny-with-volume (70) kept
    assert len(adj) == 70
    assert (vol.iloc[-1] > 0)                     # last kept bar had real volume


def test_frcb_pattern_volume_to_end_nothing_trimmed():
    """Trades at ~$0.00 WITH volume to the very end -- no frozen tail, keep everything."""
    prices = list(np.linspace(200, 0.0, 60))
    volumes = [30000] * 60                         # volume never dies
    df = _df(prices, volumes)
    assert placeholder_cutoff(df["adjClose"], df["adjVolume"]) is None
    adj, vol = clean_series(df)
    assert len(adj) == 60


def test_single_frozen_day_not_trimmed():
    """One flat zero-volume day is below the run threshold -- not a placeholder."""
    prices = [100] * 50 + [100] * (_MIN_PLACEHOLDER_RUN - 1)
    volumes = [1000] * 50 + [0] * (_MIN_PLACEHOLDER_RUN - 1)
    df = _df(prices, volumes)
    # short run -> no trim
    assert placeholder_cutoff(df["adjClose"], df["adjVolume"]) is None


def test_frozen_tail_must_be_price_flat_not_just_zero_volume():
    """
    Zero volume but MOVING price is not a placeholder (illiquid but real marks).
    Only zero-volume AND frozen-price counts.
    """
    prices = [100, 101, 102, 103, 104, 105]       # price moves every bar
    volumes = [1000, 0, 0, 0, 0, 0]               # volume dies but price still moves
    df = _df(prices, volumes)
    # price is NOT frozen, so this is not our placeholder signature
    assert placeholder_cutoff(df["adjClose"], df["adjVolume"]) is None


# ---- clean_series basics --------------------------------------------------


def test_clean_series_prefers_adjusted_close():
    idx = pd.bdate_range("2020-01-01", periods=30)
    df = pd.DataFrame({
        "adjClose": [100.0] * 30,
        "close": [200.0] * 30,
        "adjVolume": [1000] * 30,
    }, index=idx)
    adj, vol = clean_series(df)
    assert (adj == 100.0).all()


def test_clean_series_empty():
    adj, vol = clean_series(pd.DataFrame())
    assert adj.empty and vol.empty
