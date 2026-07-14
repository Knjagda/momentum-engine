"""
Weighting: ranking says WHAT to hold, weighting says HOW MUCH.

Schemes are plug-ins, exactly like signals. V0 ships equal weight -- deliberately.
It is boring, it is robust, and it is very hard to break. Fancier schemes
(mean-variance optimisation in particular) are notoriously fragile out-of-sample:
they chase estimation noise and produce unstable, extreme weights. Start simple,
earn the complexity.

Constraints (position caps, sector caps) are applied AFTER weighting, because a
weighting scheme's job is to express a view and a constraint's job is to stop
that view from becoming reckless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.data.base import PriceData


# ---------------------------------------------------------------------------
# Weighting schemes
# ---------------------------------------------------------------------------


def equal_weight(symbols: list[str], **_) -> pd.Series:
    """
    Everyone gets the same slice.

    Why this is the default: it makes no claim it cannot support. It also stops
    two mega-caps from quietly becoming 30% of the book -- the exact problem the
    pitch deck calls out ("equal weight avoids Apple + Nvidia eating 30%").
    """
    if not symbols:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(symbols), index=symbols, dtype=float)


def inverse_vol_weight(
    symbols: list[str],
    prices: PriceData | None = None,
    as_of: pd.Timestamp | None = None,
    vol_window_days: int = 126,
    **_,
) -> pd.Series:
    """
    Calmer stocks get bigger weights: w ∝ 1 / volatility.

    A gentle risk control. It stops the portfolio's total risk being dominated by
    its two most violent holdings -- which, in a momentum book, is a real danger,
    because the biggest recent winners are often the most volatile names.
    """
    if not symbols:
        return pd.Series(dtype=float)
    if prices is None or as_of is None:
        raise ValueError("inverse_vol_weight needs `prices` and `as_of`")

    history = prices.up_to(as_of)          # never peek (SPEC §4.1)
    ann = history.market.trading_days_per_year

    vols: dict[str, float] = {}
    for symbol in symbols:
        if symbol not in history.close.columns:
            vols[symbol] = np.nan
            continue
        window = history.close[symbol].tail(vol_window_days + 1)
        daily = window.pct_change().dropna()
        vols[symbol] = (
            float(daily.std(ddof=1)) * np.sqrt(ann) if len(daily) >= 2 else np.nan
        )

    vol = pd.Series(vols, dtype=float)

    # A stock with no usable volatility estimate falls back to the median, rather
    # than being dropped or given an infinite weight.
    if vol.notna().any():
        vol = vol.fillna(vol.median())
    else:
        return equal_weight(symbols)

    vol = vol.replace(0.0, np.nan).fillna(vol[vol > 0].median() if (vol > 0).any() else 1.0)

    inv = 1.0 / vol
    return inv / inv.sum()


_SCHEMES = {
    "equal": equal_weight,
    "inverse_vol": inverse_vol_weight,
}


def get_weighting(method: str):
    key = method.lower()
    if key not in _SCHEMES:
        raise KeyError(f"Unknown weighting '{method}'. Available: {sorted(_SCHEMES)}")
    return _SCHEMES[key]


def registered_weightings() -> list[str]:
    return sorted(_SCHEMES)


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


def cap_position_weights(
    weights: pd.Series,
    max_weight: float,
    tolerance: float = 1e-9,
    max_iterations: int = 100,
) -> pd.Series:
    """
    No single name may exceed `max_weight`.

    Capping is iterative, not a single pass: cap the offenders, spread the excess
    across the rest -- which can push a previously-fine holding over the line, so
    repeat until nothing breaches. A naive one-pass cap silently leaves violations
    behind.
    """
    if weights.empty:
        return weights

    if max_weight * len(weights) < 1.0 - tolerance:
        raise ValueError(
            f"Cannot cap {len(weights)} positions at {max_weight:.1%} each -- "
            f"they could hold at most {max_weight * len(weights):.1%} of the portfolio. "
            f"Raise max_position_weight or hold more names."
        )

    w = weights.astype(float).copy()

    for _ in range(max_iterations):
        over = w > max_weight + tolerance
        if not over.any():
            break

        excess = float((w[over] - max_weight).sum())
        w[over] = max_weight

        room = ~over
        if not room.any():
            break

        # Redistribute proportionally to the names that still have headroom.
        under = w[room]
        if under.sum() > 0:
            w[room] = under + excess * (under / under.sum())
        else:
            w[room] = excess / room.sum()

    return w / w.sum()


def cap_sector_weights(
    weights: pd.Series,
    sectors: dict[str, str],
    max_sector_weight: float,
    tolerance: float = 1e-9,
    max_iterations: int = 100,
) -> pd.Series:
    """
    No sector may exceed `max_sector_weight`.

    This matters more for momentum than for most strategies. Momentum buys whatever
    has been winning -- and what has been winning is usually one theme. A Top-20 can
    become a 60% technology bet without anyone deciding to make one.

    (Sector caps are a blunt instrument: five "different" sectors can still share
    one economic driver. That is what HEES is for. This is the first line only.)
    """
    if weights.empty:
        return weights

    w = weights.astype(float).copy()
    sector_of = pd.Series({s: sectors.get(s, "Unknown") for s in w.index})

    n_sectors = sector_of.nunique()
    if max_sector_weight * n_sectors < 1.0 - tolerance:
        # Not enough sectors to satisfy the cap -- don't silently produce nonsense.
        raise ValueError(
            f"Cannot cap {n_sectors} sectors at {max_sector_weight:.1%} each. "
            f"Hold a broader set of names or raise the cap."
        )

    for _ in range(max_iterations):
        totals = w.groupby(sector_of).sum()
        over = totals[totals > max_sector_weight + tolerance]
        if over.empty:
            break

        excess = 0.0
        for sector, total in over.items():
            members = sector_of[sector_of == sector].index
            scale = max_sector_weight / total
            excess += float(w[members].sum()) * (1 - scale)
            w[members] = w[members] * scale

        under_members = sector_of[~sector_of.isin(over.index)].index
        if len(under_members) == 0:
            break

        under = w[under_members]
        if under.sum() > 0:
            w[under_members] = under + excess * (under / under.sum())
        else:
            w[under_members] = excess / len(under_members)

    return w / w.sum()
