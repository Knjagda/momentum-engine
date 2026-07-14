"""
Construction: turn a ranking into a portfolio.

    SignalResult (scores)  ->  select top N  ->  weight  ->  apply caps  ->  Portfolio

This is the last step of the deterministic core. Its output is the seam that
everything else -- metrics, reports, the agentic explainer, HEES -- consumes.
"""

from __future__ import annotations

import pandas as pd

from engine.data.base import PriceData
from engine.markets.market import Market
from engine.portfolio.portfolio import Portfolio, Position
from engine.portfolio.weighting import (
    cap_position_weights,
    cap_sector_weights,
    get_weighting,
)
from engine.signals.base import SignalResult
from engine.universe.universe import Membership


def select_with_buffer(
    signal_result: SignalResult,
    top_n: int,
    current_symbols: list[str],
    exit_rank: int,
) -> list[str]:
    """
    THE NO-TRADE BUFFER. The cheapest turnover reduction there is.

    The naive rule -- "hold the top 20, sell anything that drops to 21" -- is
    needlessly twitchy. A stock that slips from rank 18 to rank 22 has barely
    changed; the signal certainly has not said anything meaningful. But the naive
    rule sells it, pays the spread, buys a near-identical name, and pays again.
    Do that every month and you have built a machine for donating to your broker.

    The buffer separates ENTRY from EXIT:

        enter  if rank <= top_n        (must be genuinely among the best)
        hold   if rank <= exit_rank    (a wider band: still good enough to keep)

    So with top_n=20, exit_rank=30: a holding that slips to 24 is KEPT. It is only
    sold once it falls out of the top 30 -- or once it is pushed out by a better
    name and there is no room.

    Research: Calluzzo, Moneta & Topaloglu (2025), "Momentum at Long Holding
    Periods" -- longer holding cuts costs without surrendering the signal.
    """
    if exit_rank < top_n:
        raise ValueError(
            f"exit_rank ({exit_rank}) must be >= top_n ({top_n}); "
            f"a hold band narrower than the entry band makes no sense"
        )

    ranks = signal_result.rank()          # rank 1 = best; unscoreable are absent

    # KEEP: current holdings still inside the wider hold band, best first.
    keepers = sorted(
        (s for s in current_symbols if s in ranks.index and ranks[s] <= exit_rank),
        key=lambda s: ranks[s],
    )[:top_n]

    # FILL: any empty slots go to the best names we do not already hold.
    held = set(keepers)
    slots = top_n - len(keepers)

    if slots > 0:
        candidates = [
            s for s in ranks.sort_values().index
            if s not in held and ranks[s] <= top_n
        ]
        keepers.extend(candidates[:slots])

    return sorted(keepers, key=lambda s: ranks[s])


def build_portfolio(
    signal_result: SignalResult,
    market: Market,
    top_n: int,
    weighting: str = "equal",
    membership: Membership | None = None,
    prices: PriceData | None = None,
    max_position_weight: float | None = None,
    max_sector_weight: float | None = None,
    preselected_symbols: list[str] | None = None,
    **weighting_kwargs,
) -> Portfolio:
    """
    Build the portfolio implied by a set of scores.

    Args:
        signal_result:       scores from a Signal
        market:              the Market object (currency, annualization, ...)
        top_n:               how many names to hold
        weighting:           "equal" | "inverse_vol"
        membership:          used to attach sector labels
        prices:              required by volatility-based weighting schemes
        max_position_weight: no single name above this (e.g. 0.10)
        max_sector_weight:   no sector above this (e.g. 0.35)

    If fewer than `top_n` names are scoreable, we hold what we have rather than
    padding with names the signal could not evaluate. A portfolio of 14 is honest;
    a portfolio of 20 containing 6 guesses is not.
    """
    if top_n <= 0:
        raise ValueError(f"top_n must be positive, got {top_n}")

    if preselected_symbols is not None:
        # Selection was already decided (e.g. by the no-trade buffer).
        symbols = [s for s in preselected_symbols if s in signal_result.valid.index]
        selected = signal_result.valid[symbols]
    else:
        selected = signal_result.top(top_n)
        symbols = list(selected.index)

    if not symbols:
        raise ValueError(
            f"No scoreable securities on {signal_result.as_of.date()}. "
            f"Universe may be too small or history too short."
        )

    # -- weight -------------------------------------------------------------
    scheme = get_weighting(weighting)
    weights = scheme(
        symbols,
        prices=prices,
        as_of=signal_result.as_of,
        **weighting_kwargs,
    )

    # -- constrain ----------------------------------------------------------
    sectors = (
        {s: membership.sector_of(s) for s in symbols} if membership else {s: "" for s in symbols}
    )

    if max_sector_weight is not None:
        weights = cap_sector_weights(weights, sectors, max_sector_weight)

    if max_position_weight is not None:
        weights = cap_position_weights(weights, max_position_weight)
        # Capping positions can nudge a sector back over its limit; one more pass
        # keeps both constraints roughly satisfied, position cap taking priority.
        if max_sector_weight is not None:
            weights = cap_sector_weights(weights, sectors, max_sector_weight)
            weights = cap_position_weights(weights, max_position_weight)

    # -- assemble -----------------------------------------------------------
    ranks = signal_result.rank()

    positions = [
        Position(
            symbol=symbol,
            weight=float(weights[symbol]),
            score=float(selected[symbol]),
            rank=int(ranks.get(symbol, 0)),
            sector=sectors.get(symbol, ""),
        )
        for symbol in symbols
    ]
    positions.sort(key=lambda p: p.rank)

    portfolio = Portfolio(
        market_id=market.market_id,
        currency=market.currency,
        as_of=signal_result.as_of,
        positions=positions,
        cash_weight=0.0,
        metadata={
            "signal": signal_result.name,
            "signal_params": signal_result.params,
            "weighting": weighting,
            "top_n_requested": top_n,
            "top_n_held": len(positions),
            "universe_scored": len(signal_result.valid),
            "max_position_weight": max_position_weight,
            "max_sector_weight": max_sector_weight,
        },
    )

    portfolio.validate()
    return portfolio
