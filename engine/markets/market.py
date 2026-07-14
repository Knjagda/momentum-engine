"""
Market configuration loader.

THE RULE (see SPEC.md §2):
    A Market is a configuration object. The engine reads EVERYTHING from it.
    No market-specific constant ("USD", "NYSE", ".NS", any tax rate) may ever
    be hard-coded in engine logic.

Switching from the US to India is this, and nothing more:

    market = load_market("us")      ->  market = load_market("india")

Every downstream module (data, universe, signals, costs, backtest) receives
this object and asks it questions. It never assumes a country.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Where market configs live, relative to the repo root.
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "markets"


# ---------------------------------------------------------------------------
# Sub-objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Benchmark:
    """The index a strategy is measured against."""

    ticker: str
    name: str


@dataclass(frozen=True)
class Universe:
    """A pool of eligible securities within a market."""

    key: str
    name: str
    source: str
    path: str
    survivorship_bias: bool

    @property
    def disclaimer(self) -> str | None:
        """Text that MUST appear on any backtest using this universe."""
        if self.survivorship_bias:
            return (
                f"⚠️ '{self.name}' uses a CURRENT membership list. Companies that were "
                f"removed from the index in the past are missing, so historical results "
                f"are optimistic (survivorship bias). Not a prediction."
            )
        return None


@dataclass(frozen=True)
class Liquidity:
    """Minimum tradability filter. A stock we cannot trade is not investable."""

    min_avg_daily_value: float
    lookback_days: int


@dataclass(frozen=True)
class CostModel:
    """
    Every simulated trade pays these. No exceptions. (SPEC.md §4.3)

    Costs are expressed in basis points (1 bp = 0.01%).
    India carries extra legs (STT, stamp duty, GST) that the US does not --
    which is precisely why this lives in config and not in code.
    """

    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    buy_extra_bps: float = 0.0        # e.g. India stamp duty (buy side only)
    sell_extra_bps: float = 0.0       # e.g. India STT (sell side)
    exchange_txn_bps: float = 0.0
    gst_on_charges_pct: float = 0.0   # GST applies to charges, not to trade value

    def _charges_bps(self) -> float:
        """Broker + exchange charges, which GST is levied on top of."""
        taxable = self.commission_bps + self.exchange_txn_bps
        return taxable * (1.0 + self.gst_on_charges_pct)

    def buy_cost_bps(self) -> float:
        """Total round-trip-free cost of BUYING, in basis points of trade value."""
        return self._charges_bps() + self.slippage_bps + self.buy_extra_bps

    def sell_cost_bps(self) -> float:
        """Total cost of SELLING, in basis points of trade value."""
        return self._charges_bps() + self.slippage_bps + self.sell_extra_bps

    def round_trip_bps(self) -> float:
        """Cost of buying then later selling the same position."""
        return self.buy_cost_bps() + self.sell_cost_bps()


@dataclass(frozen=True)
class TaxModel:
    """Informational for V0 reporting. Not yet applied to returns."""

    short_term_rate: float = 0.0
    long_term_rate: float = 0.0
    long_term_threshold_days: int = 365


# ---------------------------------------------------------------------------
# The Market object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Market:
    """Everything the engine needs to know about a country's equity market."""

    market_id: str
    name: str
    currency: str
    currency_symbol: str
    calendar: str
    data_adapter: str
    ticker_suffix: str
    benchmark: Benchmark
    universes: dict[str, Universe]
    liquidity: Liquidity
    costs: CostModel
    tax: TaxModel

    # -- ticker handling ----------------------------------------------------

    def resolve_ticker(self, symbol: str) -> str:
        """
        Turn a plain symbol into the data provider's convention.

            US    : "AAPL"     -> "AAPL"
            India : "RELIANCE" -> "RELIANCE.NS"

        The engine never writes ".NS" itself. It asks the market.
        """
        symbol = symbol.strip().upper()
        if self.ticker_suffix and not symbol.endswith(self.ticker_suffix):
            return f"{symbol}{self.ticker_suffix}"
        return symbol

    def strip_ticker(self, symbol: str) -> str:
        """Inverse of resolve_ticker -- back to the display symbol."""
        if self.ticker_suffix and symbol.endswith(self.ticker_suffix):
            return symbol[: -len(self.ticker_suffix)]
        return symbol

    # -- universes ----------------------------------------------------------

    def get_universe(self, key: str) -> Universe:
        if key not in self.universes:
            available = ", ".join(sorted(self.universes))
            raise KeyError(
                f"Universe '{key}' not defined for market '{self.market_id}'. "
                f"Available: {available}"
            )
        return self.universes[key]

    # -- formatting ---------------------------------------------------------

    def format_money(self, amount: float) -> str:
        """Display a number in this market's currency. Never assumes dollars."""
        return f"{self.currency_symbol}{amount:,.2f}"

    def __repr__(self) -> str:  # friendlier than the default dataclass dump
        return (
            f"<Market {self.market_id}: {self.name} | {self.currency} | "
            f"{self.calendar} | benchmark={self.benchmark.ticker} | "
            f"round-trip cost={self.costs.round_trip_bps():.1f}bps>"
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_market(market_key: str, config_dir: Path | str | None = None) -> Market:
    """
    Load a market by its config file name.

        load_market("us")     -> config/markets/us.yaml
        load_market("india")  -> config/markets/india.yaml

    This function is THE toggle. Nothing else in the engine changes.
    """
    directory = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    path = directory / f"{market_key.lower()}.yaml"

    if not path.exists():
        available = sorted(p.stem for p in directory.glob("*.yaml"))
        raise FileNotFoundError(
            f"No market config at {path}. Available markets: {available or 'none'}"
        )

    with path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    return _build_market(raw)


def available_markets(config_dir: Path | str | None = None) -> list[str]:
    """List every market the engine can currently run. Adding one = adding a YAML file."""
    directory = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    return sorted(p.stem for p in directory.glob("*.yaml"))


def _build_market(raw: dict[str, Any]) -> Market:
    """Turn parsed YAML into a validated Market object."""
    _require(raw, ["market_id", "name", "currency", "calendar", "benchmark", "universes"])

    universes = {
        key: Universe(
            key=key,
            name=spec.get("name", key),
            source=spec.get("source", "static_csv"),
            path=spec.get("path", ""),
            survivorship_bias=bool(spec.get("survivorship_bias", True)),
        )
        for key, spec in raw["universes"].items()
    }

    return Market(
        market_id=raw["market_id"],
        name=raw["name"],
        currency=raw["currency"],
        currency_symbol=raw.get("currency_symbol", ""),
        calendar=raw["calendar"],
        data_adapter=raw.get("data_adapter", "yfinance"),
        ticker_suffix=raw.get("ticker_suffix", ""),
        benchmark=Benchmark(
            ticker=raw["benchmark"]["ticker"],
            name=raw["benchmark"]["name"],
        ),
        universes=universes,
        liquidity=Liquidity(**raw.get("liquidity", {"min_avg_daily_value": 0, "lookback_days": 60})),
        costs=CostModel(**raw.get("costs", {})),
        tax=TaxModel(**raw.get("tax", {})),
    )


def _require(raw: dict[str, Any], keys: list[str]) -> None:
    missing = [k for k in keys if k not in raw]
    if missing:
        raise ValueError(f"Market config is missing required keys: {missing}")
