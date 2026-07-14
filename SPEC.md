# SPEC — Architecture Contract

This document is the **contract**. Code that violates it is wrong, even if it works.
Read this before writing any module.

---

## 1. The two-layer rule

| Layer | What it does | Rule |
|---|---|---|
| **Deterministic core** (`engine/`) | Ranking, scoring, portfolio construction, backtesting, costs, metrics | **No AI. Ever.** Pure functions. Same inputs → byte-identical outputs. Fully auditable. |
| **Agentic / analytics layer** (later) | Natural-language strategy building, explanation, portfolio X-ray, alerts, reporting | Wraps the core. **Consumes** core output; never decides trades. |

> An LLM must never compute a number that moves money. It may *explain* a number the core computed.

---

## 2. The "full dynamic" rule (no hard-coded wiring)

**A `Market` is a configuration object. The engine reads everything from it.**

The following strings must **never** appear inside `engine/` logic:
`"USD"` · `"INR"` · `"NYSE"` · `"NSE"` · `"S&P 500"` · `"Nifty"` · `".NS"` · any tax rate · any holiday date

A market config carries:

| Field | US | India |
|---|---|---|
| `currency` | USD | INR |
| `calendar` | NYSE | NSE |
| `ticker_suffix` | *(none)* | `.NS` |
| `data_adapter` | yfinance | yfinance (→ India provider later) |
| `benchmark` | `^GSPC` | `^NSEI` |
| `universes` | S&P 500, Nasdaq 100 | Nifty 50/200/500 |
| `costs` | commission + spread | brokerage + **STT** + stamp duty + GST + spread |

**The US/India toggle is nothing more than selecting a market config file.**
Adding a third market later = adding a YAML file, not changing engine code.

---

## 3. The pipeline (every strategy is the same machine)

```
Market config
     ↓
  [DATA]        adapter fetches adjusted prices (splits/dividends handled)
     ↓
  [UNIVERSE]    which securities are eligible on date D
     ↓
  [SIGNAL]      score every security  (momentum_12_2, volar, ...)   ← plug-ins
     ↓
  [RANK]        sort by score (or blended composite of several signals)
     ↓
  [SELECT]      take top N
     ↓
  [WEIGHT]      equal / inverse-vol / capped                        ← plug-ins
     ↓
  [OVERLAY]     risk guardrails: trend filter, cash switch, vol target
     ↓
  [REBALANCE]   on schedule, applying costs + no-trade buffers
     ↓
  [PORTFOLIO]   ← the structured output object
     ↓
  [METRICS]     CAGR, max drawdown, Sharpe, turnover, vs benchmark
```

**Momentum is just the first signal plug-in.** Value, quality, low-vol slot into the same machine
without touching the pipeline. Build the *pipeline*, not the strategy.

---

## 4. Non-negotiable backtest rules

1. **No look-ahead.** At rebalance date D, a signal may only use data with timestamp **< D**.
   Every signal function receives a `as_of` date and must respect it. This is enforced in tests.
2. **Reproducible.** Same config + same data → identical numbers. No randomness without a fixed seed.
3. **Every trade pays a cost.** No cost-free simulated fills. Ever.
4. **Survivorship bias is disclosed.** V0 uses a *current* index membership list. Every backtest
   output must carry the disclaimer. Fix later with point-in-time data.
5. **Config-driven.** A strategy is a YAML file, not code. (This is what later becomes the
   no-code strategy builder.)

---

## 5. The Portfolio object (the seam)

The core's output is a structured object:

```
Portfolio
  ├── market_id
  ├── as_of date
  ├── holdings: [ {ticker, weight, score, rank, sector} ]
  ├── cash_weight
  └── currency
```

**Everything downstream consumes this**: metrics, reporting, the agentic explainer, and the
analytics layer below. Nothing downstream reaches back into the engine's internals.

---

## 6. Analytics & Risk layer (built later — seam exists now)

These sit **on top of** the Portfolio object. They analyse what the engine produced.
They are **not** ranking signals.

| Module | Purpose | Status |
|---|---|---|
| **HEES** — Hidden Economic Exposure Score | Momentum's #1 failure mode is **crowding**: a Top-20 can silently become one big AI/semi/Taiwan bet. HEES multiplies portfolio weights by a company→factor exposure matrix to expose the true bet. Can feed **back** as a construction guardrail ("cap hidden AI exposure at 30%"). | **First analytics module** |
| **Relationship graph** | The grown-up version of HEES: full network of companies ↔ suppliers ↔ macro factors (NetworkX, centrality, communities). | Later evolution of HEES |
| **Earnings propagation** | Event-study of how one company's earnings move connected names. **Alerting only — never a trading signal**, since event-driven trading fights our low-churn design. | Monitoring layer |

---

## 7. Build sequence

- **V0** — pipeline skeleton · `momentum_12_2` + `volar` signals · top-20 equal weight · monthly
  rebalance · trend/cash overlay · bps cost model · core metrics · **US + India both working**
- **Later** — vol-targeting · no-trade buffers · signal blending · walk-forward · HEES ·
  point-in-time universe data
- **Advanced** — turning-point (4-state) dynamic momentum · factor momentum · relationship graph

See `STRATEGIES_AND_RESEARCH.md` for the full menu and the research behind each item.

---

## 8. What this product is (and is not)

- It is **software**: it computes and displays rules-based signals.
- Users connect **their own** brokerage accounts and make **their own** decisions.
- Revenue is a **flat subscription** — never a share of anyone's profits.
- Every backtest output carries a disclaimer. Backtests are not predictions.
