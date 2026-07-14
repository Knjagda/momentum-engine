# Momentum Engine

A deterministic, auditable, **multi-market** momentum investing engine.

Build a strategy → backtest it honestly → get a decision → execute in your own account → track results.

**Markets supported from day 1: 🇺🇸 US and 🇮🇳 India.** Switching between them is a config toggle,
not a code change.

---

## What this is

**Software.** It computes rules-based signals and shows you what a strategy would have done.

**What it is not:** it does not manage anyone's money, give personalised advice, or charge a share
of anyone's profits. Users connect their own brokerage accounts and make their own decisions.
Every backtest output carries a disclaimer — **backtests are not predictions.**

---

## The one big idea

> Score every security → rank → select top N → weight → hold → rebalance on schedule.

Momentum is just the **first plug-in signal**. Value, quality, and low-volatility slot into the same
machine without touching the pipeline. We build the *pipeline*, not one strategy.

---

## Repo structure

```
momentum-engine/
├── SPEC.md                     ← THE ARCHITECTURE CONTRACT. Read this first.
├── STRATEGIES_AND_RESEARCH.md  ← the research menu (what to build, and why)
│
├── config/
│   ├── markets/                ← us.yaml, india.yaml  (currency, calendar, costs, universes)
│   └── strategies/             ← a strategy is a YAML FILE, not code
│
├── engine/                     ← THE DETERMINISTIC CORE. No AI here. Ever.
│   ├── markets/                  market config loader
│   ├── data/                     price data adapters
│   ├── universe/                 who is eligible on date D
│   ├── signals/                  momentum_12_2, volar, ...   ← plug-ins
│   ├── ranking/                  sort / blend scores
│   ├── portfolio/                select top N, assign weights
│   ├── costs/                    every trade pays. no exceptions.
│   ├── backtest/                 walk history with NO look-ahead
│   ├── metrics/                  CAGR, max drawdown, Sharpe, turnover
│   └── analytics/                HEES portfolio X-ray (built later)
│
├── tests/                      ← the no-look-ahead rule is enforced here
└── scripts/                    ← save.sh (add + commit + push)
```

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                             # should pass (no tests yet — that's expected)
```

---

## The two rules that govern everything

**1. Deterministic core, agentic edges.**
The engine computes numbers. An LLM may *explain* a number the engine computed — it may never
*compute* a number that moves money.

**2. No hard-coded market wiring.**
`"USD"`, `"NYSE"`, `"S&P 500"`, `".NS"`, any tax rate — none of these appear in `engine/` logic.
A market is a config object. Adding a third market is a YAML file, not a rewrite.

See `SPEC.md` for the full contract.

---

## Status

**Phase 0 — scaffold.** Structure and contracts in place. Engine modules next.
