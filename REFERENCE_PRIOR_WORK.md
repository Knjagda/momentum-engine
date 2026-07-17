# Reference: Prior Work (Meta_PortOpt+ / FintaSense)

**What this is.** A distilled map of the logic in Kunal's earlier `Meta_PortOpt+`
codebase — a months-long, notebook-driven body of work that explored the same problem
space this engine tackles systematically. We keep the *ideas* here as a reference to
draw on when we build the matching piece, deliberately and on verified data.

**What this is NOT.** Not code to clone, and not data to trust. The prior work's data
authenticity is unverified (that's the whole reason `momentum-engine` exists — to get
the data right first). So this file records *approaches worth revisiting*, not
implementations to paste. When we build each piece, we re-derive it cleanly on our
point-in-time, survivorship-checked data.

The prior system called itself **FintaSense** and was built around **VQM** — Value,
Quality, Momentum — plus a technical-signal layer and a macro regime model.

---

## The big picture (from the two reports)

The prior system was a full pipeline: S&P 500 universe → score each stock on
Value/Quality/Momentum → select a subset by a portfolio-level *reward* function →
size positions by several allocation styles → generate execution-ready trade plans →
backtest → optionally walk-forward quarterly → persist to SQLite per investor.

Notably, it had already reached several things on our roadmap: a database layer
(SQLite snapshots keyed by portfolio_id), quarterly walk-forward rebalancing, and
turnover-based transaction costs. It confirms the destination; we're building the
trustworthy version of the road.

---

## Ideas worth borrowing, by file

### `sp500_portfolio_final.py` — the cleanest, most relevant file

This is the best-organized piece and maps most directly onto our engine. Key ideas:

**1. Reward-based greedy subset selection (`forward_select_by_reward`).**
Instead of ranking stocks individually and taking the top N, it builds the portfolio
*greedily*: start empty, and repeatedly add whichever stock most improves a
*portfolio-level* reward score, until you have K names. This is meaningfully different
from our current "rank then slice top-20" and worth considering as an alternative
construction mode. The reward (`reward_of_set`) is multi-objective:

```
score =  ALPHA_EXCESS * excess_return_vs_SPY      (1.00)
       + BETA_YIELD   * avg_dividend_yield         (0.10)
       + GAMMA_DIV    * (1 - avg_pairwise_corr)    (0.50)   ← diversification bonus
       - DELTA_DD     * abs(max_drawdown)          (0.50)
       - EPS_TE       * tracking_error_vs_SPY      (0.25)
```

The interesting part is the **diversification bonus**: it explicitly rewards low
average pairwise correlation among holdings. That's a direct, if greedy, attack on the
crowding problem — conceptually adjacent to the HEES idea in our roadmap. Worth
remembering when we build portfolio construction and the HEES layer.

*Caveat to carry:* greedy selection optimizes the reward in-sample over the lookback
window — it's prone to the same overfitting we've been disciplined about. If we adopt
it, it needs walk-forward / out-of-sample validation, not just a good backtest number.

**2. Four allocation styles behind one interface** (`weights_eqw`, `weights_rp`,
`weights_mv`, `weights_max_sharpe`):
- Equal weight.
- Risk parity: `w_i ∝ 1/σ_i` (inverse-vol, normalized). Simple, robust.
- Mean-variance: maximize `μᵀw − λ·wᵀΣw` via cvxpy, with per-name cap (0.12), sector
  cap (0.30, soft), long-only, sum-to-1.
- Max-Sharpe: sweep λ across `logspace(-3, 2, 12)`, solve MV at each, keep the weights
  with the best realized Sharpe.

This is a clean menu we could offer as weighting options. Our engine already has
equal/vol-weighting; risk-parity and the cvxpy-based MV/max-Sharpe are natural
additions *later* (they add a cvxpy dependency and optimization, so V-later, not V0).

**3. Execution plan (`build_execution_plan`).** This is genuinely practical and
something we haven't touched: given target weights and capital, it produces a
trade plan with limit prices (at ask if spread ≤ 2%, else near mid), tranche sizing,
ADV-based daily caps (don't exceed 10% of average daily volume), days-to-fill
estimates, and liquidity flags (spread > 5%, ADV < 10× position, may need > 5
sessions). When we get to "turn a target portfolio into actual orders," this is a
strong template — it encodes real trading hygiene.

**4. `bootstrap_prob_beat_spy`.** Bootstraps the annual excess return to estimate the
*probability* of beating SPY, rather than a point estimate. A nice honesty tool —
fits our "a backtest is an argument" philosophy. Cheap to add to our metrics.

**5. SQLite persistence** (`db_init`, `db_create_portfolio`, `db_save_snapshot`,
`db_load_latest_snapshot`). Portfolio snapshots keyed by `portfolio_id` per investor.
Confirms the staging we discussed: this is *application-layer* state (whose portfolio
is what), separate from the research engine. Reference point for when we build the
multi-tenant app layer.

### `market_direction_dashboard.py` — macro regime model

A `Signals` class that pulls FRED macro series + market prices and computes a basket
of regime indicators, then blends them into a composite "market direction" score with
regime labels. Signals include:
- Yield-curve spread (10y − 2y), real-yield proxy (10y − CPI YoY), M2 YoY.
- ISM PMI level + momentum, unemployment level + 3m change, consumer sentiment.
- Breadth proxy (SPY above its 200-day MA), VIX level + 21-day change.
- USD proxy (UUP) 3m change, crude & copper 3m change.

This is a whole **macro overlay** we don't have. It's conceptually a cousin of our
trend filter, but top-down and multi-signal rather than price-only. Worth revisiting
if we ever want a regime-aware overlay — though note our own finding that overlays are
"insurance with a premium," so any macro overlay must clear the same both-eras bar.
The *data* here (FRED pulls) would need our verification treatment.

### `Finta_Composite_Final_Engine_120325.py` & `tech_signal_codeonly.py` — technical-signal engine

These two (≈4,300 and ≈4,800 lines) are a deep **technical-analysis timing engine** —
far more elaborate than anything in our momentum core. They compute, per ticker:
- RSI with overbought/oversold pullback & rebound episode statistics, entry-band and
  entry-ladder construction, forward-return and forward-drawdown conditional stats.
- Moving-average regime & extension (50/200), "MA magnet" levels, distance-to-MA.
- ATR, Bollinger bands, historical vol, volatility regime classification.
- Fibonacci retracement levels and backtested variants.
- A valuation overlay (P/E, P/B, P/S, P/FCF, dividend yield vs sector medians).
- Per-signal backtest + parameter optimization (`optimize_rsi/ma/vol/fib/val`).
- Even a "lunar" (moon-phase) experiment — interesting, almost certainly noise; a good
  example of the multiple-testing trap we've been careful about.

**How this relates to us — important framing.** This is a **timing / entry-and-exit**
engine, which is a *different problem* from our **cross-sectional momentum ranking**.
Ours asks "which stocks to hold this month"; this asks "at what price to enter a stock
you've chosen." They're complementary, not competing. Most of this is well beyond V0,
and much of it (per-ticker parameter optimization) is exactly where overfitting lives.

What's worth extracting *as concepts*, if/when we add an entry-timing or monitoring
layer on top of the Portfolio object (not inside the ranking core):
- The **entry-ladder** idea (stage entries across a band of prices rather than one
  market order) — pairs naturally with the execution-plan tranching above.
- **Conditional forward-return/drawdown stats** (given an RSI/MA condition, what
  historically happened next) — an honest, distributional way to frame signals.
- The **valuation-vs-sector-median** overlay — relevant to our value screen; a
  reminder to compare cheapness *within sector*, not absolutely.

*Strong caveat:* the per-signal `optimize_*` functions tune parameters on historical
data. That's the multiple-testing / deflated-Sharpe problem our SPEC explicitly warns
about. Borrow the *structure* (conditional stats, laddering), not the optimized
parameters.

### `full_optimizer.py`, `FintaSense_portfolio_vqm_102425_main-Copy.py`, `fintasense_final_v6.py`

Larger orchestration variants tying the VQM scoring, optimization, and reporting
together. Same ideas as above at greater scale, with more Excel-reporting machinery.
The VQM scoring itself (Value/Quality/Momentum composite) is the conceptual heart and
overlaps directly with our `CompositeSignal` — the difference is we compute ours on
verified point-in-time fundamentals, which is the whole point.

### `app.py`, `run_all.py` — orchestration/CLI

Thin drivers that wire the pieces into a runnable flow. Structurally similar to our
`scripts/`. Not much to borrow; our architecture is cleaner.

---

## Cross-cutting takeaways

1. **The destination is validated.** The prior work independently arrived at VQM
   scoring, multi-style allocation, walk-forward rebalancing, turnover costs, and a
   database layer — the same shape as our roadmap. We're building the auditable
   version.

2. **Three genuinely new ideas we didn't have:**
   - Reward-based *greedy portfolio selection* with an explicit diversification bonus
     (crowding-aware) — revisit for construction + HEES.
   - A practical *execution-plan* generator (limit rules, tranching, ADV caps) —
     revisit when we turn portfolios into orders.
   - A *macro regime* model — revisit if we ever want a top-down overlay.

3. **A repeated caution, confirmed.** The prior work leans heavily on in-sample
   parameter optimization (per-signal `optimize_*`, greedy in-sample reward). That's
   the exact overfitting our SPEC guards against. Every idea borrowed from here must
   pass our discipline: point-in-time data, both-eras robustness, out-of-sample /
   walk-forward validation. Borrow the shape, re-earn the number.

4. **Data stays ours.** Nothing from the prior caches/spreadsheets enters the engine.
   Ideas only.
