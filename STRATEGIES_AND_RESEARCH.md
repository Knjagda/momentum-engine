# Strategies & Research Reference

**Purpose:** A curated menu of systematic investing strategies, factors, portfolio-construction methods, risk overlays, and backtesting standards — organized so we can design the engine as a *general* signal-ranking pipeline rather than a one-strategy hard-wire.

**How to read the "Build tier" tags:**
- **[V0]** — build this first; simple and high-value.
- **[Later]** — valuable, but adds real complexity or needs paid data.
- **[Advanced]** — powerful but easy to get wrong; only after the core is solid.

> ⚠️ **Honesty note.** Every strategy below has published historical evidence, but *none is a guarantee.* Factor premia decay, get crowded, and go through long dead periods. A signal that backtests beautifully can still lose money live. This document is a menu of well-studied ideas, not a set of promises — and that framing is exactly what keeps our product on the "software/education" side of the line.

---

## Part 1 — The organizing idea

Almost every strategy in this document fits one abstraction:

> **Score every security in a universe by a signal → rank them → select the top N → assign weights → hold → rebalance on a schedule.**

If we build that pipeline generically, then *every strategy becomes a plug-in signal function.* Momentum is just the first signal. Value, quality, low-volatility, and combinations all slot into the same machine. **This is the single most important design decision for a "robust" engine** — build the pipeline, not the strategy.

Two broad flavors:
- **Cross-sectional (relative):** rank stocks *against each other*, own the best. (Most of this doc.)
- **Time-series (absolute):** judge each asset against *its own* history / against cash. (The basis for trend-following and crash protection.)

---

## Part 2 — The momentum family (our core)

### 2.1 Cross-sectional relative momentum — **[V0]**
The original anomaly. Rank stocks by trailing return; buy past winners, avoid past losers.
- **Research:** Jegadeesh & Titman (1993), *Returns to Buying Winners and Selling Losers* — the paper the pitch deck already cites.
- **Standard lookback:** 3–12 months.
- **Engine role:** our default ranking signal.

### 2.2 The skip-month (12–1) refinement — **[V0]**
Use the return from 12 months ago up to *1 month ago*, skipping the most recent month.
- **Why:** stocks show short-term *reversal* over the last month; skipping it removes noise and improves the signal. This is why "12-1" is the academic default rather than plain "12-month return."

### 2.3 Volatility-adjusted momentum ("Volar") — **[V0]**
Rank by **return ÷ volatility** instead of raw return. Favors steady climbers over jumpy ones.
- **Engine role:** Definedge's signature method and a differentiator in the pitch. Cheap to compute, so it belongs in V0 alongside 12-1.

### 2.4 Risk-adjusted / Sharpe ranking — **[V0]**
Rank by a Sharpe-like measure (excess return per unit of risk) over the lookback. A close cousin of Volar.

### 2.5 Time-series / absolute momentum — **[V0 as a risk switch]**
Own an asset only if *its own* trailing return is positive (or beats cash/T-bills); otherwise go to cash/bonds.
- **Research:** Moskowitz, Ooi & Pedersen (2012), *Time Series Momentum*.
- **Engine role:** this is the mechanism behind the deck's "shift to cash if the market weakens" guardrail. Simple and hugely valuable for drawdown control.

### 2.6 Dual momentum — **[Later]**
Combine both: use *relative* momentum to pick the strongest assets, and *absolute* momentum to decide whether to be invested at all.
- **Research/book:** Gary Antonacci, *Dual Momentum Investing* (2014).
- **Engine role:** a clean, popular framework that packages 2.1 + 2.5 into one rule set.

### 2.7 52-week-high momentum — **[Later]**
Signal = how close a stock is to its 52-week high. Nearness-to-high predicts continuation.
- **Research:** George & Hwang (2004), *The 52-Week High and Momentum Investing*.

### 2.8 Residual / idiosyncratic momentum — **[Advanced]**
Run momentum on returns *after* stripping out market/factor exposure. More stable, lower crash risk than raw momentum.
- **Research:** Blitz, Huij & Martens (2011), *Residual Momentum*.
- **Cost:** needs a factor model first, so it's later.

### 2.9 "Frog-in-the-pan" / path quality — **[Advanced]**
Momentum from many small steady moves continues better than momentum from a few big jumps (information arrives "continuously").
- **Research:** Da, Gutierrez & Warachka (2014), *Frog in the Pan*.

### 2.10 Momentum crashes — **[V0 awareness, Later to fully mitigate]**
Critical risk fact: momentum has rare but *brutal* crashes, typically when a beaten-down market rebounds sharply (e.g., 2009). Past losers rocket up and a "buy winners" book gets run over.
- **Research:** Daniel & Moskowitz (2016), *Momentum Crashes*.
- **Mitigation:** scale exposure down when momentum's own volatility spikes ("dynamic momentum"). Even in V0, we should *disclose* this risk in outputs; full volatility-scaling can come later.

---

## Part 3 — Complementary factors (for a multi-factor, robust engine)

Momentum is one factor. A robust engine supports several, because **factors are imperfectly (sometimes negatively) correlated**, so blending them smooths the ride. Momentum + value in particular tend to zig when the other zags.

- **Value — [Later].** Cheap stocks (low price relative to fundamentals) beat expensive ones over time. Research: Fama & French (1993), HML factor.
- **Quality / profitability — [Later].** Profitable, stable, low-debt firms outperform. Research: Novy-Marx (2013), *The Other Side of Value*; Asness, Frazzini & Pedersen, *Quality Minus Junk*.
- **Low volatility / low beta — [Later].** Lower-risk stocks have historically delivered *better* risk-adjusted returns — an anomaly vs. textbook theory. Research: Frazzini & Pedersen (2014), *Betting Against Beta*; Baker, Bradley & Wurgler (2011).
- **Size — [reference only].** Small caps historically outperformed, but the effect is weak/unstable in recent decades. Research: Fama-French SMB.
- **The canonical model set — [reference].** Fama-French 3-factor (1993) → Carhart 4-factor (1997, *adds momentum*) → Fama-French 5-factor (2015, adds profitability & investment). Good mental scaffolding for what "a factor" means.

> **Engine implication:** design signals so multiple can be **combined** (e.g., rank on a blended z-score of momentum + quality). Build the combination machinery even if V0 ships with only momentum signals turned on.

---

## Part 4 — Portfolio construction & weighting

Ranking picks *what* to hold; construction decides *how much* of each.

- **Equal weight — [V0].** Every holding gets the same %. Simple, robust, hard to break, avoids one mega-cap dominating. Our default.
- **Market-cap weight — [Later].** Weight by size. Lower turnover, but concentrates in the biggest names.
- **Inverse-volatility weight — [Later].** Give calmer stocks bigger weights. A gentle risk control.
- **Risk parity — [Advanced].** Size positions so each contributes *equal risk*. Research: Qian, and others.
- **Volatility targeting — [Later].** Scale total exposure up/down to hit a target portfolio volatility (e.g., 12%/yr). One of the most effective single risk tools; strong pairing with momentum to tame crashes.
- **Mean-variance optimization (Markowitz) — [Advanced, use with caution].** Theoretically optimal, but *extremely* fragile to estimation error — tends to produce crazy, unstable weights out-of-sample. Research: Markowitz (1952); critiques and the **Hierarchical Risk Parity (HRP)** alternative from López de Prado (2016). Prefer simple weighting until much later.

**Constraints worth supporting:** max weight per position, max weight per sector, minimum liquidity / average-daily-volume filter (so you can actually trade it), and turnover caps.

---

## Part 5 — Risk management & regime overlays

These sit *on top of* the strategy and are what turn a fair-weather backtest into something survivable.

- **Trend filter / moving-average overlay — [V0].** Only stay invested while the market (or the asset) is above its long moving average (e.g., 200-day); otherwise de-risk. Research: Faber (2007), *A Quantitative Approach to Tactical Asset Allocation*. Simple, effective, cheap.
- **Absolute-momentum "cash switch" — [V0].** See 2.5 — the direct implementation of the deck's ">15% drawdown → shift to cash" idea.
- **Drawdown guardrails — [V0].** Hard rule: if portfolio drawdown exceeds a threshold, cut exposure. Blunt but reassuring for users.
- **Volatility scaling — [Later].** See Part 4; the more elegant version of drawdown control.
- **Stop-losses — [reference, be skeptical].** Intuitive but, in systematic rules-based strategies, fixed stop-losses often *hurt* returns (they lock in whipsaw losses). Include only if a backtest earns it — don't assume.

---

## Part 6 — Rebalancing & implementation frictions

Where paper returns quietly leak away in real life.

- **Frequency — [V0].** Monthly is the standard sweet spot: captures momentum cycles without excessive churn. Weekly = more turnover, more tax, usually worse *after costs*. Quarterly = cheaper but slower to react.
- **Calendar vs. threshold rebalancing — [Later].** Rebalance on a fixed date vs. only when weights drift past a band. Threshold cuts needless trades.
- **No-trade buffers / ranking bands — [Later].** Don't sell a holding the instant it slips from rank 20 to 22; use a buffer (e.g., hold until it drops below rank 25). Dramatically cuts turnover with little performance loss.
- **Transaction costs, slippage, tax drag, wash sales — [V0 to model, at least crudely].** Every simulated trade *must* pay a cost. Ignoring this is the #1 way backtests lie.

---

## Part 7 — Backtesting rigor (non-negotiable, CFA-grade)

This is the part that separates a trustworthy engine from a fantasy generator. These are the ways a backtest fools you:

- **Survivorship bias — [V0 to disclose, Later to fully fix].** Testing today's index members against the past ignores every company that failed or was removed → inflated returns. The real fix is *point-in-time* index membership data (paid, fiddly). V0 ships with a fixed universe **and a loud disclaimer**.
- **Look-ahead bias — [V0, must prevent].** Using any data you wouldn't have had on the decision date. Our hard rule: at each rebalance, rank using *only* data up to that date.
- **Data-snooping / multiple testing — [V0 awareness].** If you try 300 strategies, some look great by pure luck. Research: Harvey, Liu & Zhu (2016), *…and the Cross-Section of Expected Returns* — a large share of "discovered" factors are likely false positives.
- **In-sample vs. out-of-sample / walk-forward — [Later].** Tune on one period, validate on another you never looked at. The deck's "walk-forward testing" promise lives here.
- **Deflated Sharpe ratio — [Advanced].** Adjusts a strategy's Sharpe downward for how many variations you tried. Research: Bailey & López de Prado (2014). Good honesty tool once we're testing many configs.
- **Regime dependence / non-stationarity — [awareness].** Markets change; a strategy that worked 2003–2013 may not 2013–2023. Always inspect performance *by sub-period*, not just the headline number.

---

## Part 8 — Metrics the backtester must output

Return, risk, and relative-to-benchmark — the deck already shows most of these.

- **Return:** CAGR, total return, yearly return table.
- **Risk-adjusted:** Sharpe (return per unit of total risk), Sortino (penalizes only downside), Calmar (return ÷ max drawdown), Information Ratio (vs. benchmark).
- **Risk:** annualized volatility, **max drawdown** (the number users feel most), downside deviation; optionally VaR / CVaR later.
- **Relative:** alpha, beta, tracking error, up-capture / down-capture vs. S&P 500.
- **Practical:** turnover (drives cost & tax), win rate (% profitable months), and — importantly — **after-cost and, later, after-tax** returns.
- **Professional standard:** the **GIPS** (Global Investment Performance Standards) are the industry reference for *honest* performance reporting — a good north star even though we're not claiming compliance.

---

## Part 9 — Honest caveats to bake into the product

- Factor premia **decay and get crowded** once widely known.
- Backtests are **best-case fiction** until proven with out-of-sample and live results.
- Every strategy has **long painful stretches** of underperformance — users quit at exactly the wrong time.
- More strategies tested = higher odds one looks good **by luck**.
- **Disclaimers on every backtest output** aren't just legal cover; they're intellectual honesty, and they're core to staying a "software/education" product.

---

## Part 10 — Recommended build sequence (ties back to Phase 0)

Design the **general pipeline** first, then light up signals one at a time:

1. **Pipeline skeleton** — data → universe → signal → rank → select top N → weight → rebalance → metrics. (Generic; strategy-agnostic.)
2. **First signals [V0]** — 12-1 cross-sectional momentum, and Volar (return/vol).
3. **First construction [V0]** — Top 20, equal weight, monthly rebalance.
4. **First risk overlay [V0]** — absolute-momentum / trend "cash switch" for crash protection.
5. **Cost model [V0]** — per-trade basis-points on every simulated trade.
6. **Metrics + honest report [V0]** — CAGR, max drawdown, Sharpe, win rate, vs. S&P 500, with survivorship disclaimer.
7. **Then [Later]:** more signals (value, quality, low-vol), signal blending, volatility targeting, no-trade buffers, walk-forward, point-in-time data.

Everything in tiers **[Later]** and **[Advanced]** is a *feature we grow into* — the architecture in step 1 just needs to leave room for them.

---

### Key references (verify exact parameters against primary sources when implementing)
- Jegadeesh & Titman (1993) — cross-sectional momentum
- Moskowitz, Ooi & Pedersen (2012) — time-series momentum
- Daniel & Moskowitz (2016) — momentum crashes
- Antonacci (2014) — dual momentum
- George & Hwang (2004) — 52-week high
- Blitz, Huij & Martens (2011) — residual momentum
- Da, Gutierrez & Warachka (2014) — frog-in-the-pan
- Fama & French (1993, 2015); Carhart (1997) — factor models
- Novy-Marx (2013); Asness, Frazzini & Pedersen — quality
- Frazzini & Pedersen (2014); Baker, Bradley & Wurgler (2011) — low risk
- Faber (2007) — trend/tactical overlay
- Harvey, Liu & Zhu (2016) — multiple-testing problem
- Bailey & López de Prado (2014) — deflated Sharpe; López de Prado (2016) — HRP
- Markowitz (1952) — mean-variance; Qian — risk parity

---

## Part 11 — Incremental research addendum (2023–2026 literature scan)

New, underrated, and recently-verified threads found in a fresh scan of the literature. Same **[V0]/[Later]/[Advanced]** tiering. Parameters below were checked against the source papers; still verify against the primary PDF before coding.

### 11.1 Dynamic / turning-point momentum — the "four-state" model — **[Later]**
The single most interesting recent idea for us. Instead of one fixed lookback, run **two speeds** — a *slow* signal (e.g., 12-month) and a *fast* signal (e.g., 1-month) — and classify each asset (or the whole market) into four states by whether the two agree:
- **Bull** (both up), **Bear** (both down) → strong trend, lean in.
- **Correction** (slow up, fast down) and **Rebound** (slow down, fast up) → *turning points*, where trend-following places its worst bets.
- Rule of thumb from the research: **slow down after Corrections, speed up after Rebounds.** Intermediate-speed blends beat pure-slow or pure-fast on Sharpe, drawdown, and skew.
- **Research:** Goulding, Harvey & Mazzoleni (2023), *Momentum Turning Points*, Journal of Financial Economics; extended to 43 futures markets in *Breaking Bad Trends* (2024).
- **Engine role:** a smarter successor to our V0 cash-switch. The state classification is cheap once we already compute a slow and a fast momentum signal — so the *plumbing* fits our pipeline naturally even if we enable it later.

### 11.2 Volatility-scaled / risk-managed momentum — **[Later, high value]**
Momentum's *own* risk is time-varying and predictable from its recent volatility, and high-volatility periods precede crashes. Fix: scale position size by **(target volatility ÷ recent realized volatility)**.
- **Verified parameters:** realized vol estimated over ~**6 months** of daily momentum returns; constant **target ≈ 12% annualized**. In the original study this lifted the Sharpe from roughly **0.53 to ~0.97** and "virtually eliminated" crashes.
- **Research:** Barroso & Santa-Clara (2015), *Momentum Has Its Moments*, JFE (constant-vol scaling); Daniel & Moskowitz (2016) add a *dynamic* version that also uses forecast return.
- **Honest caveat:** the 12% target is somewhat arbitrary (it sets your risk level, not your Sharpe) — treat it as a **tunable config knob**, not a magic number.

### 11.3 Multidimensional / composite momentum — **[Later; architect for it now]**
Combining price momentum with ~10 *alternative* momentum signals (equal-weighted composite) beats price momentum alone on both return and risk-adjusted metrics, across 150 years and 46 countries. Also documents that traditional price momentum has suffered drawdowns as deep as **~ -88%** — a sober reminder of crash risk.
- **Research:** Baltussen, van Vliet, Dom & Vidojevic (2025), *Momentum Factor Investing: Evidence and Evolution* (forthcoming, Journal of Portfolio Management).
- **Engine role:** direct validation of our "many signals, blended" pipeline. Build the *composite/blending machinery* from day 1 even if V0 ships with only two signals lit up.

### 11.4 Factor momentum — **[Advanced; conceptually important]**
A provocative, underrated finding: momentum in individual stocks may largely **emanate from momentum in factor returns.** Most factors are positively autocorrelated — the average factor earned roughly **6 bps/month after a down year vs. ~51 bps after an up year.** One interpretation: stock momentum "times" other factors rather than being a standalone factor.
- **Research:** Ehsani & Linnainmaa (2021/2022), *Factor Momentum and the Momentum Factor*, Review of Financial Studies.
- **Honest debate:** later international studies (51 countries, 145 anomalies) find factor momentum does **not** fully subsume price momentum — so treat this as an important lens, not settled law.
- **Engine role:** once we have several factors, we can run momentum *on the factors themselves* — a natural extension of the same pipeline.

### 11.5 Momentum at long holding periods — **[Later]**
Confirms the academic standard is **12-2** momentum (rank on cumulative return from month t-12 to t-2, i.e. skip the most recent month to dodge short-term reversal) and studies *holding longer* to cut turnover and costs for bigger portfolios — directly relevant to our no-trade-buffer idea.
- **Research:** Calluzzo, Moneta & Topaloglu (2025), *Momentum at Long Holding Periods*.

### 11.6 Machine-learning trend (frontier — awareness only) — **[Not for us yet]**
Deep "momentum networks," change-point detection, and CNN-on-price-charts approaches (Wood/Roberts/Zohren; Jiang, Kelly & Xiu, *(Re-)Imag(in)ing Price Trends*) can outperform but are heavy, opaque, overfitting-prone, and **hard to audit** — the opposite of our deterministic-and-explainable principle. Flagged so we know it exists; not a fit for the core engine.

### Verified-parameter quick reference
| Concept | Verified setting | Note |
|---|---|---|
| Academic momentum lookback | **12-2** (skip most recent month) | practitioners also use 12-1 / 6-1 — make it config |
| Volatility scaling | ~**6-month** realized vol, **~12%** annualized target | target is tunable; doesn't change Sharpe |
| Turning-point states | slow = **12mo**, fast = **1mo** | four states: Bull/Correction/Bear/Rebound |
| Factor autocorrelation | ~6 bps (post-down-year) vs ~51 bps (post-up-year) | basis of factor momentum |

---

## Part 12 — India market evidence & the multi-market ("full dynamic") design

**Requirement:** the engine serves **US + India from day 1** via a toggle. This section records what the research says differs between the two markets, and turns "no hard-coded wiring" into a concrete design rule.

### India evidence (momentum works, often *stronger*)
- Official NSE momentum indices already exist and have beaten the broad market: **Nifty 200 Momentum 30** and **Nifty Alpha 50**, generally outperforming the Nifty 500 over most measured periods.
- A published long-only test on the **NIFTY100** (top-decile, monthly rebalance) outperformed the index by roughly **+10.7% per year** — but with **~32% monthly turnover** (much higher than typical US momentum). Source: Raju & Chandrasekaran (2020), *Implementing a Systematic Long-only Momentum Strategy: Evidence From India*.
- The momentum premium appears **stronger in mid/small caps** and in the **broader Nifty 500** than in the Nifty 50.
- **Quality-momentum (Q-Mom)** and low-volatility-tilted momentum showed **lower drawdowns** in Indian backtests — a good India-flavored composite to support later.

### The "full dynamic" rule (this is the architecture takeaway)
Because India differs from the US on nearly every practical axis, **a `Market` is a configuration object, and the engine reads everything from it — never from hard-coded constants.** A market config carries:

- **Universe:** US = S&P 500 / Nasdaq 100 / sector ETFs. India = Nifty 50 / 200 / 500 / Midcap.
- **Data adapter + ticker convention:** US = plain tickers. India = NSE `.NS` / BSE `.BO` suffixes (or an India-specific data provider later).
- **Currency:** USD vs **INR** — every price, cost, and equity-curve number must carry a currency; nothing assumes dollars.
- **Trading calendar & holidays:** NYSE vs **NSE** calendars differ; rebalance dates must snap to the correct exchange's calendar.
- **Cost & tax model:** US = ~zero commission + spread. India = brokerage + **STT (Securities Transaction Tax)** + stamp duty + GST + wider spreads on small caps. Given ~32%/month turnover, **honest India cost/tax modeling matters even more than in the US** — a strategy that looks great gross can bleed out after STT and short-term capital-gains tax.
- **Benchmark:** S&P 500 vs Nifty 50 / Nifty 500.

**The toggle is just "select a Market config."** No `"USD"`, `"NYSE"`, or `"S&P 500"` string is ever written into engine logic. That is exactly the "full dynamic, no hard-coded wiring" design — and it's also what makes adding a *third* market later (Europe, etc.) a config file rather than a rewrite.

---

### Added references (Part 11–12)
- Goulding, Harvey & Mazzoleni (2023) — *Momentum Turning Points*, JFE; and *Breaking Bad Trends* (2024)
- Barroso & Santa-Clara (2015) — *Momentum Has Its Moments*, JFE (constant-vol scaling)
- Baltussen, van Vliet, Dom & Vidojevic (2025) — *Momentum Factor Investing: Evidence and Evolution*
- Ehsani & Linnainmaa (2021/2022) — *Factor Momentum and the Momentum Factor*, RFS
- Calluzzo, Moneta & Topaloglu (2025) — *Momentum at Long Holding Periods*
- Wood, Roberts & Zohren; Jiang, Kelly & Xiu — ML trend / price-image approaches (awareness only)
- Raju & Chandrasekaran (2020) — *Systematic Long-only Momentum: Evidence From India*
- NSE Indices — Nifty 200 Momentum 30, Nifty Alpha 50 (official Indian momentum indices)
