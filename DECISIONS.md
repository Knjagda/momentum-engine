# Decisions Log

A running record of foundational calls and *why* they were made, so future-us (and
any collaborator) can see the reasoning, not just the outcome. Newest first.

---

## 2026-07 - PARAMETER SENSITIVITY: broad plateau, not a spike (research phase closed)

**The last overfitting check.** Swept lookback [6,9,11,12,13,15] x skip [0,1,2] x
topN [10,15,20,25,30] = 90 cells, on survivorship-free data, trend-filtered,
2005-2026. Question: is our 12-1-top20 choice special (overfit) or typical among its
neighbours (robust)?

**Result: ALL 90 cells beat SPY (100%).** Excess +2.29% to +17.81%, median +8.66%,
Sharpe 0.62-0.93. There is no parameter neighbourhood where momentum fails -- the edge
is a property of the strategy family, not a lucky pick.

**Our (12,1,20) sits at the 36th percentile -- BELOW median.** We did not cherry-pick
a hot cell; if anything our choice is conservative (64% of neighbours scored higher).
This rules out implicit tuning to a peak.

**Slices are smooth (plateau signature):**
- by lookback (skip1,top20): 9.5 / 10.5 / 9.5 / 7.5 / 7.8 / 7.3% -- no cliffs
- by topN (lb12,skip1): 10.4 / 8.8 / 7.5 / 7.7 / 6.4% -- smooth

**Structural patterns worth noting (NOT acting on):** shorter lookbacks and smaller
topN score higher. Concentration lifts return (top10 > top30). BUT:
- The sweep measures RETURN, not risk-adjusted outcome; concentration and short
  lookbacks carry more volatility, turnover, and cost that the excess column
  understates.
- Switching to the best cell NOW would be in-sample optimization through the back
  door -- the exact overfitting this test exists to detect. It would require a fresh
  out-of-sample test to trust. We do NOT do this.

**Conclusion.** 12-1-top20 stays. Being unremarkable among 90 winners is the
validation. Momentum has now survived: inclusion-bias correction, point-in-time
membership, survivorship-free prices, honest-drawdown measurement, out-of-sample
consistency, AND parameter robustness. The RESEARCH phase is substantially complete;
the standing estimate (~+5% vs SPY forward, -34% DD, multi-year lag stretches) is as
honest as this engine can currently make it. Next frontier is the PRODUCT layer
(live signals, accounts), not more strategy research.

---

## 2026-07 - OUT-OF-SAMPLE VALIDATION: the edge holds (the one test that could only deflate it)

**Why this test is different.** Every prior result was measured on the same history we
built against. This asks whether the FIXED strategy (12-1, top 20, trend-filtered --
nothing tuned here) is CONSISTENT across sub-periods it was not selected for. Unlike
inflation-prone backtests, an OOS consistency test can only expose weakness, not
manufacture strength. It held.

**Split-sample (survivorship-free, sp900_pit, 2005-2026):**

| half | period | CAGR | SPY | excess | Sharpe | max DD |
|---|---|---|---|---|---|---|
| first | 2005-03..2015-11 | 14.5% | 5.3% | +9.23% | 1.00 | -16.9% |
| second | 2015-11..2026-07 | 17.4% | 12.5% | +4.92% | 0.68 | -34.5% |

Edge present in BOTH halves -- not a regime artifact. It DECAYS second-half
(+9.2%->+4.9%), consistent with a real, publicly-known factor getting more crowded
over time rather than a fitted fluke.

**Per-year:** beat SPY in 12/22 years (55% hit rate). Cumulative annual excess +193pt;
dropping the single best year still leaves +118pt -- the edge is DISTRIBUTED across
years, not one lucky bounce (this was the main overfitting worry, and it failed to
materialise).

**Rolling 3-year:** 222 windows, beats SPY in 58%, median +1.8%/yr, WORST -12.2%/yr.

**Honest tempering -- this is not a green light:**
- 55% yearly hit rate is only modestly better than a coin flip; the edge is in
  magnitude (wins bigger than losses), not frequency. Multi-year lag stretches (2014,
  2019, 2023, 2025) are normal and a real investor will feel them.
- Worst 3-year window trails SPY by ~12%/yr -- a badly-timed client could underperform
  for years, exactly the person most likely to quit at the bottom. Must be disclosed.
- Second-half decay means the FORWARD edge is better estimated by the recent ~+5% than
  the full-sample ~+9%.

**Standing conclusion.** After inclusion-bias correction, point-in-time membership,
survivorship-free prices, honest-drawdown measurement, AND out-of-sample consistency,
a real distributed edge survives: **~+5% vs SPY forward-looking, -34% max drawdown,
with multi-year underperformance stretches that must be tolerated.** Modest, bounded,
and defensible -- which is more than most retail quant projects can claim. This is NOT
"just buy SPMO"; it is a genuine, honestly-caveated edge.

---

## 2026-07 - REVERSAL: adopt the trend filter as default (honest data changes the call)

**This reverses an earlier decision.** In prior sessions we evaluated risk overlays
on survivors-only data, found them to be "insurance with a premium" (cut drawdown but
cost return), and declined to adopt any. On survivorship-free data that judgement no
longer holds.

**Evidence** (survivorship-free, `sp900_pit`, momentum 12-1 top 20, 2005-01 on):

| overlay | CAGR | vs SPY | max DD | Sharpe | Sortino |
|---|---|---|---|---|---|
| always_on | 21.09% | +6.06% | **-64.9%** | 0.71 | 1.24 |
| trend_filter | 19.71% | +4.68% | **-34.0%** | 0.82 | 1.49 |
| absolute_momentum | 20.07% | +5.05% | -46.6% | 0.75 | 1.34 |

**The trend filter trade:** costs 1.38pt of CAGR, buys back 30.9pt of drawdown
(-64.9% -> -34.0%), AND improves both risk-adjusted metrics (Sharpe 0.71->0.82,
Sortino 1.24->1.49).

**Why the verdict flips - two reasons, same direction.**
1. On survivors-only data the overlay mostly moved risk and return together (pure
   insurance). On honest data it improves return-PER-UNIT-RISK - a strictly better
   way to hold the strategy, not a reluctant hedge.
2. Threshold effect. -64.9% turns $100k into $35k; almost nobody holds through that,
   so the ungated 21% CAGR is partly fictional - it assumes a behaviour (not
   capitulating at the bottom) that does not happen in practice. -34.0% ($100k ->
   $66k) is survivable. An overlay that keeps the strategy actually holdable is worth
   more than its CAGR cost implies.

**Decision.** Adopt `trend_filter` (200-day MA on the benchmark) as the DEFAULT
overlay. `always_on` remains available for research/attribution; `absolute_momentum`
is a middle option but is dominated by trend_filter on every risk metric here.

**Why the earlier call was not wrong - it was under-informed.** The prior "don't
adopt" was correct on survivors-only evidence. The lesson is not "we erred" but
"survivorship bias distorts RISK decisions, not just return estimates" - it made a
worthwhile overlay look not worth it. This is a second, independent cost of
survivorship bias beyond the 1.31% return illusion.

**Standing summary, updated.** Momentum 12-1, top 20, monthly, trend-filtered, on
survivorship-free `sp900_pit`, 2005-2026: ~+4.7% vs SPY, -34% max DD, Sharpe 0.82.
Still mid-cap-inflated and not out-of-sample validated - but now with a drawdown a
real investor could plausibly survive.

---

## 2026-07 - CRISIS-ERA RUN: the survivorship illusion nearly DOUBLES (0.77% -> 1.31%)

**Why we ran it.** The first honest backtest started 2010-06 and measured a 0.77%/yr
survivorship illusion. We flagged that as a LOWER BOUND, because the window excluded
2008-09 - the era when index members actually failed en masse. Momentum needs no
fundamentals, so the EDGAR 2009 wall does not block an earlier price-only run.

**Result.** Same strategy, window extended to 2005-01-01 (prices from 2003):

| window | survivors-only | survivorship-free | illusion | max DD (survivors -> honest) |
|---|---|---|---|---|
| 2010-06 on | +8.74% | +7.97% | **0.77%/yr** | -36.9% -> -37.8% (0.9pt) |
| 2005-01 on | +7.37% | +6.06% | **1.31%/yr** | -61.0% -> -64.9% (3.9pt) |

(SPY over the longer window: 10.93% CAGR, -52.9% max DD - versus 15.13% CAGR from
2010, confirming how exceptional the post-crisis decade was.)

**The illusion nearly doubles once the crisis is included.** The lower-bound reading
we recorded was correct.

**The more important finding is the DRAWDOWN, not the return.** Including failures
worsened max drawdown by 0.9pt in the calm window but 3.9pt in the crisis window.
Survivorship bias does not merely flatter returns - it HIDES RISK, and it hides risk
most precisely when risk matters most. An honest -64.9% max drawdown is the number
that would actually end a strategy in real life, because almost nobody holds through
it. Any presentation of this engine must lead with that, not with CAGR. (This is also
the momentum crash of the Daniel & Moskowitz literature showing up in our own data.)

**1.31% is STILL understated, in a knowable direction.** The 94 dead names we could
not price skew heavily toward 2008 casualties: LEH (Lehman), FNM/FRE (Fannie/Freddie),
CFC (Countrywide), ABK (Ambac), BS (Bethlehem Steel), MEE (Massey), ANR (Alpha
Natural). The crisis-era test is most handicapped exactly where it matters most, so
the true full-cycle illusion is somewhere ABOVE 1.31%.

**Honest standing estimate of the strategy.** Over 2005-2026 with survivorship-free
prices: ~+6% vs SPY, with a -64.9% max drawdown, on `sp900_pit` (mid-caps, which
inflate it) and with no out-of-sample validation. That is the most truthful summary
the engine can currently produce.

**Minor data note:** yfinance returned AES as a failed download in this run but not
the previous one (1,199 vs 1,200 priced). A vendor flake, not a code issue, but a
reminder that yfinance is nondeterministic at the edges.

---

## 2026-07 - THE HONEST BACKTEST: survivorship was worth 0.77%/yr, not the whole edge

**Result.** Same momentum strategy (12-1, top 20, monthly, `sp900_pit`, 2010-06 on),
run on two price sets:

| price set | CAGR | vs SPY | max DD | Sharpe | Sortino |
|---|---|---|---|---|---|
| survivors only (yfinance, 1,200 names) | 23.87% | +8.74% | -36.9% | 0.84 | 1.61 |
| survivorship-free (merged, 1,512 names) | 23.10% | +7.97% | -37.8% | 0.82 | 1.52 |

**The survivorship illusion: +0.77% per year.** Every metric moved the correct
direction once failures were included (return down, drawdown deeper, Sharpe and
Sortino down) - internally consistent, which is itself evidence the run is sound.

**How we got survivorship-free prices for free.** yfinance cannot price delisted
names. Tiingo's free tier can, but caps unique symbols per month - so we used each
vendor for its strength (Option C): yfinance for the 1,192 living names, Tiingo for
the dead ones, merged with Tiingo authoritative on any overlap. 320 of 414 dead names
obtained; 94 are absent from Tiingo under those tickers.

**Why the illusion is smaller than expected - a mechanism, not luck.** Momentum
SELLS losers. A company sliding toward failure loses momentum and is dropped at the
next rebalance, usually well before it dies, so the strategy rarely holds a disaster
through its collapse. This is the mirror image of the value finding: a value screen
BUYS falling knives (it would have loaded up on SIVB and FRC precisely because they
looked cheap before zero). Survivorship bias is therefore far more dangerous for
value than for momentum - which retroactively strengthens the decision not to adopt
the value gate.

**What this does NOT establish.** We answered one question only. The remaining +7.97%
is NOT a validated edge:
- Universe: `sp900_pit` includes mid-caps, which inflates it. Our honest `sp500_pit`
  baseline was **+4.37%**.
- Period: 2010-06 onward excludes 2008-09, the era when index members failed en
  masse. Across a full cycle survivorship would likely cost more than 0.77%, so treat
  this as a LOWER BOUND measured in a favourable stretch.
- No out-of-sample / walk-forward validation has been done.

**Correct reading:** "survivorship accounted for less of our excess than feared,"
NOT "we have an 8% edge."

**Also fixed en route (all committed):** a recycled-ticker hazard - a ticker is a
slot, not an identity (Dean Foods was DF; another company holds DF now). A scan of
all 320 cached dead names found 1 genuine splice inside a traded window, ~5
successor-entity cases, and 36 no-data cases; the rest decomposed into
membership-metadata placeholders and Tiingo history truncation. `eligible_universe`
gained two guards with regression tests: `history_gap` (the recent window must be
contiguous, so a splice cannot be ranked across a dormancy) and `stale_prices` (the
newest bar must be recent, so a dead name cannot be ranked on years-old prices).

**Known data-quality issue, not yet addressed:** many membership records carry a
placeholder `2012-01-13` join date rather than the real one. Harmless to correctness
(the filters catch the consequences) but it means we sometimes believe a company was
investable before it existed.

---

## 2025 — Value gate not adopted; survivorship-free PRICES are now the binding constraint

**Decision.** The value screen (positive earnings AND price-to-book ≤ median) is **not
adopted as a default**. It remains available as an optional screen, but it does not earn
a place in the standard pipeline.

**Evidence.** Re-ran the value experiment on the *corrected* equity data (after the
tag-priority fix that fixed book values for ~100 names), `sp900_pit`, 2010+:

| config | CAGR | vs SPY | max DD | Sharpe | Sortino |
|---|---|---|---|---|---|
| momentum only | 23.87% | +8.74% | −36.9% | 0.84 | 1.61 |
| momentum + value | 17.47% | +2.34% | −37.4% | 0.89 | 1.13 |

The value gate **cut CAGR by 6.4%**, nudged Sharpe up marginally (0.84→0.89), did **not**
improve drawdown (slightly worse), and **cut Sortino** (1.61→1.13). Clean fundamentals
did not rescue value — the earlier "inconclusive" now reads "not worth it, on this data."

**The bigger finding — survivorship-free prices are now the weakest link.** This run
made the price-side hole vivid: **410 of 1606 universe names (~25%) could not be priced
by yfinance** — every delisted company, including LEH, SIVB, FRC, FNM/FRE, CFC, CELG,
ATVI, XLNX, WFM, HNZ. The point-in-time universe *correctly includes* these (that part
works); yfinance simply cannot price a dead ticker. So every backtest silently runs on
survivors only.

This biases the value result **specifically and in the flattering direction**: a value
gate buys cheap stocks, and dying companies get cheap right before zero (SIVB, FRC, LEH
were all "cheap by P/B" shortly before collapse). The value portfolio would have bought
them; we can't price the collapse; so the value line **never takes those losses**. The
value CAGR above is therefore optimistic by an unknown amount. Momentum-only is cleaner
(momentum sells falling names) but still misses all 410.

**Consequence.** No fundamentals work can fix this — it is a *price data* problem. After
several sessions making fundamentals trustworthy, PRICES are now the binding constraint
on backtest honesty. Every excess-return number (incl. momentum's +8.74%) is inflated by
an unknown survivorship amount.

**Revisit / next lever:** buy survivorship-free price data (Sharadar ~$50 one-month pull,
or Norgate) so dead tickers get priced through their decline. That single step would make
every backtest honest and would let us re-judge value on data that includes the failures
it would have bought. Until then, all excess-return figures carry a stated upward bias.

---

## 2025 — Fundamentals source: keep EDGAR default, retain SEC bulk as validated alternative

**Decision.** The engine's default fundamentals adapter stays **per-company EDGAR**
(`get_fundamental_adapter("edgar")`). The **SEC bulk** adapter
(`get_fundamental_adapter("sec_bulk")`) is kept as a fully-built, tested alternative,
not made the default.

**Why we built the bulk pipeline at all.** To fix a suspected coverage gap in the
per-company fetch (which had silently missed live companies via ticker→CIK errors),
and to get a natively point-in-time fundamentals source.

**What we actually found — and it was not what we expected.** Building the bulk source
and comparing it against EDGAR *number by number* (`scripts/compare_fundamentals.py`)
surfaced two real bugs that were silently corrupting equity for ~100 S&P 500 names in
**both** sources:

1. **Segment-leak (bulk parser).** Real SEC `num.txt` writes an empty `segments`
   field as the string `'nan'`, and puts equity-component breakdowns
   (`EquityComponents=RetainedEarnings`, etc.) in that field. The parser's `== ""`
   filter matched almost nothing, so breakdown rows leaked in and `as_of()` sometimes
   picked a component instead of the company-wide total. Fixed in `sec_bulk_parse.py`.

2. **Tag-priority non-determinism (BOTH adapters).** Companies report equity under two
   tags — `StockholdersEquity` (parent-only) and
   `StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest` (total,
   incl. minority interest) — differing by billions. Neither adapter had a
   deterministic rule for which wins, so the choice was incidental ordering. Fixed by
   giving BOTH the same rule: prefer the tag earlier in `CONCEPT_TAGS` (parent-only),
   per-period, so tag-switch history stays unbroken.

**Result.** After both fixes, the sources agree on equity for **518 of 521** checked
companies (was ~420). The remaining 3 are edge cases, not the systematic bug:
- `STI` (SunTrust, merged into Truist 2019) — both ≈ −0.0B, a rounding display artifact.
- `EQIX` (REIT) — ~1.6% gap, likely period-timing (one source sees a newer quarter).
- `LW` (Lamb Weston) — larger gap, likely its unusual May fiscal-year-end. Noted, not chased.

**Why EDGAR stays default despite all this.** The original hypothesis — that bulk would
*win on coverage* — was **wrong**. Final coverage: EDGAR **632/874 (72%)** vs bulk
**527/874 (60%)**. EDGAR maps 106 names bulk misses, because bulk's CIK→ticker map is
SEC's *current* list and can't name delisted tickers (a mapping-side survivorship
flavour). Coverage is the thing we most need, so EDGAR wins on the axis that matters.

**What the bulk effort was worth.** It paid for itself not by replacing EDGAR, but by
being the independent second source that *caught two real bugs* and *validated EDGAR's
correctness*. It also remains the faster source (local reads vs thousands of throttled
HTTP calls) and the natively point-in-time one — useful for fast iteration and as an
ongoing correctness check. Both are correct now; we default to the one with more
coverage.

**Revisit if:** we buy survivorship-free data with historical ticker maps (then bulk's
coverage limitation disappears and speed could tip it to default), or if the 3
stragglers turn out to matter for a specific screen.

---
