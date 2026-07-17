# Decisions Log

A running record of foundational calls and *why* they were made, so future-us (and
any collaborator) can see the reasoning, not just the outcome. Newest first.

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
