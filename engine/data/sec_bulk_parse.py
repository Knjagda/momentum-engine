"""
SEC bulk parser (piece 2 of 3): a cached quarter ZIP -> clean, point-in-time facts.

Input:  one quarterly ZIP (from piece 1) containing sub.txt + num.txt.
Output: a tidy DataFrame in our standard FACT_COLUMNS shape
        [symbol, concept, period_end, filed, value, form, fiscal_year],
        ready to hand straight to FundamentalData.

The three real problems this solves, none of them glamorous:

1. VOLUME. num.txt holds ~3.6M rows per quarter -- almost all of them tags we never
   use (lease schedules, tax footnotes, per-segment breakdowns). We keep only the
   ~10 concepts in CONCEPT_TAGS. That is a >99% cut, and it is why the stored data
   ends up small enough to be a plain file, not a database (yet).

2. IDENTITY. sub.txt identifies filers by CIK; our universe speaks in tickers. We
   translate via a CIK->ticker map. A company we cannot map is dropped -- and COUNTED,
   never silently lost (that silent loss is the exact bug the per-company fetch had).

3. THE Dec-2024 FORMAT CHANGE. Reprocessed num.txt gained a 'segments' field. A row
   WITH a segment value is a slice (e.g. revenue for one business unit); the
   company-wide total is the row with EMPTY segments. We keep only the empty-segment
   rows, or we would double-count. On older files without the column, everything is
   a total already.

We reuse CONCEPT_TAGS from the existing EDGAR adapter verbatim -- one definition of
what "equity" means, shared by both sources, so they can never silently disagree.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from engine.data.edgar_adapter import ACCEPTED_FORMS, CONCEPT_TAGS

# Reverse the concept->tags map into tag->concept for a fast lookup while filtering.
# If two concepts ever claimed the same tag we would have a problem; assert they don't.
_TAG_TO_CONCEPT: dict[str, str] = {}
for _concept, _tags in CONCEPT_TAGS.items():
    for _tag in _tags:
        if _tag in _TAG_TO_CONCEPT:
            raise RuntimeError(f"Tag {_tag} claimed by two concepts -- ambiguous.")
        _TAG_TO_CONCEPT[_tag] = _concept

# Tag priority = position within its concept's tag list (earlier = preferred). This
# is what makes tag selection DETERMINISTIC: when a filing reports equity under both
# StockholdersEquity and StockholdersEquityIncludingPortion..., the first-listed
# (parent-only) wins. Single source of truth: the ORDER of CONCEPT_TAGS.
_TAG_PRIORITY: dict[str, int] = {}
for _concept, _tags in CONCEPT_TAGS.items():
    for _rank, _tag in enumerate(_tags):
        _TAG_PRIORITY[_tag] = _rank


def _load_table(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    with zf.open(name) as f:
        return pd.read_csv(f, sep="\t", dtype=str, low_memory=False)


def parse_quarter(
    zip_path: Path | str,
    cik_to_ticker: dict[int, str],
) -> pd.DataFrame:
    """
    Parse one cached quarter ZIP into FACT_COLUMNS-shaped rows.

    `cik_to_ticker` maps integer CIK -> ticker for the companies we care about.
    Companies not in the map are dropped (and countable via parse_quarter_report).

    Returns a DataFrame with columns:
        symbol, concept, period_end, filed, value, form, fiscal_year
    """
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        sub = _load_table(zf, "sub.txt")
        num = _load_table(zf, "num.txt")

    # --- sub.txt: one row per filing. Keep id, who, when, what form. -----------
    sub = sub[["adsh", "cik", "form", "period", "filed", "fy"]].copy()
    sub["cik_int"] = pd.to_numeric(sub["cik"], errors="coerce")
    sub = sub[sub["form"].isin(ACCEPTED_FORMS)]                 # 10-K/10-Q/20-F/40-F
    sub["ticker"] = sub["cik_int"].map(cik_to_ticker)
    sub = sub[sub["ticker"].notna()]                           # drop unmappable filers

    if sub.empty:
        return _empty_facts()

    sub["filed_dt"] = pd.to_datetime(sub["filed"], format="%Y%m%d", errors="coerce")

    # --- num.txt: one row per fact. Filter HARD to our tags first. -------------
    keep_tags = set(_TAG_TO_CONCEPT)
    num = num[num["tag"].isin(keep_tags)].copy()

    # Dec-2024 format: keep ONLY company-wide totals, drop the EquityComponents
    # breakdown rows (CommonStock, RetainedEarnings, NoncontrollingInterest, ...).
    # CRITICAL: a company-wide total's segments field is EMPTY, but in the raw TSV
    # that empty reads back as the string 'nan' (pandas NaN -> "nan"), NOT "". The
    # original `== ""` test matched almost nothing, so every breakdown row leaked
    # through and corrupted equity for ~100 S&P 500 names. Treat NaN/''/'nan' as
    # "no segment".
    def _is_blank(series: pd.Series) -> pd.Series:
        s = series.astype(str).str.strip().str.lower()
        return series.isna() | s.isin(["", "nan", "none"])

    if "segments" in num.columns:
        num = num[_is_blank(num["segments"])]
    if "coreg" in num.columns:
        # co-registrant rows are subsidiaries filing jointly; keep only the parent.
        num = num[_is_blank(num["coreg"])]

    # 'qtrs' = number of quarters the value covers. For balance-sheet items (equity,
    # assets, shares) it is 0 (a point in time). For flow items (revenue, net income)
    # we keep both annual (4) and quarterly (1); the concept dedup downstream and
    # form field let us pick. We keep all and let FundamentalData.as_of choose the
    # most recent as-filed value per concept.
    num["value_num"] = pd.to_numeric(num["value"], errors="coerce")
    num = num[num["value_num"].notna()]
    num["period_end_dt"] = pd.to_datetime(num["ddate"], format="%Y%m%d", errors="coerce")
    num["concept"] = num["tag"].map(_TAG_TO_CONCEPT)

    # --- join facts to their filing (for filed date, form, ticker) -------------
    merged = num.merge(
        sub[["adsh", "ticker", "form", "filed_dt", "fy"]],
        on="adsh",
        how="inner",
    )
    if merged.empty:
        return _empty_facts()

    out = pd.DataFrame({
        "symbol": merged["ticker"],
        "concept": merged["concept"],
        "period_end": merged["period_end_dt"],
        "filed": merged["filed_dt"],
        "value": merged["value_num"],
        "form": merged["form"],
        "fiscal_year": pd.to_numeric(merged["fy"], errors="coerce"),
        "_tag": merged["tag"],
    })

    # DETERMINISTIC TAG PRIORITY. A company can report equity under two tags in the
    # same filing -- StockholdersEquity (parent only) AND ...IncludingPortion...
    # (with minority interest) -- differing by billions. For book value per COMMON
    # share, parent-only is correct. We rank each row by its tag's position in
    # CONCEPT_TAGS (earlier = preferred) so the right one wins the dedup below,
    # rather than whichever happened to sort last. This is the OTHER half of the
    # WMT bug: even company-wide totals had two tags to choose between.
    out["_tag_rank"] = out["_tag"].map(_TAG_PRIORITY).fillna(9_999).astype(int)

    # A filing restates prior periods too; the same (symbol, concept, period_end)
    # can appear from multiple filings. Keep the EARLIEST-filed value -- that is the
    # number as FIRST reported, which is what an honest backtest would have seen.
    # Within the same filing date, break ties by TAG PRIORITY (parent-only equity
    # beats total-with-minority-interest).
    out = (
        out.dropna(subset=["symbol", "concept", "period_end", "filed"])
        .sort_values(["filed", "_tag_rank"])
        .drop_duplicates(subset=["symbol", "concept", "period_end"], keep="first")
        .drop(columns=["_tag", "_tag_rank"])
        .reset_index(drop=True)
    )
    return out


def _empty_facts() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["symbol", "concept", "period_end", "filed", "value", "form", "fiscal_year"]
    )


def parse_quarter_report(
    zip_path: Path | str,
    cik_to_ticker: dict[int, str],
) -> dict:
    """
    Parse + return a small diagnostics dict alongside the facts, so a bulk build can
    report coverage honestly instead of silently dropping the unmappable.
    """
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        sub = _load_table(zf, "sub.txt")

    sub_cik = pd.to_numeric(sub["cik"], errors="coerce")
    total_filers = sub_cik.nunique()
    mappable = sub_cik[sub_cik.isin(cik_to_ticker)].nunique()

    facts = parse_quarter(zip_path, cik_to_ticker)

    return {
        "facts": facts,
        "total_filers": int(total_filers),
        "mappable_filers": int(mappable),
        "unmapped_filers": int(total_filers - mappable),
        "rows_kept": len(facts),
        "concepts_seen": sorted(facts["concept"].unique()) if not facts.empty else [],
    }
