"""
What does the 'segments' field ACTUALLY contain? Show its raw values.

    python -m scripts.inspect_segments your@email.com

The equity disagreement traced to the parser keeping per-segment rows it should
drop. My filter tested `segments == ""`, but every real row came back flagged as
having a segment -- so the real "no segment" value is NOT an empty string. This dumps
the actual distinct segments values (and how WMT's real 83.9B total row is tagged vs
its segment slices) so we filter on the RIGHT thing.
"""
from __future__ import annotations
import sys, zipfile
import pandas as pd
from engine.data.sec_bulk_download import ensure_quarter
from engine.data.sec_bulk_parse import _load_table


def main() -> None:
    if len(sys.argv) < 2 or "@" not in sys.argv[1]:
        print("\n  python -m scripts.inspect_segments you@email.com\n"); return
    email = sys.argv[1]
    path = ensure_quarter("2024q4", email)
    with zipfile.ZipFile(path) as zf:
        num = _load_table(zf, "num.txt")
        sub = _load_table(zf, "sub.txt")

    print("\n=== 'segments' column: distinct value COUNT ===")
    seg = num["segments"] if "segments" in num.columns else pd.Series([], dtype=str)
    print(f"  total rows: {len(num):,}")
    print(f"  segments is NaN/empty:  {seg.isna().sum():,} NaN, "
          f"{(seg.fillna('')=='').sum():,} empty-string")
    print(f"  segments non-empty:     {(seg.fillna('').astype(str).str.len()>0).sum():,}")

    print("\n=== 10 most common 'segments' values (repr, to see whitespace) ===")
    vc = seg.fillna("<<NaN>>").astype(str).value_counts().head(10)
    for val, n in vc.items():
        print(f"  {n:>10,}  {val[:70]!r}")

    # WMT's equity rows: show segments value for the 83.9B total vs a slice
    print("\n=== WMT StockholdersEquity rows: segments value per row ===")
    wmt = sub[sub["name"].str.contains("WALMART", case=False, na=False)]
    if not wmt.empty:
        adshes = set(wmt["adsh"])
        eqrows = num[(num["adsh"].isin(adshes)) &
                     (num["tag"].str.startswith("StockholdersEquity")) &
                     (num["ddate"] == "20240131")]
        for _, r in eqrows.iterrows():
            v = pd.to_numeric(r["value"], errors="coerce")
            sv = r.get("segments", "")
            print(f"  {r['tag'][:46]:<46} {v/1e9:>7.1f}B  segments={str(sv)[:50]!r}")
    print()


if __name__ == "__main__":
    main()
