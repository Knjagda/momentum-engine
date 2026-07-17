"""
Why do the two fundamental sources disagree on equity? Show the raw tags.

    python -m scripts.diagnose_equity_disagreement your@email.com

The coverage comparison found ~100 S&P 500 companies where bulk and per-company
EDGAR report DIFFERENT stockholders' equity. Before trusting either, we need to see
WHY. This dumps, for a few of the worst offenders, every equity-related XBRL fact
each source has for the relevant period -- the actual tag names and values.

The hypothesis: companies that report BOTH
    StockholdersEquity                                          (parent only)
    StockholdersEquityIncludingPortionAttributableToNoncontrolling... (total)
get resolved differently by the two adapters, because neither has a DETERMINISTIC
rule for which tag wins when a filing contains both. If that is the cause, the fix
is to make the rule explicit and identical in both -- not to declare one "right".

We check WMT, FCX, IBKR, AIG -- big, clear disagreements from the comparison.
"""

from __future__ import annotations

import sys
import zipfile

import pandas as pd

from engine.data import get_adapter, get_fundamental_adapter
from engine.data.edgar_adapter import CONCEPT_TAGS
from engine.data.sec_bulk_download import ensure_quarter, quarters_between
from engine.data.sec_bulk_parse import _load_table
from engine.markets.market import load_market

SUSPECTS = ["WMT", "FCX", "IBKR", "AIG"]
EQUITY_TAGS = CONCEPT_TAGS["equity"]


def main() -> None:
    if len(sys.argv) < 2 or "@" not in sys.argv[1]:
        print("\n  python -m scripts.diagnose_equity_disagreement you@email.com\n")
        return
    email = sys.argv[1]
    market = load_market("us")

    # Map suspects -> CIK using the cached map.
    edgar = get_fundamental_adapter("edgar", market=market,
                                    user_agent=f"momentum-engine research {email}")
    cik_of = {s: edgar.resolve_cik(s) for s in SUSPECTS}
    print("\n  CIKs:", cik_of, "\n")

    # Pull the RAW num.txt rows for these companies from the 2024 quarters, showing
    # EVERY equity tag they filed -- this is the ground truth the parser sees.
    print("=" * 84)
    print("  RAW SEC BULK FACTS — every equity tag these companies filed (2024)")
    print("=" * 84)

    cik_set = {c for c in cik_of.values() if c}
    for q in quarters_between("2024q1", "2024q4"):
        path = ensure_quarter(q, email)
        with zipfile.ZipFile(path) as zf:
            sub = _load_table(zf, "sub.txt")
            num = _load_table(zf, "num.txt")

        sub["cik_int"] = pd.to_numeric(sub["cik"], errors="coerce")
        sub_ours = sub[sub["cik_int"].isin(cik_set)]
        if sub_ours.empty:
            continue

        adsh_to_sym = {}
        for _, r in sub_ours.iterrows():
            sym = next((s for s, c in cik_of.items() if c == r["cik_int"]), "?")
            adsh_to_sym[r["adsh"]] = (sym, r["form"], r["filed"])

        eq = num[(num["adsh"].isin(adsh_to_sym)) & (num["tag"].isin(EQUITY_TAGS))]
        for _, r in eq.iterrows():
            sym, form, filed = adsh_to_sym[r["adsh"]]
            seg = str(r.get("segments", "")).strip()
            coreg = str(r.get("coreg", "")).strip()
            flag = ""
            if seg:
                flag += " [SEGMENT]"
            if coreg:
                flag += " [COREG]"
            val = pd.to_numeric(r["value"], errors="coerce")
            print(f"  {q} {sym:<5} {form:<6} filed {filed}  "
                  f"period {r['ddate']}  {r['tag'][:52]:<52} "
                  f"{val/1e9:>8.1f}B{flag}")
        print()

    print("=" * 84)
    print("  WHAT TO LOOK FOR")
    print("=" * 84)
    print("  If a company shows TWO different equity tags with different values, that is")
    print("  the disagreement's source: one adapter grabbed the parent-only figure, the")
    print("  other the total-with-minority-interest. Whichever we CHOOSE, both adapters")
    print("  must choose the SAME one -- deterministically, by explicit tag priority.")
    print()
    print("  Note [SEGMENT]/[COREG] flags: if those appear, a mis-filter is the cause")
    print("  instead, and the fix is on the parse side.")
    print()


if __name__ == "__main__":
    main()
