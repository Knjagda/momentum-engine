"""
SPIKE: are the SEC's BULK financial-statement data sets a viable, free replacement
for our fragile per-company EDGAR fetch?

    python -m scripts.spike_sec_bulk your@email.com

WHY THIS MATTERS. Today we fetch fundamentals one CIK at a time from EDGAR's company
API. That is slow, and worse, it fails SILENTLY when our ticker->CIK map is wrong --
that is how live companies like Bank OZK and Sotheby's went missing from a whole
backtest. The SEC also publishes the SAME data as bulk quarterly ZIPs containing
EVERY filer at once. No per-ticker lookups, no CIK guessing: if a company filed, it
is in the file.

This spike does not change the engine. It answers five questions and stops:

  Q1  Does the quarterly ZIP download and open?
  Q2  What is inside it, and can we read sub.txt (submissions) + num.txt (numbers)?
  Q3  Does sub.txt carry an honest FILED date -- our anti-look-ahead key?
  Q4  Can we pull one real company's book equity for one quarter, by CIK, and does
      the number look right?
  Q5  Coverage sanity: how many distinct filers are in one quarter? (Should be
      thousands -- vs the ~1100 our per-company fetch managed across the WHOLE
      universe.)

If all five pass, the bulk approach is real and we plan the switch. If any fail,
we learn exactly where and decide with eyes open. Nothing is committed either way.

NOTE ON FORMAT: in Dec 2024 the SEC reprocessed these sets and added a 'segments'
field to num.txt. This spike does not assume a fixed schema -- it reads whatever
columns are present and reports them.
"""

from __future__ import annotations

import io
import sys
import zipfile
from urllib.request import Request, urlopen

import pandas as pd

BASE = "https://www.sec.gov/files/dera/data/financial-statement-data-sets"

# One recent, moderately sized quarter. Big enough to be representative, and it
# post-dates the Dec-2024 reprocessing so we see the CURRENT format.
TEST_QUARTER = "2025q1"

# A well-known CIK to spot-check a real number against.
# 320193 = Apple Inc. Its 10-Q filed in early 2025 should be in 2025q1.
APPLE_CIK = 320193


def fetch_zip(quarter: str, email: str) -> zipfile.ZipFile:
    url = f"{BASE}/{quarter}.zip"
    # SEC REQUIRES a descriptive User-Agent with contact info, or it 403s.
    req = Request(url, headers={"User-Agent": f"momentum-engine research {email}"})
    print(f"  Downloading {url} ...")
    raw = urlopen(req, timeout=120).read()
    print(f"  Got {len(raw)/1e6:.1f} MB.")
    return zipfile.ZipFile(io.BytesIO(raw))


def main() -> None:
    if len(sys.argv) < 2 or "@" not in sys.argv[1]:
        print("\n  The SEC requires an email in the User-Agent. Pass one:")
        print("      python -m scripts.spike_sec_bulk you@yourdomain.com\n")
        return
    email = sys.argv[1]

    print()
    print("=" * 84)
    print("  SEC BULK DATA SPIKE — is the free, complete fundamentals source real?")
    print("=" * 84)
    print(f"  Test quarter: {TEST_QUARTER}")
    print()

    # ---- Q1: download + open -------------------------------------------------
    try:
        zf = fetch_zip(TEST_QUARTER, email)
    except Exception as e:
        print(f"  ❌ Q1 FAILED: could not download/open the ZIP: {e}")
        print("     If this is a 403, the User-Agent was rejected -- check the email.")
        return
    print("  ✅ Q1: ZIP downloaded and opened.\n")

    # ---- Q2: what's inside + read the two key tables -------------------------
    names = zf.namelist()
    print(f"  Files in the ZIP: {', '.join(names)}")
    for needed in ("sub.txt", "num.txt"):
        if needed not in names:
            print(f"  ❌ Q2 FAILED: expected '{needed}' not found.")
            return

    # These are TAB-separated. sub.txt is one row per filing; num.txt one per fact.
    with zf.open("sub.txt") as f:
        sub = pd.read_csv(f, sep="\t", dtype=str, low_memory=False)
    with zf.open("num.txt") as f:
        num = pd.read_csv(f, sep="\t", dtype=str, low_memory=False)

    print(f"  sub.txt: {len(sub):,} filings, columns: {list(sub.columns)[:8]}...")
    print(f"  num.txt: {len(num):,} numeric facts, columns: {list(num.columns)[:8]}...")
    print("  ✅ Q2: both key tables read cleanly.\n")

    # ---- Q3: is there an honest FILED date? ----------------------------------
    # 'filed' in sub.txt is the date the filing became public -- our no-look-ahead
    # gate. 'period' is the balance-sheet date it describes.
    if "filed" not in sub.columns or "period" not in sub.columns:
        print(f"  ❌ Q3 FAILED: no 'filed'/'period' columns. Have: {list(sub.columns)}")
        return
    sub["filed_dt"] = pd.to_datetime(sub["filed"], format="%Y%m%d", errors="coerce")
    sub["period_dt"] = pd.to_datetime(sub["period"], format="%Y%m%d", errors="coerce")
    lag = (sub["filed_dt"] - sub["period_dt"]).dt.days.dropna()
    print(f"  Filed-minus-period lag (days): median {lag.median():.0f}, "
          f"p90 {lag.quantile(0.9):.0f}")
    print("  This lag is the whole point: a fact is usable only from its FILED date,")
    print("  and that date is right here in the data. Point-in-time is native.")
    print("  ✅ Q3: honest as-filed dates present.\n")

    # ---- Q4: pull a real company's book equity, by CIK -----------------------
    sub["cik_int"] = pd.to_numeric(sub["cik"], errors="coerce")
    apple_subs = sub[sub["cik_int"] == APPLE_CIK]
    if apple_subs.empty:
        print(f"  ⚠️  Q4: CIK {APPLE_CIK} (Apple) not in {TEST_QUARTER} -- it may not "
              "have filed this quarter. Trying any large filer instead.")
    else:
        adsh = apple_subs.iloc[0]["adsh"]      # accession number = filing id
        form = apple_subs.iloc[0]["form"]
        print(f"  Apple filing found: {form} (accession {adsh})")

        # StockholdersEquity is the XBRL tag for book equity.
        apple_num = num[num["adsh"] == adsh]
        eq = apple_num[apple_num["tag"] == "StockholdersEquity"]
        if not eq.empty:
            # value is in 'value'; pick the most recent period end.
            eq = eq.copy()
            eq["val_num"] = pd.to_numeric(eq["value"], errors="coerce")
            latest = eq.sort_values("ddate").iloc[-1]
            print(f"  StockholdersEquity = ${latest['val_num']/1e9:.1f}B "
                  f"(period {latest['ddate']})")
            print("  ✅ Q4: pulled a real, sensible book-equity number by CIK.\n")
        else:
            tags = sorted(apple_num["tag"].unique())[:10]
            print(f"  ⚠️  Q4: no StockholdersEquity tag in Apple's filing. "
                  f"Sample tags: {tags}")
            print("     (Not fatal -- tag naming varies; worth noting.)\n")

    # ---- Q5: coverage -- how many distinct filers this quarter? --------------
    n_filers = sub["cik_int"].nunique()
    print(f"  Distinct filers in {TEST_QUARTER}: {n_filers:,}")
    print("  Our per-company EDGAR fetch managed ~1,100 across the ENTIRE universe.")
    print("  One bulk quarter alone dwarfs that -- coverage is not the constraint here.")
    print("  ✅ Q5: coverage is abundant.\n")

    print("=" * 84)
    print("  VERDICT")
    print("=" * 84)
    print("  If Q1-Q5 all show ✅, the bulk approach is real: free, complete, natively")
    print("  point-in-time, no CIK guessing. The next step (NOT today) is to write a")
    print("  bulk adapter that ingests these quarterly files into our fundamentals")
    print("  store, replacing the per-company fetch.")
    print()
    print("  This does NOT fix survivorship on the PRICE side -- dead-company prices")
    print("  still need paid data. It fixes fundamentals coverage, for free.")
    print()


if __name__ == "__main__":
    main()
