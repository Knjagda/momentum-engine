"""
Fetch fundamentals from SEC EDGAR, and MEASURE WHAT WE ARE MISSING.

    python -m scripts.fetch_fundamentals you@yourdomain.com
    python -m scripts.fetch_fundamentals you@yourdomain.com sp900_pit

The second half of this script matters more than the first.

Every backtesting tool on earth prints a disclaimer: "results may be affected by
survivorship bias". That sentence is worthless. It admits a problem without
measuring it, which is the same as hiding it.

EDGAR lets us do better. EDGAR knows WHO EXISTED: if a company filed 10-Ks until
2015 and then stopped, it was real, and then it wasn't. So for any universe we can
ask a question nobody usually asks:

    How many companies were in this index -- and for how many do we
    actually have the data to trade them?

The gap is our bias. Not a disclaimer. A NUMBER.
"""

from __future__ import annotations

import sys

import pandas as pd

from engine.data import get_adapter, get_fundamental_adapter
from engine.markets.market import load_market
from engine.universe.universe import load_membership

DEFAULT_UNIVERSE = "sp900_pit"


def main() -> None:
    if len(sys.argv) < 2 or "@" not in sys.argv[1]:
        print()
        print("  The SEC requires you to identify yourself. Pass your email:")
        print("      python -m scripts.fetch_fundamentals you@yourdomain.com [universe]")
        print()
        print("  This is not bureaucracy. They will 403 you, and they are right to.")
        print()
        return

    email = sys.argv[1]
    universe = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_UNIVERSE

    market = load_market("us")
    membership = load_membership(market, universe)

    user_agent = f"momentum-engine research {email}"
    edgar = get_fundamental_adapter("edgar", market=market, user_agent=user_agent)
    prices_adapter = get_adapter(market)

    symbols = membership.symbols

    print()
    print("=" * 86)
    print("  FUNDAMENTALS — SEC EDGAR")
    print("=" * 86)
    print(f"  Universe : {universe}  ({len(symbols)} symbols)")
    print(f"  Source   : SEC EDGAR (free, official, point-in-time)")
    print()
    print("  Fetching... (first run is slow — ~8/sec, then cached to parquet)")
    print()

    data = edgar.fetch(symbols)

    print(f"  {data}")
    print()

    # ---- what we actually got ---------------------------------------------
    print("=" * 86)
    print("  CONCEPT COVERAGE")
    print("=" * 86)
    for concept in data.concepts:
        pct = data.coverage(concept)
        bar = "█" * int(pct * 30)
        flag = "  ⚠️ thin" if pct < 0.8 else ""
        print(f"  {concept:<22}{pct:>7.1%}  {bar}{flag}")
    print()

    # ---- THE TRUE FILING LAG ----------------------------------------------
    lag = data.filing_lag()
    if len(lag):
        print("=" * 86)
        print("  THE TRUE FILING LAG  (as-first-reported, comparatives excluded)")
        print("=" * 86)
        print(f"  Median : {lag.median():>5.0f} days")
        print(f"  P90    : {lag.quantile(0.90):>5.0f} days")
        print(f"  Max    : {lag.max():>5.0f} days")
        print()
        print("  This is how long a company's annual numbers stay SECRET after its")
        print("  year ends. Any screen that uses them sooner is reading the future.")
        print("  Our engine cannot: FundamentalData.as_of() will not show them.")
        print()

    # ---- THE COVERAGE GAP — the honest bit --------------------------------
    print("=" * 86)
    print("  THE SURVIVORSHIP GAP — measured, not disclaimed")
    print("=" * 86)

    print("  Checking which universe members we can actually PRICE...")
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    prices = prices_adapter.fetch(symbols, "2009-01-01", today)

    priced = set(prices.symbols)
    have_facts = set(data.symbols)
    everyone = set(symbols)

    no_price = everyone - priced
    no_facts = everyone - have_facts

    print()
    print(f"  Universe members (all time)   : {len(everyone):>5}")
    print(f"  We have PRICES for            : {len(priced):>5}  "
          f"({len(priced)/len(everyone):.1%})")
    print(f"  We have FUNDAMENTALS for      : {len(have_facts):>5}  "
          f"({len(have_facts)/len(everyone):.1%})")
    print()

    if no_price:
        print(f"  ⚠️  {len(no_price)} companies we CANNOT PRICE.")
        print("      These were genuinely in the index. Our engine cannot hold them,")
        print("      so it never takes their losses. THIS IS OUR RESIDUAL SURVIVORSHIP BIAS.")
        print()

        # The ones that hurt: companies that went to ZERO, not the ones acquired.
        # An acquisition usually pays a PREMIUM -- missing it makes us look WORSE,
        # which is the safe direction. A bankruptcy is what we cannot afford to miss.
        notorious = {
            "LEH": "Lehman Brothers (2008, → zero)",
            "FNM": "Fannie Mae (2008, → near-zero)",
            "FRE": "Freddie Mac (2008, → near-zero)",
            "CFC": "Countrywide (2008, distressed sale)",
            "ABK": "Ambac (2008, → near-zero)",
            "GGP": "General Growth (2009, bankruptcy)",
            "SIVB": "Silicon Valley Bank (2023, → ZERO)",
            "FRC": "First Republic (2023, → ZERO)",
            "MEE": "Massey Energy (2011)",
            "EK": "Eastman Kodak (2012, bankruptcy)",
            "JCP": "JC Penney (2020, bankruptcy)",
            "RAD": "Rite Aid (2023, bankruptcy)",
        }

        hits = sorted(no_price & set(notorious))
        if hits:
            print("      The ones that matter — these went to ZERO and we cannot own them:")
            for sym in hits:
                print(f"         {sym:<7}{notorious[sym]}")
            print()
            print("      Note SIVB and FRC. Both collapsed in 2023 — INSIDE our window.")
            print("      In early 2023 distressed banks traded at very low price-to-book.")
            print("      A VALUE SCREEN WOULD HAVE BOUGHT THEM. We cannot, so we never")
            print("      take the loss. Keep that in mind before believing any value")
            print("      backtest we produce.")
            print()

    if no_facts:
        sample = sorted(no_facts)[:10]
        print(f"  ⚠️  {len(no_facts)} companies with NO fundamentals.")
        print(f"      {', '.join(sample)}{'...' if len(no_facts) > 10 else ''}")
        print("      Some are pre-2009 delistings (no XBRL ever existed).")
        print("      Some are BAD CIK MAPPINGS — a live company silently missing.")
        print("      Do not assume. Check a few by hand.")
        print()

    print("=" * 86)
    print("  Fundamentals cached to data/fundamentals/. Subsequent runs are instant.")
    print("=" * 86)
    print()


if __name__ == "__main__":
    main()
