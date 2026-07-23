"""
Apply the settled portfolio construction to a strategy config, in place.

    python -m scripts.apply_config_25                          # default strategy
    python -m scripts.apply_config_25 us_sp500_top20_momentum
    python -m scripts.apply_config_25 --dry-run                # show the diff only

WHY A SCRIPT RATHER THAN A REPLACEMENT FILE. Hand-editing YAML is easy to get subtly
wrong, and a wholesale replacement risks dropping a field the engine needs. This makes
three targeted edits, prints a diff, and writes a .bak first.

WHAT IT CHANGES (all decided on survivorship-free data -- see DECISIONS.md):
  selection.top_n            20 -> 25
      A 25% sector cap is arithmetically impossible below 4 sectors, and 20 names
      spans as few as 4. The cap then binds so hard the portfolio buys the best of a
      WEAK sector, which measurably DEEPENED drawdown. At 25 names it is satisfiable.

  weighting.max_sector_weight  (absent) -> 0.25
      25% is the SEC's and SEBI's own threshold for "concentrated". Costs 0.41pt of
      excess return, buys 3.1pt of drawdown and the highest Sharpe measured (0.89).

  weighting.method           left as 'equal'
      Tested against inverse-volatility: inverse-vol cost 1.85pt of return, returned
      only 0.3pt of drawdown, and lost on BOTH Sharpe and Sortino.

It does NOT change rebalance frequency (monthly confirmed: quarterly cost 5.78pt and
deepened drawdown 15.6pt) and does NOT enable a no-trade buffer (every buffer tested
cost return, because momentum decays).
"""

from __future__ import annotations

import argparse
import difflib
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STRATEGY_DIR = REPO_ROOT / "config" / "strategies"
DEFAULT = "us_sp500_top20_momentum"

SECTOR_CAP_BLOCK = """
  # 25% is the recognised global threshold for "concentrated": the SEC treats >25%
  # of assets in one industry as concentration, and SEBI uses 25% for sector
  # exposure. At 25 holdings this costs 0.41pt of excess return and buys 3.1pt of
  # drawdown plus the highest Sharpe we have measured (0.89). Below 25 holdings it
  # is infeasible (needs 4+ sectors) and actively harmful.
  #
  # Note this is STRICTER than the large momentum funds: SPMO (Invesco S&P 500
  # Momentum) applies no sector cap at all, and MTUM has run ~39% in one sector.
  # Being stricter is a deliberate choice for a family-and-friends product.
  max_sector_weight: 0.25
"""


def patch(text: str) -> tuple[str, list[str]]:
    notes = []
    out = text

    # 1. top_n: 20 -> 25
    m = re.search(r"(^\s*top_n:\s*)(\d+)", out, flags=re.M)
    if m:
        if m.group(2) == "25":
            notes.append("top_n already 25 -- unchanged")
        else:
            out = out[:m.start()] + m.group(1) + "25" + out[m.end():]
            notes.append(f"top_n {m.group(2)} -> 25")
    else:
        notes.append("WARNING: no 'top_n:' found -- set it to 25 by hand")

    # 2. max_sector_weight -> 0.25 (add under weighting: if absent)
    m = re.search(r"(^\s*max_sector_weight:\s*)([\d.]+|null|~)", out, flags=re.M)
    if m:
        if m.group(2) == "0.25":
            notes.append("max_sector_weight already 0.25 -- unchanged")
        else:
            out = out[:m.start()] + m.group(1) + "0.25" + out[m.end():]
            notes.append(f"max_sector_weight {m.group(2)} -> 0.25")
    else:
        # insert after max_position_weight, else after the weighting: header
        anchor = re.search(r"^\s*max_position_weight:.*$", out, flags=re.M)
        if not anchor:
            anchor = re.search(r"^weighting:.*$", out, flags=re.M)
        if anchor:
            idx = anchor.end()
            out = out[:idx] + "\n" + SECTOR_CAP_BLOCK.rstrip() + out[idx:]
            notes.append("max_sector_weight added: 0.25")
        else:
            notes.append("WARNING: no 'weighting:' block found -- add "
                         "max_sector_weight: 0.25 by hand")

    # 3. stale description mentioning 12-2 when skip_months is 1
    if re.search(r"skip_months:\s*1\b", out) and "12-2" in out:
        out = out.replace("12-2", "12-1")
        notes.append("corrected stale '12-2' description to '12-1'")

    return out, notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("strategy", nargs="?", default=DEFAULT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = STRATEGY_DIR / f"{args.strategy}.yaml"
    if not path.exists():
        avail = sorted(p.stem for p in STRATEGY_DIR.glob("*.yaml"))
        print(f"\n  No strategy '{args.strategy}'. Available: {avail}\n")
        sys.exit(1)

    original = path.read_text(encoding="utf-8")
    updated, notes = patch(original)

    print()
    print("=" * 78)
    print(f"  PATCHING {path.relative_to(REPO_ROOT)}")
    print("=" * 78)
    for n in notes:
        print(f"  - {n}")

    if updated == original:
        print("\n  Nothing to change; file already matches the settled config.\n")
        return

    print("\n" + "-" * 78)
    print("  DIFF")
    print("-" * 78)
    for line in difflib.unified_diff(
        original.splitlines(), updated.splitlines(),
        fromfile="before", tofile="after", lineterm="", n=2,
    ):
        print("  " + line)

    if args.dry_run:
        print("\n  --dry-run: nothing written.\n")
        return

    backup = path.with_suffix(".yaml.bak")
    shutil.copy2(path, backup)
    path.write_text(updated, encoding="utf-8")

    print()
    print(f"  Backup written to {backup.name}")
    print(f"  Updated {path.name}")
    print()
    print("  Next:")
    print("    python -m pytest tests/ -q")
    print("    python -m scripts.signal_report")
    print()


if __name__ == "__main__":
    main()
