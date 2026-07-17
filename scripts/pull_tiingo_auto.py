"""
Hands-off wrapper: pull all dead names, auto-waiting out the hourly rate limit.

    python -m scripts.pull_tiingo_auto YOUR_TIINGO_KEY

Kick this off ONCE and walk away. It runs the normal puller, and when Tiingo's
~50/hour limit stops it, this waits ~62 minutes and automatically resumes -- looping
until every target name is cached. ~414 names at 50/hr means it finishes in ~9 waits
(roughly half a day of wall-clock, unattended). Progress is cached continuously, so
you can Ctrl-C and re-run this any time without losing work.

This is purely a convenience loop around scripts.pull_tiingo_prices -- same caching,
same resume behaviour, just without you re-typing the command every hour.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

WAIT_SECONDS = 62 * 60          # a hair over an hour, to be safe with the rolling window
MAX_ROUNDS = 30                 # generous ceiling; 414 names need ~9


def _remaining_after_run(key: str) -> int:
    """Run the puller once; return how many target names are still uncached."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.pull_tiingo_prices", key],
        capture_output=True, text=True,
    )
    print(result.stdout, end="")
    if result.stderr.strip():
        print(result.stderr, end="")
    # Parse the "Remaining : N" line the puller prints.
    for line in result.stdout.splitlines():
        if "Remaining" in line and ":" in line:
            try:
                return int(line.split(":")[1].strip())
            except ValueError:
                pass
    return -1  # unknown; treat as "keep going" cautiously


def main() -> None:
    if len(sys.argv) < 2:
        print("\n  python -m scripts.pull_tiingo_auto YOUR_TIINGO_KEY\n")
        return
    key = sys.argv[1]

    print("\n  AUTO-PULL: will resume every ~hour until all dead names are cached.")
    print("  Safe to Ctrl-C and restart anytime -- progress is cached.\n")

    for rnd in range(1, MAX_ROUNDS + 1):
        print(f"\n{'='*72}\n  ROUND {rnd} — {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*72}")
        remaining = _remaining_after_run(key)

        if remaining == 0:
            print("\n  ✅ All target names cached. Auto-pull complete.")
            print("  Next: python -m scripts.backtest_honest <KEY>\n")
            return
        if remaining < 0:
            print("\n  Couldn't read remaining count; stopping to be safe. "
                  "Re-run to continue.\n")
            return

        print(f"\n  {remaining} names left. Waiting ~62 min for the rate limit to "
              f"reset, then resuming automatically...")
        print(f"  (Round {rnd}/{MAX_ROUNDS}. Ctrl-C to stop; progress is saved.)")
        try:
            time.sleep(WAIT_SECONDS)
        except KeyboardInterrupt:
            print("\n  Stopped. Re-run scripts.pull_tiingo_auto to continue.\n")
            return

    print(f"\n  Hit the {MAX_ROUNDS}-round ceiling. Re-run to finish the rest.\n")


if __name__ == "__main__":
    main()
