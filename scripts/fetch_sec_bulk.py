"""
Download SEC bulk financial-statement quarters into the local cache.

    # one quarter (quick check):
    python -m scripts.fetch_sec_bulk you@email.com 2024q1

    # a range (the real backfill, ~5GB for a full history -- do this once):
    python -m scripts.fetch_sec_bulk you@email.com 2012q1 2026q1

Downloads each quarter exactly once. Re-running skips anything already cached, so
it is safe to stop and resume. This only DOWNLOADS -- parsing comes next.
"""

import sys
import time

from engine.data.sec_bulk_download import (
    SecBulkDownloadError,
    cache_path,
    ensure_quarter,
    quarters_between,
)


def main() -> None:
    if len(sys.argv) < 3 or "@" not in sys.argv[1]:
        print("\n  Usage:")
        print("    python -m scripts.fetch_sec_bulk you@email.com 2024q1")
        print("    python -m scripts.fetch_sec_bulk you@email.com 2012q1 2026q1\n")
        return

    email = sys.argv[1]
    start = sys.argv[2]
    end = sys.argv[3] if len(sys.argv) > 3 else start
    quarters = quarters_between(start, end)

    print(f"\n  {len(quarters)} quarter(s) to ensure: {quarters[0]} .. {quarters[-1]}")
    print("  Already-cached quarters are skipped instantly.\n")

    got, skipped, failed = 0, 0, 0
    for q in quarters:
        if cache_path(q).exists():
            print(f"    {q}  ✓ cached")
            skipped += 1
            continue
        try:
            t0 = time.time()
            path = ensure_quarter(q, email)
            mb = path.stat().st_size / 1e6
            print(f"    {q}  ↓ {mb:.0f} MB in {time.time()-t0:.0f}s")
            got += 1
            time.sleep(0.5)   # be gentle to the SEC
        except SecBulkDownloadError as e:
            print(f"    {q}  ✗ {e}")
            failed += 1

    print(f"\n  Done. Downloaded {got}, skipped {skipped}, failed {failed}.")
    print("  Raw ZIPs cached under data/fundamentals/sec_bulk_raw/ (gitignored).\n")


if __name__ == "__main__":
    main()
