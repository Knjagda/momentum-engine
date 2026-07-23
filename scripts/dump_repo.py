"""
Dump the whole repo -- tree plus every source file -- into one text file.

    python -m scripts.dump_repo                    # -> repo_dump.txt
    python -m scripts.dump_repo --max-mb 4         # split into parts of ~4MB
    python -m scripts.dump_repo --out snapshot.txt

WHY THIS EXISTS. Claude's sandbox is wiped between sessions, so it repeatedly ends up
writing code against APIs it cannot read -- guessing at signatures, occasionally
getting them wrong, and burning a round trip on a traceback. Uploading one file with
the tree and all the source removes that guesswork entirely.

WHAT IT INCLUDES. Source and config only: .py, .yaml, .yml, .md, .toml, .cfg, .txt,
.sh, .json (small ones). It SKIPS anything that is large, binary, generated, or
private:
  - .git, .venv, __pycache__, .pytest_cache, node_modules, .idea, .vscode
  - data/ (price caches, universes -- big and reproducible)
  - any file over --max-file-kb (default 200)

SECRETS. The dump is scanned for anything that looks like an API key or token and
those lines are redacted before writing. That is a safety net, not a guarantee --
skim the output before sharing it.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    "node_modules", ".idea", ".vscode", ".ruff_cache", "dist", "build",
    ".eggs", "htmlcov",
}
# Directories whose CONTENTS we skip but whose existence we still show in the tree.
SKIP_CONTENTS = {"data"}

INCLUDE_SUFFIXES = {
    ".py", ".yaml", ".yml", ".md", ".toml", ".cfg", ".ini",
    ".txt", ".sh", ".json",
}

# Patterns that look like credentials. Redacted line-by-line.
SECRET_PATTERNS = [
    re.compile(r"(?i)\b[0-9a-f]{40}\b"),                    # 40-char hex token
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*\S+"),
    re.compile(r"(?i)\bsk-[A-Za-z0-9]{20,}\b"),
]


def _skip_dir(p: Path) -> bool:
    return any(part in SKIP_DIRS for part in p.parts)


def _redact(text: str) -> tuple[str, int]:
    out, n = [], 0
    for line in text.splitlines():
        red = line
        for pat in SECRET_PATTERNS:
            if pat.search(red):
                red = pat.sub("[REDACTED]", red)
        if red != line:
            n += 1
        out.append(red)
    return "\n".join(out), n


def build_tree(root: Path) -> list[str]:
    """A readable tree, showing skipped directories without descending into them."""
    lines = []

    def walk(d: Path, prefix: str = ""):
        try:
            entries = sorted(
                [e for e in d.iterdir() if e.name not in SKIP_DIRS],
                key=lambda e: (e.is_file(), e.name.lower()),
            )
        except PermissionError:
            return
        for i, e in enumerate(entries):
            last = i == len(entries) - 1
            branch = "`-- " if last else "|-- "
            if e.is_dir():
                if e.name in SKIP_CONTENTS:
                    lines.append(f"{prefix}{branch}{e.name}/  (contents omitted)")
                    continue
                lines.append(f"{prefix}{branch}{e.name}/")
                walk(e, prefix + ("    " if last else "|   "))
            else:
                try:
                    kb = e.stat().st_size / 1024
                    lines.append(f"{prefix}{branch}{e.name}  ({kb:,.0f} KB)")
                except OSError:
                    lines.append(f"{prefix}{branch}{e.name}")

    lines.append(f"{root.name}/")
    walk(root)
    return lines


def collect_files(root: Path, max_file_kb: int) -> list[Path]:
    files = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or _skip_dir(p.relative_to(root)):
            continue
        rel = p.relative_to(root)
        if rel.parts and rel.parts[0] in SKIP_CONTENTS:
            continue
        if p.suffix.lower() not in INCLUDE_SUFFIXES:
            continue
        try:
            if p.stat().st_size > max_file_kb * 1024:
                continue
        except OSError:
            continue
        files.append(p)
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="repo_dump.txt")
    ap.add_argument("--max-mb", type=float, default=0,
                    help="split into parts of roughly this size (0 = single file)")
    ap.add_argument("--max-file-kb", type=int, default=200,
                    help="skip individual files larger than this")
    args = ap.parse_args()

    root = REPO_ROOT
    tree = build_tree(root)
    files = collect_files(root, args.max_file_kb)

    header = [
        "=" * 78,
        "  REPOSITORY SNAPSHOT",
        "=" * 78,
        f"  Root      : {root.name}",
        f"  Files     : {len(files)} source/config files included",
        "  Omitted   : data/, .venv, __pycache__, .git, binaries, files >"
        f" {args.max_file_kb} KB",
        "  Secrets   : lines matching key/token patterns are redacted",
        "=" * 78,
        "",
        "DIRECTORY TREE",
        "-" * 78,
        *tree,
        "",
        "=" * 78,
        "  FILE CONTENTS",
        "=" * 78,
        "",
    ]

    chunks: list[str] = ["\n".join(header)]
    redacted_total = 0

    for p in files:
        rel = p.relative_to(root).as_posix()
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            chunks.append(f"\n### {rel}\n[could not read: {e}]\n")
            continue
        text, n = _redact(text)
        redacted_total += n
        chunks.append(
            f"\n{'=' * 78}\n### FILE: {rel}  ({len(text.splitlines())} lines)\n"
            f"{'=' * 78}\n{text}\n"
        )

    body = "".join(chunks)
    total_mb = len(body.encode("utf-8")) / (1024 * 1024)

    out_paths = []
    if args.max_mb and total_mb > args.max_mb:
        # split on file boundaries so no source file is cut in half
        limit = int(args.max_mb * 1024 * 1024)
        part, size, idx = [], 0, 1
        for c in chunks:
            b = len(c.encode("utf-8"))
            if size + b > limit and part:
                path = Path(f"{Path(args.out).stem}_part{idx}.txt")
                path.write_text("".join(part), encoding="utf-8")
                out_paths.append(path)
                part, size, idx = [], 0, idx + 1
            part.append(c)
            size += b
        if part:
            path = Path(f"{Path(args.out).stem}_part{idx}.txt")
            path.write_text("".join(part), encoding="utf-8")
            out_paths.append(path)
    else:
        path = Path(args.out)
        path.write_text(body, encoding="utf-8")
        out_paths.append(path)

    print()
    print(f"  Included {len(files)} files, {total_mb:.2f} MB total.")
    if redacted_total:
        print(f"  Redacted {redacted_total} line(s) that looked like credentials.")
    for p in out_paths:
        print(f"  Wrote {p}  ({p.stat().st_size / 1024:,.0f} KB)")
    print()
    print("  Skim it before sharing, then upload to the chat.")
    print()


if __name__ == "__main__":
    main()
