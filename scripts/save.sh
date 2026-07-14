#!/usr/bin/env bash
# save.sh — add, commit, push. Same workflow as ResearchMind.
# Usage:  ./scripts/save.sh "your commit message"

set -e

MSG="${1:-checkpoint}"

echo "→ Staging changes..."
git add -A

echo "→ Committing: $MSG"
git commit -m "$MSG"

echo "→ Pushing to origin..."
git push

echo "✅ Saved."
