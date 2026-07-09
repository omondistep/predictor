#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Step 1: Deduplicate links ==="
python3 "$DIR/dedup_links.py"

echo ""
echo "=== Step 2: Run predictions ==="
pr "$DIR/links.txt"

echo ""
echo "=== Step 3: Auto-learn from outcomes ==="
python3 "$DIR/auto_learn.py" --days-back 365 --max-matches 2000

echo ""
echo "=== Pipeline complete ==="
