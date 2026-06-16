#!/usr/bin/env bash
# ============================================================================
# Football Predictor — Shell Aliases
# ============================================================================
# Source this file in your ~/.bashrc or ~/.zshrc:
#   source /path/to/predictor/aliases.sh
#
# On Arch, just add the source line to ~/.bashrc.
# The scripts auto-create symlinks in ~/.local/bin/ on first run,
# so if that's on your PATH you don't even need this file.
# ============================================================================

PREDICTOR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Main predictor ──────────────────────────────────────────────────────────
alias predict="python3 '$PREDICTOR_DIR/predict.py'"

# Calibration shortcut
alias predict-calibrate="python3 '$PREDICTOR_DIR/predict.py' --calibrate"

# Review mode (interactive)
alias predict-review="python3 '$PREDICTOR_DIR/predict.py' --review"

# Auto-review (re-scrapes Forebet for scores)
alias predict-autoreview="python3 '$PREDICTOR_DIR/predict.py' --review --auto"

# High-confidence only
alias predict-high="python3 '$PREDICTOR_DIR/predict.py' --high-only"

# JSON output
alias predict-json="python3 '$PREDICTOR_DIR/predict.py' --json"

# ── Confidence updater ──────────────────────────────────────────────────────
alias cnf="python3 '$PREDICTOR_DIR/cnfupdate.py'"
alias cnfupdate="python3 '$PREDICTOR_DIR/cnfupdate.py'"

# ── Results scraper ─────────────────────────────────────────────────────────
alias scrape-results="python3 '$PREDICTOR_DIR/scrape_results.py'"

# ── Historical calibration ──────────────────────────────────────────────────
alias histcal="python3 '$PREDICTOR_DIR/historical_calibrate.py'"
alias historical-calibrate="python3 '$PREDICTOR_DIR/historical_calibrate.py'"

# ── Development / testing ───────────────────────────────────────────────────
alias predict-test="python3 '$PREDICTOR_DIR/predict.py' '$PREDICTOR_DIR/test_links.txt'"
alias predict-test-high="python3 '$PREDICTOR_DIR/predict.py' --high-only '$PREDICTOR_DIR/test_links.txt'"

# ── Convenience: predict from a single URL ──────────────────────────────────
predict-url() {
    python3 "$PREDICTOR_DIR/predict.py" "$1"
}

# ── Reports ─────────────────────────────────────────────────────────────────
predict-report() {
    python3 "$PREDICTOR_DIR/predict.py" --calibrate
    echo
    echo "── League profiles ──────────────────────────────"
    python3 -c "
from predict import LEAGUE_PROFILES
for name, p in sorted(LEAGUE_PROFILES.items()):
    vol = p['volatility']
    tag = '⚡' if vol >= 0.25 else '~' if vol >= 0.15 else ' '
    print(f'{tag} {name:<30s} avg_g={p[\"avg_goals\"]:.1f}  vol={vol:.2f}')
"
}

echo "predictor aliases loaded. Available:"
echo "  predict, predict-calibrate, predict-review, predict-high"
echo "  cnf / cnfupdate, scrape-results, predict-url"
