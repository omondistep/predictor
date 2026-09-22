#!/usr/bin/env bash
# ============================================================================
# Football Predictor — Shell Aliases
# ============================================================================
# Source this file in your ~/.bashrc or ~/.zshrc:
#   source /path/to/predictor/scripts/aliases.sh
#
# Predictor root = parent of this script's directory (scripts/).
# All commands use the repo's .venv python and absolute paths, so every
# command works from any terminal / directory (no cd needed).
# ============================================================================

_PRED_ALIASES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREDICTOR_DIR="$(cd "$_PRED_ALIASES_DIR/.." && pwd)"
unset _PRED_ALIASES_DIR

if [ -x "$PREDICTOR_DIR/.venv/bin/python" ]; then
    PREDICTOR_PY="$PREDICTOR_DIR/.venv/bin/python"
else
    echo "predictor: no .venv found (run scripts/install.sh)" >&2
    PREDICTOR_PY="python3"
fi

# ── Main predictor ──────────────────────────────────────────────────────────
predict() { "$PREDICTOR_PY" "$PREDICTOR_DIR/predict.py" "$@"; }

predict-calibrate()  { "$PREDICTOR_PY" "$PREDICTOR_DIR/predict.py" --calibrate "$@"; }
predict-review()     { "$PREDICTOR_PY" "$PREDICTOR_DIR/predict.py" --review "$@"; }
predict-autoreview() { "$PREDICTOR_PY" "$PREDICTOR_DIR/predict.py" --review --auto "$@"; }
predict-high()       { "$PREDICTOR_PY" "$PREDICTOR_DIR/predict.py" --high-only "$@"; }
predict-json()       { "$PREDICTOR_PY" "$PREDICTOR_DIR/predict.py" --json "$@"; }
predict-url()        { "$PREDICTOR_PY" "$PREDICTOR_DIR/predict.py" "$@"; }
predict-test()       { "$PREDICTOR_PY" "$PREDICTOR_DIR/predict.py" "$PREDICTOR_DIR/test.txt"; }
predict-test-high()  { "$PREDICTOR_PY" "$PREDICTOR_DIR/predict.py" --high-only "$PREDICTOR_DIR/test.txt"; }

# ── Reports ─────────────────────────────────────────────────────────────────
predict-report() {
    "$PREDICTOR_PY" "$PREDICTOR_DIR/predict.py" --calibrate "$@"
    echo
    echo "── League profiles ──────────────────────────────"
    ( cd "$PREDICTOR_DIR" && "$PREDICTOR_PY" -c '
from predict import LEAGUE_PROFILES
for name, p in sorted(LEAGUE_PROFILES.items()):
    vol = p["volatility"]
    tag = "*" if vol >= 0.25 else "~" if vol >= 0.15 else " "
    print(f"{tag} {name:<30s} avg_g={p[\"avg_goals\"]:.1f}  vol={vol:.2f}")
' )
}

# ── Auto-Learn (continuous learning pipeline) ───────────────────────────────
auto-learn()    { "$PREDICTOR_PY" "$PREDICTOR_DIR/scripts/auto_learn.py" "$@"; }
learn-status()  { "$PREDICTOR_PY" "$PREDICTOR_DIR/scripts/auto_learn.py" --status "$@"; }
learn-daemon()  { "$PREDICTOR_PY" "$PREDICTOR_DIR/scripts/auto_learn.py" --daemon "$@"; }

# ── Bets / Bettor agent tools ───────────────────────────────────────────────
# book <links.txt | preds.json>   — bettor's own card: de-vigged edge, EV%,
#                                   Quarter-Kelly stakes, league reliability.
#                                   Pass a links file and it predicts first.
book()        { "$PREDICTOR_PY" "$PREDICTOR_DIR/scripts/bettor.py" card "$@"; }
bettor-audit(){ "$PREDICTOR_PY" "$PREDICTOR_DIR/scripts/bettor.py" audit "$@"; }

# ── Confidence updater ──────────────────────────────────────────────────────
cnf()      { "$PREDICTOR_PY" "$PREDICTOR_DIR/scripts/cnfupdate.py" "$@"; }
cnfupdate(){ "$PREDICTOR_PY" "$PREDICTOR_DIR/scripts/cnfupdate.py" "$@"; }

# ── Results scraper ─────────────────────────────────────────────────────────
scrape-results() { "$PREDICTOR_PY" "$PREDICTOR_DIR/scripts/scrape_results.py" "$@"; }

# ── Output directory ─────────────────────────────────────────────────────────
# output — navigate to / list the predictions output folder.
# Tries to open the folder in a GUI file manager; always prints the listing.
output() {
    local pred_dir="$PREDICTOR_DIR/predictions"
    if command -v xdg-open &>/dev/null; then
        xdg-open "$pred_dir" 2>/dev/null &
    elif command -v open &>/dev/null; then
        open "$pred_dir" 2>/dev/null
    fi
    echo "📁  $pred_dir"
    echo
    ls -lhtr --color=always "$pred_dir"/*.html 2>/dev/null | tail -20 \
        || ls -lhtr "$pred_dir"/*.html 2>/dev/null | tail -20
}

# ── Historical calibration ──────────────────────────────────────────────────
histcal()             { "$PREDICTOR_PY" "$PREDICTOR_DIR/scripts/historical_calibrate.py" "$@"; }
historical-calibrate(){ "$PREDICTOR_PY" "$PREDICTOR_DIR/scripts/historical_calibrate.py" "$@"; }