#!/usr/bin/env bash
# ============================================================================
# Football Match Predictor — Installer (Linux / macOS)
#
# Replicates the entire system on a new machine:
#   git clone <repo> predictor
#   cd predictor && bash scripts/install.sh
#
# - DB (history.db), calibration params and trained ML models come from git
# - Regenerated state (data/pending_results.json, leagues_db.json,
#   predictions.json) is restored from seed/data/
# - The `pr` command is symlinked into ~/.local/bin
#
# For native Windows (cmd / PowerShell) run scripts/install.ps1 instead.
# If this script is run from Git-Bash / MSYS it auto-delegates there.
# ============================================================================
set -e

# Repo root = parent of scripts/
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

# ── Detect Windows (Git-Bash / MSYS / Cygwin) ────────────────────────────────
if command -v uname >/dev/null 2>&1; then
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*)
            echo "==> Git-Bash/MSYS detected. Delegating to the PowerShell installer..."
            exec powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\\install.ps1"
            ;;
    esac
fi

echo "==> Setting up Football Match Predictor in $DIR"

# ── Python ────────────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON="python"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "ERROR: Python 3 not found. Install Python 3 first." >&2
    exit 1
fi

# ── Virtual environment ───────────────────────────────────────────────────────
if [ ! -x "$DIR/.venv/bin/python" ]; then
    echo "==> Creating virtual environment..."
    if ! "$PYTHON" -m venv "$DIR/.venv"; then
        echo "ERROR: Could not create venv. On Debian/Ubuntu install python3-venv first." >&2
        exit 1
    fi
fi
VENV_PY="$DIR/.venv/bin/python"

# ── Dependencies ──────────────────────────────────────────────────────────────
echo "==> Installing dependencies..."
"$DIR/.venv/bin/pip" install --quiet --upgrade pip || true
# xgboost declares nvidia-nccl as a hard dependency on Linux (~305 MB) even
# though CPU inference/training never loads it. Install every other package
# normally, then xgboost itself with --no-deps so no CUDA download is needed.
REQ_NO_XGB="$(mktemp)"
grep -viE '^[[:space:]]*xgboost([<>=!~ ]|$)' "$DIR/requirements.txt" > "$REQ_NO_XGB"
"$DIR/.venv/bin/pip" install --quiet -r "$REQ_NO_XGB"
rm -f "$REQ_NO_XGB"
"$DIR/.venv/bin/pip" install --quiet --no-deps xgboost

# ── Restore regenerated state (data/ is gitignored) ──────────────────────────
echo "==> Restoring data files..."
mkdir -p "$DIR/data"
for f in pending_results.json leagues_db.json predictions.json; do
    if [ ! -f "$DIR/data/$f" ] && [ -f "$DIR/seed/data/$f" ]; then
        cp "$DIR/seed/data/$f" "$DIR/data/$f"
        echo "    restored data/$f"
    fi
done

# ── ML models (trained models ship in git; only retrain if broken/missing) ──
if [ -f "$DIR/ml_models/ml_predictor/meta.json" ]; then
    echo "==> Verifying ML models..."
    if "$VENV_PY" -c "from ml_model import MLPredictor; m=MLPredictor.load(auto_train=False); assert m is not None and m.is_trained" 2>/dev/null; then
        echo "    models OK (loaded from git)"
    else
        echo "    models missing/incompatible — retraining from history.db..."
        "$VENV_PY" "$DIR/ml_model.py" --train
    fi
else
    echo "==> Training ML model..."
    "$VENV_PY" "$DIR/ml_model.py" --train
fi

# ── pr command (alias) ────────────────────────────────────────────────────────
# Every executable entry point gets a symlink, so `pr`, `pr2`, `book` and
# `pr-retrain` all work from any directory without activating the venv.
BINDIR="$HOME/.local/bin"
mkdir -p "$BINDIR"
for entry in pr pr2 book pr-retrain; do
    chmod +x "$DIR/$entry"
    ln -sf "$DIR/$entry" "$BINDIR/$entry"
done
ln -sf "$DIR/pr" "$BINDIR/predict"
ln -sf "$DIR/pr" "$BINDIR/predictor"
echo "==> Aliases created in $BINDIR: pr, pr2, book, pr-retrain, predict, predictor"

# ── Ensure ~/.local/bin is FIRST on PATH (so `pr` beats /usr/bin/pr) ────────
# ...and source the predictor aliases so every command works from any shell.
# The block is marker-wrapped and replaced on every re-install (never doubled).
# It goes into every rc file that exists: a login shell reads ~/.bash_profile
# (which may return from ~/.bashrc early when non-interactive), so fixing only
# ~/.bashrc would leave `bash -lc pr` resolving to coreutils' /usr/bin/pr.
PRED_BEGIN="# >>> predictor aliases >>>"
PRED_END="# <<< predictor aliases <<<"
PRED_BLOCK="$(cat <<EOF
$PRED_BEGIN
# Put ~/.local/bin FIRST so \`pr\` beats /usr/bin/pr (coreutils).
PATH="\$(printf ':%s:' "\$PATH" | sed "s#:$BINDIR:#:#g")"
PATH="\${PATH#:}"; PATH="\${PATH%:}"
export PATH="$BINDIR:\$PATH"
[ -f "$DIR/scripts/aliases.sh" ] && source "$DIR/scripts/aliases.sh"
$PRED_END
EOF
)"
INSTALLED_RC=""
for RC_FILE in "$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.zshrc" "$HOME/.zprofile" "$HOME/.profile"; do
    [ -f "$RC_FILE" ] || continue
    if grep -qF "$PRED_BEGIN" "$RC_FILE"; then
        TMP_RC="$(mktemp)"
        awk -v b="$PRED_BEGIN" -v e="$PRED_END" '
            $0 == b { skip = 1; next }
            $0 == e { skip = 0; next }
            !skip   { print }
        ' "$RC_FILE" > "$TMP_RC" && mv "$TMP_RC" "$RC_FILE"
    fi
    printf '\n%s\n' "$PRED_BLOCK" >> "$RC_FILE"
    INSTALLED_RC="$INSTALLED_RC $RC_FILE"
done
if [ -n "$INSTALLED_RC" ]; then
    echo "==> Added $BINDIR to PATH (prepended) + sources aliases in:$INSTALLED_RC"
else
    echo "==> Add to your shell rc: export PATH=\"$BINDIR:\$PATH\" && source $DIR/scripts/aliases.sh"
fi

# ── Verify install ────────────────────────────────────────────────────────────
echo "==> Verifying install (DB + models)..."
"$VENV_PY" -c "from database import DB_PATH; from ml_model import MLPredictor; m=MLPredictor.load(auto_train=False); print('    DB :', DB_PATH); print('    ML :', 'trained' if (m and m.is_trained) else 'NOT trained')"

echo ""
echo "✅  Ready! Usage:  pr https://www.forebet.com/en/football/matches/..."
echo "    pr <file>       Predict matches from a links file"
echo "    pr results      Scrape Forebet for scores -> update DB + HTML"
echo "    pr --calibrate  Show accuracy stats"
echo ""
echo "NOTE: ml_models (XGB/LGB) and seed/data/ must be committed to git for"
echo "      a fresh clone to carry them over.  See git status for untracked files."
