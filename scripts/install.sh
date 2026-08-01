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
"$DIR/.venv/bin/pip" install --quiet -r "$DIR/requirements.txt"

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
BINDIR="$HOME/.local/bin"
mkdir -p "$BINDIR"
chmod +x "$DIR/pr"
ln -sf "$DIR/pr" "$BINDIR/pr"
ln -sf "$DIR/pr" "$BINDIR/predict"
ln -sf "$DIR/pr" "$BINDIR/predictor"
echo "==> Alias created: $BINDIR/pr -> $DIR/pr"

# ── Ensure ~/.local/bin is on PATH ───────────────────────────────────────────
if [[ ":$PATH:" != *":$BINDIR:"* ]]; then
    RC_FILE=""
    [ -f "$HOME/.zshrc" ] && RC_FILE="$HOME/.zshrc"
    [ -z "$RC_FILE" ] && [ -f "$HOME/.bashrc" ] && RC_FILE="$HOME/.bashrc"
    if [ -n "$RC_FILE" ]; then
        if ! grep -qF "PATH.*$BINDIR" "$RC_FILE" 2>/dev/null; then
            printf '\n# Predictor\ncase ":$PATH:" in *:%s:*) ;; *) export PATH="$PATH:%s" ;; esac\n' "$BINDIR" "$BINDIR" >> "$RC_FILE"
            echo "==> Added $BINDIR to PATH ($RC_FILE)"
        fi
    else
        echo "==> Add to your shell rc: export PATH=\"\$PATH:$BINDIR\""
    fi
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
