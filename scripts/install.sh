#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "==> Setting up Football Match Predictor"

# Create virtual environment if missing
if [ ! -d "$DIR/.venv" ]; then
    echo "==> Creating virtual environment..."
    python3 -m venv "$DIR/.venv"
fi

# Install dependencies
echo "==> Installing dependencies..."
"$DIR/.venv/bin/pip" install --quiet -r "$DIR/requirements.txt"

# Create symlink in ~/.local/bin
BINDIR="$HOME/.local/bin"
mkdir -p "$BINDIR"
if [ -f "$BINDIR/pr" ] && [ "$(readlink -f "$BINDIR/pr")" = "$DIR/predict.py" ]; then
    echo "==> Alias 'pr' already installed"
else
    # Use the wrapper script so ML mode is default from first run
    ln -sf "$DIR/pr" "$BINDIR/pr"
    chmod +x "$DIR/pr"
    echo "==> Alias created: $BINDIR/pr -> $DIR/pr"
fi

# Ensure ~/.local/bin is in PATH
case ":$PATH:" in
    *:$BINDIR:*) ;;
    *) echo "==> Add to your shell rc: export PATH=\"\$PATH:$BINDIR\"" ;;
esac

# Train ML model
echo "==> Training ML model..."
"$DIR/.venv/bin/python" "$DIR/ml_model.py" --train

# Run once to initialize DB + symlinks
echo "==> Initializing..."
"$DIR/.venv/bin/python" "$DIR/predict.py" --calibrate 2>/dev/null || true

echo ""
echo "✅  Ready! Usage:  pr https://www.forebet.com/en/football/matches/..."
