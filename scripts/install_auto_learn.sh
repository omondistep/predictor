#!/usr/bin/env bash
# ============================================================================
# Auto-Learn Scheduler — Install systemd timer or cron for continuous learning
# ============================================================================
# This script sets up automated learning so the predictor continuously
# improves by scraping results, analyzing bias, and retraining models.
#
# Usage:
#   ./install_auto_learn.sh              Interactive (choose method)
#   ./install_auto_learn.sh --status     Show current schedule
#   ./install_auto_learn.sh --systemd    Install systemd timer
#   ./install_auto_learn.sh --cron       Install crontab entry
#   ./install_auto_learn.sh --remove     Remove all scheduling
# ============================================================================

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$DIR/.venv/bin/python"

SYSTEMD_SERVICE="predictor-auto-learn.service"
SYSTEMD_TIMER="predictor-auto-learn.timer"

echo "========================================"
echo "  Auto-Learn Scheduler Installer"
echo "========================================"

if [ ! -f "$PYTHON" ]; then
    echo "Error: No virtual environment found. Run ./install.sh first."
    exit 1
fi

show_status() {
    echo ""
    echo "── Current Schedule ──"

    # Check systemd
    if systemctl --user list-timers 2>/dev/null | grep -q predictor-auto-learn; then
        echo "  ✓ Systemd timer: ACTIVE"
        systemctl --user status "$SYSTEMD_TIMER" 2>/dev/null | grep -E "Trigger|Active" | sed 's/^/    /'
    else
        echo "  ✗ Systemd timer: NOT INSTALLED"
    fi

    # Check cron
    if crontab -l 2>/dev/null | grep -q auto_learn.py; then
        echo "  ✓ Crontab: ACTIVE"
        crontab -l 2>/dev/null | grep auto_learn.py | sed 's/^/    /'
    else
        echo "  ✗ Crontab: NOT INSTALLED"
    fi

    # Show last run info from DB
    if [ -f "$DIR/history.db" ]; then
        echo ""
        "$PYTHON" -c "
from database import get_calibration_data_for_retraining, get_calibration_summary
d = get_calibration_data_for_retraining()
s = get_calibration_summary()
print(f'  Calibration entries: {d[\"total_calibration_entries\"]}')
print(f'  Last retrain:        {d[\"last_retrain_time\"] or \"never\"}')
print(f'  Overall accuracy:    {s[\"our_pct\"]}%')
"
    fi
}

install_systemd() {
    echo "  Installing systemd user service & timer..."

    mkdir -p "$HOME/.config/systemd/user"

    # Service unit
    cat > "$HOME/.config/systemd/user/$SYSTEMD_SERVICE" << EOF
[Unit]
Description=Predictor Auto-Learn (scrape results + calibrate + retrain)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=$PYTHON $DIR/auto_learn.py
WorkingDirectory=$DIR
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
EOF

    # Timer unit
    cat > "$HOME/.config/systemd/user/$SYSTEMD_TIMER" << EOF
[Unit]
Description=Run predictor auto-learn daily

[Timer]
OnCalendar=daily
Persistent=true
RandomizedDelaySec=1800

[Install]
WantedBy=timers.target
EOF

    systemctl --user daemon-reload
    systemctl --user enable "$SYSTEMD_TIMER"
    systemctl --user start "$SYSTEMD_TIMER"

    echo "  ✓ Systemd timer installed (runs daily with 30min random delay)"
    echo "  Next run:"
    systemctl --user status "$SYSTEMD_TIMER" 2>/dev/null | grep Trigger | sed 's/^/    /'
}

install_cron() {
    echo "  Installing crontab entry..."

    # Add twice-daily entry (6am and 6pm)
    CRON_LINE="0 6,18 * * * cd $DIR && $PYTHON $DIR/auto_learn.py >> $DIR/auto_learn.log 2>&1"

    # Check if already exists
    if crontab -l 2>/dev/null | grep -qF "$DIR/auto_learn.py"; then
        echo "  Crontab entry already exists. Skipping."
        echo "  Current entry:"
        crontab -l 2>/dev/null | grep auto_learn.py | sed 's/^/    /'
        return
    fi

    (crontab -l 2>/dev/null; echo "$CRON_LINE") | crontab -
    echo "  ✓ Crontab installed (runs at 6:00 and 18:00 daily)"
    echo "  Log: $DIR/auto_learn.log"
}

remove_scheduling() {
    echo "  Removing all scheduling..."

    # Remove systemd
    systemctl --user stop "$SYSTEMD_TIMER" 2>/dev/null || true
    systemctl --user disable "$SYSTEMD_TIMER" 2>/dev/null || true
    rm -f "$HOME/.config/systemd/user/$SYSTEMD_SERVICE" 2>/dev/null || true
    rm -f "$HOME/.config/systemd/user/$SYSTEMD_TIMER" 2>/dev/null || true
    systemctl --user daemon-reload 2>/dev/null || true

    # Remove cron
    crontab -l 2>/dev/null | grep -vF "$DIR/auto_learn.py" | crontab - 2>/dev/null || true

    echo "  ✓ All auto-learn scheduling removed"
}

# ── Main ──────────────────────────────────────────────────────────────────

case "${1:-}" in
    --status|-s)
        show_status
        exit 0
        ;;
    --systemd)
        install_systemd
        exit 0
        ;;
    --cron)
        install_cron
        exit 0
        ;;
    --remove)
        remove_scheduling
        exit 0
        ;;
esac

# Interactive mode
echo ""
echo "How would you like to schedule auto-learning?"
echo "  1) Systemd timer (recommended — runs daily, integrates with journal)"
echo "  2) Crontab (runs twice daily, logs to file)"
echo "  3) Show current status"
echo "  4) Remove all scheduling"
echo "  q) Quit"
echo ""
read -rp "Choice [1-4/q]: " choice

case "$choice" in
    1) install_systemd ;;
    2) install_cron ;;
    3) show_status ;;
    4) remove_scheduling ;;
    q|Q) echo "  No changes made." ;;
    *) echo "  Invalid choice." ;;
esac
