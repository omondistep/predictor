#!/usr/bin/env python3
"""Update prediction results by scraping Forebet for finished match scores.

Reads from data/pending_results.json (additive tracking file) instead of
scanning HTML reports.  After updating the DB and any matching HTML files,
completed entries are removed from the pending file.
"""

import re
import sys
import json
import sqlite3
import time
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))
from database import (
    get_db, update_result, store_market_results,
    get_pending_results, remove_pending_result,
)
from predict import log


def _match_started(match_id: int) -> bool:
    """Check if a match has started based on its scheduled date/time.

    Returns True if the match time is in the past or unknown.
    Returns False if the match is still in the future.
    """
    conn = get_db()
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT match_date, match_time FROM matches WHERE id = ?", (match_id,)
    ).fetchone()
    conn.close()

    if not row or not row["match_date"] or not row["match_time"]:
        return True  # unknown → don't skip

    try:
        dt_str = f"{row['match_date']} {row['match_time']}"
        match_dt = datetime.strptime(dt_str, "%d/%m/%Y %H:%M")
        return datetime.now() >= match_dt
    except Exception:
        return True  # parse error → don't skip


def update_html_with_result(html_path: Path, match_id: int, home_goals: int, away_goals: int,
                            ht_home: int = None, ht_away: int = None):
    """Update a specific HTML file with match result."""
    html = html_path.read_text(encoding="utf-8")

    if home_goals > away_goals:
        outcome = "Home win"
    elif away_goals > home_goals:
        outcome = "Away win"
    else:
        outcome = "Draw"

    conn = get_db()
    conn.row_factory = sqlite3.Row
    match = conn.execute("""
        SELECT our_market, our_prediction, our_confidence, home_team, away_team
        FROM matches WHERE id = ?
    """, (match_id,)).fetchone()
    conn.close()

    if not match:
        return False

    mkt = match["our_market"] or ""
    pick = match["our_prediction"] or ""

    correct = False
    if mkt == "1X2":
        correct = (pick == outcome)
    elif mkt == "O/U":
        total = home_goals + away_goals
        if "Over" in pick:
            correct = (total > float(pick.split()[-1]))
        elif "Under" in pick:
            correct = (total <= float(pick.split()[-1]))
    elif mkt == "BTTS":
        both = home_goals > 0 and away_goals > 0
        correct = (pick == "Yes" and both) or (pick == "No" and not both)
    elif mkt == "DNB":
        if outcome == "Draw":
            correct = None
        else:
            correct = (pick == "Home" and outcome == "Home win") or (pick == "Away" and outcome == "Away win")
    elif mkt == "DC":
        if pick == "1X": correct = outcome in ("Home win", "Draw")
        elif pick == "X2": correct = outcome in ("Away win", "Draw")
        elif pick == "12": correct = outcome in ("Home win", "Away win")

    ht_tag = f"  HT: {ht_home}-{ht_away}" if ht_home is not None and ht_away is not None else ""
    if correct is True:
        verdict = '<span style="color:#22c55e;font-weight:700">Correct!</span>'
    elif correct is False:
        verdict = f'<span style="color:#ef4444;font-weight:700">Incorrect</span> (picked {pick})'
    else:
        verdict = '<span style="color:#94a3b8;font-weight:700">Push (DNB)</span>'

    result_html = f'<div class="pick-line" style="color:#60a5fa;font-weight:700">RESULT: {home_goals} - {away_goals} ({outcome}){ht_tag}  {verdict}</div>'

    card_pattern = rf'(<div class="card" data-match-id="{match_id}"[^>]*>.*?<div class="card-body">)'
    replacement = r'\1\n  ' + result_html

    new_html = re.sub(card_pattern, replacement, html, count=1, flags=re.DOTALL)

    if new_html != html:
        html_path.write_text(new_html, encoding="utf-8")
        return True

    return False


def main():
    """Main update function."""
    log("Checking pending results...")

    pending = get_pending_results()
    if not pending:
        log("No pending results to check.")
        return

    log(f"Found {len(pending)} pending matches")

    from forebet_scraper import scrape_url

    updated_count = 0
    for entry in list(pending):
        url = entry.get("forebet_url", "")
        match_id = entry.get("match_id")

        if not url or not match_id:
            continue

        if not _match_started(match_id):
            log(f"  Skipping {url.split('/')[-1][:40]}... (match not started yet)")
            continue

        try:
            slug = url.split("/")[-1][:40]
            log(f"  Checking {slug}...")

            data = scrape_url(url)
            if not data or not data.get("home_team"):
                time.sleep(0.5)
                continue

            if data.get("actual_home_goals") is None:
                time.sleep(0.5)
                continue

            home_goals = data["actual_home_goals"]
            away_goals = data["actual_away_goals"]

            conn = get_db()
            conn.row_factory = sqlite3.Row
            match = conn.execute("""
                SELECT id, home_team, away_team, actual_home_goals
                FROM matches WHERE id = ?
            """, (match_id,)).fetchone()
            conn.close()

            if not match:
                log(f"  Match {match_id} not in DB, skipping")
                continue

            if match["actual_home_goals"] is not None:
                remove_pending_result(url)
                continue

            update_result(match_id, home_goals, away_goals)

            pred_dir = Path(__file__).parent / "predictions"
            if pred_dir.exists():
                for html_file in pred_dir.glob("*.html"):
                    if html_file.name == "index.html":
                        continue
                    card_pattern = r'<div class="card" data-match-id="(\d*)"'
                    html = html_file.read_text(encoding="utf-8")
                    if re.search(card_pattern, html) and f'data-match-id="{match_id}"' in html:
                        if update_html_with_result(html_file, match_id, home_goals, away_goals):
                            log(f"  Updated {html_file.name}: {match['home_team']} vs {match['away_team']} {home_goals}-{away_goals}")
                            break

            remove_pending_result(url)
            updated_count += 1

        except Exception as e:
            log(f"  Error processing {url}: {e}")
            continue

    if updated_count > 0:
        log(f"Updated {updated_count} matches.")

        # ── Auto-learn: calibrate + retrain if enough new data ──
        log("Running calibration learning...")
        try:
            from calibration_learner import run_calibration_learning
            run_calibration_learning(analyze=True, retrain=True, report=False)
        except Exception as e:
            log(f"  Calibration learning skipped: {e}")
    else:
        log("No new results found.")


if __name__ == "__main__":
    main()
