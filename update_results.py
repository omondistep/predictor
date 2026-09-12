#!/usr/bin/env python3
"""Update prediction results by scraping Forebet for finished match scores.

Reads from data/pending_results.json (additive tracking file) instead of
scanning HTML reports.  After updating the DB and any matching HTML files,
completed entries are removed from the pending file.

Fetching strategy (defeats Cloudflare rate-limiting):
  1. Batch pass — one getrs.php AJAX call *per distinct match date*. That
     endpoint returns every finished match (with scores) for a day in a single
     JSON response, so ~700 per-match pages collapse into ~50 date requests.
  2. Fallback pass — any match the date pages no longer carry is fetched via
     the per-page scraper (which itself reuses baked cf_clearance cookies).
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


def _fetch_results_by_date(date_iso: str):
    """Fetch ALL finished matches for one date via Forebet's getrs.php AJAX.

    Returns (map, result) where map is ``str(forebet_id) -> (hg, ag, comment)``
    for finished matches, or None when the fetch was blocked/failed, plus the
    FetchResult so callers can distinguish a Cloudflare block from a glitch.
    """
    from webfetch import fetch_with_retry

    url = ("https://www.forebet.com/scripts/getrs.php?ln=en&tp=1x2"
           f"&in={date_iso}&ord=0&tz=%2B180&tzs=0&tze=0")
    res = fetch_with_retry(url, timeout=30, max_attempts=4)
    if not res.ok:
        return None, res
    try:
        data = json.loads(res.text)
    except (ValueError, TypeError):
        return None, res
    if not isinstance(data, list) or not data:
        return {}, res
    matches = data[0]
    out = {}
    for m in matches:
        comment = (m.get("comment") or "").strip()
        if comment not in ("FT", "AET", "Pen."):
            continue
        try:
            hg = int(m["Host_SC"])
            ag = int(m["Guest_SC"])
        except (KeyError, TypeError, ValueError):
            continue
        out[str(m.get("id"))] = (hg, ag, comment)
    return out, res


def _looks_cloudflare_blocked(res) -> bool:
    """True when a fetch result is a Cloudflare block, not a mere miss."""
    text = getattr(res, "text", "") or ""
    if text:
        from webfetch import _looks_blocked
        return _looks_blocked(text)
    return getattr(res, "status_code", 0) == 403


def _forebet_id_from_url(url: str):
    """Extract the trailing Forebet match id from a match URL (or None).

    e.g. https://www.forebet.com/en/football/matches/bokelj-kotor-fk-otrant-2484797
    -> "2484797".  getrs.php identifies matches by this id, which is NOT the
    local DB match_id.
    """
    if not url:
        return None
    tail = url.rstrip("/").rsplit("-", 1)[-1]
    return tail if tail.isdigit() else None


def _apply_result(match_id: int, url: str, row, home_goals: int, away_goals: int,
                  ht_home_goals=None, ht_away_goals=None) -> bool:
    """Write one result to the DB + HTML card and drop it from pending.

    Returns True when the result was newly written (not already present).
    """
    conn = get_db()
    cur = conn.execute(
        "SELECT actual_home_goals FROM matches WHERE id = ?", (match_id,)
    ).fetchone()
    conn.close()
    if cur and cur[0] is not None:
        remove_pending_result(url)
        return False

    update_result(match_id, home_goals, away_goals, ht_home_goals, ht_away_goals)

    pred_dir = Path(__file__).parent / "predictions"
    if pred_dir.exists():
        for html_file in pred_dir.glob("*.html"):
            if html_file.name == "index.html":
                continue
            card_pattern = r'<div class="card" data-match-id="(\d*)"'
            html = html_file.read_text(encoding="utf-8")
            if re.search(card_pattern, html) and f'data-match-id="{match_id}"' in html:
                if update_html_with_result(html_file, match_id, home_goals, away_goals,
                                           ht_home=ht_home_goals, ht_away=ht_away_goals):
                    log(f"  Updated {html_file.name}: {row['home_team']} vs "
                        f"{row['away_team']} {home_goals}-{away_goals}")
                    break

    remove_pending_result(url)
    return True


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
    elif mkt == "1H":
        # First-Half picks are decided by the HT score, not the final one
        if ht_home is not None and ht_away is not None:
            if ht_home > ht_away:
                ht_outcome = "Home win"
            elif ht_home < ht_away:
                ht_outcome = "Away win"
            else:
                ht_outcome = "Draw"
            correct = (pick == ht_outcome)
        else:
            correct = None

    ht_tag = f"  HT: {ht_home}-{ht_away}" if ht_home is not None and ht_away is not None else ""
    if correct is True:
        verdict = '<span style="color:#22c55e;font-weight:700">Correct!</span>'
    elif correct is False:
        verdict = f'<span style="color:#ef4444;font-weight:700">Incorrect</span> (picked {pick})'
    elif mkt == "1H":
        verdict = '<span style="color:#94a3b8;font-weight:700">Awaiting HT settle (1H)</span>'
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

    # Load DB metadata for all pending matches once.
    conn = get_db()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, home_team, away_team, actual_home_goals, "
        "match_date, match_time FROM matches"
    ).fetchall()
    conn.close()
    db_by_id = {r["id"]: r for r in rows}

    # ── Pass 1: batch fetch — one getrs.php AJAX call per distinct date ──
    by_date = {}
    for entry in pending:
        mid = entry.get("match_id")
        row = db_by_id.get(mid)
        if not row or not row["match_date"]:
            continue
        by_date.setdefault(row["match_date"], []).append(entry)

    updated_count = 0
    handled_urls = set()
    today = datetime.now().date()

    date_keys = sorted(by_date)
    for i, d in enumerate(date_keys, 1):
        try:
            iso = datetime.strptime(d, "%d/%m/%Y").strftime("%Y-%m-%d")
            is_future = datetime.strptime(d, "%d/%m/%Y").date() > today
        except ValueError:
            continue
        if is_future:
            continue  # no results exist yet

        log(f"  [{i}/{len(date_keys)}] {d}: fetch finished results ...")
        results, res = _fetch_results_by_date(iso)

        # A Cloudflare block on the AJAX endpoint aborts the whole run.
        if results is None:
            if _looks_cloudflare_blocked(res):
                log("  Cloudflare rate-limited (blocked) — stopping results "
                    "update; run again later")
                return
            log(f"  [scraper] no data for {iso} — falling back per-page")
            continue

        for entry in by_date[d]:
            url = entry.get("forebet_url", "")
            mid = entry.get("match_id")
            if not url or not mid:
                continue
            row = db_by_id.get(mid)
            if not row:
                log(f"  Match {mid} not in DB, skipping")
                continue
            fid = _forebet_id_from_url(url)
            if fid is None:
                continue
            score = results.get(fid)
            if score is None:
                continue  # not finished / absent on the date page → fallback
            hg, ag, _comment = score
            if _apply_result(mid, url, row, hg, ag):
                updated_count += 1
            handled_urls.add(url)

    # ── Pass 2: fallback — per-page scrape for anything the batch missed ──
    backlog = [e for e in pending if e.get("forebet_url") not in handled_urls]
    if backlog:
        log(f"  {len(backlog)} matches not covered by batch — per-page fallback")
    for entry in backlog:
        url = entry.get("forebet_url", "")
        mid = entry.get("match_id")
        if not url or not mid:
            continue
        row = db_by_id.get(mid)
        if not row:
            continue
        if row["actual_home_goals"] is not None:
            remove_pending_result(url)
            continue
        if not _match_started(mid):
            log(f"  Skipping {url.split('/')[-1][:40]}... (match not started yet)")
            continue
        try:
            slug = url.split("/")[-1][:40]
            log(f"  Checking {slug}...")
            data = scrape_url(url)
            if not data or not data.get("home_team"):
                if data and data.get("_blocked"):
                    log("  Cloudflare rate-limited (blocked) — stopping results "
                        "update; run again later")
                    return
                time.sleep(0.5)
                continue
            if data.get("actual_home_goals") is None:
                time.sleep(0.5)
                continue
            if _apply_result(mid, url, row, data["actual_home_goals"],
                             data["actual_away_goals"],
                             data.get("ht_home_goals"),
                             data.get("ht_away_goals")):
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

        # ── Retrain ensemble weights from history ──
        log("Retraining ensemble weights from history...")
        try:
            from database import train_weights_from_history
            result = train_weights_from_history(market="1X2", min_samples=10)
            log(f"  Updated weights for {result['leagues_updated']} leagues")
        except Exception as e:
            log(f"  Weight retraining skipped: {e}")
    else:
        log("No new results found.")


if __name__ == "__main__":
    main()
