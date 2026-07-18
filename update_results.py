#!/usr/bin/env python3
"""Update existing prediction reports with match results.

Scans all HTML reports in predictions/ for matches without results,
re-scrapes them from Forebet, updates the HTML files, and regenerates index.html.
Avoids duplicating teams on the index.
"""

import re
import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from database import get_db, update_result, store_market_results
from predict import log, _update_index, _write_html


def parse_html_report(html_path: Path) -> dict:
    """Parse an HTML report to extract match info and URLs."""
    html = html_path.read_text(encoding="utf-8")
    
    # Extract all card data
    cards = []
    # Find cards with data-match-id (new format)
    card_pattern = r'<div class="card" data-match-id="(\d*)"'
    for match in re.finditer(card_pattern, html):
        match_id = int(match.group(1)) if match.group(1) else None
        if match_id:
            cards.append({"match_id": match_id})
    
    # Also extract from Forebet links
    link_pattern = r'<a href="(https://www\.forebet\.com/en/football/matches/[^"]+)">Forebet</a>'
    links = re.findall(link_pattern, html)
    
    # Extract teams
    team_pattern = r'<span class="teams">([^<]+) vs ([^<]+)</span>'
    teams = re.findall(team_pattern, html)
    
    # Check which have results (RESULT: line)
    has_result = 'RESULT:' in html
    
    # For old format without data-match-id, try to match by team names
    if not cards and teams and links:
        # Try to find matches in DB by team names
        conn = get_db()
        conn.row_factory = sqlite3.Row
        for i, (home, away) in enumerate(teams):
            match = conn.execute("""
                SELECT id FROM matches 
                WHERE home_team LIKE ? AND away_team LIKE ?
                ORDER BY id DESC LIMIT 1
            """, (f"%{home[:20]}%", f"%{away[:20]}%")).fetchone()
            if match:
                cards.append({"match_id": match["id"]})
        conn.close()
    
    return {
        "cards": cards,
        "links": links,
        "teams": teams,
        "has_result": has_result,
        "html": html,
    }


def check_unfinished_matches() -> list:
    """Find all HTML reports with matches that don't have results yet."""
    pred_dir = Path("predictions")
    if not pred_dir.exists():
        return []
    
    unfinished = []
    for html_file in sorted(pred_dir.glob("*.html")):
        if html_file.name == "index.html":
            continue
        
        report = parse_html_report(html_file)
        if not report["has_result"] and report["cards"]:
            unfinished.append({
                "path": html_file,
                "report": report,
            })
    
    return unfinished


def update_html_with_result(html_path: Path, match_id: int, home_goals: int, away_goals: int, 
                            ht_home: int = None, ht_away: int = None):
    """Update a specific HTML file with match result."""
    html = html_path.read_text(encoding="utf-8")
    
    # Determine outcome
    if home_goals > away_goals:
        outcome = "Home win"
    elif away_goals > home_goals:
        outcome = "Away win"
    else:
        outcome = "Draw"
    
    # Get our prediction from DB
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
    
    # Check if our pick was correct
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
            correct = None  # Push — stake returned
        else:
            correct = (pick == "Home" and outcome == "Home win") or (pick == "Away" and outcome == "Away win")
    elif mkt == "DC":
        if pick == "1X": correct = outcome in ("Home win", "Draw")
        elif pick == "X2": correct = outcome in ("Away win", "Draw")
        elif pick == "12": correct = outcome in ("Home win", "Away win")
    
    # Build result HTML
    ht_tag = f"  HT: {ht_home}-{ht_away}" if ht_home is not None and ht_away is not None else ""
    if correct is True:
        verdict = '<span style="color:#22c55e;font-weight:700">Correct!</span>'
    elif correct is False:
        verdict = f'<span style="color:#ef4444;font-weight:700">Incorrect</span> (picked {pick})'
    else:
        verdict = '<span style="color:#94a3b8;font-weight:700">Push (DNB)</span>'
    
    result_html = f'<div class="pick-line" style="color:#60a5fa;font-weight:700">RESULT: {home_goals} - {away_goals} ({outcome}){ht_tag}  {verdict}</div>'
    
    # Find the card and insert result after card-body div
    card_pattern = rf'(<div class="card" data-match-id="{match_id}"[^>]*>.*?<div class="card-body">)'
    replacement = r'\1\n  ' + result_html
    
    new_html = re.sub(card_pattern, replacement, html, count=1, flags=re.DOTALL)
    
    if new_html != html:
        html_path.write_text(new_html, encoding="utf-8")
        return True
    
    return False


def main():
    """Main update function."""
    log("Checking for unfinished matches...")
    
    unfinished = check_unfinished_matches()
    if not unfinished:
        log("No unfinished matches found.")
        return
    
    log(f"Found {len(unfinished)} reports with unfinished matches")
    
    # Collect all Forebet URLs to scrape
    all_urls = []
    for item in unfinished:
        for link in item["report"]["links"]:
            if link not in all_urls:
                all_urls.append(link)
    
    log(f"Need to check {len(all_urls)} Forebet URLs for results")
    
    # Limit to most recent 50 URLs to avoid timeout
    if len(all_urls) > 50:
        log(f"Limiting to most recent 50 URLs (of {len(all_urls)})")
        all_urls = all_urls[-50:]
    
    # Scrape results from Forebet
    from forebet_scraper import scrape_url
    import time
    
    updated_count = 0
    for i, url in enumerate(all_urls):
        try:
            # Skip if URL looks like a preview article (not a match page)
            if "match-previews" in url:
                continue
            
            log(f"  [{i+1}/{len(all_urls)}] Checking {url.split('/')[-1][:40]}...")
            data = scrape_url(url)
            if not data or not data.get("home_team"):
                time.sleep(0.5)
                continue
            
            # Check if this match has a result
            if data.get("actual_home_goals") is None:
                time.sleep(0.5)
                continue
            
            home_goals = data["actual_home_goals"]
            away_goals = data["actual_away_goals"]
            
            # Find match in DB
            conn = get_db()
            conn.row_factory = sqlite3.Row
            match = conn.execute("""
                SELECT id, home_team, away_team, actual_home_goals
                FROM matches 
                WHERE home_team LIKE ? AND away_team LIKE ?
                AND actual_home_goals IS NULL
                ORDER BY id DESC LIMIT 1
            """, (f"%{data.get('home_team', '')[:15]}%", 
                  f"%{data.get('away_team', '')[:15]}%",)).fetchone()
            conn.close()
            
            if not match or match["actual_home_goals"] is not None:
                continue
            
            match_id = match["id"]
            
            # Update DB
            update_result(match_id, home_goals, away_goals)
            
            # Update all HTML files that contain this match
            pred_dir = Path("predictions")
            for html_file in pred_dir.glob("*.html"):
                if html_file.name == "index.html":
                    continue
                
                report = parse_html_report(html_file)
                for card in report["cards"]:
                    if card["match_id"] == match_id:
                        if update_html_with_result(html_file, match_id, home_goals, away_goals):
                            log(f"  Updated {html_file.name}: {match['home_team']} vs {match['away_team']} {home_goals}-{away_goals}")
                            updated_count += 1
                            break
            
        except Exception as e:
            log(f"  Error processing {url}: {e}")
            continue
    
    if updated_count > 0:
        # Regenerate index.html
        log(f"Updated {updated_count} matches, regenerating index...")
        _update_index(Path("predictions"), datetime.now().strftime("%Y-%m-%d %H:%M"))
        log("Index updated.")
    else:
        log("No new results found.")


if __name__ == "__main__":
    main()
