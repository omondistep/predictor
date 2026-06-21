#!/usr/bin/env python3
"""Update DB with results from scraped_results.json using a single connection."""
import json
import sys
from database import get_db, _prediction_correct, _update_league_stats, _update_component_accuracy

RESULTS_FILE = "scraped_results.json"

results = json.load(open(RESULTS_FILE))
print(f"Total results in file: {len(results)}", file=sys.stderr)

conn = get_db()

# Build URL → match_id lookup
url_to_id = {}
for row in conn.execute("SELECT id, forebet_url FROM matches WHERE forebet_url IS NOT NULL"):
    url_to_id[row["forebet_url"]] = row["id"]
print(f"DB has {len(url_to_id)} matches with URLs", file=sys.stderr)

updated = 0
skipped_no_match = 0
errors = 0

for r in results:
    url = r["url"]
    match_id = url_to_id.get(url)
    if not match_id:
        skipped_no_match += 1
        continue

    home_goals, away_goals = r["home_goals"], r["away_goals"]
    result_label = (
        "Home win" if home_goals > away_goals
        else "Away win" if away_goals > home_goals
        else "Draw"
    )

    try:
        # Update match record
        conn.execute("""
            UPDATE matches SET
                actual_home_goals = ?, actual_away_goals = ?,
                actual_result = ?, reviewed = 1,
                reviewed_at = datetime('now')
            WHERE id = ?
        """, (home_goals, away_goals, result_label, match_id))

        # Fetch match for calibration log
        match = conn.execute("""
            SELECT our_prediction, our_confidence, forebet_pred, league,
                   our_stake, our_market, method_used, odds_home, odds_draw, odds_away,
                   odds_over25, odds_under25
            FROM matches WHERE id = ?
        """, (match_id,)).fetchone()

        if match:
            our_correct = 1 if _prediction_correct(match["our_prediction"], home_goals, away_goals) else 0
            fb_correct = 1 if match["forebet_pred"] and _prediction_correct(match["forebet_pred"], home_goals, away_goals) else 0

            # Pick the relevant odds for this market
            odds = None
            market = match["our_market"] or ""
            pick = match["our_prediction"] or ""
            if market == "1X2":
                odds_map = {"Home win": match["odds_home"], "Draw": match["odds_draw"], "Away win": match["odds_away"]}
                odds = odds_map.get(pick)
            elif market == "O/U":
                odds = match["odds_over25"] if "Over" in pick else match["odds_under25"]
            elif market == "BTTS":
                odds = match["odds_btts_yes"] if pick == "Yes" else match["odds_btts_no"]
            elif market == "DNB":
                odds = match["odds_home"] if "Home" in pick else match["odds_away"]
            elif market == "DC":
                odds = match["odds_home"]

            conn.execute("""
                INSERT INTO calibration_log
                    (league, match_id, our_prediction, actual_result,
                     correct, confidence, forebet_pred, forebet_correct,
                     method_used, market, stake, odds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                match["league"], match_id, match["our_prediction"], result_label,
                our_correct, match["our_confidence"], match["forebet_pred"], fb_correct,
                match["method_used"], match["our_market"],
                match["our_stake"], odds
            ))
            _update_league_stats(conn, match["league"])
            # We need to pass dict to _update_component_accuracy
            match_dict = dict(match)
            _update_component_accuracy(conn, match_dict, our_correct)

        updated += 1
        if updated % 25 == 0:
            conn.commit()
            print(f"  [{updated}]...", file=sys.stderr, flush=True)

    except Exception as e:
        print(f"  [ER] match_id={match_id} ({url.split('/')[-1][:40]}): {e}", file=sys.stderr)
        errors += 1

conn.commit()
conn.close()

print(f"\nDone. Updated: {updated}, Skipped (no DB match): {skipped_no_match}, Errors: {errors}", file=sys.stderr)
