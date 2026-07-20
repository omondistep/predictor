#!/usr/bin/env python3
"""Backtest: re-predict finished matches from DB data and compare old vs new."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from database import init_db, get_db
from predict import analyze_from_data, detect_league, get_profile

def db_row_to_data(row: dict) -> dict:
    """Map DB row to the data dict format expected by analyze_from_data."""
    data = {}
    for k, v in row.items():
        if v is not None:
            data[k] = v

    # Ensure odds are floats
    for k in ('odds_home', 'odds_draw', 'odds_away', 'odds_over25', 'odds_under25',
              'odds_btts_yes', 'odds_btts_no', 'odds_over15', 'odds_under15'):
        if data.get(k) is not None:
            try:
                data[k] = float(data[k])
            except (ValueError, TypeError):
                pass

    # Ensure pct fields are floats
    for k in ('forebet_home_pct', 'forebet_draw_pct', 'forebet_away_pct',
              'forebet_over25_pct', 'forebet_btts_yes_pct',
              'home_avg_goals_for', 'home_avg_goals_against',
              'away_avg_goals_for', 'away_avg_goals_against',
              'home_home_avg_goals_for', 'home_home_avg_goals_against',
              'away_away_avg_goals_for', 'away_away_avg_goals_against'):
        if data.get(k) is not None:
            try:
                data[k] = float(data[k])
            except (ValueError, TypeError):
                pass

    # actual_home_goals / actual_away_goals
    if data.get('actual_home_goals') is not None:
        data['actual_home_goals'] = int(data['actual_home_goals'])
    if data.get('actual_away_goals') is not None:
        data['actual_away_goals'] = int(data['actual_away_goals'])

    return data


def check_correct(pred: dict, data: dict) -> bool | None:
    """Check if a prediction is correct. None = push."""
    hg = data.get('actual_home_goals')
    ag = data.get('actual_away_goals')
    if hg is None or ag is None:
        return None
    if hg > ag:
        actual = "Home win"
    elif hg < ag:
        actual = "Away win"
    else:
        actual = "Draw"

    pick = pred.get("pick", "")
    market = pred.get("market", "")
    if market == "1X2":
        return pick == actual
    elif market == "O/U":
        total = hg + ag
        if "Over" in pick:
            return total > float(pick.split()[-1])
        elif "Under" in pick:
            return total <= float(pick.split()[-1])
    elif market == "BTTS":
        both = hg > 0 and ag > 0
        return (pick == "Yes" and both) or (pick == "No" and not both)
    elif market == "DNB":
        if actual == "Draw":
            return None
        return (pick == "Home" and actual == "Home win") or (pick == "Away" and actual == "Away win")
    elif market == "DC":
        if pick == "1X": return actual in ("Home win", "Draw")
        elif pick == "X2": return actual in ("Away win", "Draw")
        elif pick == "12": return actual in ("Home win", "Away win")
    return None


def main():
    init_db()
    conn = get_db()

    rows = conn.execute("""
        SELECT * FROM matches
        WHERE actual_result IS NOT NULL AND match_date = '19/07/2026'
        ORDER BY id
    """).fetchall()

    col_names = [c["name"] for c in conn.execute("PRAGMA table_info(matches)").fetchall()]

    print(f"Re-predicting {len(rows)} matches from 19/07/2026...\n")

    old_correct = 0
    new_correct = 0
    total = 0
    old_by_market = {}
    new_by_market = {}
    old_by_league = {}
    new_by_league = {}
    market_switches = 0

    for row_vals in rows:
        row = dict(zip(col_names, row_vals))
        data = db_row_to_data(row)

        # Old prediction
        old_pick = row.get("our_prediction", "?")
        old_market = row.get("our_market", "?")
        old_conf = row.get("our_confidence", "?")
        old_method = row.get("method_used", "?")

        # New prediction
        try:
            pred = analyze_from_data(data, use_ml=True)
        except Exception as e:
            print(f"  ERROR: {row['home_team']} vs {row['away_team']}: {e}")
            continue

        new_pick = pred.get("pick", "?")
        new_market = pred.get("market", "?")
        new_conf = pred.get("confidence", "?")

        # Check correctness
        hg = data.get("actual_home_goals", 0)
        ag = data.get("actual_away_goals", 0)
        actual = "Home win" if hg > ag else ("Away win" if hg < ag else "Draw")
        score = f"{hg}-{ag}"

        old_c = check_correct({"pick": old_pick, "market": old_market}, data)
        new_c = check_correct({"pick": new_pick, "market": new_market}, data)

        if old_c is not None:
            old_correct += int(old_c)
        if new_c is not None:
            new_correct += int(new_c)
        total += 1

        # Per-market stats
        if old_c is not None:
            old_by_market.setdefault(old_market, [0, 0])
            old_by_market[old_market][0] += 1
            if old_c:
                old_by_market[old_market][1] += 1
        if new_c is not None:
            new_by_market.setdefault(new_market, [0, 0])
            new_by_market[new_market][0] += 1
            if new_c:
                new_by_market[new_market][1] += 1

        # Per-league
        league = detect_league(row.get("league", ""))
        if old_c is not None:
            old_by_league.setdefault(league, [0, 0])
            old_by_league[league][0] += 1
            if old_c:
                old_by_league[league][1] += 1
        if new_c is not None:
            new_by_league.setdefault(league, [0, 0])
            new_by_league[league][0] += 1
            if new_c:
                new_by_league[league][1] += 1

        # Track market switches
        if old_market != new_market:
            market_switches += 1

        # Print detail
        mark = ""
        if old_c != new_c:
            if new_c and not old_c:
                mark = " \033[92m← FIXED\033[0m"
            elif not new_c and old_c:
                mark = " \033[91m← BROKE\033[0m"

        old_sym = "✓" if old_c else ("—" if old_c is None else "✗")
        new_sym = "✓" if new_c else ("—" if new_c is None else "✗")

        print(f"{row['home_team'][:18]:18s} vs {row['away_team'][:18]:18s} {score:5s} | "
              f"Old: {old_pick[:15]:15s} ({old_market:4s}) {old_sym} | "
              f"New: {new_pick[:15]:15s} ({new_market:4s}) {new_sym}{mark}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {total} matches")
    print(f"Old model: {old_correct}/{total} = {old_correct/total*100:.1f}%")
    print(f"New model: {new_correct}/{total} = {new_correct/total*100:.1f}%")
    print(f"Market switches: {market_switches}/{total}")

    print(f"\n{'='*60}")
    print("BY MARKET (New model):")
    for mkt, (t, c) in sorted(new_by_market.items()):
        print(f"  {mkt:6s}: {c}/{t} = {c/t*100:.1f}%")

    print(f"\nOLD BY MARKET:")
    for mkt, (t, c) in sorted(old_by_market.items()):
        print(f"  {mkt:6s}: {c}/{t} = {c/t*100:.1f}%")

    conn.close()


if __name__ == "__main__":
    main()
