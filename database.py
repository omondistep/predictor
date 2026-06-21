"""
SQLite database module for prediction storage, review, and calibration.

Stores every prediction with all scraped features, tracks actual results
after review, and supports model calibration based on historical accuracy.
"""

import sqlite3
import os
import re
import json
from datetime import datetime
from pathlib import Path

DB_DIR = Path(__file__).parent
DB_PATH = DB_DIR / "history.db"

SCHEMA_TABLES = """
CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY,
    forebet_url TEXT UNIQUE,
    home_team TEXT,
    away_team TEXT,
    league TEXT,
    match_date TEXT,
    match_time TEXT,

    -- Scraped features
    home_form TEXT,
    away_form TEXT,
    home_pos INTEGER,
    away_pos INTEGER,
    home_pts REAL,
    away_pts REAL,
    home_games_played INTEGER,
    away_games_played INTEGER,
    h2h_home_wins INTEGER,
    h2h_draws INTEGER,
    h2h_away_wins INTEGER,
    h2h_matches INTEGER,
    home_avg_goals_for REAL,
    home_avg_goals_against REAL,
    away_avg_goals_for REAL,
    away_avg_goals_against REAL,

    -- Scraped odds
    odds_home REAL,
    odds_draw REAL,
    odds_away REAL,
    odds_over25 REAL,
    odds_under25 REAL,
    odds_btts_yes REAL,
    odds_btts_no REAL,

    -- Forebet prediction
    forebet_pred TEXT,
    forebet_home_pct REAL,
    forebet_draw_pct REAL,
    forebet_away_pct REAL,
    forebet_over25_pct REAL,
    forebet_btts_yes_pct REAL,

    -- Our prediction
    our_prediction TEXT,
    our_confidence TEXT,
    our_score_lean TEXT,
    our_stake REAL DEFAULT 0,
    our_market TEXT,

    -- Which model method was used
    method_used TEXT,
    poisson_prob_home REAL,
    poisson_prob_draw REAL,
    poisson_prob_away REAL,
    ml_prob_home REAL,
    ml_prob_draw REAL,
    ml_prob_away REAL,
    forebet_prob_home REAL,
    forebet_prob_draw REAL,
    forebet_prob_away REAL,

    -- Actual result (filled on review)
    actual_home_goals INTEGER,
    actual_away_goals INTEGER,
    actual_result TEXT,
    reviewed INTEGER DEFAULT 0,

    created_at TEXT DEFAULT (datetime('now')),
    reviewed_at TEXT
);

CREATE TABLE IF NOT EXISTS calibration_log (
    id INTEGER PRIMARY KEY,
    league TEXT,
    match_id INTEGER,
    our_prediction TEXT,
    actual_result TEXT,
    correct INTEGER,
    confidence TEXT,
    forebet_pred TEXT,
    forebet_correct INTEGER,
    method_used TEXT,
    market TEXT,
    stake REAL,
    odds REAL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS league_stats (
    league TEXT PRIMARY KEY,
    total_predictions INTEGER DEFAULT 0,
    correct_predictions INTEGER DEFAULT 0,
    under25_pct REAL,
    btts_no_pct REAL,
    draw_pct REAL,
    home_win_pct REAL,
    last_updated TEXT
);

CREATE TABLE IF NOT EXISTS component_accuracy (
    component TEXT,
    league TEXT,
    market TEXT,
    total INTEGER DEFAULT 0,
    correct INTEGER DEFAULT 0,
    last_updated TEXT,
    PRIMARY KEY (component, league, market)
);

CREATE TABLE IF NOT EXISTS kelly_log (
    id INTEGER PRIMARY KEY,
    match_id INTEGER,
    market TEXT,
    pick TEXT,
    model_prob REAL,
    implied_prob REAL,
    kelly_stake REAL,
    odds REAL,
    result INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);
"""

SCHEMA_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_matches_league ON matches(league);
CREATE INDEX IF NOT EXISTS idx_matches_reviewed ON matches(reviewed);
CREATE INDEX IF NOT EXISTS idx_calibration_league ON calibration_log(league);
CREATE INDEX IF NOT EXISTS idx_calibration_method ON calibration_log(method_used);
"""


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_db()
    # 1. Create tables (no-op if they exist)
    conn.executescript(SCHEMA_TABLES)
    # 2. Migrate existing tables that may be missing new columns
    _migrate(conn)
    # 3. Create indexes (after migration so new columns exist)
    conn.executescript(SCHEMA_INDEXES)
    conn.commit()
    conn.close()


_MIGRATIONS = [
    ("matches", "our_stake", "REAL DEFAULT 0"),
    ("matches", "our_market", "TEXT"),
    ("matches", "method_used", "TEXT"),
    ("matches", "poisson_prob_home", "REAL"),
    ("matches", "poisson_prob_draw", "REAL"),
    ("matches", "poisson_prob_away", "REAL"),
    ("matches", "ml_prob_home", "REAL"),
    ("matches", "ml_prob_draw", "REAL"),
    ("matches", "ml_prob_away", "REAL"),
    ("matches", "forebet_prob_home", "REAL"),
    ("matches", "forebet_prob_draw", "REAL"),
    ("matches", "forebet_prob_away", "REAL"),
    ("matches", "h2h_goals_for", "INTEGER DEFAULT 0"),
    ("matches", "h2h_goals_against", "INTEGER DEFAULT 0"),
    ("matches", "h2h_avg_total_goals", "REAL DEFAULT 0"),
    ("matches", "h2h_weighted_form", "REAL DEFAULT 0.5"),
    ("matches", "home_home_avg_goals_for", "REAL"),
    ("matches", "home_home_avg_goals_against", "REAL"),
    ("matches", "away_away_avg_goals_for", "REAL"),
    ("matches", "away_away_avg_goals_against", "REAL"),
    ("matches", "home_over15_pct", "INTEGER"),
    ("matches", "home_under15_pct", "INTEGER"),
    ("matches", "away_over15_pct", "INTEGER"),
    ("matches", "away_under15_pct", "INTEGER"),
    ("matches", "home_over25_pct", "INTEGER"),
    ("matches", "home_under25_pct", "INTEGER"),
    ("matches", "away_over25_pct", "INTEGER"),
    ("matches", "away_under25_pct", "INTEGER"),
    ("matches", "home_over35_pct", "INTEGER"),
    ("matches", "home_under35_pct", "INTEGER"),
    ("matches", "away_over35_pct", "INTEGER"),
    ("matches", "away_under35_pct", "INTEGER"),
    ("matches", "home_btts_yes_pct", "INTEGER"),
    ("matches", "home_btts_no_pct", "INTEGER"),
    ("matches", "away_btts_yes_pct", "INTEGER"),
    ("matches", "away_btts_no_pct", "INTEGER"),
    ("matches", "home_scored_pct", "INTEGER"),
    ("matches", "home_conceded_pct", "INTEGER"),
    ("matches", "away_scored_pct", "INTEGER"),
    ("matches", "away_conceded_pct", "INTEGER"),
    ("matches", "home_total_shots_pg", "REAL"),
    ("matches", "home_shots_ontarget_pct", "INTEGER"),
    ("matches", "away_total_shots_pg", "REAL"),
    ("matches", "away_shots_ontarget_pct", "INTEGER"),
    ("matches", "home_clean_sheets_pct", "REAL"),
    ("matches", "away_clean_sheets_pct", "REAL"),
    ("calibration_log", "method_used", "TEXT"),
    ("calibration_log", "market", "TEXT"),
    ("calibration_log", "stake", "REAL"),
    ("calibration_log", "odds", "REAL"),
]


def _migrate(conn):
    """Add missing columns to existing tables."""
    existing = set(
        row[1] for row in conn.execute("SELECT * FROM sqlite_master WHERE sql IS NOT NULL")
    )
    for table, column, col_def in _MIGRATIONS:
        if table in existing:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_def}")
            except sqlite3.OperationalError:
                pass  # Column already exists


def save_prediction(data: dict) -> int:
    """Save a prediction to the database. Returns match ID."""
    conn = get_db()
    defaults = {
        "our_stake": 0, "our_market": None, "method_used": None,
        "poisson_prob_home": None, "poisson_prob_draw": None, "poisson_prob_away": None,
        "ml_prob_home": None, "ml_prob_draw": None, "ml_prob_away": None,
        "forebet_prob_home": None, "forebet_prob_draw": None, "forebet_prob_away": None,
    }
    for k, v in defaults.items():
        data.setdefault(k, v)
    conn.execute("""
        INSERT OR REPLACE INTO matches (
            forebet_url, home_team, away_team, league,
            match_date, match_time,
            home_form, away_form, home_pos, away_pos,
            home_pts, away_pts, home_games_played, away_games_played,
            h2h_home_wins, h2h_draws, h2h_away_wins, h2h_matches,
            h2h_goals_for, h2h_goals_against, h2h_avg_total_goals, h2h_weighted_form,
            home_avg_goals_for, home_avg_goals_against,
            away_avg_goals_for, away_avg_goals_against,
            home_home_avg_goals_for, home_home_avg_goals_against,
            away_away_avg_goals_for, away_away_avg_goals_against,
            home_over15_pct, home_under15_pct, away_over15_pct, away_under15_pct,
            home_over25_pct, home_under25_pct, away_over25_pct, away_under25_pct,
            home_over35_pct, home_under35_pct, away_over35_pct, away_under35_pct,
            home_btts_yes_pct, home_btts_no_pct, away_btts_yes_pct, away_btts_no_pct,
            home_scored_pct, home_conceded_pct, away_scored_pct, away_conceded_pct,
            home_total_shots_pg, home_shots_ontarget_pct,
            away_total_shots_pg, away_shots_ontarget_pct,
            home_clean_sheets_pct, away_clean_sheets_pct,
            odds_home, odds_draw, odds_away,
            odds_over25, odds_under25, odds_btts_yes, odds_btts_no,
            forebet_pred, forebet_home_pct, forebet_draw_pct, forebet_away_pct,
            forebet_over25_pct, forebet_btts_yes_pct,
            our_prediction, our_confidence, our_score_lean,
            our_stake, our_market, method_used,
            poisson_prob_home, poisson_prob_draw, poisson_prob_away,
            ml_prob_home, ml_prob_draw, ml_prob_away,
            forebet_prob_home, forebet_prob_draw, forebet_prob_away
        ) VALUES (
            :forebet_url, :home_team, :away_team, :league,
            :match_date, :match_time,
            :home_form, :away_form, :home_pos, :away_pos,
            :home_pts, :away_pts, :home_games_played, :away_games_played,
            :h2h_home_wins, :h2h_draws, :h2h_away_wins, :h2h_matches,
            :h2h_goals_for, :h2h_goals_against, :h2h_avg_total_goals, :h2h_weighted_form,
            :home_avg_goals_for, :home_avg_goals_against,
            :away_avg_goals_for, :away_avg_goals_against,
            :home_home_avg_goals_for, :home_home_avg_goals_against,
            :away_away_avg_goals_for, :away_away_avg_goals_against,
            :home_over15_pct, :home_under15_pct, :away_over15_pct, :away_under15_pct,
            :home_over25_pct, :home_under25_pct, :away_over25_pct, :away_under25_pct,
            :home_over35_pct, :home_under35_pct, :away_over35_pct, :away_under35_pct,
            :home_btts_yes_pct, :home_btts_no_pct, :away_btts_yes_pct, :away_btts_no_pct,
            :home_scored_pct, :home_conceded_pct, :away_scored_pct, :away_conceded_pct,
            :home_total_shots_pg, :home_shots_ontarget_pct,
            :away_total_shots_pg, :away_shots_ontarget_pct,
            :home_clean_sheets_pct, :away_clean_sheets_pct,
            :odds_home, :odds_draw, :odds_away,
            :odds_over25, :odds_under25, :odds_btts_yes, :odds_btts_no,
            :forebet_pred, :forebet_home_pct, :forebet_draw_pct, :forebet_away_pct,
            :forebet_over25_pct, :forebet_btts_yes_pct,
            :our_prediction, :our_confidence, :our_score_lean,
            :our_stake, :our_market, :method_used,
            :poisson_prob_home, :poisson_prob_draw, :poisson_prob_away,
            :ml_prob_home, :ml_prob_draw, :ml_prob_away,
            :forebet_prob_home, :forebet_prob_draw, :forebet_prob_away
        )
    """, data)
    conn.commit()
    match_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return match_id


def get_unreviewed_matches(limit: int = 50) -> list:
    """Get unreviewed matches that have already been played (past dates only)."""
    from datetime import datetime
    today = datetime.now().strftime("%d/%m/%Y")
    conn = get_db()
    rows = conn.execute("""
        SELECT id, forebet_url, home_team, away_team, match_date, league
        FROM matches
        WHERE reviewed = 0
          AND match_date IS NOT NULL
          AND match_date < ?
        ORDER BY match_date DESC LIMIT ?
    """, (today, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _prediction_correct(pred: str, home_goals: int, away_goals: int) -> bool:
    """Check if a prediction matches the actual result.
    Supports: Home win, Away win, Draw, Home, Away,
    Over X.5, Under X.5, 1X, X2, 12, and Forebet format (1/X/2)."""
    if not pred:
        return False
    p = pred.strip()
    total = home_goals + away_goals

    # Forebet format: 1=Home, X=Draw, 2=Away
    if p in ("1", "2", "X"):
        if p == "1": return home_goals > away_goals
        if p == "2": return away_goals > home_goals
        if p == "X": return home_goals == away_goals

    # Over/Under goals
    m = re.match(r"(Over|Under)\s+(\d+\.?\d*)", p)
    if m:
        direction = m.group(1)
        threshold = float(m.group(2))
        if direction == "Over":
            return total > threshold
        else:
            return total <= threshold

    # Match result predictions
    if p in ("Home win", "Home"):
        return home_goals > away_goals
    if p in ("Away win", "Away"):
        return away_goals > home_goals
    if p == "Draw":
        return home_goals == away_goals

    # Double chance
    if p == "1X":
        return home_goals >= away_goals
    if p == "X2":
        return away_goals >= home_goals
    if p == "12":
        return home_goals != away_goals

    return False


def update_result(match_id: int, home_goals: int, away_goals: int):
    """Record actual result for a match."""
    result = "Home win" if home_goals > away_goals else (
        "Away win" if away_goals > home_goals else "Draw"
    )
    conn = get_db()
    conn.execute("""
        UPDATE matches SET
            actual_home_goals = ?, actual_away_goals = ?,
            actual_result = ?, reviewed = 1,
            reviewed_at = datetime('now')
        WHERE id = ?
    """, (home_goals, away_goals, result, match_id))
    conn.commit()

    # Log calibration
    match = conn.execute("""
        SELECT our_prediction, our_confidence, forebet_pred, league,
               our_stake, our_market, method_used, odds_home, odds_draw, odds_away,
               odds_over25, odds_under25
        FROM matches WHERE id = ?
    """, (match_id,)).fetchone()
    if match:
        our_correct = 1 if _prediction_correct(match["our_prediction"], home_goals, away_goals) else 0
        fb_correct = 1 if match["forebet_pred"] and _prediction_correct(match["forebet_pred"], home_goals, away_goals) else 0

        odds = None
        market = match["our_market"] or ""
        pick = match["our_prediction"] or ""
        if market == "1X2":
            odds = {"Home win": match["odds_home"], "Draw": match["odds_draw"], "Away win": match["odds_away"]}.get(pick)
        elif market == "O/U":
            if "Over" in pick:
                odds = match["odds_over25"]
            else:
                odds = match["odds_under25"]
        elif market == "BTTS":
            odds = match["odds_btts_yes"] if pick == "Yes" else match["odds_btts_no"]
        elif market == "DNB":
            odds = match["odds_home"] if "Home" in pick else match["odds_away"]

        conn.execute("""
            INSERT INTO calibration_log
                (league, match_id, our_prediction, actual_result,
                 correct, confidence, forebet_pred, forebet_correct,
                 method_used, market, stake, odds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            match["league"], match_id, match["our_prediction"], result,
            our_correct, match["our_confidence"], match["forebet_pred"], fb_correct,
            match["method_used"], match["our_market"],
            match["our_stake"], odds
        ))
        # Update league stats
        _update_league_stats(conn, match["league"])
        # Update component accuracy
        _update_component_accuracy(conn, match, our_correct)
    conn.commit()
    conn.close()


def _update_component_accuracy(conn, match: dict, our_correct: int):
    """Track accuracy per component, league, and market for dynamic weighting."""
    method = match["method_used"] or "unknown"
    league = match["league"] or "unknown"
    market = match["our_market"] or "unknown"
    # Map method to component
    if "ensemble" in method:
        if "poisson" in method and "ml" in method:
            components = ["ml", "poisson"]
        elif "poisson" in method:
            components = ["poisson"]
        else:
            components = ["ml"]
    else:
        components = ["poisson"]

    for comp in components:
        conn.execute("""
            INSERT OR REPLACE INTO component_accuracy
                (component, league, market, total, correct, last_updated)
            VALUES (?, ?, ?,
                COALESCE((SELECT total + 1 FROM component_accuracy
                    WHERE component=? AND league=? AND market=?), 1),
                COALESCE((SELECT correct + ? FROM component_accuracy
                    WHERE component=? AND league=? AND market=?), ?),
                datetime('now')
            )
        """, (comp, league, market, comp, league, market,
              our_correct, comp, league, market, our_correct))


def _update_league_stats(conn, league: str):
    """Update aggregated stats for a league."""
    stats = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN our_prediction = actual_result THEN 1 ELSE 0 END) as correct
        FROM calibration_log WHERE league = ?
    """, (league,)).fetchone()
    if stats and stats["total"] > 0:
        conn.execute("""
            INSERT OR REPLACE INTO league_stats
                (league, total_predictions, correct_predictions, last_updated)
            VALUES (?, ?, ?, datetime('now'))
        """, (league, stats["total"], stats["correct"]))


def get_calibration_summary() -> dict:
    """Get overall accuracy stats."""
    conn = get_db()
    rows = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(correct) as our_correct,
            SUM(forebet_correct) as fb_correct
        FROM calibration_log
    """).fetchone()

    by_league = conn.execute("""
        SELECT league,
               COUNT(*) as total,
               SUM(correct) as our_correct,
               ROUND(100.0 * SUM(correct) / COUNT(*), 1) as our_pct
        FROM calibration_log
        GROUP BY league
        ORDER BY total DESC
    """).fetchall()

    by_confidence = conn.execute("""
        SELECT confidence,
               COUNT(*) as total,
               SUM(correct) as correct,
               ROUND(100.0 * SUM(correct) / COUNT(*), 1) as pct
        FROM calibration_log
        GROUP BY confidence
        ORDER BY CASE confidence
            WHEN 'Near Certain' THEN 1
            WHEN 'High' THEN 2
            WHEN 'Medium-High' THEN 3
            WHEN 'Medium' THEN 4
            WHEN 'Low' THEN 5
        END
    """).fetchall()

    conn.close()
    return {
        "total": rows["total"] if rows else 0,
        "our_correct": rows["our_correct"] if rows else 0,
        "our_pct": round(100.0 * rows["our_correct"] / rows["total"], 1)
            if rows and rows["total"] else 0,
        "fb_correct": rows["fb_correct"] if rows else 0,
        "fb_pct": round(100.0 * rows["fb_correct"] / rows["total"], 1)
            if rows and rows["total"] else 0,
        "by_league": [dict(r) for r in by_league],
        "by_confidence": [dict(r) for r in by_confidence],
    }


def get_component_accuracy(component: str = None, league: str = None, market: str = None) -> list:
    """Get accuracy stats by component, optionally filtered."""
    conn = get_db()
    query = "SELECT component, league, market, total, correct FROM component_accuracy WHERE 1=1"
    params = []
    if component:
        query += " AND component=?"
        params.append(component)
    if league:
        query += " AND league=?"
        params.append(league)
    if market:
        query += " AND market=?"
        params.append(market)
    query += " ORDER BY total DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_league_accuracy(league: str) -> float:
    """Get accuracy for a specific league."""
    conn = get_db()
    row = conn.execute("""
        SELECT ROUND(100.0 * SUM(correct) / COUNT(*), 1) as pct
        FROM calibration_log WHERE league = ?
    """, (league,)).fetchone()
    conn.close()
    return row["pct"] if row and row["pct"] else 0


def get_dynamic_weights(league: str = None, market: str = None, min_samples: int = 5) -> dict:
    """Compute dynamic ensemble weights based on tracked component accuracy."""
    conn = get_db()
    rows = conn.execute("""
        SELECT component, SUM(total) as total, SUM(correct) as correct
        FROM component_accuracy
        WHERE (league = ? OR ? IS NULL)
          AND (market = ? OR ? IS NULL)
        GROUP BY component
    """, (league, league, market, market)).fetchall()
    conn.close()

    weights = {"ml": 0.25, "poisson": 0.35, "forebet": 0.25, "default": 0.15}
    if not rows:
        return weights

    total_weight = 0
    accuracies = {}
    for r in rows:
        comp = r["component"]
        total = r["total"]
        correct = r["correct"]
        if total >= min_samples:
            acc = correct / total
            accuracies[comp] = acc
            total_weight += acc

    if total_weight > 0 and accuracies:
        raw = {k: v / total_weight for k, v in accuracies.items()}
        # Blend with default weights to avoid overfitting
        blend = 0.7
        for k in raw:
            weights[k] = raw[k] * blend + weights.get(k, 0.2) * (1 - blend)
        # Normalize
        tw = sum(weights.values())
        if tw > 0:
            for k in weights:
                weights[k] /= tw

    return weights


def get_predictions_for_review() -> list:
    """Get predictions with their Forebet URLs for review process."""
    conn = get_db()
    rows = conn.execute("""
        SELECT id, forebet_url, home_team, away_team, match_date,
               our_prediction, our_confidence, our_score_lean,
               forebet_pred, reviewed
        FROM matches
        ORDER BY match_date DESC LIMIT 200
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def import_betting_results(filepath: str) -> int:
    """Parse betting_results.txt and import into DB as unreviewed matches.
    Format: id, odds, date, teams, market, pick, score
    (improvement 8: feed betting results into model training)
    """
    try:
        with open(filepath) as f:
            lines = [l.strip() for l in f if l.strip()]
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return 0

    imported = 0
    conn = get_db()
    i = 0
    while i < len(lines):
        try:
            # Parse blocks: match_id, odds, date, teams, market, pick, score
            if not lines[i][0].isdigit():
                i += 1
                continue
            match_id = int(lines[i])
            odds = float(lines[i+1]) if i+1 < len(lines) and lines[i+1].replace('.','',1).isdigit() else 0
            date = lines[i+2] if i+2 < len(lines) else ""
            teams = lines[i+3] if i+3 < len(lines) else ""
            market = lines[i+4] if i+4 < len(lines) else ""
            pick = lines[i+5] if i+5 < len(lines) else ""
            score = lines[i+6] if i+6 < len(lines) else ""

            # Parse teams
            parts = teams.split(" – ")
            home = parts[0].strip() if parts else ""
            away = parts[1].strip() if len(parts) > 1 else ""

            # Parse score
            actual_h = actual_a = None
            score_m = re.match(r"(\d+):(\d+)", score)
            if score_m:
                actual_h, actual_a = int(score_m.group(1)), int(score_m.group(2))

            if home and away:
                conn.execute("""
                    INSERT OR REPLACE INTO matches
                        (forebet_url, home_team, away_team, match_date,
                         odds_home, our_prediction, our_confidence,
                         actual_home_goals, actual_away_goals, actual_result,
                         reviewed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """, (
                    f"betting_results_{match_id}", home, away, date,
                    odds, pick, "Medium",
                    actual_h, actual_a,
                    "Home win" if actual_h and actual_a and actual_h > actual_a
                    else "Away win" if actual_h and actual_a and actual_h < actual_a
                    else "Draw" if actual_h is not None and actual_a is not None
                    else None
                ))
                imported += 1
            i += 7
        except (ValueError, IndexError):
            i += 1
            continue

    conn.commit()
    conn.close()
    print(f"Imported {imported} betting results from {filepath}")
    return imported


# Auto-initialize on import
if not DB_PATH.exists():
    init_db()
