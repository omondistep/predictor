"""
SQLite database module for prediction storage, review, and calibration.

Stores every prediction with all scraped features, tracks actual results
after review, and supports model calibration based on historical accuracy.
"""

import sqlite3
import os
import json
from datetime import datetime
from pathlib import Path

DB_DIR = Path(__file__).parent
DB_PATH = DB_DIR / "history.db"

SCHEMA = """
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

CREATE INDEX IF NOT EXISTS idx_matches_league ON matches(league);
CREATE INDEX IF NOT EXISTS idx_matches_reviewed ON matches(reviewed);
CREATE INDEX IF NOT EXISTS idx_calibration_league ON calibration_log(league);
"""


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_db()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def save_prediction(data: dict) -> int:
    """Save a prediction to the database. Returns match ID."""
    conn = get_db()
    conn.execute("""
        INSERT OR REPLACE INTO matches (
            forebet_url, home_team, away_team, league,
            match_date, match_time,
            home_form, away_form, home_pos, away_pos,
            home_pts, away_pts, home_games_played, away_games_played,
            h2h_home_wins, h2h_draws, h2h_away_wins, h2h_matches,
            home_avg_goals_for, home_avg_goals_against,
            away_avg_goals_for, away_avg_goals_against,
            odds_home, odds_draw, odds_away,
            odds_over25, odds_under25, odds_btts_yes, odds_btts_no,
            forebet_pred, forebet_home_pct, forebet_draw_pct, forebet_away_pct,
            forebet_over25_pct, forebet_btts_yes_pct,
            our_prediction, our_confidence, our_score_lean
        ) VALUES (
            :forebet_url, :home_team, :away_team, :league,
            :match_date, :match_time,
            :home_form, :away_form, :home_pos, :away_pos,
            :home_pts, :away_pts, :home_games_played, :away_games_played,
            :h2h_home_wins, :h2h_draws, :h2h_away_wins, :h2h_matches,
            :home_avg_goals_for, :home_avg_goals_against,
            :away_avg_goals_for, :away_avg_goals_against,
            :odds_home, :odds_draw, :odds_away,
            :odds_over25, :odds_under25, :odds_btts_yes, :odds_btts_no,
            :forebet_pred, :forebet_home_pct, :forebet_draw_pct, :forebet_away_pct,
            :forebet_over25_pct, :forebet_btts_yes_pct,
            :our_prediction, :our_confidence, :our_score_lean
        )
    """, data)
    conn.commit()
    match_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return match_id


def get_unreviewed_matches(limit: int = 50) -> list:
    """Get matches that haven't been reviewed yet."""
    conn = get_db()
    rows = conn.execute("""
        SELECT id, forebet_url, home_team, away_team, match_date, league
        FROM matches WHERE reviewed = 0
        ORDER BY match_date DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


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
    match = conn.execute(
        "SELECT our_prediction, our_confidence, forebet_pred, league FROM matches WHERE id = ?",
        (match_id,)
    ).fetchone()
    if match:
        our_correct = 1 if match["our_prediction"] == result else 0
        fb_correct = 1 if match["forebet_pred"] == result else 0
        conn.execute("""
            INSERT INTO calibration_log
                (league, match_id, our_prediction, actual_result,
                 correct, confidence, forebet_pred, forebet_correct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            match["league"], match_id, match["our_prediction"], result,
            our_correct, match["our_confidence"], match["forebet_pred"], fb_correct
        ))
        # Update league stats
        _update_league_stats(conn, match["league"])
    conn.commit()
    conn.close()


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


def get_league_accuracy(league: str) -> float:
    """Get accuracy for a specific league."""
    conn = get_db()
    row = conn.execute("""
        SELECT ROUND(100.0 * SUM(correct) / COUNT(*), 1) as pct
        FROM calibration_log WHERE league = ?
    """, (league,)).fetchone()
    conn.close()
    return row["pct"] if row and row["pct"] else 0


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


# Auto-initialize on import
if not DB_PATH.exists():
    init_db()
