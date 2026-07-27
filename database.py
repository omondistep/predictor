"""
SQLite database module for prediction storage, review, and calibration.

Stores every prediction with all scraped features, tracks actual results
after review, and supports model calibration based on historical accuracy.
"""

import sqlite3
import os
import re
import json
from datetime import datetime, timedelta
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

    -- Half-time scores
    ht_home_goals INTEGER,
    ht_away_goals INTEGER,

    -- Injury/suspension data (scraped from /en/injured-players)
    home_injured_total INTEGER DEFAULT 0,
    home_forwards_out INTEGER DEFAULT 0,
    home_midfielders_out INTEGER DEFAULT 0,
    home_defenders_out INTEGER DEFAULT 0,
    home_key_players_out INTEGER DEFAULT 0,
    home_suspended INTEGER DEFAULT 0,
    away_injured_total INTEGER DEFAULT 0,
    away_forwards_out INTEGER DEFAULT 0,
    away_midfielders_out INTEGER DEFAULT 0,
    away_defenders_out INTEGER DEFAULT 0,
    away_key_players_out INTEGER DEFAULT 0,
    away_suspended INTEGER DEFAULT 0,

    -- FBref xG data
    home_squad_xg REAL,
    home_squad_xga REAL,
    home_squad_xgd REAL,
    away_squad_xg REAL,
    away_squad_xga REAL,
    away_squad_xgd REAL,

    home_xg_proxy REAL,
    away_xg_proxy REAL,

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

CREATE TABLE IF NOT EXISTS calibration_bias (
    id INTEGER PRIMARY KEY,
    league TEXT,
    market TEXT,
    threshold TEXT,
    bucket TEXT,
    predicted_mean REAL,
    actual_mean REAL,
    bias REAL,
    sample_count INTEGER DEFAULT 0,
    last_updated TEXT DEFAULT (datetime('now')),
    UNIQUE(league, market, threshold, bucket)
);

CREATE TABLE IF NOT EXISTS model_retrain_log (
    id INTEGER PRIMARY KEY,
    triggered_by TEXT,
    examples_before INTEGER,
    examples_after INTEGER,
    accuracy_1x2_before REAL,
    accuracy_1x2_after REAL,
    accuracy_ou_before REAL,
    accuracy_ou_after REAL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS ml_league_accuracy (
    league TEXT,
    market TEXT,
    ml_correct INTEGER DEFAULT 0,
    ml_total INTEGER DEFAULT 0,
    poisson_correct INTEGER DEFAULT 0,
    poisson_total INTEGER DEFAULT 0,
    last_updated TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (league, market)
);
"""

SCHEMA_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_matches_league ON matches(league);
CREATE INDEX IF NOT EXISTS idx_matches_reviewed ON matches(reviewed);
CREATE UNIQUE INDEX IF NOT EXISTS idx_matches_forebet_url ON matches(forebet_url);
CREATE INDEX IF NOT EXISTS idx_calibration_league ON calibration_log(league);
CREATE INDEX IF NOT EXISTS idx_calibration_method ON calibration_log(method_used);
"""


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _migrate(conn)
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
    ("matches", "home_possession_pct", "INTEGER"),
    ("matches", "away_possession_pct", "INTEGER"),
    ("matches", "home_passes_per_game", "REAL"),
    ("matches", "away_passes_per_game", "REAL"),
    ("matches", "home_pass_accuracy_pct", "INTEGER"),
    ("matches", "away_pass_accuracy_pct", "INTEGER"),
    ("matches", "home_total_attacks_pg", "REAL"),
    ("matches", "away_total_attacks_pg", "REAL"),
    ("matches", "home_dangerous_attacks_pg", "REAL"),
    ("matches", "away_dangerous_attacks_pg", "REAL"),
    ("matches", "home_corners_avg", "REAL"),
    ("matches", "away_corners_avg", "REAL"),
    ("matches", "home_fouls_avg", "REAL"),
    ("matches", "away_fouls_avg", "REAL"),
    ("matches", "home_yellow_cards_avg", "REAL"),
    ("matches", "away_yellow_cards_avg", "REAL"),
    ("matches", "home_total_shots", "INTEGER"),
    ("matches", "away_total_shots", "INTEGER"),
    ("matches", "home_clean_sheets", "INTEGER"),
    ("matches", "away_clean_sheets", "INTEGER"),
    ("matches", "ht_home_goals", "INTEGER"),
    ("matches", "ht_away_goals", "INTEGER"),
    ("matches", "home_injured_total", "INTEGER DEFAULT 0"),
    ("matches", "home_forwards_out", "INTEGER DEFAULT 0"),
    ("matches", "home_midfielders_out", "INTEGER DEFAULT 0"),
    ("matches", "home_defenders_out", "INTEGER DEFAULT 0"),
    ("matches", "home_key_players_out", "INTEGER DEFAULT 0"),
    ("matches", "home_suspended", "INTEGER DEFAULT 0"),
    ("matches", "away_injured_total", "INTEGER DEFAULT 0"),
    ("matches", "away_forwards_out", "INTEGER DEFAULT 0"),
    ("matches", "away_midfielders_out", "INTEGER DEFAULT 0"),
    ("matches", "away_defenders_out", "INTEGER DEFAULT 0"),
    ("matches", "away_key_players_out", "INTEGER DEFAULT 0"),
    ("matches", "away_suspended", "INTEGER DEFAULT 0"),
    # FBref xG data
    ("matches", "home_squad_xg", "REAL"),
    ("matches", "home_squad_xga", "REAL"),
    ("matches", "home_squad_xgd", "REAL"),
    ("matches", "away_squad_xg", "REAL"),
    ("matches", "away_squad_xga", "REAL"),
    ("matches", "away_squad_xgd", "REAL"),
    # Proxy xG (computed from Forebet shot/attack data)
    ("matches", "home_xg_proxy", "REAL"),
    ("matches", "away_xg_proxy", "REAL"),
    ("calibration_log", "method_used", "TEXT"),
    ("calibration_log", "market", "TEXT"),
    ("calibration_log", "stake", "REAL"),
    ("calibration_log", "odds", "REAL"),
    ("calibration_log", "model_prob", "REAL"),
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
    # Reuse the existing row id for the same Forebet URL so re-running a link
    # updates the prior record instead of inserting a duplicate (this keeps
    # match ids stable for report de-duplication and prevents DB bloat).
    existing = conn.execute(
        "SELECT id FROM matches WHERE forebet_url = ?", (data.get("forebet_url"),)
    ).fetchone()
    data["id"] = existing[0] if existing else None
    conn.execute("""
        INSERT OR REPLACE INTO matches (
            id, forebet_url, home_team, away_team, league,
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
            home_total_shots_pg, home_total_shots, home_shots_ontarget_pct,
            away_total_shots_pg, away_total_shots, away_shots_ontarget_pct,
            home_clean_sheets_pct, away_clean_sheets_pct,
            home_clean_sheets, away_clean_sheets,
            home_possession_pct, away_possession_pct,
            home_passes_per_game, away_passes_per_game,
            home_pass_accuracy_pct, away_pass_accuracy_pct,
            home_total_attacks_pg, away_total_attacks_pg,
            home_dangerous_attacks_pg, away_dangerous_attacks_pg,
            home_corners_avg, away_corners_avg,
            home_fouls_avg, away_fouls_avg,
            home_yellow_cards_avg, away_yellow_cards_avg,
            odds_home, odds_draw, odds_away,
            odds_over25, odds_under25, odds_btts_yes, odds_btts_no,
            forebet_pred, forebet_home_pct, forebet_draw_pct, forebet_away_pct,
            forebet_over25_pct, forebet_btts_yes_pct,
            our_prediction, our_confidence, our_score_lean,
            our_stake, our_market, method_used,
            poisson_prob_home, poisson_prob_draw, poisson_prob_away,
            ml_prob_home, ml_prob_draw, ml_prob_away,
            forebet_prob_home, forebet_prob_draw, forebet_prob_away,
            ht_home_goals, ht_away_goals,
            home_injured_total, home_forwards_out, home_midfielders_out,
            home_defenders_out, home_key_players_out, home_suspended,
            away_injured_total, away_forwards_out, away_midfielders_out,
            away_defenders_out, away_key_players_out, away_suspended
        ) VALUES (
            :id, :forebet_url, :home_team, :away_team, :league,
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
            :home_total_shots_pg, :home_total_shots, :home_shots_ontarget_pct,
            :away_total_shots_pg, :away_total_shots, :away_shots_ontarget_pct,
            :home_clean_sheets_pct, :away_clean_sheets_pct,
            :home_clean_sheets, :away_clean_sheets,
            :home_possession_pct, :away_possession_pct,
            :home_passes_per_game, :away_passes_per_game,
            :home_pass_accuracy_pct, :away_pass_accuracy_pct,
            :home_total_attacks_pg, :away_total_attacks_pg,
            :home_dangerous_attacks_pg, :away_dangerous_attacks_pg,
            :home_corners_avg, :away_corners_avg,
            :home_fouls_avg, :away_fouls_avg,
            :home_yellow_cards_avg, :away_yellow_cards_avg,
            :odds_home, :odds_draw, :odds_away,
            :odds_over25, :odds_under25, :odds_btts_yes, :odds_btts_no,
            :forebet_pred, :forebet_home_pct, :forebet_draw_pct, :forebet_away_pct,
            :forebet_over25_pct, :forebet_btts_yes_pct,
            :our_prediction, :our_confidence, :our_score_lean,
            :our_stake, :our_market, :method_used,
            :poisson_prob_home, :poisson_prob_draw, :poisson_prob_away,
            :ml_prob_home, :ml_prob_draw, :ml_prob_away,
            :forebet_prob_home, :forebet_prob_draw, :forebet_prob_away,
            :ht_home_goals, :ht_away_goals,
            :home_injured_total, :home_forwards_out, :home_midfielders_out,
            :home_defenders_out, :home_key_players_out, :home_suspended,
            :away_injured_total, :away_forwards_out, :away_midfielders_out,
            :away_defenders_out, :away_key_players_out, :away_suspended
        )
    """, data)
    conn.commit()
    match_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return match_id


def update_injury_data(match_id: int, home_summary: dict, away_summary: dict):
    """Update injury data for an existing match prediction."""
    conn = get_db()
    conn.execute("""
        UPDATE matches SET
            home_injured_total = ?,
            home_forwards_out = ?,
            home_midfielders_out = ?,
            home_defenders_out = ?,
            home_key_players_out = ?,
            home_suspended = ?,
            away_injured_total = ?,
            away_forwards_out = ?,
            away_midfielders_out = ?,
            away_defenders_out = ?,
            away_key_players_out = ?,
            away_suspended = ?
        WHERE id = ?
    """, (
        home_summary.get("total_injured", 0),
        home_summary.get("forwards_out", 0),
        home_summary.get("midfielders_out", 0),
        home_summary.get("defenders_out", 0),
        home_summary.get("key_players_out", 0),
        home_summary.get("suspended", 0),
        away_summary.get("total_injured", 0),
        away_summary.get("forwards_out", 0),
        away_summary.get("midfielders_out", 0),
        away_summary.get("defenders_out", 0),
        away_summary.get("key_players_out", 0),
        away_summary.get("suspended", 0),
        match_id,
    ))
    conn.commit()
    conn.close()


def populate_injury_data():
    """Scrape injuries and update all unreviewed matches with injury data."""
    from forebet_scraper import scrape_injuries, get_team_injury_summary
    injuries = scrape_injuries()
    if not injuries:
        print("  No injury data scraped")
        return 0

    conn = get_db()
    rows = conn.execute("""
        SELECT id, home_team, away_team FROM matches
        WHERE reviewed = 0 AND home_injured_total = 0
    """).fetchall()
    conn.close()

    updated = 0
    for row in rows:
        home_summary = get_team_injury_summary(injuries, row["home_team"])
        away_summary = get_team_injury_summary(injuries, row["away_team"])
        if home_summary["total_injured"] > 0 or away_summary["total_injured"] > 0:
            update_injury_data(row["id"], home_summary, away_summary)
            updated += 1

    print(f"  Updated injury data for {updated} matches")
    return updated


def compute_proxy_xg(match_id: int, data: dict):
    """Compute proxy xG from Forebet shot/attack stats and save to DB.
    
    Uses: shots_pg, shots_on_target_pct, dangerous_attacks_pg, possession_pct
    Formula: xG = shots * SOT% * league_avg_sot_conversion * attack_quality * possession_weight
    
    Calibration constants from 2661 historical matches:
      - SOT conversion rate: 0.4025 goals per shot on target
      - Average dangerous attacks: 48.1 per game
    """
    SOT_CONV = 0.4025
    AVG_DANG = 48.1

    def _proxy_xg(shots_pg, sot_pct, dang_pg, poss_pct):
        if not shots_pg or not sot_pct:
            return None
        sot = shots_pg * (sot_pct / 100.0)
        quality = min((dang_pg or AVG_DANG) / AVG_DANG, 2.0)
        poss_w = 0.7 + 0.6 * ((poss_pct or 50) / 100.0)
        return round(sot * SOT_CONV * quality * poss_w, 4)

    hxg = _proxy_xg(
        data.get("home_total_shots_pg"),
        data.get("home_shots_ontarget_pct"),
        data.get("home_dangerous_attacks_pg"),
        data.get("home_possession_pct"),
    )
    axg = _proxy_xg(
        data.get("away_total_shots_pg"),
        data.get("away_shots_ontarget_pct"),
        data.get("away_dangerous_attacks_pg"),
        data.get("away_possession_pct"),
    )

    if hxg is not None or axg is not None:
        conn = get_db()
        conn.execute(
            "UPDATE matches SET home_xg_proxy = ?, away_xg_proxy = ? WHERE id = ?",
            (hxg, axg, match_id),
        )
        conn.commit()
        conn.close()


def get_unreviewed_matches(limit: int = 50) -> list:
    """Get unreviewed matches that have already been played (past dates only)."""
    today = datetime.now().strftime("%d/%m/%Y")
    today_parts = today.split("/")
    today_int = int(today_parts[2]) * 10000 + int(today_parts[1]) * 100 + int(today_parts[0])
    conn = get_db()
    rows = conn.execute("""
        SELECT id, forebet_url, home_team, away_team, match_date, league
        FROM matches
        WHERE reviewed = 0
          AND match_date IS NOT NULL
          AND CAST(SUBSTR(match_date, 7, 4) AS INTEGER) * 10000
              + CAST(SUBSTR(match_date, 4, 2) AS INTEGER) * 100
              + CAST(SUBSTR(match_date, 1, 2) AS INTEGER) < ?
        ORDER BY match_date DESC LIMIT ?
    """, (today_int, limit)).fetchall()
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
               odds_over25, odds_under25, odds_btts_yes, odds_btts_no
        FROM matches WHERE id = ?
    """, (match_id,)).fetchone()
    if match:
        market = match["our_market"] or ""
        pick = match["our_prediction"] or ""
        # DNB picks on a draw are pushes — skip calibration log
        if market == "DNB" and home_goals == away_goals:
            our_correct = None
        else:
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

        if our_correct is not None:
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

    def _upsert_comp(comp, correct):
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
              correct, comp, league, market, correct))

    if "ensemble" in method:
        if "poisson" in method and "ml" in method:
            _upsert_comp("ml", our_correct)
            _upsert_comp("poisson", our_correct)
        elif "poisson" in method:
            _upsert_comp("poisson", our_correct)
        else:
            _upsert_comp("ml", our_correct)
        # Track forebet component when ensemble used
        if "forebet" in method or "fb=" in method or "fb-weights" in method:
            _upsert_comp("forebet", our_correct)
    else:
        _upsert_comp("poisson", our_correct)


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


def store_market_results(match_id: int, all_picks: list, actual_home_goals: int, actual_away_goals: int, actual_outcome: str, match_data: dict = None):
    """Store accuracy results for all market picks from a match."""
    if not all_picks:
        return

    conn = get_db()
    match = conn.execute("SELECT league, method_used FROM matches WHERE id = ?", (match_id,)).fetchone()
    if not match:
        conn.close()
        return

    total_goals = actual_home_goals + actual_away_goals
    league = match["league"] or "unknown"
    method = match["method_used"] or "unknown"

    odds_map = {}
    if match_data:
        odds_map = {
            "Home win": match_data.get("odds_home"),
            "Draw": match_data.get("odds_draw"),
            "Away win": match_data.get("odds_away"),
            "Over 2.5": match_data.get("odds_over25"),
            "Under 2.5": match_data.get("odds_under25"),
            "Yes": match_data.get("odds_btts_yes"),
            "No": match_data.get("odds_btts_no"),
        }

    # Extract Forebet's prediction for each market from match_data
    fb_pred_1x2 = (match_data or {}).get("forebet_pred") if match_data else None
    fb_over25_pct = (match_data or {}).get("forebet_over25_pct") if match_data else None
    fb_btts_yes_pct = (match_data or {}).get("forebet_btts_yes_pct") if match_data else None

    for p in all_picks:
        market = p.get("market", "")
        pick = p.get("pick", "")
        model_prob = p.get("model_prob")
        confidence = p.get("confidence", "")

        correct = None
        if market == "1X2":
            correct = 1 if pick == actual_outcome else 0
        elif market == "O/U":
            if "Over" in pick:
                thresh = float(pick.split()[-1])
                correct = 1 if total_goals > thresh else 0
            elif "Under" in pick:
                thresh = float(pick.split()[-1])
                correct = 1 if total_goals <= thresh else 0
        elif market == "BTTS":
            both_scored = actual_home_goals > 0 and actual_away_goals > 0
            if pick == "Yes":
                correct = 1 if both_scored else 0
            elif pick == "No":
                correct = 1 if not both_scored else 0
        elif market == "DNB":
            if actual_outcome == "Draw":
                correct = None  # Push — stake returned
            elif pick == "Home":
                correct = 1 if actual_outcome == "Home win" else 0
            elif pick == "Away":
                correct = 1 if actual_outcome == "Away win" else 0
        elif market == "DC":
            if pick == "1X":
                correct = 1 if actual_outcome in ("Home win", "Draw") else 0
            elif pick == "X2":
                correct = 1 if actual_outcome in ("Away win", "Draw") else 0
            elif pick == "12":
                correct = 1 if actual_outcome in ("Home win", "Away win") else 0

        if correct is None:
            continue

        odds = odds_map.get(pick)

        # Compute Forebet's prediction and correctness for this market
        fb_pred = None
        fb_correct = None
        if market == "1X2" and fb_pred_1x2:
            fb_pred = fb_pred_1x2
            fb_correct = 1 if _prediction_correct(fb_pred_1x2, actual_home_goals, actual_away_goals) else 0
        elif market == "O/U" and fb_over25_pct is not None:
            fb_pred = "Over 2.5" if fb_over25_pct > 50 else "Under 2.5"
            fb_correct = 1 if _prediction_correct(fb_pred, actual_home_goals, actual_away_goals) else 0
        elif market == "BTTS" and fb_btts_yes_pct is not None:
            fb_pred = "Yes" if fb_btts_yes_pct > 50 else "No"
            fb_correct = 1 if _prediction_correct(fb_pred, actual_home_goals, actual_away_goals) else 0

        conn.execute("""
            INSERT INTO calibration_log
                (league, match_id, our_prediction, actual_result,
                 correct, confidence, forebet_pred, forebet_correct,
                 method_used, market, stake, odds, model_prob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            league, match_id, pick, actual_outcome,
            correct, confidence, fb_pred, fb_correct,
            method, market, 0, odds, model_prob
        ))

    conn.commit()
    conn.close()


def get_market_accuracy(market: str = None, league: str = None, days_back: int = 365) -> list:
    """Get per-market accuracy stats from calibration_log."""
    conn = get_db()
    cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    query = """
        SELECT market,
               COUNT(*) as total,
               SUM(correct) as correct,
               ROUND(100.0 * SUM(correct) / COUNT(*), 1) as accuracy
        FROM calibration_log
        WHERE created_at >= ?
    """
    params = [cutoff]

    if market:
        query += " AND market = ?"
        params.append(market)
    if league:
        query += " AND league = ?"
        params.append(league)

    query += " GROUP BY market ORDER BY total DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_market_accuracy_history(market: str, days_back: int = 365, window: int = 50) -> list:
    """Get rolling accuracy history for a market (for trend charts)."""
    conn = get_db()
    cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    rows = conn.execute("""
        SELECT created_at, correct
        FROM calibration_log
        WHERE market = ? AND created_at >= ?
        ORDER BY created_at ASC
    """, (market, cutoff)).fetchall()
    conn.close()

    if not rows:
        return []

    # Calculate rolling accuracy
    results = []
    for i in range(len(rows)):
        start = max(0, i - window + 1)
        window_data = rows[start:i+1]
        total = len(window_data)
        correct = sum(1 for r in window_data if r["correct"])
        accuracy = round(100.0 * correct / total, 1) if total > 0 else 0
        results.append({
            "date": rows[i]["created_at"],
            "accuracy": accuracy,
            "total": total,
            "correct": correct
        })

    return results


def get_calibration_data_for_market(market: str, min_samples: int = 10) -> list:
    """Get calibration data for isotonic regression (predicted prob vs actual outcome)."""
    conn = get_db()
    rows = conn.execute("""
        SELECT odds, correct
        FROM calibration_log
        WHERE market = ? AND odds IS NOT NULL AND odds > 1.0
        ORDER BY created_at ASC
    """, (market,)).fetchall()
    conn.close()

    if len(rows) < min_samples:
        return []

    # Convert odds to implied probability
    results = []
    for r in rows:
        implied_prob = 1.0 / r["odds"]
        results.append({
            "predicted_prob": implied_prob,
            "actual": r["correct"]
        })

    return results


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


def train_weights_from_history(market: str = "1X2", min_samples: int = 10):
    """Train ensemble weights from historical predictions.
    
    For each league, computes per-source accuracy from calibration_log
    and updates component_accuracy table. This allows the model to learn
    which source (Forebet, Poisson, ML) is most accurate per league.
    
    Returns dict with summary of changes.
    """
    conn = get_db()
    
    # Get all leagues with enough 1X2 predictions
    leagues = conn.execute("""
        SELECT league, COUNT(*) as total,
               SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as ensemble_correct,
               SUM(CASE WHEN forebet_correct = 1 THEN 1 ELSE 0 END) as forebet_correct,
               SUM(CASE WHEN forebet_correct IS NOT NULL THEN 1 ELSE 0 END) as has_forebet
        FROM calibration_log
        WHERE market = ? AND actual_result IS NOT NULL
        GROUP BY league
        HAVING total >= ?
    """, (market, min_samples)).fetchall()
    
    updated = 0
    results = {}
    
    for row in leagues:
        league = row["league"]
        total = row["total"]
        ens_correct = row["ensemble_correct"] or 0
        fb_correct = row["forebet_correct"] or 0
        has_fb = row["has_forebet"] or 0
        
        # Skip if no forebet data
        if has_fb < min_samples:
            continue
        
        # Compute accuracies
        ens_acc = ens_correct / total if total > 0 else 0
        fb_acc = fb_correct / has_fb if has_fb > 0 else 0
        
        # Infer Poisson+ML contribution
        # If ensemble matches forebet pick and both correct: forebet contributed
        # If ensemble differs from forebet and ensemble correct: poisson/ml contributed
        # Simple approximation: poisson_ml_acc = (ensemble_correct - forebet_contribution) / total
        # where forebet_contribution = forebet_correct * (overlap with ensemble)
        
        # Get overlap: how often ensemble pick matches forebet pick
        # forebet_pred can be: "1" (home), "2" (away), "X" (draw), or "H", "A", "D"
        overlap_row = conn.execute("""
            SELECT COUNT(*) as overlap,
                   SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as overlap_correct
            FROM calibration_log
            WHERE market = ? AND league = ? 
              AND actual_result IS NOT NULL
              AND forebet_pred IS NOT NULL
              AND (
                (forebet_pred IN ('H', '1') AND our_prediction = 'Home win')
                OR (forebet_pred IN ('D', 'X') AND our_prediction = 'Draw')
                OR (forebet_pred IN ('A', '2') AND our_prediction = 'Away win')
              )
        """, (market, league)).fetchone()
        
        overlap = overlap_row["overlap"] or 0
        overlap_correct = overlap_row["overlap_correct"] or 0
        
        # When ensemble agrees with forebet, forebet accuracy = overlap_correct/overlap
        # When ensemble disagrees, poisson/ml accuracy = (ens_correct - overlap_correct)/(total - overlap)
        fb_when_agree = overlap_correct / overlap if overlap > 0 else fb_acc
        
        # Poisson/ML accuracy: when ensemble pick != forebet pick, the ensemble is driven by poisson/ml
        non_fb_total = total - overlap
        non_fb_correct = ens_correct - overlap_correct
        pm_acc = non_fb_correct / non_fb_total if non_fb_total > 0 else ens_acc
        
        # Also compute: what's the ensemble accuracy when it disagrees with forebet?
        # This tells us how well poisson/ml perform independently
        ensemble_vs_forebet_acc = non_fb_correct / non_fb_total if non_fb_total > 0 else 0
        
        # Ensure we have minimum accuracy (don't let 0% destroy a source)
        fb_acc_eff = max(fb_acc, 0.15)
        pm_acc_eff = max(pm_acc, 0.15)
        
        # Compute optimal weights (proportional to accuracy)
        total_acc = fb_acc_eff + pm_acc_eff
        w_fb_raw = fb_acc_eff / total_acc
        w_pm_raw = pm_acc_eff / total_acc
        
        # Split poisson/ml 50/50 (we don't have separate tracking yet)
        w_poisson = w_pm_raw * 0.55
        w_ml = w_pm_raw * 0.45
        w_forebet = w_fb_raw
        
        # Store in component_accuracy
        conn.execute("""
            INSERT OR REPLACE INTO component_accuracy
                (component, league, market, total, correct, last_updated)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
        """, ("forebet", league, market, has_fb, fb_correct))
        
        # For poisson+ml, we approximate from ensemble performance when forebet wasn't the driver
        if non_fb_total > 0:
            conn.execute("""
                INSERT OR REPLACE INTO component_accuracy
                    (component, league, market, total, correct, last_updated)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
            """, ("poisson", league, market, non_fb_total, int(pm_acc * non_fb_total)))
            conn.execute("""
                INSERT OR REPLACE INTO component_accuracy
                    (component, league, market, total, correct, last_updated)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
            """, ("ml", league, market, non_fb_total, int(pm_acc * non_fb_total)))
        
        results[league] = {
            "total": total,
            "forebet_acc": round(fb_acc * 100, 1),
            "poisson_ml_acc": round(pm_acc * 100, 1),
            "ensemble_acc": round(ens_acc * 100, 1),
            "weights": {
                "forebet": round(w_forebet, 3),
                "poisson": round(w_poisson, 3),
                "ml": round(w_ml, 3)
            }
        }
        updated += 1
    
    conn.commit()
    conn.close()
    
    return {"leagues_updated": updated, "details": results}


def record_ml_league_result(league: str, market: str, ml_correct: bool, poisson_correct: bool):
    """Record ML vs Poisson accuracy for a specific league and market."""
    conn = get_db()
    conn.execute("""
        INSERT INTO ml_league_accuracy (league, market, ml_correct, ml_total, poisson_correct, poisson_total, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(league, market) DO UPDATE SET
            ml_correct = ml_correct + excluded.ml_correct,
            ml_total = ml_total + excluded.ml_total,
            poisson_correct = poisson_correct + excluded.poisson_correct,
            poisson_total = poisson_total + excluded.poisson_total,
            last_updated = datetime('now')
    """, (league, market, int(ml_correct), 1, int(poisson_correct), 1))
    conn.commit()
    conn.close()


def get_ml_league_accuracy(league: str, market: str, min_samples: int = 5) -> dict:
    """Get ML vs Poisson accuracy for a league/market.
    
    Returns dict with:
        ml_accuracy: float or None
        poisson_accuracy: float or None
        ml_total: int
        poisson_total: int
        use_ml: bool (True if ML is more accurate and has enough samples)
    """
    conn = get_db()
    row = conn.execute("""
        SELECT ml_correct, ml_total, poisson_correct, poisson_total
        FROM ml_league_accuracy
        WHERE league = ? AND market = ?
    """, (league, market)).fetchone()
    conn.close()
    
    result = {
        "ml_accuracy": None,
        "poisson_accuracy": None,
        "ml_total": 0,
        "poisson_total": 0,
        "use_ml": False,
    }
    
    if not row:
        return result
    
    ml_total = row["ml_total"] or 0
    poisson_total = row["poisson_total"] or 0
    ml_correct = row["ml_correct"] or 0
    poisson_correct = row["poisson_correct"] or 0
    
    result["ml_total"] = ml_total
    result["poisson_total"] = poisson_total
    
    if ml_total >= min_samples:
        result["ml_accuracy"] = ml_correct / ml_total
    if poisson_total >= min_samples:
        result["poisson_accuracy"] = poisson_correct / poisson_total
    
    if ml_total >= min_samples and poisson_total >= min_samples:
        result["use_ml"] = result["ml_accuracy"] >= result["poisson_accuracy"]
    elif ml_total >= min_samples:
        result["use_ml"] = result["ml_accuracy"] > 0.45
    
    return result


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


def save_calibration_bias(league: str, market: str, threshold: str, bucket: str,
                         predicted_mean: float, actual_mean: float, sample_count: int):
    """Store a bias correction entry for a league/market/bucket."""
    bias = actual_mean - predicted_mean
    conn = get_db()
    conn.execute("""
        INSERT OR REPLACE INTO calibration_bias
            (league, market, threshold, bucket,
             predicted_mean, actual_mean, bias, sample_count, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
    """, (league or "unknown", market, threshold, bucket,
          predicted_mean, actual_mean, bias, sample_count))
    conn.commit()
    conn.close()
    return bias


def get_calibration_biases(league: str = None, market: str = None, min_samples: int = 10) -> list:
    """Get bias corrections, optionally filtered."""
    conn = get_db()
    query = """
        SELECT league, market, threshold, bucket,
               predicted_mean, actual_mean, bias, sample_count, last_updated
        FROM calibration_bias
        WHERE sample_count >= ?
    """
    params = [min_samples]
    if league:
        query += " AND league = ?"
        params.append(league)
    if market:
        query += " AND market = ?"
        params.append(market)
    query += " ORDER BY ABS(bias) DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_calibration_data_for_retraining(min_samples: int = 20) -> dict:
    """Get aggregated calibration data to decide if retraining is needed."""
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) as cnt FROM calibration_log").fetchone()["cnt"]
    recent = conn.execute("""
        SELECT COUNT(*) as cnt FROM calibration_log
        WHERE created_at >= datetime('now', '-30 days')
    """).fetchone()["cnt"]

    acc_by_market = conn.execute("""
        SELECT market,
               COUNT(*) as total,
               SUM(correct) as correct,
               ROUND(100.0 * SUM(correct) / COUNT(*), 1) as pct
        FROM calibration_log
        WHERE market IS NOT NULL
        GROUP BY market
    """).fetchall()

    last_retrain = conn.execute("""
        SELECT MAX(created_at) as last_time, examples_after as last_examples
        FROM model_retrain_log
    """).fetchone()

    conn.close()
    return {
        "total_calibration_entries": total,
        "recent_30d": recent,
        "accuracy_by_market": [dict(r) for r in acc_by_market],
        "last_retrain_time": last_retrain["last_time"] if last_retrain else None,
        "last_retrain_examples": last_retrain["last_examples"] if last_retrain else 0,
    }


def log_retrain(triggered_by: str, examples_before: int, examples_after: int,
                acc_1x2_before: float, acc_1x2_after: float,
                acc_ou_before: float, acc_ou_after: float):
    """Log a model retrain event."""
    conn = get_db()
    conn.execute("""
        INSERT INTO model_retrain_log
            (triggered_by, examples_before, examples_after,
             accuracy_1x2_before, accuracy_1x2_after,
             accuracy_ou_before, accuracy_ou_after)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (triggered_by, examples_before, examples_after,
          acc_1x2_before, acc_1x2_after,
          acc_ou_before, acc_ou_after))
    conn.commit()
    conn.close()


# ── Pending results tracking ─────────────────────────────────────
# A lightweight JSON file that tracks which matches still need their
# results scraped.  Every predict run appends; update_results removes
# entries once scores are written to the DB.

PENDING_PATH = DB_DIR / "data" / "pending_results.json"


def _load_pending() -> list:
    if PENDING_PATH.exists():
        try:
            return json.loads(PENDING_PATH.read_text())
        except Exception:
            return []
    return []


def _save_pending(entries: list):
    PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    PENDING_PATH.write_text(json.dumps(entries, indent=2))


def add_pending_result(match_id: int, forebet_url: str, home_team: str, away_team: str):
    """Add a match to the pending-results tracker (idempotent by URL)."""
    entries = _load_pending()
    if any(e.get("forebet_url") == forebet_url for e in entries):
        return
    entries.append({
        "match_id": match_id,
        "forebet_url": forebet_url,
        "home_team": home_team,
        "away_team": away_team,
        "added": datetime.now().isoformat(),
    })
    _save_pending(entries)


def remove_pending_result(forebet_url: str):
    """Remove a match from the pending-results tracker."""
    entries = _load_pending()
    entries = [e for e in entries if e.get("forebet_url") != forebet_url]
    _save_pending(entries)


def get_pending_results() -> list:
    """Return all pending result entries."""
    return _load_pending()


# Auto-initialize on import
if not DB_PATH.exists():
    init_db()
