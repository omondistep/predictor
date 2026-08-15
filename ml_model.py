#!/usr/bin/env python3
"""
ML Prediction Module — trained models + hybrid ensemble for football prediction.

Borrows best practices from the game/ system:
  - RandomForest + GradientBoosting classifiers for 1X2 and O/U
  - Attack/defense strength matchup via independent Poisson distributions
  - Weighted factor scoring (form, position, goals, H2H)
  - Probability calibration with draw inflation
  - Hybrid ensemble combining ML + Poisson + Forebet
  - Mutual information feature analysis

Usage:
  python ml_model.py --train                 Train models from game dataset + history.db
  python ml_model.py --train --predict       Train then predict
"""

import json
import math
import os
import pickle
import re
import sqlite3
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

_PARALLEL_MISUSE = "`sklearn.utils.parallel.delayed` should be used with"
if _PARALLEL_MISUSE not in os.environ.get("PYTHONWARNINGS", ""):
    _cur_warn = os.environ.get("PYTHONWARNINGS", "")
    os.environ["PYTHONWARNINGS"] = (_cur_warn + "," if _cur_warn else "") + f"ignore:{_PARALLEL_MISUSE}:UserWarning"
warnings.filterwarnings("ignore", message=_PARALLEL_MISUSE)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent
MODELS_DIR = BASE / "ml_models"
DB_PATH = BASE / "history.db"
GAME_DATA = Path(os.environ.get("GAME_DATA_PATH", "/home/stdk/game/data/historical_matches_combined.json"))
GAME_MODELS = Path(os.environ.get("GAME_MODELS_PATH", "/home/stdk/game/models"))
# Override via env var
if os.environ.get("ML_MODELS_DIR"):
    MODELS_DIR = Path(os.environ.get("ML_MODELS_DIR"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature engineering (borrowed from game/ prediction_model.py)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    # Form features
    "home_form_pts", "away_form_pts", "form_diff",
    # Time-weighted form (from game system - more recent results weighted higher)
    "home_form_pts_tw", "away_form_pts_tw", "form_diff_tw",
    # Position features
    "home_pos_score", "away_pos_score", "pos_diff",
    # Goal average features
    "home_gf_avg", "home_ga_avg", "away_gf_avg", "away_ga_avg",
    "home_gd_per_game", "away_gd_per_game",
    # Expected goals (Poisson matchup)
    "exp_home_goals", "exp_away_goals", "exp_total_goals",
    # H2H features
    "h2h_home_wins", "h2h_draws", "h2h_away_wins", "h2h_total",
    # League profile features
    "league_avg_goals", "league_draw_rate", "league_home_win_rate",
    # Forebet probabilities (when available)
    "fb_home_pct", "fb_draw_pct", "fb_away_pct",
    # Odds features
    "odds_home", "odds_draw", "odds_away",
    # League volatility
    "league_volatility",
    # League encoding
    "league_encoding",
    # Derived features
    "prob_diff_home_away", "implied_home_prob",
    # Odds-Forebet relationship features
    "odds_fb_value_home", "draw_concentration",
    "goals_vs_league", "home_advantage_signal",
    "prob_entropy", "odds_implied_overround",
    "favorite_strength", "gap_favorite_underdog",
    # Possession and passing
    "poss_diff", "passes_pg_diff", "pass_acc_diff",
    # Shots
    "home_shots_pg", "away_shots_pg", "shots_diff",
    "home_sot_pct", "away_sot_pct", "sot_diff",
    # Corners, fouls, cards
    "home_corners_avg", "away_corners_avg", "corners_diff",
    "home_fouls_avg", "away_fouls_avg", "fouls_diff",
    "home_yellows_avg", "away_yellows_avg", "yellows_diff",
    # Half-time goals
    "ht_home_goals", "ht_away_goals", "ht_total_goals",
    # Dangerous attacks
    "home_dang_attacks", "away_dang_attacks", "dang_attacks_diff",
    # Injury/suspension features
    "home_injured_total", "away_injured_total", "injured_diff",
    "home_forwards_out", "away_forwards_out", "forwards_out_diff",
    "home_midfielders_out", "away_midfielders_out", "midfielders_out_diff",
    "home_defenders_out", "away_defenders_out", "defenders_out_diff",
    "home_key_players_out", "away_key_players_out", "key_players_out_diff",
    "home_suspended", "away_suspended", "suspended_diff",
    # Missing data indicators
    "home_form_missing", "away_form_missing",
    "home_goals_missing", "away_goals_missing",
    "h2h_missing",
    # Proxy xG (computed from Forebet shot/attack data)
    "home_xg_proxy", "away_xg_proxy", "xg_proxy_diff",
    # ── NEW: Venue-specific performance (from game system) ──
    "home_home_gf_avg", "home_home_ga_avg",  # Home team's performance at home
    "away_away_gf_avg", "away_away_ga_avg",  # Away team's performance away
    # ── NEW: Attack strength engineered features ──
    "home_attack_strength", "away_attack_strength", "attack_ratio",
    # ── NEW: Form components (W/D/L counts from form string) ──
    "home_wins_l6", "home_draws_l6", "home_losses_l6",
    "away_wins_l6", "away_draws_l6", "away_losses_l6",
    # ── NEW: Combined strength scores ──
    "home_strength_score", "away_strength_score",
    # ── NEW: Actual xG data (from Forebet squad xG) ──
    "home_xg", "away_xg", "home_xga", "away_xga",
    # ── NEW: Raw possession/pass values ──
    "home_possession", "away_possession",
    "home_pass_acc", "away_pass_acc",
    # ── NEW: Clean sheets and scoring consistency ──
    "home_clean_sheets_pct", "away_clean_sheets_pct",
    "home_scored_pct", "away_scored_pct",
    # ── NEW: Common opponent scoring analysis (from Forebet insights) ──
    "home_common_scoring_rate",    # % of common opponents home team scored against
    "away_common_scoring_rate",    # % of common opponents away team scored against
    "home_common_defense_vuln",    # Avg goals conceded vs common opponents
    "away_common_defense_vuln",    # Avg goals conceded vs common opponents
    "common_btts_rate",            # BTTS rate against common opponents
    "common_over25_rate",          # O2.5 rate against common opponents
    "home_hidden_strength",        # 1 if team scores ≥80% despite poor record
    "away_hidden_strength",        # 1 if team scores ≥80% despite poor record
]

# ── Feature categories: market-derived vs independent ──
# MARKET features are derived from odds/forebet — they encode what the market
# already knows. The model should NOT rely on these to make predictions,
# because the market is already efficient at incorporating this information.
# INDEPENDENT features come from team performance data and capture
# information the market may miss.
MARKET_FEATURES = frozenset({
    25, 26, 27,   # fb_home_pct, fb_draw_pct, fb_away_pct
    28, 29, 30,   # odds_home, odds_draw, odds_away
    33,           # prob_diff_home_away (= fb_home - fb_away)
    34,           # implied_home_prob (= 1/odds_home)
    35,           # odds_fb_value_home (= fb_home - implied_home)
    36,           # draw_concentration (= fb_draw*2 - 1)
    38,           # home_advantage_signal (fb-based)
    39,           # prob_entropy (Shannon entropy of fb probs)
    40,           # odds_implied_overround (= sum(1/odds) - 1)
    41,           # favorite_strength (= max of fb probs)
    42,           # gap_favorite_underdog (= sorted fb range)
})
INDEPENDENT_FEATURES = frozenset(i for i in range(len(FEATURE_NAMES))
                                  if i not in MARKET_FEATURES)

TARGET_1X2 = "target_1x2"     # 0=away, 1=draw, 2=home
TARGET_OU = "target_ou"        # 0=under, 1=over


def _ppg(form_str: str) -> float:
    """Points per game from form string."""
    pts = sum(3 if c == "W" else 1 if c == "D" else 0 for c in form_str if c in "WDL")
    n = sum(1 for c in form_str if c in "WDL")
    return pts / n if n >= 3 else 1.2


def _form_wdl(form_str: str) -> tuple:
    """Extract W/D/L counts from form string (e.g., 'WWDWL' -> (3, 1, 1))."""
    w = sum(1 for c in form_str if c == "W")
    d = sum(1 for c in form_str if c == "D")
    l = sum(1 for c in form_str if c == "L")
    return w, d, l


def _time_weighted_form(form_str: str) -> float:
    """Compute time-weighted form giving more weight to recent results.
    
    From game system: recent form is more predictive than older form.
    Uses exponential decay: most recent match gets weight 1.0, 
    second most recent gets 0.8, etc.
    """
    if not form_str:
        return 1.2  # Default neutral form
    
    form_results = [c for c in form_str if c in "WDL"]
    if not form_results:
        return 1.2
    
    # Exponential decay weights (most recent = highest weight)
    weights = [0.85 ** i for i in range(len(form_results))]
    total_weight = sum(weights)
    
    weighted_pts = 0.0
    for i, c in enumerate(form_results):
        pts = 3 if c == "W" else 1 if c == "D" else 0
        weighted_pts += pts * weights[i]
    
    return weighted_pts / total_weight if total_weight > 0 else 1.2


def _compute_attack_defense(home_gf, home_ga, away_gf, away_ga):
    """Compute expected goals using attack/defense matchup."""
    h_gf = home_gf or 1.3
    h_ga = home_ga or 1.0
    a_gf = away_gf or 1.1
    a_ga = away_ga or 1.2
    exp_h = (h_gf + a_ga) / 2.0
    exp_a = (a_gf + h_ga) / 2.0
    return max(0.1, exp_h), max(0.1, exp_a)


def extract_features_from_db_row(row: dict) -> np.ndarray:
    """Build feature vector from a history.db row dict.
    
    Enhanced with time-weighted form features from game system.
    """
    f = []

    # Form features (basic and time-weighted)
    hfp = _ppg(row.get("home_form", ""))
    afp = _ppg(row.get("away_form", ""))
    f.append(hfp)
    f.append(afp)
    f.append(hfp - afp)
    
    # Time-weighted form (from game system - more recent results weighted higher)
    hfp_tw = _time_weighted_form(row.get("home_form", ""))
    afp_tw = _time_weighted_form(row.get("away_form", ""))
    f.append(hfp_tw)
    f.append(afp_tw)
    f.append(hfp_tw - afp_tw)

    # Position features
    hp = row.get("home_pos") or 10
    ap = row.get("away_pos") or 10
    max_pos = 20
    f.append(max(0, 1 - (hp - 1) / (max_pos - 1)))
    f.append(max(0, 1 - (ap - 1) / (max_pos - 1)))
    f.append(hp - ap)

    # Goal averages
    h_gf = row.get("home_avg_goals_for") or 1.3
    h_ga = row.get("home_avg_goals_against") or 1.0
    a_gf = row.get("away_avg_goals_for") or 1.1
    a_ga = row.get("away_avg_goals_against") or 1.2
    f.append(h_gf)
    f.append(h_ga)
    f.append(a_gf)
    f.append(a_ga)
    f.append((h_gf - h_ga) / max(h_ga, 0.1))
    f.append((a_gf - a_ga) / max(a_ga, 0.1))

    # Expected goals
    exp_h, exp_a = _compute_attack_defense(h_gf, h_ga, a_gf, a_ga)
    f.append(exp_h)
    f.append(exp_a)
    f.append(exp_h + exp_a)

    # H2H
    f.append(row.get("h2h_home_wins") or 0)
    f.append(row.get("h2h_draws") or 0)
    f.append(row.get("h2h_away_wins") or 0)
    f.append(row.get("h2h_matches") or 5)

    # League profile (default values if not available)
    f.append(2.8)  # avg_goals
    f.append(0.25)  # draw_rate
    f.append(0.45)  # home_win_rate

    # Forebet probs
    f.append((row.get("forebet_home_pct") or 33) / 100.0)
    f.append((row.get("forebet_draw_pct") or 33) / 100.0)
    f.append((row.get("forebet_away_pct") or 33) / 100.0)

    # Odds
    f.append(row.get("odds_home") or 2.5)
    f.append(row.get("odds_draw") or 3.2)
    f.append(row.get("odds_away") or 3.0)

    # Volatility
    f.append(0.15)

    # League encoding (use profile-based features instead of hash noise)
    league_code = row.get("league", "default")
    lp = _lookup_league_profile(league_code)
    f.append(lp["avg_goals"] / 4.0)

    # Derived features
    fb_h = (row.get("forebet_home_pct") or 33) / 100.0
    fb_d = (row.get("forebet_draw_pct") or 33) / 100.0
    fb_a = (row.get("forebet_away_pct") or 33) / 100.0
    f.append(fb_h - fb_a)
    odds_h = row.get("odds_home") or 2.5
    f.append(1.0 / odds_h if odds_h > 1 else 0.5)

    # Additional features (must match game record extraction indices 32-41)
    league_code = row.get("league", "default")
    lp = _lookup_league_profile(league_code)
    # Odds-Forebet value gap (home)
    odds_implied_h = 1.0 / odds_h if odds_h > 1 else 0.4
    f.append(fb_h - odds_implied_h)  # odds_fb_value_home

    # Draw concentration
    f.append(fb_d * 2 - 1)  # draw_concentration

    # Goals vs league average
    exp_total = (row.get("home_avg_goals_for") or 1.3) + (row.get("away_avg_goals_for") or 1.1)
    f.append(exp_total - lp["avg_goals"])  # goals_vs_league

    # Home advantage signal
    f.append(fb_h - fb_a - (lp["home_win_rate"] - (1 - lp["home_win_rate"] - lp["draw_rate"])))

    # Prob entropy (uncertainty)
    probs = [max(fb_h, 0.01), max(fb_d, 0.01), max(fb_a, 0.01)]
    f.append(-sum(p * math.log(p) for p in probs) / math.log(3))

    # Odds-implied overround
    odds_d = row.get("odds_draw") or 3.2
    odds_a = row.get("odds_away") or 3.0
    f.append((1.0 / odds_h + 1.0 / odds_d + 1.0 / odds_a) - 1.0 if odds_h > 1 else 0.0)

    # Favorite strength
    f.append(max(fb_h, fb_d, fb_a))

    # Gap between favorite and underdog
    sorted_probs = sorted([fb_h, fb_d, fb_a])
    f.append(sorted_probs[2] - sorted_probs[0])

    # Possession and passing (home - away difference)
    h_poss = row.get("home_possession_pct") or 50
    a_poss = row.get("away_possession_pct") or 50
    f.append(h_poss - a_poss)

    h_ppg = row.get("home_passes_per_game") or 40
    a_ppg = row.get("away_passes_per_game") or 40
    f.append(h_ppg - a_ppg)

    h_acc = row.get("home_pass_accuracy_pct") or 75
    a_acc = row.get("away_pass_accuracy_pct") or 75
    f.append(h_acc - a_acc)

    # Shots
    h_shots = row.get("home_total_shots_pg") or 12.0
    a_shots = row.get("away_total_shots_pg") or 10.0
    f.append(h_shots)
    f.append(a_shots)
    f.append(h_shots - a_shots)
    h_sot = row.get("home_shots_ontarget_pct") or 35
    a_sot = row.get("away_shots_ontarget_pct") or 30
    f.append(h_sot)
    f.append(a_sot)
    f.append(h_sot - a_sot)

    # Corners
    h_corners = row.get("home_corners_avg") or 5.0
    a_corners = row.get("away_corners_avg") or 4.5
    f.append(h_corners)
    f.append(a_corners)
    f.append(h_corners - a_corners)

    # Fouls
    h_fouls = row.get("home_fouls_avg") or 11.0
    a_fouls = row.get("away_fouls_avg") or 11.0
    f.append(h_fouls)
    f.append(a_fouls)
    f.append(h_fouls - a_fouls)

    # Yellow cards
    h_yellows = row.get("home_yellow_cards_avg") or 1.8
    a_yellows = row.get("away_yellow_cards_avg") or 1.8
    f.append(h_yellows)
    f.append(a_yellows)
    f.append(h_yellows - a_yellows)

    # Half-time goals
    ht_h = row.get("ht_home_goals")
    ht_a = row.get("ht_away_goals")
    f.append(ht_h if ht_h is not None else -1.0)
    f.append(ht_a if ht_a is not None else -1.0)
    f.append((ht_h + ht_a) if ht_h is not None and ht_a is not None else -1.0)

    # Dangerous attacks
    h_dang = row.get("home_dangerous_attacks_pg") or 12.0
    a_dang = row.get("away_dangerous_attacks_pg") or 10.0
    f.append(h_dang)
    f.append(a_dang)
    f.append(h_dang - a_dang)

    # Injury/suspension features
    h_inj = row.get("home_injured_total") or 0
    a_inj = row.get("away_injured_total") or 0
    f.append(h_inj)
    f.append(a_inj)
    f.append(h_inj - a_inj)
    h_for = row.get("home_forwards_out") or 0
    a_for = row.get("away_forwards_out") or 0
    f.append(h_for)
    f.append(a_for)
    f.append(h_for - a_for)
    h_mid = row.get("home_midfielders_out") or 0
    a_mid = row.get("away_midfielders_out") or 0
    f.append(h_mid)
    f.append(a_mid)
    f.append(h_mid - a_mid)
    h_def = row.get("home_defenders_out") or 0
    a_def = row.get("away_defenders_out") or 0
    f.append(h_def)
    f.append(a_def)
    f.append(h_def - a_def)
    h_key = row.get("home_key_players_out") or 0
    a_key = row.get("away_key_players_out") or 0
    f.append(h_key)
    f.append(a_key)
    f.append(h_key - a_key)
    h_sus = row.get("home_suspended") or 0
    a_sus = row.get("away_suspended") or 0
    f.append(h_sus)
    f.append(a_sus)
    f.append(h_sus - a_sus)

    # Missing data indicators
    f.append(1.0 if not row.get("home_form") else 0.0)
    f.append(1.0 if not row.get("away_form") else 0.0)
    f.append(1.0 if not row.get("home_avg_goals_for") else 0.0)
    f.append(1.0 if not row.get("away_avg_goals_for") else 0.0)
    f.append(1.0 if not row.get("h2h_matches") else 0.0)

    # Proxy xG (computed from Forebet shot/attack data)
    hxg = row.get("home_xg_proxy") or 0
    axg = row.get("away_xg_proxy") or 0
    f.append(hxg)
    f.append(axg)
    f.append(hxg - axg)

    # ── NEW: Venue-specific performance ──
    h_home_gf = row.get("home_home_avg_goals_for") or h_gf
    h_home_ga = row.get("home_home_avg_goals_against") or h_ga
    a_away_gf = row.get("away_away_avg_goals_for") or a_gf
    a_away_ga = row.get("away_away_avg_goals_against") or a_ga
    f.append(h_home_gf)
    f.append(h_home_ga)
    f.append(a_away_gf)
    f.append(a_away_ga)

    # ── NEW: Attack strength engineered features ──
    home_attack = h_gf * a_ga  # Home attack vs away defense
    away_attack = a_gf * h_ga  # Away attack vs home defense
    attack_ratio = home_attack / max(away_attack, 0.01)
    f.append(home_attack)
    f.append(away_attack)
    f.append(attack_ratio)

    # ── NEW: Form components (W/D/L counts) ──
    hw, hd, hl = _form_wdl(row.get("home_form", ""))
    aw, ad, al = _form_wdl(row.get("away_form", ""))
    f.append(hw)
    f.append(hd)
    f.append(hl)
    f.append(aw)
    f.append(ad)
    f.append(al)

    # ── NEW: Combined strength scores ──
    home_strength = (h_gf - h_ga) * 0.4 + (hfp - 1.5) * 0.35 + (1 - (hp - 1) / 19) * 0.25
    away_strength = (a_gf - a_ga) * 0.4 + (afp - 1.5) * 0.35 + (1 - (ap - 1) / 19) * 0.25
    f.append(home_strength)
    f.append(away_strength)

    # ── NEW: Actual xG data ──
    f.append(row.get("home_squad_xg") or 0)
    f.append(row.get("away_squad_xg") or 0)
    f.append(row.get("home_squad_xga") or 0)
    f.append(row.get("away_squad_xga") or 0)

    # ── NEW: Raw possession/pass values ──
    f.append(h_poss)
    f.append(a_poss)
    f.append(h_acc)
    f.append(a_acc)

    # ── NEW: Clean sheets and scoring consistency ──
    f.append(row.get("home_clean_sheets_pct") or 0)
    f.append(row.get("away_clean_sheets_pct") or 0)
    f.append(row.get("home_scored_pct") or 0)
    f.append(row.get("away_scored_pct") or 0)

    # ── NEW: Common opponent scoring analysis ──
    # These features capture insights from Forebet analysis:
    # - Scoring consistency against shared opponents
    # - Defensive vulnerability against shared opponents
    # - BTTS and O/U rates against shared opponents
    # - Hidden strength (scoring despite poor record)
    home_form_str = row.get("home_form") or ""
    away_form_str = row.get("away_form") or ""
    home_form_details = row.get("home_form_details") or []
    away_form_details = row.get("away_form_details") or []

    if home_form_details and away_form_details:
        # Build opponent maps
        home_opp_map = {}
        for m in home_form_details:
            opp = m.get("opponent", "")
            if opp:
                if opp not in home_opp_map:
                    home_opp_map[opp] = []
                home_opp_map[opp].append(m)

        away_opp_map = {}
        for m in away_form_details:
            opp = m.get("opponent", "")
            if opp:
                if opp not in away_opp_map:
                    away_opp_map[opp] = []
                away_opp_map[opp].append(m)

        common_opps = set(home_opp_map.keys()) & set(away_opp_map.keys())

        if common_opps:
            home_scored_count = 0
            home_conceded_total = 0
            away_scored_count = 0
            away_conceded_total = 0
            btts_count = 0
            over25_count = 0
            total_matches = 0

            for opp in common_opps:
                h_matches = home_opp_map[opp]
                a_matches = away_opp_map[opp]

                # Use most recent match against each common opponent
                h_match = max(h_matches, key=lambda m: m.get("date", ""))
                a_match = max(a_matches, key=lambda m: m.get("date", ""))

                h_gf = h_match.get("gf", 0)
                h_ga = h_match.get("ga", 0)
                a_gf = a_match.get("gf", 0)
                a_ga = a_match.get("ga", 0)

                # Scoring consistency
                if h_gf > 0:
                    home_scored_count += 1
                if a_gf > 0:
                    away_scored_count += 1

                # Defensive vulnerability
                home_conceded_total += h_ga
                away_conceded_total += a_ga

                # BTTS check
                if h_gf > 0 and a_gf > 0:
                    btts_count += 1

                # O/U check
                total_goals = h_gf + h_ga + a_gf + a_ga
                if total_goals > 2.5:
                    over25_count += 1

                total_matches += 1

            if total_matches > 0:
                home_scoring_rate = home_scored_count / total_matches
                away_scoring_rate = away_scored_count / total_matches
                home_defense_vuln = home_conceded_total / total_matches
                away_defense_vuln = away_conceded_total / total_matches
                btts_rate = btts_count / total_matches
                over25_rate = over25_count / total_matches

                # Hidden strength: team scores ≥80% despite poor record (≥3 losses)
                home_losses = sum(1 for c in home_form_str if c == "L")
                away_losses = sum(1 for c in away_form_str if c == "L")
                home_hidden = 1.0 if (home_scoring_rate >= 0.8 and home_losses >= 3) else 0.0
                away_hidden = 1.0 if (away_scoring_rate >= 0.8 and away_losses >= 3) else 0.0

                f.append(home_scoring_rate)
                f.append(away_scoring_rate)
                f.append(home_defense_vuln)
                f.append(away_defense_vuln)
                f.append(btts_rate)
                f.append(over25_rate)
                f.append(home_hidden)
                f.append(away_hidden)
            else:
                # No common opponents found
                f.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            # No common opponents found
            f.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    else:
        # No form details available
        f.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    return np.array(f, dtype=np.float32)


def _lookup_league_profile(league_code: str) -> dict:
    """Look up league profile by code for game records."""
    profiles = {
        "Br1": {"avg_goals": 2.47, "draw_rate": 0.21, "home_win_rate": 0.52},
        "Br2": {"avg_goals": 2.61, "draw_rate": 0.27, "home_win_rate": 0.43},
        "Br3": {"avg_goals": 2.27, "draw_rate": 0.31, "home_win_rate": 0.42},
        "Ar1": {"avg_goals": 2.26, "draw_rate": 0.29, "home_win_rate": 0.49},
        "Cl1": {"avg_goals": 2.77, "draw_rate": 0.22, "home_win_rate": 0.48},
        "Us1": {"avg_goals": 2.71, "draw_rate": 0.36, "home_win_rate": 0.36},
        "Ec1": {"avg_goals": 2.31, "draw_rate": 0.19, "home_win_rate": 0.44},
        "Pe1": {"avg_goals": 2.44, "draw_rate": 0.28, "home_win_rate": 0.49},
        "Uy1": {"avg_goals": 1.86, "draw_rate": 0.57, "home_win_rate": 0.14},
        "Se1": {"avg_goals": 3.47, "draw_rate": 0.13, "home_win_rate": 0.40},
        "Se2": {"avg_goals": 3.05, "draw_rate": 0.25, "home_win_rate": 0.42},
        "Fi1": {"avg_goals": 2.50, "draw_rate": 0.25, "home_win_rate": 0.47},
        "Ma1": {"avg_goals": 3.00, "draw_rate": 0.25, "home_win_rate": 0.52},
        "Co1": {"avg_goals": 2.52, "draw_rate": 0.30, "home_win_rate": 0.45},
        "Pa1": {"avg_goals": 3.25, "draw_rate": 0.29, "home_win_rate": 0.29},
        "MX1": {"avg_goals": 2.74, "draw_rate": 0.26, "home_win_rate": 0.45},
    }
    return profiles.get(league_code, {"avg_goals": 2.8, "draw_rate": 0.25, "home_win_rate": 0.45})


def extract_features_from_game_record(r: dict) -> np.ndarray:
    """Build feature vector from a game/ historical record.
    
    Enhanced with time-weighted form features from game system.
    """
    f = np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    predicted_avg_goals = r.get("predicted_avg_goals")
    prob_home = r.get("prob_home")
    prob_draw = r.get("prob_draw")
    prob_away = r.get("prob_away")
    odds = r.get("odds")
    league_code = r.get("league_code") or r.get("short_code") or r.get("league", "default")

    # --- Form features (unavailable → neutral) ---
    f[0] = 1.2
    f[1] = 1.2
    f[2] = 0.0
    
    # --- Time-weighted form (unavailable → neutral) ---
    f[3] = 1.2
    f[4] = 1.2
    f[5] = 0.0

    # --- Position features (unavailable → neutral) ---
    f[6] = 0.5
    f[7] = 0.5
    f[8] = 0.0

    # --- Goal averages from predicted_avg_goals (derive home/away split) ---
    avg = float(predicted_avg_goals) if predicted_avg_goals and predicted_avg_goals > 0 else 2.3
    # Use forebet probs to split expected goals between home and away
    if prob_home is not None and prob_away is not None and (prob_home + prob_away) > 0:
        home_share = prob_home / (prob_home + prob_away)
    else:
        home_share = 0.55
    exp_h = avg * home_share
    exp_a = avg * (1 - home_share)
    # Derive GF/GA estimates from expected goals
    h_gf = exp_h * 1.15  # home advantage factor
    h_ga = exp_a
    a_gf = exp_a * 1.05
    a_ga = exp_h
    f[9] = h_gf
    f[10] = h_ga
    f[11] = a_gf
    f[12] = a_ga
    f[13] = (h_gf - h_ga) / max(h_ga, 0.1)
    f[14] = (a_gf - a_ga) / max(a_ga, 0.1)

    # --- Expected goals ---
    f[15] = exp_h
    f[16] = exp_a
    f[17] = avg

    # --- H2H (unavailable) ---
    f[18] = 0
    f[19] = 0
    f[20] = 0
    f[21] = 5

    # --- League profile from lookup ---
    lp = _lookup_league_profile(league_code)
    f[22] = lp["avg_goals"]
    f[23] = lp["draw_rate"]
    f[24] = lp["home_win_rate"]

    # --- Forebet probabilities (most informative features) ---
    f[25] = (prob_home or 33) / 100.0
    f[26] = (prob_draw or 33) / 100.0
    f[27] = (prob_away or 33) / 100.0

    # --- Odds (infer draw/away odds from home odds using typical ratios) ---
    if odds and odds > 1:
        f[28] = float(odds)
        # Infer draw odds: typical ratio is 1.3-1.5x home odds
        implied_home = 1.0 / odds
        implied_rest = 1.0 - implied_home
        if prob_draw and prob_away and (prob_draw + prob_away) > 0:
            draw_share = prob_draw / (prob_draw + prob_away)
        else:
            draw_share = 0.42
        implied_draw = implied_rest * draw_share
        implied_away = implied_rest * (1 - draw_share)
        f[29] = 1.0 / max(implied_draw, 0.05) if implied_draw > 0 else 3.2
        f[30] = 1.0 / max(implied_away, 0.05) if implied_away > 0 else 3.0
    else:
        f[28] = 2.5
        f[29] = 3.2
        f[30] = 3.0

    # --- Volatility ---
    f[31] = 0.15

    # --- League encoding (use profile-based features instead of hash noise) ---
    f[32] = lp["avg_goals"] / 4.0  # normalized avg goals

    # --- Derived features ---
    f[33] = f[25] - f[27]  # prob_diff_home_away
    f[34] = 1.0 / f[28] if f[28] > 1 else 0.5  # implied_home_prob

    # --- Additional features (must match DB row extraction indices 35-44) ---
    # Odds-Forebet value gap (home)
    odds_implied_h = 1.0 / f[28] if f[28] > 1 else 0.4
    f[35] = f[25] - odds_implied_h  # odds_fb_value_home

    # Draw concentration
    f[36] = f[26] * 2 - 1  # draw_concentration (centered at 0)

    # Goals vs league average
    f[37] = avg - lp["avg_goals"]  # goals_vs_league

    # Home advantage signal
    f[38] = f[25] - f[27] - (lp["home_win_rate"] - (1 - lp["home_win_rate"] - lp["draw_rate"]))

    # Prob entropy (uncertainty)
    probs = [max(f[25], 0.01), max(f[26], 0.01), max(f[27], 0.01)]
    f[39] = -sum(p * math.log(p) for p in probs) / math.log(3)

    # Odds-implied overround
    f[40] = (1.0 / f[28] + 1.0 / f[29] + 1.0 / f[30]) - 1.0 if f[28] > 1 else 0.0

    # Favorite strength
    f[41] = max(f[25], f[26], f[27])

    # Gap between favorite and underdog
    sorted_probs = sorted([f[25], f[26], f[27]])
    f[42] = sorted_probs[2] - sorted_probs[0]

    # Possession/passing (unavailable in game records → neutral)
    f[43] = 0.0
    f[44] = 0.0
    f[45] = 0.0

    # Shots (neutral defaults for game records)
    f[46] = 12.0   # home_shots_pg
    f[47] = 10.0   # away_shots_pg
    f[48] = 2.0    # shots_diff
    f[49] = 35.0   # home_sot_pct
    f[50] = 30.0   # away_sot_pct
    f[51] = 5.0    # sot_diff

    # Corners, fouls, cards
    f[52] = 5.0    # home_corners_avg
    f[53] = 4.5    # away_corners_avg
    f[54] = 0.5    # corners_diff
    f[55] = 11.0   # home_fouls_avg
    f[56] = 11.0   # away_fouls_avg
    f[57] = 0.0    # fouls_diff
    f[58] = 1.8    # home_yellows_avg
    f[59] = 1.8    # away_yellows_avg
    f[60] = 0.0    # yellows_diff

    # Half-time goals (unavailable in game records)
    f[61] = -1.0   # ht_home_goals
    f[62] = -1.0   # ht_away_goals
    f[63] = -1.0   # ht_total_goals

    # Dangerous attacks
    f[64] = 12.0   # home_dang_attacks
    f[65] = 10.0   # away_dang_attacks
    f[66] = 2.0    # dang_attacks_diff

    # Injury/suspension features (unavailable in game records → neutral)
    f[67] = 0.0    # home_injured_total
    f[68] = 0.0    # away_injured_total
    f[69] = 0.0    # injured_diff
    f[70] = 0.0    # home_forwards_out
    f[71] = 0.0    # away_forwards_out
    f[72] = 0.0    # forwards_out_diff
    f[73] = 0.0    # home_midfielders_out
    f[74] = 0.0    # away_midfielders_out
    f[75] = 0.0    # midfielders_out_diff
    f[76] = 0.0    # home_defenders_out
    f[77] = 0.0    # away_defenders_out
    f[78] = 0.0    # defenders_out_diff
    f[79] = 0.0    # home_key_players_out
    f[80] = 0.0    # away_key_players_out
    f[81] = 0.0    # key_players_out_diff
    f[82] = 0.0    # home_suspended
    f[83] = 0.0    # away_suspended
    f[84] = 0.0    # suspended_diff

    # Missing data indicators (game records always missing form/goals/h2h)
    f[85] = 1.0  # home_form_missing
    f[86] = 1.0  # away_form_missing
    f[87] = 1.0  # home_goals_missing
    f[88] = 1.0  # away_goals_missing
    f[89] = 1.0  # h2h_missing

    # Proxy xG (unavailable in game records → zero)
    f[90] = 0.0    # home_xg_proxy
    f[91] = 0.0    # away_xg_proxy
    f[92] = 0.0    # xg_proxy_diff

    return f


def extract_targets_from_game_record(r: dict) -> Tuple[int, int]:
    """Extract 1X2 and O/U targets from game record."""
    hs = int(r.get("home_score", 0))
    aws = int(r.get("away_score", 0))
    if hs > aws:
        t1x2 = 2  # home
    elif hs == aws:
        t1x2 = 1  # draw
    else:
        t1x2 = 0  # away
    tou = 1 if (hs + aws) > 2 else 0  # Over 2.5
    return t1x2, tou


# ---------------------------------------------------------------------------
# ML Model
# ---------------------------------------------------------------------------

class MLPredictor:
    """ML-based predictor with RandomForest + GradientBoosting + XGBoost + LightGBM.

    Uses 4-model ensemble for better generalization. Supports probability
    calibration via Platt scaling (sigmoid) or isotonic regression.
    """

    def __init__(self):
        # 1X2 models
        self.rf_model_1x2: Optional[RandomForestClassifier] = None
        self.gb_model_1x2: Optional[GradientBoostingClassifier] = None
        self.xgb_model_1x2: Optional[xgb.XGBClassifier] = None
        self.lgb_model_1x2: Optional[lgb.LGBMClassifier] = None
        # O/U models
        self.rf_model_ou: Optional[RandomForestClassifier] = None
        self.gb_model_ou: Optional[GradientBoostingClassifier] = None
        self.xgb_model_ou: Optional[xgb.XGBClassifier] = None
        self.lgb_model_ou: Optional[lgb.LGBMClassifier] = None
        self.scaler: Optional[StandardScaler] = None

        # Calibrated versions
        self.cal_rf_1x2: Optional[CalibratedClassifierCV] = None
        self.cal_gb_1x2: Optional[CalibratedClassifierCV] = None
        self.cal_xgb_1x2: Optional[CalibratedClassifierCV] = None
        self.cal_lgb_1x2: Optional[CalibratedClassifierCV] = None
        self.cal_rf_ou: Optional[CalibratedClassifierCV] = None
        self.cal_gb_ou: Optional[CalibratedClassifierCV] = None
        self.cal_xgb_ou: Optional[CalibratedClassifierCV] = None
        self.cal_lgb_ou: Optional[CalibratedClassifierCV] = None

        # Feature selection: indices of features used for training/prediction
        self.feature_indices: Optional[np.ndarray] = None

        self.is_trained = False
        self.training_examples = 0
        self.accuracy_1x2 = 0.0
        self.accuracy_ou = 0.0
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def _calibrate_classifier(self, clf, X_calib, y_calib, method: str = "sigmoid"):
        """Calibrate classifier probabilities using held-out set.

        Fits Platt scaling (sigmoid) on calibration data via cross-validation.
        Falls back to the raw classifier if calibration fails.
        """
        if X_calib is None or y_calib is None or len(X_calib) < 100:
            return None
        try:
            calibrated = CalibratedClassifierCV(clf, method=method, cv=3)
            calibrated.fit(X_calib, y_calib)
            return calibrated
        except Exception:
            return None

    def train(self, X: np.ndarray, y_1x2: np.ndarray, y_ou: np.ndarray,
              sample_weights: Optional[np.ndarray] = None,
              calibration_method: Optional[str] = "sigmoid",
              calibration_split: float = 0.15,
              dates: Optional[list] = None,
              feature_indices: Optional[np.ndarray] = None):
        """Train all models with optional sample weights and probability calibration.

        When feature_indices is provided, only those features are used for
        training. This prevents the model from over-relying on market odds.
        """
        n = len(X)
        if n < 100:
            print(f"Warning: only {n} training examples, need at least 100")
            return

        # Apply feature selection
        if feature_indices is not None and len(feature_indices) < X.shape[1]:
            self.feature_indices = feature_indices
            X = X[:, feature_indices]
            print(f"   Using {len(feature_indices)}/{len(FEATURE_NAMES)} selected features")
        else:
            self.feature_indices = None

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Split calibration set if requested
        X_train, X_calib, y1_train, y1_calib, y2_train, y2_calib = None, None, None, None, None, None
        sw_train = None
        if calibration_method and n >= 200:
            # Chronological split: take last calibration_split fraction as calibration
            # This prevents future information leakage
            cal_size = int(n * calibration_split)
            if dates and len(dates) == n:
                # Sort by date, then take last cal_size as calibration
                sorted_indices = sorted(range(n), key=lambda i: dates[i] or datetime(2020, 1, 1))
                train_indices = sorted_indices[:n - cal_size]
                cal_indices = sorted_indices[n - cal_size:]
            else:
                # Fallback: take last cal_size rows (roughly chronological if data was appended in order)
                train_indices = list(range(n - cal_size))
                cal_indices = list(range(n - cal_size, n))

            X_train = X_scaled[train_indices]
            X_calib = X_scaled[cal_indices]
            y1_train = y_1x2[train_indices]
            y1_calib = y_1x2[cal_indices]
            y2_train = y_ou[train_indices]
            y2_calib = y_ou[cal_indices]
            if sample_weights is not None:
                sw_train = sample_weights[train_indices]
            print(f"   Training: {len(X_train)}, Calibration: {len(X_calib)} (chronological)")
        else:
            X_train = X_scaled
            y1_train = y_1x2
            y2_train = y_ou
            sw_train = sample_weights

        print(f"Training RandomForest for 1X2 ({len(X_train)} examples)...")
        self.rf_model_1x2 = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=15,
            min_samples_split=30, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1, max_features="sqrt",
        )
        self.rf_model_1x2.fit(X_train, y1_train, sample_weight=sw_train)

        print(f"Training GradientBoosting for 1X2 ({len(X_train)} examples)...")
        self.gb_model_1x2 = GradientBoostingClassifier(
            n_estimators=250, max_depth=4, min_samples_leaf=12,
            learning_rate=0.05, subsample=0.80, random_state=42,
            max_features="sqrt",
        )
        self.gb_model_1x2.fit(X_train, y1_train, sample_weight=sw_train)

        print(f"Training RandomForest for O/U ({len(X_train)} examples)...")
        self.rf_model_ou = RandomForestClassifier(
            n_estimators=250, max_depth=6, min_samples_leaf=20,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1,
            max_features="sqrt",
        )
        self.rf_model_ou.fit(X_train, y2_train, sample_weight=sw_train)

        print(f"Training GradientBoosting for O/U ({len(X_train)} examples)...")
        self.gb_model_ou = GradientBoostingClassifier(
            n_estimators=180, max_depth=4, min_samples_leaf=12,
            learning_rate=0.06, subsample=0.80, random_state=42,
            max_features="sqrt",
        )
        self.gb_model_ou.fit(X_train, y2_train, sample_weight=sw_train)

        # XGBoost for 1X2
        print(f"Training XGBoost for 1X2 ({len(X_train)} examples)...")
        self.xgb_model_1x2 = xgb.XGBClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.05,
            subsample=0.80, colsample_bytree=0.80, min_child_weight=10,
            random_state=42, n_jobs=-1, eval_metric="mlogloss",
            use_label_encoder=False,
        )
        self.xgb_model_1x2.fit(X_train, y1_train, sample_weight=sw_train)

        # LightGBM for 1X2
        print(f"Training LightGBM for 1X2 ({len(X_train)} examples)...")
        self.lgb_model_1x2 = lgb.LGBMClassifier(
            n_estimators=250, max_depth=5, learning_rate=0.05,
            subsample=0.80, colsample_bytree=0.80, min_child_samples=20,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        self.lgb_model_1x2.fit(X_train, y1_train, sample_weight=sw_train)

        # XGBoost for O/U
        print(f"Training XGBoost for O/U ({len(X_train)} examples)...")
        self.xgb_model_ou = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.06,
            subsample=0.80, colsample_bytree=0.80, min_child_weight=10,
            random_state=42, n_jobs=-1, eval_metric="mlogloss",
            use_label_encoder=False,
        )
        self.xgb_model_ou.fit(X_train, y2_train, sample_weight=sw_train)

        # LightGBM for O/U
        print(f"Training LightGBM for O/U ({len(X_train)} examples)...")
        self.lgb_model_ou = lgb.LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.06,
            subsample=0.80, colsample_bytree=0.80, min_child_samples=20,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        self.lgb_model_ou.fit(X_train, y2_train, sample_weight=sw_train)

        # Probability calibration on held-out set
        if calibration_method and X_calib is not None and len(X_calib) >= 50:
            print(f"   Calibrating probabilities using {calibration_method}...")
            # 1X2 calibration
            self.cal_rf_1x2 = self._calibrate_classifier(
                self.rf_model_1x2, X_calib, y1_calib, method=calibration_method)
            self.cal_gb_1x2 = self._calibrate_classifier(
                self.gb_model_1x2, X_calib, y1_calib, method=calibration_method)
            self.cal_xgb_1x2 = self._calibrate_classifier(
                self.xgb_model_1x2, X_calib, y1_calib, method=calibration_method)
            self.cal_lgb_1x2 = self._calibrate_classifier(
                self.lgb_model_1x2, X_calib, y1_calib, method=calibration_method)
            # O/U calibration
            self.cal_rf_ou = self._calibrate_classifier(
                self.rf_model_ou, X_calib, y2_calib, method=calibration_method)
            self.cal_gb_ou = self._calibrate_classifier(
                self.gb_model_ou, X_calib, y2_calib, method=calibration_method)
            self.cal_xgb_ou = self._calibrate_classifier(
                self.xgb_model_ou, X_calib, y2_calib, method=calibration_method)
            self.cal_lgb_ou = self._calibrate_classifier(
                self.lgb_model_ou, X_calib, y2_calib, method=calibration_method)
            n_cal = sum(1 for c in [self.cal_rf_1x2, self.cal_gb_1x2, self.cal_xgb_1x2, self.cal_lgb_1x2,
                                    self.cal_rf_ou, self.cal_gb_ou, self.cal_xgb_ou, self.cal_lgb_ou] if c is not None)
            print(f"   Calibrated {n_cal}/8 classifiers")
            self.calibration_method = calibration_method
        else:
            self.cal_rf_1x2 = self.cal_gb_1x2 = self.cal_xgb_1x2 = self.cal_lgb_1x2 = None
            self.cal_rf_ou = self.cal_gb_ou = self.cal_xgb_ou = self.cal_lgb_ou = None
            self.calibration_method = None

        self.is_trained = True
        self.training_examples = n

        # Cross-validation accuracy
        try:
            tscv = TimeSeriesSplit(n_splits=3)
            models_1x2 = [("RF", self.rf_model_1x2), ("GB", self.gb_model_1x2),
                          ("XGB", self.xgb_model_1x2), ("LGB", self.lgb_model_1x2)]
            models_ou = [("RF", self.rf_model_ou), ("GB", self.gb_model_ou),
                         ("XGB", self.xgb_model_ou), ("LGB", self.lgb_model_ou)]

            best_cv_1x2, best_cv_ou = 0.0, 0.0
            for name, model in models_1x2:
                scores = cross_val_score(model, X_scaled, y_1x2, cv=tscv, scoring='accuracy')
                print(f"   CV {name} 1X2: {scores.mean():.3f} (+/-{scores.std() * 2:.3f})")
                best_cv_1x2 = max(best_cv_1x2, scores.mean())
            for name, model in models_ou:
                scores = cross_val_score(model, X_scaled, y_ou, cv=tscv, scoring='accuracy')
                print(f"   CV {name} O/U: {scores.mean():.3f} (+/-{scores.std() * 2:.3f})")
                best_cv_ou = max(best_cv_ou, scores.mean())

            self.cv_accuracy_1x2 = best_cv_1x2
            self.cv_accuracy_ou = best_cv_ou
        except Exception as e:
            print(f"   CV skipped: {e}")
            self.cv_accuracy_1x2 = 0.0
            self.cv_accuracy_ou = 0.0

        # In-sample accuracy
        all_1x2 = [self.rf_model_1x2, self.gb_model_1x2, self.xgb_model_1x2, self.lgb_model_1x2]
        all_ou = [self.rf_model_ou, self.gb_model_ou, self.xgb_model_ou, self.lgb_model_ou]
        self.accuracy_1x2 = max((m.predict(X_scaled) == y_1x2).mean() for m in all_1x2)
        self.accuracy_ou = max((m.predict(X_scaled) == y_ou).mean() for m in all_ou)

        accs_1x2 = [(m.__class__.__name__, (m.predict(X_scaled) == y_1x2).mean()) for m in all_1x2]
        accs_ou = [(m.__class__.__name__, (m.predict(X_scaled) == y_ou).mean()) for m in all_ou]
        print(f"   In-sample 1X2: {', '.join(f'{n}={a:.3f}' for n, a in accs_1x2)}")
        print(f"   In-sample O/U: {', '.join(f'{n}={a:.3f}' for n, a in accs_ou)}")

    def predict_proba_1x2(self, X: np.ndarray) -> np.ndarray:
        """Return ensemble probabilities for [away, draw, home].

        Uses calibrated models when available. Averages 4 models (RF, GB, XGB, LGB).
        """
        X_scaled = self.scaler.transform(X)

        def _get_proba(model, cal_model):
            if cal_model is not None:
                return cal_model.predict_proba(X_scaled)
            return model.predict_proba(X_scaled)

        probas = [
            _get_proba(self.rf_model_1x2, self.cal_rf_1x2),
            _get_proba(self.gb_model_1x2, self.cal_gb_1x2),
            _get_proba(self.xgb_model_1x2, self.cal_xgb_1x2),
            _get_proba(self.lgb_model_1x2, self.cal_lgb_1x2),
        ]

        return sum(probas) / len(probas)

    def predict_proba_ou(self, X: np.ndarray) -> np.ndarray:
        """Return ensemble probabilities for [under, over].

        Uses calibrated models when available. Averages 4 models (RF, GB, XGB, LGB).
        """
        X_scaled = self.scaler.transform(X)

        def _get_proba(model, cal_model):
            if cal_model is not None:
                return cal_model.predict_proba(X_scaled)
            return model.predict_proba(X_scaled)

        probas = [
            _get_proba(self.rf_model_ou, self.cal_rf_ou),
            _get_proba(self.gb_model_ou, self.cal_gb_ou),
            _get_proba(self.xgb_model_ou, self.cal_xgb_ou),
            _get_proba(self.lgb_model_ou, self.cal_lgb_ou),
        ]

        return sum(probas) / len(probas)

    def predict_from_row(self, row: dict) -> dict:
        """Predict using one row of features."""
        fv = extract_features_from_db_row(row).reshape(1, -1)
        # Apply feature selection if model was trained with a subset
        if self.feature_indices is not None:
            fv = fv[:, self.feature_indices]
        try:
            expected = getattr(self.scaler, "n_features_in_", None)
        except Exception:
            expected = None
        if expected is not None and int(fv.shape[1]) != int(expected):
            import logging
            logging.getLogger("predictor").warning(
                "ML feature mismatch: got %d features, scaler expects %d — "
                "falling back to Poisson-only for this match. Retrain the model "
                "after feature-set changes.", int(fv.shape[1]), int(expected)
            )
            return {}
        p_1x2 = self.predict_proba_1x2(fv)[0]
        p_ou = self.predict_proba_ou(fv)[0]
        return {
            "ml_prob_away": float(p_1x2[0]),
            "ml_prob_draw": float(p_1x2[1]),
            "ml_prob_home": float(p_1x2[2]),
            "ml_prob_under": float(p_ou[0]),
            "ml_prob_over": float(p_ou[1]),
            "ml_prediction": "Home" if p_1x2[2] > max(p_1x2[0], p_1x2[1])
                            else "Away" if p_1x2[0] > p_1x2[1]
                            else "Draw",
            "ml_ou_prediction": "Over" if p_ou[1] > p_ou[0] else "Under",
        }

    def save(self):
        """Save individual model components to disk, including calibrated versions."""
        import joblib
        path = MODELS_DIR / "ml_predictor"
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path / "scaler.joblib")
        joblib.dump(self.rf_model_1x2, path / "rf_1x2.joblib")
        joblib.dump(self.gb_model_1x2, path / "gb_1x2.joblib")
        joblib.dump(self.xgb_model_1x2, path / "xgb_1x2.joblib")
        joblib.dump(self.lgb_model_1x2, path / "lgb_1x2.joblib")
        joblib.dump(self.rf_model_ou, path / "rf_ou.joblib")
        joblib.dump(self.gb_model_ou, path / "gb_ou.joblib")
        joblib.dump(self.xgb_model_ou, path / "xgb_ou.joblib")
        joblib.dump(self.lgb_model_ou, path / "lgb_ou.joblib")
        # Save calibrated models if they exist
        for name, model in [("cal_rf_1x2", self.cal_rf_1x2), ("cal_gb_1x2", self.cal_gb_1x2),
                            ("cal_xgb_1x2", self.cal_xgb_1x2), ("cal_lgb_1x2", self.cal_lgb_1x2),
                            ("cal_rf_ou", self.cal_rf_ou), ("cal_gb_ou", self.cal_gb_ou),
                            ("cal_xgb_ou", self.cal_xgb_ou), ("cal_lgb_ou", self.cal_lgb_ou)]:
            if model is not None:
                joblib.dump(model, path / f"{name}.joblib")
        meta = {
            "is_trained": self.is_trained,
            "training_examples": self.training_examples,
            "accuracy_1x2": self.accuracy_1x2,
            "accuracy_ou": self.accuracy_ou,
            "cv_accuracy_1x2": getattr(self, 'cv_accuracy_1x2', 0.0),
            "cv_accuracy_ou": getattr(self, 'cv_accuracy_ou', 0.0),
            "calibration_method": getattr(self, 'calibration_method', None),
            "models": ["RF", "GB", "XGB", "LGB"],
            "feature_indices": self.feature_indices.tolist() if self.feature_indices is not None else None,
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f)
        print(f"Saved ML model components to {path}/")

    @staticmethod
    def load(auto_train: bool = True) -> Optional["MLPredictor"]:
        """
        Load trained model. If none exists and auto_train=True, train one.
        (improvement 1: auto-train on startup)
        """
        import joblib, shutil
        path = MODELS_DIR / "ml_predictor"
        meta_path = path / "meta.json"
        if not meta_path.exists():
            if auto_train:
                print("[ML] No trained model found. Auto-training...")
                try:
                    return train()
                except Exception as e:
                    print(f"[ML] Auto-training failed: {e}")
            return None
        ml = MLPredictor()
        try:
            ml.scaler = joblib.load(path / "scaler.joblib")
            ml.rf_model_1x2 = joblib.load(path / "rf_1x2.joblib")
            ml.gb_model_1x2 = joblib.load(path / "gb_1x2.joblib")
            ml.rf_model_ou = joblib.load(path / "rf_ou.joblib")
            ml.gb_model_ou = joblib.load(path / "gb_ou.joblib")
            # Load XGBoost/LightGBM if they exist (backward compatible)
            xgb_1x2_path = path / "xgb_1x2.joblib"
            lgb_1x2_path = path / "lgb_1x2.joblib"
            if xgb_1x2_path.exists():
                ml.xgb_model_1x2 = joblib.load(xgb_1x2_path)
                ml.lgb_model_1x2 = joblib.load(lgb_1x2_path)
                ml.xgb_model_ou = joblib.load(path / "xgb_ou.joblib")
                ml.lgb_model_ou = joblib.load(path / "lgb_ou.joblib")
            else:
                print("[ML] XGBoost/LightGBM not found — retraining with all 4 models...")
                try:
                    shutil.rmtree(path, ignore_errors=True)
                    return train()
                except Exception as e2:
                    print(f"[ML] Retraining failed: {e2}")
                    return None
        except Exception as e:
            print(f"[ML] Model files incompatible with current environment ({e}). Retraining...")
            try:
                shutil.rmtree(path, ignore_errors=True)
                return train()
            except Exception as e2:
                print(f"[ML] Retraining failed: {e2}")
                return None
        # Load calibrated models if they exist
        cal_paths = {
            "cal_rf_1x2": "cal_rf_1x2.joblib",
            "cal_gb_1x2": "cal_gb_1x2.joblib",
            "cal_xgb_1x2": "cal_xgb_1x2.joblib",
            "cal_lgb_1x2": "cal_lgb_1x2.joblib",
            "cal_rf_ou": "cal_rf_ou.joblib",
            "cal_gb_ou": "cal_gb_ou.joblib",
            "cal_xgb_ou": "cal_xgb_ou.joblib",
            "cal_lgb_ou": "cal_lgb_ou.joblib",
        }
        for attr, fname in cal_paths.items():
            fpath = path / fname
            if fpath.exists():
                try:
                    setattr(ml, attr, joblib.load(fpath))
                except Exception:
                    setattr(ml, attr, None)
            else:
                setattr(ml, attr, None)
        with open(meta_path) as f:
            meta = json.load(f)
        ml.is_trained = meta["is_trained"]
        ml.training_examples = meta["training_examples"]
        ml.accuracy_1x2 = meta.get("accuracy_1x2", 0.0)
        ml.accuracy_ou = meta.get("accuracy_ou", 0.0)
        ml.cv_accuracy_1x2 = meta.get("cv_accuracy_1x2", 0.0)
        ml.cv_accuracy_ou = meta.get("cv_accuracy_ou", 0.0)
        ml.calibration_method = meta.get("calibration_method", None)
        # Load feature selection indices (None means use all features)
        fi = meta.get("feature_indices")
        ml.feature_indices = np.array(fi, dtype=np.int32) if fi else None
        return ml


# ---------------------------------------------------------------------------
# Dixon-Coles Bivariate Poisson (improvement 4: goal correlation)
# ---------------------------------------------------------------------------

def dixon_coles_prob(exp_h: float, exp_a: float, rho: float = -0.15, max_goals: int = 8) -> Tuple[float, float, float]:
    """
    Dixon-Coles adjusted probabilities accounting for goal correlation.
    rho < 0 means low-scoring draws are less likely than independent Poisson predicts.
    Typical rho values: -0.10 to -0.20 for football.
    """
    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0
    for i in range(max_goals):
        for j in range(max_goals):
            prob = poisson_prob(exp_h, i) * poisson_prob(exp_a, j)
            # Dixon-Coles adjustment for low scores
            if i <= 1 and j <= 1:
                adj = 1.0 + rho * (1 - i / exp_h if exp_h > 0 else 0) * (1 - j / exp_a if exp_a > 0 else 0)
                prob *= max(0.0, adj)
            if i > j:
                p_home += prob
            elif i == j:
                p_draw += prob
            else:
                p_away += prob
    total = p_home + p_draw + p_away
    if total > 0:
        p_home /= total
        p_draw /= total
        p_away /= total
    return p_home, p_draw, p_away


def dixon_coles_draw_inflation(exp_h: float, exp_a: float, rho: float = -0.15) -> float:
    """Compute draw probability with Dixon-Coles correlation."""
    _, p_d, _ = dixon_coles_prob(exp_h, exp_a, rho)
    return p_d


# ---------------------------------------------------------------------------
# Enhanced Poisson + Weighted Factor Engine
# Borrows from game/ WeightedPredictor + predict.py estimate_goals
# ---------------------------------------------------------------------------

def poisson_prob(goals: float, k: int) -> float:
    """P(X=k) for Poisson(goals)."""
    return math.exp(-goals) * (goals ** k) / math.factorial(k)


def prob_home_win(exp_h: float, exp_a: float) -> float:
    """P(Home win) from independent Poissons."""
    return sum(
        poisson_prob(exp_h, h) * poisson_prob(exp_a, a)
        for h in range(8) for a in range(8) if h > a
    )


def prob_draw(exp_h: float, exp_a: float) -> float:
    return sum(poisson_prob(exp_h, s) * poisson_prob(exp_a, s) for s in range(8))


def prob_over(exp_h: float, exp_a: float, threshold: float = 2.5) -> float:
    return 1.0 - sum(poisson_prob(exp_h + exp_a, i) for i in range(int(threshold) + 1))


def compute_attack_defense_strength(
    home_gf: float, home_ga: float, away_gf: float, away_ga: float,
    league_avg_goals: float, home_adv: float = 1.10,
    form_len_h: int = 6, form_len_a: int = 6,
) -> Tuple[float, float]:
    """Compute expected goals using attack/defense strength (Dixon-Coles style).
    Regresses toward league mean when form sample is small.
    
    Improved home_advantage from 1.15 to 1.10 to reduce home bias.
    Actual home win rate is ~41%, so home advantage should be modest.
    """
    league_avg = league_avg_goals / 2.0
    # Shrink team-specific gf/ga toward the league mean by sample size BEFORE
    # deriving strength. This prevents raw season averages (e.g. 2.4 gf) from
    # being taken at face value and over-predicting goals for low-sample/volatile
    # teams. ~10 games → ~50% trust in team data, 6 games → ~38%.
    trust = lambda n: min(1.0, (n or 0) / 16.0)
    h_t = trust(form_len_h)
    a_t = trust(form_len_a)
    home_gf = (home_gf or league_avg) * h_t + league_avg * (1 - h_t)
    home_ga = (home_ga or league_avg) * h_t + league_avg * (1 - h_t)
    away_gf = (away_gf or league_avg) * a_t + league_avg * (1 - a_t)
    away_ga = (away_ga or league_avg) * a_t + league_avg * (1 - a_t)

    home_strength = home_gf / league_avg if league_avg > 0 else 1.0
    home_defense = home_ga / league_avg if league_avg > 0 else 1.0
    away_strength = away_gf / league_avg if league_avg > 0 else 1.0
    away_defense = away_ga / league_avg if league_avg > 0 else 1.0

    exp_h = home_strength * away_defense * league_avg * home_adv
    exp_a = away_strength * home_defense * league_avg * (2.0 - home_adv)

    # Sample-size regression: fewer form games = less trust in team-specific data
    min_games = 8
    h_factor = min(1.0, form_len_h / min_games)
    a_factor = min(1.0, form_len_a / min_games)
    exp_h = exp_h * h_factor + league_avg * home_adv * (1 - h_factor)
    exp_a = exp_a * a_factor + league_avg * (2.0 - home_adv) * (1 - a_factor)

    return max(0.1, exp_h), max(0.1, exp_a)


def poisson_predict(data: dict, profile: dict, use_dixon_coles: bool = True) -> dict:
    """
    Enhanced Poisson prediction with attack/defense strength.
    Uses Dixon-Coles bivariate Poisson when use_dixon_coles=True (improvement 4).
    """
    h_gf = data.get("home_avg_goals_for")
    h_ga = data.get("home_avg_goals_against")
    a_gf = data.get("away_avg_goals_for")
    a_ga = data.get("away_avg_goals_against")
    h_f = _ppg(data.get("home_form", ""))
    a_f = _ppg(data.get("away_form", ""))
    hp = data.get("home_pos")
    ap = data.get("away_pos")

    # Form length (games actually played) for sample-size regression
    hf_len = sum(1 for c in data.get("home_form", "") if c in "WDL")
    af_len = sum(1 for c in data.get("away_form", "") if c in "WDL")

    # Attack/defense strength matchup (with sample-size regression)
    exp_h, exp_a = compute_attack_defense_strength(
        h_gf, h_ga, a_gf, a_ga,
        profile["avg_goals"], profile.get("home_adv", 1.15),
        form_len_h=hf_len, form_len_a=af_len,
    )

    # Form adjustment — capped to avoid streak overreaction
    form_mult_h = min(1.25, max(0.75, h_f / 1.2))
    form_mult_a = min(1.25, max(0.75, a_f / 1.2))
    # Shorter form sequences get more regression toward neutral (1.0)
    form_conf_h = min(1.0, hf_len / 6)
    form_conf_a = min(1.0, af_len / 6)
    form_mult_h = 1.0 + (form_mult_h - 1.0) * form_conf_h
    form_mult_a = 1.0 + (form_mult_a - 1.0) * form_conf_a
    exp_h *= form_mult_h
    exp_a *= form_mult_a

    # Position adjustment
    if hp and ap:
        total_teams = max(hp, ap) + 5
        exp_h *= max(0.7, 1.0 + (total_teams - hp) / total_teams * 0.3)
        exp_a *= max(0.7, 1.0 + (total_teams - ap) / total_teams * 0.3)
        exp_a *= max(0.7, 1.0 - (total_teams - hp) / total_teams * 0.2)
        exp_h *= max(0.7, 1.0 - (total_teams - ap) / total_teams * 0.2)

    # ── No-goal / clean-sheet discount ──
    # Average goals alone double-counts high-scoring freaks and ignores how often a
    # team actually FAILS to score (or keeps a clean sheet). Blend in the observed
    # frequency of goalless matches so exp reflects "scores in X% of games":
    #   exp_h discounted by home's fail-to-score rate AND away's clean-sheet rate
    # (0.5 blend keeps it a nudge, not overriding the attack/defense model above).
    home_score_rate = (data.get("home_scored_pct") or 100) / 100.0
    away_score_rate = (data.get("away_scored_pct") or 100) / 100.0
    home_cs_rate = (data.get("home_clean_sheets_pct") or 0) / 100.0
    away_cs_rate = (data.get("away_clean_sheets_pct") or 0) / 100.0
    exp_h *= (1.0 - 0.5 * (1.0 - home_score_rate)) * (1.0 - 0.5 * away_cs_rate)
    exp_a *= (1.0 - 0.5 * (1.0 - away_score_rate)) * (1.0 - 0.5 * home_cs_rate)

    # Volatility regression
    vol = profile.get("volatility", 0.1)
    base = profile["avg_goals"] / 2.0
    exp_h = exp_h * (1.0 - vol) + base * vol
    exp_a = exp_a * (1.0 - vol) + base * vol
    exp_h, exp_a = max(0.1, exp_h), max(0.1, exp_a)

    # Dixon-Coles or independent Poisson
    if use_dixon_coles:
        rho = profile.get("dixon_coles_rho", -0.12)
        p_home, p_draw, p_away = dixon_coles_prob(exp_h, exp_a, rho)
    else:
        p_home = prob_home_win(exp_h, exp_a)
        p_draw = prob_draw(exp_h, exp_a)
        p_away = 1.0 - p_home - p_draw

    # Draw probability adjustment based on expected goals proximity
    # When expected goals are close (balanced match), draw probability should be higher
    # This fixes the issue where Poisson underestimates draws in balanced matches
    goal_diff = abs(exp_h - exp_a)
    avg_goals = (exp_h + exp_a) / 2.0
    
    # Draw adjustment: balanced matches (goal_diff < 0.3) with low expected goals
    # should have higher draw probability
    if goal_diff < 0.3 and avg_goals < 2.5:
        # Boost draw probability by 5-10% based on how balanced the match is
        draw_boost = (0.3 - goal_diff) / 0.3 * 0.08
        # Reduce home/away proportionally
        p_home *= (1.0 - draw_boost)
        p_away *= (1.0 - draw_boost)
        p_draw += draw_boost * (p_home + p_away)
    
    # Also boost draws when expected goals are very low (< 2.0)
    if avg_goals < 2.0:
        low_goals_boost = (2.0 - avg_goals) / 2.0 * 0.05
        p_home *= (1.0 - low_goals_boost)
        p_away *= (1.0 - low_goals_boost)
        p_draw += low_goals_boost * (p_home + p_away)

    # Normalize
    total = p_home + p_draw + p_away
    if total > 0:
        p_home /= total
        p_draw /= total
        p_away /= total

    # Over/Under
    expected_total = exp_h + exp_a
    p_ov = 1.0 - sum(poisson_prob(expected_total, i) for i in range(3))
    p_un = 1.0 - p_ov

    # Overdispersion adjustment for O/U — real football scores have variance > mean
    vol = profile.get("volatility", 0.10)
    overdisp_factor = 1.0 + vol * 2.0
    adjusted_total = expected_total * overdisp_factor
    p_ov_adj = 1.0 - sum(poisson_prob(adjusted_total, i) for i in range(3))
    # Blend: 70% original Poisson, 30% overdispersed
    p_ov = 0.7 * p_ov + 0.3 * p_ov_adj
    p_un = 1.0 - p_ov

    # BTTS (with Dixon-Coles correlation adjustment)
    p_home_scores = 1.0 - poisson_prob(exp_h, 0)
    p_away_scores = 1.0 - poisson_prob(exp_a, 0)
    p_btts_indep = p_home_scores * p_away_scores
    dc_rho = profile.get("dixon_coles_rho", -0.12)
    if dc_rho < 0:
        p_both_zero = poisson_prob(exp_h, 0) * poisson_prob(exp_a, 0)
        p_btts = p_btts_indep + dc_rho * p_both_zero
    else:
        p_btts = p_btts_indep

    return {
        "prob_home": p_home,
        "prob_draw": p_draw,
        "prob_away": p_away,
        "prob_over": p_ov,
        "prob_under": p_un,
        "prob_btts": p_btts,
        "prob_btts_no": 1.0 - p_btts,
        "exp_home_goals": exp_h,
        "exp_away_goals": exp_a,
        "exp_total": expected_total,
    }


# ---------------------------------------------------------------------------
# Hybrid Ensemble — combine Poisson + ML + Forebet
# ---------------------------------------------------------------------------

# Market-specific default weights based on proven track records:
#   Forebet dominates 1X2 (90.5%), O/U (89%), BTTS (88%)
#   We dominate DC (93.2%)
#   Both weak on DNB (22%)
MARKET_WEIGHT_PROFILES = {
    "1X2":  {"poisson": 0.40, "ml": 0.35, "forebet": 0.25},  # Forebet reduced: learn from results
    "O/U":  {"poisson": 0.30, "ml": 0.20, "forebet": 0.50},
    "BTTS": {"poisson": 0.25, "ml": 0.25, "forebet": 0.50},
    "DC":   {"poisson": 0.50, "ml": 0.20, "forebet": 0.30},
    "DNB":  {"poisson": 0.25, "ml": 0.25, "forebet": 0.50},
}


def ensemble_predict(
    data: dict,
    profile: dict,
    ml_model: Optional[MLPredictor] = None,
    dynamic_weights: Optional[dict] = None,
    league: Optional[str] = None,
    market: Optional[str] = None,
) -> dict:
    """
    Combine Poisson, ML, and Forebet predictions into a single ensemble.

    Weighting strategy:
      - Market-specific profiles when no dynamic weights available
      - Dynamic weights from DB blended 50/50 with market profiles
      - Forebet gets high weight on markets where it's proven strong
    """
    # Poisson prediction (always use Dixon-Coles)
    poisson = poisson_predict(data, profile, use_dixon_coles=True)

    # ML prediction
    ml_pred = None
    if ml_model and ml_model.is_trained:
        ml_pred = ml_model.predict_from_row(data)

    # Forebet probabilities
    fb_h = (data.get("forebet_home_pct") or 0) / 100.0
    fb_d = (data.get("forebet_draw_pct") or 0) / 100.0
    fb_a = (data.get("forebet_away_pct") or 0) / 100.0
    has_forebet = fb_h + fb_d + fb_a > 0

    # Normalize forebet
    fb_total = fb_h + fb_d + fb_a
    if fb_total > 0:
        fb_h /= fb_total
        fb_d /= fb_total
        fb_a /= fb_total

    # Determine weights based on available sources
    # Start with market-specific profiles as defaults
    market_defaults = MARKET_WEIGHT_PROFILES.get(market, {"poisson": 0.50, "ml": 0.20, "forebet": 0.30})

    if dynamic_weights:
        # Blend 80% dynamic (learned) + 20% market profile (proven track records)
        blend = 0.80
        w_poisson = dynamic_weights.get("poisson", market_defaults["poisson"]) * blend + market_defaults["poisson"] * (1 - blend)
        w_ml = dynamic_weights.get("ml", market_defaults["ml"]) * blend + market_defaults["ml"] * (1 - blend)
        w_fb = dynamic_weights.get("forebet", market_defaults["forebet"]) * blend + market_defaults["forebet"] * (1 - blend)
        # Renormalize if a source is missing
        if not ml_pred:
            w_poisson += w_ml
            w_ml = 0.0
        if not has_forebet:
            w_poisson += w_fb
            w_fb = 0.0
        total_w = w_poisson + w_ml + w_fb
        if total_w > 0:
            w_poisson /= total_w
            w_ml /= total_w
            w_fb /= total_w
    else:
        # Use market-specific default profiles
        w_poisson = market_defaults["poisson"]
        w_ml = market_defaults["ml"]
        w_fb = market_defaults["forebet"]
        # Renormalize if a source is missing
        if not ml_pred:
            w_poisson += w_ml
            w_ml = 0.0
        if not has_forebet:
            w_poisson += w_fb
            w_fb = 0.0
        total_w = w_poisson + w_ml + w_fb
        if total_w > 0:
            w_poisson /= total_w
            w_ml /= total_w
            w_fb /= total_w

    # Cap ML weight based on per-league accuracy (improvement 14)
    ml_effective_weight = w_ml
    if ml_pred and w_ml > 0 and ml_model is not None:
        league_key = league or data.get("league", "default")
        try:
            from database import get_ml_league_accuracy
            ml_acc = get_ml_league_accuracy(league_key, "1X2", min_samples=5)
            ml_total = ml_acc.get("ml_total", 0)
            if ml_total >= 5 and ml_acc.get("use_ml"):
                ml_effective_weight = w_ml
            elif ml_total >= 5 and not ml_acc.get("use_ml"):
                ml_effective_weight = w_ml * 0.3  # reduce but don't kill ML
            else:
                # Unknown league — use ML at full weight if model is trained
                ml_effective_weight = w_ml
        except Exception:
            ml_effective_weight = w_ml

    # Disagreement penalty: if ML and Poisson disagree on top pick, reduce ML weight
    if ml_pred and ml_effective_weight > 0.01:
        p_h_poisson = poisson["prob_home"]
        p_a_poisson = poisson["prob_away"]
        p_h_ml = ml_pred["ml_prob_home"]
        p_a_ml = ml_pred["ml_prob_away"]

        poisson_top = "Home" if p_h_poisson >= p_a_poisson else "Away"
        ml_top = "Home" if p_h_ml >= p_a_ml else "Away"

        if poisson_top != ml_top:
            poisson_top_prob = p_h_poisson if poisson_top == "Home" else p_a_poisson
            ml_top_prob = p_h_ml if ml_top == "Home" else p_a_ml
            # If both are reasonably confident and disagree, trust Poisson more
            if poisson_top_prob >= 0.35 and ml_top_prob >= 0.40:
                ml_effective_weight *= 0.5
                eff_total = w_poisson + ml_effective_weight + w_fb
                if eff_total > 0:
                    w_poisson = w_poisson / eff_total
                    ml_effective_weight = ml_effective_weight / eff_total
                    w_fb = w_fb / eff_total

    # Blend probabilities
    p_h = poisson["prob_home"] * w_poisson
    p_d = poisson["prob_draw"] * w_poisson
    p_a = poisson["prob_away"] * w_poisson

    if ml_pred:
        p_h += ml_pred["ml_prob_home"] * ml_effective_weight
        p_d += ml_pred["ml_prob_draw"] * ml_effective_weight
        p_a += ml_pred["ml_prob_away"] * ml_effective_weight

    if has_forebet:
        p_h += fb_h * w_fb
        p_d += fb_d * w_fb
        p_a += fb_a * w_fb

    # Normalize
    total = p_h + p_d + p_a
    p_h /= total
    p_d /= total
    p_a /= total

    # Blend O/U
    p_over = poisson["prob_over"] * w_poisson
    p_under = poisson["prob_under"] * w_poisson
    if ml_pred:
        p_over += ml_pred["ml_prob_over"] * ml_effective_weight
        p_under += ml_pred["ml_prob_under"] * ml_effective_weight
    if has_forebet:
        fb_ou_pct = data.get("forebet_over25_pct") or 50
        fb_ou = fb_ou_pct / 100.0
        p_over += fb_ou * w_fb
        p_under += (1 - fb_ou) * w_fb
    ou_total = p_over + p_under
    p_over /= ou_total
    p_under /= ou_total

    # Determine prediction and confidence
    max_prob = max(p_h, p_d, p_a)
    if p_h == max_prob:
        pred = "Home"
    elif p_d == max_prob:
        pred = "Draw"
    else:
        pred = "Away"

    # Confidence level
    if max_prob >= 0.50:
        confidence = "High"
    elif max_prob >= 0.40:
        confidence = "Medium-High"
    elif max_prob >= 0.35:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "prob_home": round(p_h, 4),
        "prob_draw": round(p_d, 4),
        "prob_away": round(p_a, 4),
        "prob_over": round(p_over, 4),
        "prob_under": round(p_under, 4),
        "prediction": pred,
        "confidence": confidence,
        "max_probability": round(max_prob, 4),
        "method": f"ensemble(mkt={market or 'default'},poisson={w_poisson:.0%},ml={w_ml:.0%},fb={w_fb:.0%})",
        "_poisson": poisson,
        "_ml": ml_pred,
        "_forebet_home": fb_h,
        "_forebet_draw": fb_d,
        "_forebet_away": fb_a,
    }


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def load_training_data(with_weights: bool = False) -> Tuple:
    """Load training data from game dataset + history.db.
    
    When with_weights=True, returns (X, y1, y2, sample_weights, dates) with
    time decay weights (improvement 5: recent matches weighted more).
    DB records get higher base weight since they have richer features.
    dates is a list of datetime objects for chronological splitting.
    """
    X_list, y1_list, y2_list, date_list = [], [], [], []
    weight_list = []
    cutoff = datetime.now() - timedelta(days=365)

    # 1. Game dataset (primary - large but sparse features)
    if GAME_DATA.exists():
        with open(GAME_DATA) as f:
            game_data = json.load(f)
        valid = [r for r in game_data if isinstance(r, dict)
                 and r.get("has_result")
                 and r.get("home_score") is not None
                 and r.get("away_score") is not None]
        print(f"Game dataset: {len(valid)} records with results")
        for r in valid:
            try:
                fv = extract_features_from_game_record(r)
                t1, t2 = extract_targets_from_game_record(r)
                X_list.append(fv)
                y1_list.append(t1)
                y2_list.append(t2)
                # Parse date for chronological ordering
                match_date_str = r.get("date", "")
                match_date = None
                for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y", "%Y/%m/%d"):
                    try:
                        match_date = datetime.strptime(match_date_str, fmt)
                        break
                    except ValueError:
                        continue
                date_list.append(match_date or datetime(2020, 1, 1))
                # Time decay weight - game data gets base weight 1.0
                if with_weights:
                    w = _time_decay_weight(match_date_str, cutoff)
                    weight_list.append(w)
            except Exception:
                continue
    else:
        print(f"Game dataset not found at {GAME_DATA}")

    # 2. History.db (reviewed predictions - fewer records but rich features)
    db_count = 0
    if DB_PATH.exists():
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT * FROM matches
            WHERE reviewed = 1
              AND actual_home_goals IS NOT NULL
              AND actual_away_goals IS NOT NULL
        """).fetchall()
        conn.close()
        print(f"History.db: {len(rows)} reviewed records")
        for row in rows:
            r = dict(row)
            try:
                fv = extract_features_from_db_row(r)
                hs = r["actual_home_goals"]
                aws = r["actual_away_goals"]
                if hs > aws:
                    t1 = 2
                elif hs == aws:
                    t1 = 1
                else:
                    t1 = 0
                t2 = 1 if (hs + aws) > 2 else 0
                X_list.append(fv)
                y1_list.append(t1)
                y2_list.append(t2)
                db_count += 1
                # Parse date for chronological ordering
                match_date_str = r.get("match_date", "")
                match_date = None
                for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y", "%Y/%m/%d"):
                    try:
                        match_date = datetime.strptime(match_date_str, fmt)
                        break
                    except ValueError:
                        continue
                date_list.append(match_date or datetime(2020, 1, 1))
                # DB records get higher weight (2.5x) due to richer features
                if with_weights:
                    w = _time_decay_weight(match_date_str, cutoff) * 2.5
                    weight_list.append(w)
            except Exception:
                continue

    X = np.array(X_list, dtype=np.float32)
    y1 = np.array(y1_list, dtype=np.int32)
    y2 = np.array(y2_list, dtype=np.int32)
    print(f"Total training examples: {len(X)} (game: {len(X)-db_count}, db: {db_count})")
    
    if with_weights and weight_list:
        sw = np.array(weight_list, dtype=np.float32)
        sw = sw / sw.mean()  # Normalize so mean weight = 1.0
        return X, y1, y2, sw, date_list
    
    return X, y1, y2, date_list


def _time_decay_weight(date_str: str, cutoff: datetime, half_life_days: int = 90) -> float:
    """Exponential time decay weight (improvement 5).
    Matches older than cutoff get weight 0.5, recent matches weight 1.0.
    """
    if not date_str:
        return 0.7
    try:
        # Try common date formats
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y", "%Y/%m/%d"):
            try:
                d = datetime.strptime(date_str, fmt)
                break
            except ValueError:
                continue
        else:
            return 0.7
        days_old = (datetime.now() - d).days
        if days_old < 0:
            return 1.0
        weight = 0.5 ** (days_old / half_life_days)
        return max(0.1, weight)
    except Exception:
        return 0.7


def analyze_feature_importance(X: np.ndarray, y1: np.ndarray, y2: np.ndarray = None):
    """Compute mutual information for each feature across both targets."""
    try:
        mi_1x2 = mutual_info_classif(X, y1, random_state=42)
        mi_ou = mutual_info_classif(X, y2, random_state=42) if y2 is not None else mi_1x2
        mi = np.maximum(mi_1x2, mi_ou)
        ranked = sorted(zip(FEATURE_NAMES, mi, mi_1x2, mi_ou), key=lambda x: -x[1])
        independent_count = sum(1 for name, _, _, _ in ranked[:50] if name not in {FEATURE_NAMES[i] for i in MARKET_FEATURES})
        print(f"\nTop 15 features by mutual information (max of 1X2/O/U):")
        for name, score, mi_h, mi_o in ranked[:15]:
            tag = " [MKT]" if name in {FEATURE_NAMES[i] for i in MARKET_FEATURES} else ""
            print(f"  {name:30s} {score:.4f}  (1X2:{mi_h:.4f} O/U:{mi_o:.4f}){tag}")
        print(f"  ... independent in top 50: {independent_count}/50")
    except Exception as e:
        print(f"Feature importance analysis skipped: {e}")


def select_features(X: np.ndarray, y1: np.ndarray, y2: np.ndarray,
                    max_features: int = 50, min_independent: int = 30,
                    ) -> np.ndarray:
    """Select the most predictive features while ensuring independent features
    are well-represented.

    Strategy:
      1. Compute mutual information for both 1X2 and O/U targets.
      2. Rank all features by max(mi_1x2, mi_ou).
      3. Greedily select top features, but require at least `min_independent`
         features from the INDEPENDENT set (not market-derived).
      4. Return the selected feature indices.

    This prevents the model from over-relying on market odds, which are
    already efficient and don't add predictive value beyond what the
    market already prices in.
    """
    n_features = X.shape[1]
    if n_features <= max_features:
        return np.arange(n_features)

    # Compute mutual information for both targets
    try:
        mi_1x2 = mutual_info_classif(X, y1, random_state=42)
        mi_ou = mutual_info_classif(X, y2, random_state=42)
    except Exception:
        return np.arange(min(max_features, n_features))

    # Score = max MI across both targets
    mi_score = np.maximum(mi_1x2, mi_ou)

    # Separate independent and market features by MI score
    independent_ranked = sorted(
        [(i, mi_score[i]) for i in INDEPENDENT_FEATURES if i < n_features],
        key=lambda x: -x[1])
    market_ranked = sorted(
        [(i, mi_score[i]) for i in MARKET_FEATURES if i < n_features],
        key=lambda x: -x[1])

    selected = set()
    # First: pick top min_independent independent features
    for idx, score in independent_ranked[:min_independent]:
        selected.add(idx)

    # Fill remaining slots from all features (independent + market)
    all_ranked = sorted(
        [(i, mi_score[i]) for i in range(n_features)],
        key=lambda x: -x[1])
    for idx, score in all_ranked:
        if len(selected) >= max_features:
            break
        selected.add(idx)

    result = np.array(sorted(selected), dtype=np.int32)
    n_ind = sum(1 for i in result if i in INDEPENDENT_FEATURES)
    n_mkt = sum(1 for i in result if i in MARKET_FEATURES)
    print(f"Feature selection: {len(result)}/{n_features} features "
          f"({n_ind} independent, {n_mkt} market)")
    return result


def train(force: bool = True):
    """Train ML models from available data with time decay and probability calibration.

    Uses Platt scaling (sigmoid) calibration by default when sufficient data exists.
    """
    result = load_training_data(with_weights=True)
    if len(result) == 5:
        X, y1, y2, sw, dates = result
    else:
        X, y1, y2, dates = result
        sw = None

    if len(X) < 100:
        print("Not enough training data. Skipping ML training.")
        return

    analyze_feature_importance(X, y1, y2)

    # Feature selection: pick best features, ensuring independent features
    # are well-represented so the model doesn't just copy market odds
    feature_indices = select_features(X, y1, y2, max_features=50, min_independent=30)

    ml = MLPredictor()
    # Use sigmoid calibration when enough data, fall back otherwise
    cal_method = "sigmoid" if len(X) >= 200 else None
    ml.train(X, y1, y2, sample_weights=sw, calibration_method=cal_method,
             dates=dates, feature_indices=feature_indices)
    ml.save()

    print(f"\nTraining complete: {ml.training_examples} examples, "
          f"1X2 in-sample acc={ml.accuracy_1x2:.3f}, CV acc={getattr(ml, 'cv_accuracy_1x2', 0):.3f}, "
          f"O/U in-sample acc={ml.accuracy_ou:.3f}, CV acc={getattr(ml, 'cv_accuracy_ou', 0):.3f}")
    if ml.calibration_method:
        print(f"  Probability calibration: {ml.calibration_method}")
    return ml


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ML-enhanced predictor")
    parser.add_argument("--train", action="store_true", help="Train ML models")
    parser.add_argument("--predict", action="store_true", help="Run prediction test")
    parser.add_argument("--analyze", action="store_true", help="Analyze feature importance")
    parser.add_argument("--load", action="store_true", help="Load existing model and test")
    args = parser.parse_args()

    ml = None

    if args.train:
        ml = train()
    elif args.load:
        ml = MLPredictor.load()
        if ml:
            print(f"Loaded ML model: {ml.training_examples} examples, "
                  f"1X2 acc={ml.accuracy_1x2:.3f}")
        else:
            print("No trained model found. Run with --train first.")

    if args.analyze and not args.train:
        result = load_training_data()
        X, y1, y2 = result[0], result[1], result[2]
        if len(X) > 0:
            analyze_feature_importance(X, y1, y2)

    if args.predict:
        if ml is None:
            ml = MLPredictor.load()
        if ml and ml.is_trained:
            print("\nTest prediction with sample data:")
            sample = {
                "home_form": "WWDWL", "away_form": "LLDLL",
                "home_pos": 5, "away_pos": 18,
                "home_avg_goals_for": 1.8, "home_avg_goals_against": 0.9,
                "away_avg_goals_for": 0.8, "away_avg_goals_against": 1.9,
                "h2h_home_wins": 3, "h2h_draws": 1, "h2h_away_wins": 1, "h2h_matches": 5,
                "forebet_home_pct": 45, "forebet_draw_pct": 28, "forebet_away_pct": 27,
                "forebet_over25_pct": 52,
                "odds_home": 2.1, "odds_draw": 3.4, "odds_away": 3.5,
                "home_avg_goals_for": 1.8, "home_avg_goals_against": 0.9,
                "away_avg_goals_for": 0.8, "away_avg_goals_against": 1.9,
            }
            profile = {"avg_goals": 2.8, "home_adv": 1.15, "volatility": 0.15}

            # ML only
            ml_result = ml.predict_from_row(sample)
            print(f"  ML: {ml_result['ml_prediction']} "
                  f"(H={ml_result['ml_prob_home']:.2f} D={ml_result['ml_prob_draw']:.2f} "
                  f"A={ml_result['ml_prob_away']:.2f})")

            # Poisson only
            poisson_result = poisson_predict(sample, profile)
            print(f"  Poisson: H={poisson_result['prob_home']:.2f} "
                  f"D={poisson_result['prob_draw']:.2f} "
                  f"A={poisson_result['prob_away']:.2f}")

            # Ensemble
            ensemble_result = ensemble_predict(sample, profile, ml)
            print(f"  Ensemble: {ensemble_result['prediction']} "
                  f"(H={ensemble_result['prob_home']:.2f} "
                  f"D={ensemble_result['prob_draw']:.2f} "
                  f"A={ensemble_result['prob_away']:.2f}) "
                  f"[{ensemble_result['method']}]")
        else:
            print("No trained model available. Run with --train first.")


# ─────────────────────────────────────────────
# ML → FB Model Signal Functions
# ─────────────────────────────────────────────

# Mapping from feature indices to FB model signal categories
_SIGNAL_MAP = {
    "form":        [0, 1, 2, 3, 4, 5, 100, 101, 102, 103, 104, 105],
    "position":    [6, 7, 8],
    "goals":       [9, 10, 11, 12, 13, 14, 15, 16, 17],
    "h2h":         [18, 19, 20, 21],
    "league":      [22, 23, 24, 31, 32],
    "draw":        [23, 36, 6, 7, 8, 116, 117, 118, 119],
    "shots":       [46, 47, 48, 49, 50, 51],
    "attacks":     [64, 65, 66],
    "possession":  [43, 44, 45, 112, 113, 114, 115],
    "ht_goals":    [61, 62, 63],
    "injuries":    list(range(67, 85)),
    "venue":       [93, 94, 95, 96],
    "attack_str":  [97, 98, 99],
    "xg":          [90, 91, 92, 108, 109, 110, 111],
    "consistency": [116, 117, 118, 119],
    "market":      [25, 26, 27, 28, 29, 30, 33, 34, 35, 37, 38, 39, 40, 41, 42],
}


def get_feature_importance_weights(ml_model) -> dict:
    """Extract feature importances from ML classifiers and map to FB signal weights.

    Returns a dict of signal_name → weight (0.0–1.0) indicating how important
    each signal category is according to the ML model's learned patterns.
    """
    if not ml_model or not ml_model.is_trained:
        return {k: 1.0 for k in _SIGNAL_MAP}

    # Collect importances from all available classifiers
    importances = np.zeros(len(FEATURE_NAMES))
    count = 0
    for attr in ("rf_model_1x2", "gb_model_1x2", "xgb_model_1x2", "lgb_model_1x2"):
        clf = getattr(ml_model, attr, None)
        if clf is not None and hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
            if len(imp) == len(FEATURE_NAMES):
                importances += imp
                count += 1
    if count == 0:
        return {k: 1.0 for k in _SIGNAL_MAP}
    importances /= count

    # If feature selection was applied, map back to full feature space
    if ml_model.feature_indices is not None:
        full_imp = np.zeros(len(FEATURE_NAMES))
        for i, idx in enumerate(ml_model.feature_indices):
            if i < len(importances):
                full_imp[idx] = importances[i]
        importances = full_imp

    # Map to signal categories
    weights = {}
    for signal, feat_indices in _SIGNAL_MAP.items():
        valid = [i for i in feat_indices if i < len(importances)]
        if valid:
            weights[signal] = float(np.mean(importances[valid]))
        else:
            weights[signal] = 1.0

    # Normalize so max = 1.0
    max_w = max(weights.values()) if weights else 1.0
    if max_w > 0:
        weights = {k: v / max_w for k, v in weights.items()}

    return weights


def get_ensemble_agreement(ml_model, data: dict) -> float:
    """Compute agreement score between ML classifiers (0.0–1.0).

    High agreement (close to 1.0) means all classifiers agree → more reliable.
    Low agreement (close to 0.0) means classifiers disagree → less reliable.
    """
    if not ml_model or not ml_model.is_trained:
        return 0.5

    try:
        features = ml_model.extract_features_from_row(data)
        if ml_model.feature_indices is not None:
            features = features[ml_model.feature_indices]
        X = ml_model.scaler.transform([features])

        probs_list = []
        for attr in ("rf_model_1x2", "gb_model_1x2", "xgb_model_1x2", "lgb_model_1x2"):
            clf = getattr(ml_model, attr, None)
            if clf is not None:
                p = clf.predict_proba(X)[0]
                probs_list.append(p)

        if len(probs_list) < 2:
            return 0.5

        # Agreement = 1 - mean pairwise variance across classes
        probs_arr = np.array(probs_list)
        variance = np.mean(np.var(probs_arr, axis=0))
        # Map variance [0, ~0.25] to agreement [1, 0]
        agreement = max(0.0, 1.0 - 4.0 * variance)
        return float(agreement)
    except Exception:
        return 0.5


def get_feature_quality_score(ml_model, data: dict) -> float:
    """Return a data quality score (0.0–1.0) based on feature completeness.

    Uses ML's feature extraction to detect missing/noisy data.
    High score = data is complete and reliable.
    Low score = data is sparse or missing key features.
    """
    if not ml_model or not ml_model.is_trained:
        return 0.85

    try:
        features = ml_model.extract_features_from_row(data)
        # Count how many features are zero (missing/Default)
        total = len(features)
        missing = np.sum(np.abs(features) < 1e-6)
        # Also count features that are exactly 0 (explicit missing flags)
        flags = sum(1 for i in [85, 86, 87, 88, 89] if i < total and features[i] > 0)
        completeness = 1.0 - (missing / total)
        flag_penalty = flags * 0.05
        score = max(0.3, min(1.0, completeness - flag_penalty))
        return float(score)
    except Exception:
        return 0.85


if __name__ == "__main__":
    main()
