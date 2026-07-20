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
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, train_test_split
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

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
]

TARGET_1X2 = "target_1x2"     # 0=away, 1=draw, 2=home
TARGET_OU = "target_ou"        # 0=under, 1=over


def _ppg(form_str: str) -> float:
    """Points per game from form string."""
    pts = sum(3 if c == "W" else 1 if c == "D" else 0 for c in form_str if c in "WDL")
    n = sum(1 for c in form_str if c in "WDL")
    return pts / n if n >= 3 else 1.2


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
    """Build feature vector from a history.db row dict."""
    f = []

    # Form features
    hfp = _ppg(row.get("home_form", ""))
    afp = _ppg(row.get("away_form", ""))
    f.append(hfp)
    f.append(afp)
    f.append(hfp - afp)

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
    """Build feature vector from a game/ historical record."""
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

    # --- Position features (unavailable → neutral) ---
    f[3] = 0.5
    f[4] = 0.5
    f[5] = 0.0

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
    f[6] = h_gf
    f[7] = h_ga
    f[8] = a_gf
    f[9] = a_ga
    f[10] = (h_gf - h_ga) / max(h_ga, 0.1)
    f[11] = (a_gf - a_ga) / max(a_ga, 0.1)

    # --- Expected goals ---
    f[12] = exp_h
    f[13] = exp_a
    f[14] = avg

    # --- H2H (unavailable) ---
    f[15] = 0
    f[16] = 0
    f[17] = 0
    f[18] = 5

    # --- League profile from lookup ---
    lp = _lookup_league_profile(league_code)
    f[19] = lp["avg_goals"]
    f[20] = lp["draw_rate"]
    f[21] = lp["home_win_rate"]

    # --- Forebet probabilities (most informative features) ---
    f[22] = (prob_home or 33) / 100.0
    f[23] = (prob_draw or 33) / 100.0
    f[24] = (prob_away or 33) / 100.0

    # --- Odds (infer draw/away odds from home odds using typical ratios) ---
    if odds and odds > 1:
        f[25] = float(odds)
        # Infer draw odds: typical ratio is 1.3-1.5x home odds
        implied_home = 1.0 / odds
        implied_rest = 1.0 - implied_home
        if prob_draw and prob_away and (prob_draw + prob_away) > 0:
            draw_share = prob_draw / (prob_draw + prob_away)
        else:
            draw_share = 0.42
        implied_draw = implied_rest * draw_share
        implied_away = implied_rest * (1 - draw_share)
        f[26] = 1.0 / max(implied_draw, 0.05) if implied_draw > 0 else 3.2
        f[27] = 1.0 / max(implied_away, 0.05) if implied_away > 0 else 3.0
    else:
        f[25] = 2.5
        f[26] = 3.2
        f[27] = 3.0

    # --- Volatility ---
    f[28] = 0.15

    # --- League encoding (use profile-based features instead of hash noise) ---
    f[29] = lp["avg_goals"] / 4.0  # normalized avg goals

    # --- Derived features ---
    f[30] = f[22] - f[24]  # prob_diff_home_away
    f[31] = 1.0 / f[25] if f[25] > 1 else 0.5  # implied_home_prob

    # --- Additional features (must match DB row extraction indices 32-41) ---
    # Odds-Forebet value gap (home)
    odds_implied_h = 1.0 / f[25] if f[25] > 1 else 0.4
    f[32] = f[22] - odds_implied_h  # odds_fb_value_home

    # Draw concentration
    f[33] = f[23] * 2 - 1  # draw_concentration (centered at 0)

    # Goals vs league average
    f[34] = avg - lp["avg_goals"]  # goals_vs_league

    # Home advantage signal
    f[35] = f[22] - f[24] - (lp["home_win_rate"] - (1 - lp["home_win_rate"] - lp["draw_rate"]))

    # Prob entropy (uncertainty)
    probs = [max(f[22], 0.01), max(f[23], 0.01), max(f[24], 0.01)]
    f[36] = -sum(p * math.log(p) for p in probs) / math.log(3)

    # Odds-implied overround
    f[37] = (1.0 / f[25] + 1.0 / f[26] + 1.0 / f[27]) - 1.0 if f[25] > 1 else 0.0

    # Favorite strength
    f[38] = max(f[22], f[23], f[24])

    # Gap between favorite and underdog
    sorted_probs = sorted([f[22], f[23], f[24]])
    f[39] = sorted_probs[2] - sorted_probs[0]

    # Possession/passing (unavailable in game records → neutral)
    f[40] = 0.0
    f[41] = 0.0
    f[42] = 0.0

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
    """ML-based predictor with RandomForest + GradientBoosting.

    Supports probability calibration via Platt scaling (sigmoid) or
    isotonic regression, learned from a held-out calibration set.
    """

    def __init__(self):
        self.rf_model_1x2: Optional[RandomForestClassifier] = None
        self.gb_model_1x2: Optional[GradientBoostingClassifier] = None
        self.rf_model_ou: Optional[RandomForestClassifier] = None
        self.gb_model_ou: Optional[GradientBoostingClassifier] = None
        self.scaler: Optional[StandardScaler] = None

        # Calibrated versions (wrapping the base models)
        self.cal_rf_1x2: Optional[CalibratedClassifierCV] = None
        self.cal_gb_1x2: Optional[CalibratedClassifierCV] = None
        self.cal_rf_ou: Optional[CalibratedClassifierCV] = None
        self.cal_gb_ou: Optional[CalibratedClassifierCV] = None

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
              calibration_split: float = 0.15):
        """Train all models with optional sample weights and probability calibration.

        When calibration_method is set, holds out 'calibration_split' fraction
        of data to fit Platt scaling (sigmoid) or isotonic regression, producing
        better-calibrated probabilities.
        """
        n = len(X)
        if n < 100:
            print(f"Warning: only {n} training examples, need at least 100")
            return

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Split calibration set if requested
        X_train, X_calib, y1_train, y1_calib, y2_train, y2_calib = None, None, None, None, None, None
        sw_train = None
        if calibration_method and n >= 200:
            stratify = y_1x2 if len(np.unique(y_1x2)) >= 3 else None
            split_result = train_test_split(
                X_scaled, y_1x2, y_ou,
                test_size=calibration_split,
                random_state=42,
                stratify=stratify,
            )
            X_train, X_calib, y1_train, y1_calib, y2_train, y2_calib = split_result
            if sample_weights is not None:
                sw_train = sample_weights[:len(X_train)]
            print(f"   Training: {len(X_train)}, Calibration: {len(X_calib)}")
        else:
            X_train = X_scaled
            y1_train = y_1x2
            y2_train = y_ou
            sw_train = sample_weights

        print(f"Training RandomForest for 1X2 ({len(X_train)} examples)...")
        self.rf_model_1x2 = RandomForestClassifier(
            n_estimators=400, max_depth=15, min_samples_leaf=8,
            min_samples_split=20, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1, max_features="sqrt",
        )
        self.rf_model_1x2.fit(X_train, y1_train, sample_weight=sw_train)

        print(f"Training GradientBoosting for 1X2 ({len(X_train)} examples)...")
        self.gb_model_1x2 = GradientBoostingClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=8,
            learning_rate=0.05, subsample=0.85, random_state=42,
            max_features="sqrt",
        )
        self.gb_model_1x2.fit(X_train, y1_train, sample_weight=sw_train)

        print(f"Training RandomForest for O/U ({len(X_train)} examples)...")
        self.rf_model_ou = RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=10,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1,
            max_features="sqrt",
        )
        self.rf_model_ou.fit(X_train, y2_train, sample_weight=sw_train)

        print(f"Training GradientBoosting for O/U ({len(X_train)} examples)...")
        self.gb_model_ou = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=10,
            learning_rate=0.08, subsample=0.85, random_state=42,
            max_features="sqrt",
        )
        self.gb_model_ou.fit(X_train, y2_train, sample_weight=sw_train)

        # Probability calibration on held-out set
        if calibration_method and X_calib is not None and len(X_calib) >= 50:
            print(f"   Calibrating probabilities using {calibration_method}...")
            self.cal_rf_1x2 = self._calibrate_classifier(
                self.rf_model_1x2, X_calib, y1_calib, method=calibration_method)
            self.cal_gb_1x2 = self._calibrate_classifier(
                self.gb_model_1x2, X_calib, y1_calib, method=calibration_method)
            self.cal_rf_ou = self._calibrate_classifier(
                self.rf_model_ou, X_calib, y2_calib, method=calibration_method)
            self.cal_gb_ou = self._calibrate_classifier(
                self.gb_model_ou, X_calib, y2_calib, method=calibration_method)
            n_cal = sum(1 for c in [self.cal_rf_1x2, self.cal_gb_1x2,
                                    self.cal_rf_ou, self.cal_gb_ou] if c is not None)
            print(f"   Calibrated {n_cal}/4 classifiers")
            self.calibration_method = calibration_method
        else:
            self.cal_rf_1x2 = None
            self.cal_gb_1x2 = None
            self.cal_rf_ou = None
            self.cal_gb_ou = None
            self.calibration_method = None

        self.is_trained = True
        self.training_examples = n

        # Cross-validation accuracy (improvement 2)
        try:
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores_1x2_rf = cross_val_score(self.rf_model_1x2, X_scaled, y_1x2, cv=tscv, scoring='accuracy')
            cv_scores_1x2_gb = cross_val_score(self.gb_model_1x2, X_scaled, y_1x2, cv=tscv, scoring='accuracy')
            cv_scores_ou_rf = cross_val_score(self.rf_model_ou, X_scaled, y_ou, cv=tscv, scoring='accuracy')
            cv_scores_ou_gb = cross_val_score(self.gb_model_ou, X_scaled, y_ou, cv=tscv, scoring='accuracy')
            print(f"   CV RF 1X2: {cv_scores_1x2_rf.mean():.3f} (+/-{cv_scores_1x2_rf.std() * 2:.3f})")
            print(f"   CV GB 1X2: {cv_scores_1x2_gb.mean():.3f} (+/-{cv_scores_1x2_gb.std() * 2:.3f})")
            print(f"   CV RF O/U: {cv_scores_ou_rf.mean():.3f} (+/-{cv_scores_ou_rf.std() * 2:.3f})")
            print(f"   CV GB O/U: {cv_scores_ou_gb.mean():.3f} (+/-{cv_scores_ou_gb.std() * 2:.3f})")
            self.cv_accuracy_1x2 = max(cv_scores_1x2_rf.mean(), cv_scores_1x2_gb.mean())
            self.cv_accuracy_ou = max(cv_scores_ou_rf.mean(), cv_scores_ou_gb.mean())
        except Exception as e:
            print(f"   CV skipped: {e}")
            self.cv_accuracy_1x2 = 0.0
            self.cv_accuracy_ou = 0.0

        # In-sample accuracy (for reference)
        rf_acc = (self.rf_model_1x2.predict(X_scaled) == y_1x2).mean()
        gb_acc = (self.gb_model_1x2.predict(X_scaled) == y_1x2).mean()
        self.accuracy_1x2 = max(rf_acc, gb_acc)

        rf_acc_ou = (self.rf_model_ou.predict(X_scaled) == y_ou).mean()
        gb_acc_ou = (self.gb_model_ou.predict(X_scaled) == y_ou).mean()
        self.accuracy_ou = max(rf_acc_ou, gb_acc_ou)

        print(f"   In-sample RF 1X2: {rf_acc:.3f}, GB 1X2: {gb_acc:.3f}")
        print(f"   In-sample RF O/U: {rf_acc_ou:.3f}, GB O/U: {gb_acc_ou:.3f}")

    def predict_proba_1x2(self, X: np.ndarray) -> np.ndarray:
        """Return ensemble probabilities for [away, draw, home].

        Uses calibrated models when available for better-calibrated probabilities.
        """
        X_scaled = self.scaler.transform(X)

        # Use calibrated models if available
        if self.cal_rf_1x2 is not None and self.cal_gb_1x2 is not None:
            rf_proba = self.cal_rf_1x2.predict_proba(X_scaled)
            gb_proba = self.cal_gb_1x2.predict_proba(X_scaled)
        elif self.cal_rf_1x2 is not None:
            rf_proba = self.cal_rf_1x2.predict_proba(X_scaled)
            gb_proba = self.gb_model_1x2.predict_proba(X_scaled)
        elif self.cal_gb_1x2 is not None:
            rf_proba = self.rf_model_1x2.predict_proba(X_scaled)
            gb_proba = self.cal_gb_1x2.predict_proba(X_scaled)
        else:
            rf_proba = self.rf_model_1x2.predict_proba(X_scaled)
            gb_proba = self.gb_model_1x2.predict_proba(X_scaled)

        return (rf_proba + gb_proba) / 2.0

    def predict_proba_ou(self, X: np.ndarray) -> np.ndarray:
        """Return ensemble probabilities for [under, over].

        Uses calibrated models when available for better-calibrated probabilities.
        """
        X_scaled = self.scaler.transform(X)

        if self.cal_rf_ou is not None and self.cal_gb_ou is not None:
            rf_proba = self.cal_rf_ou.predict_proba(X_scaled)
            gb_proba = self.cal_gb_ou.predict_proba(X_scaled)
        elif self.cal_rf_ou is not None:
            rf_proba = self.cal_rf_ou.predict_proba(X_scaled)
            gb_proba = self.gb_model_ou.predict_proba(X_scaled)
        elif self.cal_gb_ou is not None:
            rf_proba = self.rf_model_ou.predict_proba(X_scaled)
            gb_proba = self.cal_gb_ou.predict_proba(X_scaled)
        else:
            rf_proba = self.rf_model_ou.predict_proba(X_scaled)
            gb_proba = self.gb_model_ou.predict_proba(X_scaled)

        return (rf_proba + gb_proba) / 2.0

    def predict_from_row(self, row: dict) -> dict:
        """Predict using one row of features."""
        fv = extract_features_from_db_row(row).reshape(1, -1)
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
        joblib.dump(self.rf_model_ou, path / "rf_ou.joblib")
        joblib.dump(self.gb_model_ou, path / "gb_ou.joblib")
        # Save calibrated models if they exist
        if self.cal_rf_1x2 is not None:
            joblib.dump(self.cal_rf_1x2, path / "cal_rf_1x2.joblib")
        if self.cal_gb_1x2 is not None:
            joblib.dump(self.cal_gb_1x2, path / "cal_gb_1x2.joblib")
        if self.cal_rf_ou is not None:
            joblib.dump(self.cal_rf_ou, path / "cal_rf_ou.joblib")
        if self.cal_gb_ou is not None:
            joblib.dump(self.cal_gb_ou, path / "cal_gb_ou.joblib")
        meta = {
            "is_trained": self.is_trained,
            "training_examples": self.training_examples,
            "accuracy_1x2": self.accuracy_1x2,
            "accuracy_ou": self.accuracy_ou,
            "cv_accuracy_1x2": getattr(self, 'cv_accuracy_1x2', 0.0),
            "cv_accuracy_ou": getattr(self, 'cv_accuracy_ou', 0.0),
            "calibration_method": getattr(self, 'calibration_method', None),
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
            "cal_rf_ou": "cal_rf_ou.joblib",
            "cal_gb_ou": "cal_gb_ou.joblib",
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
    league_avg_goals: float, home_adv: float = 1.15,
    form_len_h: int = 6, form_len_a: int = 6,
) -> Tuple[float, float]:
    """Compute expected goals using attack/defense strength (Dixon-Coles style).
    Regresses toward league mean when form sample is small."""
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

        # Legacy draw inflation
        goal_diff = abs(exp_h - exp_a)
        if goal_diff < 0.4:
            draw_boost = (0.4 - goal_diff) / 0.4 * 0.05
            p_draw += draw_boost
            p_home *= (1.0 - draw_boost) / (p_home + p_away + 1e-10)
            p_away = 1.0 - p_home - p_draw

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
    "1X2":  {"poisson": 0.20, "ml": 0.25, "forebet": 0.55},
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
        # Blend 50% dynamic (learned) + 50% market profile (proven track records)
        blend = 0.50
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
    
    When with_weights=True, returns (X, y1, y2, sample_weights) with
    time decay weights (improvement 5: recent matches weighted more).
    DB records get higher base weight since they have richer features.
    """
    X_list, y1_list, y2_list = [], [], []
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
                # Time decay weight - game data gets base weight 1.0
                if with_weights:
                    match_date = r.get("date", "")
                    w = _time_decay_weight(match_date, cutoff)
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
                # DB records get higher weight (2.5x) due to richer features
                if with_weights:
                    w = _time_decay_weight(r.get("match_date", ""), cutoff) * 2.5
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
        return X, y1, y2, sw
    
    return X, y1, y2


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


def analyze_feature_importance(X: np.ndarray, y: np.ndarray):
    """Compute mutual information for each feature."""
    try:
        mi = mutual_info_classif(X, y, random_state=42)
        ranked = sorted(zip(FEATURE_NAMES, mi), key=lambda x: -x[1])
        print(f"\nTop 15 features by mutual information:")
        for name, score in ranked[:15]:
            print(f"  {name:25s} {score:.4f}")
    except Exception as e:
        print(f"Feature importance analysis skipped: {e}")


def train(force: bool = True):
    """Train ML models from available data with time decay and probability calibration.

    Uses Platt scaling (sigmoid) calibration by default when sufficient data exists.
    """
    result = load_training_data(with_weights=True)
    if len(result) == 4:
        X, y1, y2, sw = result
    else:
        X, y1, y2 = result
        sw = None

    if len(X) < 100:
        print("Not enough training data. Skipping ML training.")
        return

    analyze_feature_importance(X, y1)

    ml = MLPredictor()
    # Use sigmoid calibration when enough data, fall back otherwise
    cal_method = "sigmoid" if len(X) >= 200 else None
    ml.train(X, y1, y2, sample_weights=sw, calibration_method=cal_method)
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
        X, y1, _ = load_training_data()
        if len(X) > 0:
            analyze_feature_importance(X, y1)

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


if __name__ == "__main__":
    main()
