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

    return np.array(f, dtype=np.float32)


def extract_features_from_game_record(r: dict) -> np.ndarray:
    """Build feature vector from a game/ historical record."""
    f = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    f[0] = 1.2  # home_form_pts (default)
    f[1] = 1.2  # away_form_pts
    f[2] = 0.0  # form_diff
    f[3] = 0.5  # home_pos_score
    f[4] = 0.5  # away_pos_score
    f[5] = 0.0  # pos_diff
    f[6] = 1.3  # home_gf_avg
    f[7] = 1.0  # home_ga_avg
    f[8] = 1.1  # away_gf_avg
    f[9] = 1.2  # away_ga_avg
    f[10] = 0.3  # home_gd_per_game
    f[11] = -0.1  # away_gd_per_game
    f[12] = 1.2  # exp_home_goals
    f[13] = 1.1  # exp_away_goals
    f[14] = 2.3  # exp_total
    f[15] = 0  # h2h_home_wins
    f[16] = 0  # h2h_draws
    f[17] = 0  # h2h_away_wins
    f[18] = 5  # h2h_total
    f[19] = 2.8  # league_avg_goals
    f[20] = 0.25  # league_draw_rate
    f[21] = 0.45  # league_home_win_rate
    f[22] = (r.get("prob_home") or 33) / 100.0
    f[23] = (r.get("prob_draw") or 33) / 100.0
    f[24] = (r.get("prob_away") or 33) / 100.0
    f[25] = 1 / ((r.get("prob_home") or 33) / 100.0) if (r.get("prob_home") or 33) > 0 else 2.5
    f[26] = 1 / ((r.get("prob_draw") or 33) / 100.0) if (r.get("prob_draw") or 33) > 0 else 3.2
    f[27] = 1 / ((r.get("prob_away") or 33) / 100.0) if (r.get("prob_away") or 33) > 0 else 3.0
    f[28] = 0.15  # volatility
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
    """ML-based predictor with RandomForest + GradientBoosting."""

    def __init__(self):
        self.rf_model_1x2: Optional[RandomForestClassifier] = None
        self.gb_model_1x2: Optional[GradientBoostingClassifier] = None
        self.rf_model_ou: Optional[RandomForestClassifier] = None
        self.gb_model_ou: Optional[GradientBoostingClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False
        self.training_examples = 0
        self.accuracy_1x2 = 0.0
        self.accuracy_ou = 0.0
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def train(self, X: np.ndarray, y_1x2: np.ndarray, y_ou: np.ndarray,
              sample_weights: Optional[np.ndarray] = None):
        """Train all models with optional sample weights (improvement 5: time decay)."""
        n = len(X)
        if n < 100:
            print(f"Warning: only {n} training examples, need at least 100")
            return

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        print(f"Training RandomForest for 1X2 ({n} examples)...")
        self.rf_model_1x2 = RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=10,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        self.rf_model_1x2.fit(X_scaled, y_1x2, sample_weight=sample_weights)

        print(f"Training GradientBoosting for 1X2 ({n} examples)...")
        self.gb_model_1x2 = GradientBoostingClassifier(
            n_estimators=150, max_depth=6, min_samples_leaf=10,
            learning_rate=0.1, subsample=0.8, random_state=42,
        )
        self.gb_model_1x2.fit(X_scaled, y_1x2, sample_weight=sample_weights)

        print(f"Training RandomForest for O/U ({n} examples)...")
        self.rf_model_ou = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=10,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        self.rf_model_ou.fit(X_scaled, y_ou, sample_weight=sample_weights)

        print(f"Training GradientBoosting for O/U ({n} examples)...")
        self.gb_model_ou = GradientBoostingClassifier(
            n_estimators=150, max_depth=5, min_samples_leaf=10,
            learning_rate=0.1, subsample=0.8, random_state=42,
        )
        self.gb_model_ou.fit(X_scaled, y_ou, sample_weight=sample_weights)

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
        """Return ensemble probabilities for [away, draw, home]."""
        X_scaled = self.scaler.transform(X)
        rf_proba = self.rf_model_1x2.predict_proba(X_scaled)
        gb_proba = self.gb_model_1x2.predict_proba(X_scaled)
        # Average ensemble
        return (rf_proba + gb_proba) / 2.0

    def predict_proba_ou(self, X: np.ndarray) -> np.ndarray:
        """Return ensemble probabilities for [under, over]."""
        X_scaled = self.scaler.transform(X)
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
        """Save individual model components to disk."""
        import joblib
        path = MODELS_DIR / "ml_predictor"
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path / "scaler.joblib")
        joblib.dump(self.rf_model_1x2, path / "rf_1x2.joblib")
        joblib.dump(self.gb_model_1x2, path / "gb_1x2.joblib")
        joblib.dump(self.rf_model_ou, path / "rf_ou.joblib")
        joblib.dump(self.gb_model_ou, path / "gb_ou.joblib")
        meta = {
            "is_trained": self.is_trained,
            "training_examples": self.training_examples,
            "accuracy_1x2": self.accuracy_1x2,
            "accuracy_ou": self.accuracy_ou,
            "cv_accuracy_1x2": getattr(self, 'cv_accuracy_1x2', 0.0),
            "cv_accuracy_ou": getattr(self, 'cv_accuracy_ou', 0.0),
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
        import joblib
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
        ml.scaler = joblib.load(path / "scaler.joblib")
        ml.rf_model_1x2 = joblib.load(path / "rf_1x2.joblib")
        ml.gb_model_1x2 = joblib.load(path / "gb_1x2.joblib")
        ml.rf_model_ou = joblib.load(path / "rf_ou.joblib")
        ml.gb_model_ou = joblib.load(path / "gb_ou.joblib")
        with open(meta_path) as f:
            meta = json.load(f)
        ml.is_trained = meta["is_trained"]
        ml.training_examples = meta["training_examples"]
        ml.accuracy_1x2 = meta.get("accuracy_1x2", 0.0)
        ml.accuracy_ou = meta.get("accuracy_ou", 0.0)
        ml.cv_accuracy_1x2 = meta.get("cv_accuracy_1x2", 0.0)
        ml.cv_accuracy_ou = meta.get("cv_accuracy_ou", 0.0)
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
    home_strength = (home_gf or 1.3) / league_avg if league_avg > 0 else 1.0
    home_defense = (home_ga or 1.0) / league_avg if league_avg > 0 else 1.0
    away_strength = (away_gf or 1.1) / league_avg if league_avg > 0 else 1.0
    away_defense = (away_ga or 1.2) / league_avg if league_avg > 0 else 1.0

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

def ensemble_predict(
    data: dict,
    profile: dict,
    ml_model: Optional[MLPredictor] = None,
    dynamic_weights: Optional[dict] = None,
) -> dict:
    """
    Combine Poisson, ML, and Forebet predictions into a single ensemble.

    Weighting strategy (improvement 3):
      - When dynamic_weights provided from DB tracking: use those
      - Otherwise: 40% ML, 35% Poisson, 25% Forebet (or fallback based on availability)
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
    if dynamic_weights:
        w_poisson = dynamic_weights.get("poisson", 0.35)
        w_ml = dynamic_weights.get("ml", 0.25)
        w_fb = dynamic_weights.get("forebet", 0.25)
        w_default = dynamic_weights.get("default", 0.15)
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
    elif ml_pred and has_forebet:
        w_ml, w_poisson, w_fb = 0.40, 0.35, 0.25
    elif ml_pred:
        w_ml, w_poisson, w_fb = 0.50, 0.50, 0.0
    elif has_forebet:
        w_ml, w_poisson, w_fb = 0.0, 0.60, 0.40
    else:
        w_ml, w_poisson, w_fb = 0.0, 1.0, 0.0

    # Blend probabilities
    p_h = poisson["prob_home"] * w_poisson
    p_d = poisson["prob_draw"] * w_poisson
    p_a = poisson["prob_away"] * w_poisson

    if ml_pred:
        p_h += ml_pred["ml_prob_home"] * w_ml
        p_d += ml_pred["ml_prob_draw"] * w_ml
        p_a += ml_pred["ml_prob_away"] * w_ml

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
        p_over += ml_pred["ml_prob_over"] * w_ml
        p_under += ml_pred["ml_prob_under"] * w_ml
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
        "method": f"ensemble(poisson={w_poisson:.0%}, ml={w_ml:.0%}, forebet={w_fb:.0%})",
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
    """
    X_list, y1_list, y2_list = [], [], []
    weight_list = []
    cutoff = datetime.now() - timedelta(days=365)

    # 1. Game dataset (primary)
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
                # Time decay weight
                if with_weights:
                    match_date = r.get("date", "")
                    w = _time_decay_weight(match_date, cutoff)
                    weight_list.append(w)
            except Exception:
                continue
    else:
        print(f"Game dataset not found at {GAME_DATA}")

    # 2. History.db (reviewed predictions)
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
                if with_weights:
                    w = _time_decay_weight(r.get("match_date", ""), cutoff)
                    weight_list.append(w)
            except Exception:
                continue

    X = np.array(X_list, dtype=np.float32)
    y1 = np.array(y1_list, dtype=np.int32)
    y2 = np.array(y2_list, dtype=np.int32)
    print(f"Total training examples: {len(X)}")
    
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
    """Train ML models from available data with time decay (improvement 5)."""
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
    ml.train(X, y1, y2, sample_weights=sw)
    ml.save()

    print(f"\nTraining complete: {ml.training_examples} examples, "
          f"1X2 in-sample acc={ml.accuracy_1x2:.3f}, CV acc={getattr(ml, 'cv_accuracy_1x2', 0):.3f}, "
          f"O/U in-sample acc={ml.accuracy_ou:.3f}, CV acc={getattr(ml, 'cv_accuracy_ou', 0):.3f}")
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
