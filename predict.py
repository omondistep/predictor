#!/usr/bin/env python3
"""
Football Match Predictor — v2 (Forebet-powered)

Modes:
  predict links.txt                    Scrape Forebet links → predict → save to DB
  predict --review                     Review past predictions vs actual results
  predict --calibrate                  Show calibration stats
  predict --odds <file> <flags>        Original odds-based mode (v1)

Scrapes deep match data from Forebet, applies rule-based analysis,
stores everything in SQLite, and supports post-match review + calibration.
"""

import argparse
import os
import sys
import re
import json
from datetime import datetime
from pathlib import Path

# Local modules
from database import (
    init_db, save_prediction, get_unreviewed_matches, update_result,
    get_calibration_summary, get_predictions_for_review, get_league_accuracy
)
from forebet_scraper import scrape_url, scrape_and_save, ForebetScraper

# ─────────────────────────────────────────────
# League Profiles
# ─────────────────────────────────────────────

LEAGUE_PROFILES = {
    "brazil-serie-a":      {"avg_goals": 3.01, "u25_rate": 0.4, "btts_no_rate": 0.49, "draw_rate": 0.21, "home_win_rate": 0.53, "home_adv": 1.15, "volatility": 0.05},
    "brazil-serie-b":      {"avg_goals": 2.1, "u25_rate": 0.58, "btts_no_rate": 0.55, "draw_rate": 0.30, "home_win_rate": 0.44, "home_adv": 1.15, "volatility": 0.05},
    "brazil-serie-c":      {"avg_goals": 2.18, "u25_rate": 0.45, "btts_no_rate": 0.45, "draw_rate": 0.36, "home_win_rate": 0.45, "home_adv": 1.20, "volatility": 0.10},
    "brazil-serie-d":      {"avg_goals": 2.2, "u25_rate": 0.72, "btts_no_rate": 0.68, "draw_rate": 0.2, "home_win_rate": 0.64, "home_adv": 1.25, "volatility": 0.15},
    "brazil-u20":          {"avg_goals": 4.0, "u25_rate": 0.2, "btts_no_rate": 0.4, "draw_rate": 0.4, "home_win_rate": 0.6, "home_adv": 1.10, "volatility": 0.30},
    "argentina-b-nacional": {"avg_goals": 2.81, "u25_rate": 0.52, "btts_no_rate": 0.48, "draw_rate": 0.22, "home_win_rate": 0.52, "home_adv": 1.15, "volatility": 0.10},
    "argentina-primera-b":  {"avg_goals": 2.36, "u25_rate": 0.55, "btts_no_rate": 0.64, "draw_rate": 0.18, "home_win_rate": 0.55, "home_adv": 1.15, "volatility": 0.10},
    "argentina-primera-c":  {"avg_goals": 1.71, "u25_rate": 0.64, "btts_no_rate": 0.71, "draw_rate": 0.29, "home_win_rate": 0.57, "home_adv": 1.15, "volatility": 0.15},
    "argentina-federal-a":  {"avg_goals": 1.55, "u25_rate": 0.91, "btts_no_rate": 0.91, "draw_rate": 0.18, "home_win_rate": 0.73, "home_adv": 1.20, "volatility": 0.15},
    "chile-primera":        {"avg_goals": 2.56, "u25_rate": 0.44, "btts_no_rate": 0.44, "draw_rate": 0.33, "home_win_rate": 0.33, "home_adv": 1.15, "volatility": 0.05},
    "chile-primera-b":      {"avg_goals": 2.25, "u25_rate": 0.62, "btts_no_rate": 0.5, "draw_rate": 0.25, "home_win_rate": 0.62, "home_adv": 1.15, "volatility": 0.10},
    "usl-championship":     {"avg_goals": 2.71, "u25_rate": 0.71, "btts_no_rate": 0.57, "draw_rate": 0.36, "home_win_rate": 0.43, "home_adv": 1.15, "volatility": 0.10},
    "usl-league-one":       {"avg_goals": 3.35, "u25_rate": 0.35, "btts_no_rate": 0.47, "draw_rate": 0.12, "home_win_rate": 0.65, "home_adv": 1.15, "volatility": 0.15},
    "usl-league-two":       {"avg_goals": 3.67, "u25_rate": 0.26, "btts_no_rate": 0.39, "draw_rate": 0.14, "home_win_rate": 0.47, "home_adv": 1.15, "volatility": 0.30},
    "mls-next-pro":         {"avg_goals": 3.1, "u25_rate": 0.38, "btts_no_rate": 0.38, "draw_rate": 0.20, "home_win_rate": 0.50, "home_adv": 1.10, "volatility": 0.25},
    "nwsl":                {"avg_goals": 2.4, "u25_rate": 0.50, "btts_no_rate": 0.48, "draw_rate": 0.25, "home_win_rate": 0.46, "home_adv": 1.10, "volatility": 0.10},
    "uruguay-primera":      {"avg_goals": 1.62, "u25_rate": 0.75, "btts_no_rate": 0.75, "draw_rate": 0.38, "home_win_rate": 0.38, "home_adv": 1.10, "volatility": 0.10},
    "uruguay-segunda":      {"avg_goals": 2.33, "u25_rate": 0.56, "btts_no_rate": 0.56, "draw_rate": 0.33, "home_win_rate": 0.33, "home_adv": 1.15, "volatility": 0.15},
    "ecuador-serie-a":      {"avg_goals": 2.2, "u25_rate": 0.6, "btts_no_rate": 0.4, "draw_rate": 0.2, "home_win_rate": 0.6, "home_adv": 1.25, "volatility": 0.10},
    "ecuador-serie-b":      {"avg_goals": 1.9, "u25_rate": 0.62, "btts_no_rate": 0.58, "draw_rate": 0.32, "home_win_rate": 0.42, "home_adv": 1.25, "volatility": 0.15},
    "peru-primera":         {"avg_goals": 2.3, "u25_rate": 0.52, "btts_no_rate": 0.50, "draw_rate": 0.28, "home_win_rate": 0.46, "home_adv": 1.30, "volatility": 0.10},
    "paraguay-primera":     {"avg_goals": 3.25, "u25_rate": 0.25, "btts_no_rate": 0.38, "draw_rate": 0.25, "home_win_rate": 0.25, "home_adv": 1.15, "volatility": 0.10},
    "paraguay-segunda":     {"avg_goals": 1.9, "u25_rate": 0.62, "btts_no_rate": 0.58, "draw_rate": 0.32, "home_win_rate": 0.42, "home_adv": 1.15, "volatility": 0.15},
    "spain-segunda":        {"avg_goals": 2.2, "u25_rate": 0.55, "btts_no_rate": 0.52, "draw_rate": 0.30, "home_win_rate": 0.44, "home_adv": 1.15, "volatility": 0.05},
    "austria-landesliga":   {"avg_goals": 4.33, "u25_rate": 0.11, "btts_no_rate": 0.27, "draw_rate": 0.09, "home_win_rate": 0.53, "home_adv": 1.15, "volatility": 0.25},
    "reserve-leagues":      {"avg_goals": 3.0, "u25_rate": 0.35, "btts_no_rate": 0.35, "draw_rate": 0.24, "home_win_rate": 0.42, "home_adv": 1.05, "volatility": 0.35},
    "sweden-allsvenskan":   {"avg_goals": 2.6, "u25_rate": 0.45, "btts_no_rate": 0.44, "draw_rate": 0.24, "home_win_rate": 0.48, "home_adv": 1.15, "volatility": 0.08},
    "sweden-superettan":    {"avg_goals": 3.6, "u25_rate": 0.2, "btts_no_rate": 0.2, "draw_rate": 0.4, "home_win_rate": 0.4, "home_adv": 1.15, "volatility": 0.12},
    "sweden-ettan":         {"avg_goals": 2.8, "u25_rate": 0.40, "btts_no_rate": 0.38, "draw_rate": 0.23, "home_win_rate": 0.47, "home_adv": 1.12, "volatility": 0.20},
    "sweden-division-2":    {"avg_goals": 3.62, "u25_rate": 0.38, "btts_no_rate": 0.38, "draw_rate": 0.12, "home_win_rate": 0.25, "home_adv": 1.10, "volatility": 0.25},
    "finland-veikkausliiga":{"avg_goals": 2.5, "u25_rate": 0.48, "btts_no_rate": 0.46, "draw_rate": 0.25, "home_win_rate": 0.47, "home_adv": 1.12, "volatility": 0.12},
    "finland-ykkonen":      {"avg_goals": 2.6, "u25_rate": 0.45, "btts_no_rate": 0.44, "draw_rate": 0.24, "home_win_rate": 0.46, "home_adv": 1.10, "volatility": 0.18},
    "finland-kakkonen":     {"avg_goals": 2.82, "u25_rate": 0.55, "btts_no_rate": 0.55, "draw_rate": 0.36, "home_win_rate": 0.18, "home_adv": 1.10, "volatility": 0.25},
    "morocco-botola":       {"avg_goals": 1.73, "u25_rate": 0.73, "btts_no_rate": 0.55, "draw_rate": 0.27, "home_win_rate": 0.64, "home_adv": 1.12, "volatility": 0.08},
    "iceland":              {"avg_goals": 2.4, "u25_rate": 0.55, "btts_no_rate": 0.50, "draw_rate": 0.28, "home_win_rate": 0.44, "home_adv": 1.10, "volatility": 0.15},
    "iceland-women":        {"avg_goals": 2.0, "u25_rate": 0.65, "btts_no_rate": 0.55, "draw_rate": 0.30, "home_win_rate": 0.40, "home_adv": 1.10, "volatility": 0.20},
    "estonia":              {"avg_goals": 2.2, "u25_rate": 0.60, "btts_no_rate": 0.55, "draw_rate": 0.30, "home_win_rate": 0.42, "home_adv": 1.10, "volatility": 0.20},
    "georgia":              {"avg_goals": 2.3, "u25_rate": 0.55, "btts_no_rate": 0.50, "draw_rate": 0.28, "home_win_rate": 0.46, "home_adv": 1.15, "volatility": 0.20},
    "lithuania":            {"avg_goals": 2.1, "u25_rate": 0.60, "btts_no_rate": 0.55, "draw_rate": 0.30, "home_win_rate": 0.42, "home_adv": 1.10, "volatility": 0.20},
    "women-football":       {"avg_goals": 3.52, "u25_rate": 0.39, "btts_no_rate": 0.43, "draw_rate": 0.17, "home_win_rate": 0.52, "home_adv": 1.05, "volatility": 0.20},
    "default":              {"avg_goals": 2.8, "u25_rate": 0.45, "btts_no_rate": 0.50, "draw_rate": 0.25, "home_win_rate": 0.45, "home_adv": 1.10, "volatility": 0.20},
}


def detect_league(text: str) -> str:
    """Detect league key from competition text."""
    t = text.lower()
    
    # Prefix matches (Forebet short codes)
    if t.startswith("br"):
        code = t[2:3].lower()
        if "u20" in t or "sub" in t: return "brazil-u20"
        if code in ("1", "a"): return "brazil-serie-a"
        if code in ("2", "b"): return "brazil-serie-b"
        if code in ("3", "c"): return "brazil-serie-c"
        if code in ("4", "d"): return "brazil-serie-d"
        return "brazil-serie-a"
    if t.startswith("ar"):
        if "res" in t: return "reserve-leagues"
        if "b nacional" in t or "2" in t: return "argentina-b-nacional"
        if "primera b" in t or "3" in t[:4]: return "argentina-primera-b"
        if "primera c" in t or "4" in t[:4]: return "argentina-primera-c"
        if "federal a" in t: return "argentina-federal-a"
        return "argentina-b-nacional"
    if t.startswith("es"):
        if "2" in t: return "spain-segunda"
        if "estonia" in t or "eesti" in t or "meistriliiga" in t: return "estonia"
        return "default"
    if t.startswith("at"):
        return "austria-landesliga"
    if t.startswith("cl"):
        if "2" in t or "b" in t: return "chile-primera-b"
        return "chile-primera"
    if t.startswith("uy"):
        if "2" in t: return "uruguay-segunda"
        return "uruguay-primera"
    if t.startswith("kr"):
        return "default"
    if t.startswith("se"):
        short = t[:4].lower()
        if "1" in short: return "sweden-allsvenskan"
        if "2" in short: return "sweden-superettan"
        if "3" in short: return "sweden-ettan"
        return "sweden-division-2"
    if t.startswith("fi"):
        short = t[:4].lower()
        if "1" in short: return "finland-veikkausliiga"
        if "2" in short: return "finland-ykkonen"
        return "finland-kakkonen"
    if t.startswith("ma"):
        return "morocco-botola"

    # Brazil
    if "brazil" in t or "brasil" in t:
        if "u20" in t or "sub-20" in t or "sub 20" in t:
            return "brazil-u20"
        if "serie d" in t or "série d" in t:
            return "brazil-serie-d"
        if "serie c" in t or "série c" in t:
            return "brazil-serie-c"
        if "serie b" in t or "série b" in t:
            return "brazil-serie-b"
        if "serie a" in t or "série a" in t or "brasileir" in t:
            return "brazil-serie-a"
    # Argentina
    if "argentina" in t:
        if "primera b nacional" in t or "b nacional" in t or "primera nacional" in t:
            return "argentina-b-nacional"
        if "primera b" in t:
            return "argentina-primera-b"
        if "primera c" in t:
            return "argentina-primera-c"
        if "federal a" in t:
            return "argentina-federal-a"
    # Chile
    if "chile" in t:
        if "primera b" in t or "torneo transicion" in t:
            return "chile-primera-b"
        if "primera" in t:
            return "chile-primera"
    # USA
    if "usa" in t or "usl" in t:
        if "championship" in t:
            return "usl-championship"
        if "league one" in t:
            return "usl-league-one"
        if "league two" in t:
            return "usl-league-two"
        if "mls next pro" in t or "mls" in t:
            return "mls-next-pro"
    if "nwsl" in t or "national women" in t:
        return "nwsl"
    # Austria
    if "austria" in t or "österreich" in t:
        if "landesliga" in t or "oberliga" in t or "regionalliga" in t:
            return "austria-landesliga"
    # Uruguay
    if "uruguay" in t:
        if "segunda" in t:
            return "uruguay-segunda"
        if "primera" in t:
            return "uruguay-primera"
    # Ecuador
    if "ecuador" in t:
        if "serie b" in t:
            return "ecuador-serie-b"
        if "serie a" in t:
            return "ecuador-serie-a"
    # Peru
    if "peru" in t:
        return "peru-primera"
    # Paraguay
    if "paraguay" in t:
        if "segunda" in t:
            return "paraguay-segunda"
        return "paraguay-primera"
    # Sweden
    if "sweden" in t or "sverige" in t or "suecia" in t:
        if "allsvenskan" in t:
            return "sweden-allsvenskan"
        if "superettan" in t:
            return "sweden-superettan"
        if "ettan" in t or "division 2" in t:
            return "sweden-ettan"
        return "sweden-division-2"
    # Finland
    if "finland" in t or "finland" in t or "suomi" in t:
        if "veikkausliiga" in t:
            return "finland-veikkausliiga"
        if "ykkonen" in t or "ykkönen" in t:
            return "finland-ykkonen"
        return "finland-kakkonen"
    # Morocco
    if "morocco" in t or "botola" in t or "maroc" in t:
        return "morocco-botola"
    # Spain
    if "spain" in t or "espana" in t or "espa" in t:
        if "segunda" in t:
            return "spain-segunda"
    # Women's football — lower scoring on average
    if (" w" in t or " women" in t or " wfc " in t or " wfc" in t
        or t.endswith(" w") or t.endswith(" women")
        or "(w)" in t or "/w " in t):
        if "iceland" in t or "island" in t: return "iceland-women"
        if "sweden" in t or "sverige" in t or "suecia" in t: return "sweden-allsvenskan"
        return "women-football"
    # Iceland
    if "iceland" in t or "island" in t:
        if " w" in t or " women" in t or "(w)" in t: return "iceland-women"
        return "iceland"
    # Estonia
    if "estonia" in t or "eesti" in t:
        return "estonia"
    # Georgia
    if "georgia" in t or "sakartvelo" in t:
        return "georgia"
    # Lithuania
    if "lithuania" in t or "lietuva" in t:
        return "lithuania"
    # General Reserve / Youth catch-all
    if "reserve" in t or "u21" in t or "u23" in t or "juniors" in t:
        return "reserve-leagues"
    return "default"


def get_profile(league_key: str) -> dict:
    return LEAGUE_PROFILES.get(league_key, LEAGUE_PROFILES["default"])


# ─────────────────────────────────────────────
# Analysis Engine
# ─────────────────────────────────────────────

CONF_RANK = {"Near Certain": 0, "High": 1, "Medium-High": 2, "Medium": 3, "Low": 4}
CONF_LABELS = ["Near Certain", "High", "Medium-High", "Medium", "Low"]


def _ppg(form_str: str) -> float:
    """Points per game from a form string like 'WDLDDL'."""
    pts = sum(3 if c == "W" else 1 if c == "D" else 0 for c in form_str if c in "WDL")
    n = sum(1 for c in form_str if c in "WDL")
    return pts / n if n >= 3 else 1.2


def _ord(n):
    """Ordinal suffix for a number."""
    if not n:
        return ""
    n = int(n)
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{'th' if n % 10 not in (1,2,3) else {1:'st',2:'nd',3:'rd'}[n%10]}"


def _val(v, default=1.2):
    return v if v is not None else default


def estimate_goals(data: dict, profile: dict) -> tuple:
    """Estimate expected home/away goals from available data."""
    hf, af = data.get("home_form", ""), data.get("away_form", "")
    h_f = _ppg(hf) if hf else None
    a_f = _ppg(af) if af else None

    hp, ap = data.get("home_pos"), data.get("away_pos")
    total_teams = max(hp or 20, ap or 20) + 5

    # Base expected goals from league average
    base = profile["avg_goals"] / 2.0
    h_adv = profile.get("home_adv", 1.15)
    a_adv = 2.0 - h_adv  # Balanced advantage

    exp_h = base * h_adv
    exp_a = base * a_adv

    # Adjust for form
    if h_f is not None:
        exp_h *= max(0.5, h_f / 1.2)
    if a_f is not None:
        exp_a *= max(0.5, a_f / 1.2)

    # Adjust for standings
    if hp and ap and total_teams:
        # Higher position → more goals scored, fewer conceded
        exp_h *= max(0.7, 1.0 + (total_teams - hp) / total_teams * 0.3)
        exp_a *= max(0.7, 1.0 + (total_teams - ap) / total_teams * 0.3)
        # Defensive adjustment: higher position → fewer conceded
        exp_a *= max(0.7, 1.0 - (total_teams - hp) / total_teams * 0.2)
        exp_h *= max(0.7, 1.0 - (total_teams - ap) / total_teams * 0.2)

    # Override with actual avg goals data if available
    h_gf = data.get("home_avg_goals_for")
    a_gf = data.get("away_avg_goals_for")
    h_ga = data.get("home_avg_goals_against")
    a_ga = data.get("away_avg_goals_against")
    if h_gf:
        exp_h = (exp_h + h_gf) / 2
    if a_gf:
        exp_a = (exp_a + a_gf) / 2
    if h_ga:
        exp_a = (exp_a + h_ga) / 2
    if a_ga:
        exp_h = (exp_h + a_ga) / 2

    # Volatility regression: higher volatility -> regress toward league mean
    vol = profile.get("volatility", 0.1)
    exp_h = exp_h * (1.0 - vol) + base * vol
    exp_a = exp_a * (1.0 - vol) + base * vol

    return max(0.1, exp_h), max(0.1, exp_a)


def poisson_prob(goals: float, k: int) -> float:
    """P(X = k) for Poisson(goals)."""
    import math
    return math.exp(-goals) * (goals ** k) / math.factorial(k)


def poisson_cdf(goals: float, k: int) -> float:
    """P(X <= k) for Poisson(goals)."""
    return sum(poisson_prob(goals, i) for i in range(k + 1))


def prob_exact_score(exp_h: float, exp_a: float, h: int, a: int) -> float:
    return poisson_prob(exp_h, h) * poisson_prob(exp_a, a)


def prob_home_win(exp_h: float, exp_a: float) -> float:
    """P(Home win) from independent Poissons."""
    total = 0.0
    # Sum over reasonable score range (0-7 goals)
    for h in range(8):
        for a in range(8):
            if h > a:
                total += poisson_prob(exp_h, h) * poisson_prob(exp_a, a)
    return total


def prob_draw(exp_h: float, exp_a: float) -> float:
    total = 0.0
    for s in range(8):
        total += poisson_prob(exp_h, s) * poisson_prob(exp_a, s)
    return total


def prob_away_win(exp_h: float, exp_a: float) -> float:
    return 1.0 - prob_home_win(exp_h, exp_a) - prob_draw(exp_h, exp_a)


def prob_over(exp_h: float, exp_a: float, threshold: float) -> float:
    """P(Total goals > threshold)."""
    total = exp_h + exp_a
    return 1.0 - poisson_cdf(total, int(threshold))


def prob_btts(exp_h: float, exp_a: float) -> float:
    """P(Both teams score)."""
    return (1.0 - poisson_prob(exp_h, 0)) * (1.0 - poisson_prob(exp_a, 0))


def pick_from_odds(odds: tuple, our_prob: float, label_h: str, label_a: str):
    """Pick the side with best odds value vs our probability."""
    o_h, o_a = odds
    if not o_h or not o_a:
        return "", "Low", 0

    implied_h = 1.0 / o_h
    implied_a = 1.0 / o_a
    value_h = (our_prob - implied_h) / implied_h if implied_h > 0 else 0
    value_a = ((1 - our_prob) - implied_a) / implied_a if implied_a > 0 else 0

    if value_h > value_a and value_h > 0.05:
        return label_h, _value_to_conf(value_h, o_h), value_h
    elif value_a > value_h and value_a > 0.05:
        return label_a, _value_to_conf(value_a, o_a), value_a
    return "", "Low", 0


def _value_to_conf(value: float, odds: float) -> str:
    if odds < 1.25:
        return "Near Certain"
    if odds < 1.50 or value > 0.30:
        return "High"
    if odds < 1.70 or value > 0.15:
        return "Medium-High"
    if value > 0.05:
        return "Medium"
    return "Low"


def conv_label(score: int) -> str:
    """Convert 0-100 conviction score to confidence label."""
    if score >= 85:
        return "Near Certain"
    if score >= 70:
        return "High"
    if score >= 55:
        return "Medium-High"
    if score >= 40:
        return "Medium"
    return "Low"


def analyze_from_data(data: dict) -> dict:
    """Analyze all markets, recommend highest-conviction pick."""
    profile = get_profile(detect_league(data.get("league", "")))
    reasoning = []
    candidates = []

    hf, af = data.get("home_form", ""), data.get("away_form", "")
    h_ppg = _ppg(hf) if hf else None
    a_ppg = _ppg(af) if af else None
    hp, ap = data.get("home_pos"), data.get("away_pos")
    hm_ = data.get("h2h_matches", 0)
    hw_ = data.get("h2h_home_wins", 0) if hm_ >= 3 else 0
    ha_ = data.get("h2h_away_wins", 0) if hm_ >= 3 else 0

    # Expected goals model
    exp_h, exp_a = estimate_goals(data, profile)
    exp_total = exp_h + exp_a

    # Estimated probabilities from Poisson model
    p_home = prob_home_win(exp_h, exp_a)
    p_draw = prob_draw(exp_h, exp_a)
    p_away = prob_away_win(exp_h, exp_a)

    # Draw inflation (Poisson often underestimates draws)
    # Use higher boost for low-scoring matches or high-draw leagues
    draw_rate = profile.get("draw_rate", 0.25)
    draw_boost = 0.07 if exp_total < 2.5 else 0.04
    if draw_rate >= 0.32:
        draw_boost += 0.04
    p_draw += draw_boost

    # Re-normalize
    total_p = p_home + p_draw + p_away
    p_home /= total_p
    p_draw /= total_p
    p_away /= total_p

    # ── Odds-based value infrastructure ──
    odds_h = data.get("odds_home")
    odds_d = data.get("odds_draw")
    odds_a = data.get("odds_away")
    odds_o25 = data.get("odds_over25")
    odds_u25 = data.get("odds_under25")
    odds_btts_y = data.get("odds_btts_yes")
    odds_btts_n = data.get("odds_btts_no")

    def _pick_odds(market: str, pick: str):
        if market == "1X2":
            return {"Home win": odds_h, "Draw": odds_d, "Away win": odds_a}.get(pick)
        if market == "DNB":
            return {"Home": odds_h, "Away": odds_a}.get(pick)
        if market == "DC":
            if pick == "1X" and odds_h and odds_d:
                return 1.0 / (1.0/odds_h + 1.0/odds_d)
            if pick == "X2" and odds_a and odds_d:
                return 1.0 / (1.0/odds_a + 1.0/odds_d)
            if pick == "12" and odds_h and odds_a:
                return 1.0 / (1.0/odds_h + 1.0/odds_a)
        if market == "O/U":
            return {"Over 0.5": None, "Under 0.5": None,
                    "Over 1.5": None, "Under 1.5": None,
                    "Over 2.5": odds_o25, "Under 2.5": odds_u25,
                    "Over 3.5": None, "Under 3.5": None}.get(pick)
        if market == "BTTS":
            return {"Yes": odds_btts_y, "No": odds_btts_n}.get(pick)
        return None

    # Minimum odds floor by confidence level — prevents value destruction
    ODDS_FLOORS = {"Near Certain": 1.10, "High": 1.18, "Medium-High": 1.28, "Medium": 1.50}

    def _value_adjust(conf: str, market: str, pick: str):
        """Return adjusted confidence or None to skip the pick based on odds value."""
        po = _pick_odds(market, pick)
        if po is None or po <= 1.0:
            return conf  # No market odds available — trust model
        implied = 1.0 / po
        # Cap confidence if odds are too low for that level
        for level, floor in sorted(ODDS_FLOORS.items(), key=lambda x: CONF_RANK[x[0]]):
            if CONF_RANK.get(conf, 99) <= CONF_RANK[level] and po < floor:
                conf = level
        # Skip Medium picks with odds below floor (no lower level to downgrade to)
        if conf == "Medium" and po < ODDS_FLOORS.get("Medium", 1.5):
            return None
        return conf

    def add(market: str, pick: str, conf: str, reason: str = "", model_prob: float = None):
        conf = _value_adjust(conf, market, pick)
        if conf is None:
            return
        rank = CONF_RANK.get(conf, 99)
        po = _pick_odds(market, pick)
        implied_prob = 1.0 / po if po and po > 1.0 else None
        value_ratio = model_prob / implied_prob if (model_prob and implied_prob) else None
        candidates.append({
            "market": market, "pick": pick, "confidence": conf,
            "rank": rank, "reason": reason,
            "model_prob": model_prob, "implied_prob": implied_prob,
            "value_ratio": value_ratio,
        })

    # ── 1X2 (model-driven from Poisson probabilities) ──
    probs = [("Home win", p_home), ("Draw", p_draw), ("Away win", p_away)]
    probs.sort(key=lambda x: x[1], reverse=True)
    top_pick, top_prob = probs[0]
    second_prob = probs[1][1]
    margin = top_prob - second_prob

    best_12 = ""
    best_12_conf = "Low"
    best_12_reason = ""

    # Draws are harder to predict — use tighter thresholds
    if top_pick == "Draw":
        if top_prob >= 0.36 and margin >= 0.04:
            best_12_conf = "Medium-High"
        elif top_prob >= 0.33:
            best_12_conf = "Medium"
    else:
        # Tightened thresholds for Home/Away win
        if top_prob >= 0.58:
            best_12_conf = "Near Certain"
        elif top_prob >= 0.50:
            best_12_conf = "High" if margin >= 0.10 else "Medium-High"
        elif top_prob >= 0.42:
            best_12_conf = "Medium-High" if margin >= 0.06 else "Medium"
        elif top_prob >= 0.38 and margin >= 0.04:
            best_12_conf = "Medium"

    # Volatility Capping: Reduce confidence for volatile leagues
    vol = profile.get("volatility", 0.1)
    if vol >= 0.25:
        # Strict cap at Medium-High for highly volatile leagues
        if best_12_conf in ("Near Certain", "High"):
            best_12_conf = "Medium-High"
    elif vol >= 0.15:
        if best_12_conf == "Near Certain":
            best_12_conf = "High"

    if best_12_conf != "Low":
        best_12 = top_pick
        parts = [f"model {p_home:.0%}/{p_draw:.0%}/{p_away:.0%}", f"exp {exp_h:.1f}-{exp_a:.1f}"]
        if hp and ap:
            parts.append(f"pos {_ord(hp)}-{_ord(ap)}")
        best_12_reason = " ".join(parts)
        add("1X2", top_pick, best_12_conf, best_12_reason,
            model_prob={"Home win": p_home, "Draw": p_draw, "Away win": p_away}.get(top_pick))

    # Also consider Draw as a secondary candidate if it's close but not top
    if top_pick != "Draw":
        if p_draw >= 0.32 and (top_prob - p_draw) <= 0.12:
            add("1X2", "Draw", "Medium", f"model {p_draw:.0%} (close to top)", model_prob=p_draw)

    # Always show all three 1X2 outcomes (home/draw/away) regardless of probability
    existing_12 = {(c['market'], c['pick']) for c in candidates if c['market'] == '1X2'}
    for outcome_name, outcome_prob in [("Home win", p_home), ("Draw", p_draw), ("Away win", p_away)]:
        if ("1X2", outcome_name) not in existing_12:
            if outcome_prob >= 0.50:
                cnf = "High"
            elif outcome_prob >= 0.38:
                cnf = "Medium-High"
            elif outcome_prob >= 0.30:
                cnf = "Medium"
            elif outcome_prob >= 0.20:
                cnf = "Low"
            else:
                cnf = "Low"
            candidates.append({
                "market": "1X2", "pick": outcome_name,
                "confidence": cnf, "rank": 99,
                "reason": "", "model_prob": outcome_prob,
                "implied_prob": None, "value_ratio": None,
                "_always_show": True,
            })

    # ── Draw No Bet (derived from 1X2) — stricter after calibration ──
    dnb_home_conf = "Low"
    dnb_away_conf = "Low"
    if p_home > p_away + 0.08:
        if top_prob >= 0.55 and best_12_conf in ("Near Certain", "High"):
            dnb_home_conf = best_12_conf
        elif top_prob >= 0.50:
            dnb_home_conf = "Medium-High"
        elif top_prob >= 0.46:
            dnb_home_conf = "Medium"
    elif p_away > p_home + 0.10:  # Away DNB needs bigger margin
        if top_prob >= 0.58 and best_12_conf in ("Near Certain", "High"):
            dnb_away_conf = best_12_conf
        elif top_prob >= 0.52:
            dnb_away_conf = "Medium-High"
        elif top_prob >= 0.48:
            dnb_away_conf = "Medium"

    # Volatility capping for DNB too
    if vol >= 0.25:
        if dnb_home_conf in ("Near Certain", "High"): dnb_home_conf = "Medium-High"
        if dnb_away_conf in ("Near Certain", "High"): dnb_away_conf = "Medium-High"
        if dnb_away_conf == "Medium-High": dnb_away_conf = "Medium"
    elif vol >= 0.15:
        if dnb_home_conf == "Near Certain": dnb_home_conf = "High"
        if dnb_away_conf == "Near Certain": dnb_away_conf = "High"
        if dnb_away_conf == "Medium-High": dnb_away_conf = "Medium"  # Away penalty

    dnb_denom_h = p_home + p_draw
    dnb_denom_a = p_away + p_draw
    if dnb_home_conf != "Low":
        add("DNB", "Home", dnb_home_conf, "derived from model",
            model_prob=(p_home / dnb_denom_h) if dnb_denom_h > 0 else None)
    if dnb_away_conf != "Low":
        add("DNB", "Away", dnb_away_conf, "derived from model",
            model_prob=(p_away / dnb_denom_a) if dnb_denom_a > 0 else None)

    # ── Double Chance (derived from 1X2) — tightened thresholds ──
    if p_home + p_draw > 0.72:
        add("DC", "1X", "Medium-High" if p_home + p_draw > 0.82 else "Medium", "derived from model",
            model_prob=p_home + p_draw)
    if p_away + p_draw > 0.72:
        add("DC", "X2", "Medium-High" if p_away + p_draw > 0.82 else "Medium", "derived from model",
            model_prob=p_away + p_draw)
    # '12' only in low-draw leagues with strong separation
    if p_home + p_away > 0.86 and p_draw < 0.22:
        add("DC", "12", "Medium-High" if p_home + p_away > 0.92 else "Medium", "derived from model",
            model_prob=p_home + p_away)

    # ── O/U Multi-threshold (model-driven) — 0.5 is too trivial to include ──
    for thresh, label_u, label_o in [(1.5, "Under 1.5", "Over 1.5"),
                                      (2.5, "Under 2.5", "Over 2.5"),
                                      (3.5, "Under 3.5", "Over 3.5")]:
        p_o = prob_over(exp_h, exp_a, thresh)
        p_u = 1.0 - p_o

        # Use deviation from 50% as signal
        value_o = p_o - 0.5
        value_u = p_u - 0.5

        if p_o > p_u and value_o > 0:
            ou_pick = label_o
            ou_val = value_o
        elif p_u > p_o and value_u > 0:
            ou_pick = label_u
            ou_val = value_u
        else:
            continue

        if ou_val > 0.45:
            ou_conf = "Near Certain"
        elif ou_val > 0.35:
            ou_conf = "High"
        elif ou_val > 0.18:
            ou_conf = "Medium-High"
        elif ou_val > 0.10:
            ou_conf = "Medium"
        else:
            ou_conf = "Low"

        # Volatility capping for O/U
        if vol >= 0.25 and ou_conf in ("Near Certain", "High"):
            ou_conf = "Medium-High"
        elif vol >= 0.15 and ou_conf == "Near Certain":
            ou_conf = "High"

        # Cap O/U 1.5 — market odds are typically very low, never Near Certain
        if thresh == 1.5 and ou_conf == "Near Certain":
            ou_conf = "High"

        # Calibration: O1.5 needs sufficient expected goals to be reliable
        if thresh == 1.5 and "Over" in ou_pick:
            if exp_total < 2.5:
                ou_conf = "Low"  # Not enough expected goals to trust
            elif exp_total < 3.0:
                ou_conf = "Medium" if CONF_RANK.get(ou_conf, 99) < CONF_RANK["Medium"] else ou_conf
            # League-based cap for high u25_rate (many under-2.5 matches)
            if profile.get("u25_rate", 0.5) > 0.55:
                if CONF_RANK.get(ou_conf, 99) < CONF_RANK["Medium"]:
                    ou_conf = "Medium"
            # Women's football: naturally lower scoring, cap at Medium
            if profile.get("avg_goals", 3) < 2.2:
                if CONF_RANK.get(ou_conf, 99) < CONF_RANK["Medium"]:
                    ou_conf = "Medium"

        if thresh == 2.5 and "Under" in ou_pick and profile["u25_rate"] > 0.65:
            ou_conf = "High" if profile["u25_rate"] > 0.75 else "Medium-High"
            if vol >= 0.25: ou_conf = "Medium-High"
            ou_val = max(ou_val, 0.3)

        if ou_conf != "Low":
            ou_reason = f"exp goals {exp_total:.1f} model {p_o:.0%}o/{p_u:.0%}u"
            add("O/U", ou_pick, ou_conf, ou_reason,
                model_prob=p_o if "Over" in ou_pick else p_u)

    # ── BTTS (model-driven) ──
    p_btss = prob_btts(exp_h, exp_a)
    p_btn = 1.0 - p_btss

    value_yes = p_btss - 0.5
    value_no = p_btn - 0.5

    if value_yes > 0.08 and value_yes >= value_no:
        btss_conf = conv_label(50 + int(value_yes * 80))
        if vol >= 0.25 and btss_conf in ("Near Certain", "High"): btss_conf = "Medium-High"
        elif vol >= 0.15 and btss_conf == "Near Certain": btss_conf = "High"
        add("BTTS", "Yes", btss_conf, f"model {p_btss:.0%}y/{p_btn:.0%}n", model_prob=p_btss)
    elif value_no > 0.06:
        btss_conf = conv_label(50 + int(value_no * 80))
        if vol >= 0.25 and btss_conf in ("Near Certain", "High"): btss_conf = "Medium-High"
        elif vol >= 0.15 and btss_conf == "Near Certain": btss_conf = "High"
        add("BTTS", "No", btss_conf, f"model {p_btss:.0%}y/{p_btn:.0%}n", model_prob=p_btn)

    # ── Rank candidates and pick primary ──
    candidates.sort(key=lambda c: (
        -(c.get("model_prob") or 0),
        c["rank"],
    ))

    non_show = [c for c in candidates if not c.get('_always_show')]
    primary = non_show[0] if non_show else candidates[0] if candidates else {"market": "1X2", "pick": "Draw", "confidence": "Low"}

    # ── Build reasoning ──
    for c in candidates[:6]:  # top 6
        line = f"{c['market']}: {c['pick']} ({c['confidence']})"
        if c.get("reason"):
            line += f" — {c['reason']}"
        reasoning.append(line)

    # ── Correct score estimate ──
    cs_h, cs_a = round(exp_h), round(exp_a)
    cs_h = max(0, min(cs_h, 5))
    cs_a = max(0, min(cs_a, 5))
    correct_score = f"{cs_h}-{cs_a}"

    # ── Build picks summary for display ──
    picks_summary = []
    for c in candidates:
        star = "★" if c == primary else " "
        picks_summary.append(f"{star}{c['market']}: {c['pick']} ({c['confidence']})")

    return {
        "pick": primary["pick"],
        "market": primary["market"],
        "confidence": primary["confidence"],
        "all_picks": candidates,
        "picks_summary": picks_summary,
        "score_lean": correct_score,
        "reasoning": reasoning,
        "supporting_markets": [],
        "_exp_goals": (exp_h, exp_a),
        "_volatility": vol,
    }


# ─────────────────────────────────────────────
# Prediction runner
# ─────────────────────────────────────────────

def log(msg, end="\n"):
    """Print progress to stderr so stdout stays clean for JSON."""
    print(msg, end=end, file=sys.stderr, flush=True)


def run_forebet_predictions(links_path: str, show_reasoning: bool = True,
                            high_only: bool = False, json_out: bool = False,
                            compare_forebet: bool = True):
    """Read Forebet links, scrape, analyze, store, and output predictions."""
    with open(links_path) as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if not urls:
        log("No URLs found in " + links_path)
        return

    # Filter to only /football/matches/ URLs (skip /previews/)
    match_urls = [u for u in urls if "/football/matches/" in u]
    skipped = len(urls) - len(match_urls)
    if skipped:
        log(f"[Skipped {skipped} preview/other URLs — only match pages used]")

    log(f"Processing {len(match_urls)} Forebet match links...\n")

    results = []
    for i, url in enumerate(match_urls, 1):
        log(f"[{i}/{len(match_urls)}]", end=" ")
        data = scrape_and_save(url)
        if not data.get("home_team"):
            log("  [Skipped — no data]")
            continue

        # Analyze
        pred = analyze_from_data(data)

        # Store in DB (map analysis keys to DB column names)
        db_data = {
            **data,
            "our_prediction": pred["pick"],
            "our_confidence": pred["confidence"],
            "our_score_lean": pred["score_lean"],
        }
        match_id = save_prediction(db_data)

        results.append({
            "url": url,
            "home": data.get("home_team", "?"),
            "away": data.get("away_team", "?"),
            "league": data.get("league", ""),
            "date": data.get("match_date", ""),
            **pred,
            "forebet": data.get("forebet_pred", ""),
            "forebet_pct": (data.get("forebet_home_pct"),
                           data.get("forebet_draw_pct"),
                           data.get("forebet_away_pct")),
            "forebet_over25_pct": data.get("forebet_over25_pct"),
            "forebet_btts_yes_pct": data.get("forebet_btts_yes_pct"),
            # Raw data for table display
            "home_form": data.get("home_form", ""),
            "away_form": data.get("away_form", ""),
            "home_pos": data.get("home_pos"),
            "away_pos": data.get("away_pos"),
            "odds_home": data.get("odds_home"),
            "odds_draw": data.get("odds_draw"),
            "odds_away": data.get("odds_away"),
            "odds_over25": data.get("odds_over25"),
            "odds_under25": data.get("odds_under25"),
            "odds_btts_yes": data.get("odds_btts_yes"),
            "odds_btts_no": data.get("odds_btts_no"),
            "h2h_matches": data.get("h2h_matches", 0),
            "h2h_home_wins": data.get("h2h_home_wins", 0),
            "h2h_draws": data.get("h2h_draws", 0),
            "h2h_away_wins": data.get("h2h_away_wins", 0),
            "match_id": match_id,
        })

    # ── Output ──
    if json_out:
        json.dump(results, indent=2, ensure_ascii=False, fp=sys.stdout)
        return

    # Filter by confidence
    if high_only:
        results = [r for r in results if r["confidence"] in ("Near Certain", "High")]

    # Print
    preds_made = 0
    for r in results:
        preds_made += 1
        print()

        W = 68
        C = W - 4  # content width between borders

        # Color setup
        conf = r['confidence']
        color = {"Near Certain": "\033[92m", "High": "\033[94m",
                 "Medium-High": "\033[93m", "Medium": "\033[93m", "Low": "\033[91m"}.get(conf, "")
        reset = "\033[0m"

        def box(content):
            return f"\033[38;5;244m│\033[0m {content:<{C}} \033[38;5;244m│\033[0m"
        def hline(char="─"):
            return f"\033[38;5;244m├{char * C}┤\033[0m"
        def top():
            return f"\033[38;5;244m┌{'─' * C}┐\033[0m"
        def bottom():
            return f"\033[38;5;244m└{'─' * C}┘\033[0m"

        # ── HEADER ──
        print(top())

        # Volatility badge
        vol_val = r.get('_volatility', 0)
        if vol_val >= 0.25:
            vol_tag = f" \033[91m⚡VOL\033[0m"
        elif vol_val >= 0.15:
            vol_tag = f" \033[93m⚡vol\033[0m"
        else:
            vol_tag = f" \033[92m---\033[0m"

        print(box(f" {r['home']} vs {r['away']} "))
        print(box(f" {r.get('league', '')}{vol_tag}  |  {r.get('date', '')} "))
        print(hline())

        # ── FORM + STANDINGS ──
        fmt_pos = lambda p: (f"{p}" if p else "—") + "th" if p else "—"
        hf = r.get('home_form', '')
        af = r.get('away_form', '')
        h_ppg = _ppg(hf) if sum(1 for c in hf if c in 'WDL') >= 3 else None
        a_ppg = _ppg(af) if sum(1 for c in af if c in 'WDL') >= 3 else None

        home_line = f"Home: pos {r.get('home_pos', '—')}  Form: {hf or '—'}"
        if h_ppg:
            home_line += f"  ({h_ppg:.1f} ppg)"
        away_line = f"Away: pos {r.get('away_pos', '—')}  Form: {af or '—'}"
        if a_ppg:
            away_line += f"  ({a_ppg:.1f} ppg)"
        print(box(home_line))
        print(box(away_line))
        print(hline())

        # ── PRIMARY PICK ──
        market_tag = f" ({r['market']})" if r['market'] else ""
        score_str = f"  Score: {r['score_lean']}" if r['score_lean'] else ""
        exp_str = ""
        if '_exp_goals' in r and r['_exp_goals']:
            eh, ea = r['_exp_goals']
            exp_str = f"  Exp: {eh:.1f}-{ea:.1f}"
        pick_line = f"{color}★ {r['pick']}{market_tag}{reset}  | {conf}{score_str}{exp_str}"
        print(box(pick_line))

        # ── ODDS ──
        nl = lambda v: f"{v:.2f}" if isinstance(v, (int, float)) else str(v) if v else "—"
        odds_12 = f"{nl(r.get('odds_home'))}/{nl(r.get('odds_draw'))}/{nl(r.get('odds_away'))}"
        odds_ou = f"O/U 2.5: {nl(r.get('odds_over25'))}/{nl(r.get('odds_under25'))}"
        odds_bt = f"BTTS: {nl(r.get('odds_btts_yes'))}/{nl(r.get('odds_btts_no'))}"
        print(box(f"1X2: {odds_12}  |  {odds_ou}  |  {odds_bt}"))

        # ── H2H ──
        hm = r.get('h2h_matches', 0)
        if hm >= 3:
            hw = r.get('h2h_home_wins', 0)
            hd = r.get('h2h_draws', 0)
            ha = r.get('h2h_away_wins', 0)
            print(box(f"H2H: {hw}W-{hd}D-{ha}L  ({hm} matches)"))
        print(hline())

        # ── MODEL vs FOREBET (ranked by probability) ──
        all_picks = r.get('all_picks') or []
        if all_picks:
            fb_pred = r.get('forebet')
            fb_pcts = r.get('forebet_pct', (None, None, None))
            fb_o25 = r.get('forebet_over25_pct')
            fb_btts = r.get('forebet_btts_yes_pct')
            green = "\033[92m"
            reset = "\033[0m"

            oh = r.get('odds_home')
            od = r.get('odds_draw')
            oa = r.get('odds_away')
            oo25 = r.get('odds_over25')
            ou25 = r.get('odds_under25')
            ob_y = r.get('odds_btts_yes')
            ob_n = r.get('odds_btts_no')

            def _implied(odds_dict):
                """Return (pick_name, pct) from odds implied probabilities."""
                imp = {}
                total = 0.0
                for name, odd_val in odds_dict.items():
                    if odd_val and odd_val > 1.0:
                        imp[name] = 1.0 / odd_val
                        total += imp[name]
                if not imp or total <= 0:
                    return None, None
                for k in imp:
                    imp[k] = imp[k] / total
                best = max(imp, key=imp.get)
                return best, imp[best] * 100

            def _fb_for(market: str):
                if market == "1X2" and fb_pred and fb_pcts[0] is not None:
                    pick = {"1": "Home win", "X": "Draw", "2": "Away win"}.get(fb_pred, fb_pred)
                    pct = {"1": fb_pcts[0], "X": fb_pcts[1], "2": fb_pcts[2]}.get(fb_pred, 0)
                    return pick, pct
                if market == "O/U" and fb_o25 is not None:
                    pick = "Over" if fb_o25 > 50 else "Under"
                    pct = fb_o25 if fb_o25 > 50 else 100 - fb_o25
                    return pick, pct
                if market == "BTTS" and fb_btts is not None:
                    pick = "Yes" if fb_btts > 50 else "No"
                    pct = fb_btts if fb_btts > 50 else 100 - fb_btts
                    return pick, pct
                # Fallback to odds-implied probabilities
                if market == "1X2":
                    return _implied({"Home win": oh, "Draw": od, "Away win": oa})
                if market == "O/U":
                    return _implied({"Over 2.5": oo25, "Under 2.5": ou25})
                if market == "BTTS":
                    return _implied({"Yes": ob_y, "No": ob_n})
                return None, None

            def _short(pick: str, n: int = 11) -> str:
                """Truncate long pick names."""
                return pick[:n] if len(pick) > n else pick

            print(box(" Model Pick       Prob Cnf │ Forebet Pick      Prob"))
            print(box("─────────────────────────────┼─────────────────────"))
            for idx, p in enumerate(all_picks):
                if p.get('_always_show'):
                    star = " "
                else:
                    star = "★" if idx == 0 else " "
                pm = p['market']
                pp = _short(p['pick'], 11)
                mp = p.get('model_prob')
                mp_str = f"{mp:.0%}" if mp else ""
                pc = {'Near Certain': 'NC', 'High': 'Hi', 'Medium-High': 'MH', 'Medium': 'Me', 'Low': 'Lo'}.get(p['confidence'], '')
                vr = p.get('value_ratio')
                vr_str = f" ({vr:.2f})" if vr else ""
                left = f"{star} {pm:5s} {pp:11s} {mp_str:>4s} {pc:2s}{vr_str}"

                # Forebet column
                fb_pick, fb_pct = _fb_for(pm)
                if fb_pick:
                    if pm == "O/U":
                        agree = pp.split()[0] == fb_pick
                    else:
                        agree = pp == fb_pick
                    right = f"{_short(fb_pick, 18):18s} {fb_pct:3.0f}%"
                    if agree:
                        right = f"{green}{right} ✓{reset}"
                        left = f"{green}{left}{reset}"
                else:
                    right = "—"
                    agree = False

                print(box(f" {left:29s} │ {right}"))

        # ── REASONING ──
        if show_reasoning and r.get('reasoning'):
            for reason in r['reasoning'][:4]:
                print(box(f" {reason}"))

        # ── FOOTER ──
        short_url = r['url'][:45] + "..." if len(r['url']) > 48 else r['url']
        tag = f"\033[38;5;245mID: {r['match_id']}  |  {short_url}\033[0m"
        print(box(tag))
        print(bottom())

    # Summary
    print()
    Ws = 68
    C = Ws - 4
    print(f"\033[38;5;244m┌{'─' * C}┐\033[0m")
    conf_counts = {}
    for r in results:
        conf_counts[r['confidence']] = conf_counts.get(r['confidence'], 0) + 1
    print(f"\033[38;5;244m│\033[0m Predictions made: {preds_made} ({(len(results)/len(match_urls)*100) if preds_made else 0:.0f}% pick rate)")
    for c in ["Near Certain", "High", "Medium-High", "Medium", "Low"]:
        if c in conf_counts:
            count = conf_counts[c]
            suffix = "match" if count == 1 else "matches"
            print(f"\033[38;5;244m│\033[0m   {c}: {count} {suffix}")
    print(f"\033[38;5;244m│\033[0m")
    if compare_forebet:
        agreements = 0
        total_fb = 0
        for r in results:
            if r.get('forebet'):
                picks_12 = [p for p in (r.get('all_picks') or []) if p['market'] == '1X2']
                our_12 = picks_12[0]['pick'] if picks_12 else r['pick']
                fb_val = {"Home win": "1", "Draw": "X", "Away win": "2"}.get(our_12, "")
                if fb_val and r['forebet'] == fb_val:
                    agreements += 1
                total_fb += 1
        if total_fb:
            pct = 100 * agreements // total_fb
            agree_str = f"✓ {agreements}/{total_fb} ({pct}%)" if pct >= 50 else f"✗ {agreements}/{total_fb} ({pct}%)"
            print(f"\033[38;5;244m│\033[0m Forebet 1X2 agreement: {agree_str}")
    print(f"\033[38;5;244m└{'─' * C}┘\033[0m")
    print(f"\nSaved to database: history.db")


# ─────────────────────────────────────────────
# Review mode
# ─────────────────────────────────────────────

def _extract_result_from_forebet(soup) -> tuple | None:
    """Try to extract final score from Forebet page. Returns (h, a) or None."""
    if not soup:
        return None
    # Check h1 or title for score pattern
    h1 = soup.find("h1")
    if h1:
        m = re.search(r"(\d+)\s*[-–:]\s*(\d+)", h1.get_text())
        if m:
            return (int(m.group(1)), int(m.group(2)))
    # Check stat-content tables for recent result
    tables = soup.find_all("table", {"class": "stat-content"})
    for table in tables:
        rows = table.find_all("tr")
        for row in rows[:5]:
            cells = row.find_all("td")
            if len(cells) >= 3:
                m = re.search(r"(\d+)\s*-\s*(\d+)", cells[-1].get_text())
                if m:
                    return (int(m.group(1)), int(m.group(2)))
    return None


def run_review(mark_reviewed: bool = False):
    """Review past predictions and record actual results."""
    init_db()
    pending = get_unreviewed_matches(limit=100)

    if not pending:
        print("No unreviewed matches found.")
        return

    print(f"Found {len(pending)} unreviewed matches.\n")

    for m in pending:
        print(f"ID {m['id']}: {m['home_team']} vs {m['away_team']} ({m['match_date']})")
        if mark_reviewed and m.get('forebet_url'):
            scraper = ForebetScraper(m['forebet_url'])
            if scraper.fetch():
                score = _extract_result_from_forebet(scraper.soup)
                if score:
                    update_result(m['id'], score[0], score[1])
                    print(f"  ✓ Auto: {score[0]}-{score[1]}")
                else:
                    print(f"  - No score on page yet")
                    continue
            else:
                print(f"  - Could not fetch")
        else:
            print(f"  URL: {m.get('forebet_url', 'N/A')}")
            try:
                resp = input("  Score (e.g. 2-1) or Enter to skip: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if resp and re.match(r"\d+-\d+", resp):
                parts = resp.split("-")
                update_result(m['id'], int(parts[0]), int(parts[1]))
                print(f"  ✓ Result recorded: {resp}")
            else:
                print("  — Skipped")


def run_learn(url: str):
    """Scrape a results list page and update local database."""
    from forebet_scraper import scrape_results_list
    log(f"Learning from results: {url}")
    
    results = scrape_results_list(url)
    if not results:
        log("No results found on page.")
        return
        
    log(f"Found {len(results)} scores on page. Matching against history.db...")
    
    init_db()
    pending = get_unreviewed_matches(limit=1000)
    pending_map = {m['forebet_url']: m['id'] for m in pending if m.get('forebet_url')}
    
    log(f"Debug: pending_map has {len(pending_map)} entries")
    if results:
        log(f"Debug: first result url: {results[0]['url']}")
    if pending_map:
        log(f"Debug: first pending url: {list(pending_map.keys())[0]}")

    updated = 0
    for res in results:
        match_url = res['url']
        # Try exact match or match without query params
        match_id = pending_map.get(match_url) or pending_map.get(match_url.split('?')[0])
        
        if match_id:
            update_result(match_id, res['home_goals'], res['away_goals'])
            updated += 1
            
    log(f"Successfully updated {updated} match results.")
    if updated > 0:
        print("\n" + "="*55)
        print("NEW CALIBRATION INSIGHTS")
        run_calibration()


# ─────────────────────────────────────────────
# Calibration mode
# ─────────────────────────────────────────────

def run_calibration():
    """Show accuracy stats and calibration data."""
    init_db()
    stats = get_calibration_summary()

    if stats["total"] == 0:
        print("No calibration data yet. Review predictions first with --review.")
        return

    print("=" * 55)
    print("MODEL CALIBRATION REPORT")
    print("=" * 55)
    print(f"\nTotal reviewed: {stats['total']}")
    print(f"Our accuracy:    {stats['our_correct']}/{stats['total']} ({stats['our_pct']}%)")
    print(f"Forebet acc:     {stats['fb_correct']}/{stats['total']} ({stats['fb_pct']}%)")

    if stats["by_confidence"]:
        print(f"\n{'='*55}")
        print("ACCURACY BY CONFIDENCE LEVEL")
        print(f"{'Confidence':<20} {'Total':<8} {'Correct':<8} {'Rate':<8}")
        print("-" * 44)
        for row in stats["by_confidence"]:
            print(f"{row['confidence']:<20} {row['total']:<8} {row['correct']:<8} {row['pct']}%")

    if stats["by_league"]:
        print(f"\n{'='*55}")
        print("ACCURACY BY LEAGUE")
        print(f"{'League':<30} {'Vol':<6} {'Total':<8} {'Correct':<8} {'Rate':<8}")
        print("-" * 60)
        for row in stats["by_league"]:
            league_key = detect_league(row["league"])
            profile = get_profile(league_key)
            vol = profile.get("volatility", 0.1)
            if vol >= 0.25:
                vol_str = f"\033[91m{vol:.2f}\033[0m"
            elif vol >= 0.15:
                vol_str = f"\033[93m{vol:.2f}\033[0m"
            else:
                vol_str = f"\033[92m{vol:.2f}\033[0m"
            # Strip ANSI for width calculation
            plain_vol = f"{vol:.2f}"
            padding = 6 - len(plain_vol)
            print(f"{row['league'][:28]:<30} {vol_str}{' '*padding} {row['total']:<8} {row['our_correct']:<8} {row['our_pct']}%")

    # Suggest profile adjustments
    print(f"\n{'='*55}")
    print("CALIBRATION SUGGESTIONS (LEARNING)")
    for row in stats["by_league"]:
        # Get actual avg goals vs predicted if possible
        league = row["league"]
        total = row["total"]
        acc = row["our_pct"]
        
        if total >= 3:
            if acc < 45:
                print(f"  ⚠ {league}: Low accuracy ({acc}%). Try increasing 'volatility' or 'draw_boost'.")
            elif acc > 75:
                print(f"  ✓ {league}: High accuracy ({acc}%). Profile is well-calibrated.")
            
            # Note: A more advanced version would query goals from DB here

    print(f"\n{'='*55}")
    print("ACTIVE FILTERS (from calibration)")
    print(f"  Min odds: Near Certain ≥ 1.10, High ≥ 1.18, Medium-High ≥ 1.28, Medium ≥ 1.50")
    print(f"  1X2: Near Certain ≥ 58% (≥60% margin≥10%), High ≥ 50% (margin≥10%), MH ≥ 42% (margin≥6%), Medium ≥ 38% (margin≥4%)")
    print(f"  Draw: MH ≥ 36% (margin≥4%), Medium ≥ 33%")
    print(f"  DNB: Home margin ≥ 8% (Medium ≥ 46%), Away margin ≥ 10% (Medium ≥ 48%). Away penalized in volatile leagues.")
    print(f"  DC:  Threshold at 72% combined prob (Medium); MH ≥ 82%")
    print(f"  O/U: Near Certain needs 45% deviation, High ≥ 35%, MH ≥ 18%, Medium ≥ 10%")
    print(f"  BTTS: Yes requires value > 8% (>58% prob), No requires value > 6% (>56% prob)")
    print(f"  All picks filtered through odds-based value check at recommendation time")
    print()


# ─────────────────────────────────────────────
# Legacy odds-based mode
# ─────────────────────────────────────────────

def ensure_alias():
    """Create symlinks in ~/.local/bin/ for easy access."""
    bindir = Path.home() / ".local" / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve()
    for name in ("predict", "predictor"):
        bin_path = bindir / name
        if bin_path.exists() and bin_path.samefile(src):
            continue
        try:
            if bin_path.exists() or bin_path.is_symlink():
                bin_path.unlink()
            bin_path.symlink_to(src)
            print(f"Alias created: {bin_path} -> {src}")
        except Exception as e:
            print(f"Warning: could not create alias for {name}: {e}")


# ─────────────────────────────────────────────
# Main CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Football Match Predictor v2 — Forebet-powered analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  predict.py links.txt              Scrape Forebet links → predict → save to DB
  predict.py --review               Review past predictions vs actual results
  predict.py --learn <url>           Automated learning from results page
  predict.py --calibrate             Show calibration/accuracy stats

Options:
  --high-only    Show only High / Near Certain predictions
  --json         JSON output for scripting
  --no-compare   Skip Forebet comparison display
  --no-reasoning Hide reasoning
        """
    )
    parser.add_argument("file", nargs="?", help="File with Forebet URLs")
    parser.add_argument("--review", action="store_true", help="Review past predictions")
    parser.add_argument("--auto", action="store_true", help="Auto-review by re-scraping")
    parser.add_argument("--learn", help="URL of Forebet results page to learn from")
    parser.add_argument("--calibrate", action="store_true", help="Show calibration stats")
    parser.add_argument("--high-only", action="store_true", help="Show only confident picks")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--no-compare", action="store_true", help="Skip Forebet comparison")
    parser.add_argument("--no-reasoning", action="store_true", help="Hide reasoning")

    args = parser.parse_args()

    ensure_alias()
    init_db()

    if args.learn:
        run_learn(args.learn)
        return

    if args.calibrate:
        run_calibration()
        return

    if args.review:
        run_review(mark_reviewed=args.auto)
        return

    if args.file:
        # Detect if argument is a URL or a file path
        if args.file.startswith("http://") or args.file.startswith("https://"):
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
                f.write(args.file + "\n")
                tmp_path = f.name
            run_forebet_predictions(
                tmp_path,
                show_reasoning=not args.no_reasoning,
                high_only=args.high_only,
                json_out=args.json,
                compare_forebet=not args.no_compare,
            )
            os.unlink(tmp_path)
        else:
            run_forebet_predictions(
                args.file,
                show_reasoning=not args.no_reasoning,
                high_only=args.high_only,
                json_out=args.json,
                compare_forebet=not args.no_compare,
            )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
