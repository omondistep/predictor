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
    get_db, init_db, save_prediction, get_unreviewed_matches, update_result,
    get_calibration_summary, get_predictions_for_review, get_league_accuracy,
    store_market_results, get_market_accuracy, get_market_accuracy_history
)
from calibration_learner import retrain_from_results, apply_calibration
from forebet_scraper import scrape_url, scrape_and_save, ForebetScraper

# ML-enhanced modules (optional)
_ML_MODEL = None
_DYNAMIC_WEIGHTS = None  # Cached per-league dynamic weights

# Calibration learning module
_CALIBRATION_LEARNER = None
_BIAS_CORRECTIONS_LOADED = False


def schedule_retrain(delay_hours: float = 18.0):
    """Schedule isotonic regression retrain via cron.

    Args:
        delay_hours: Hours to wait before retraining (default 18h)
    """
    import subprocess
    from datetime import timedelta

    retrain_time = datetime.now() + timedelta(hours=delay_hours)
    hour = retrain_time.hour
    minute = retrain_time.minute
    parent = Path(__file__).parent

    cron_line = (
        f"{minute} {hour} * * * cd {parent} && "
        f".venv/bin/python3 -c "
        f"\"from calibration_learner import retrain_from_results; retrain_from_results(force=True)\" "
        f">> /tmp/retrain.log 2>&1"
    )

    try:
        result = subprocess.run("crontab -l 2>/dev/null", shell=True,
                                capture_output=True, text=True, timeout=10)
        existing = result.stdout if result.returncode == 0 else ""

        lines = [l for l in existing.splitlines()
                 if "retrain_from_results" not in l]
        lines.append(cron_line)
        new_crontab = "\n".join(lines) + "\n"

        proc = subprocess.run("crontab -", shell=True, input=new_crontab,
                              capture_output=True, text=True, timeout=10)
        if proc.returncode == 0:
            log(f"Retrain cron scheduled for {retrain_time.strftime('%H:%M %Y-%m-%d')}")
        else:
            log(f"Failed to set cron: {proc.stderr}")
    except Exception as e:
        log(f"Failed to schedule retrain: {e}")
        log("Running retrain immediately instead...")
        retrain_from_results(force=True)

# ─────────────────────────────────────────────
# League Profiles
# ─────────────────────────────────────────────

LEAGUE_PROFILES = {
    "brazil-serie-a":      {"avg_goals": 2.47, "u25_rate": 0.53, "btts_no_rate": 0.58, "draw_rate": 0.21, "home_win_rate": 0.52, "home_adv": 1.15, "volatility": 0.05},
    "brazil-serie-b":      {"avg_goals": 2.61, "u25_rate": 0.54, "btts_no_rate": 0.5, "draw_rate": 0.27, "home_win_rate": 0.43, "home_adv": 1.15, "volatility": 0.05},
    "brazil-serie-c":      {"avg_goals": 2.27, "u25_rate": 0.61, "btts_no_rate": 0.57, "draw_rate": 0.31, "home_win_rate": 0.42, "home_adv": 1.20, "volatility": 0.10},
    "brazil-serie-d":      {"avg_goals": 2.49, "u25_rate": 0.54, "btts_no_rate": 0.56, "draw_rate": 0.26, "home_win_rate": 0.45, "home_adv": 1.25, "volatility": 0.15},
    "brazil-u20":          {"avg_goals": 2.9, "u25_rate": 0.42, "btts_no_rate": 0.48, "draw_rate": 0.21, "home_win_rate": 0.47, "home_adv": 1.10, "volatility": 0.30},
    "argentina-b-nacional": {"avg_goals": 2.26, "u25_rate": 0.63, "btts_no_rate": 0.46, "draw_rate": 0.29, "home_win_rate": 0.49, "home_adv": 1.15, "volatility": 0.10},
    "argentina-primera-b":  {"avg_goals": 2.32, "u25_rate": 0.45, "btts_no_rate": 0.36, "draw_rate": 0.41, "home_win_rate": 0.27, "home_adv": 1.15, "volatility": 0.10},
    "argentina-primera-c":  {"avg_goals": 1.7, "u25_rate": 0.77, "btts_no_rate": 0.73, "draw_rate": 0.33, "home_win_rate": 0.3, "home_adv": 1.15, "volatility": 0.15},
    "argentina-federal-a":  {"avg_goals": 1.88, "u25_rate": 0.64, "btts_no_rate": 0.7, "draw_rate": 0.33, "home_win_rate": 0.48, "home_adv": 1.20, "volatility": 0.15},
    "chile-primera":        {"avg_goals": 2.77, "u25_rate": 0.47, "btts_no_rate": 0.51, "draw_rate": 0.22, "home_win_rate": 0.48, "home_adv": 1.15, "volatility": 0.05},
    "chile-primera-b":      {"avg_goals": 2.79, "u25_rate": 0.53, "btts_no_rate": 0.41, "draw_rate": 0.32, "home_win_rate": 0.46, "home_adv": 1.15, "volatility": 0.10},
    "usl-championship":     {"avg_goals": 2.71, "u25_rate": 0.43, "btts_no_rate": 0.29, "draw_rate": 0.36, "home_win_rate": 0.36, "home_adv": 1.15, "volatility": 0.10},
    "usl-league-one":       {"avg_goals": 2.21, "u25_rate": 0.5, "btts_no_rate": 0.57, "draw_rate": 0.21, "home_win_rate": 0.43, "home_adv": 1.15, "volatility": 0.15},
    "usl-league-two":       {"avg_goals": 3.88, "u25_rate": 0.27, "btts_no_rate": 0.38, "draw_rate": 0.16, "home_win_rate": 0.45, "home_adv": 1.15, "volatility": 0.30},
    "mls-next-pro":         {"avg_goals": 3.1, "u25_rate": 0.38, "btts_no_rate": 0.38, "draw_rate": 0.20, "home_win_rate": 0.50, "home_adv": 1.10, "volatility": 0.25},
    "nwsl":                {"avg_goals": 2.4, "u25_rate": 0.50, "btts_no_rate": 0.48, "draw_rate": 0.25, "home_win_rate": 0.46, "home_adv": 1.10, "volatility": 0.10},
    "uruguay-primera":      {"avg_goals": 1.86, "u25_rate": 0.71, "btts_no_rate": 0.43, "draw_rate": 0.57, "home_win_rate": 0.14, "home_adv": 1.10, "volatility": 0.10},
    "uruguay-segunda":      {"avg_goals": 2.15, "u25_rate": 0.69, "btts_no_rate": 0.46, "draw_rate": 0.31, "home_win_rate": 0.38, "home_adv": 1.15, "volatility": 0.15},
    "ecuador-serie-a":      {"avg_goals": 2.31, "u25_rate": 0.56, "btts_no_rate": 0.62, "draw_rate": 0.19, "home_win_rate": 0.44, "home_adv": 1.25, "volatility": 0.10},
    "ecuador-serie-b":      {"avg_goals": 1.58, "u25_rate": 0.75, "btts_no_rate": 0.75, "draw_rate": 0.33, "home_win_rate": 0.42, "home_adv": 1.25, "volatility": 0.15},
    "peru-primera":         {"avg_goals": 2.44, "u25_rate": 0.53, "btts_no_rate": 0.49, "draw_rate": 0.28, "home_win_rate": 0.49, "home_adv": 1.30, "volatility": 0.10},
    "paraguay-primera":     {"avg_goals": 2.2, "u25_rate": 0.69, "btts_no_rate": 0.51, "draw_rate": 0.36, "home_win_rate": 0.29, "home_adv": 1.15, "volatility": 0.10},
    "paraguay-segunda":     {"avg_goals": 1.9, "u25_rate": 0.62, "btts_no_rate": 0.58, "draw_rate": 0.32, "home_win_rate": 0.42, "home_adv": 1.15, "volatility": 0.15},
    "spain-segunda":        {"avg_goals": 2.6, "u25_rate": 0.5, "btts_no_rate": 0.47, "draw_rate": 0.26, "home_win_rate": 0.45, "home_adv": 1.15, "volatility": 0.05},
    "austria-landesliga":   {"avg_goals": 3.37, "u25_rate": 0.35, "btts_no_rate": 0.42, "draw_rate": 0.19, "home_win_rate": 0.47, "home_adv": 1.15, "volatility": 0.25},
    "reserve-leagues":      {"avg_goals": 3.0, "u25_rate": 0.35, "btts_no_rate": 0.35, "draw_rate": 0.24, "home_win_rate": 0.42, "home_adv": 1.05, "volatility": 0.35},
    "sweden-allsvenskan":   {"avg_goals": 3.47, "u25_rate": 0.33, "btts_no_rate": 0.4, "draw_rate": 0.13, "home_win_rate": 0.4, "home_adv": 1.15, "volatility": 0.08},
    "sweden-superettan":    {"avg_goals": 3.05, "u25_rate": 0.41, "btts_no_rate": 0.39, "draw_rate": 0.25, "home_win_rate": 0.42, "home_adv": 1.15, "volatility": 0.12},
    "sweden-ettan":         {"avg_goals": 2.86, "u25_rate": 0.43, "btts_no_rate": 0.49, "draw_rate": 0.19, "home_win_rate": 0.41, "home_adv": 1.12, "volatility": 0.20},
    "sweden-division-2":    {"avg_goals": 2.94, "u25_rate": 0.45, "btts_no_rate": 0.47, "draw_rate": 0.27, "home_win_rate": 0.41, "home_adv": 1.10, "volatility": 0.25},
    "finland-veikkausliiga":{"avg_goals": 2.5, "u25_rate": 0.48, "btts_no_rate": 0.46, "draw_rate": 0.25, "home_win_rate": 0.47, "home_adv": 1.12, "volatility": 0.12},
    "finland-ykkonen":      {"avg_goals": 2.6, "u25_rate": 0.45, "btts_no_rate": 0.44, "draw_rate": 0.24, "home_win_rate": 0.46, "home_adv": 1.10, "volatility": 0.18},
    "finland-kakkonen":     {"avg_goals": 2.70, "u25_rate": 0.42, "btts_no_rate": 0.40, "draw_rate": 0.22, "home_win_rate": 0.46, "home_adv": 1.10, "volatility": 0.25},
    "morocco-botola":       {"avg_goals": 3.0, "u25_rate": 0.55, "btts_no_rate": 0.5, "draw_rate": 0.25, "home_win_rate": 0.52, "home_adv": 1.12, "volatility": 0.08},
    "iceland":              {"avg_goals": 4.17, "u25_rate": 0.33, "btts_no_rate": 0.28, "draw_rate": 0.33, "home_win_rate": 0.39, "home_adv": 1.10, "volatility": 0.15},
    "iceland-women":        {"avg_goals": 2.0, "u25_rate": 0.65, "btts_no_rate": 0.55, "draw_rate": 0.30, "home_win_rate": 0.40, "home_adv": 1.10, "volatility": 0.20},
    "estonia":              {"avg_goals": 4.55, "u25_rate": 0.05, "btts_no_rate": 0.27, "draw_rate": 0.27, "home_win_rate": 0.36, "home_adv": 1.10, "volatility": 0.20},
    "georgia":              {"avg_goals": 2.51, "u25_rate": 0.52, "btts_no_rate": 0.46, "draw_rate": 0.29, "home_win_rate": 0.42, "home_adv": 1.15, "volatility": 0.20},
    "lithuania":            {"avg_goals": 2.58, "u25_rate": 0.33, "btts_no_rate": 0.5, "draw_rate": 0.25, "home_win_rate": 0.25, "home_adv": 1.10, "volatility": 0.20},
    "women-football":       {"avg_goals": 2.96, "u25_rate": 0.44, "btts_no_rate": 0.44, "draw_rate": 0.25, "home_win_rate": 0.38, "home_adv": 1.05, "volatility": 0.20},
    "algeria-ligue-2": {"avg_goals": 2.36, "u25_rate": 0.57, "btts_no_rate": 0.6, "draw_rate": 0.22, "home_win_rate": 0.51, "home_adv": 1.15, "volatility": 0.12},  # Algeria - Ligue 2
    "colombia-a": {"avg_goals": 2.52, "u25_rate": 0.54, "btts_no_rate": 0.48, "draw_rate": 0.3, "home_win_rate": 0.45, "home_adv": 1.15, "volatility": 0.12},  # Colombia - Primera A
    "colombia-b": {"avg_goals": 2.18, "u25_rate": 0.67, "btts_no_rate": 0.56, "draw_rate": 0.31, "home_win_rate": 0.38, "home_adv": 1.1, "volatility": 0.12},  # Colombia - Primera B
    "costa-rica-liga-de-ascenso": {"avg_goals": 2.98, "u25_rate": 0.43, "btts_no_rate": 0.49, "draw_rate": 0.21, "home_win_rate": 0.5, "home_adv": 1.15, "volatility": 0.12},  # Costa Rica - Liga de Ascenso
    "dr-congo-ligue-1": {"avg_goals": 1.98, "u25_rate": 0.69, "btts_no_rate": 0.6, "draw_rate": 0.35, "home_win_rate": 0.41, "home_adv": 1.15, "volatility": 0.08},  # DR Congo - Ligue 1
    "el-salvador-primera": {"avg_goals": 2.58, "u25_rate": 0.55, "btts_no_rate": 0.49, "draw_rate": 0.28, "home_win_rate": 0.39, "home_adv": 1.1, "volatility": 0.12},  # El Salvador - Primera Division
    "guatemala-liga-nacional": {"avg_goals": 2.43, "u25_rate": 0.59, "btts_no_rate": 0.53, "draw_rate": 0.24, "home_win_rate": 0.6, "home_adv": 1.15, "volatility": 0.12},  # Guatemala - Liga Nacional
    "guatemala-primera": {"avg_goals": 2.44, "u25_rate": 0.55, "btts_no_rate": 0.51, "draw_rate": 0.24, "home_win_rate": 0.66, "home_adv": 1.2, "volatility": 0.12},  # Guatemala - Primera Division
    "honduras-liga-nacional": {"avg_goals": 2.63, "u25_rate": 0.52, "btts_no_rate": 0.41, "draw_rate": 0.3, "home_win_rate": 0.43, "home_adv": 1.1, "volatility": 0.12},  # Honduras - Liga Nacional
    "libya-premier": {"avg_goals": 2.35, "u25_rate": 0.57, "btts_no_rate": 0.56, "draw_rate": 0.26, "home_win_rate": 0.42, "home_adv": 1.15, "volatility": 0.12},  # Libya - Premier League
    "mexico-liga-de-expansion-mx": {"avg_goals": 2.77, "u25_rate": 0.49, "btts_no_rate": 0.43, "draw_rate": 0.28, "home_win_rate": 0.56, "home_adv": 1.15, "volatility": 0.12},  # Mexico - Liga de Expansion MX
    "mexico-liga-mx": {"avg_goals": 2.74, "u25_rate": 0.45, "btts_no_rate": 0.41, "draw_rate": 0.26, "home_win_rate": 0.45, "home_adv": 1.15, "volatility": 0.12},  # Mexico - Liga MX
    "mexico-liga-serie-a": {"avg_goals": 2.78, "u25_rate": 0.5, "btts_no_rate": 0.51, "draw_rate": 0.26, "home_win_rate": 0.48, "home_adv": 1.15, "volatility": 0.12},  # Mexico - Liga Premier Serie A
    "nicaragua-primera": {"avg_goals": 2.43, "u25_rate": 0.56, "btts_no_rate": 0.57, "draw_rate": 0.24, "home_win_rate": 0.59, "home_adv": 1.15, "volatility": 0.12},  # Nicaragua - Primera Division
    "panama-football": {"avg_goals": 3.25, "u25_rate": 0.39, "btts_no_rate": 0.36, "draw_rate": 0.29, "home_win_rate": 0.29, "home_adv": 1.1, "volatility": 0.12},  # Panama - Football League
    "saudi-arabia-1st": {"avg_goals": 3.01, "u25_rate": 0.41, "btts_no_rate": 0.42, "draw_rate": 0.22, "home_win_rate": 0.44, "home_adv": 1.15, "volatility": 0.12},  # Saudi Arabia - 1st Division
    "sudan-premier": {"avg_goals": 2.38, "u25_rate": 0.55, "btts_no_rate": 0.56, "draw_rate": 0.3, "home_win_rate": 0.38, "home_adv": 1.1, "volatility": 0.08},  # Sudan - Premier League
    "syria-premier": {"avg_goals": 2.67, "u25_rate": 0.56, "btts_no_rate": 0.56, "draw_rate": 0.0, "home_win_rate": 0.56, "home_adv": 1.1, "volatility": 0.12},  # Syria - Premier League
    "thailand-thai-3": {"avg_goals": 2.68, "u25_rate": 0.5, "btts_no_rate": 0.49, "draw_rate": 0.27, "home_win_rate": 0.46, "home_adv": 1.15, "volatility": 0.12},  # Thailand - Thai League 3
    "turkiye-tff-3-lig": {"avg_goals": 2.78, "u25_rate": 0.48, "btts_no_rate": 0.51, "draw_rate": 0.23, "home_win_rate": 0.46, "home_adv": 1.15, "volatility": 0.12},  # Türkiye - TFF 3. Lig
    "venezuela-primera": {"avg_goals": 2.43, "u25_rate": 0.56, "btts_no_rate": 0.46, "draw_rate": 0.3, "home_win_rate": 0.41, "home_adv": 1.1, "volatility": 0.12},  # Venezuela - Primera Division
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
    # Colombia
    if t.startswith("co"):
        if "2" in t[:4] or "b" in t[:4]: return "colombia-b"
        return "colombia-a"
    if "colombia" in t:
        if "primera b" in t or "segunda" in t: return "colombia-b"
        return "colombia-a"
    # Mexico
    if t.startswith("mx"):
        if "w" in t[:4]: return "women-football"
        if "2" in t[:4]: return "mexico-liga-de-expansion-mx"
        if "3" in t[:4] or "4" in t[:4]: return "mexico-liga-serie-a"
        return "mexico-liga-mx"
    if "mexico" in t or "mx" in t[:3]:
        if "liga mx women" in t or " women" in t: return "women-football"
        if "expansion" in t: return "mexico-liga-de-expansion-mx"
        if "premier" in t: return "mexico-liga-serie-a"
        if "liga mx" in t: return "mexico-liga-mx"
    # Venezuela
    if t.startswith("ve"):
        if "2" in t[:4]: return "default"
        return "venezuela-primera"
    if "venezuela" in t:
        if "segunda" in t: return "default"
        return "venezuela-primera"
    # Guatemala
    if t.startswith("gt"):
        if "1" in t[:4]: return "guatemala-liga-nacional"
        return "guatemala-primera"
    if "guatemala" in t:
        if "liga nacional" in t: return "guatemala-liga-nacional"
        if "primera" in t: return "guatemala-primera"
    # El Salvador
    if t.startswith("sv"):
        return "el-salvador-primera"
    if "el salvador" in t:
        return "el-salvador-primera"
    # Honduras
    if t.startswith("hn"):
        return "honduras-liga-nacional"
    if "honduras" in t:
        return "honduras-liga-nacional"
    # Nicaragua
    if t.startswith("ni"):
        return "nicaragua-primera"
    if "nicaragua" in t:
        return "nicaragua-primera"
    # Costa Rica
    if t.startswith("cr"):
        if "1" in t[:4]: return "default"
        return "costa-rica-liga-de-ascenso"
    if "costa rica" in t:
        if "ascenso" in t: return "costa-rica-liga-de-ascenso"
        return "default"
    # Panama
    if t.startswith("pa"):
        return "panama-football"
    if "panama" in t:
        return "panama-football"
    # Libya
    if t.startswith("ly"):
        return "libya-premier"
    if "libya" in t:
        return "libya-premier"
    # Sudan
    if t.startswith("sd"):
        return "sudan-premier"
    if "sudan" in t:
        return "sudan-premier"
    # Syria
    if t.startswith("sy"):
        return "syria-premier"
    if "syria" in t:
        return "syria-premier"
    # DR Congo
    if t.startswith("cd"):
        return "dr-congo-ligue-1"
    if "dr congo" in t:
        return "dr-congo-ligue-1"
    # Saudi Arabia
    if t.startswith("sa"):
        if "1" in t[:4]: return "default"
        return "saudi-arabia-1st"
    if "saudi" in t:
        if "professional" in t or "1st" in t: return "default"
        return "saudi-arabia-1st"
    # Turkey (Turkiye)
    if t.startswith("tr"):
        if "1" in t[:4]: return "default"
        if "2" in t[:4]: return "default"
        if "3" in t[:4]: return "turkiye-tff-3-lig"
        if "4" in t[:4]: return "turkiye-tff-3-lig"
        if "c" in t[-1:].lower(): return "default"
        return "turkiye-tff-3-lig"
    if "turkiye" in t or "türkiye" in t or "turkey" in t:
        if "super lig" in t: return "default"
        if "1. lig" in t or "tff 1" in t: return "default"
        if "2. lig" in t or "tff 2" in t: return "default"
        if "3. lig" in t or "tff 3" in t: return "turkiye-tff-3-lig"
        if "kupasi" in t: return "default"
        return "turkiye-tff-3-lig"
    # Thailand
    if t.startswith("th"):
        if "1" in t[:4]: return "default"
        if "2" in t[:4]: return "default"
        if "3" in t[:4]: return "thailand-thai-3"
        if "c" in t[-1:].lower(): return "default"
        if "l" in t[-1:].lower(): return "default"
        return "thailand-thai-3"
    if "thailand" in t or "thai" in t:
        if "premier" in t or "league 1" in t: return "default"
        if "league 2" in t: return "default"
        if "league 3" in t: return "thailand-thai-3"
        if "fa cup" in t or "league cup" in t: return "default"
        return "thailand-thai-3"
    # Algeria
    if t.startswith("dz"):
        if "1" in t[:4]: return "default"
        return "algeria-ligue-2"
    if "algeria" in t or "algerie" in t:
        if "ligue 1" in t: return "default"
        return "algeria-ligue-2"
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

# ── Auto-calibrated thresholds (improvement 10) ──
# These get updated from history.db calibration data on each run
CALIBRATED_THRESHOLDS = {
    "near_certain": 0.58,
    "near_certain_margin": 0.12,
    "high": 0.50,
    "high_margin": 0.10,
    "medium_high": 0.42,
    "medium_high_margin": 0.06,
    "medium": 0.38,
    "medium_margin": 0.04,
    "draw_medium_high": 0.36,
    "draw_medium_high_margin": 0.04,
    "draw_medium": 0.33,
}

def _get_league_difficulty(league: str) -> dict:
    """Return difficulty info for a league based on historical accuracy.
    
    Returns dict with:
      - level: "hard" | "medium" | "easy"
      - accuracy: float (0-100)
      - matches: int
      - reason: str
    """
    try:
        import re
        from database import get_db
        conn = get_db()
        
        # Extract league prefix (e.g. "BrC", "Ar3", "Pe2") from the league string
        m = re.match(r'^([A-Za-z]+\d?)\s', league or '')
        prefix = m.group(1) if m else ''
        
        if prefix:
            # Search calibration_log for all entries starting with this prefix
            row = conn.execute("""
                SELECT ROUND(100.0 * SUM(correct) / COUNT(*), 1) as pct, COUNT(*) as cnt
                FROM calibration_log WHERE league LIKE ?
            """, (f"{prefix}%",)).fetchone()
        else:
            # Fallback: search by full league name
            row = conn.execute("""
                SELECT ROUND(100.0 * SUM(correct) / COUNT(*), 1) as pct, COUNT(*) as cnt
                FROM calibration_log WHERE league LIKE ?
            """, (f"%{league}%",)).fetchone()
        
        conn.close()
        
        if not row or not row["pct"] or row["cnt"] < 5:
            return {"level": "medium", "accuracy": 0, "matches": row["cnt"] if row else 0,
                    "reason": "Insufficient data for this league"}
        
        pct = row["pct"]
        cnt = row["cnt"]
        
        if pct >= 80:
            level = "easy"
            reason = f"Historically reliable ({pct:.0f}% accuracy, {cnt} matches)"
        elif pct >= 65:
            level = "medium"
            reason = f"Moderate difficulty ({pct:.0f}% accuracy, {cnt} matches)"
        else:
            level = "hard"
            reason = f"Unreliable league ({pct:.0f}% accuracy, {cnt} matches) — bet with caution"
        
        return {"level": level, "accuracy": pct, "matches": cnt, "reason": reason}
    except Exception:
        return {"level": "medium", "accuracy": 0, "matches": 0, "reason": "Could not determine league difficulty"}


def _auto_calibrate_thresholds():
    """Load calibration data from DB and conservatively adjust thresholds.
    Requires sufficient sample size and only tightens (raises) thresholds
    when overconfidence is detected — never loosens with small samples."""
    try:
        from database import get_db
        conn = get_db()

        # Require minimum total pool to avoid noisy adjustments
        total_pool = conn.execute("SELECT COUNT(*) as cnt FROM calibration_log").fetchone()["cnt"]
        if total_pool < 50:
            conn.close()
            return

        rows = conn.execute("""
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

        adjusted = 0
        min_samples = 25  # Increased from 10 — need 25+ records per level
        target_map = {
            "Near Certain": 0.78,
            "High": 0.65,
            "Medium-High": 0.55,
            "Medium": 0.50,
        }

        for row in rows:
            conf = row["confidence"]
            actual_pct = row["pct"] / 100.0
            target = target_map.get(conf, 0.50)
            n = row["total"]
            if n < min_samples:
                continue

            # Only tighten (raise thresholds) when overconfident
            # Never loosen (lower thresholds) automatically — that introduces risk
            if actual_pct < target - 0.03:
                if conf == "Near Certain":
                    CALIBRATED_THRESHOLDS["near_certain"] = min(0.72, CALIBRATED_THRESHOLDS["near_certain"] + 0.02)
                    adjusted += 1
                elif conf == "High":
                    CALIBRATED_THRESHOLDS["high"] = min(0.65, CALIBRATED_THRESHOLDS["high"] + 0.02)
                    adjusted += 1
                elif conf == "Medium-High":
                    CALIBRATED_THRESHOLDS["medium_high"] = min(0.55, CALIBRATED_THRESHOLDS["medium_high"] + 0.02)
                    adjusted += 1
                elif conf == "Medium":
                    CALIBRATED_THRESHOLDS["medium"] = min(0.50, CALIBRATED_THRESHOLDS["medium"] + 0.02)
                    adjusted += 1

        # Validate hierarchy: Near_Certain > High > Medium-High > Medium
        nc = CALIBRATED_THRESHOLDS["near_certain"]
        hi = CALIBRATED_THRESHOLDS["high"]
        mh = CALIBRATED_THRESHOLDS["medium_high"]
        me = CALIBRATED_THRESHOLDS["medium"]
        if not (nc > hi > mh > me):
            CALIBRATED_THRESHOLDS["near_certain"] = max(nc, hi + 0.05)
            CALIBRATED_THRESHOLDS["high"] = max(hi, mh + 0.05)
            CALIBRATED_THRESHOLDS["medium_high"] = max(mh, me + 0.05)
            adjusted += 1

        if adjusted:
            print(f"[calibrate] Thresholds tightened from {total_pool} calibration records ({adjusted} changes)")
    except Exception as e:
        print(f"[calibrate] Could not auto-calibrate: {e}")


def _ppg(form_str: str) -> float:
    """Points per game from a form string like 'WDLDDL'."""
    pts = sum(3 if c == "W" else 1 if c == "D" else 0 for c in form_str if c in "WDL")
    n = sum(1 for c in form_str if c in "WDL")
    return pts / n if n >= 3 else 1.2


def _form_rates(matches: list) -> dict:
    """Compute W/D/L rates from a list of form match dicts."""
    n = len(matches)
    if n == 0:
        return {"w": 0, "d": 0, "l": 0, "gf_avg": 0, "ga_avg": 0, "pts_avg": 0, "n": 0}
    w = sum(1 for m in matches if m.get("result") == "W")
    d = sum(1 for m in matches if m.get("result") == "D")
    l = n - w - d
    gf = sum(m.get("gf", 0) for m in matches)
    ga = sum(m.get("ga", 0) for m in matches)
    pts = w * 3 + d
    return {
        "w": w, "d": d, "l": l,
        "gf_avg": gf / n, "ga_avg": ga / n,
        "pts_avg": pts / n, "n": n,
        "wins": w, "draws": d, "losses": l,
        "gf_total": gf, "ga_total": ga, "gd": gf - ga,
    }


def build_form_analysis(data: dict) -> dict:
    """Build comprehensive form analysis from scraped match details.

    Analyzes:
      1. Last 6 overall matches for H and A teams
      2. Venue-specific: last 6 home for H, last 6 away for A
      3. Common opponents: compare goals vs shared opponents
      4. League standing differential (pos, GF, GA)
      5. Returns reasoning strings + signal for probability adjustment
    """
    hd = data.get("home_form_details", [])
    ad = data.get("away_form_details", [])

    result = {
        "reasoning": [],
        "signal": 0.0,
        "signal_parts": [],
    }

    if not hd and not ad:
        return result

    # ── Step 1: Overall form (last 6) ──
    h_last6 = hd[:6]
    a_last6 = ad[:6]
    h_overall = _form_rates(h_last6)
    a_overall = _form_rates(a_last6)

    if h_last6:
        result["reasoning"].append(
            f"H last 6: {h_overall['w']}W-{h_overall['d']}D-{h_overall['l']}L "
            f"({h_overall['gf_avg']:.1f}gf {h_overall['ga_avg']:.1f}ga, "
            f"{h_overall['pts_avg']:.1f} ppg)"
        )
    if a_last6:
        result["reasoning"].append(
            f"A last 6: {a_overall['w']}W-{a_overall['d']}D-{a_overall['l']}L "
            f"({a_overall['gf_avg']:.1f}gf {a_overall['ga_avg']:.1f}ga, "
            f"{a_overall['pts_avg']:.1f} ppg)"
        )

    # ── Step 2: Venue-specific form ──
    h_home_matches = [m for m in hd if m.get("venue") == "home"][:6]
    a_away_matches = [m for m in ad if m.get("venue") == "away"][:6]
    h_home = _form_rates(h_home_matches)
    a_away = _form_rates(a_away_matches)

    if h_home_matches:
        result["reasoning"].append(
            f"H at home: {h_home['w']}W-{h_home['d']}D-{h_home['l']}L "
            f"({h_home['gf_avg']:.1f}gf {h_home['ga_avg']:.1f}ga)"
        )
    if a_away_matches:
        result["reasoning"].append(
            f"A away: {a_away['w']}W-{a_away['d']}D-{a_away['l']}L "
            f"({a_away['gf_avg']:.1f}gf {a_away['ga_avg']:.1f}ga)"
        )

    # ── Step 3: Common opponents analysis ──
    h_opps = {m.get("opponent", "").lower().strip(): m for m in h_last6 if m.get("opponent")}
    a_opps = {m.get("opponent", "").lower().strip(): m for m in a_last6 if m.get("opponent")}
    common = set(h_opps.keys()) & set(a_opps.keys())

    if common:
        h_gd_common = 0
        a_gd_common = 0
        h_gf_common = 0
        a_gf_common = 0
        details = []
        for opp in sorted(common):
            hm = h_opps[opp]
            am = a_opps[opp]
            h_gd_common += hm.get("gf", 0) - hm.get("ga", 0)
            a_gd_common += am.get("gf", 0) - am.get("ga", 0)
            h_gf_common += hm.get("gf", 0)
            a_gf_common += am.get("gf", 0)
            details.append(
                f"  vs {opp[:25]}: H {hm['gf']}-{hm['ga']}({hm['result']}) "
                f"A {am['gf']}-{am['ga']}({am['result']})"
            )
        result["reasoning"].append(
            f"Common opponents ({len(common)}): H GD {h_gd_common:+d} vs A GD {a_gd_common:+d}"
        )
        result["reasoning"].extend(details)

        # Signal: positive = away advantage, negative = home advantage
        if len(common) >= 2:
            gd_diff = a_gd_common - h_gd_common
            sig = max(-0.5, min(0.5, gd_diff * 0.10))
            result["signal"] += sig
            if abs(sig) >= 0.05:
                fav = "A" if sig > 0 else "H"
                result["signal_parts"].append(
                    f"Common opp edge: {fav} (GD diff {gd_diff:+d}, sig {sig:+.2f})"
                )

    # ── Step 4: League standing differential ──
    h_pos = data.get("home_pos")
    a_pos = data.get("away_pos")
    h_gf = data.get("home_avg_goals_for")
    h_ga = data.get("home_avg_goals_against")
    a_gf = data.get("away_avg_goals_for")
    a_ga = data.get("away_avg_goals_against")

    if h_pos and a_pos:
        pos_gap = h_pos - a_pos  # negative = H is higher
        if abs(pos_gap) >= 3:
            sig = max(-0.3, min(0.3, pos_gap * 0.04))
            result["signal"] += sig
            fav = "H" if pos_gap < 0 else "A"
            result["reasoning"].append(
                f"Standing: H #{h_pos} vs A #{a_pos} "
                f"({_ord(abs(pos_gap))} place gap, fav {fav}, sig {sig:+.2f})"
            )
            result["signal_parts"].append(f"Standing edge: {fav} ({abs(pos_gap)} pos)")

    if h_gf and h_ga and a_gf and a_ga:
        h_gd = h_gf - h_ga
        a_gd = a_gf - a_ga
        gd_diff = a_gd - h_gd  # positive = A has better GD
        if abs(gd_diff) >= 0.3:
            sig = max(-0.25, min(0.25, gd_diff * 0.08))
            result["signal"] += sig
            fav = "H" if gd_diff < 0 else "A"
            result["reasoning"].append(
                f"Goal diff: H {h_gd:+.1f} vs A {a_gd:+.1f} "
                f"(diff {gd_diff:+.1f}, fav {fav}, sig {sig:+.2f})"
            )
            result["signal_parts"].append(f"GD edge: {fav} ({gd_diff:+.1f})")

    # ── Step 5: Form momentum (recent trend) ──
    if len(h_last6) >= 3:
        h_recent3 = h_last6[:3]
        h_older3 = h_last6[3:6] if len(h_last6) >= 6 else h_last6[3:]
        h_r3 = _form_rates(h_recent3)
        h_o3 = _form_rates(h_older3) if h_older3 else {"pts_avg": h_r3["pts_avg"]}
        if h_r3["pts_avg"] - h_o3["pts_avg"] > 0.5:
            result["reasoning"].append("H trending up (recent 3 better than older)")
        elif h_o3["pts_avg"] - h_r3["pts_avg"] > 0.5:
            result["reasoning"].append("H trending down (recent 3 worse than older)")

    if len(a_last6) >= 3:
        a_recent3 = a_last6[:3]
        a_older3 = a_last6[3:6] if len(a_last6) >= 6 else a_last6[3:]
        a_r3 = _form_rates(a_recent3)
        a_o3 = _form_rates(a_older3) if a_older3 else {"pts_avg": a_r3["pts_avg"]}
        if a_r3["pts_avg"] - a_o3["pts_avg"] > 0.5:
            result["reasoning"].append("A trending up (recent 3 better than older)")
        elif a_o3["pts_avg"] - a_r3["pts_avg"] > 0.5:
            result["reasoning"].append("A trending down (recent 3 worse than older)")

    # ── Summary signal ──
    if result["signal_parts"]:
        result["reasoning"].insert(0,
            f"Form analysis signal: {result['signal']:+.2f} "
            f"({'favors A' if result['signal'] > 0 else 'favors H' if result['signal'] < 0 else 'neutral'})"
        )

    return result


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

    # Adjust for form — capped to avoid streak overreaction.
    # Multiplier range is narrowed and sample-weighted more conservatively so a
    # short hot/cold streak cannot blow up expected goals (e.g. 4.7 home goals).
    hf_len = sum(1 for c in hf if c in "WDL") if hf else 0
    af_len = sum(1 for c in af if c in "WDL") if af else 0
    if h_f is not None:
        f = min(1.15, max(0.85, h_f / 1.2))
        # sqrt weighting: 6 games → 0.62, 10 → 0.80, 20 → ~1.0 (more shrinkage for short form)
        f = 1.0 + (f - 1.0) * min(1.0, (hf_len / 6) ** 0.5)
        exp_h *= f
    if a_f is not None:
        f = min(1.15, max(0.85, a_f / 1.2))
        f = 1.0 + (f - 1.0) * min(1.0, (af_len / 6) ** 0.5)
        exp_a *= f

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

    # Venue-specific goal averages (home-at-home, away-at-away).
    # These come from a SMALL sample (often 3-5 home/away games) and are noisy
    # (e.g. 3.7 GF from 3 home games). Shrink them toward the season/league mean
    # with a conservative weight rather than a 50/50 blend, so a hot/cold venue
    # streak can't dominate expected goals.
    hh_gf = data.get("home_home_avg_goals_for")
    hh_ga = data.get("home_home_avg_goals_against")
    aa_gf = data.get("away_away_avg_goals_for")
    aa_ga = data.get("away_away_avg_goals_against")
    venue_w = 0.30  # venue stats get at most 30% weight; rest = current exp (season/league mean)
    if hh_gf:
        exp_h = exp_h * (1 - venue_w) + hh_gf * venue_w
    if aa_gf:
        exp_a = exp_a * (1 - venue_w) + aa_gf * venue_w
    if hh_ga:
        exp_a = exp_a * (1 - venue_w) + hh_ga * venue_w
    if aa_ga:
        exp_h = exp_h * (1 - venue_w) + aa_ga * venue_w

    # Shots-on-target proxy for xG
    h_sot = data.get("home_shots_ontarget_pct")
    a_sot = data.get("away_shots_ontarget_pct")
    h_tsh = data.get("home_total_shots_pg")
    a_tsh = data.get("away_total_shots_pg")
    if h_sot and h_tsh:
        # Expected goals ≈ shots_on_target * ~0.32 (realistic SoT conversion rate)
        h_xg_proxy = h_tsh * (h_sot / 100.0) * 0.32
        exp_h = (exp_h + h_xg_proxy) / 2
    if a_sot and a_tsh:
        a_xg_proxy = a_tsh * (a_sot / 100.0) * 0.32
        exp_a = (exp_a + a_xg_proxy) / 2

    # ── No-goal / clean-sheet discount (mirrors poisson_predict) ──
    home_score_rate = (data.get("home_scored_pct") or 100) / 100.0
    away_score_rate = (data.get("away_scored_pct") or 100) / 100.0
    home_cs_rate = (data.get("home_clean_sheets_pct") or 0) / 100.0
    away_cs_rate = (data.get("away_clean_sheets_pct") or 0) / 100.0
    exp_h *= (1.0 - 0.5 * (1.0 - home_score_rate)) * (1.0 - 0.5 * away_cs_rate)
    exp_a *= (1.0 - 0.5 * (1.0 - away_score_rate)) * (1.0 - 0.5 * home_cs_rate)

    # H2H goal average adjustment
    h2h_avg = data.get("h2h_avg_total_goals")
    h2h_m = data.get("h2h_matches", 0) or 0
    if h2h_avg and h2h_m >= 3:
        league_avg = profile.get("avg_goals", 2.5)
        h2h_adj = h2h_avg / league_avg
        h2h_adj = max(0.8, min(1.2, h2h_adj))  # clamp
        h2h_adj = 1.0 + (h2h_adj - 1.0) * min(1.0, h2h_m / 6)
        exp_h *= h2h_adj
        exp_a *= h2h_adj

    # Volatility regression: higher volatility -> regress toward league mean.
    # Also apply a mild general shrinkage toward league mean so that teams with
    # small/volatile samples (e.g. a 6-game form streak) cannot produce extreme
    # expected goals. This is the primary guard against over-confident extremes.
    vol = profile.get("volatility", 0.1)
    shrink = 0.10 + vol  # baseline 10% shrinkage, more for volatile leagues
    exp_h = exp_h * (1.0 - shrink) + base * shrink
    exp_a = exp_a * (1.0 - shrink) + base * shrink

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


def prob_btts(exp_h: float, exp_a: float, rho: float = 0.0) -> float:
    """P(Both teams score), with optional Dixon-Coles goal correlation adjustment.
    When rho < 0 (typical -0.12), reduces BTTS probability because low-scoring
    draws are less likely than independent Poisson suggests."""
    import math
    p_h_scores = 1.0 - math.exp(-exp_h)
    p_a_scores = 1.0 - math.exp(-exp_a)
    if rho < 0:
        p_both_zero = math.exp(-exp_h - exp_a)
        return p_h_scores * p_a_scores + rho * p_both_zero
    return p_h_scores * p_a_scores


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


def _load_ml_model():
    """Lazy-load the trained ML model (auto-trains if needed)."""
    global _ML_MODEL
    if _ML_MODEL is None:
        try:
            from ml_model import MLPredictor
            _ML_MODEL = MLPredictor.load(auto_train=True)
        except Exception as e:
            print(f"[ml] Failed to load ML model: {e}")
            _ML_MODEL = None
    return _ML_MODEL if _ML_MODEL and _ML_MODEL.is_trained else None


def _get_dynamic_weights(league_key: str, market: str = "1X2"):
    """Get dynamic ensemble weights from DB tracking (improvement 3)."""
    try:
        from database import get_dynamic_weights
        return get_dynamic_weights(league=league_key, market=market)
    except Exception:
        return None


def _load_calibration_biases():
    """Lazy-load bias corrections from calibration_learner."""
    global _BIAS_CORRECTIONS_LOADED
    if not _BIAS_CORRECTIONS_LOADED:
        _BIAS_CORRECTIONS_LOADED = True
    return _BIAS_CORRECTIONS_LOADED


def _apply_bias_corrections(league_key: str, p_home: float, p_draw: float, p_away: float,
                             p_over: float, p_under: float) -> tuple:
    """Apply learned bias corrections to probabilities for a league."""
    try:
        from calibration_learner import apply_all_bias_corrections
        probs_1x2 = {"Home win": p_home, "Draw": p_draw, "Away win": p_away}
        probs_ou = {"Over": p_over, "Under": p_under}
        corrected_1x2, corrected_ou = apply_all_bias_corrections(
            league_key, probs_1x2, probs_ou, min_samples=10
        )
        return (
            corrected_1x2["Home win"],
            corrected_1x2["Draw"],
            corrected_1x2["Away win"],
            corrected_ou["Over"],
            corrected_ou["Under"],
        )
    except Exception:
        return p_home, p_draw, p_away, p_over, p_under


def _maybe_auto_calibrate():
    """Run automated learning on startup:
    1. Scrape results for past unreviewed matches
    2. Analyze calibration bias
    3. Retrain ML model if enough new data accumulated
    This enables continuous learning without manual intervention.
    """
    try:
        try:
            from auto_learn import step_scrape_results, step_calibrate
        except ImportError:
            from scripts.auto_learn import step_scrape_results, step_calibrate
        from database import get_calibration_data_for_retraining
        stats = get_calibration_data_for_retraining()

        # Step 1: Scrape results for past predictions
        scrape = step_scrape_results(days_back=14, delay=0.3, max_matches=200)
        if scrape["updated"] > 0:
            print(f"[auto-learn] Updated {scrape['updated']} match results from Forebet")

        # Step 2: Analyze calibration if enough entries
        if stats["total_calibration_entries"] >= 50:
            step_calibrate()
    except Exception:
        pass


# ─────────────────────────────────────────────
# Transitive Common-Opponent Analysis
# ─────────────────────────────────────────────

def _team_names_match(a: str, b: str) -> bool:
    """Fuzzy team name matching via substring."""
    if not a or not b:
        return False
    al, bl = a.lower().strip(), b.lower().strip()
    return al in bl or bl in al


def _recency_weight(date_str: str, decay_days: float = 30) -> float:
    """Weight recent results higher (exponential decay)."""
    if not date_str:
        return 0.5
    try:
        dt = datetime.strptime(date_str, "%d/%m/%Y")
        days_ago = (datetime.now() - dt).days
        return max(0.2, min(1.0, 1.0 - days_ago / decay_days))
    except (ValueError, TypeError):
        return 0.5


def _get_team_form_details(data: dict, side: str) -> list:
    """Get form match details for a side, preferring scraped data over DB.

    side: 'home' or 'away'
    Returns list of dicts with opponent, venue, gf, ga, result, date.
    """
    team_name = data.get(f"{side}_team", "")
    if not team_name:
        return []

    # Primary source: scraped form details from the Forebet page
    scraped = data.get(f"{side}_form_details")
    if scraped and len(scraped) >= 3:
        return scraped

    # Fallback: query the DB for actual historical results
    return _get_team_results_from_db(team_name, limit=30)


def _get_team_results_from_db(team_name: str, limit: int = 8) -> list:
    """Get recent match results for a team from the DB (history)."""
    from database import get_db
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT home_team, away_team, actual_home_goals, actual_away_goals,
                   match_date
            FROM matches
            WHERE (home_team LIKE ? ESCAPE '\\' OR away_team LIKE ? ESCAPE '\\')
              AND actual_home_goals IS NOT NULL
              AND actual_away_goals IS NOT NULL
            ORDER BY id DESC
            LIMIT ?
        """, (f'%{team_name}%', f'%{team_name}%', limit))
        rows = cur.fetchall()
        conn.close()

        results = []
        seen = set()
        for row in rows:
            home = row["home_team"] or ""
            away = row["away_team"] or ""
            hg = row["actual_home_goals"]
            ag = row["actual_away_goals"]
            date = row["match_date"] or ""

            # Deduplicate by scoreline
            key = (home, away, hg, ag)
            if key in seen:
                continue
            seen.add(key)

            if team_name.lower() in home.lower():
                results.append({
                    "opponent": away,
                    "venue": "home", "gf": int(hg), "ga": int(ag),
                    "result": "W" if hg > ag else "D" if hg == ag else "L",
                    "date": date,
                })
            elif team_name.lower() in away.lower():
                results.append({
                    "opponent": home,
                    "venue": "away", "gf": int(ag), "ga": int(hg),
                    "result": "W" if ag > hg else "D" if ag == hg else "L",
                    "date": date,
                })
        return results[:limit]
    except Exception:
        return []


def _transitive_common_opponent_analysis(data: dict) -> dict:
    """
    Transitive common-opponent analysis.

    Core insight:
      If Team A (home) lost at home to X, but Team B (away) beat X,
      then B has a transitive edge over A — "what X did to A, B did to X."

    Also checks the reverse (away team's away losses → home team beat that opponent)
    and pure common-opponent performance comparison.

    Returns:
      {"signal": float, "confidence": str, "reasoning": list, "prediction": str|None}
      signal > 0  → favors away team; signal < 0 → favors home team
    """
    home_team = data.get("home_team", "")
    away_team = data.get("away_team", "")
    if not home_team or not away_team:
        return {"signal": 0.0, "confidence": "Low", "reasoning": [], "prediction": None}

    home_form = _get_team_form_details(data, "home")
    away_form = _get_team_form_details(data, "away")

    if not home_form or not away_form:
        return {"signal": 0.0, "confidence": "Low", "reasoning": [], "prediction": None}

    reasoning = []
    signal = 0.0
    signals_found = 0

    # -- Signal 1: Home team's home losses → away team beat that opponent --
    home_home_losses = [m for m in home_form if m["venue"] == "home" and m["result"] == "L"]

    for hl in home_home_losses:
        opponent = hl["opponent"]
        if not opponent:
            continue
        for aw in away_form:
            if aw["result"] == "W" and _team_names_match(aw["opponent"], opponent):
                margin = (hl["ga"] - hl["gf"]) + (aw["gf"] - aw["ga"])
                recency = (_recency_weight(hl["date"]) + _recency_weight(aw["date"])) / 2
                venue_bonus = 1.2 if aw["venue"] == "away" else 1.0
                strength = (0.30 + 0.08 * min(margin, 3)) * recency * venue_bonus
                signal += strength
                signals_found += 1
                reasoning.append(
                    f"[TRANS] {home_team} lost {hl['gf']}-{hl['ga']} at home to {opponent}, "
                    f"but {away_team} beat them {aw['gf']}-{aw['ga']} ({aw['venue']})"
                )
                break

    # -- Signal 2: Away team's away losses → home team beat that opponent --
    away_away_losses = [m for m in away_form if m["venue"] == "away" and m["result"] == "L"]

    for al in away_away_losses:
        opponent = al["opponent"]
        if not opponent:
            continue
        for hw in home_form:
            if hw["result"] == "W" and _team_names_match(hw["opponent"], opponent):
                margin = (al["ga"] - al["gf"]) + (hw["gf"] - hw["ga"])
                recency = (_recency_weight(al["date"]) + _recency_weight(hw["date"])) / 2
                venue_bonus = 1.2 if hw["venue"] == "home" else 1.0
                strength = -(0.30 + 0.08 * min(margin, 3)) * recency * venue_bonus
                signal += strength
                signals_found += 1
                reasoning.append(
                    f"[TRANS-R] {away_team} lost {al['gf']}-{al['ga']} away to {opponent}, "
                    f"but {home_team} beat them {hw['gf']}-{hw['ga']} ({hw['venue']})"
                )
                break

    # -- Signal 3: Pure common-opponent performance comparison --
    home_opp_map = {}
    for m in home_form:
        opp = m["opponent"]
        if opp not in home_opp_map:
            home_opp_map[opp] = []
        home_opp_map[opp].append(m)

    away_opp_map = {}
    for m in away_form:
        opp = m["opponent"]
        if opp not in away_opp_map:
            away_opp_map[opp] = []
        away_opp_map[opp].append(m)

    home_opp_set = set(home_opp_map.keys())
    away_opp_set = set(away_opp_map.keys())
    common_opps = home_opp_set & away_opp_set

    for opp in common_opps:
        h_results = home_opp_map[opp]
        a_results = away_opp_map[opp]

        h_score = sum(m["gf"] - m["ga"] for m in h_results)
        a_score = sum(m["gf"] - m["ga"] for m in a_results)

        diff = a_score - h_score
        if abs(diff) < 1:
            continue

        capped = max(-3, min(3, diff))
        sig = 0.08 * capped
        signal += sig
        signals_found += 1

        if diff > 0:
            reasoning.append(
                f"[TRANS-C] {away_team} (GD {a_score:+d}) outperforms {home_team} (GD {h_score:+d}) vs {opp}"
            )
        else:
            reasoning.append(
                f"[TRANS-C] {home_team} (GD {h_score:+d}) outperforms {away_team} (GD {a_score:+d}) vs {opp}"
            )

    # -- Signal 4: League position advantage (when contradicting form) --
    hp = data.get("home_pos")
    ap = data.get("away_pos")
    if hp is not None and ap is not None:
        pos_gap = hp - ap  # positive → away is higher ranked
        if abs(pos_gap) >= 3:
            capped = max(-5, min(5, pos_gap))
            pos_sig = capped * 0.03
            signal += pos_sig
            signals_found += 1
            if pos_gap > 0:
                reasoning.append(
                    f"[TRANS-P] {away_team} (#{ap}) {abs(pos_gap)} positions above {home_team} (#{hp}) — standing advantage"
                )
            else:
                reasoning.append(
                    f"[TRANS-P] {home_team} (#{hp}) {abs(pos_gap)} positions above {away_team} (#{ap}) — standing advantage"
                )

    abs_sig = abs(signal)
    if abs_sig >= 1.0:
        confidence = "High"
    elif abs_sig >= 0.55:
        confidence = "Medium-High"
    elif abs_sig >= 0.25:
        confidence = "Medium"
    else:
        confidence = "Low"

    # -- Draw signal 1: weak/no clear edge from common opponents --
    draw_signal = 0.0
    if signals_found > 0 and abs_sig < 0.25:
        draw_signal = 0.12
        reasoning.append(
            f"[TRANS-D] Common opponents show no clear edge ({abs_sig:.2f}) — evenly matched"
        )

    # -- Draw signal 2: H2H draw rate --
    h2h_m = data.get("h2h_matches", 0) or 0
    h2h_d = data.get("h2h_draws", 0) or 0
    h2h_draw_rate = h2h_d / h2h_m if h2h_m >= 5 else 0.0
    if h2h_draw_rate >= 0.35:
        draw_signal = max(draw_signal, h2h_draw_rate * 0.15)
        if signals_found == 0:
            reasoning.append(
                f"[TRANS-D] No common opponents found — H2H draw rate {h2h_draw_rate:.0%} ({h2h_d}/{h2h_m} matches)"
            )
        else:
            reasoning.append(
                f"[TRANS-D] H2H draw rate {h2h_draw_rate:.0%} ({h2h_d}/{h2h_m} matches)"
            )

    if signal > 0:
        prediction = "Away win"
    else:
        prediction = "Home win"

    return {
        "signal": signal,
        "confidence": confidence,
        "reasoning": reasoning,
        "prediction": prediction,
        "draw_signal": draw_signal,
    }


def _common_opponent_strength(data: dict) -> dict:
    """Derive attack/defense strength multipliers from head-to-head form vs the
    SAME opponents (the fairest available strength comparison).

    For each common opponent O, we have:
      H scored gf_H vs O, conceded ga_H vs O   (home team's view)
      A scored gf_A vs O, conceded ga_A vs O   (away team's view)
    Per-game goal difference vs O:  gd_H = gf_H - ga_H,  gd_A = gf_A - ga_A.
    If gd_H > gd_A, the home team is relatively stronger (it out-scored/out-defended
    the same opposition), so we nudge exp_h up and exp_a down proportionally — and
    vice-versa. This makes expected goals reflect "scores X against THIS opponent"
    rather than just adding two independent season rates together.

    Returns {"h_mult": float, "a_mult": float, "reason": str|None}.
    """
    home_form = _get_team_form_details(data, "home")
    away_form = _get_team_form_details(data, "away")
    if not home_form or not away_form:
        return {"h_mult": 1.0, "a_mult": 1.0, "reason": None}

    home_opp = {}
    for m in home_form:
        if m.get("opponent"):
            home_opp.setdefault(m["opponent"], []).append(m)
    away_opp = {}
    for m in away_form:
        if m.get("opponent"):
            away_opp.setdefault(m["opponent"], []).append(m)

    common = set(home_opp) & set(away_opp)
    if not common:
        return {"h_mult": 1.0, "a_mult": 1.0, "reason": None}

    h_gd = 0.0
    a_gd = 0.0
    n = 0
    for opp in common:
        h_g = sum(m["gf"] - m["ga"] for m in home_opp[opp])
        a_g = sum(m["gf"] - m["ga"] for m in away_opp[opp])
        h_gd += h_g
        a_gd += a_g
        n += 1

    if n == 0:
        return {"h_mult": 1.0, "a_mult": 1.0, "reason": None}

    # Average per-common-opponent goal difference gap (home relative to away)
    gap = (h_gd - a_gd) / n  # positive → home stronger
    # Gentle scaling: each 1.0 GD gap → ±10% on the relevant rate (clamped)
    scale = max(-0.30, min(0.30, gap * 0.10))
    h_mult = 1.0 + scale
    a_mult = 1.0 - scale
    reason = (f"Common-opp strength: H GD {h_gd:+.1f}/{n} vs A GD {a_gd:+.1f}/{n} "
              f"→ H×{h_mult:.2f} A×{a_mult:.2f}")
    return {"h_mult": h_mult, "a_mult": a_mult, "reason": reason}


def analyze_from_data(data: dict, use_ml: bool = False) -> dict:
    """Analyze all markets, recommend highest-conviction pick.
    
    When use_ml=True, replaces the simple Poisson model with the enhanced
    attack/defense strength model and blends with ML probabilities.
    Uses Dixon-Coles bivariate Poisson for goal correlation (improvement 4).
    Uses dynamic ensemble weights from DB (improvement 3).
    """
    league_key = detect_league(data.get("league", ""))
    profile = get_profile(league_key)
    reasoning = []
    candidates = []

    # ── League reliability factor ──
    # Lower historical accuracy → dampen O/U/BTTS confidence (the model tends to
    # over-predict goals in unreliable leagues). 1.0 = fully reliable.
    _ld = _get_league_difficulty(data.get("league", ""))
    _league_acc = _ld.get("accuracy", 0) or 0
    if _league_acc >= 65:
        league_reliability = 1.0
    elif _league_acc >= 45:
        league_reliability = 0.7
    elif _league_acc > 0:
        league_reliability = 0.45
    else:
        league_reliability = 0.7  # unknown → moderately cautious
    if _ld.get("level") == "hard":
        reasoning.append(f"⚠ {_ld.get('reason', 'Unreliable league')}")

    hf, af = data.get("home_form", ""), data.get("away_form", "")
    h_ppg = _ppg(hf) if hf else None
    a_ppg = _ppg(af) if af else None
    hp, ap = data.get("home_pos"), data.get("away_pos")
    hm_ = data.get("h2h_matches", 0)
    hw_ = data.get("h2h_home_wins", 0) if hm_ >= 3 else 0
    ha_ = data.get("h2h_away_wins", 0) if hm_ >= 3 else 0

    # Auto-calibrate thresholds from DB (improvement 10)
    _auto_calibrate_thresholds()

    # ── Form analysis (runs early to inform predictions) ──
    form_analysis = build_form_analysis(data)

    # ── ML-enhanced probability computation ──
    ml_model = _load_ml_model() if use_ml else None
    method_parts = []
    # Accumulates how strongly the adjustment signals fired; drives the
    # single final blend weight between ML/DC base and exp-derived probs.
    signal_blend = 0.0

    if ml_model:
        from ml_model import poisson_predict, ensemble_predict
        # Use enhanced attack/defense Poisson with Dixon-Coles
        enhanced = poisson_predict(data, profile, use_dixon_coles=True)
        p_home = enhanced["prob_home"]
        p_draw = enhanced["prob_draw"]
        p_away = enhanced["prob_away"]
        p_over = enhanced["prob_over"]
        p_under = enhanced["prob_under"]
        exp_h = enhanced["exp_home_goals"]
        exp_a = enhanced["exp_away_goals"]
        exp_total = exp_h + exp_a
        method_parts.append("dc-poisson")

        # ── Common-opponent strength: scale exp by how each team performed vs
        # the SAME opponents (fairer than adding two independent season rates). ──
        _cos = _common_opponent_strength(data)
        if _cos["reason"]:
            exp_h = max(0.1, exp_h * _cos["h_mult"])
            exp_a = max(0.1, exp_a * _cos["a_mult"])
            exp_total = exp_h + exp_a
            signal_blend = min(0.65, signal_blend + 0.12)
            reasoning.append(f"⚠ {_cos['reason']}")

        # Get dynamic weights from DB (improvement 3)
        dynamic_weights = _get_dynamic_weights(league_key)

        # Blend with ML model using market-specific weights
        ensemble = ensemble_predict(data, profile, ml_model, dynamic_weights=dynamic_weights, league=league_key, market="1X2")
        p_home = ensemble["prob_home"]
        p_draw = ensemble["prob_draw"]
        p_away = ensemble["prob_away"]
        p_over = ensemble["prob_over"]
        p_under = ensemble["prob_under"]
        method_parts.append(f"ml({getattr(ml_model, 'cv_accuracy_1x2', 0):.2f})")
        if dynamic_weights:
            method_parts.append("dyn-weights")

        # ── Form signal: shift expected goals only (probabilities recomputed once at end) ──
        fsig = form_analysis.get("signal", 0.0)
        if abs(fsig) >= 0.05:
            shift = fsig * 0.08
            exp_h -= shift
            exp_a += shift
            exp_h = max(exp_h, 0.05)
            exp_a = max(exp_a, 0.05)
            signal_blend = min(0.65, signal_blend + abs(fsig))
            method_parts.append("form")

        # Concordance boost: Forebet + Poisson agreement
        fb_h_raw = (data.get("forebet_home_pct") or 0) / 100.0
        fb_d_raw = (data.get("forebet_draw_pct") or 0) / 100.0
        fb_a_raw = (data.get("forebet_away_pct") or 0) / 100.0
        fb_total_raw = fb_h_raw + fb_d_raw + fb_a_raw
        if fb_total_raw > 0:
            fb_h_n = fb_h_raw / fb_total_raw
            fb_d_n = fb_d_raw / fb_total_raw
            fb_a_n = fb_a_raw / fb_total_raw
            fb_probs = {"Home win": fb_h_n, "Draw": fb_d_n, "Away win": fb_a_n}
            fb_top = max(fb_probs, key=fb_probs.get)
            our_probs = {"Home win": p_home, "Draw": p_draw, "Away win": p_away}
            our_top = max(our_probs, key=our_probs.get)
            if fb_top == our_top and fb_probs[fb_top] > 0.50:
                boost = 0.05 if fb_probs[fb_top] > 0.60 else 0.03
                if fb_top == "Home win":
                    p_home = min(p_home + boost, 0.95)
                elif fb_top == "Draw":
                    p_draw = min(p_draw + boost, 0.95)
                else:
                    p_away = min(p_away + boost, 0.95)
                total_p = p_home + p_draw + p_away
                p_home /= total_p
                p_draw /= total_p
                p_away /= total_p
                method_parts.append("concordance")
    else:
        # Enhanced Poisson with Dixon-Coles even without ML
        from ml_model import poisson_predict as ml_poisson_predict
        try:
            enhanced = ml_poisson_predict(data, profile, use_dixon_coles=True)
            p_home = enhanced["prob_home"]
            p_draw = enhanced["prob_draw"]
            p_away = enhanced["prob_away"]
            p_over = enhanced["prob_over"]
            p_under = enhanced["prob_under"]
            exp_h = enhanced["exp_home_goals"]
            exp_a = enhanced["exp_away_goals"]
            exp_total = exp_h + exp_a
            method_parts.append("dc-poisson")
        except Exception:
            # Original simple Poisson model
            exp_h, exp_a = estimate_goals(data, profile)
            exp_total = exp_h + exp_a
            p_home = prob_home_win(exp_h, exp_a)
            p_draw = prob_draw(exp_h, exp_a)
            p_away = prob_away_win(exp_h, exp_a)

            # Draw inflation
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
            method_parts.append("simple-poisson")

        # ── Form signal (non-ML path): shift expected goals only ──
        fsig = form_analysis.get("signal", 0.0)
        if abs(fsig) >= 0.05:
            shift = fsig * 0.08
            exp_h -= shift
            exp_a += shift
            exp_h = max(exp_h, 0.05)
            exp_a = max(exp_a, 0.05)
            signal_blend = min(0.65, signal_blend + abs(fsig))
            method_parts.append("form")

    # ── Transitive common-opponent analysis: adjust expected goals ──
    _trans_analysis = _transitive_common_opponent_analysis(data)
    trans_adjusted = False
    draw_adjusted = False
    trans_signal = 0.0  # default; only overridden when transitivity fires
    if _trans_analysis and _trans_analysis["reasoning"]:
        trans_signal = _trans_analysis.get("signal", 0.0)
        trans_conf = _trans_analysis.get("confidence", "Low")
        trans_weight = {"High": 0.30, "Medium-High": 0.20, "Medium": 0.12}.get(trans_conf, 0.0)
        abs_sig = min(abs(trans_signal), 1.0)
        if trans_weight > 0 and abs_sig >= 0.15:
            shift = trans_weight * abs_sig * 0.15
            if trans_signal < 0:
                exp_h += shift
                exp_a = max(exp_a - shift * 0.3, 0.05)
            else:
                exp_a += shift
                exp_h = max(exp_h - shift * 0.3, 0.05)
            signal_blend = min(0.65, signal_blend + trans_weight * abs_sig)
            method_parts.append("trans")
            trans_adjusted = True

        # -- Draw tendency: reduce expected-goals gap when teams are evenly matched --
        draw_signal_val = _trans_analysis.get("draw_signal", 0.0)
        if draw_signal_val > 0 and trans_conf not in ("High", "Medium-High"):
            exp_avg = (exp_h + exp_a) / 2
            gap = abs(exp_h - exp_a)
            reduction = gap * draw_signal_val * 0.3
            if exp_h > exp_a:
                exp_h = max(exp_h - reduction, exp_avg)
                exp_a = min(exp_a + reduction, exp_avg)
            else:
                exp_a = max(exp_a - reduction, exp_avg)
                exp_h = min(exp_h + reduction, exp_avg)
            signal_blend = min(0.65, signal_blend + min(0.5, draw_signal_val * 0.3))
            method_parts.append("draw")
            draw_adjusted = True

    # ── Single final probability recompute from adjusted expected goals ──
    # Every signal above only modified exp_h/exp_a; now derive probabilities once
    # and blend with the ML/DC base, weighted by how strongly the signals fired.
    # This removes the repeated prob↔goal round-tripping that produced extremes.
    exp_total = exp_h + exp_a
    _ph = prob_home_win(exp_h, exp_a)
    _pd = prob_draw(exp_h, exp_a)
    _pa = prob_away_win(exp_h, exp_a)
    _po = prob_over(exp_h, exp_a, 2.5)
    _pu = 1.0 - _po
    _tp = _ph + _pd + _pa
    if _tp > 0:
        _ph /= _tp
        _pd /= _tp
        _pa /= _tp
    _w = min(0.65, signal_blend)
    p_home = p_home * (1 - _w) + _ph * _w
    p_draw = p_draw * (1 - _w) + _pd * _w
    p_away = p_away * (1 - _w) + _pa * _w
    p_over = p_over * (1 - _w) + _po * _w
    p_under = 1.0 - p_over

    # ── Apply learned bias corrections from calibration learning ──
    _load_calibration_biases()
    p_home_bias, p_draw_bias, p_away_bias, p_over_bias, p_under_bias = \
        _apply_bias_corrections(league_key, p_home, p_draw, p_away, p_over, p_under)
    bias_applied = (
        abs(p_home - p_home_bias) > 0.005 or
        abs(p_over - p_over_bias) > 0.005
    )
    if bias_applied:
        method_parts.append("bias-corrected")
        p_home, p_draw, p_away = p_home_bias, p_draw_bias, p_away_bias
        p_over, p_under = p_over_bias, p_under_bias

    # ── Apply isotonic regression calibration (improvement 13) ──
    p_home = apply_calibration("1X2", p_home)
    p_draw = apply_calibration("1X2", p_draw)
    p_away = apply_calibration("1X2", p_away)
    p_over = apply_calibration("O/U", p_over)
    p_under = apply_calibration("O/U", p_under)

    # Re-normalize 1X2 after calibration
    total_1x2 = p_home + p_draw + p_away
    if total_1x2 > 0:
        p_home /= total_1x2
        p_draw /= total_1x2
        p_away /= total_1x2

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

    # Use calibrated thresholds (improvement 10)
    nc_thresh = CALIBRATED_THRESHOLDS["near_certain"]
    nc_margin = CALIBRATED_THRESHOLDS["near_certain_margin"]
    hi_thresh = CALIBRATED_THRESHOLDS["high"]
    hi_margin = CALIBRATED_THRESHOLDS["high_margin"]
    mh_thresh = CALIBRATED_THRESHOLDS["medium_high"]
    mh_margin = CALIBRATED_THRESHOLDS["medium_high_margin"]
    med_thresh = CALIBRATED_THRESHOLDS["medium"]
    med_margin = CALIBRATED_THRESHOLDS["medium_margin"]

    # Draws are harder to predict — use tighter thresholds
    if top_pick == "Draw":
        draw_mh = CALIBRATED_THRESHOLDS["draw_medium_high"]
        draw_mh_margin = CALIBRATED_THRESHOLDS["draw_medium_high_margin"]
        draw_med = CALIBRATED_THRESHOLDS["draw_medium"]
        if top_prob >= draw_mh and margin >= draw_mh_margin:
            best_12_conf = "Medium-High"
        elif top_prob >= draw_med:
            best_12_conf = "Medium"
    else:
        # Calibrated thresholds for Home/Away win
        if top_prob >= nc_thresh and margin >= nc_margin:
            best_12_conf = "Near Certain"
        elif top_prob >= hi_thresh and margin >= hi_margin:
            best_12_conf = "High" if margin >= hi_margin else "Medium-High"
        elif top_prob >= mh_thresh:
            best_12_conf = "Medium-High" if margin >= mh_margin else "Medium"
        elif top_prob >= med_thresh and margin >= med_margin:
            best_12_conf = "Medium"

    # Draw proximity penalty: when Draw is close to top pick, downgrade confidence
    # Draws are frequently the "default" result when model is uncertain
    if top_pick != "Draw" and p_draw >= 0.28:
        if margin <= 0.10:
            # Draw within 10% of top — strong draw signal, cap at Medium
            if best_12_conf in ("Near Certain", "High", "Medium-High"):
                best_12_conf = "Medium"
        elif margin <= 0.15 and p_draw >= 0.30:
            # Draw within 15% AND above 30% — moderate draw signal
            if best_12_conf in ("Near Certain", "High"):
                best_12_conf = "Medium-High"

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
        # Add form summary
        h_form = data.get("home_form", "") or ""
        a_form = data.get("away_form", "") or ""
        if h_form:
            parts.append(f"H:{h_form[:6]}")
        if a_form:
            parts.append(f"A:{a_form[:6]}")
        best_12_reason = " ".join(parts)
        add("1X2", top_pick, best_12_conf, best_12_reason,
            model_prob={"Home win": p_home, "Draw": p_draw, "Away win": p_away}.get(top_pick))

    # Also consider Draw as a secondary candidate if it's close but not top
    if top_pick != "Draw":
        draw_conf = "Low"
        if p_draw >= 0.38:
            draw_conf = "Medium-High"
        elif p_draw >= 0.32:
            draw_conf = "Medium"
        elif p_draw >= 0.26:
            draw_conf = "Low"
        # Boost if margin is small (draw is competitive)
        if margin <= 0.10 and p_draw >= 0.30:
            if draw_conf == "Low":
                draw_conf = "Medium"
        if draw_conf != "Low":
            add("1X2", "Draw", draw_conf, f"model {p_draw:.0%} (close to top)", model_prob=p_draw)

    # Draw tendency signal: flag when Draw is close to primary 1X2 pick
    _draw_tendency = False
    if top_pick != "Draw" and p_draw >= 0.28 and margin <= 0.12:
        _draw_tendency = True
        reasoning.append(f"⚠ Draw tendency: {p_draw:.0%} vs {top_pick} {top_prob:.0%} (margin {margin:.0%})")

    # ── Composite draw signal: multi-factor draw detection ──
    _draw_factors = []
    h_form = data.get("home_form") or ""
    a_form = data.get("away_form") or ""
    h_d_count = sum(1 for c in h_form[:6] if c == "D")
    a_d_count = sum(1 for c in a_form[:6] if c == "D")

    # Venue stats for draw signal factors
    hh_gf = data.get("home_home_avg_goals_for")
    hh_ga = data.get("home_home_avg_goals_against")
    aa_gf = data.get("away_away_avg_goals_for")
    aa_ga = data.get("away_away_avg_goals_against")
    h_ou15 = data.get("home_over15_pct")

    # Factor 1: Home team has ≥2 draws in recent form
    if h_d_count >= 2:
        _draw_factors.append(f"H:{h_d_count}D")

    # Factor 2: Away team has ≥2 draws in recent form
    if a_d_count >= 2:
        _draw_factors.append(f"A:{a_d_count}D")

    # Factor 3: Expected goals differential ≤ 0.5
    exp_diff = abs(exp_h - exp_a)
    if exp_diff <= 0.5 and exp_h + exp_a > 0:
        _draw_factors.append(f"expΔ{exp_diff:.1f}")

    # Factor 4: Draw odds ≤ 3.0 (bookmaker signals draw)
    draw_odds = data.get("odds_draw")
    if draw_odds and draw_odds <= 3.0:
        _draw_factors.append(f"odds{draw_odds:.1f}")

    # Factor 5: Form analysis signal is neutral (|signal| ≤ 0.2)
    if abs(fsig) <= 0.2:
        _draw_factors.append("form-neutral")

    # Factor 6: Transitive draw tendency
    if _trans_analysis and _trans_analysis.get("draw_signal", 0) > 0:
        _draw_factors.append("trans-draw")

    # Factor 7: Both teams score ≤1.0 goals at venue (low-scoring game likely)
    if hh_gf is not None and aa_gf is not None:
        if hh_gf <= 1.0 and aa_gf <= 1.0:
            _draw_factors.append("venue-low")

    # Factor 8: Home O15 rate < 60% (many low-scoring games)
    if h_ou15 is not None and h_ou15 < 60:
        _draw_factors.append(f"O15:{h_ou15}%")

    # Factor 9: Home BTTS rate < 45% (one team often fails to score)
    btts_h = data.get("home_btts_yes_pct")
    if btts_h is not None and btts_h < 45:
        _draw_factors.append(f"BTTS:{btts_h}%")

    # Factor 10: Home CS rate > 25% (home keeps clean sheets → 0-0/1-0 draws)
    cs_h = data.get("home_clean_sheets_pct")
    if cs_h is not None and cs_h > 25:
        _draw_factors.append(f"CS:{cs_h}%")

    # Boost draw confidence when multiple factors align
    if len(_draw_factors) >= 2 and top_pick != "Draw":
        _draw_boost = min(len(_draw_factors) * 0.03, 0.10)
        p_draw_boosted = min(p_draw + _draw_boost, 0.60)
        # Upgrade Draw confidence
        if p_draw_boosted >= 0.35 and draw_conf == "Low":
            draw_conf = "Medium"
            # Re-add Draw with upgraded confidence
            candidates = [c for c in candidates if not (c["market"] == "1X2" and c["pick"] == "Draw")]
            add("1X2", "Draw", draw_conf, f"model {p_draw:.0%} ({'+'.join(_draw_factors)})", model_prob=p_draw_boosted)
        reasoning.append(f"⚠ Draw signal ({len(_draw_factors)}f): {', '.join(_draw_factors)}")

    # ── Draw override: when composite signal is strong + margin small, make Draw the primary ──
    _draw_override = False
    _draw_override_reason = ""
    # Ensure p_draw_boosted is defined (fallback to p_draw if boost block didn't run)
    try:
        p_draw_boosted
    except NameError:
        p_draw_boosted = p_draw
    if len(_draw_factors) >= 5 and top_pick != "Draw":
        # Only override to Draw when it is the genuine model leader BEFORE the
        # cosmetic draw boost — compare the unboosted model probabilities so the
        # boost block (which can lift Draw above 0.60) cannot manufacture a Draw
        # primary. Require Draw to clearly top the best side win probability.
        _side_max = max(p_home, p_away)
        if p_draw >= 0.42 and p_draw - _side_max >= 0.04:
            _draw_override = True
            _draw_override_reason = f"Draw signal override ({len(_draw_factors)}f, margin {margin:.0%})"

    if _draw_override:
        # Remove the current primary pick (Home/Away win) and any existing Draw candidate
        candidates = [c for c in candidates if not (c["market"] == "1X2" and c["pick"] == top_pick)]
        candidates = [c for c in candidates if not (c["market"] == "1X2" and c["pick"] == "Draw")]
        # Determine Draw override confidence based on boosted probability
        if p_draw_boosted >= 0.40:
            draw_override_conf = "Medium-High"
        elif p_draw_boosted >= 0.35:
            draw_override_conf = "Medium"
        else:
            draw_override_conf = "Medium"
        add("1X2", "Draw", draw_override_conf,
            f"model {p_draw:.0%} ({'+'.join(_draw_factors)})", model_prob=p_draw_boosted)
        reasoning.append(f"⚠ {_draw_override_reason}")
        # Update top_pick/top_prob for downstream logic (DC, DNB, etc.)
        top_pick = "Draw"
        top_prob = p_draw
        margin = 0.0  # Draw is now the reference
        _draw_tendency = True  # Ensure backup pick logic activates

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

    # ── Data quality assessment (used by DC and DNB) ──
    data_quality = 1.0
    missing_data = 0
    if not data.get("home_pos") or not data.get("away_pos"):
        missing_data += 1
    if not data.get("home_avg_goals_for") or not data.get("away_avg_goals_for"):
        missing_data += 1
    hf_len = sum(1 for c in (data.get("home_form") or "") if c in "WDL")
    af_len = sum(1 for c in (data.get("away_form") or "") if c in "WDL")
    if hf_len < 4 or af_len < 4:
        missing_data += 1
    if missing_data >= 2:
        data_quality = 0.85
    elif missing_data == 1:
        data_quality = 0.92

    # ── Draw No Bet (derived from 1X2) — volatility-gated ──
    # Skip DNB entirely in very high volatility (unpredictable leagues)
    dnb_home_conf = "Low"
    dnb_away_conf = "Low"
    if vol < 0.25:
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

    # Data quality penalty for DNB: downgrade when key data missing
    if missing_data >= 2:
        if dnb_home_conf == "High": dnb_home_conf = "Medium-High"
        if dnb_away_conf == "High": dnb_away_conf = "Medium-High"
    elif missing_data == 1:
        if dnb_home_conf == "Near Certain": dnb_home_conf = "High"
        if dnb_away_conf == "Near Certain": dnb_away_conf = "High"

    # Volatility capping for DNB
    if vol >= 0.20:
        if dnb_home_conf in ("Near Certain", "High"): dnb_home_conf = "Medium-High"
        if dnb_away_conf != "Low": dnb_away_conf = "Medium"  # Skip away DNB in moderate-high vol
    elif vol >= 0.15:
        if dnb_home_conf == "Near Certain": dnb_home_conf = "High"
        if dnb_away_conf == "Near Certain": dnb_away_conf = "High"
        if dnb_away_conf == "Medium-High": dnb_away_conf = "Medium"

    # DNB-specific odds floor: require minimum odds for any DNB pick
    dnb_odds_home = data.get("odds_home")
    dnb_odds_away = data.get("odds_away")
    if dnb_home_conf != "Low" and dnb_odds_home and dnb_odds_home < 1.40:
        dnb_home_conf = "Low"
    if dnb_away_conf != "Low" and dnb_odds_away and dnb_odds_away < 1.45:
        dnb_away_conf = "Low"

    dnb_denom_h = p_home + p_draw
    dnb_denom_a = p_away + p_draw
    if dnb_home_conf != "Low":
        add("DNB", "Home", dnb_home_conf, "derived from model",
            model_prob=(p_home / dnb_denom_h) if dnb_denom_h > 0 else None)
    if dnb_away_conf != "Low":
        add("DNB", "Away", dnb_away_conf, "derived from model",
            model_prob=(p_away / dnb_denom_a) if dnb_denom_a > 0 else None)

    # ── Double Chance (derived from 1X2) — tightened thresholds ──
    # data_quality and missing_data computed above (before DNB)
    dc_threshold = 0.72 / data_quality  # raise threshold when data is poor

    if p_home + p_draw > dc_threshold:
        dc_prob = (p_home + p_draw) * data_quality
        add("DC", "1X", "Medium-High" if dc_prob > 0.82 else "Medium", "derived from model",
            model_prob=dc_prob)
    if p_away + p_draw > dc_threshold:
        dc_prob = (p_away + p_draw) * data_quality
        add("DC", "X2", "Medium-High" if dc_prob > 0.82 else "Medium", "derived from model",
            model_prob=dc_prob)
    # '12' only in low-draw leagues with strong separation
    if p_home + p_away > 0.86 and p_draw < 0.22:
        dc_prob = (p_home + p_away) * data_quality
        add("DC", "12", "Medium-High" if dc_prob > 0.92 else "Medium", "derived from model",
            model_prob=dc_prob)

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

        # League-reliability damping: unreliable leagues over-predict goals,
        # so cap O/U confidence regardless of the model's certainty.
        if league_reliability < 1.0:
            _max_conf = "Medium" if league_reliability < 0.5 else "Medium-High"
            if CONF_RANK.get(ou_conf, 99) < CONF_RANK[_max_conf]:
                ou_conf = _max_conf

        # Under 3.5 expected-goals gate — prevent overconfident picks when exp goals are marginal
        if thresh == 3.5 and "Under" in ou_pick:
            if exp_total > 3.5:
                ou_conf = "Low"  # Too many expected goals for Under 3.5
            elif exp_total > 3.2:
                if CONF_RANK.get(ou_conf, 99) < CONF_RANK["Medium-High"]:
                    ou_conf = "Medium-High"

        # Cap O/U 1.5 — too many 1-0/0-0 results even with high exp goals
        if thresh == 1.5:
            if CONF_RANK.get(ou_conf, 99) < CONF_RANK["Medium-High"]:
                ou_conf = "Medium-High"

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

        # Forebet O/U % cross-check — adjust confidence when Forebet disagrees
        if thresh == 1.5:
            fb_over = data.get("forebet_over25_pct")  # Forebet gives O25 not O15
            fb_under = data.get("forebet_under25_pct")
        else:
            # Use venue-specific O/U % from stats section as Forebet-style signal
            fb_over = data.get(f"{'home' if 'Over' in ou_pick else 'away'}_over{int(thresh)}_pct")
            fb_under = data.get(f"{'home' if 'Under' in ou_pick else 'away'}_under{int(thresh)}_pct")
        fb_ou_diff = 0.0
        if fb_over is not None and fb_under is not None:
            fb_ou_diff = (fb_over - fb_under) / 100.0 if "Over" in ou_pick else (fb_under - fb_over) / 100.0
            # If Forebet is neutral (40-60 range), cap our confidence
            if abs(fb_ou_diff) < 0.15 and ou_conf in ("Near Certain", "High"):
                ou_conf = "Medium-High"
            # If Forebet strongly agrees (>20% edge), boost
            if fb_ou_diff > 0.20 and ou_conf == "Medium":
                ou_conf = "Medium-High"

        # Venue-specific O/U % cross-check
        home_ou_pct = data.get(f"home_over{int(thresh)}_pct")
        away_ou_pct = data.get(f"away_over{int(thresh)}_pct")
        if home_ou_pct is not None and away_ou_pct is not None:
            combined_ou_avg = (home_ou_pct + away_ou_pct) / 2.0
            if "Over" in ou_pick and combined_ou_avg < 40:
                if CONF_RANK.get(ou_conf, 99) > CONF_RANK["Medium"]:
                    ou_conf = "Medium"
            elif "Under" in ou_pick and combined_ou_avg > (50 if thresh == 3.5 else 60):
                if CONF_RANK.get(ou_conf, 99) > CONF_RANK["Medium"]:
                    ou_conf = "Medium"

        if ou_conf != "Low":
            ou_reason = f"exp goals {exp_total:.1f} model {p_o:.0%}o/{p_u:.0%}u"
            # Add venue goal stats
            hh_gf = data.get("home_home_avg_goals_for")
            aa_gf = data.get("away_away_avg_goals_for")
            if hh_gf is not None and aa_gf is not None:
                ou_reason += f" vG:{hh_gf:.1f}-{aa_gf:.1f}"
            # Append Forebet agreement indicator
            if fb_over is not None and abs(fb_ou_diff) >= 0.15:
                ou_reason += f" fb{'✓' if fb_ou_diff > 0 else '✗'}"
            add("O/U", ou_pick, ou_conf, ou_reason,
                model_prob=p_o if "Over" in ou_pick else p_u)

    # ── BTTS (blended: Poisson + Forebet) ──
    dc_rho = profile.get("dixon_coles_rho", -0.12)
    p_btss_poisson = prob_btts(exp_h, exp_a, rho=dc_rho)

    # Blend with Forebet BTTS probability when available (Forebet BTTS ~88% accurate)
    fb_btts_yes = data.get("home_btts_yes_pct")
    fb_btts_no = data.get("home_btts_no_pct")
    if fb_btts_yes is not None:
        fb_btts_prob = fb_btts_yes / 100.0
        # Forebet 40% weight, Poisson 60% — Poisson is more reliable for BTTS
        p_btss = p_btss_poisson * 0.60 + fb_btts_prob * 0.40
    else:
        p_btss = p_btss_poisson
    p_btn = 1.0 - p_btss

    value_yes = p_btss - 0.5
    value_no = p_btn - 0.5

    # Higher threshold for YES (was 0.08) to reduce false positives
    if value_yes > 0.10 and value_yes >= value_no:
        btss_conf = conv_label(50 + int(value_yes * 80))
        if vol >= 0.25 and btss_conf in ("Near Certain", "High"): btss_conf = "Medium-High"
        elif vol >= 0.15 and btss_conf == "Near Certain": btss_conf = "High"
        if profile.get("avg_goals", 2.8) < 2.5 and btss_conf in ("Near Certain", "High"):
            btss_conf = "Medium-High"
        # League-reliability damping (same logic as O/U)
        if league_reliability < 1.0:
            _max_conf = "Medium" if league_reliability < 0.5 else "Medium-High"
            if CONF_RANK.get(btss_conf, 99) < CONF_RANK[_max_conf]:
                btss_conf = _max_conf
        btss_reason = f"blended {p_btss:.0%}y/{p_btn:.0%}n"
        if fb_btts_yes is not None:
            btss_reason += f" fb{fb_btts_yes}%"
        add("BTTS", "Yes", btss_conf, btss_reason, model_prob=p_btss)
    elif value_no > 0.08:
        btss_conf = conv_label(50 + int(value_no * 80))
        if vol >= 0.25 and btss_conf in ("Near Certain", "High"): btss_conf = "Medium-High"
        elif vol >= 0.15 and btss_conf == "Near Certain": btss_conf = "High"
        # High-scoring league cap — BTTS NO less reliable when avg_goals > 3.0
        if profile.get("avg_goals", 2.8) > 3.0 and btss_conf in ("Near Certain", "High"):
            btss_conf = "Medium-High"
        # League-reliability damping (same logic as O/U)
        if league_reliability < 1.0:
            _max_conf = "Medium" if league_reliability < 0.5 else "Medium-High"
            if CONF_RANK.get(btss_conf, 99) < CONF_RANK[_max_conf]:
                btss_conf = _max_conf
        btss_reason = f"blended {p_btss:.0%}y/{p_btn:.0%}n"
        if fb_btts_no:
            btss_reason += f" fb{fb_btts_no}%"
        add("BTTS", "No", btss_conf, btss_reason, model_prob=p_btn)

    # ── Suppress combined picks (e.g. "1 and NO") when favorite odds are low ──
    for c in candidates:
        pick_str = c.get("pick", "")
        if " and " in pick_str:
            c_odds = _pick_odds(c["market"], pick_str)
            if c_odds and c_odds < 1.30:
                c["confidence"] = "Low"

    # ── Rank candidates and pick primary ──
    # Coverage: how many 1X2 outcomes does each pick cover?
    # DC 1X covers Home+Draw, DC X2 covers Away+Draw, DC 12 covers Home+Away
    # O/U covers all outcomes (it's goal-based, not result-based)
    # DNB covers 1 outcome (Home or Away)
    # 1X2 covers 1 outcome
    COVERAGE = {
        ("DC", "1X"): 2, ("DC", "X2"): 2, ("DC", "12"): 2,
        ("O/U", "Over 1.5"): 3, ("O/U", "Under 1.5"): 3,
        ("O/U", "Over 2.5"): 3, ("O/U", "Under 2.5"): 3,
        ("O/U", "Over 3.5"): 3, ("O/U", "Under 3.5"): 3,
        ("BTTS", "Yes"): 3, ("BTTS", "No"): 3,
        ("DNB", "Home"): 1, ("DNB", "Away"): 1,
        ("1X2", "Home win"): 1, ("1X2", "Draw"): 1, ("1X2", "Away win"): 1,
    }
    for c in candidates:
        c["coverage"] = COVERAGE.get((c["market"], c["pick"]), 1)

    # ── Holistic synthesis ──
    # Fuse every match signal (model probs, market edge, component agreement,
    # uncertainty, draw tendency, coverage) into one decision value per
    # candidate, then re-rank. This replaces the old isolated confidence sort
    # so picks are chosen on the whole picture, not one threshold at a time.
    try:
        from synthesis import (
            synthesize, build_synthesis_rationale, context_from_pred,
        )
        _synth_ctx = context_from_pred(
            pred={},  # not used; fields passed explicitly below
            data=data, vol=vol, form_signal=fsig,
            trans_signal=trans_signal, draw_tendency=_draw_tendency,
            draw_factors=len(_draw_factors), top_pick=top_pick, margin=margin,
            league_reliability=league_reliability,
            ml_dir=None,
        )
        # inject the computed 1X2 probs + exp goals directly (pred dict is empty)
        _synth_ctx.p_home, _synth_ctx.p_draw, _synth_ctx.p_away = p_home, p_draw, p_away
        _synth_ctx.exp_h, _synth_ctx.exp_a = exp_h, exp_a
        candidates = synthesize(_synth_ctx, candidates)
        _synth_rationale = build_synthesis_rationale(_synth_ctx, candidates, candidates[0])
        from synthesis import component_agreement
        _synth_consensus, _synth_n_sources = component_agreement(_synth_ctx)
        method_parts.append("synthesis")
    except Exception as _syn_err:
        # Fallback: keep the legacy score so the model still produces picks
        for c in candidates:
            cov_bonus = (c["coverage"] - 1) * 0.5
            if _draw_tendency and c["market"] != "1X2":
                cov_bonus *= 0.5
            c["score"] = c["rank"] - cov_bonus - (c.get("model_prob") or 0) * 2
        candidates.sort(key=lambda c: c["score"])
        _synth_rationale = f"(synthesis unavailable: {_syn_err})"

    non_show = [c for c in candidates if not c.get('_always_show')]
    primary = non_show[0] if non_show else candidates[0] if candidates else {"market": "1X2", "pick": "Draw", "confidence": "Low"}

    # Backup pick: best alternative if primary fails
    backup = None
    # If Draw is close to primary 1X2 pick, make it the backup
    if primary["market"] == "1X2" and _draw_tendency:
        for c in non_show[1:]:
            if c["market"] == "1X2" and c["pick"] == "Draw":
                backup = c
                break
    # Otherwise, best alternative from different market
    if not backup:
        for c in non_show[1:]:
            if c["market"] != primary["market"] or c["pick"] != primary["pick"]:
                backup = c
                break

    # ── Build reasoning ──
    for c in candidates[:6]:  # top 6
        line = f"{c['market']}: {c['pick']} ({c['confidence']})"
        if c.get("reason"):
            line += f" — {c['reason']}"
        if c.get("decision_value") is not None:
            line += f"  [synth {c['decision_value']:.3f}]"
        reasoning.append(line)

    # ── Holistic synthesis verdict ──
    if _synth_rationale:
        reasoning.append("")
        reasoning.append("── Synthesis ──")
        reasoning.append(_synth_rationale)

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

    # ── Data quality warnings ──
    hf = data.get("home_form", "") or ""
    af = data.get("away_form", "") or ""
    hf_len = sum(1 for c in hf if c in "WDL")
    af_len = sum(1 for c in af if c in "WDL")
    hm = data.get("h2h_matches", 0) or 0
    h2h_avg = data.get("h2h_avg_total_goals") or 0
    h_gf = data.get("home_avg_goals_for")
    a_gf = data.get("away_avg_goals_for")
    warnings = []
    if hf_len < 3:
        warnings.append(f"Home form: only {hf_len} games")
    if af_len < 3:
        warnings.append(f"Away form: only {af_len} games")
    if hm < 3:
        warnings.append(f"H2H: only {hm} meetings")
    elif h2h_avg and h2h_avg < 1.5:
        warnings.append(f"H2H avg {h2h_avg:.1f} goals — low scoring history")
    elif h2h_avg and h2h_avg > 4.5:
        warnings.append(f"H2H avg {h2h_avg:.1f} goals — high scoring history")
    if not h_gf or not a_gf:
        warnings.append("No attack/defense data")
    if vol >= 0.25:
        warnings.append(f"High volatility ({vol:.2f})")
    if not data.get("home_pos") or not data.get("away_pos"):
        warnings.append("No league position data")

    # New stat-based warnings
    hh_gf = data.get("home_home_avg_goals_for")
    aa_gf = data.get("away_away_avg_goals_for")
    if hh_gf is None:
        warnings.append("No home-venue attack data")
    if aa_gf is None:
        warnings.append("No away-venue attack data")
    h_ou15 = data.get("home_over15_pct")
    a_ou15 = data.get("away_over15_pct")
    if h_ou15 is not None and a_ou15 is not None:
        avg_ou15 = (h_ou15 + a_ou15) / 2.0
        if avg_ou15 > 80:
            warnings.append(f"Very high O15 ({avg_ou15:.0f}%)")
        elif avg_ou15 < 30:
            warnings.append(f"Very low O15 ({avg_ou15:.0f}%)")
    h_sot = data.get("home_shots_ontarget_pct")
    if h_sot is None:
        warnings.append("No shots data")

    # ── Component availability warnings ──
    if use_ml and not ml_model:
        warnings.append("ML model unavailable — using Poisson only")
    elif not use_ml:
        warnings.append("ML disabled (--no-ml)")
    fb_h = data.get("forebet_home_pct")
    fb_d = data.get("forebet_draw_pct")
    fb_a = data.get("forebet_away_pct")
    if not fb_h and not fb_d and not fb_a:
        warnings.append("No Forebet data for this match")

    # ── Transitive common-opponent display ──
    if _trans_analysis and _trans_analysis["reasoning"]:
        for r in _trans_analysis["reasoning"]:
            reasoning.append(r)
        if trans_adjusted:
            tdir = "home" if _trans_analysis["signal"] < 0 else "away"
            tconf = _trans_analysis["confidence"]
            tsig = _trans_analysis["signal"]
            reasoning.append(
                f"†Trans: {tdir} bias ({tconf}, sig {tsig:+.2f})"
            )
        if draw_adjusted:
            ds = _trans_analysis.get("draw_signal", 0.0)
            reasoning.append(f"†Trans: draw tendency ({ds:.2f})")

    # ── Form analysis reasoning output ──
    if form_analysis["reasoning"]:
        reasoning.append("")
        reasoning.append("── Form Analysis ──")
        for r in form_analysis["reasoning"]:
            reasoning.append(r)

    # ── Combined Synthesis + Form verdict (standout line) ──
    # Brings together the holistic synthesis direction and the recent-form
    # signal into one plain-language sentence, so the two lenses are read as a
    # single conclusion rather than separate sections.
    try:
        _form_dir = "home" if fsig < -0.05 else "away" if fsig > 0.05 else "balanced"
        _form_strength = "strongly" if abs(fsig) >= 0.30 else "moderately" if abs(fsig) >= 0.12 else "slightly"
        _syn_dir = "home" if _synth_consensus < -0.05 else "away" if _synth_consensus > 0.05 else "balanced"

        # The decisive conclusion: the actual top-ranked pick from the holistic
        # fusion, plus the single strongest factor that drove it.
        _top = primary
        _top_pick = f"{_top['market']} {_top['pick']}"
        _top_dv = _top.get("decision_value")
        _top_comp = _top.get("components") or {}
        _top_edge = _top_comp.get("edge")
        if _top_edge is not None and _top_edge > 0.02:
            _driver = f"model edge vs market +{_top_edge:.0%}"
        elif _top["market"] == "O/U":
            _driver = f"expected total {exp_h+exp_a:.1f} goals (model {_top_comp.get('prob'):.0%})"
        else:
            _driver = f"model probability {_top_comp.get('prob'):.0%}"

        # Opening clause: how synthesis and form relate
        if _form_dir == _syn_dir and _form_dir != "balanced":
            _lead = (f"Synthesis and recent form align on the {_form_dir} side "
                     f"({_form_strength} on form, consensus {_synth_consensus:+.2f}); ")
        elif _form_dir == "balanced":
            _lead = (f"Recent form is balanced, so the call rests on the holistic synthesis "
                     f"(consensus {_synth_consensus:+.2f}); ")
        elif _syn_dir == "balanced":
            _lead = (f"Synthesis is inconclusive, so recent form ({_form_dir}, {_form_strength}) "
                     f"decides; ")
        else:
            _lead = (f"Synthesis favours {_syn_dir} while recent form favours {_form_dir} "
                     f"— the split keeps conviction modest; ")

        _combo = (f"⟁SYNTHFORM⟁ {_lead}overall the model settles on "
                  f"{_top_pick} ({_top['confidence']}), driven by {_driver} "
                  f"(decision value {_top_dv:.2f}).")
        reasoning.append("")
        reasoning.append(_combo)
    except Exception:
        pass

    # ── Kelly Criterion stake sizing (improvement 9) ──
    kelly_stake = 0.0
    model_prob = primary.get("model_prob")
    implied_prob = primary.get("implied_prob")
    odds_val = _pick_odds(primary["market"], primary["pick"])
    if model_prob and implied_prob and implied_prob > 0 and odds_val and odds_val > 1.0:
        edge = (model_prob / implied_prob) - 1.0
        if edge > 0:
            kelly_fraction = edge / (odds_val - 1) if odds_val > 1 else 0
            kelly_stake = round(min(kelly_fraction * 0.25, 0.05), 4)  # Quarter-Kelly, max 5%

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
        "_method": "+".join(method_parts) if method_parts else "unknown",
        "_kelly_stake": kelly_stake,
        "_model_prob": model_prob,
        "_implied_prob": implied_prob,
        "_odds": odds_val,
        "_poisson_probs": (p_home, p_draw, p_away),
        "_synthesis_rationale": _synth_rationale if "_synth_rationale" in dir() else None,
        "_synthesis_ranked": [
            {"market": c["market"], "pick": c["pick"], "confidence": c["confidence"],
             "decision_value": c.get("decision_value"), "components": c.get("components")}
            for c in candidates[:6]
        ],
        "_warnings": warnings,
        "_backup": {"pick": backup["pick"], "market": backup["market"],
                    "confidence": backup["confidence"], "coverage": backup["coverage"]} if backup else None,
    }


# ─────────────────────────────────────────────
# Prediction runner
# ─────────────────────────────────────────────

def log(msg, end="\n"):
    """Print progress to stderr so stdout stays clean for JSON."""
    print(msg, end=end, file=sys.stderr, flush=True)


def _write_html(results, all_urls, compare_forebet, high_only):
    """Generate an HTML report of predictions and update index."""
    import webbrowser
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    now_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    filtered = [r for r in results if r["confidence"] in ("Near Certain", "High")] if high_only else results

    # ── Calculate per-report accuracy ──
    report_correct = 0
    report_total = 0
    for r in filtered:
        if r.get("actual_home_goals") is not None and r.get("actual_away_goals") is not None:
            if r.get("correct_pick") is True:
                report_correct += 1
            report_total += 1

    conf_counts = {}
    for r in filtered:
        conf_counts[r["confidence"]] = conf_counts.get(r["confidence"], 0) + 1

    agreements = 0
    total_fb = 0
    if compare_forebet:
        for r in filtered:
            if r.get("forebet"):
                picks_12 = [p for p in (r.get("all_picks") or []) if p["market"] == "1X2"]
                our_12 = picks_12[0]["pick"] if picks_12 else r["pick"]
                fb_val = {"Home win": "1", "Draw": "X", "Away win": "2"}.get(our_12, "")
                if fb_val and r["forebet"] == fb_val:
                    agreements += 1
                total_fb += 1

    EXP_COLOR = "#c4b5fd"  # violet — distinct from result(blue)/correct(green)/incorrect(red)

    def _c(val):
        m = {"Near Certain": "#22c55e", "High": "#3b82f6", "Medium-High": "#eab308", "Medium": "#f97316", "Low": "#ef4444"}
        return m.get(val, "#888")

    def _star(conf):
        return {3: "★★★", 2: "★★☆", 1: "★☆☆"}.get({"Near Certain": 3, "High": 2, "Medium-High": 1}.get(conf, 0), "")

    def _highlight_exp(text: str) -> str:
        """Colorize expected-goals mentions so they aren't confused with real scores."""
        text = re.sub(r'exp goals (\d+\.\d+)',
                      rf'<span style="color:{EXP_COLOR};font-weight:700">exp goals \1</span>', text)
        text = re.sub(r'exp (\d+\.\d+-\d+\.\d+)',
                      rf'<span style="color:{EXP_COLOR};font-weight:700">exp \1</span>', text)
        return text

    def _venue_stats_html(r):
        parts = []
        hh_gf = r.get("home_home_avg_goals_for")
        hh_ga = r.get("home_home_avg_goals_against")
        if hh_gf is not None:
            parts.append(f"Home(H): {hh_gf:.1f}GF/{hh_ga:.1f}GA")
        aa_gf = r.get("away_away_avg_goals_for")
        aa_ga = r.get("away_away_avg_goals_against")
        if aa_gf is not None:
            parts.append(f"Away(A): {aa_gf:.1f}GF/{aa_ga:.1f}GA")
        ou15 = r.get("home_over15_pct")
        if ou15 is not None:
            parts.append(f"O15: {ou15}% (Forebet)")
        btts_h = r.get("home_btts_yes_pct")
        if btts_h is not None:
            parts.append(f"BTTS: {btts_h}% (Forebet)")
        sot_h = r.get("home_shots_ontarget_pct")
        ts_h = r.get("home_total_shots_pg")
        if sot_h is not None and ts_h is not None:
            sot_est = round(ts_h * (sot_h / 100.0) * 0.32, 1)
            parts.append(f"SoT: {sot_h}% ({sot_est:.1f} xG)")
        cs_h = r.get("home_clean_sheets_pct")
        if cs_h is not None:
            parts.append(f"CS: {cs_h}%")
        if parts:
            return '<p class="venue-stats">' + " &middot; ".join(parts) + "</p>"
        return ""

    rows = []
    for r in filtered:
        eh, ea = r.get("_exp_goals", (None, None))
        exp_str = f"{eh:.1f}-{ea:.1f}" if eh is not None else "—"
        hf = r.get("home_form", "")
        af = r.get("away_form", "")
        picks_rows = ""
        has_result_h = r.get("actual_home_goals") is not None and r.get("actual_away_goals") is not None
        for p in r.get("all_picks") or []:
            mp = p.get("model_prob")
            mp_s = f"{mp:.0%}" if mp else ""
            vr = p.get("value_ratio")
            vr_s = f" ({vr:.2f})" if vr else ""

            result_cell = ""
            if has_result_h:
                hg_h, ag_h = r["actual_home_goals"], r["actual_away_goals"]
                total_g = hg_h + ag_h
                actual_out = r.get("actual_outcome", "")
                pmk, ppk = p["market"], p["pick"]
                pc = None
                if pmk == "1X2":
                    pc = (ppk == actual_out)
                elif pmk == "O/U":
                    if "Over" in ppk:
                        pc = (total_g > float(ppk.split()[-1]))
                    elif "Under" in ppk:
                        pc = (total_g <= float(ppk.split()[-1]))
                elif pmk == "BTTS":
                    both = hg_h > 0 and ag_h > 0
                    pc = (ppk == "Yes" and both) or (ppk == "No" and not both)
                elif pmk == "DNB":
                    if actual_out == "Draw":
                        pc = None  # Push — stake returned
                    else:
                        pc = (ppk == "Home" and actual_out == "Home win") or (ppk == "Away" and actual_out == "Away win")
                elif pmk == "DC":
                    if ppk == "1X": pc = actual_out in ("Home win", "Draw")
                    elif ppk == "X2": pc = actual_out in ("Away win", "Draw")
                    elif ppk == "12": pc = actual_out in ("Home win", "Away win")
                if pc is True:
                    result_cell = '<td style="color:#22c55e;font-weight:700">✓</td>'
                elif pc is False:
                    result_cell = '<td style="color:#ef4444;font-weight:700">✗</td>'
                elif pmk == "DNB" and actual_out == "Draw":
                    result_cell = '<td style="color:#94a3b8;font-weight:700">push</td>'
                else:
                    result_cell = '<td style="color:#64748b">—</td>'

            picks_rows += f"<tr><td>{p['market']}</td><td>{p['pick']}</td><td>{mp_s}</td><td style='color:{_c(p['confidence'])}'>{p['confidence']}</td><td>{vr_s}</td>{result_cell}</tr>\n"

        reason_html = ""
        if r.get("reasoning"):
            for reason in r["reasoning"]:
                if reason.startswith("⟁SYNTHFORM⟁"):
                    _body = _highlight_exp(reason.replace("⟁SYNTHFORM⟁", "", 1).strip())
                    reason_html += f'<li class="synthform-line">✦ {_body}</li>\n'
                else:
                    reason_html += f"<li>{_highlight_exp(reason)}</li>\n"
            reason_html = f'<div class="reasoning"><strong>Reasoning</strong><ul>{reason_html}</ul></div>'

        kelly_tag = f" &middot; Kelly: {r.get('kelly_stake', 0)*100:.1f}%" if r.get('kelly_stake', 0) > 0 else ""
        method_tag = f" &middot; {r.get('method', '')}" if r.get('method') else ""

        result_html = ""
        if r.get("actual_home_goals") is not None and r.get("actual_away_goals") is not None:
            hg, ag = r["actual_home_goals"], r["actual_away_goals"]
            outcome = r.get("actual_outcome", "")
            ht_h, ht_a = r.get("ht_home_goals"), r.get("ht_away_goals")
            ht_tag = f"  HT: {ht_h}-{ht_a}" if ht_h is not None and ht_a is not None else ""
            if r.get("correct_pick") is True:
                verdict = '<span style="color:#22c55e;font-weight:700">Correct!</span>'
            elif r.get("correct_pick") is False:
                our_12 = [p for p in (r.get("all_picks") or []) if p["market"] == "1X2"]
                our_main = our_12[0]["pick"] if our_12 else r["pick"]
                verdict = f'<span style="color:#ef4444;font-weight:700">Incorrect</span> (picked {our_main})'
            else:
                verdict = ""
            result_html = f'<div class="pick-line" style="color:#60a5fa;font-weight:700">RESULT: {hg} - {ag} ({outcome}){ht_tag}  {verdict}</div>'

        exp_home = f"{eh:.2f}" if eh is not None else ""
        exp_away = f"{ea:.2f}" if ea is not None else ""
        exp_total = f"{eh + ea:.2f}" if eh is not None and ea is not None else ""
        rows.append(f"""<div class="card" data-match-id="{r.get('match_id', '')}" data-exp-home="{exp_home}" data-exp-away="{exp_away}" data-exp-total="{exp_total}" data-market="{r['market']}" data-pick="{r['pick']}" style="border-left: 4px solid {_c(r['confidence'])};">
<div class="card-header">
  <span class="teams">{r['home']} vs {r['away']}</span>
  <span class="conf-badge" style="background:{_c(r['confidence'])}">{_star(r['confidence'])} {r['confidence']}</span>
</div>
<div class="card-meta">{r.get('league', '')} &middot; {r.get('date', '')} &middot; <a href="{r['url']}">Forebet</a>{method_tag}</div>
{"".join(f'<div class="league-warning" style="background:#7f1d1d;color:#fca5a5;padding:4px 10px;border-radius:4px;font-size:0.78rem;margin-bottom:8px;display:inline-block;">⚠ {r["league_difficulty"]["reason"]}</div>' if r.get("league_difficulty", {}).get("level") == "hard" and r.get("league_difficulty", {}).get("matches", 0) >= 5 else [])}
<div class="card-body">
  {result_html}
  <div class="pick-line"><strong>{r['pick']}</strong> ({r['market']}) &middot; Score lean: {r['score_lean'] or '—'} &middot; Exp: <span style="color:{EXP_COLOR};font-weight:700">{exp_str}</span>{kelly_tag}</div>
  <table>
    <tr><th>Home</th><td>Pos {r.get('home_pos', '—')}</td><td>Form {hf or '—'}</td><td>{r.get('odds_home', '—')}</td></tr>
    <tr><th>Draw</th><td></td><td></td><td>{r.get('odds_draw', '—')}</td></tr>
    <tr><th>Away</th><td>Pos {r.get('away_pos', '—')}</td><td>Form {af or '—'}</td><td>{r.get('odds_away', '—')}</td></tr>
  </table>
  <table><tr><th>O/U 2.5</th><td>{r.get('odds_over25', '—')}/{r.get('odds_under25', '—')}</td><th>BTTS</th><td>{r.get('odds_btts_yes', '—')}/{r.get('odds_btts_no', '—')}</td></tr></table>
  {"<p>H2H: " + str(r.get('h2h_home_wins', 0)) + "W-" + str(r.get('h2h_draws', 0)) + "D-" + str(r.get('h2h_away_wins', 0)) + "L &ndash; GF/GA: " + str(r.get('h2h_goals_for', 0)) + "/" + str(r.get('h2h_goals_against', 0)) + " &ndash; avg " + str(r.get('h2h_avg_total_goals', 0)) + " goals (" + str(r.get('h2h_matches', 0)) + " matches)</p>" if r.get('h2h_matches', 0) >= 3 else ""}
  {_venue_stats_html(r)}
  {reason_html}
  {("<table><tr><th>Market</th><th>Pick</th><th>Prob</th><th>Conf</th><th>Value</th>" + ("<th>Result</th>" if has_result_h else "") + "</tr>" + picks_rows + "</table>") if picks_rows else ""}
  {("<div style='margin-top:8px;padding:6px 10px;background:rgba(99,102,241,0.1);border-radius:6px;font-size:0.88em;'><strong>Best alternative:</strong> <span style='color:" + _c(r.get('_backup', {}).get('confidence', 'Low')) + "'>" + r['_backup']['market'] + ": " + r['_backup']['pick'] + " (" + r['_backup']['confidence'] + ")</span> — covers " + str(r['_backup']['coverage']) + " outcomes</div>") if r.get('_backup') else ""}
</div>
</div>""")

    # ── Get market accuracy data for charts ──
    market_accuracy_data = get_market_accuracy()
    market_labels = json.dumps([m["market"] for m in market_accuracy_data if m["market"]])
    market_totals = json.dumps([m["total"] for m in market_accuracy_data if m["market"]])
    market_correct = json.dumps([m["correct"] for m in market_accuracy_data if m["market"]])
    market_accuracy = json.dumps([m["accuracy"] for m in market_accuracy_data if m["market"]])

    # ── Get accuracy trend data for charts ──
    trend_data = {}
    for market in ["1X2", "O/U", "BTTS"]:
        history = get_market_accuracy_history(market, window=20)
        if history:
            trend_data[market] = {
                "dates": [h["date"][:10] for h in history[-30:]],
                "accuracy": [h["accuracy"] for h in history[-30:]]
            }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Predictions — {now}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#0f172a; color:#e2e8f0; padding:20px; }}
h1 {{ font-size:1.4rem; margin-bottom:4px; }}
h2 {{ font-size:1.1rem; margin:16px 0 8px; color:#94a3b8; }}
.sub {{ color:#94a3b8; font-size:0.85rem; margin-bottom:20px; }}
.stats {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:24px; }}
.stat {{ background:#1e293b; padding:10px 16px; border-radius:8px; font-size:0.85rem; }}
.stat span {{ font-weight:700; font-size:1.1rem; }}
.card {{ background:#1e293b; border-radius:8px; padding:16px; margin-bottom:12px; }}
.card-header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:4px; }}
.teams {{ font-size:1.1rem; font-weight:700; }}
.conf-badge {{ font-size:0.75rem; padding:2px 8px; border-radius:4px; color:#fff; font-weight:600; }}
.card-meta {{ color:#94a3b8; font-size:0.8rem; margin-bottom:10px; }}
.card-body {{ font-size:0.85rem; line-height:1.6; }}
.pick-line {{ font-size:1rem; margin-bottom:8px; }}
table {{ width:100%; border-collapse:collapse; margin:6px 0; }}
th, td {{ text-align:left; padding:2px 8px 2px 0; }}
th {{ color:#94a3b8; font-weight:500; width:60px; }}
 details {{ margin-top:6px; }}
  .reasoning {{ margin-top:6px; }}
  .reasoning strong {{ color:#94a3b8; font-weight:500; }}
  .synthform-line {{ list-style:none; margin:10px 0 4px 0 !important; padding:9px 12px;
    background:linear-gradient(90deg, rgba(250,204,21,0.16), rgba(250,204,21,0.04));
    border-left:4px solid #facc15; border-radius:6px; color:#fde68a; font-weight:600;
    font-size:0.92rem; box-shadow:0 1px 4px rgba(0,0,0,0.25); }}
 summary {{ cursor:pointer; color:#60a5fa; font-weight:500; }}
ul {{ margin:4px 0 0 18px; color:#94a3b8; }}
a {{ color:#60a5fa; }}
.chart-container {{ background:#1e293b; border-radius:8px; padding:16px; margin-bottom:12px; }}
canvas {{ max-height:300px; }}
select {{ cursor:pointer; }}
</style>
</head>
<body>
<h1>⚽ Predictions Report</h1>
<p class="sub">Generated {now} &middot; {len(filtered)} picks ({(len(filtered)/len(all_urls)*100) if filtered else 0:.0f}% pick rate)</p>
<div class="stats">
<div class="stat">Total <span>{len(filtered)}</span></div>
{"".join(f'<div class="stat" style="border-left:3px solid {_c(c)}">{c} <span style="color:{_c(c)}">{n}</span></div>' for c, n in conf_counts.items())}
{f'<div class="stat">Forebet 1X2 agreement <span>{agreements}/{total_fb} ({100*agreements//total_fb if total_fb else 0}%)</span></div>' if compare_forebet and total_fb else ""}
</div>

<h2>Model Accuracy</h2>
<div class="chart-container">
  <canvas id="marketChart"></canvas>
</div>

<div class="chart-container">
  <canvas id="trendChart"></canvas>
</div>

<h2>Predictions <span id="matchCount" style="font-size:0.8rem;color:#94a3b8;font-weight:400">{len(filtered)} / {len(filtered)}</span></h2>
<div style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:14px;align-items:center;">
<input type="text" id="matchSearch" placeholder="Search teams…" style="flex:1;min-width:180px;max-width:400px;padding:8px 12px;border-radius:6px;border:1px solid #334155;background:#1e293b;color:#e2e8f0;font-size:0.9rem;outline:none;">
<select id="filterExpTotal" style="padding:8px 12px;border-radius:6px;border:1px solid #334155;background:#1e293b;color:#e2e8f0;font-size:0.9rem;outline:none;">
  <option value="">Exp Goals: Any</option>
  <option value="2.0">Under 2.0</option>
  <option value="2.5">Under 2.5</option>
  <option value="3.0">Under 3.0</option>
  <option value="3.5">Under 3.5</option>
  <option value="4.0">Under 4.0</option>
</select>
<select id="filterOutcome" style="padding:8px 12px;border-radius:6px;border:1px solid #334155;background:#1e293b;color:#e2e8f0;font-size:0.9rem;outline:none;">
  <option value="">1X2: Any</option>
  <option value="Home win">Home Win</option>
  <option value="Draw">Draw</option>
  <option value="Away win">Away Win</option>
</select>
<select id="filterConf" style="padding:8px 12px;border-radius:6px;border:1px solid #334155;background:#1e293b;color:#e2e8f0;font-size:0.9rem;outline:none;">
  <option value="">Confidence: Any</option>
  <option value="Near Certain">Near Certain</option>
  <option value="High">High</option>
  <option value="Medium-High">Medium-High</option>
  <option value="Medium">Medium</option>
  <option value="Low">Low</option>
</select>
<select id="filterMarket" style="padding:8px 12px;border-radius:6px;border:1px solid #334155;background:#1e293b;color:#e2e8f0;font-size:0.9rem;outline:none;">
  <option value="">Market: Any</option>
  <option value="1X2">1X2</option>
  <option value="O/U">O/U</option>
  <option value="BTTS">BTTS</option>
  <option value="DC">DC</option>
  <option value="DNB">DNB</option>
</select>
</div>
{"".join(rows)}

<script>
const cards = document.querySelectorAll('.card');
const searchInput = document.getElementById('matchSearch');
const filterExp = document.getElementById('filterExpTotal');
const filterOutcome = document.getElementById('filterOutcome');
const filterConf = document.getElementById('filterConf');
const filterMarket = document.getElementById('filterMarket');
const matchCount = document.getElementById('matchCount');

function applyFilters() {{
  const q = (searchInput.value || '').toLowerCase().trim();
  const maxExp = filterExp.value ? parseFloat(filterExp.value) : null;
  const outcome = filterOutcome.value;
  const conf = filterConf.value;
  const market = filterMarket.value;
  let visible = 0;
  cards.forEach(card => {{
    let show = true;
    // Team search
    if (q) {{
      const teams = card.querySelector('.teams');
      if (!teams || !q.split(/\\s+/).every(w => teams.textContent.toLowerCase().includes(w))) show = false;
    }}
    // Exp goals filter (total expected goals)
    if (show && maxExp !== null) {{
      const t = parseFloat(card.dataset.expTotal);
      if (isNaN(t) || t >= maxExp) show = false;
    }}
    // 1X2 outcome filter
    if (show && outcome) {{
      if (card.dataset.pick !== outcome) show = false;
    }}
    // Confidence filter
    if (show && conf) {{
      const badge = card.querySelector('.conf-badge');
      if (!badge || !badge.textContent.includes(conf)) show = false;
    }}
    // Market filter
    if (show && market) {{
      if (card.dataset.market !== market) show = false;
    }}
    card.style.display = show ? '' : 'none';
    if (show) visible++;
  }});
  if (matchCount) matchCount.textContent = visible + ' / ' + cards.length;
}}

[searchInput, filterExp, filterOutcome, filterConf, filterMarket].forEach(el => {{
  if (el) el.addEventListener(el.tagName === 'INPUT' ? 'input' : 'change', applyFilters);
}});
</script>

<script>
const marketCtx = document.getElementById('marketChart').getContext('2d');
new Chart(marketCtx, {{
  type: 'bar',
  data: {{
    labels: {market_labels},
    datasets: [{{
      label: 'Total Picks',
      data: {market_totals},
      backgroundColor: 'rgba(59, 130, 246, 0.5)',
      borderColor: 'rgba(59, 130, 246, 1)',
      borderWidth: 1
    }}, {{
      label: 'Correct',
      data: {market_correct},
      backgroundColor: 'rgba(34, 197, 94, 0.5)',
      borderColor: 'rgba(34, 197, 94, 1)',
      borderWidth: 1
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{
      title: {{ display: true, text: 'Market Accuracy', color: '#e2e8f0' }},
      legend: {{ labels: {{ color: '#94a3b8' }} }}
    }},
    scales: {{
      x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }},
      y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }}
    }}
  }}
}});

const trendCtx = document.getElementById('trendChart').getContext('2d');
const trendDatasets = [];
{"".join(f'''
trendDatasets.push({{
  label: '{m}',
  data: {json.dumps(trend_data[m]["accuracy"]) if m in trend_data else "[]"},
  borderColor: '{"#22c55e" if m == "1X2" else "#3b82f6" if m == "O/U" else "#eab308"}',
  tension: 0.1,
  fill: false
}});''' for m in ["1X2", "O/U", "BTTS"] if m in trend_data)}

new Chart(trendCtx, {{
  type: 'line',
  data: {{
    labels: {json.dumps(trend_data["1X2"]["dates"]) if "1X2" in trend_data else "[]"},
    datasets: trendDatasets
  }},
  options: {{
    responsive: true,
    plugins: {{
      title: {{ display: true, text: 'Accuracy Trend (Rolling 20)', color: '#e2e8f0' }},
      legend: {{ labels: {{ color: '#94a3b8' }} }}
    }},
    scales: {{
      x: {{ ticks: {{ color: '#94a3b8', maxTicksLimit: 10 }}, grid: {{ color: '#334155' }} }},
      y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }}, min: 0, max: 100 }}
    }}
  }}
}});
</script>
</body>
</html>"""

    # ── Save numbered report (dedupe by match id) ──
    pred_dir = Path("predictions")
    pred_dir.mkdir(exist_ok=True)

    # Build the set of match ids for this run. Forebet match URLs end in a
    # numeric id (e.g. ...-2428062); use that as the stable unique identifier
    # so re-running the same URL(s) reuses the existing report instead of
    # creating a duplicate.
    import re as _re
    current_ids = set()
    for r in (results or []):
        mid = r.get("match_id")
        if mid:
            current_ids.add(str(mid))
    # Fallback: derive id from the trailing digits of each URL
    for u in (all_urls or []):
        m = _re.search(r"/(\d+)(?:[/?]|$)", u)
        if m:
            current_ids.add(m.group(1))

    # Find next available number
    existing = sorted(pred_dir.glob("*.html"))
    existing_nums = []
    for f in existing:
        if f.name == "index.html":
            continue
        try:
            existing_nums.append(int(f.stem))
        except ValueError:
            pass

    # Reuse an existing report that already contains the exact same match set
    report_path = None
    if current_ids:
        for f in existing:
            if f.name == "index.html":
                continue
            try:
                int(f.stem)
            except ValueError:
                continue
            content = f.read_text(encoding="utf-8", errors="ignore")
            file_ids = set(_re.findall(r'data-match-id="(\d+)"', content))
            if current_ids and file_ids == current_ids:
                report_path = f
                break

    if report_path is None:
        next_num = max(existing_nums, default=0) + 1
        report_path = pred_dir / f"{next_num:03d}.html"

    report_path.write_text(html)
    if report_path.stem in {str(n) for n in existing_nums}:
        log(f"HTML report (reused): {report_path.resolve()}")
    else:
        log(f"HTML report: {report_path.resolve()}")

    # ── Update index.html ──
    _update_index(pred_dir, now)

    # ── Auto-open in browser ──
    index_path = pred_dir / "index.html"
    webbrowser.open(str(index_path.resolve()))

    return report_path


def _update_index(pred_dir: Path, current_time: str):
    """Generate/update index.html with links to all reports."""
    reports = []
    for f in sorted(pred_dir.glob("*.html"), reverse=True):
        if f.name == "index.html":
            continue
        try:
            num = int(f.stem)
        except ValueError:
            continue

        # Extract basic info from the HTML file
        content = f.read_text()
        # Extract title (match names)
        title_match = re.search(r'<title>(.*?)</title>', content)
        title = title_match.group(1) if title_match else f"Report {num}"

        # Extract match info from first card
        teams_match = re.search(r'class="teams">(.*?) vs (.*?)</span>', content)
        home = teams_match.group(1) if teams_match else "?"
        away = teams_match.group(2) if teams_match else "?"

        # Extract date from card-meta
        meta_match = re.search(r'class="card-meta">(.*?)</div>', content)
        meta = meta_match.group(1) if meta_match else ""
        date_str = ""
        if " · " in meta:
            parts = meta.split(" · ")
            date_str = parts[1] if len(parts) > 1 else ""

        # Extract result if present
        result_match = re.search(r'RESULT: (\d+) - (\d+) \((.*?)\)', content)
        score = f"{result_match.group(1)}-{result_match.group(2)}" if result_match else "—"
        outcome = result_match.group(3) if result_match else ""

        # Extract verdict
        verdict_match = re.search(r'(Correct!|Incorrect)', content)
        verdict = verdict_match.group(1) if verdict_match else "—"

        # Extract accuracy from picks table (only if match has result)
        if result_match:
            correct_count = content.count('color:#22c55e;font-weight:700">✓</td>')
            incorrect_count = content.count('color:#ef4444;font-weight:700">✗</td>')
            total_evaluated = correct_count + incorrect_count
            accuracy = f"{correct_count}/{total_evaluated}" if total_evaluated > 0 else "—"
        else:
            accuracy = "Pending"

        reports.append({
            'num': num,
            'home': home,
            'away': away,
            'date': date_str,
            'score': score,
            'verdict': verdict,
            'accuracy': accuracy,
            'filename': f.name,
        })

    # Keep only last 100
    reports = reports[:100]

    # Generate index HTML
    index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Predictions Index</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#0f172a; color:#e2e8f0; padding:20px; }}
h1 {{ font-size:1.4rem; margin-bottom:4px; }}
.sub {{ color:#94a3b8; font-size:0.85rem; margin-bottom:20px; }}
table {{ width:100%; border-collapse:collapse; margin:6px 0; }}
th, td {{ text-align:left; padding:8px 12px; }}
th {{ color:#94a3b8; font-weight:500; border-bottom:1px solid #334155; }}
tr:hover {{ background:#1e293b; }}
a {{ color:#60a5fa; text-decoration:none; }}
a:hover {{ text-decoration:underline; }}
.correct {{ color:#22c55e; }}
.incorrect {{ color:#ef4444; }}
</style>
</head>
<body>
<h1>⚽ Predictions Index</h1>
<p class="sub">Last updated: {current_time} &middot; {len(reports)} reports</p>
<table>
<tr><th>#</th><th>Match</th><th>Date</th><th>Score</th><th>Accuracy</th><th>Result</th><th>Link</th></tr>
"""
    for r in reports:
        verdict_class = "correct" if r['verdict'] == "Correct!" else "incorrect" if r['verdict'] == "Incorrect" else ""
        index_html += f"""<tr>
<td>{r['num']}</td>
<td>{r['home']} vs {r['away']}</td>
<td>{r['date']}</td>
<td>{r['score']}</td>
<td>{r['accuracy']}</td>
<td class="{verdict_class}">{r['verdict']}</td>
<td><a href="{r['filename']}">View</a></td>
</tr>
"""
    index_html += """</table>
</body>
</html>"""

    index_path = pred_dir / "index.html"
    index_path.write_text(index_html)
    log(f"Index updated: {index_path.resolve()}")


def run_forebet_predictions(links_path: str, show_reasoning: bool = True,
                            high_only: bool = False, json_out: bool = False, html_out: bool = False,
                            compare_forebet: bool = True,
                            use_ml: bool = False):
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
        pred = analyze_from_data(data, use_ml=use_ml)

        # ── Minimum odds check for primary pick (improvement 12) ──
        min_odds = 1.10
        if pred["confidence"] == "Near Certain":
            min_odds = 1.08
        elif pred["confidence"] == "High":
            min_odds = 1.15
        elif pred["confidence"] == "Medium-High":
            min_odds = 1.25
        elif pred["confidence"] == "Medium":
            min_odds = 1.50

        # Get the pick's odds and skip if too low
        pick = pred.get("pick", "")
        market = pred.get("market", "")
        pick_odds_val = None
        if market == "1X2":
            pick_odds_val = {"Home win": data.get("odds_home"), "Draw": data.get("odds_draw"), "Away win": data.get("odds_away")}.get(pick)
        elif market == "O/U":
            if "Over" in pick:
                pick_odds_val = data.get("odds_over25")
            elif "Under" in pick:
                pick_odds_val = data.get("odds_under25")

        if pick_odds_val and pick_odds_val <= min_odds:
            # Downgrade confidence — market too short for meaningful value
            conf_rank = CONF_RANK.get(pred["confidence"], 99)
            if conf_rank < CONF_RANK["Medium-High"]:
                pred["confidence"] = "Medium-High"
                log(f"  [odds] Downgraded {pick} (odds {pick_odds_val:.2f} < min {min_odds:.2f})")

        # Detect finished match result
        actual_hg = data.get("actual_home_goals")
        actual_ag = data.get("actual_away_goals")
        actual_outcome = None
        correct_pick = None
        if actual_hg is not None and actual_ag is not None:
            if actual_hg > actual_ag:
                actual_outcome = "Home win"
            elif actual_hg < actual_ag:
                actual_outcome = "Away win"
            else:
                actual_outcome = "Draw"
            # Check if our primary pick matches
            our_12 = [p for p in (pred.get("all_picks") or []) if p["market"] == "1X2"]
            our_main_12 = our_12[0]["pick"] if our_12 else pred["pick"]
            correct_pick = (our_main_12 == actual_outcome)

        # Store in DB (map analysis keys to DB column names)
        poisson_probs = pred.get("_poisson_probs", (None, None, None))
        db_data = {
            **data,
            "our_prediction": pred["pick"],
            "our_confidence": pred["confidence"],
            "our_score_lean": pred["score_lean"],
            "our_stake": pred.get("_kelly_stake", 0.0),
            "our_market": pred.get("market", ""),
            "method_used": pred.get("_method", ""),
            "poisson_prob_home": poisson_probs[0] if poisson_probs else None,
            "poisson_prob_draw": poisson_probs[1] if poisson_probs else None,
            "poisson_prob_away": poisson_probs[2] if poisson_probs else None,
        }
        match_id = save_prediction(db_data)

        # Update DB with result if match is finished
        if actual_hg is not None and actual_ag is not None and match_id:
            update_result(match_id, actual_hg, actual_ag)
            # Store all market picks for accuracy tracking
            store_market_results(match_id, pred.get("all_picks", []),
                                actual_hg, actual_ag, actual_outcome, data)
            
            # Record ML vs Poisson accuracy for per-league tracking
            if use_ml:
                try:
                    from database import record_ml_league_result
                    from predict import detect_league
                    league_key = detect_league(data.get("league", ""))
                    
                    # Get pure Poisson 1X2 pick from stored Poisson probs
                    poisson_probs = pred.get("_poisson_probs", (None, None, None))
                    if poisson_probs and all(p is not None for p in poisson_probs):
                        p_h, p_d, p_a = poisson_probs
                        poisson_top = max(p_h, p_d, p_a)
                        if poisson_top == p_h:
                            poisson_12_pick = "Home win"
                        elif poisson_top == p_d:
                            poisson_12_pick = "Draw"
                        else:
                            poisson_12_pick = "Away win"
                    else:
                        poisson_12_pick = pred.get("pick", "")
                    
                    # Check 1X2 accuracy for both
                    ml_12 = [p for p in pred.get("all_picks", []) if p["market"] == "1X2"]
                    ml_12_pick = ml_12[0]["pick"] if ml_12 else pred.get("pick", "")
                    
                    ml_correct = (ml_12_pick == actual_outcome)
                    poisson_correct = (poisson_12_pick == actual_outcome)
                    
                    record_ml_league_result(league_key, "1X2", ml_correct, poisson_correct)
                except Exception:
                    pass

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
            # Match result (finished games)
            "actual_home_goals": actual_hg,
            "actual_away_goals": actual_ag,
            "actual_outcome": actual_outcome,
            "correct_pick": correct_pick,
            "ht_home_goals": data.get("ht_home_goals"),
            "ht_away_goals": data.get("ht_away_goals"),
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
            "h2h_goals_for": data.get("h2h_goals_for", 0),
            "h2h_goals_against": data.get("h2h_goals_against", 0),
            "h2h_avg_total_goals": data.get("h2h_avg_total_goals", 0),
            "home_home_avg_goals_for": data.get("home_home_avg_goals_for"),
            "home_home_avg_goals_against": data.get("home_home_avg_goals_against"),
            "away_away_avg_goals_for": data.get("away_away_avg_goals_for"),
            "away_away_avg_goals_against": data.get("away_away_avg_goals_against"),
            "home_over15_pct": data.get("home_over15_pct"),
            "home_under15_pct": data.get("home_under15_pct"),
            "away_over15_pct": data.get("away_over15_pct"),
            "away_under15_pct": data.get("away_under15_pct"),
            "home_over25_pct": data.get("home_over25_pct"),
            "home_under25_pct": data.get("home_under25_pct"),
            "away_over25_pct": data.get("away_over25_pct"),
            "away_under25_pct": data.get("away_under25_pct"),
            "home_over35_pct": data.get("home_over35_pct"),
            "home_under35_pct": data.get("home_under35_pct"),
            "away_over35_pct": data.get("away_over35_pct"),
            "away_under35_pct": data.get("away_under35_pct"),
            "home_btts_yes_pct": data.get("home_btts_yes_pct"),
            "home_btts_no_pct": data.get("home_btts_no_pct"),
            "away_btts_yes_pct": data.get("away_btts_yes_pct"),
            "away_btts_no_pct": data.get("away_btts_no_pct"),
            "home_scored_pct": data.get("home_scored_pct"),
            "home_conceded_pct": data.get("home_conceded_pct"),
            "away_scored_pct": data.get("away_scored_pct"),
            "away_conceded_pct": data.get("away_conceded_pct"),
            "home_total_shots_pg": data.get("home_total_shots_pg"),
            "home_shots_ontarget_pct": data.get("home_shots_ontarget_pct"),
            "away_total_shots_pg": data.get("away_total_shots_pg"),
            "away_shots_ontarget_pct": data.get("away_shots_ontarget_pct"),
            "home_clean_sheets_pct": data.get("home_clean_sheets_pct"),
            "away_clean_sheets_pct": data.get("away_clean_sheets_pct"),
            "match_id": match_id,
            # New fields
            "method": pred.get("_method", ""),
            "kelly_stake": pred.get("_kelly_stake", 0),
            "pick_odds": pred.get("_odds"),
            "league_difficulty": _get_league_difficulty(data.get("league", "")),
        })

    # ── Output ──
    if json_out:
        json.dump(results, indent=2, ensure_ascii=False, fp=sys.stdout)
        return

    # ── Always generate HTML ──
    report_path = _write_html(results, match_urls, compare_forebet, high_only)

    # Filter by confidence
    if high_only:
        results = [r for r in results if r["confidence"] in ("Near Certain", "High")]

    # ── Minimal text summary ──
    preds_made = 0

    ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def visible_len(text: str) -> int:
        clean = ANSI_ESCAPE.sub('', text)
        extra_width = 0
        for char in clean:
            cp = ord(char)
            if cp >= 0x2600 and cp <= 0x27BF:
                extra_width += 1
        return len(clean) + extra_width

    for r in results:
        preds_made += 1

        # ── Minimal summary line ──
        home = r['home']
        away = r['away']
        league = r.get('league', '')
        date = r.get('date', '')
        pick = r['pick']
        market = r.get('market', '')
        conf = r['confidence']

        # Line 1: Match info
        match_info = f"{home} vs {away}"
        if league:
            match_info += f"  •  {league}"
        if date:
            match_info += f"  •  {date}"
        print(f"\033[1m{match_info}\033[0m")

        # Line 2: Result (if finished)
        if r.get("actual_home_goals") is not None and r.get("actual_away_goals") is not None:
            hg = r["actual_home_goals"]
            ag = r["actual_away_goals"]
            outcome = r.get("actual_outcome", "")
            ht_h = r.get("ht_home_goals")
            ht_a = r.get("ht_away_goals")
            ht_str = f"  HT: {ht_h}-{ht_a}" if ht_h is not None and ht_a is not None else ""

            if r.get("correct_pick") is True:
                verdict = "\033[1;32m✓ Correct\033[0m"
            elif r.get("correct_pick") is False:
                our_12 = [p for p in (r.get("all_picks") or []) if p["market"] == "1X2"]
                our_main = our_12[0]["pick"] if our_12 else r["pick"]
                verdict = f"\033[1;31m✗ Incorrect\033[0m (picked {our_main})"
            else:
                verdict = ""

            print(f"RESULT: \033[1m{hg} - {ag}\033[0m ({outcome}){ht_str}  {verdict}")

        # Line 3: Pick info
        pick_line = f"Pick: \033[1m{pick}\033[0m ({market}) • {conf}"
        if r.get("score_lean"):
            pick_line += f" • Score: {r['score_lean']}"
        eh, ea = r.get("_exp_goals", (None, None))
        if eh is not None and ea is not None:
            pick_line += f" • \033[35mExp: {eh:.1f}-{ea:.1f}\033[0m"
        fb = r.get("forebet")
        if fb:
            pick_line += f" • Forebet: {fb}"
        print(pick_line)

        # Line 3b: Backup pick
        backup = r.get("_backup")
        if backup:
            print(f"  → Backup: {backup['market']}: {backup['pick']} ({backup['confidence']}) — covers {backup['coverage']} outcomes")

        # Line 4: HTML path
        print(f"→ predictions/{report_path.stem}.html")
    print(f"\nSaved to database: history.db")

    # Schedule retrain ~18h after games finish
    if preds_made > 0:
        schedule_retrain(delay_hours=18.0)


# ─────────────────────────────────────────────
# Review mode
# ─────────────────────────────────────────────

def _extract_result_from_forebet(soup) -> tuple | None:
    """Try to extract final score from Forebet page. Returns (h, a) or None."""
    if not soup:
        return None

    candidates = []

    # 1. Check all divs for clean score text (e.g. "3 - 2", "1-0")
    for div in soup.find_all("div"):
        text = div.get_text(strip=True)
        m = re.match(r"^(\d+)\s*[-–:]\s*(\d+)$", text)
        if m:
            h, a = int(m.group(1)), int(m.group(2))
            # Filter out implausible scores (times like "13-2", max realistic football score ~10)
            if h <= 10 and a <= 10 and h + a <= 15 and not (h == 0 and a == 0):
                candidates.append((h, a))

    # 2. Check h1
    h1 = soup.find("h1")
    if h1:
        m = re.search(r"(\d+)\s*[-–:]\s*(\d+)", h1.get_text())
        if m:
            h, a = int(m.group(1)), int(m.group(2))
            if h <= 10 and a <= 10 and h + a <= 15:
                candidates.append((h, a))

    # 3. Check stat-content tables
    tables = soup.find_all("table", {"class": "stat-content"})
    for table in tables:
        rows = table.find_all("tr")
        for row in rows[:5]:
            cells = row.find_all("td")
            if len(cells) >= 3:
                m = re.search(r"(\d+)\s*[-–:]\s*(\d+)", cells[-1].get_text())
                if m:
                    h, a = int(m.group(1)), int(m.group(2))
                    if h <= 10 and a <= 10 and h + a <= 15:
                        candidates.append((h, a))

    if not candidates:
        return None

    # Return the most common score (likely the match result, not a timestamp or odd)
    from collections import Counter
    best = Counter(candidates).most_common(1)[0][0]
    return best


def run_review(urls_file: str | None = None):
    """Review predictions — auto-fetch from Forebet, fall back to manual input.

    If urls_file is given, read URLs from that file, fetch each, extract score, update DB.
    Otherwise, read unreviewed matches from history.db.
    """
    init_db()
    updated = 0

    if urls_file:
        with open(urls_file) as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        print(f"Reviewing {len(urls)} URLs from {urls_file}\n")
        for i, url in enumerate(urls, 1):
            print(f"[{i}/{len(urls)}]", end=" ")
            scraper = ForebetScraper(url)
            if scraper.fetch():
                score = _extract_result_from_forebet(scraper.soup)
                if score:
                    # Match against DB by URL
                    conn = get_db()
                    row = conn.execute("SELECT id FROM matches WHERE forebet_url = ?", (url,)).fetchone()
                    conn.close()
                    if row:
                        update_result(row["id"], score[0], score[1])
                        print(f"✓ {score[0]}-{score[1]} (ID {row['id']})")
                        updated += 1
                    else:
                        print(f"  {score[0]}-{score[1]} (no matching DB record)")
                else:
                    print(f"  No score on page")
            else:
                print(f"  Could not fetch")
        print(f"\nUpdated {updated} match results.")
        return

    # DB-based review (existing behavior)
    pending = get_unreviewed_matches(limit=100)
    if not pending:
        print("No unreviewed matches found.")
        return

    print(f"Found {len(pending)} unreviewed matches.\n")
    for m in pending:
        print(f"ID {m['id']}: {m['home_team']} vs {m['away_team']} ({m['match_date']})")
        score = None
        if m.get('forebet_url'):
            scraper = ForebetScraper(m['forebet_url'])
            if scraper.fetch():
                score = _extract_result_from_forebet(scraper.soup)
                if score:
                    update_result(m['id'], score[0], score[1])
                    print(f"  ✓ Auto: {score[0]}-{score[1]}")
        if not score:
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

    # Show active bias corrections
    try:
        from database import get_calibration_biases
        biases = get_calibration_biases(min_samples=10)
        if biases:
            print(f"\n{'='*55}")
            print("ACTIVE BIAS CORRECTIONS (from calibration learning)")
            print(f"  {'League':<20} {'Market':<8} {'Threshold':<12} {'Bucket':<10} {'Bias':<8} {'Samples':<8}")
            print(f"  {'-'*66}")
            for b in biases[:15]:
                direction = "+" if b['bias'] >= 0 else ""
                print(f"  {b['league'][:18]:<20} {b['market'][:6]:<8} {b['threshold'][:10]:<12} "
                      f"{b['bucket']:<10} {direction}{b['bias']:.3f}  {int(b['sample_count']):<8}")
            if len(biases) > 15:
                print(f"  ... and {len(biases) - 15} more")
    except Exception:
        pass

    print()


# ─────────────────────────────────────────────
# Legacy odds-based mode
# ─────────────────────────────────────────────

def ensure_alias():
    """Create symlinks in ~/.local/bin/ for easy access."""
    bindir = Path.home() / ".local" / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve()
    for name in ("pr", "predict", "predictor"):
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
  predict.py --auto-learn            Run continuous learning pipeline (results + calibrate + retrain)
  predict.py --calibrate             Show calibration/accuracy stats
  predict.py --learn-calibration     Analyze bias, store corrections, retrain ML if needed
  predict.py --calibration-report    Generate detailed calibration quality report
  predict.py --force-retrain         Force ML model retraining from all data

Options:
  --high-only    Show only High / Near Certain predictions
  --json         JSON output for scripting
  --no-compare   Skip Forebet comparison display
  --no-reasoning Hide reasoning
  --no-ml, --classic  Disable ML-enhanced model (use classic Poisson only)
        """
    )
    parser.add_argument("file", nargs="?", help="File with Forebet URLs")
    parser.add_argument("--review", nargs="?", const=True, default=None, help="Review past predictions, or review URLs from a file")
    parser.add_argument("--auto", action="store_true", help="Auto-review by re-scraping")
    parser.add_argument("--learn", help="URL of Forebet results page to learn from")
    parser.add_argument("--auto-learn", action="store_true", help="Run continuous learning pipeline (scrape results + calibrate + retrain)")
    parser.add_argument("--calibrate", action="store_true", help="Show calibration stats")
    parser.add_argument("--learn-calibration", action="store_true", help="Run calibration learning: analyze bias and retrain ML model if needed")
    parser.add_argument("--calibration-report", action="store_true", help="Generate detailed calibration quality report")
    parser.add_argument("--force-retrain", action="store_true", help="Force ML model retraining from all available data")
    parser.add_argument("--high-only", action="store_true", help="Show only confident picks")
    parser.add_argument("--html", action="store_true", help="Output as HTML file (predictions.html)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--no-compare", action="store_true", help="Skip Forebet comparison")
    parser.add_argument("--no-reasoning", action="store_true", help="Hide reasoning")
    parser.add_argument("--no-ml", "--classic", action="store_true", help="Disable ML-enhanced prediction (use classic model)")

    args = parser.parse_args()

    ensure_alias()
    init_db()

    # Auto-run calibration learning on prediction runs (analyze bias + check retrain)
    # Disabled by default: step_scrape_results is slow and blocks predictions.
    # Run explicit modes instead: --auto-learn, --learn-calibration, --calibration-report
    if args.file and not args.no_ml and False:
        _maybe_auto_calibrate()

    if args.auto_learn:
        from auto_learn import run_full_pipeline
        run_full_pipeline(
            days_back=30,
            delay=0.3,
            max_matches=500,
        )
        return

    if args.learn:
        run_learn(args.learn)
        return

    if args.learn_calibration:
        from calibration_learner import run_calibration_learning
        run_calibration_learning(analyze=True, retrain=True, report=False)
        return

    if args.calibration_report:
        from calibration_learner import calibration_report
        calibration_report(verbose=True)
        return

    if args.force_retrain:
        from calibration_learner import auto_retrain
        did = auto_retrain(force=True)
        print(f"Retrain {'succeeded' if did else 'failed or not needed'}")
        return

    if args.calibrate:
        run_calibration()
        return

    if args.review:
        run_review(urls_file=None if args.review is True else args.review)
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
                html_out=args.html,
                compare_forebet=not args.no_compare,
                use_ml=not args.no_ml,
            )
            os.unlink(tmp_path)
        else:
            run_forebet_predictions(
                args.file,
                show_reasoning=not args.no_reasoning,
                high_only=args.high_only,
                json_out=args.json,
                html_out=args.html,
                compare_forebet=not args.no_compare,
                use_ml=not args.no_ml,
            )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
