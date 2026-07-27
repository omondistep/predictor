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
    store_market_results, get_market_accuracy, get_market_accuracy_history,
    add_pending_result,
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
    "near_certain": 0.68,
    "near_certain_margin": 0.15,
    "high": 0.55,
    "high_margin": 0.12,
    "medium_high": 0.44,
    "medium_high_margin": 0.08,
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
        from database import get_db
        conn = get_db()
        
        prefix = _resolve_league_prefix(league)
        
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


# Full league name → Forebet prefix mapping (used in calibration_log)
_LEAGUE_TO_FOREBET_PREFIX = {
    "romania": "Ro", "argentina": "Ar", "australia": "Au", "brazil": "Br",
    "bolivia": "Bo", "chile": "Cl", "colombia": "Co", "costa rica": "Cr",
    "czech": "Cz", "denmark": "Dk", "ecuador": "Ec", "estonia": "Ee",
    "finland": "Fi", "georgia": "Ge", "guatemala": "Gt", "hungary": "Hu",
    "ireland": "Ie", "iceland": "Is", "israel": "Il", "kazakhstan": "Kz",
    "kenya": "Ke", "korea": "Kr", "kuwait": "Kw", "kyrgyzstan": "Kg",
    "lebanon": "Lb", "lithuania": "Lt", "latvia": "Lv", "moldova": "Md",
    "mexico": "Mx", "morocco": "Mo", "malawi": "Mw", "macedonia": "Mk",
    "nicaragua": "Ni", "norway": "No", "panama": "Pa", "peru": "Pe",
    "paraguay": "Py", "poland": "Pl", "russia": "Ru", "serbia": "Rs",
    "sweden": "Se", "slovenia": "Si", "slovakia": "Sk", "switzerland": "Ch",
    "scotland": "Sc", "el salvador": "Sv", "ukraine": "Ua", "usa": "Us",
    "uruguay": "Uy", "venezuela": "Ve", "zimbabwe": "Zw",
    "conmebol": "CS", "uefa champions": "CL", "uefa europa": "EL",
    "conference league": "ECL", "copa libertadores": "CL",
    "copa sudamericana": "CS", "caribbean": "Ca",
}


def _resolve_league_prefix(league: str) -> str:
    """Resolve a full league name (e.g. 'Romania Divizia A') to its Forebet prefix (e.g. 'Ro1').

    The calibration_log stores leagues as 'Ro1 Team1 Team2', so we need the
    short prefix to query it.  Falls back to extracting the first word if no
    mapping is found.
    """
    if not league:
        return ""
    t = league.lower().strip()

    # 1. Try known full-name → prefix mapping + optional division number
    for name, prefix in _LEAGUE_TO_FOREBET_PREFIX.items():
        if name in t:
            # Extract division number if present (e.g. "Divizia A" → 1, "Serie B" → 2)
            div_m = re.search(r'(?:serie|division|liga|league|division)\s*([1-9])', t)
            div_num = div_m.group(1) if div_m else ""
            # Some leagues have letter suffixes (e.g. BrC, AuN)
            letter_m = re.search(r'(?:cup|copa|taça|Shield|NPL|state)', t)
            letter = ""
            if "cup" in t or "copa" in t or "taça" in t:
                letter = "C"
            elif "npl" in t or "national premier" in t:
                letter = "N"
            elif "serie a" in t or "primera" in t or "division 1" in t:
                div_num = "1"
            elif "serie b" in t or "segunda" in t or "division 2" in t:
                div_num = "2"
            return f"{prefix}{letter or div_num}"

    # 2. Fallback: first word (handles "Ro1 ..." or "Ar3 ..." style inputs)
    m = re.match(r'^([A-Za-z]+\d?)\s', league or '')
    if m:
        return m.group(1)
    m2 = re.match(r'^([A-Za-z]+\d?)$', league or '')
    if m2:
        return m2.group(1)
    return ""


def _get_league_recent_performance(league: str) -> dict:
    """Return recent (last 3 days) prediction accuracy for a league, broken down by market.

    Returns dict with:
      - overall: {total, correct, pct}
      - markets: { "1X2": {total, correct, pct}, "O/U": {...}, "BTTS": {...} }
      - rating: "hot" | "warm" | "cold" | "unknown"
      - summary: human-readable string
    """
    try:
        import datetime
        from database import get_db
        conn = get_db()

        prefix = _resolve_league_prefix(league)
        if not prefix:
            return {"overall": {"total": 0, "correct": 0, "pct": 0},
                    "markets": {}, "rating": "unknown", "summary": "No league data"}

        cutoff = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
        rows = conn.execute("""
            SELECT market, correct
            FROM calibration_log
            WHERE league LIKE ? AND created_at >= ?
        """, (f"{prefix}%", cutoff)).fetchall()
        conn.close()

        if not rows:
            return {"overall": {"total": 0, "correct": 0, "pct": 0},
                    "markets": {}, "rating": "unknown", "summary": "No recent data (7d)"}

        total = len(rows)
        correct = sum(r["correct"] for r in rows)
        pct = round(100.0 * correct / total, 1) if total else 0

        markets = {}
        for mk in ("1X2", "O/U", "BTTS"):
            mk_rows = [r for r in rows if r["market"] == mk]
            if mk_rows:
                mk_total = len(mk_rows)
                mk_correct = sum(r["correct"] for r in mk_rows)
                mk_pct = round(100.0 * mk_correct / mk_total, 1)
                markets[mk] = {"total": mk_total, "correct": mk_correct, "pct": mk_pct}

        if pct >= 65:
            rating = "hot"
        elif pct >= 50:
            rating = "warm"
        else:
            rating = "cold"

        mk_parts = []
        for mk, v in markets.items():
            mk_parts.append(f"{mk} {v['pct']:.0f}%({v['correct']}/{v['total']})")
        summary = f"Last 7d: {pct:.0f}% ({correct}/{total})" + (f" | {' '.join(mk_parts)}" if mk_parts else "")

        return {"overall": {"total": total, "correct": correct, "pct": pct},
                "markets": markets, "rating": rating, "summary": summary}
    except Exception:
        return {"overall": {"total": 0, "correct": 0, "pct": 0},
                "markets": {}, "rating": "unknown", "summary": "Error loading data"}


def _get_league_market_accuracy(league: str, market: str) -> dict:
    """Get accuracy for a specific league+market combo from last 7 days.

    Returns dict with: total, correct, pct (0-100), or empty if insufficient data.
    """
    try:
        import datetime
        from database import get_db
        conn = get_db()
        prefix = _resolve_league_prefix(league)
        if not prefix:
            return {"total": 0, "correct": 0, "pct": 0}
        cutoff = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
        row = conn.execute("""
            SELECT COUNT(*) as total, SUM(correct) as correct
            FROM calibration_log
            WHERE league LIKE ? AND market = ? AND created_at >= ?
        """, (f"{prefix}%", market, cutoff)).fetchone()
        conn.close()
        if not row or row["total"] < 5:
            return {"total": row["total"] if row else 0, "correct": row["correct"] if row else 0, "pct": 0}
        return {"total": row["total"], "correct": row["correct"],
                "pct": round(100.0 * row["correct"] / row["total"], 1)}
    except Exception:
        return {"total": 0, "correct": 0, "pct": 0}


def _kelly_interpretation(kelly_pct: float) -> str:
    """Human-readable Kelly stake interpretation."""
    if kelly_pct <= 0:
        return "No edge"
    if kelly_pct < 1.0:
        return "Minimal edge — skip or paper trade"
    if kelly_pct < 2.0:
        return "Small edge — fractional stake"
    if kelly_pct < 3.5:
        return "Moderate edge — measured bet"
    if kelly_pct < 5.0:
        return "Strong edge — confident position"
    return "Very strong edge — max stake (5% cap)"


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
        min_samples = 15  # Lowered from 25 — adjust sooner
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
                    CALIBRATED_THRESHOLDS["near_certain"] = min(0.78, CALIBRATED_THRESHOLDS["near_certain"] + 0.04)
                    adjusted += 1
                elif conf == "High":
                    CALIBRATED_THRESHOLDS["high"] = min(0.65, CALIBRATED_THRESHOLDS["high"] + 0.04)
                    adjusted += 1
                elif conf == "Medium-High":
                    CALIBRATED_THRESHOLDS["medium_high"] = min(0.55, CALIBRATED_THRESHOLDS["medium_high"] + 0.03)
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

    # Adjust for form — ADDITIVE offsets to avoid multiplicative compounding.
    # Form signal maps to ±0.15 goals max, weighted by sample size.
    hf_len = sum(1 for c in hf if c in "WDL") if hf else 0
    af_len = sum(1 for c in af if c in "WDL") if af else 0
    if h_f is not None:
        # h_f ranges ~0.5 (bad) to ~2.0 (good); map to ±0.15 goal offset
        form_offset = max(-0.15, min(0.15, (h_f - 1.2) * 0.2))
        form_offset *= min(1.0, (hf_len / 6) ** 0.5)  # shrink for short form
        exp_h += form_offset
    if a_f is not None:
        form_offset = max(-0.15, min(0.15, (a_f - 1.2) * 0.2))
        form_offset *= min(1.0, (af_len / 6) ** 0.5)
        exp_a += form_offset

    # Adjust for standings — ADDITIVE offsets, not multiplicative.
    # Top team gets +0.1 goals, bottom team gets -0.1 goals.
    if hp and ap and total_teams:
        # Offensive: higher position → slightly more goals
        exp_h += (total_teams - hp) / total_teams * 0.15
        exp_a += (total_teams - ap) / total_teams * 0.15
        # Defensive: higher position → slightly fewer conceded
        exp_a -= (total_teams - hp) / total_teams * 0.10
        exp_h -= (total_teams - ap) / total_teams * 0.10

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
    venue_w = 0.15  # venue stats get at most 15% weight (small samples of 3-5 games)
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

    # ── High-scoring league adjustment ──
    # For leagues with avg_goals > 3.0, allow higher expected goals when venue
    # stats support it (prevents underestimation in volatile high-scoring leagues)
    if profile.get("avg_goals", 2.5) > 3.0:
        # Check if venue stats show high-scoring tendency
        hh_gf_val = data.get("home_home_avg_goals_for")
        aa_gf_val = data.get("away_away_avg_goals_for")
        if hh_gf_val is not None and aa_gf_val is not None:
            venue_avg = (hh_gf_val + aa_gf_val) / 2.0
            # If venue average is significantly higher than current estimate, allow upward adjustment
            if venue_avg > exp_h + exp_a + 0.5:
                # Blend toward venue average (conservative: 25% weight)
                target_total = (exp_h + exp_a) * 0.75 + venue_avg * 0.25
                ratio = target_total / (exp_h + exp_a) if (exp_h + exp_a) > 0 else 1.0
                exp_h *= ratio
                exp_a *= ratio

    # Cap expected total at 3.5 to prevent extreme overestimation
    exp_h, exp_a = _cap_expected_goals(exp_h, exp_a, profile.get("avg_goals", 2.5))

    return max(0.1, exp_h), max(0.1, exp_a)


def _cap_expected_goals(exp_h: float, exp_a: float, league_avg: float = 2.5) -> tuple:
    """Cap expected total goals at 3.5 to prevent extreme overestimation.
    
    Real football matches rarely average >3.5 total goals. The expected goals
    model can inflate to 4.0+ when adjustments compound multiplicatively.
    This cap brings unrealistic estimates back to reality.
    """
    total = exp_h + exp_a
    if total > 3.5:
        # Scale down proportionally to cap at 3.5
        ratio = 3.5 / total
        exp_h *= ratio
        exp_a *= ratio
    return exp_h, exp_a
    """Cap expected total goals at 3.5 to prevent extreme overestimation.
    
    Real football matches rarely average >3.5 total goals. The expected goals
    model can inflate to 4.0+ when adjustments compound multiplicatively.
    This cap brings unrealistic estimates back to reality.
    """
    total = exp_h + exp_a
    if total > 3.5:
        # Scale down proportionally to cap at 3.5
        ratio = 3.5 / total
        exp_h *= ratio
        exp_a *= ratio
    return exp_h, exp_a


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
    nc = CALIBRATED_THRESHOLDS["near_certain"]
    hi = CALIBRATED_THRESHOLDS["high"]
    mh = CALIBRATED_THRESHOLDS["medium_high"]
    if odds < 1.25 and value > nc:
        return "Near Certain"
    if odds < 1.50 or value > hi:
        return "High"
    if odds < 1.70 or value > mh:
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


def _common_opponent_scoring_analysis(data: dict) -> dict:
    """Analyze scoring consistency and defensive vulnerability against common opponents.

    Key insights from Forebet analysis:
    1. Scoring consistency: Did each team score against shared opponents?
    2. Defensive vulnerability: How many goals conceded against common opponents?
    3. BTTS rate: Did both teams score in common opponent matches?
    4. O/U rate: Did matches against common opponents go over 2.5 goals?

    Returns dict with:
    - home_scoring_rate: float (0-1) - % of common opponent matches where home team scored
    - away_scoring_rate: float (0-1) - % of common opponent matches where away team scored
    - home_defense_vulnerability: float - avg goals conceded per match vs common opponents
    - away_defense_vulnerability: float - avg goals conceded per match vs common opponents
    - btts_rate: float (0-1) - % of common opponent matches where both teams scored
    - over25_rate: float (0-1) - % of common opponent matches with >2.5 goals
    - home_scoring_consistency: str - "High"/"Medium"/"Low"
    - away_scoring_consistency: str - "High"/"Medium"/"Low"
    - reasoning: list - human-readable insights
    """
    home_team = data.get("home_team", "")
    away_team = data.get("away_team", "")

    home_form = _get_team_form_details(data, "home")
    away_form = _get_team_form_details(data, "away")

    result = {
        "home_scoring_rate": 0.0,
        "away_scoring_rate": 0.0,
        "home_defense_vulnerability": 0.0,
        "away_defense_vulnerability": 0.0,
        "btts_rate": 0.0,
        "over25_rate": 0.0,
        "home_scoring_consistency": "Unknown",
        "away_scoring_consistency": "Unknown",
        "reasoning": [],
        "home_team": home_team,
        "away_team": away_team,
    }

    if not home_form or not away_form:
        return result

    # Build opponent maps
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

    common_opps = set(home_opp_map.keys()) & set(away_opp_map.keys())

    if not common_opps:
        return result

    # Analyze scoring against common opponents
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

        # BTTS check (did both teams score in their respective matches?)
        if h_gf > 0 and a_gf > 0:
            btts_count += 1

        # O/U check (combined goals from both matches)
        total_goals = h_gf + h_ga + a_gf + a_ga
        if total_goals > 2.5:
            over25_count += 1

        total_matches += 1

    if total_matches == 0:
        return result

    # Calculate rates
    result["home_scoring_rate"] = home_scored_count / total_matches
    result["away_scoring_rate"] = away_scored_count / total_matches
    result["home_defense_vulnerability"] = home_conceded_total / total_matches
    result["away_defense_vulnerability"] = away_conceded_total / total_matches
    result["btts_rate"] = btts_count / total_matches
    result["over25_rate"] = over25_count / total_matches

    # Consistency labels
    result["home_scoring_consistency"] = (
        "High" if result["home_scoring_rate"] >= 0.8 else
        "Medium" if result["home_scoring_rate"] >= 0.5 else
        "Low"
    )
    result["away_scoring_consistency"] = (
        "High" if result["away_scoring_rate"] >= 0.8 else
        "Medium" if result["away_scoring_rate"] >= 0.5 else
        "Low"
    )

    # Generate reasoning
    result["reasoning"].append(
        f"[SCORING] vs {len(common_opps)} common opponents: "
        f"{home_team} scored in {home_scored_count}/{total_matches} ({result['home_scoring_rate']:.0%}), "
        f"{away_team} scored in {away_scored_count}/{total_matches} ({result['away_scoring_rate']:.0%})"
    )

    if result["home_defense_vulnerability"] > 1.5 or result["away_defense_vulnerability"] > 1.5:
        result["reasoning"].append(
            f"[DEFENSE] Vulnerable defenses: {home_team} concedes {result['home_defense_vulnerability']:.1f}/game, "
            f"{away_team} concedes {result['away_defense_vulnerability']:.1f}/game vs common opponents"
        )

    if result["btts_rate"] >= 0.7:
        result["reasoning"].append(
            f"[BTTS] High BTTS rate ({result['btts_rate']:.0%}) against common opponents — both teams score consistently"
        )

    if result["over25_rate"] >= 0.7:
        result["reasoning"].append(
            f"[O/U] High O2.5 rate ({result['over25_rate']:.0%}) against common opponents — expect goals"
        )

    # Hidden strength detection: team that scores despite losing
    if result["home_scoring_rate"] >= 0.8 and data.get("home_form", "").count("L") >= 3:
        result["reasoning"].append(
            f"[HIDDEN] {home_team} scores consistently ({result['home_scoring_rate']:.0%}) despite poor record — dangerous underdog"
        )

    if result["away_scoring_rate"] >= 0.8 and data.get("away_form", "").count("L") >= 3:
        result["reasoning"].append(
            f"[HIDDEN] {away_team} scores consistently ({result['away_scoring_rate']:.0%}) despite poor record — dangerous underdog"
        )

    return result


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


def compute_ml_odds(data: dict) -> dict:
    """Compute independent ML odds from match data without using Forebet odds.
    
    Returns:
        dict with ml_1x2_odds, ml_ou_odds, ml_btts_odds, ml_dc_odds
        and underlying probabilities
    """
    ml_model = _load_ml_model()
    if not ml_model:
        return {}
    
    # Get ML predictions from the model
    ml_pred = ml_model.predict_from_row(data)
    
    # Extract probabilities
    ml_prob_home = ml_pred.get("ml_prob_home", 0.33)
    ml_prob_draw = ml_pred.get("ml_prob_draw", 0.33)
    ml_prob_away = ml_pred.get("ml_prob_away", 0.33)
    ml_prob_over = ml_pred.get("ml_prob_over", 0.5)
    ml_prob_under = ml_pred.get("ml_prob_under", 0.5)
    
    # Compute decimal odds (fair odds, no margin)
    def prob_to_odds(prob, min_prob=0.01):
        """Convert probability to decimal odds."""
        prob = max(prob, min_prob)
        return round(1 / prob, 2)
    
    # 1X2 odds
    ml_1x2_odds = {
        "home": prob_to_odds(ml_prob_home),
        "draw": prob_to_odds(ml_prob_draw),
        "away": prob_to_odds(ml_prob_away)
    }
    
    # O/U odds
    ml_ou_odds = {
        "over": prob_to_odds(ml_prob_over),
        "under": prob_to_odds(ml_prob_under)
    }
    
    # BTTS estimate (using form and scoring data)
    home_scored = data.get("home_scored_pct", 50) or 50
    away_scored = data.get("away_scored_pct", 50) or 50
    home_conceded = data.get("home_conceded_pct", 50) or 50
    away_conceded = data.get("away_conceded_pct", 50) or 50
    
    # Simple BTTS model: both teams score if both have >40% scoring rate
    ml_prob_btts_yes = (home_scored / 100 * away_scored / 100) * 1.2  # boost for correlation
    ml_prob_btts_yes = max(0.15, min(0.85, ml_prob_btts_yes))
    ml_prob_btts_no = 1 - ml_prob_btts_yes
    
    ml_btts_odds = {
        "yes": prob_to_odds(ml_prob_btts_yes),
        "no": prob_to_odds(ml_prob_btts_no)
    }
    
    # DC odds (Double Chance)
    ml_prob_1x = ml_prob_home + ml_prob_draw
    ml_prob_12 = ml_prob_home + ml_prob_away
    ml_prob_x2 = ml_prob_draw + ml_prob_away
    
    ml_dc_odds = {
        "1X": prob_to_odds(ml_prob_1x),
        "12": prob_to_odds(ml_prob_12),
        "X2": prob_to_odds(ml_prob_x2)
    }
    
    # DNB odds (Draw No Bet)
    ml_prob_dnb_home = ml_prob_home / (ml_prob_home + ml_prob_away) if (ml_prob_home + ml_prob_away) > 0 else 0.5
    ml_prob_dnb_away = ml_prob_away / (ml_prob_home + ml_prob_away) if (ml_prob_home + ml_prob_away) > 0 else 0.5
    
    ml_dnb_odds = {
        "home": prob_to_odds(ml_prob_dnb_home),
        "away": prob_to_odds(ml_prob_dnb_away)
    }
    
    return {
        "ml_1x2": {
            "probs": {"home": ml_prob_home, "draw": ml_prob_draw, "away": ml_prob_away},
            "odds": ml_1x2_odds
        },
        "ml_ou": {
            "probs": {"over": ml_prob_over, "under": ml_prob_under},
            "odds": ml_ou_odds
        },
        "ml_btts": {
            "probs": {"yes": ml_prob_btts_yes, "no": ml_prob_btts_no},
            "odds": ml_btts_odds
        },
        "ml_dc": {
            "probs": {"1X": ml_prob_1x, "12": ml_prob_12, "X2": ml_prob_x2},
            "odds": ml_dc_odds
        },
        "ml_dnb": {
            "probs": {"home": ml_prob_dnb_home, "away": ml_prob_dnb_away},
            "odds": ml_dnb_odds
        },
        "ml_prediction": ml_pred.get("ml_prediction"),
        "ml_ou_prediction": ml_pred.get("ml_ou_prediction")
    }


def analyze_ml_only(data: dict) -> dict:
    """Run a complete ML-only analysis pipeline with draw signal, form analysis,
    venue stats, and synthesis adjustments — independent of Forebet odds.
    
    Returns a dict structured identically to analyze_from_data() output.
    """
    ml_model = _load_ml_model()
    if not ml_model:
        return {}

    ml_pred = ml_model.predict_from_row(data)
    ph = ml_pred.get("ml_prob_home", 0.33)
    pd = ml_pred.get("ml_prob_draw", 0.33)
    pa = ml_pred.get("ml_prob_away", 0.33)
    p_over = ml_pred.get("ml_prob_over", 0.5)
    p_under = ml_pred.get("ml_prob_under", 0.5)

    # ── Convert 1X2 probabilities to expected goals ──
    best_exp_h, best_exp_a = 1.5, 1.2
    best_err = 1e9
    for eh_i in range(5, 60):
        eh = eh_i / 10.0
        for ea_i in range(3, 50):
            ea = ea_i / 10.0
            p_h_test = prob_home_win(eh, ea)
            p_d_test = prob_draw(eh, ea)
            p_a_test = prob_away_win(eh, ea)
            err = abs(p_h_test - ph) + abs(p_a_test - pa) + abs(p_d_test - pd)
            if err < best_err:
                best_err = err
                best_exp_h, best_exp_a = eh, ea
    exp_h, exp_a = best_exp_h, best_exp_a

    # ── Blend with venue stats (same as estimate_goals) ──
    # Venue stats are small-sample (3-5 games) but ground the model in reality.
    venue_w = 0.25  # ML-only gets more venue weight since inversion can be wild
    hh_gf = data.get("home_home_avg_goals_for")
    hh_ga = data.get("home_home_avg_goals_against")
    aa_gf = data.get("away_away_avg_goals_for")
    aa_ga = data.get("away_away_avg_goals_against")
    if hh_gf is not None:
        exp_h = exp_h * (1 - venue_w) + hh_gf * venue_w
    if aa_gf is not None:
        exp_a = exp_a * (1 - venue_w) + aa_gf * venue_w
    if hh_ga is not None:
        exp_a = exp_a * (1 - venue_w) + hh_ga * venue_w
    if aa_ga is not None:
        exp_h = exp_h * (1 - venue_w) + aa_ga * venue_w

    # Shots-on-target xG proxy
    h_sot = data.get("home_shots_ontarget_pct")
    a_sot = data.get("away_shots_ontarget_pct")
    h_tsh = data.get("home_total_shots_pg")
    a_tsh = data.get("away_total_shots_pg")
    if h_sot and h_tsh:
        h_xg = h_tsh * (h_sot / 100.0) * 0.32
        exp_h = (exp_h + h_xg) / 2
    if a_sot and a_tsh:
        a_xg = a_tsh * (a_sot / 100.0) * 0.32
        exp_a = (exp_a + a_xg) / 2

    # No-goal / clean-sheet discount
    home_score_rate = (data.get("home_scored_pct") or 100) / 100.0
    away_score_rate = (data.get("away_scored_pct") or 100) / 100.0
    home_cs_rate = (data.get("home_clean_sheets_pct") or 0) / 100.0
    away_cs_rate = (data.get("away_clean_sheets_pct") or 0) / 100.0
    exp_h *= (1.0 - 0.5 * (1.0 - home_score_rate)) * (1.0 - 0.5 * away_cs_rate)
    exp_a *= (1.0 - 0.5 * (1.0 - away_score_rate)) * (1.0 - 0.5 * home_cs_rate)
    exp_h = max(exp_h, 0.1)
    exp_a = max(exp_a, 0.1)
    exp_total = exp_h + exp_a

    # ── BTTS from scoring data ──
    home_scored = (data.get("home_scored_pct", 50) or 50) / 100.0
    away_scored = (data.get("away_scored_pct", 50) or 50) / 100.0
    p_btts_yes = min(0.85, max(0.15, home_scored * away_scored * 1.2))
    p_btts_no = 1 - p_btts_yes

    # DC / DNB from 1X2
    p_1x, p_x2, p_12 = ph + pd, pd + pa, ph + pa
    p_dnb_h = ph / (ph + pa) if (ph + pa) > 0 else 0.5
    p_dnb_a = pa / (ph + pa) if (ph + pa) > 0 else 0.5

    # ── Volatility ──
    league_key = detect_league(data.get("league", ""))
    profile = get_profile(league_key)
    vol = profile.get("volatility", 0.1)

    # ── Form analysis from string ──
    hf, af = data.get("home_form", ""), data.get("away_form", "")
    h_ppg = _ppg(hf) if hf else 1.0
    a_ppg = _ppg(af) if af else 1.0
    fsig = (a_ppg - h_ppg) / 3.0

    def _form_stats(form_str):
        """Parse form string into W/D/L counts and ppg."""
        w = sum(1 for c in form_str[:6] if c == "W")
        d = sum(1 for c in form_str[:6] if c == "D")
        l = sum(1 for c in form_str[:6] if c == "L")
        n = w + d + l
        ppg = (w * 3 + d) / n if n else 1.0
        return {"w": w, "d": d, "l": l, "n": n, "ppg": ppg}

    h_stats = _form_stats(hf)
    a_stats = _form_stats(af)

    # ── Form reasoning ──
    form_reasoning = []
    if hf:
        form_reasoning.append(
            f"H last 6: {h_stats['w']}W-{h_stats['d']}D-{h_stats['l']}L "
            f"({h_stats['ppg']:.1f} ppg)"
        )
    if af:
        form_reasoning.append(
            f"A last 6: {a_stats['w']}W-{a_stats['d']}D-{a_stats['l']}L "
            f"({a_stats['ppg']:.1f} ppg)"
        )

    # Venue-specific form from DB columns
    hh_gf = data.get("home_home_avg_goals_for")
    hh_ga = data.get("home_home_avg_goals_against")
    aa_gf = data.get("away_away_avg_goals_for")
    aa_ga = data.get("away_away_avg_goals_against")
    if hh_gf is not None:
        form_reasoning.append(f"H at home: {hh_gf:.1f}GF/{hh_ga:.1f}GA")
    if aa_gf is not None:
        form_reasoning.append(f"A away: {aa_gf:.1f}GF/{aa_ga:.1f}GA")

    # Trending (recent 3 vs older 3)
    if len(hf) >= 3:
        h_r3 = _form_stats(hf[:3])
        h_o3 = _form_stats(hf[3:6]) if len(hf) >= 6 else h_r3
        if h_r3["ppg"] - h_o3["ppg"] > 0.5:
            form_reasoning.append("H trending up (recent 3 better than older)")
        elif h_o3["ppg"] - h_r3["ppg"] > 0.5:
            form_reasoning.append("H trending down (recent 3 worse than older)")
    if len(af) >= 3:
        a_r3 = _form_stats(af[:3])
        a_o3 = _form_stats(af[3:6]) if len(af) >= 6 else a_r3
        if a_r3["ppg"] - a_o3["ppg"] > 0.5:
            form_reasoning.append("A trending up (recent 3 better than older)")
        elif a_o3["ppg"] - a_r3["ppg"] > 0.5:
            form_reasoning.append("A trending down (recent 3 worse than older)")

    # ── Draw tendency & draw signal factors ──
    _draw_factors = []
    h_d_count = sum(1 for c in hf[:6] if c == "D")
    a_d_count = sum(1 for c in af[:6] if c == "D")

    # Factor 1: Home team has ≥3 draws
    if h_d_count >= 3:
        _draw_factors.append(f"H:{h_d_count}D")

    # Factor 2: Away team has ≥3 draws
    if a_d_count >= 3:
        _draw_factors.append(f"A:{a_d_count}D")

    # Factor 3: Expected goals differential ≤ 0.3
    exp_diff = abs(exp_h - exp_a)
    if exp_diff <= 0.3 and exp_h + exp_a > 0:
        _draw_factors.append(f"expΔ{exp_diff:.1f}")

    # Factor 4: Form signal neutral (|signal| ≤ 0.15)
    if abs(fsig) <= 0.15:
        _draw_factors.append("form-neutral")

    # Factor 5: Venue low-scoring
    if hh_gf is not None and aa_gf is not None:
        if hh_gf <= 0.8 and aa_gf <= 0.8:
            _draw_factors.append("venue-low")

    # Factor 6: Home O15 rate < 50%
    h_ou15 = data.get("home_over15_pct")
    if h_ou15 is not None and h_ou15 < 50:
        _draw_factors.append(f"O15:{h_ou15}%")

    # Factor 7: Home BTTS rate < 40%
    btts_h = data.get("home_btts_yes_pct")
    if btts_h is not None and btts_h < 40:
        _draw_factors.append(f"BTTS:{btts_h}%")

    # Factor 8: Home CS rate > 30%
    cs_h = data.get("home_clean_sheets_pct")
    if cs_h is not None and cs_h > 30:
        _draw_factors.append(f"CS:{cs_h}%")

    # Factor 9: Home has ≥2D in venue-specific stats (from form string)
    if h_d_count >= 2 and a_d_count >= 1:
        _draw_factors.append("form-draws")

    # Draw tendency detection
    top_prob = max(ph, pd, pa)
    top_pick = "Home win" if ph == top_prob else ("Away win" if pa == top_prob else "Draw")
    margin = top_prob - pd if top_pick != "Draw" else 0
    _draw_tendency = False
    if top_pick != "Draw" and pd >= 0.28 and margin <= 0.12:
        _draw_tendency = True

    # Boost draw when multiple factors align
    p_draw_boosted = pd
    if len(_draw_factors) >= 3 and top_pick != "Draw":
        _draw_boost = min(len(_draw_factors) * 0.02, 0.06)
        p_draw_boosted = min(pd + _draw_boost, 0.55)

    # ── Build candidates ──
    candidates = []

    def add(market, pick, conf, reason, model_prob=None, odds=None):
        candidates.append({
            "market": market, "pick": pick, "confidence": conf,
            "rank": 0, "reason": reason, "model_prob": model_prob,
            "implied_prob": None, "value_ratio": None,
            "_always_show": True, "odds": odds,
        })

    def prob_to_odds(prob, min_p=0.01):
        return round(1 / max(prob, min_p), 2)

    # ── 1X2 with draw signal ──
    nc_thresh = CALIBRATED_THRESHOLDS["near_certain"]
    nc_margin = CALIBRATED_THRESHOLDS["near_certain_margin"]
    for name, prob in [("Home win", ph), ("Away win", pa)]:
        if prob >= nc_thresh and abs(ph - pa) >= nc_margin:
            conf = "Near Certain"
        elif prob >= CALIBRATED_THRESHOLDS["high"]:
            conf = "High"
        elif prob >= CALIBRATED_THRESHOLDS["medium_high"]:
            conf = "Medium-High"
        elif prob >= CALIBRATED_THRESHOLDS["medium"]:
            conf = "Medium"
        else: conf = "Low"
        add("1X2", name, conf, f"ML model {prob:.0%}", model_prob=prob,
            odds=prob_to_odds(prob))

    # Draw with boosted probability when draw signal is present
    draw_conf = "Medium" if pd >= 0.30 else "Low"
    if len(_draw_factors) >= 3:
        if p_draw_boosted >= 0.36: draw_conf = "Medium-High"
        elif p_draw_boosted >= 0.30: draw_conf = "Medium"
    add("1X2", "Draw", draw_conf, f"ML model {pd:.0%} ({'+'.join(_draw_factors[:3]) if _draw_factors else 'no signal'})",
        model_prob=p_draw_boosted, odds=prob_to_odds(p_draw_boosted))

    # ── DNB ──
    if vol < 0.25:
        if ph > pa + 0.08:
            if p_dnb_h >= 0.55: conf = "High"
            elif p_dnb_h >= 0.50: conf = "Medium-High"
            elif p_dnb_h >= 0.46: conf = "Medium"
            else: conf = "Low"
            add("DNB", "Home", conf, "derived from ML", model_prob=p_dnb_h,
                odds=prob_to_odds(p_dnb_h))
        elif pa > ph + 0.10:
            if p_dnb_a >= 0.58: conf = "High"
            elif p_dnb_a >= 0.52: conf = "Medium-High"
            elif p_dnb_a >= 0.48: conf = "Medium"
            else: conf = "Low"
            add("DNB", "Away", conf, "derived from ML", model_prob=p_dnb_a,
                odds=prob_to_odds(p_dnb_a))

    # ── DC ──
    dc_thresh = 0.72
    if p_1x > dc_thresh:
        add("DC", "1X", "Medium-High" if p_1x > 0.82 else "Medium", "derived from ML",
            model_prob=p_1x, odds=prob_to_odds(p_1x))
    if p_x2 > dc_thresh:
        add("DC", "X2", "Medium-High" if p_x2 > 0.82 else "Medium", "derived from ML",
            model_prob=p_x2, odds=prob_to_odds(p_x2))
    if p_12 > 0.86 and pd < 0.22:
        add("DC", "12", "Medium-High" if p_12 > 0.92 else "Medium", "derived from ML",
            model_prob=p_12, odds=prob_to_odds(p_12))

    # ── O/U multi-threshold ──
    for thresh, label_u, label_o in [(1.5, "Under 1.5", "Over 1.5"),
                                       (2.5, "Under 2.5", "Over 2.5"),
                                       (3.5, "Under 3.5", "Over 3.5")]:
        if thresh == 2.5:
            p_o = p_over
            p_u = p_under
        else:
            p_o = prob_over(exp_h, exp_a, thresh)
            p_u = 1.0 - p_o
        val_o, val_u = p_o - 0.5, p_u - 0.5

        if p_o > p_u and val_o > 0:
            ou_pick, ou_val = label_o, val_o
        elif p_u > p_o and val_u > 0:
            ou_pick, ou_val = label_u, val_u
        else:
            continue

        if ou_val > 0.45: ou_conf = "Near Certain"
        elif ou_val > 0.35: ou_conf = "High"
        elif ou_val > 0.18: ou_conf = "Medium-High"
        elif ou_val > 0.10: ou_conf = "Medium"
        else: ou_conf = "Low"

        if vol >= 0.25 and ou_conf in ("Near Certain", "High"):
            ou_conf = "Medium-High"
        if thresh == 3.5 and "Under" in ou_pick and exp_total > 4.0:
            ou_conf = "Low"
        if thresh == 1.5:
            if ou_conf not in ("Near Certain", "High", "Medium-High"):
                ou_conf = "Medium-High"
            if exp_total < 2.0: ou_conf = "Low"
            elif exp_total < 2.5 and ou_conf == "Low":
                ou_conf = "Medium"

        add("O/U", ou_pick, ou_conf, f"ML exp {exp_total:.1f}g model {max(p_o,p_u):.0%}" + (" (ML direct)" if thresh == 2.5 else ""),
            model_prob=max(p_o, p_u), odds=prob_to_odds(max(p_o, p_u)))

    # ── BTTS ──
    btts_conf = "Medium-High" if p_btts_yes > 0.62 else "Medium" if p_btts_yes > 0.52 else "Low"
    btts_no_conf = "Medium-High" if p_btts_no > 0.62 else "Medium" if p_btts_no > 0.52 else "Low"
    add("BTTS", "Yes", btts_conf, f"ML {p_btts_yes:.0%}", model_prob=p_btts_yes,
        odds=prob_to_odds(p_btts_yes))
    add("BTTS", "No", btts_no_conf, f"ML {p_btts_no:.0%}", model_prob=p_btts_no,
        odds=prob_to_odds(p_btts_no))

    # ── Synthesis ──
    reasoning = []

    # ── League reliability factor ──
    _ld = _get_league_difficulty(data.get("league", ""))
    if _ld.get("level") == "hard":
        reasoning.append(f"⚠ {_ld.get('reason', 'Unreliable league')}")

    # ── Common-opponent strength ──
    _cos = _common_opponent_strength(data)
    if _cos["reason"]:
        reasoning.append(f"⚠ {_cos['reason']}")

    # ── Transitive common-opponent analysis ──
    _trans = _transitive_common_opponent_analysis(data)
    if _trans and _trans.get("reasoning"):
        for tr in _trans["reasoning"]:
            reasoning.append(tr)

    # Draw tendency warning
    if _draw_tendency:
        reasoning.append(f"⚠ Draw tendency: {pd:.0%} vs {top_pick} {top_prob:.0%} (margin {margin:.0%})")

    # Draw signal warning
    if len(_draw_factors) >= 3:
        reasoning.append(f"⚠ Draw signal ({len(_draw_factors)}f): {', '.join(_draw_factors)}")

    reasoning.append(f"── ML-Only Synthesis ──")

    # Build MatchContext for synthesis
    _synth_ctx = None
    try:
        from synthesis import MatchContext, synthesize, build_synthesis_rationale, context_from_pred
        _synth_ctx = context_from_pred(
            {"_poisson_probs": (ph, pd, pa), "_exp_goals": (exp_h, exp_a),
             "_volatility": vol, "_warnings": []},
            data, vol=vol, form_signal=fsig,
            draw_tendency=_draw_tendency, draw_factors=len(_draw_factors),
            top_pick=top_pick, margin=margin,
        )
        candidates = synthesize(_synth_ctx, candidates, ml_only=True)
        _synth_rationale = build_synthesis_rationale(_synth_ctx, candidates, candidates[0])
        reasoning.append(_synth_rationale)
        method_parts = ["ml-only"]
    except Exception as e:
        reasoning.append(f"(ML synthesis error: {e})")
        candidates.sort(key=lambda c: c.get("model_prob") or 0, reverse=True)
        _synth_rationale = None
        method_parts = ["ml-odds"]

    primary = candidates[0] if candidates else {"pick": "—", "market": "—", "confidence": "Low"}

    # ── Picks summary ──
    picks_summary = []
    for c in candidates[:5]:
        star = "★" if c["confidence"] in ("Near Certain", "High") else "☆" if c["confidence"] == "Medium-High" else ""
        picks_summary.append(f"{star}{c['market']}: {c['pick']} ({c['confidence']})")

    # ── Synthform summary line (like the Forebet-based one) ──
    try:
        _top = candidates[0] if candidates else None
        _top_comp = _top.get("components", {}) if _top else {}
        _top_dv = _top.get("decision_value", 0) if _top else 0
        _top_pick = f"{_top['market']}: {_top['pick']}" if _top else "—"

        # Form direction and synthesis direction
        if fsig > 0.12:
            _form_dir, _form_strength = "away", f"+{fsig:.2f}"
        elif fsig < -0.12:
            _form_dir, _form_strength = "home", f"{fsig:.2f}"
        else:
            _form_dir, _form_strength = "balanced", "neutral"

        _syn_dir = "balanced"
        if _synth_ctx:
            if _synth_ctx.p_home > _synth_ctx.p_away + 0.10:
                _syn_dir = "home"
            elif _synth_ctx.p_away > _synth_ctx.p_home + 0.10:
                _syn_dir = "away"
            elif abs(_synth_ctx.p_home - _synth_ctx.p_away) <= 0.10:
                _syn_dir = "balanced"

        _synth_consensus = 0.0
        if _synth_ctx:
            _synth_consensus = _synth_ctx.form_signal

        _top_edge = None
        if _top and _top.get("model_prob"):
            implied = 1.0 / _top.get("odds", 2.0) if _top.get("odds") else 0.5
            if implied > 0:
                _top_edge = _top["model_prob"] - implied

        if _top_edge is not None and _top_edge > 0.02:
            _driver = f"model edge vs market +{_top_edge:.0%}"
        elif _top and _top.get("market") == "O/U":
            _driver = f"expected total {exp_total:.1f}g (model {_top_comp.get('prob', 0):.0%})"
        else:
            _driver = f"model probability {_top_comp.get('prob', 0):.0%}"

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

        _1x2_picks = [c for c in candidates if c["market"] == "1X2" and c["pick"] in ("Home win", "Away win")]
        _best_1x2 = _1x2_picks[0] if _1x2_picks else None
        if _best_1x2 and _top and _top["market"] != "1X2":
            _top_dv_val = _top.get("decision_value", 0)
            _1x2_dv_val = _best_1x2.get("decision_value", 0)
            _cov_diff = _top.get("coverage", 1) - _best_1x2.get("coverage", 1)
            if _cov_diff > 0:
                _driver += f" (O/U covers {_top.get('coverage', 1)} outcomes vs 1X2's 1)"
            elif _top_dv_val > _1x2_dv_val:
                _driver += f" (scored {_top_dv_val:.2f} vs {_best_1x2['pick']} {_1x2_dv_val:.2f})"

        _combo = (f"⟁SYNTHFORM⟁ {_form_dir} side ({_form_strength} form, "
                  f"{_synth_consensus:+.2f} consensus); settles on "
                  f"{_top_pick} ({_top['confidence']}, dv {_top_dv:.2f}).")

        _1x2_picks_all = [c for c in candidates if c["market"] == "1X2"]
        if _1x2_picks_all:
            _best_1x2_all = _1x2_picks_all[0]
            _1x2_prob = _best_1x2_all.get("model_prob")
            if _1x2_prob:
                _combo += f" 1X2: {_best_1x2_all['pick']} ({_1x2_prob:.0%})."

        reasoning.append("")
        reasoning.append(_combo)
    except Exception:
        pass

    # ── Form analysis section ──
    reasoning.append("")
    reasoning.append("── Form Analysis ──")
    reasoning.extend(form_reasoning)

    # ── Odds dict ──
    ml_odds_1x2 = {"home": prob_to_odds(ph), "draw": prob_to_odds(pd), "away": prob_to_odds(pa)}
    ml_odds_ou = {"over": prob_to_odds(p_over), "under": prob_to_odds(p_under)}
    ml_odds_btts = {"yes": prob_to_odds(p_btts_yes), "no": prob_to_odds(p_btts_no)}
    ml_odds_dc = {"1X": prob_to_odds(p_1x), "12": prob_to_odds(p_12), "X2": prob_to_odds(p_x2)}
    ml_odds_dnb = {"home": prob_to_odds(p_dnb_h), "away": prob_to_odds(p_dnb_a)}

    # ── Kelly Criterion for ML top pick ──
    ml_kelly = 0.0
    ml_top_prob = primary.get("model_prob")
    ml_top_odds = primary.get("odds")
    if ml_top_prob and ml_top_odds and ml_top_odds > 1.0:
        ml_implied = 1.0 / ml_top_odds
        ml_edge = ml_top_prob - ml_implied
        if ml_edge > 0:
            ml_kelly_f = ml_edge / (ml_top_odds - 1)
            ml_kelly = round(min(ml_kelly_f * 0.25, 0.05), 4)

    return {
        "pick": primary["pick"],
        "market": primary["market"],
        "confidence": primary["confidence"],
        "all_picks": candidates,
        "picks_summary": picks_summary,
        "reasoning": reasoning,
        "_exp_goals": (exp_h, exp_a),
        "_method": "+".join(method_parts),
        "_kelly_stake": ml_kelly,
        "_ml_odds": {
            "ml_1x2": {"probs": {"home": ph, "draw": pd, "away": pa}, "odds": ml_odds_1x2},
            "ml_ou": {"probs": {"over": p_over, "under": p_under}, "odds": ml_odds_ou},
            "ml_btts": {"probs": {"yes": p_btts_yes, "no": p_btts_no}, "odds": ml_odds_btts},
            "ml_dc": {"probs": {"1X": p_1x, "12": p_12, "X2": p_x2}, "odds": ml_odds_dc},
            "ml_dnb": {"probs": {"home": p_dnb_h, "away": p_dnb_a}, "odds": ml_odds_dnb},
            "ml_prediction": ml_pred.get("ml_prediction"),
            "ml_ou_prediction": ml_pred.get("ml_ou_prediction"),
        },
        "_synthesis_rationale": _synth_rationale if "_synth_rationale" in dir() else None,
        "_synthesis_ranked": [
            {"market": c["market"], "pick": c["pick"], "confidence": c["confidence"],
             "decision_value": c.get("decision_value"), "components": c.get("components")}
            for c in candidates[:6]
        ],
    }


def _get_ml_signal_weights(ml_model) -> dict:
    """Get ML-derived signal weights for FB model adjustments.

    Returns weights for how much each FB signal should be adjusted
    based on the ML model's learned feature importances.
    """
    try:
        from ml_model import get_feature_importance_weights
        return get_feature_importance_weights(ml_model)
    except Exception:
        return {}


# ── Derby detection ─────────────────────────────────────────────
# Known derby city/region patterns (lowercase). If both team names
# contain any matching pattern, the match is flagged as a derby.
_DERBY_REGIONS = [
    "newcastle", "sydney", "melbourne", "brisbane", "perth", "adelaide",
    "hobart", "canberra", "geelong", "gold coast", "sunshine coast",
    "wollongong", "central coast", "north shore", "western sydney",
    "eastern suburbs", "inner west", "northern beaches",
    "manchester", "liverpool", "london", "birmingham", "leeds", "sheffield",
    "glasgow", "edinburgh", "cardiff", "swansea", "bristol",
    "madrid", "barcelona", "seville", "valencia", "bilbao",
    "milan", "rome", "turin", "naples", "florence",
    "munich", "berlin", "dortmund", "hamburg", "frankfurt",
    "paris", "marseille", "lyon", "toulouse",
    "buenos aires", "sao paulo", "rio de janeiro", "bogota", "lima",
    "istanbul", "ankara", "izmir",
    "cairo", "alexandria", "cape town", "johannesburg",
    # Common local derivations
    "united", "city", "fc", "rovers", "wanderers", "rangers",
]

# City name extraction patterns (team name → city/region)
_CITY_PATTERNS = [
    (r'\b(Newcastle)\b', "newcastle"),
    (r'\b(Sydney)\b', "sydney"),
    (r'\b(Melbourne)\b', "melbourne"),
    (r'\b(Brisbane)\b', "brisbane"),
    (r'\b(Perth)\b', "perth"),
    (r'\b(Adelaide)\b', "adelaide"),
    (r'\b(Geelong)\b', "geelong"),
    (r'\b(Gold\s*Coast)\b', "gold coast"),
    (r'\b(Wollongong)\b', "wollongong"),
    (r'\b(Central\s*Coast)\b', "central coast"),
    (r'\b(Kahibah)\b', "newcastle"),  # Kahibah is a suburb of Newcastle
    (r'\b(Belmont)\b', "newcastle"),  # Belmont Swansea is in Newcastle area
    (r'\b(Swansea)\b', "newcastle"),  # Swansea is near Newcastle
    (r'\b(Lake\s*Macquarie)\b', "newcastle"),  # Lake Macquarie is in Newcastle area
    (r'\b(Charlestown)\b', "newcastle"),  # Charlestown is in Newcastle
    (r'\b(Valentine)\b', "newcastle"),  # Valentine is in Newcastle
    (r'\b(Edgeworth)\b', "newcastle"),  # Edgeworth is in Newcastle
    (r'\b(Adamstown)\b', "newcastle"),  # Adamstown is in Newcastle
    (r'\b(Broadmeadow)\b', "newcastle"),  # Broadmeadow is in Newcastle
    (r'\b(Cooks\s*Hill)\b', "newcastle"),  # Cooks Hill is in Newcastle
    (r'\b(Maitland)\b', "newcastle"),  # Maitland is near Newcastle
    (r'\b(Manchester)\b', "manchester"),
    (r'\b(Liverpool)\b', "liverpool"),
    (r'\b(Arsenal)\b', "london"),
    (r'\b(Tottenham)\b', "london"),
    (r'\b(West\s*Ham)\b', "london"),
    (r'\b(Chelsea)\b', "london"),
    (r'\b(Crystal\s*Palace)\b', "london"),
    (r'\b(Fulham)\b', "london"),
    (r'\b(QPR)\b', "london"),
    (r'\b(Milan)\b', "milan"),
    (r'\b(Inter)\b', "milan"),
    (r'\b(Real)\b', "madrid"),
    (r'\b(Atletico)\b', "madrid"),
    (r'\b(Barcelona)\b', "barcelona"),
    (r'\b(Espanyol)\b', "barcelona"),
    (r'\b(Boca)\b', "buenos aires"),
    (r'\b(River)\b', "buenos aires"),
    (r'\b(Corinthians)\b', "sao paulo"),
    (r'\b(Palmeiras)\b', "sao paulo"),
    (r'\b(Sao\s*Paulo)\b', "sao paulo"),
]


def _extract_team_region(team_name: str) -> str:
    """Extract city/region from team name using pattern matching."""
    import re
    if not team_name:
        return ""
    team_lower = team_name.lower()
    for pattern, region in _CITY_PATTERNS:
        if re.search(pattern, team_lower, re.IGNORECASE):
            return region
    return ""


def _detect_derby(data: dict) -> dict:
    """Detect if a match is a derby based on team locations.

    Returns dict with:
    - is_derby: bool
    - region: str (the shared region)
    - warning: str (visible warning message)
    """
    home_team = data.get("home_team", "")
    away_team = data.get("away_team", "")
    league = data.get("league", "")

    home_region = _extract_team_region(home_team)
    away_region = _extract_team_region(away_team)

    is_derby = False
    region = ""
    warning = ""

    if home_region and away_region and home_region == away_region:
        is_derby = True
        region = home_region
        warning = (f"🏟 DERBY: {home_team} vs {away_team} — local rivalry in {region.title()}. "
                   f"Form matters less in derbies; surprises are common.")
    elif home_region and away_region and home_region != away_region:
        # Check if regions are known to be nearby (e.g., Newcastle suburbs)
        # For now, just flag if same league and both have city patterns
        pass

    return {
        "is_derby": is_derby,
        "region": region,
        "warning": warning,
        "home_region": home_region,
        "away_region": away_region,
    }


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

    # ── Load league-market accuracy for recent performance gating ──
    _league_mkt_acc = {}
    for _mk in ("1X2", "O/U", "BTTS"):
        _acc = _get_league_market_accuracy(data.get("league", ""), _mk)
        if _acc.get("total", 0) >= 5:
            _league_mkt_acc[_mk] = _acc
    if _ld.get("level") == "hard":
        reasoning.append(f"⚠ {_ld.get('reason', 'Unreliable league')}")

    # ── Derby detection: flag local rivalries ──
    derby = _detect_derby(data)
    if derby["is_derby"]:
        reasoning.append(derby["warning"])

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
    _ml_from_ensemble = None
    method_parts = []
    # Accumulates how strongly the adjustment signals fired; drives the
    # single final blend weight between ML/DC base and exp-derived probs.
    signal_blend = 0.0

    # ── ML-derived signal weights (from feature importances) ──
    ml_signal_weights = _get_ml_signal_weights(ml_model) if ml_model else {}
    ml_ensemble_agreement = 0.5
    ml_feature_quality = 0.85

    if ml_model:
        from ml_model import poisson_predict, ensemble_predict, get_ensemble_agreement, get_feature_quality_score
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

        # Get ML ensemble agreement and feature quality
        ml_ensemble_agreement = get_ensemble_agreement(ml_model, data)
        ml_feature_quality = get_feature_quality_score(ml_model, data)

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
        _ml_from_ensemble = ensemble.get("_ml")
        method_parts.append(f"ml({getattr(ml_model, 'cv_accuracy_1x2', 0):.2f})")
        if dynamic_weights:
            method_parts.append("dyn-weights")
        # Track forebet component separately for accuracy tracking
        fb_h_raw_check = (data.get("forebet_home_pct") or 0)
        fb_d_raw_check = (data.get("forebet_draw_pct") or 0)
        fb_a_raw_check = (data.get("forebet_away_pct") or 0)
        if fb_h_raw_check + fb_d_raw_check + fb_a_raw_check > 0:
            method_parts.append("fb-weights")

        # ── Form signal: shift expected goals only (probabilities recomputed once at end) ──
        # ML feature importances weight: higher ML importance → stronger form signal
        # Derby penalty: form matters less in local derbies (reduces form impact by 40%)
        _form_weight = ml_signal_weights.get("form", 1.0) if ml_signal_weights else 1.0
        if derby["is_derby"]:
            _form_weight *= 0.6  # Reduce form impact in derbies
        fsig = form_analysis.get("signal", 0.0)
        if abs(fsig) >= 0.05:
            shift = fsig * 0.08 * _form_weight
            exp_h -= shift
            exp_a += shift
            exp_h = max(exp_h, 0.05)
            exp_a = max(exp_a, 0.05)
            signal_blend = min(0.65, signal_blend + abs(fsig) * _form_weight)
            method_parts.append("form")

        # ── Away win probability boost when form strongly favors away ──
        # Prevents systematic under-prediction of away wins
        if fsig > 0.20:
            away_boost = min(fsig * 0.10, 0.08)  # Max 8% boost
            p_away = min(p_away + away_boost, 0.65)
            # Renormalize
            total_p = p_home + p_draw + p_away
            p_home /= total_p
            p_draw /= total_p
            p_away /= total_p

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

            # Draw inflation (reduced to avoid over-predicting draws)
            draw_rate = profile.get("draw_rate", 0.25)
            draw_boost = 0.03 if exp_total < 2.5 else 0.02  # Reduced from 0.07/0.04
            if draw_rate >= 0.32:
                draw_boost += 0.02  # Reduced from 0.04
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

        # ── Away win probability boost when form strongly favors away (non-ML path) ──
        if fsig > 0.20:
            away_boost = min(fsig * 0.10, 0.08)
            p_away = min(p_away + away_boost, 0.65)
            total_p = p_home + p_draw + p_away
            p_home /= total_p
            p_draw /= total_p
            p_away /= total_p

    # ── Transitive common-opponent analysis: adjust expected goals ──
    _trans_analysis = _transitive_common_opponent_analysis(data)
    trans_adjusted = False
    draw_adjusted = False
    trans_signal = 0.0  # default; only overridden when transitivity fires

    # ── Common opponent scoring analysis (BTTS/O/U insights) ──
    _scoring_analysis = _common_opponent_scoring_analysis(data)
    if _scoring_analysis["reasoning"]:
        for r in _scoring_analysis["reasoning"]:
            reasoning.append(r)

    if _trans_analysis and _trans_analysis["reasoning"]:
        trans_signal = _trans_analysis.get("signal", 0.0)
        trans_conf = _trans_analysis.get("confidence", "Low")
        trans_weight = {"High": 0.30, "Medium-High": 0.20, "Medium": 0.12}.get(trans_conf, 0.0)
        # ML H2H feature importance weight
        _h2h_weight = ml_signal_weights.get("h2h", 1.0) if ml_signal_weights else 1.0
        trans_weight *= _h2h_weight
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

        # -- Away win probability floor when trans signal favors away --
        # When trans signal strongly favors away, ensure away win prob isn't suppressed
        if trans_signal > 0.25 and trans_conf in ("High", "Medium-High", "Medium"):
            # Boost away win probability floor based on trans signal strength
            away_floor = 0.25 + min(trans_signal * 0.15, 0.15)  # Floor 25-40%
            if p_away < away_floor:
                p_away = away_floor
                # Renormalize
                total_p = p_home + p_draw + p_away
                p_home /= total_p
                p_draw /= total_p
                p_away /= total_p
                method_parts.append("trans-away-boost")

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
    # ML ensemble agreement adjusts blend weight: high agreement → trust signals more
    _agree_boost = (ml_ensemble_agreement - 0.5) * 0.08  # ±4% max adjustment
    # ── Ensemble weight adjustment by league-market recent accuracy ──
    # If ML has been more accurate than Poisson in this league-market recently,
    # boost the ML blend weight;反之 reduce it
    _mkt_acc = _league_mkt_acc.get("1X2", {})
    _ensemble_adj = 0.0
    if _mkt_acc.get("total", 0) >= 5:
        if _mkt_acc["pct"] >= 70:
            _ensemble_adj = 0.04  # ML doing well, trust it more
        elif _mkt_acc["pct"] < 45:
            _ensemble_adj = -0.04  # ML struggling, lean toward Poisson
    _w = min(0.65, signal_blend + _agree_boost + _ensemble_adj)
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

        # ── Skip bad league-market combos ──
        _acc = _league_mkt_acc.get(market, {})
        if _acc.get("total", 0) >= 10 and _acc.get("pct", 0) < 40:
            reasoning.append(f"⚠ Skipped {market}: league accuracy {_acc['pct']:.0f}% ({_acc['total']} samples)")
            return

        # ── Confidence dampening for poor league-markets ──
        if _acc.get("total", 0) >= 5 and _acc.get("pct", 0) < 55:
            conf_rank = CONF_RANK.get(conf, 99)
            if conf_rank <= CONF_RANK["High"]:  # Near Certain or High
                conf = "Medium-High"
                reason += f" [damped: {market} {_acc['pct']:.0f}% in league]"
            elif conf == "Medium-High" and _acc["pct"] < 45:
                conf = "Medium"
                reason += f" [damped: {market} {_acc['pct']:.0f}% in league]"

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

    # ML ensemble agreement adjustment: high agreement → slightly lower thresholds
    # (easier to reach higher confidence), low agreement → raise thresholds
    _agree_adj = (ml_ensemble_agreement - 0.5) * 0.03  # ±1.5% max adjustment
    nc_thresh = max(0.55, nc_thresh - _agree_adj)
    hi_thresh = max(0.45, hi_thresh - _agree_adj)
    mh_thresh = max(0.35, mh_thresh - _agree_adj)

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

    # League reliability damping for 1X2: unreliable leagues
    # produce false-confidence picks — cap them down
    if _ld.get("level") == "hard" and _ld.get("accuracy", 0) > 0:
        if _ld["accuracy"] < 45:
            # Very unreliable: Near Certain → High, High → Medium-High
            if best_12_conf == "Near Certain":
                best_12_conf = "High"
            if best_12_conf == "High":
                best_12_conf = "Medium-High"
        elif _ld["accuracy"] < 65:
            # Unreliable: High → Medium-High
            if best_12_conf == "High":
                best_12_conf = "Medium-High"

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
    # ML draw feature importance weight: higher weight → more sensitive to draw signals
    _draw_weight = ml_signal_weights.get("draw", 1.0) if ml_signal_weights else 1.0
    _draw_tendency = False
    _draw_prob_adj = 0.28 - (_draw_weight - 1.0) * 0.05  # Adjust threshold
    _draw_margin_adj = 0.12 + (_draw_weight - 1.0) * 0.02
    if top_pick != "Draw" and p_draw >= _draw_prob_adj and margin <= _draw_margin_adj:
        _draw_tendency = True
        reasoning.append(f"⚠ Draw tendency: {p_draw:.0%} vs {top_pick} {top_prob:.0%} (margin {margin:.0%})")

    # ── Composite draw signal: multi-factor draw detection (conservative) ──
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

    # ── Skip draw factors when form/trans clearly favor a side ──
    _form_clearly_favors = abs(fsig) >= 0.25
    _trans_clearly_favors = abs(trans_signal) >= 0.30
    _model_clearly_favors = max(p_home, p_away) >= 0.48 and margin >= 0.12

    # Factor 1: Home team has ≥3 draws in recent form (raised from 2)
    if h_d_count >= 3:
        _draw_factors.append(f"H:{h_d_count}D")

    # Factor 2: Away team has ≥3 draws in recent form (raised from 2)
    if a_d_count >= 3:
        _draw_factors.append(f"A:{a_d_count}D")

    # Factor 3: Expected goals differential ≤ 0.3 (tightened from 0.5)
    exp_diff = abs(exp_h - exp_a)
    if exp_diff <= 0.3 and exp_h + exp_a > 0:
        _draw_factors.append(f"expΔ{exp_diff:.1f}")

    # Factor 4: Draw odds ≤ 2.8 (bookmaker signals draw — tightened from 3.0)
    draw_odds = data.get("odds_draw")
    if draw_odds and draw_odds <= 2.8:
        _draw_factors.append(f"odds{draw_odds:.1f}")

    # Factor 5: Form analysis signal is neutral (|signal| ≤ 0.15 — tightened from 0.2)
    if abs(fsig) <= 0.15:
        _draw_factors.append("form-neutral")

    # Factor 6: Transitive draw tendency (only when trans signal is weak)
    if _trans_analysis and _trans_analysis.get("draw_signal", 0) > 0.15:
        _draw_factors.append("trans-draw")

    # Factor 7: Both teams score ≤0.8 goals at venue (low-scoring game likely — tightened from 1.0)
    if hh_gf is not None and aa_gf is not None:
        if hh_gf <= 0.8 and aa_gf <= 0.8:
            _draw_factors.append("venue-low")

    # Factor 8: Home O15 rate < 50% (many low-scoring games — tightened from 60%)
    if h_ou15 is not None and h_ou15 < 50:
        _draw_factors.append(f"O15:{h_ou15}%")

    # Factor 9: Home BTTS rate < 40% (one team often fails to score — tightened from 45%)
    btts_h = data.get("home_btts_yes_pct")
    if btts_h is not None and btts_h < 40:
        _draw_factors.append(f"BTTS:{btts_h}%")

    # Factor 10: Home CS rate > 30% (home keeps clean sheets — tightened from 25%)
    cs_h = data.get("home_clean_sheets_pct")
    if cs_h is not None and cs_h > 30:
        _draw_factors.append(f"CS:{cs_h}%")

    # Remove draw factors when form/trans clearly favor a side
    if _form_clearly_favors or _trans_clearly_favors or _model_clearly_favors:
        # Only keep high-confidence draw factors (odds-based and venue-low)
        _draw_factors = [f for f in _draw_factors if f.startswith("odds") or f == "venue-low"]

    # Boost draw confidence when multiple factors align (reduced boost)
    p_draw_boosted = p_draw
    if len(_draw_factors) >= 3 and top_pick != "Draw":
        _draw_boost = min(len(_draw_factors) * 0.02, 0.06)  # Reduced from 0.03/0.10
        p_draw_boosted = min(p_draw + _draw_boost, 0.55)  # Capped lower (0.55 vs 0.60)
        # Upgrade Draw confidence only when boost is significant
        if p_draw_boosted >= 0.38 and draw_conf == "Low":
            draw_conf = "Medium"
            # Re-add Draw with upgraded confidence
            candidates = [c for c in candidates if not (c["market"] == "1X2" and c["pick"] == "Draw")]
            add("1X2", "Draw", draw_conf, f"model {p_draw:.0%} ({'+'.join(_draw_factors)})", model_prob=p_draw_boosted)
        reasoning.append(f"⚠ Draw signal ({len(_draw_factors)}f): {', '.join(_draw_factors)}")

    # ── Draw override: very conservative — only when model strongly agrees ──
    _draw_override = False
    _draw_override_reason = ""
    if len(_draw_factors) >= 6 and top_pick != "Draw":  # Raised from 5 to 6
        # Require Draw to clearly dominate AND not be contradicted by form/trans
        _side_max = max(p_home, p_away)
        if (p_draw >= 0.45 and p_draw - _side_max >= 0.06  # Tightened from 0.42/0.04
            and not _form_clearly_favors and not _trans_clearly_favors):
            _draw_override = True
            _draw_override_reason = f"Draw signal override ({len(_draw_factors)}f, margin {margin:.0%})"

    if _draw_override:
        # Remove the current primary pick (Home/Away win) and any existing Draw candidate
        candidates = [c for c in candidates if not (c["market"] == "1X2" and c["pick"] == top_pick)]
        candidates = [c for c in candidates if not (c["market"] == "1X2" and c["pick"] == "Draw")]
        # Determine Draw override confidence based on boosted probability
        if p_draw_boosted >= 0.42:
            draw_override_conf = "Medium-High"
        elif p_draw_boosted >= 0.38:
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
    # Blend with ML feature quality score (ML sees more data dimensions)
    if ml_feature_quality < 1.0:
        data_quality = data_quality * 0.7 + ml_feature_quality * 0.3

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
    # ML ensemble O/U probability (for 2.5 threshold) — used as blending signal
    ml_ou_signal = p_over  # ensemble-blended P(Over 2.5)
    for thresh, label_u, label_o in [(1.5, "Under 1.5", "Over 1.5"),
                                      (2.5, "Under 2.5", "Over 2.5"),
                                      (3.5, "Under 3.5", "Over 3.5")]:
        p_o_poisson = prob_over(exp_h, exp_a, thresh)
        # Blend Poisson with ML ensemble signal (threshold-dependent weight)
        # 2.5: ML is most relevant (direct match), 1.5/3.5: ML is indirect
        if thresh == 2.5:
            ml_w = 0.35
        elif thresh == 1.5:
            ml_w = 0.15
        else:  # 3.5
            ml_w = 0.25
        p_o = p_o_poisson * (1 - ml_w) + ml_ou_signal * ml_w
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

        # Common opponent O/U boost: if matches against shared opponents had high scoring
        if thresh == 2.5 and _scoring_analysis["over25_rate"] >= 0.7:
            # High O2.5 rate against common opponents → boost Over confidence
            if "Over" in ou_pick and ou_conf in ("Medium", "Medium-High"):
                ou_conf = "Medium-High" if ou_conf == "Medium" else "High"
        elif thresh == 2.5 and _scoring_analysis["over25_rate"] <= 0.3 and _scoring_analysis["over25_rate"] > 0:
            # Low O2.5 rate against common opponents → boost Under confidence
            if "Under" in ou_pick and ou_conf in ("Medium", "Medium-High"):
                ou_conf = "Medium-High" if ou_conf == "Medium" else "High"

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

    # ── BTTS (blended: Poisson + Forebet + venue rates) ──
    dc_rho = profile.get("dixon_coles_rho", -0.12)
    p_btss_poisson = prob_btts(exp_h, exp_a, rho=dc_rho)

    # Collect BTTS signals from multiple sources
    fb_btts_yes = data.get("home_btts_yes_pct")
    fb_btts_no = data.get("home_btts_no_pct")
    # Venue-specific BTTS rates (home/away splits)
    h_btts_rate = data.get("home_btts_pct")  # home team's BTTS rate
    a_btts_rate = data.get("away_btts_pct")  # away team's BTTS rate

    # Start with Poisson
    p_btss = p_btss_poisson

    # Forebet BTTS (high quality signal)
    if fb_btts_yes is not None and fb_btts_no is not None:
        fb_btts_prob = fb_btts_yes / 100.0
        # Stronger Forebet weight when both yes/no rates are available
        p_btss = p_btss_poisson * 0.45 + fb_btts_prob * 0.55
    elif fb_btts_yes is not None:
        p_btss = p_btss_poisson * 0.55 + (fb_btts_yes / 100.0) * 0.45

    # Venue-specific BTTS adjustment: blend in home/away BTTS rates
    if h_btts_rate is not None and a_btts_rate is not None:
        venue_btts = (h_btts_rate / 100.0 + a_btts_rate / 100.0) / 2.0
        p_btss = p_btss * 0.70 + venue_btts * 0.30

    # Common opponent BTTS boost: if both teams scored consistently against shared opponents
    if _scoring_analysis["btts_rate"] >= 0.7:
        # High BTTS rate against common opponents → boost BTTS YES probability
        btts_boost = (_scoring_analysis["btts_rate"] - 0.5) * 0.15  # Max 7.5% boost
        p_btss = min(p_btss + btts_boost, 0.95)
    elif _scoring_analysis["btts_rate"] <= 0.3 and _scoring_analysis["btts_rate"] > 0:
        # Low BTTS rate against common opponents → boost BTTS NO probability
        btts_no_boost = (0.5 - _scoring_analysis["btts_rate"]) * 0.10
        p_btss = max(p_btss - btts_no_boost, 0.05)

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
            draw_bias_suppressed=(_form_clearly_favors or _trans_clearly_favors or _model_clearly_favors),
            away_win_boosted=(fsig > 0.20),
        )
        # inject the computed 1X2 probs + exp goals directly (pred dict is empty)
        _synth_ctx.p_home, _synth_ctx.p_draw, _synth_ctx.p_away = p_home, p_draw, p_away
        _synth_ctx.exp_h, _synth_ctx.exp_a = exp_h, exp_a
        candidates = synthesize(_synth_ctx, candidates)
        _synth_rationale = build_synthesis_rationale(_synth_ctx, candidates, candidates[0])
        from synthesis import component_agreement
        _synth_consensus, _synth_n_sources = component_agreement(_synth_ctx)
        method_parts.append("synthesis")

        # ── Dynamic market selection: boost/penalize by league-market accuracy ──
        for c in candidates:
            _acc = _league_mkt_acc.get(c["market"], {})
            if _acc.get("total", 0) >= 5 and _acc.get("pct", 0) > 0:
                # Boost: +0.05 for >70%, +0.03 for >60%, penalty: -0.03 for <45%
                dv = c.get("decision_value", 0)
                if _acc["pct"] >= 70:
                    c["decision_value"] = dv + 0.05
                elif _acc["pct"] >= 60:
                    c["decision_value"] = dv + 0.03
                elif _acc["pct"] < 45:
                    c["decision_value"] = dv - 0.03
                c["league_mkt_boost"] = _acc["pct"]
        # Re-rank after boost
        non_show_temp = [c for c in candidates if not c.get('_always_show')]
        non_show_temp.sort(key=lambda c: c.get("decision_value", 0), reverse=True)
        if non_show_temp:
            candidates = non_show_temp + [c for c in candidates if c.get('_always_show')]
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

    # ── League-reliability pick rate gate ──
    # For leagues with <50% accuracy, suppress Medium-High picks to reduce noise
    if _ld.get("level") == "hard" and _ld.get("accuracy", 0) > 0 and _ld["accuracy"] < 50:
        # Only keep Medium picks that are truly strong (top prob >= 0.45)
        if primary.get("confidence") == "Medium-High":
            if primary.get("model_prob", 0) < 0.45:
                primary["confidence"] = "Low"
                reasoning.append(f"⚠ League reliability gate: suppressed Medium-High pick (league accuracy {_ld['accuracy']:.0f}%)")
        # Also suppress Medium picks for very unreliable leagues (<40% accuracy)
        if _ld["accuracy"] < 40 and primary.get("confidence") == "Medium":
            if primary.get("model_prob", 0) < 0.50:
                primary["confidence"] = "Low"
                reasoning.append(f"⚠ League reliability gate: suppressed Medium pick (league accuracy {_ld['accuracy']:.0f}%)")

    # Backup pick: best alternative if primary fails
    # Simply the next highest probability pick
    backup = None
    if len(non_show) > 1:
        backup = non_show[1]

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

        # Explain when primary pick differs from 1X2 consensus
        _1x2_picks = [c for c in candidates if c["market"] == "1X2" and c["pick"] in ("Home win", "Away win")]
        _best_1x2 = _1x2_picks[0] if _1x2_picks else None
        if _best_1x2 and _top["market"] != "1X2":
            # Primary pick is not 1X2 — explain why
            _top_dv_val = _top.get("decision_value", 0)
            _1x2_dv_val = _best_1x2.get("decision_value", 0)
            _cov_diff = _top.get("coverage", 1) - _best_1x2.get("coverage", 1)
            if _cov_diff > 0:
                _driver += f" (O/U covers {_top.get('coverage', 1)} outcomes vs 1X2's 1)"
            elif _top_dv_val > _1x2_dv_val:
                _driver += f" (scored {_top_dv_val:.2f} vs {_best_1x2['pick']} {_1x2_dv_val:.2f})"

        _combo = (f"⟁SYNTHFORM⟁ {_lead}overall the model settles on "
                  f"{_top_pick} ({_top['confidence']}), driven by {_driver} "
                  f"(decision value {_top_dv:.2f}).")

        # Add 1X2 prediction summary at the end
        _1x2_picks_all = [c for c in candidates if c["market"] == "1X2"]
        if _1x2_picks_all:
            _best_1x2_all = _1x2_picks_all[0]  # Already sorted by score
            _1x2_prob = _best_1x2_all.get("model_prob") or _best_1x2_all.get("components", {}).get("prob")
            _1x2_conf = _best_1x2_all.get("confidence", "")
            if _1x2_prob:
                _combo += (f" 1X2 prediction: {_best_1x2_all['pick']} "
                          f"({_1x2_prob:.0%}, {_1x2_conf}).")

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

    # ── Kelly scaling by league-market accuracy ──
    # Reduce stake in leagues where this market has poor track record
    _pm_acc = _league_mkt_acc.get(primary.get("market", ""), {})
    if _pm_acc.get("total", 0) >= 5 and _pm_acc.get("pct", 0) > 0:
        _acc_factor = _pm_acc["pct"] / 100.0
        kelly_stake = round(kelly_stake * _acc_factor, 4)

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
        "_ml": _ml_from_ensemble,
        "_synthesis_rationale": _synth_rationale if "_synth_rationale" in dir() else None,
        "_synthesis_ranked": [
            {"market": c["market"], "pick": c["pick"], "confidence": c["confidence"],
             "decision_value": c.get("decision_value"), "components": c.get("components")}
            for c in candidates[:6]
        ],
        "_warnings": warnings,
        "_backup": {"pick": backup["pick"], "market": backup["market"],
                    "confidence": backup["confidence"], "coverage": backup["coverage"]} if backup else None,
        "_ml_analysis": analyze_ml_only(data) if use_ml else None,
        "_derby": derby,
        "_scoring_analysis": _scoring_analysis,
    }


# ─────────────────────────────────────────────
# Prediction runner
# ─────────────────────────────────────────────

def log(msg, end="\n"):
    """Print progress to stderr so stdout stays clean for JSON."""
    print(msg, end=end, file=sys.stderr, flush=True)


def _write_html(results, all_urls, compare_forebet, high_only,
                _save_to=None, _title_suffix=""):
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

    def _verdict_signal(r: dict) -> str:
        """One-line signal: where does the model lean and how strongly."""
        synth = (r.get("_synthesis_rationale") or "").lower()
        if "favours home" in synth or "favors home" in synth:
            return "Home side favoured"
        if "favours away" in synth or "favors away" in synth:
            return "Away side favoured"
        if "draw tendency" in synth or "drawish" in synth:
            return "Draw-leaning matchup"
        # fallback: use primary pick direction
        pk = r.get("pick", "")
        if pk == "Home win":
            return "Home-side signal"
        if pk == "Away win":
            return "Away-side signal"
        if pk == "Draw":
            return "Neutral / draw signal"
        return "Balanced model signal"

    def _verdict_outlook(r: dict) -> str:
        """What the model expects in plain terms."""
        eh, ea = r.get("_exp_goals", (None, None))
        total = (eh + ea) if eh is not None and ea is not None else None
        pk = r.get("pick", "")
        market = r.get("market", "")
        parts = []
        if total is not None:
            parts.append(f"{total:.1f} expected goals total")
        if market == "O/U":
            parts.append(f"lean {pk}")
        elif market == "1X2":
            parts.append(f"{pk} the likely result")
        elif market == "BTTS":
            parts.append(f"{pk} both teams to score")
        elif market == "DC":
            parts.append(f"{pk} covers both outcomes")
        elif market == "DNB":
            parts.append(f"{pk} avoids the draw")
        if not parts:
            return "Check model reasoning"
        return "; ".join(parts)

    def _verdict_verdict(r: dict) -> str:
        """The final recommendation in one line."""
        market = r.get("market", "")
        pick = r.get("pick", "")
        conf = r.get("confidence", "")
        kelly = r.get("kelly_stake", 0)
        edge = ""
        if kelly and kelly > 0:
            edge = f" value present (Kelly {kelly*100:.1f}% — {_kelly_interpretation(kelly*100)})"
        comps = r.get("_synthesis_ranked") or []
        top = comps[0] if comps else {}
        dv = top.get("decision_value")
        if dv is not None:
            edge += f" synth {dv:.2f}"
        return f"{pick} ({market}) · {conf}{edge}"

    def _ml_reason_style(line: str) -> str:
        """Return inline CSS for ML-only reasoning lines."""
        if line.startswith("⚠"):
            return "color:#facc15;font-weight:600;"
        if line.startswith("──"):
            return "color:#94a3b8;font-weight:700;border-top:1px solid #334155;padding-top:6px;margin-top:8px;"
        if line.startswith("[TRANS"):
            return "color:#818cf8;font-size:0.78em;"
        if line.startswith("†Trans"):
            return "color:#818cf8;font-size:0.78em;font-style:italic;"
        if line.startswith("Form analysis"):
            return "color:#94a3b8;font-weight:600;"
        return "color:#cbd5e1;"

    def _comparison_table(r):
        """Build a consensus table for picks where both models agree (≥70% prob)."""
        ml = r.get("_ml_analysis")
        if not ml:
            return ""
        fb_picks = {f"{p['market']}|{p['pick']}": p for p in (r.get("all_picks") or []) if p.get("model_prob") and p["model_prob"] >= 0.695}
        ml_picks = {f"{p['market']}|{p['pick']}": p for p in ml.get("all_picks", []) if p.get("model_prob") and p["model_prob"] >= 0.695}
        agreed_keys = sorted(set(fb_picks) & set(ml_picks),
                             key=lambda k: (fb_picks[k]["model_prob"] + ml_picks[k]["model_prob"]) / 2,
                             reverse=True)
        if not agreed_keys:
            return ""
        rows_html = ""
        for k in agreed_keys:
            fp = fb_picks[k]
            mp = ml_picks[k]
            avg = (fp["model_prob"] + mp["model_prob"]) / 2
            mkt, pick = k.split("|", 1)
            rows_html += f'<tr><td>{mkt}</td><td>{pick}</td><td>{fp["model_prob"]:.0%}</td><td>{mp["model_prob"]:.0%}</td><td style="color:#22c55e;font-weight:600">{avg:.0%}</td></tr>'
        return (f'<div style="margin-top:10px;padding:8px 10px;background:rgba(34,197,94,0.06);border:1px solid rgba(34,197,94,0.2);border-radius:6px;font-size:0.82em;">'
                f'<div style="font-weight:700;color:#86efac;margin-bottom:4px;">🤝 Consensus picks (both ≥69.5%)</div>'
                f'<table style="width:100%;font-size:0.9em;"><tr><th>Mkt</th><th>Pick</th><th>FB</th><th>ML</th><th>Avg</th></tr>'
                f'{rows_html}</table></div>')

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

    def _pick_result(p, r):
        """Return result cell HTML for a pick against actual match result."""
        hg = r.get("actual_home_goals")
        ag = r.get("actual_away_goals")
        if hg is None or ag is None:
            return ""
        tot = hg + ag
        out = r.get("actual_outcome", "")
        pmk, ppk = p["market"], p["pick"]
        pc = None
        if pmk == "1X2":
            pc = (ppk == out)
        elif pmk == "O/U":
            if "Over" in ppk:
                pc = (tot > float(ppk.split()[-1]))
            elif "Under" in ppk:
                pc = (tot <= float(ppk.split()[-1]))
        elif pmk == "BTTS":
            both = hg > 0 and ag > 0
            pc = (ppk == "Yes" and both) or (ppk == "No" and not both)
        elif pmk == "DNB":
            if out == "Draw":
                return '<td style="color:#94a3b8;font-weight:700">push</td>'
            pc = (ppk == "Home" and out == "Home win") or (ppk == "Away" and out == "Away win")
        elif pmk == "DC":
            if ppk == "1X": pc = out in ("Home win", "Draw")
            elif ppk == "X2": pc = out in ("Away win", "Draw")
            elif ppk == "12": pc = out in ("Home win", "Away win")
        if pc is True:
            return '<td style="color:#22c55e;font-weight:700">✓</td>'
        elif pc is False:
            return '<td style="color:#ef4444;font-weight:700">✗</td>'
        return '<td style="color:#64748b">—</td>'

    rows = []
    for r in filtered:
        eh, ea = r.get("_exp_goals", (None, None))
        exp_str = f"{eh:.1f}-{ea:.1f}" if eh is not None else "—"
        hf = r.get("home_form", "")
        af = r.get("away_form", "")
        picks_rows = ""
        has_result_h = r.get("actual_home_goals") is not None and r.get("actual_away_goals") is not None
        
        # Sort picks by probability (highest first) for ranking
        all_picks_sorted = sorted(
            r.get("all_picks") or [],
            key=lambda x: x.get("model_prob") or 0,
            reverse=True
        )
        
        for p in all_picks_sorted:
            mp = p.get("model_prob")
            mp_s = f"{mp:.0%}" if mp else ""
            vr = p.get("value_ratio")
            vr_s = f" ({vr:.2f})" if vr else ""

            # Color highlighting for pick based on confidence
            _conf = p.get("confidence", "")
            if _conf == "High":
                _pick_bg = "rgba(34,197,94,0.15)"  # green
            elif _conf == "Medium-High":
                _pick_bg = "rgba(234,179,8,0.15)"  # yellow
            elif _conf == "Medium":
                _pick_bg = "rgba(249,115,22,0.12)"  # orange
            else:
                _pick_bg = "rgba(239,68,68,0.1)"  # red/light

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

            _dv_s = f"{p['decision_value']:.2f}" if p.get('decision_value') else ""
            picks_rows += f"<tr><td>{p['market']}</td><td style='background:{_pick_bg};border-radius:4px;padding:2px 6px'>{p['pick']}</td><td>{mp_s}</td><td>{_dv_s}</td>{result_cell}</tr>\n"

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
                # Use the actual primary pick (market + pick) for verdict display
                our_main = f"{r['market']}: {r['pick']}"
                verdict = f'<span style="color:#ef4444;font-weight:700">Incorrect</span> (picked {our_main})'
            else:
                verdict = ""
            result_html = f'<div class="pick-line" style="color:#60a5fa;font-weight:700">RESULT: {hg} - {ag} ({outcome}){ht_tag}  {verdict}</div>'

        exp_home = f"{eh:.2f}" if eh is not None else ""
        exp_away = f"{ea:.2f}" if ea is not None else ""
        exp_total = f"{eh + ea:.2f}" if eh is not None and ea is not None else ""
        
        # Model comparison badge
        _ml = r.get("_ml_analysis")
        if _ml and _ml.get("pick") and r.get("pick"):
            _agree = (_ml["pick"] == r["pick"] and _ml.get("market") == r.get("market"))
            if _agree:
                _compare_badge = '<div style="margin:8px 0;padding:6px 10px;border-radius:6px;font-size:0.82em;font-weight:600;background:rgba(34,197,94,0.12);color:#86efac;">✓ Models agree</div>'
            else:
                _compare_badge = (f'<div style="margin:8px 0;padding:6px 10px;border-radius:6px;font-size:0.82em;font-weight:600;background:rgba(234,179,8,0.12);color:#fde68a;">'
                                 f'⚠ Models disagree: <span style="color:{_c(r["confidence"])}">{r["market"]}: {r["pick"]}</span>'
                                 f' vs <span style="color:{_c(_ml["confidence"])}">{_ml["market"]}: {_ml["pick"]}</span></div>')
        else:
            _compare_badge = ""

        # Build league recent performance HTML badge
        _lr = r.get("league_recent", {})
        if _lr and _lr.get("overall", {}).get("total", 0) > 0:
            _rating = _lr.get("rating", "unknown")
            _lr_icon = {"hot": "🔥", "warm": "~", "cold": "❄"}.get(_rating, "?")
            _lr_bg = {"hot": "rgba(34,197,94,0.15);color:#86efac",
                      "warm": "rgba(234,179,8,0.15);color:#fde68a",
                      "cold": "rgba(239,68,68,0.12);color:#fca5a5"}.get(_rating, "rgba(148,163,184,0.15);color:#94a3b8")
            _ov = _lr["overall"]
            _mk_parts = "".join(f" | {mk} {mv['pct']:.0f}%({mv['correct']}/{mv['total']})" for mk, mv in _lr.get("markets", {}).items())
            league_perf_html = f'<div style="padding:4px 10px;border-radius:4px;font-size:0.78rem;margin-bottom:8px;display:inline-block;font-weight:600;background:{_lr_bg};">{_lr_icon} League (7d): <strong>{_ov["pct"]:.0f}%</strong> ({_ov["correct"]}/{_ov["total"]}){_mk_parts}</div>'
        else:
            league_perf_html = ""

        # Build ML-only picks table rows
        _ml_picks_html = ""
        for p in r["_ml_analysis"].get("all_picks", [])[:4]:
            _mp_s = f'{p["model_prob"]:.0%}' if p.get("model_prob") else ""
            _dv_s = f'{p.get("decision_value",0):.2f}' if p.get("decision_value") else ""
            _res_s = _pick_result(p, r) if has_result_h else ""
            _ml_picks_html += f'<tr><td>{p["market"]}</td><td>{p["pick"]}</td><td>{_mp_s}</td><td>{_dv_s}</td>{_res_s}</tr>\n'

        rows.append(f"""<div class="card" data-match-id="{r.get('match_id', '')}" data-exp-home="{exp_home}" data-exp-away="{exp_away}" data-exp-total="{exp_total}" data-market="{r['market']}" data-pick="{r['pick']}" data-date="{r.get('date', '')}" data-time="{r.get('time', '')}" style="border-left: 4px solid {_c(r['confidence'])};">
<div class="card-header">
  <span class="teams">{r['home']} vs {r['away']}</span>
  <span class="conf-badge" style="background:{_c(r['confidence'])}">{_star(r['confidence'])} {r['confidence']}</span>
</div>
<div class="card-meta">{r.get('league', '')} &middot; {r.get('date', '')} {r.get('time', '')} &middot; <a href="{r['url']}">Forebet</a>{method_tag}</div>
{league_perf_html}
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin:8px 0;font-size:0.82em;">
  <div style="background:#1e293b;border-radius:6px;padding:8px 10px;border-left:3px solid #60a5fa;">
    <div style="color:#94a3b8;font-size:0.75em;text-transform:uppercase;letter-spacing:0.05em;">Signal</div>
    <div style="color:#f1f5f9;font-weight:600;">{_verdict_signal(r)}</div>
  </div>
  <div style="background:#1e293b;border-radius:6px;padding:8px 10px;border-left:3px solid #facc15;">
    <div style="color:#94a3b8;font-size:0.75em;text-transform:uppercase;letter-spacing:0.05em;">Outlook</div>
    <div style="color:#f1f5f9;font-weight:600;">{_verdict_outlook(r)}</div>
  </div>
  <div style="background:#1e293b;border-radius:6px;padding:8px 10px;border-left:3px solid #34d399;">
    <div style="color:#94a3b8;font-size:0.75em;text-transform:uppercase;letter-spacing:0.05em;">Verdict</div>
    <div style="color:#f1f5f9;font-weight:600;">{_verdict_verdict(r)}</div>
  </div>
</div>
{"".join(f'<div class="league-warning" style="background:#7f1d1d;color:#fca5a5;padding:4px 10px;border-radius:4px;font-size:0.78rem;margin-bottom:8px;display:inline-block;">⚠ {r["league_difficulty"]["reason"]}</div>' if r.get("league_difficulty", {}).get("level") == "hard" and r.get("league_difficulty", {}).get("matches", 0) >= 5 else [])}
{"".join(f'<div style="background:#854d0e;color:#fde68a;padding:6px 10px;border-radius:6px;font-size:0.82rem;margin-bottom:8px;font-weight:600;">🏟 DERBY: {r["_derby"]["warning"]}</div>' if r.get("_derby", {}).get("is_derby") else "")}
{"".join(f'''<div style="background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.3);padding:6px 10px;border-radius:6px;font-size:0.78rem;margin-bottom:8px;">
<strong>📊 Common Opponent Scoring:</strong> {r["_scoring_analysis"]["home_team"]} scored in {r["_scoring_analysis"]["home_scoring_rate"]:.0%} of shared matches, {r["_scoring_analysis"]["away_team"]} in {r["_scoring_analysis"]["away_scoring_rate"]:.0%} | BTTS rate: {r["_scoring_analysis"]["btts_rate"]:.0%} | O2.5 rate: {r["_scoring_analysis"]["over25_rate"]:.0%}
</div>''' if r.get("_scoring_analysis", {}).get("btts_rate", 0) > 0 else "")}
{result_html}
<table style="margin:6px 0;">
  <tr><th>Home</th><td>Pos {r.get('home_pos', '—')}</td><td>Form {hf or '—'}</td><td>{r.get('odds_home', '—')}</td></tr>
  <tr><th>Draw</th><td></td><td></td><td>{r.get('odds_draw', '—')}</td></tr>
  <tr><th>Away</th><td>Pos {r.get('away_pos', '—')}</td><td>Form {af or '—'}</td><td>{r.get('odds_away', '—')}</td></tr>
</table>
<table><tr><th>O/U 2.5</th><td>{r.get('odds_over25') or '—'}/{r.get('odds_under25') or '—'}</td><th>BTTS</th><td>{r.get('odds_btts_yes') or '—'}/{r.get('odds_btts_no') or '—'}</td></tr></table>
{"<p>H2H: " + str(r.get('h2h_home_wins', 0)) + "W-" + str(r.get('h2h_draws', 0)) + "D-" + str(r.get('h2h_away_wins', 0)) + "L &ndash; GF/GA: " + str(r.get('h2h_goals_for', 0)) + "/" + str(r.get('h2h_goals_against', 0)) + " &ndash; avg " + str(r.get('h2h_avg_total_goals', 0)) + " goals (" + str(r.get('h2h_matches', 0)) + " matches)</p>" if r.get('h2h_matches', 0) >= 3 else ""}
{_venue_stats_html(r)}
{_compare_badge}
<div class="two-models">
{"".join(f'''<div class="model-panel model-forebet">
<div class="model-header">
  <span style="font-weight:600;">📊 Forebet-based</span>
  <span style="color:{_c(r['confidence'])};font-weight:700;font-size:0.85em;">{_star(r['confidence'])} {r['confidence']}</span>
</div>
<div style="font-weight:700;margin-bottom:4px;">{r['pick']} ({r['market']})</div>
<div style="color:#94a3b8;font-size:0.82em;margin-bottom:6px;">Exp: {exp_str} &middot; Kelly: {r.get('kelly_stake', 0)*100:.1f}%</div>
{f"<div style='font-size:0.78em;color:{'#10b981' if r.get('kelly_stake',0)*100>=3.5 else '#f59e0b' if r.get('kelly_stake',0)*100>=1.5 else '#64748b'};margin-bottom:6px;font-style:italic;'>{_kelly_interpretation(r.get('kelly_stake',0)*100)}</div>" if r.get('kelly_stake',0) > 0 else ""}
{reason_html}
{("<table><tr><th>Market</th><th>Pick</th><th>Prob</th><th>Score</th>" + ("<th>Result</th>" if has_result_h else "") + "</tr>" + picks_rows + "</table>") if picks_rows else ""}
{("<div style='margin-top:8px;padding:6px 10px;background:rgba(99,102,241,0.1);border-radius:6px;font-size:0.82em;'><strong>Best alt:</strong> <span style='color:" + _c(r.get('_backup', {}).get('confidence', 'Low')) + "'>" + r['_backup']['market'] + ": " + r['_backup']['pick'] + " (" + r['_backup']['confidence'] + ")</span></div>") if r.get('_backup') else ""}
</div>''')}
{"".join(f'''<div class="model-panel model-ml">
<div class="model-header">
  <span style="font-weight:600;">🤖 ML-only</span>
  <span style="color:{_c(r["_ml_analysis"]["confidence"])};font-weight:700;font-size:0.85em;">{_star(r["_ml_analysis"]["confidence"])} {r["_ml_analysis"]["confidence"]}</span>
</div>
<div style="font-weight:700;margin-bottom:4px;">{r["_ml_analysis"]["pick"]} ({r["_ml_analysis"]["market"]})</div>
<div style="color:#94a3b8;font-size:0.82em;margin-bottom:6px;">Exp: {r["_ml_analysis"]["_exp_goals"][0]:.1f}-{r["_ml_analysis"]["_exp_goals"][1]:.1f} &middot; Kelly: {r["_ml_analysis"].get("_kelly_stake", 0)*100:.1f}%</div>
{f"<div style='font-size:0.78em;color:{'#10b981' if r['_ml_analysis'].get('_kelly_stake',0)*100>=3.5 else '#f59e0b' if r['_ml_analysis'].get('_kelly_stake',0)*100>=1.5 else '#64748b'};margin-bottom:6px;font-style:italic;'>{_kelly_interpretation(r['_ml_analysis'].get('_kelly_stake',0)*100)}</div>" if r['_ml_analysis'].get('_kelly_stake',0) > 0 else ""}
<div style="font-size:0.78em;color:#94a3b8;margin-bottom:4px;">Model odds:</div>
<table style="font-size:0.85em;">
  <tr><th>1X2</th><td>H: {r["_ml_analysis"]["_ml_odds"]["ml_1x2"]["odds"]["home"]:.2f}</td><td>D: {r["_ml_analysis"]["_ml_odds"]["ml_1x2"]["odds"]["draw"]:.2f}</td><td>A: {r["_ml_analysis"]["_ml_odds"]["ml_1x2"]["odds"]["away"]:.2f}</td></tr>
  <tr><th>O/U</th><td>O: {r["_ml_analysis"]["_ml_odds"]["ml_ou"]["odds"]["over"]:.2f}</td><td>U: {r["_ml_analysis"]["_ml_odds"]["ml_ou"]["odds"]["under"]:.2f}</td><td></td></tr>
  <tr><th>BTTS</th><td>Y: {r["_ml_analysis"]["_ml_odds"]["ml_btts"]["odds"]["yes"]:.2f}</td><td>N: {r["_ml_analysis"]["_ml_odds"]["ml_btts"]["odds"]["no"]:.2f}</td><td></td></tr>
</table>
{"".join(f'<div class="synthform-line" style="margin:8px 0;padding:6px 10px;font-size:0.82em;">✦ {_highlight_exp(l.replace("⟁SYNTHFORM⟁", "", 1).strip())}</div>' if l.startswith("⟁SYNTHFORM⟁") else f'<div style="font-size:0.82em;margin:4px 0;{_ml_reason_style(l)}">{_highlight_exp(l)}</div>' if l.strip() and not l.startswith("──") else "" for l in r["_ml_analysis"].get("reasoning", []))}
<table style="font-size:0.85em;width:100%;">
  <tr><th>Market</th><th>Pick</th><th>Prob</th><th>Score</th>{"<th>Res</th>" if has_result_h else ""}</tr>
  {_ml_picks_html}
</table>
</div>''')}
{_comparison_table(r)}
</div>
</div>""")

    # ── Get market accuracy data for summary ──
    market_accuracy_data = get_market_accuracy()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Predictions{_title_suffix} — {now}</title>
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
.two-models {{ display:flex; gap:12px; margin-top:10px; }}
.model-panel {{ flex:1; min-width:0; background:rgba(15,23,42,0.5); border-radius:6px; padding:10px 12px; font-size:0.82em; }}
.model-forebet {{ border-left:3px solid #3b82f6; }}
.model-ml {{ border-left:3px solid #8b5cf6; }}
.model-header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; font-size:0.95em; }}
@media (max-width:700px) {{ .two-models {{ flex-direction: column; }} }}
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
<div style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:16px;">{"".join(f'<div class="stat" style="border-left:3px solid #888"><span style="color:#888">{m["market"]}</span> <span>{m["correct"]}/{m["total"]} ({m["accuracy"]:.0f}%)</span></div>' for m in market_accuracy_data if m["market"])}
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
<select id="filterDate" style="padding:8px 12px;border-radius:6px;border:1px solid #334155;background:#1e293b;color:#e2e8f0;font-size:0.9rem;outline:none;">
  <option value="">Date: Any</option>
  {"".join(f'<option value="{d}">{d}</option>' for d in sorted(set(r.get("date", "") for r in filtered if r.get("date"))))}
</select>
<select id="filterTime" style="padding:8px 12px;border-radius:6px;border:1px solid #334155;background:#1e293b;color:#e2e8f0;font-size:0.9rem;outline:none;">
  <option value="">Time: Any</option>
  <option value="morning">Morning (06-12)</option>
  <option value="afternoon">Afternoon (12-18)</option>
  <option value="evening">Evening (18-00)</option>
  <option value="night">Night (00-06)</option>
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
const filterDate = document.getElementById('filterDate');
const filterTime = document.getElementById('filterTime');
const matchCount = document.getElementById('matchCount');

function applyFilters() {{
  const q = (searchInput.value || '').toLowerCase().trim();
  const maxExp = filterExp.value ? parseFloat(filterExp.value) : null;
  const outcome = filterOutcome.value;
  const conf = filterConf.value;
  const market = filterMarket.value;
  const date = filterDate.value;
  const timeSlot = filterTime.value;
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
    // Date filter
    if (show && date) {{
      if (card.dataset.date !== date) show = false;
    }}
    // Time slot filter
    if (show && timeSlot) {{
      const t = card.dataset.time || '';
      const m = t.match(/(\\d{{1,2}}):(\\d{{2}})/);
      if (!m) {{ show = false; }}
      else {{
        const h = parseInt(m[1], 10);
        if (timeSlot === 'morning' && (h < 6 || h >= 12)) show = false;
        else if (timeSlot === 'afternoon' && (h < 12 || h >= 18)) show = false;
        else if (timeSlot === 'evening' && (h < 18 || h >= 24)) show = false;
        else if (timeSlot === 'night' && (h >= 6)) show = false;
      }}
    }}
    card.style.display = show ? '' : 'none';
    if (show) visible++;
  }});
  if (matchCount) matchCount.textContent = visible + ' / ' + cards.length;
}}

[searchInput, filterExp, filterOutcome, filterConf, filterMarket, filterDate, filterTime].forEach(el => {{
  if (el) el.addEventListener(el.tagName === 'INPUT' ? 'input' : 'change', applyFilters);
}});
</script>
</body>
</html>"""

    # ── Save HTML ──
    pred_dir = Path(__file__).parent / "predictions"
    pred_dir.mkdir(exist_ok=True)

    if _save_to:
        # Direct save (e.g. high-confidence report)
        _save_to.write_text(html)
        return _save_to

    report_path = pred_dir / "latest.html"

    # Archive previous latest if it exists
    if report_path.exists():
        archive_dir = pred_dir / "archive"
        archive_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path.rename(archive_dir / f"{ts}.html")

    report_path.write_text(html)
    log(f"HTML report: {report_path.resolve()}")

    # ── High-confidence-only report ──
    high_results = [r for r in results if r["confidence"] in ("Near Certain", "High")]
    high_path = None
    if high_results and len(high_results) < len(results):
        high_path = pred_dir / "high.html"
        _write_html(high_results, all_urls, compare_forebet, high_only=False,
                     _save_to=high_path, _title_suffix=" (High Confidence)")
        log(f"High-confidence report: {high_path.resolve()}")

    # ── Hot-leagues-only report ──
    hot_results = [r for r in results if r.get("league_recent", {}).get("rating") == "hot"]
    hot_path = None
    if hot_results:
        hot_path = pred_dir / "hot.html"
        _write_html(hot_results, all_urls, compare_forebet, high_only=False,
                     _save_to=hot_path, _title_suffix=" (Hot Leagues — 7d)")
        log(f"Hot-leagues report: {hot_path.resolve()} ({len(hot_results)} matches)")

    # ── Auto-open in browser ──
    webbrowser.open(str(report_path.resolve()))
    if high_path:
        webbrowser.open(str(high_path.resolve()))
    if hot_path:
        webbrowser.open(str(hot_path.resolve()))

    return report_path, hot_path


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

    # Scrape injuries page once (shared across all matches)
    from forebet_scraper import scrape_injuries, get_team_injury_summary
    log("Scraping injury data...")
    injuries = scrape_injuries()
    if injuries:
        log(f"  Found injury data for {len(injuries)} teams")
    else:
        log("  No injury data available")

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
            # Use the actual primary pick (market + pick), not just 1X2
            our_main = pred["pick"]
            our_market = pred.get("market", "")
            correct_pick = False
            if our_market == "1X2":
                correct_pick = (our_main == actual_outcome)
            elif our_market == "O/U":
                total_g = actual_hg + actual_ag
                if "Over" in our_main:
                    correct_pick = (total_g > float(our_main.split()[-1]))
                elif "Under" in our_main:
                    correct_pick = (total_g <= float(our_main.split()[-1]))
            elif our_market == "BTTS":
                both = actual_hg > 0 and actual_ag > 0
                correct_pick = (our_main == "Yes" and both) or (our_main == "No" and not both)
            elif our_market == "DNB":
                if actual_outcome == "Draw":
                    correct_pick = None  # Push
                else:
                    correct_pick = (our_main == "Home" and actual_outcome == "Home win") or (our_main == "Away" and actual_outcome == "Away win")
            elif our_market == "DC":
                if our_main == "1X": correct_pick = actual_outcome in ("Home win", "Draw")
                elif our_main == "X2": correct_pick = actual_outcome in ("Away win", "Draw")
                elif our_main == "12": correct_pick = actual_outcome in ("Home win", "Away win")

        # Store in DB (map analysis keys to DB column names)
        poisson_probs = pred.get("_poisson_probs", (None, None, None))
        ml_pred = pred.get("_ml")

        # Get injury summaries for this match
        home_team = data.get("home_team", "")
        away_team = data.get("away_team", "")
        home_summary = get_team_injury_summary(injuries, home_team) if injuries else {}
        away_summary = get_team_injury_summary(injuries, away_team) if injuries else {}

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
            "ml_prob_home": ml_pred.get("ml_prob_home") if ml_pred else None,
            "ml_prob_draw": ml_pred.get("ml_prob_draw") if ml_pred else None,
            "ml_prob_away": ml_pred.get("ml_prob_away") if ml_pred else None,
            # Injury data (map summary keys → DB column names)
            "home_injured_total": home_summary.get("total_injured", 0),
            "home_forwards_out": home_summary.get("forwards_out", 0),
            "home_midfielders_out": home_summary.get("midfielders_out", 0),
            "home_defenders_out": home_summary.get("defenders_out", 0),
            "home_key_players_out": home_summary.get("key_players_out", 0),
            "home_suspended": home_summary.get("suspended", 0),
            "away_injured_total": away_summary.get("total_injured", 0),
            "away_forwards_out": away_summary.get("forwards_out", 0),
            "away_midfielders_out": away_summary.get("midfielders_out", 0),
            "away_defenders_out": away_summary.get("defenders_out", 0),
            "away_key_players_out": away_summary.get("key_players_out", 0),
            "away_suspended": away_summary.get("suspended", 0),
        }
        match_id = save_prediction(db_data)

        # Compute proxy xG from Forebet shot/attack data
        if match_id:
            from database import compute_proxy_xg
            compute_proxy_xg(match_id, data)

        # Track unfinished matches for later result scraping
        if match_id and actual_hg is None:
            url = data.get("forebet_url", "")
            if url:
                add_pending_result(match_id, url, home_team, away_team)

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
            "time": data.get("match_time", ""),
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
            # Injury data
            "home_injured": home_summary.get("total_injured", 0),
            "away_injured": away_summary.get("total_injured", 0),
            "home_key_out": home_summary.get("key_players_out", 0),
            "away_key_out": away_summary.get("key_players_out", 0),
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
            "league_recent": _get_league_recent_performance(data.get("league", "")),
        })

    # ── Output ──
    if json_out:
        json.dump(results, indent=2, ensure_ascii=False, fp=sys.stdout)
        return

    # ── Always generate HTML ──
    report_path, hot_path = _write_html(results, match_urls, compare_forebet, high_only)

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

    for i, r in enumerate(results, 1):
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
        print(f"\n\033[1m{i}. {match_info}\033[0m")

        # Line 1b: League recent performance
        lr = r.get("league_recent", {})
        if lr and lr.get("overall", {}).get("total", 0) > 0:
            rating = lr.get("rating", "unknown")
            if rating == "hot":
                _lc = "\033[1;32m"  # green
                _icon = "🔥"
            elif rating == "warm":
                _lc = "\033[1;33m"  # yellow
                _icon = "~"
            elif rating == "cold":
                _lc = "\033[1;31m"  # red
                _icon = "❄"
            else:
                _lc = "\033[0m"
                _icon = "?"
            _ov = lr["overall"]
            _perf_str = f"{_icon} League: {_lc}{_ov['pct']:.0f}% ({_ov['correct']}/{_ov['total']})\033[0m (7d)"
            _mkt_parts = []
            for _mk, _mv in lr.get("markets", {}).items():
                _mkt_parts.append(f"{_mk} {_mv['pct']:.0f}%({_mv['correct']}/{_mv['total']})")
            if _mkt_parts:
                _perf_str += f" | {' '.join(_mkt_parts)}"
            print(f"  {_perf_str}")

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
                # Use the actual primary pick (market + pick) for verdict display
                our_main = f"{r['market']}: {r['pick']}"
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
        kelly_pct = r.get("kelly_stake", 0) * 100
        if kelly_pct > 0:
            pick_line += f" • Kelly: {kelly_pct:.1f}% ({_kelly_interpretation(kelly_pct)})"
        print(pick_line)

        # Line 3b: Backup pick
        backup = r.get("_backup")
        if backup:
            print(f"  → Backup: {backup['market']}: {backup['pick']} ({backup['confidence']}) — {backup['coverage']} outcomes")

        # Line 3c: ML Analysis
        ml_analysis = r.get("_ml_analysis")
        if ml_analysis:
            ml_conf = ml_analysis.get("confidence", "Low")
            print(f"  → ML Model: {ml_analysis['pick']} ({ml_analysis['market']}) • {ml_conf} • {ml_analysis.get('_method', '')}")
            ml_odds = ml_analysis.get("_ml_odds", {})
            ml_1x2 = ml_odds.get("ml_1x2", {})
            ml_ou = ml_odds.get("ml_ou", {})
            ml_btts = ml_odds.get("ml_btts", {})
            print(f"    Odds: 1X2 H:{ml_1x2.get('odds',{}).get('home',0):.2f} D:{ml_1x2.get('odds',{}).get('draw',0):.2f} A:{ml_1x2.get('odds',{}).get('away',0):.2f} | O/U O:{ml_ou.get('odds',{}).get('over',0):.2f} U:{ml_ou.get('odds',{}).get('under',0):.2f} | BTTS Y:{ml_btts.get('odds',{}).get('yes',0):.2f} N:{ml_btts.get('odds',{}).get('no',0):.2f}")
            # ML picks summary
            ml_picks = ml_analysis.get("all_picks", [])[:3]
            for mp in ml_picks:
                dv = mp.get("decision_value", 0)
                mp_s = f"{mp['model_prob']:.0%}" if mp.get("model_prob") else ""
                print(f"    {mp['market']:5s}: {mp['pick']:12s} {mp_s:>5s} {mp['confidence']:12s} [score {dv:.3f}]")
            # ML synthesis rationale
            ml_rationale = ml_analysis.get("reasoning", [])
            # Find and print the synthform line
            for line in ml_rationale:
                if line.startswith("⟁SYNTHFORM⟁"):
                    print(f"    ✦ {line.replace('⟁SYNTHFORM⟁', '', 1).strip()}")
                    break
            # Print last form analysis lines
            for line in ml_rationale[-3:]:
                if line and not line.startswith("──") and not line.startswith("⟁SYNTHFORM⟁"):
                    print(f"    {line}")

        # Line 4: HTML path
        print(f"→ predictions/latest.html")
    print(f"\nSaved to database: history.db")

    # ── Hot leagues summary ──
    hot_count = len([r for r in results if r.get("league_recent", {}).get("rating") == "hot"])
    if hot_count:
        # Deduplicate by league prefix
        hot_leagues = {}
        for r in results:
            lr = r.get("league_recent", {})
            if lr.get("rating") == "hot":
                league = r.get("league", "")
                prefix = league.split()[0] if league.split() else league
                if prefix not in hot_leagues:
                    hot_leagues[prefix] = lr
        print(f"\n\033[1;32m🔥 Hot Leagues (7d): {hot_count} matches from {len(hot_leagues)} leagues\033[0m")
        for prefix, lr in sorted(hot_leagues.items(), key=lambda x: x[1]["overall"]["pct"], reverse=True):
            ov = lr["overall"]
            mk_parts = " | ".join(f"{mk} {mv['pct']:.0f}%" for mk, mv in lr.get("markets", {}).items())
            print(f"  \033[32m{prefix:5s}\033[0m {ov['pct']:5.1f}% ({ov['correct']}/{ov['total']})  {mk_parts}")
        print(f"→ predictions/hot.html")

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
  predict.py results                Update existing reports with match results (no duplicates)
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
    parser.add_argument("--train-weights", action="store_true", help="Train ensemble weights from historical predictions")

    args = parser.parse_args()

    ensure_alias()
    init_db()

    # Handle "results" as a special positional command (pr results)
    if args.file and args.file.lower() == "results":
        from update_results import main as update_results_main
        update_results_main()
        return

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

    if args.train_weights:
        from database import train_weights_from_history
        print("Training ensemble weights from historical predictions...")
        result = train_weights_from_history(market="1X2", min_samples=10)
        print(f"Updated weights for {result['leagues_updated']} leagues")
        # Show top 10 by sample size
        sorted_leagues = sorted(result['details'].items(), key=lambda x: x[1]['total'], reverse=True)[:10]
        print("\nTop 10 leagues:")
        for league, info in sorted_leagues:
            print(f"  {league[:30]:30s} n={info['total']:3d} FB={info['forebet_acc']}% PM={info['poisson_ml_acc']}%")
            print(f"    Weights: FB={info['weights']['forebet']:.1%} P={info['weights']['poisson']:.1%} ML={info['weights']['ml']:.1%}")
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
