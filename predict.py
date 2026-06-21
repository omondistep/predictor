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
    get_calibration_summary, get_predictions_for_review, get_league_accuracy
)
from forebet_scraper import scrape_url, scrape_and_save, ForebetScraper

# ML-enhanced modules (optional)
_ML_MODEL = None
_DYNAMIC_WEIGHTS = None  # Cached per-league dynamic weights

# ─────────────────────────────────────────────
# League Profiles
# ─────────────────────────────────────────────

LEAGUE_PROFILES = {
    "brazil-serie-a":      {"avg_goals": 2.6, "u25_rate": 0.54, "btts_no_rate": 0.46, "draw_rate": 0.3, "home_win_rate": 0.39, "home_adv": 1.15, "volatility": 0.05},
    "brazil-serie-b":      {"avg_goals": 2.61, "u25_rate": 0.54, "btts_no_rate": 0.5, "draw_rate": 0.27, "home_win_rate": 0.43, "home_adv": 1.15, "volatility": 0.05},
    "brazil-serie-c":      {"avg_goals": 2.27, "u25_rate": 0.61, "btts_no_rate": 0.57, "draw_rate": 0.31, "home_win_rate": 0.42, "home_adv": 1.20, "volatility": 0.10},
    "brazil-serie-d":      {"avg_goals": 2.49, "u25_rate": 0.54, "btts_no_rate": 0.56, "draw_rate": 0.26, "home_win_rate": 0.45, "home_adv": 1.25, "volatility": 0.15},
    "brazil-u20":          {"avg_goals": 4.0, "u25_rate": 0.2, "btts_no_rate": 0.4, "draw_rate": 0.4, "home_win_rate": 0.6, "home_adv": 1.10, "volatility": 0.30},
    "argentina-b-nacional": {"avg_goals": 2.0, "u25_rate": 0.69, "btts_no_rate": 0.59, "draw_rate": 0.31, "home_win_rate": 0.52, "home_adv": 1.15, "volatility": 0.10},
    "argentina-primera-b":  {"avg_goals": 1.97, "u25_rate": 0.69, "btts_no_rate": 0.56, "draw_rate": 0.44, "home_win_rate": 0.39, "home_adv": 1.15, "volatility": 0.10},
    "argentina-primera-c":  {"avg_goals": 2.03, "u25_rate": 0.74, "btts_no_rate": 0.68, "draw_rate": 0.24, "home_win_rate": 0.35, "home_adv": 1.15, "volatility": 0.15},
    "argentina-federal-a":  {"avg_goals": 1.55, "u25_rate": 0.91, "btts_no_rate": 0.91, "draw_rate": 0.18, "home_win_rate": 0.73, "home_adv": 1.20, "volatility": 0.15},
    "chile-primera":        {"avg_goals": 2.31, "u25_rate": 0.57, "btts_no_rate": 0.57, "draw_rate": 0.27, "home_win_rate": 0.45, "home_adv": 1.15, "volatility": 0.05},
    "chile-primera-b":      {"avg_goals": 2.51, "u25_rate": 0.6, "btts_no_rate": 0.49, "draw_rate": 0.26, "home_win_rate": 0.54, "home_adv": 1.15, "volatility": 0.10},
    "usl-championship":     {"avg_goals": 2.71, "u25_rate": 0.71, "btts_no_rate": 0.57, "draw_rate": 0.36, "home_win_rate": 0.43, "home_adv": 1.15, "volatility": 0.10},
    "usl-league-one":       {"avg_goals": 3.35, "u25_rate": 0.35, "btts_no_rate": 0.47, "draw_rate": 0.12, "home_win_rate": 0.65, "home_adv": 1.15, "volatility": 0.15},
    "usl-league-two":       {"avg_goals": 3.67, "u25_rate": 0.26, "btts_no_rate": 0.39, "draw_rate": 0.14, "home_win_rate": 0.47, "home_adv": 1.15, "volatility": 0.30},
    "mls-next-pro":         {"avg_goals": 3.1, "u25_rate": 0.38, "btts_no_rate": 0.38, "draw_rate": 0.20, "home_win_rate": 0.50, "home_adv": 1.10, "volatility": 0.25},
    "nwsl":                {"avg_goals": 2.4, "u25_rate": 0.50, "btts_no_rate": 0.48, "draw_rate": 0.25, "home_win_rate": 0.46, "home_adv": 1.10, "volatility": 0.10},
    "uruguay-primera":      {"avg_goals": 2.96, "u25_rate": 0.4, "btts_no_rate": 0.47, "draw_rate": 0.23, "home_win_rate": 0.39, "home_adv": 1.10, "volatility": 0.10},
    "uruguay-segunda":      {"avg_goals": 2.0, "u25_rate": 0.83, "btts_no_rate": 0.33, "draw_rate": 0.67, "home_win_rate": 0.0, "home_adv": 1.15, "volatility": 0.15},
    "ecuador-serie-a":      {"avg_goals": 2.36, "u25_rate": 0.57, "btts_no_rate": 0.43, "draw_rate": 0.21, "home_win_rate": 0.64, "home_adv": 1.25, "volatility": 0.10},
    "ecuador-serie-b":      {"avg_goals": 1.9, "u25_rate": 0.62, "btts_no_rate": 0.58, "draw_rate": 0.32, "home_win_rate": 0.42, "home_adv": 1.25, "volatility": 0.15},
    "peru-primera":         {"avg_goals": 2.67, "u25_rate": 0.42, "btts_no_rate": 0.42, "draw_rate": 0.25, "home_win_rate": 0.67, "home_adv": 1.30, "volatility": 0.10},
    "paraguay-primera":     {"avg_goals": 2.2, "u25_rate": 0.69, "btts_no_rate": 0.51, "draw_rate": 0.36, "home_win_rate": 0.29, "home_adv": 1.15, "volatility": 0.10},
    "paraguay-segunda":     {"avg_goals": 1.9, "u25_rate": 0.62, "btts_no_rate": 0.58, "draw_rate": 0.32, "home_win_rate": 0.42, "home_adv": 1.15, "volatility": 0.15},
    "spain-segunda":        {"avg_goals": 2.58, "u25_rate": 0.53, "btts_no_rate": 0.56, "draw_rate": 0.19, "home_win_rate": 0.5, "home_adv": 1.15, "volatility": 0.05},
    "austria-landesliga":   {"avg_goals": 2.95, "u25_rate": 0.45, "btts_no_rate": 0.45, "draw_rate": 0.23, "home_win_rate": 0.41, "home_adv": 1.15, "volatility": 0.25},
    "reserve-leagues":      {"avg_goals": 3.0, "u25_rate": 0.35, "btts_no_rate": 0.35, "draw_rate": 0.24, "home_win_rate": 0.42, "home_adv": 1.05, "volatility": 0.35},
    "sweden-allsvenskan":   {"avg_goals": 2.6, "u25_rate": 0.45, "btts_no_rate": 0.44, "draw_rate": 0.24, "home_win_rate": 0.48, "home_adv": 1.15, "volatility": 0.08},
    "sweden-superettan":    {"avg_goals": 3.6, "u25_rate": 0.2, "btts_no_rate": 0.2, "draw_rate": 0.4, "home_win_rate": 0.4, "home_adv": 1.15, "volatility": 0.12},
    "sweden-ettan":         {"avg_goals": 2.8, "u25_rate": 0.40, "btts_no_rate": 0.38, "draw_rate": 0.23, "home_win_rate": 0.47, "home_adv": 1.12, "volatility": 0.20},
    "sweden-division-2":    {"avg_goals": 3.48, "u25_rate": 0.33, "btts_no_rate": 0.59, "draw_rate": 0.13, "home_win_rate": 0.57, "home_adv": 1.10, "volatility": 0.25},
    "finland-veikkausliiga":{"avg_goals": 2.5, "u25_rate": 0.48, "btts_no_rate": 0.46, "draw_rate": 0.25, "home_win_rate": 0.47, "home_adv": 1.12, "volatility": 0.12},
    "finland-ykkonen":      {"avg_goals": 2.6, "u25_rate": 0.45, "btts_no_rate": 0.44, "draw_rate": 0.24, "home_win_rate": 0.46, "home_adv": 1.10, "volatility": 0.18},
    "finland-kakkonen":     {"avg_goals": 2.82, "u25_rate": 0.55, "btts_no_rate": 0.55, "draw_rate": 0.36, "home_win_rate": 0.18, "home_adv": 1.10, "volatility": 0.25},
    "morocco-botola":       {"avg_goals": 2.0, "u25_rate": 0.63, "btts_no_rate": 0.57, "draw_rate": 0.3, "home_win_rate": 0.41, "home_adv": 1.12, "volatility": 0.08},
    "iceland":              {"avg_goals": 3.83, "u25_rate": 0.5, "btts_no_rate": 0.67, "draw_rate": 0.17, "home_win_rate": 0.83, "home_adv": 1.10, "volatility": 0.15},
    "iceland-women":        {"avg_goals": 2.0, "u25_rate": 0.65, "btts_no_rate": 0.55, "draw_rate": 0.30, "home_win_rate": 0.40, "home_adv": 1.10, "volatility": 0.20},
    "estonia":              {"avg_goals": 2.2, "u25_rate": 0.60, "btts_no_rate": 0.55, "draw_rate": 0.30, "home_win_rate": 0.42, "home_adv": 1.10, "volatility": 0.20},
    "georgia":              {"avg_goals": 2.3, "u25_rate": 0.55, "btts_no_rate": 0.50, "draw_rate": 0.28, "home_win_rate": 0.46, "home_adv": 1.15, "volatility": 0.20},
    "lithuania":            {"avg_goals": 2.1, "u25_rate": 0.60, "btts_no_rate": 0.55, "draw_rate": 0.30, "home_win_rate": 0.42, "home_adv": 1.10, "volatility": 0.20},
    "women-football":       {"avg_goals": 3.03, "u25_rate": 0.42, "btts_no_rate": 0.45, "draw_rate": 0.23, "home_win_rate": 0.44, "home_adv": 1.05, "volatility": 0.20},
    "algeria-ligue-2": {"avg_goals": 2.06, "u25_rate": 0.68, "btts_no_rate": 0.68, "draw_rate": 0.29, "home_win_rate": 0.55, "home_adv": 1.15, "volatility": 0.12},  # Algeria - Ligue 2
    "colombia-a": {"avg_goals": 2.46, "u25_rate": 0.55, "btts_no_rate": 0.39, "draw_rate": 0.39, "home_win_rate": 0.4, "home_adv": 1.15, "volatility": 0.12},  # Colombia - Primera A
    "colombia-b": {"avg_goals": 2.18, "u25_rate": 0.67, "btts_no_rate": 0.56, "draw_rate": 0.31, "home_win_rate": 0.38, "home_adv": 1.1, "volatility": 0.12},  # Colombia - Primera B
    "costa-rica-liga-de-ascenso": {"avg_goals": 2.75, "u25_rate": 0.56, "btts_no_rate": 0.47, "draw_rate": 0.25, "home_win_rate": 0.5, "home_adv": 1.15, "volatility": 0.12},  # Costa Rica - Liga de Ascenso
    "dr-congo-ligue-1": {"avg_goals": 1.79, "u25_rate": 0.76, "btts_no_rate": 0.61, "draw_rate": 0.36, "home_win_rate": 0.42, "home_adv": 1.15, "volatility": 0.08},  # DR Congo - Ligue 1
    "el-salvador-primera": {"avg_goals": 2.68, "u25_rate": 0.46, "btts_no_rate": 0.44, "draw_rate": 0.26, "home_win_rate": 0.36, "home_adv": 1.1, "volatility": 0.12},  # El Salvador - Primera Division
    "guatemala-liga-nacional": {"avg_goals": 2.22, "u25_rate": 0.63, "btts_no_rate": 0.48, "draw_rate": 0.26, "home_win_rate": 0.52, "home_adv": 1.15, "volatility": 0.12},  # Guatemala - Liga Nacional
    "guatemala-primera": {"avg_goals": 2.3, "u25_rate": 0.52, "btts_no_rate": 0.45, "draw_rate": 0.26, "home_win_rate": 0.61, "home_adv": 1.2, "volatility": 0.12},  # Guatemala - Primera Division
    "honduras-liga-nacional": {"avg_goals": 2.5, "u25_rate": 0.58, "btts_no_rate": 0.39, "draw_rate": 0.45, "home_win_rate": 0.32, "home_adv": 1.1, "volatility": 0.12},  # Honduras - Liga Nacional
    "libya-premier": {"avg_goals": 2.38, "u25_rate": 0.59, "btts_no_rate": 0.59, "draw_rate": 0.28, "home_win_rate": 0.41, "home_adv": 1.15, "volatility": 0.12},  # Libya - Premier League
    "mexico-liga-de-expansion-mx": {"avg_goals": 3.03, "u25_rate": 0.47, "btts_no_rate": 0.47, "draw_rate": 0.27, "home_win_rate": 0.53, "home_adv": 1.15, "volatility": 0.12},  # Mexico - Liga de Expansion MX
    "mexico-liga-mx": {"avg_goals": 2.74, "u25_rate": 0.45, "btts_no_rate": 0.52, "draw_rate": 0.19, "home_win_rate": 0.55, "home_adv": 1.15, "volatility": 0.12},  # Mexico - Liga MX
    "mexico-liga-serie-a": {"avg_goals": 2.73, "u25_rate": 0.47, "btts_no_rate": 0.5, "draw_rate": 0.23, "home_win_rate": 0.54, "home_adv": 1.15, "volatility": 0.12},  # Mexico - Liga Premier Serie A
    "nicaragua-primera": {"avg_goals": 2.66, "u25_rate": 0.57, "btts_no_rate": 0.52, "draw_rate": 0.27, "home_win_rate": 0.45, "home_adv": 1.15, "volatility": 0.12},  # Nicaragua - Primera Division
    "panama-football": {"avg_goals": 2.3, "u25_rate": 0.6, "btts_no_rate": 0.5, "draw_rate": 0.43, "home_win_rate": 0.37, "home_adv": 1.1, "volatility": 0.12},  # Panama - Football League
    "saudi-arabia-1st": {"avg_goals": 3.44, "u25_rate": 0.28, "btts_no_rate": 0.28, "draw_rate": 0.22, "home_win_rate": 0.44, "home_adv": 1.15, "volatility": 0.12},  # Saudi Arabia - 1st Division
    "sudan-premier": {"avg_goals": 1.96, "u25_rate": 0.68, "btts_no_rate": 0.57, "draw_rate": 0.4, "home_win_rate": 0.23, "home_adv": 1.1, "volatility": 0.08},  # Sudan - Premier League
    "syria-premier": {"avg_goals": 2.77, "u25_rate": 0.49, "btts_no_rate": 0.46, "draw_rate": 0.2, "home_win_rate": 0.37, "home_adv": 1.1, "volatility": 0.12},  # Syria - Premier League
    "thailand-thai-3": {"avg_goals": 2.31, "u25_rate": 0.63, "btts_no_rate": 0.64, "draw_rate": 0.22, "home_win_rate": 0.49, "home_adv": 1.15, "volatility": 0.12},  # Thailand - Thai League 3
    "turkiye-tff-3-lig": {"avg_goals": 2.39, "u25_rate": 0.57, "btts_no_rate": 0.63, "draw_rate": 0.22, "home_win_rate": 0.48, "home_adv": 1.15, "volatility": 0.12},  # Türkiye - TFF 3. Lig
    "venezuela-primera": {"avg_goals": 2.34, "u25_rate": 0.57, "btts_no_rate": 0.54, "draw_rate": 0.37, "home_win_rate": 0.37, "home_adv": 1.1, "volatility": 0.12},  # Venezuela - Primera Division
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

    # Adjust for form — capped to avoid streak overreaction
    hf_len = sum(1 for c in hf if c in "WDL") if hf else 0
    af_len = sum(1 for c in af if c in "WDL") if af else 0
    if h_f is not None:
        f = min(1.25, max(0.75, h_f / 1.2))
        f = 1.0 + (f - 1.0) * min(1.0, hf_len / 6)
        exp_h *= f
    if a_f is not None:
        f = min(1.25, max(0.75, a_f / 1.2))
        f = 1.0 + (f - 1.0) * min(1.0, af_len / 6)
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

    # Venue-specific goal averages (home-at-home, away-at-away)
    hh_gf = data.get("home_home_avg_goals_for")
    hh_ga = data.get("home_home_avg_goals_against")
    aa_gf = data.get("away_away_avg_goals_for")
    aa_ga = data.get("away_away_avg_goals_against")
    if hh_gf:
        exp_h = (exp_h + hh_gf) / 2
    if aa_gf:
        exp_a = (exp_a + aa_gf) / 2
    if hh_ga:
        exp_a = (exp_a + hh_ga) / 2
    if aa_ga:
        exp_h = (exp_h + aa_ga) / 2

    # Shots-on-target proxy for xG
    h_sot = data.get("home_shots_ontarget_pct")
    a_sot = data.get("away_shots_ontarget_pct")
    h_tsh = data.get("home_total_shots_pg")
    a_tsh = data.get("away_total_shots_pg")
    if h_sot and h_tsh:
        # Expected goals ≈ shots_on_target * 0.65 – 0.75 (league average conversion)
        h_xg_proxy = h_tsh * (h_sot / 100.0) * 0.70
        exp_h = (exp_h + h_xg_proxy) / 2
    if a_sot and a_tsh:
        a_xg_proxy = a_tsh * (a_sot / 100.0) * 0.70
        exp_a = (exp_a + a_xg_proxy) / 2

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
        from ml_model import MLPredictor
        _ML_MODEL = MLPredictor.load(auto_train=True)
    return _ML_MODEL if _ML_MODEL and _ML_MODEL.is_trained else None


def _get_dynamic_weights(league_key: str):
    """Get dynamic ensemble weights from DB tracking (improvement 3)."""
    try:
        from database import get_dynamic_weights
        return get_dynamic_weights(league=league_key, market="1X2")
    except Exception:
        return None


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

    hf, af = data.get("home_form", ""), data.get("away_form", "")
    h_ppg = _ppg(hf) if hf else None
    a_ppg = _ppg(af) if af else None
    hp, ap = data.get("home_pos"), data.get("away_pos")
    hm_ = data.get("h2h_matches", 0)
    hw_ = data.get("h2h_home_wins", 0) if hm_ >= 3 else 0
    ha_ = data.get("h2h_away_wins", 0) if hm_ >= 3 else 0

    # Auto-calibrate thresholds from DB (improvement 10)
    _auto_calibrate_thresholds()

    # ── ML-enhanced probability computation ──
    ml_model = _load_ml_model() if use_ml else None
    method_parts = []

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

        # Get dynamic weights from DB (improvement 3)
        dynamic_weights = _get_dynamic_weights(league_key)

        # Blend with ML model using dynamic weights
        ensemble = ensemble_predict(data, profile, ml_model, dynamic_weights=dynamic_weights)
        p_home = ensemble["prob_home"]
        p_draw = ensemble["prob_draw"]
        p_away = ensemble["prob_away"]
        p_over = ensemble["prob_over"]
        p_under = ensemble["prob_under"]
        method_parts.append(f"ml({getattr(ml_model, 'cv_accuracy_1x2', 0):.2f})")
        if dynamic_weights:
            method_parts.append("dyn-weights")
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
        if top_prob >= nc_thresh:
            best_12_conf = "Near Certain"
        elif top_prob >= hi_thresh:
            best_12_conf = "High" if margin >= hi_margin else "Medium-High"
        elif top_prob >= mh_thresh:
            best_12_conf = "Medium-High" if margin >= mh_margin else "Medium"
        elif top_prob >= med_thresh and margin >= med_margin:
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
            elif "Under" in ou_pick and combined_ou_avg > 60:
                if CONF_RANK.get(ou_conf, 99) > CONF_RANK["Medium"]:
                    ou_conf = "Medium"

        if ou_conf != "Low":
            ou_reason = f"exp goals {exp_total:.1f} model {p_o:.0%}o/{p_u:.0%}u"
            # Append Forebet agreement indicator
            if fb_over is not None and abs(fb_ou_diff) >= 0.15:
                ou_reason += f" fb{'✓' if fb_ou_diff > 0 else '✗'}"
            add("O/U", ou_pick, ou_conf, ou_reason,
                model_prob=p_o if "Over" in ou_pick else p_u)

    # ── BTTS (model-driven, with Dixon-Coles correlation) ──
    dc_rho = profile.get("dixon_coles_rho", -0.12)
    p_btss = prob_btts(exp_h, exp_a, rho=dc_rho)
    p_btn = 1.0 - p_btss

    value_yes = p_btss - 0.5
    value_no = p_btn - 0.5

    # Forebet BTTS cross-check
    fb_btts_yes = data.get("home_btts_yes_pct")
    fb_btts_no = data.get("home_btts_no_pct")

    # Higher threshold for YES (was 0.08) to reduce false positives
    if value_yes > 0.10 and value_yes >= value_no:
        btss_conf = conv_label(50 + int(value_yes * 80))
        if vol >= 0.25 and btss_conf in ("Near Certain", "High"): btss_conf = "Medium-High"
        elif vol >= 0.15 and btss_conf == "Near Certain": btss_conf = "High"
        if profile.get("avg_goals", 2.8) < 2.5 and btss_conf in ("Near Certain", "High"):
            btss_conf = "Medium-High"
        # Forebet cross-check: if stats show low BTTS %, cap confidence
        if fb_btts_yes is not None and fb_btts_yes < 40 and btss_conf in ("Near Certain", "High"):
            btss_conf = "Medium-High"
        btss_reason = f"model {p_btss:.0%}y/{p_btn:.0%}n"
        if fb_btts_yes:
            btss_reason += f" fb{fb_btts_yes}%"
        add("BTTS", "Yes", btss_conf, btss_reason, model_prob=p_btss)
    elif value_no > 0.08:
        btss_conf = conv_label(50 + int(value_no * 80))
        if vol >= 0.25 and btss_conf in ("Near Certain", "High"): btss_conf = "Medium-High"
        elif vol >= 0.15 and btss_conf == "Near Certain": btss_conf = "High"
        btss_reason = f"model {p_btss:.0%}y/{p_btn:.0%}n"
        if fb_btts_no:
            btss_reason += f" fb{fb_btts_no}%"
        add("BTTS", "No", btss_conf, btss_reason, model_prob=p_btn)

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
        "_warnings": warnings,
    }


# ─────────────────────────────────────────────
# Prediction runner
# ─────────────────────────────────────────────

def log(msg, end="\n"):
    """Print progress to stderr so stdout stays clean for JSON."""
    print(msg, end=end, file=sys.stderr, flush=True)


def _write_html(results, all_urls, compare_forebet, high_only):
    """Generate an HTML report of predictions."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    filtered = [r for r in results if r["confidence"] in ("Near Certain", "High")] if high_only else results

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

    def _c(val):
        m = {"Near Certain": "#22c55e", "High": "#3b82f6", "Medium-High": "#eab308", "Medium": "#f97316", "Low": "#ef4444"}
        return m.get(val, "#888")

    def _star(conf):
        return {3: "★★★", 2: "★★☆", 1: "★☆☆"}.get({"Near Certain": 3, "High": 2, "Medium-High": 1}.get(conf, 0), "")

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
            parts.append(f"O15: {ou15}%")
        btts_h = r.get("home_btts_yes_pct")
        if btts_h is not None:
            parts.append(f"BTTS: {btts_h}%")
        sot_h = r.get("home_shots_ontarget_pct")
        ts_h = r.get("home_total_shots_pg")
        if sot_h is not None and ts_h is not None:
            sot_est = round(ts_h * (sot_h / 100.0) * 0.70, 1)
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
        for p in r.get("all_picks") or []:
            mp = p.get("model_prob")
            mp_s = f"{mp:.0%}" if mp else ""
            vr = p.get("value_ratio")
            vr_s = f" ({vr:.2f})" if vr else ""
            picks_rows += f"<tr><td>{p['market']}</td><td>{p['pick']}</td><td>{mp_s}</td><td style='color:{_c(p['confidence'])}'>{p['confidence']}</td><td>{vr_s}</td></tr>\n"

        reason_html = ""
        if r.get("reasoning"):
            for reason in r["reasoning"][:4]:
                reason_html += f"<li>{reason}</li>\n"
            reason_html = f"<details><summary>Reasoning</summary><ul>{reason_html}</ul></details>"

        kelly_tag = f" &middot; Kelly: {r.get('kelly_stake', 0)*100:.1f}%" if r.get('kelly_stake', 0) > 0 else ""
        method_tag = f" &middot; {r.get('method', '')}" if r.get('method') else ""

        rows.append(f"""<div class="card" style="border-left: 4px solid {_c(r['confidence'])};">
<div class="card-header">
  <span class="teams">{r['home']} vs {r['away']}</span>
  <span class="conf-badge" style="background:{_c(r['confidence'])}">{_star(r['confidence'])} {r['confidence']}</span>
</div>
<div class="card-meta">{r.get('league', '')} &middot; {r.get('date', '')} &middot; <a href="{r['url']}">Forebet</a>{method_tag}</div>
<div class="card-body">
  <div class="pick-line"><strong>{r['pick']}</strong> ({r['market']}) &middot; Score lean: {r['score_lean'] or '—'} &middot; Exp: {exp_str}{kelly_tag}</div>
  <table>
    <tr><th>Home</th><td>Pos {r.get('home_pos', '—')}</td><td>Form {hf or '—'}</td><td>{r.get('odds_home', '—')}</td></tr>
    <tr><th>Draw</th><td></td><td></td><td>{r.get('odds_draw', '—')}</td></tr>
    <tr><th>Away</th><td>Pos {r.get('away_pos', '—')}</td><td>Form {af or '—'}</td><td>{r.get('odds_away', '—')}</td></tr>
  </table>
  <table><tr><th>O/U 2.5</th><td>{r.get('odds_over25', '—')}/{r.get('odds_under25', '—')}</td><th>BTTS</th><td>{r.get('odds_btts_yes', '—')}/{r.get('odds_btts_no', '—')}</td></tr></table>
  {"<p>H2H: " + str(r.get('h2h_home_wins', 0)) + "W-" + str(r.get('h2h_draws', 0)) + "D-" + str(r.get('h2h_away_wins', 0)) + "L &ndash; GF/GA: " + str(r.get('h2h_goals_for', 0)) + "/" + str(r.get('h2h_goals_against', 0)) + " &ndash; avg " + str(r.get('h2h_avg_total_goals', 0)) + " goals (" + str(r.get('h2h_matches', 0)) + " matches)</p>" if r.get('h2h_matches', 0) >= 3 else ""}
  {_venue_stats_html(r)}
  {reason_html}
  {("<table><tr><th>Market</th><th>Pick</th><th>Prob</th><th>Conf</th><th>Value</th></tr>" + picks_rows + "</table>") if picks_rows else ""}
</div>
</div>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Predictions — {now}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#0f172a; color:#e2e8f0; padding:20px; }}
h1 {{ font-size:1.4rem; margin-bottom:4px; }}
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
summary {{ cursor:pointer; color:#60a5fa; font-weight:500; }}
ul {{ margin:4px 0 0 18px; color:#94a3b8; }}
a {{ color:#60a5fa; }}
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
{"".join(rows)}
</body>
</html>"""
    path = Path("predictions.html")
    path.write_text(html)
    log(f"HTML report: {path.resolve()}")


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
        })

    # ── Output ──
    if json_out:
        json.dump(results, indent=2, ensure_ascii=False, fp=sys.stdout)
        return

    # ── HTML output ──
    if html_out:
        _write_html(results, match_urls, compare_forebet, high_only)
        return

    # Filter by confidence
    if high_only:
        results = [r for r in results if r["confidence"] in ("Near Certain", "High")]

    # Print
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

    def pad_visible(text: str, width: int, char: str = " ") -> str:
        """Pad a string to a given visible width, taking ANSI escapes into account."""
        v_len = visible_len(text)
        needed = max(0, width - v_len)
        return text + (char * needed)

    def color_form(form_str: str) -> str:
        """Color form string: W=green, D=yellow, L=red."""
        if not form_str or form_str == "—":
            return "\033[38;5;244m—\033[0m"
        res = []
        for char in form_str:
            if char == 'W':
                res.append("\033[1;32mW\033[0m")
            elif char == 'D':
                res.append("\033[1;33mD\033[0m")
            elif char == 'L':
                res.append("\033[1;31mL\033[0m")
            else:
                res.append(char)
        return "".join(res)

    for r in results:
        preds_made += 1
        print()

        W = 80
        C = W - 4  # content width between borders

        # Color setup
        conf = r['confidence']
        color = {"Near Certain": "\033[92m", "High": "\033[94m",
                 "Medium-High": "\033[93m", "Medium": "\033[93m", "Low": "\033[91m"}.get(conf, "")
        reset = "\033[0m"

        def box(content):
            padded = pad_visible(content, C)
            return f"\033[38;5;244m│\033[0m {padded} \033[38;5;244m│\033[0m"
        def hline(char="─"):
            return f"\033[38;5;244m├{char * C}┤\033[0m"
        def top():
            return f"\033[38;5;244m╭{'─' * C}╮\033[0m"
        def bottom():
            return f"\033[38;5;244m╰{'─' * C}╯\033[0m"

        # ── HEADER ──
        print(top())

        # Volatility badge
        vol_val = r.get('_volatility', 0)
        if vol_val >= 0.25:
            vol_tag = f"\033[1;31m⚡VOL\033[0m"
        elif vol_val >= 0.15:
            vol_tag = f"\033[1;33m⚡vol\033[0m"
        else:
            vol_tag = f"\033[38;5;244m⚡std\033[0m"

        print(box(f" \033[1m⚽ {r['home']} vs {r['away']}\033[0m "))
        print(box(f" \033[38;5;248m{r.get('league', '')}\033[0m  •  {vol_tag}  •  \033[38;5;248m{r.get('date', '')}\033[0m "))
        print(hline())

        # ── FORM + STANDINGS ──
        fmt_pos = lambda p: (f"{p}" if p else "—") + "th" if p else "—"
        hf = r.get('home_form', '')
        af = r.get('away_form', '')
        h_ppg = _ppg(hf) if sum(1 for c in hf if c in 'WDL') >= 3 else None
        a_ppg = _ppg(af) if sum(1 for c in af if c in 'WDL') >= 3 else None

        h_pos = r.get('home_pos', '—')
        a_pos = r.get('away_pos', '—')
        h_pos_str = f"#{h_pos}" if h_pos and h_pos != '—' else "—"
        a_pos_str = f"#{a_pos}" if a_pos and a_pos != '—' else "—"

        home_line = f" \033[38;5;248mHome:\033[0m \033[1m{h_pos_str:<3s}\033[0m  \033[38;5;248mForm:\033[0m {color_form(hf)}"
        if h_ppg:
            home_line += f"  \033[38;5;244m({h_ppg:.1f} ppg)\033[0m"
        away_line = f" \033[38;5;248mAway:\033[0m \033[1m{a_pos_str:<3s}\033[0m  \033[38;5;248mForm:\033[0m {color_form(af)}"
        if a_ppg:
            away_line += f"  \033[38;5;244m({a_ppg:.1f} ppg)\033[0m"
        print(box(home_line))
        print(box(away_line))
        print(hline())

        # ── PRIMARY PICK ──
        market_tag = f" ({r['market']})" if r['market'] else ""
        score_str = f"  •  \033[38;5;248mScore:\033[0m \033[1m{r['score_lean']}\033[0m" if r['score_lean'] else ""
        exp_str = ""
        if '_exp_goals' in r and r['_exp_goals']:
            eh, ea = r['_exp_goals']
            exp_str = f"  •  \033[38;5;248mExp:\033[0m \033[1m{eh:.1f}-{ea:.1f}\033[0m"
        method_str = f"  •  \033[38;5;244m[{r.get('method', '')}]\033[0m" if r.get('method') else ""
        kelly_str = f"  •  \033[1;32mKelly: {r.get('kelly_stake', 0)*100:.1f}%\033[0m" if r.get('kelly_stake', 0) > 0 else ""
        
        pick_part = f"\033[1;33m★\033[0m {color}\033[1m{r['pick']}{market_tag}\033[0m"
        conf_part = f"\033[1m{color}{conf}\033[0m"
        
        pick_line = f" {pick_part}  •  {conf_part}{score_str}{exp_str}{kelly_str}"
        print(box(pick_line))

        # Method line (separate to avoid overflow)
        if r.get('method'):
            method_line = f" \033[38;5;244mMethod:\033[0m \033[1m[{r.get('method', '')}]\033[0m"
            print(box(method_line))

        # ── ODDS ──
        nl = lambda v: f"{v:.2f}" if isinstance(v, (int, float)) else str(v) if v else "—"
        odds_12 = f"{nl(r.get('odds_home'))}/{nl(r.get('odds_draw'))}/{nl(r.get('odds_away'))}"
        odds_ou = f"{nl(r.get('odds_over25'))}/{nl(r.get('odds_under25'))}"
        odds_bt = f"{nl(r.get('odds_btts_yes'))}/{nl(r.get('odds_btts_no'))}"
        
        odds_line = (
            f" \033[38;5;248m1X2:\033[0m \033[1m{odds_12}\033[0m  •  "
            f"\033[38;5;248mO/U 2.5:\033[0m \033[1m{odds_ou}\033[0m  •  "
            f"\033[38;5;248mBTTS:\033[0m \033[1m{odds_bt}\033[0m"
        )
        print(box(odds_line))

        # ── Venue stats ──
        _vs_parts = []
        hh_gf = r.get("home_home_avg_goals_for")
        hh_ga = r.get("home_home_avg_goals_against")
        if hh_gf is not None:
            _vs_parts.append(f"H(H): {hh_gf:.1f}/{hh_ga:.1f}")
        aa_gf = r.get("away_away_avg_goals_for")
        aa_ga = r.get("away_away_avg_goals_against")
        if aa_gf is not None:
            _vs_parts.append(f"A(A): {aa_gf:.1f}/{aa_ga:.1f}")
        ou15 = r.get("home_over15_pct")
        if ou15 is not None:
            _vs_parts.append(f"O15: {ou15}%")
        btts_h = r.get("home_btts_yes_pct")
        if btts_h is not None:
            _vs_parts.append(f"BTTS: {btts_h}%")
        sot_h = r.get("home_shots_ontarget_pct")
        ts_h = r.get("home_total_shots_pg")
        if sot_h is not None and ts_h is not None:
            _vs_parts.append(f"SoT: {sot_h}%")
        cs_h = r.get("home_clean_sheets_pct")
        if cs_h is not None:
            _vs_parts.append(f"CS: {cs_h}%")
        if _vs_parts:
            vs_line = " \033[38;5;248mVenue:\033[0m " + "  ".join(_vs_parts)
            print(box(vs_line))

        # ── H2H ──
        hm = r.get('h2h_matches', 0)
        if hm >= 3:
            hw = r.get('h2h_home_wins', 0)
            hd = r.get('h2h_draws', 0)
            ha = r.get('h2h_away_wins', 0)
            h2h_gf = r.get('h2h_goals_for', 0)
            h2h_ga = r.get('h2h_goals_against', 0)
            h2h_avg = r.get('h2h_avg_total_goals', 0)
            h2h_line = (
                f" \033[38;5;248mH2H:\033[0m \033[1m{hw}W-{hd}D-{ha}L\033[0m  "
                f"\033[38;5;244m({hm} matches, GF/GA: {h2h_gf}/{h2h_ga}, avg {h2h_avg:.1f})\033[0m"
            )
            print(box(h2h_line))
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

            conf_colors = {
                'NC': '\033[1;32mNC\033[0m',
                'Hi': '\033[1;34mHi\033[0m',
                'MH': '\033[1;33mMH\033[0m',
                'Me': '\033[33mMe\033[0m',
                'Lo': '\033[31mLo\033[0m'
            }

            header_left = "\033[1;36mModel Pick\033[0m          \033[1;36mProb\033[0m \033[1;36mCn\033[0m  "
            header_right = "\033[1;35mForebet Pick\033[0m      \033[1;35mProb\033[0m"
            header_left_padded = pad_visible(header_left, 29)
            print(box(f" {header_left_padded} │ {header_right}"))
            
            divider_line = "\033[38;5;244m─────────────────────────────┼─────────────────────\033[0m"
            print(box(divider_line))

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

                star_styled = f"\033[1;33m★\033[0m" if star == "★" else " "
                pm_styled = f"\033[38;5;248m{pm:5s}\033[0m"
                mp_styled = f"\033[1m{mp_str:>4s}\033[0m" if mp_str else "    "
                pc_styled = conf_colors.get(pc, f"{pc:2s}")
                vr_styled = f"\033[38;5;244m{vr_str}\033[0m" if vr_str else ""

                # Forebet column
                fb_pick, fb_pct = _fb_for(pm)
                if fb_pick:
                    if pm == "O/U":
                        agree = pp.split()[0] == fb_pick
                    else:
                        agree = pp == fb_pick
                    
                    fb_pick_short = _short(fb_pick, 18)
                    fb_pct_str = f"{fb_pct:3.0f}%"
                    
                    if agree:
                        right = f"\033[1;32m{fb_pick_short:18s} {fb_pct_str} ✓\033[0m"
                        pp_styled = f"\033[1;32m{pp:11s}\033[0m"
                    else:
                        right = f"\033[38;5;248m{fb_pick_short:18s}\033[0m \033[1m{fb_pct_str}\033[0m"
                        pp_styled = f"\033[1m{pp:11s}\033[0m"
                else:
                    right = "\033[38;5;244m—\033[0m"
                    agree = False
                    pp_styled = f"\033[1m{pp:11s}\033[0m"

                left = f"{star_styled} {pm_styled} {pp_styled} {mp_styled} {pc_styled}{vr_styled}"
                left_padded = pad_visible(left, 29)
                print(box(f" {left_padded} │ {right}"))

        # ── REASONING ──
        if show_reasoning and r.get('reasoning'):
            for reason in r['reasoning'][:4]:
                if " — " in reason:
                    left_part, right_part = reason.split(" — ", 1)
                    styled_reason = f"\033[1m{left_part}\033[0m \033[38;5;244m— {right_part}\033[0m"
                else:
                    styled_reason = f"\033[38;5;244m{reason}\033[0m"
                print(box(f"  • {styled_reason}"))

        # ── DATA QUALITY WARNINGS ──
        warnings = r.get('_warnings', [])
        if warnings:
            print(box(f" \033[1;33m⚠\033[0m \033[33m{'  •  '.join(warnings)}\033[0m"))

        # ── FOOTER ──
        short_url = r['url'][:45] + "..." if len(r['url']) > 48 else r['url']
        tag = f"\033[38;5;244mID: {r['match_id']}  •  {short_url}\033[0m"
        print(box(tag))
        print(bottom())

    # Summary
    print()
    Ws = 80
    Cs = Ws - 4
    
    def s_box(content):
        v_len = visible_len(content)
        padding = max(0, Cs - v_len)
        return f"\033[38;5;244m│\033[0m {content}{' ' * padding} \033[38;5;244m│\033[0m"

    print(f"\033[38;5;244m╭{'─' * Cs}╮\033[0m")
    print(s_box(f"\033[1mSUMMARY STATISTICS\033[0m"))
    print(f"\033[38;5;244m├{'─' * Cs}┤\033[0m")
    
    conf_counts = {}
    for r in results:
        conf_counts[r['confidence']] = conf_counts.get(r['confidence'], 0) + 1

    pick_rate = (len(results)/len(match_urls)*100) if preds_made else 0
    print(s_box(f"Predictions made: \033[1m{preds_made}\033[0m ({pick_rate:.0f}% pick rate)"))
    
    for c in ["Near Certain", "High", "Medium-High", "Medium", "Low"]:
        if c in conf_counts:
            count = conf_counts[c]
            suffix = "match" if count == 1 else "matches"
            c_color = {"Near Certain": "\033[92m", "High": "\033[94m",
                       "Medium-High": "\033[93m", "Medium": "\033[93m", "Low": "\033[91m"}.get(c, "")
            print(s_box(f"  • {c_color}{c}\033[0m: \033[1m{count}\033[0m {suffix}"))
            
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
            agree_str = f"\033[1;32m✓ {agreements}/{total_fb} ({pct}%)\033[0m" if pct >= 50 else f"\033[1;31m✗ {agreements}/{total_fb} ({pct}%)\033[0m"
            print(s_box(f"Forebet 1X2 agreement: {agree_str}"))
            
    print(f"\033[38;5;244m╰{'─' * Cs}╯\033[0m")
    print(f"\nSaved to database: history.db")


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
  predict.py --calibrate             Show calibration/accuracy stats

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
    parser.add_argument("--calibrate", action="store_true", help="Show calibration stats")
    parser.add_argument("--high-only", action="store_true", help="Show only confident picks")
    parser.add_argument("--html", action="store_true", help="Output as HTML file (predictions.html)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--no-compare", action="store_true", help="Skip Forebet comparison")
    parser.add_argument("--no-reasoning", action="store_true", help="Hide reasoning")
    parser.add_argument("--no-ml", "--classic", action="store_true", help="Disable ML-enhanced prediction (use classic model)")

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
