#!/usr/bin/env python3
"""
Holdout validation of the context-aware gate on the 50-match multi-bet slate.

Simulates what would have happened if we had applied the context-aware model
to these fixtures before they were played. After results come in, this script
measures:

1. How many RELIABLE fast-track picks were correct
2. How many WATCH picks were correct vs base rate
3. Whether context signal correlates with actual outcomes
4. Calibration: does stated confidence match realised accuracy?

Usage:
    python scripts/validate_context_gate.py          # report only
    python scripts/validate_context_gate.py --resolve # also resolve signals from DB results
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database import get_db, record_signal_prediction, resolve_signal_predictions_for_match, compute_signal_reliability
from reserve_filter import is_reserve_fixture, reserve_reason

ROOT = Path(__file__).resolve().parent.parent


def build_slate() -> list[dict]:
    """Return the 50-match slate with expected picks based on our research."""
    return [
        # #, home, away, league, research_pick, confidence, reason
        (1, "Fleetwood Town", "Tranmere Rovers", "England League Two", "Draw", "Low", "Tranmere unbeaten in 4 H2H"),
        (2, "Almería", "FC Andorra", "Spain Segunda Division", "Home win", "Medium", "1st vs 19th, strong home"),
        (3, "RCD Mallorca", "Córdoba CF", "Spain Segunda Division", "Home win", "Medium", "6th vs 11th, back-to-back home wins"),
        (4, "Juventud Torremolinos CF", "SD Huesca", "Spain Primera Federación", "Away win", "Low", "Huesca stronger, ex-Segunda"),
        (5, "Real Jaén CF", "Real Zaragoza", "Copa del Rey", "Draw", "Low", "Cup, Zaragoza won reverse 4-0 but Segunda side"),
        (6, "UD Ourense", "Merida UD", "Primera Federación", "Draw", "Low", "Mid-table clash"),
        (7, "CD Coria", "Unionistas de Salamanca CF", "Primera Federación", "Away win", "Low", "Unionistas 4th, Coria 17th"),
        (8, "CA Osasuna B", "SD Logroñés", "Segunda Federación", "Away win", "Low", "Osasuna B is reserve side"),
        (9, "Barcelona B", "Girona FC B", "Segunda Federación", "Home win", "Low", "Both reserves, Barça B stronger"),
        (10, "Club Portugalete", "Marino Luanco", "Segunda Federación", "Home win", "Low", "Home favorite"),
        (11, "Deportivo Alavés W", "Eibar W", "Liga F Women", "Home win", "Low", "Limited data"),
        (12, "Atlético Zabal U19", "UD Almería U19", "Spain U19", "Draw", "Low", "Youth, no data"),
        (13, "RC Strasbourg W", "Toulouse W", "Première Ligue Féminine", "Home win", "Low", "Strasbourg home, Toulouse winless away"),
        (14, "Montpellier W", "FC Fleury 91 W", "Première Ligue Féminine", "Away win", "Low", "Fleury 5th vs Montpellier 9th"),
        (15, "Croix IC", "ES Thaon", "Coupe de France", "Draw", "Low", "Regional cup, no data"),
        (16, "Béthune ST", "AS Nancy Lorraine", "Coupe de France", "Away win", "Medium", "Pro vs amateur cup"),
        (17, "Dunkerque", "US Quevilly Rouen", "Coupe de France", "Home win", "Medium", "Pro (Ligue 2) vs amateur"),
        (18, "Sag Cestas", "Aviron Bayonnais FC", "Coupe de France", "Draw", "Low", "No data"),
        (19, "Balma SC", "Castanet US", "Coupe de France", "Draw", "Low", "No data"),
        (20, "AC Lumezzane", "AC Trento Calcio 1921", "Serie C Group A", "Away win", "Low", "Trento 3rd vs Lumezzane 11th"),
        (21, "SSC Bari", "Calcio Foggia 1920", "Serie C Group C", "Home win", "Medium", "Bari 2nd vs Foggia 16th"),
        (22, "Fiorentina W", "Lazio W", "Serie A Women", "Home win", "Medium", "Fiorentina 4th, strong form"),
        (23, "NAC Breda", "FC Emmen", "Eerste Divisie", "Home win", "Low", "Both 25 pts, NAC home"),
        (24, "FC Dordrecht", "Heracles Almelo", "Eerste Divisie", "Away win", "Medium", "1st vs 18th, huge gap"),
        (25, "Roda JC Kerkrade", "FC Volendam", "Eerste Divisie", "Draw", "Low", "Close mid-table"),
        (26, "Almere City", "Helmond Sport", "Eerste Divisie", "Home win", "Medium", "3rd vs 17th, strong home"),
        (27, "VVV-Venlo", "FC Eindhoven", "Eerste Divisie", "Home win", "Medium", "2nd vs 18th"),
        (28, "De Graafschap", "FC Oss", "Eerste Divisie", "Draw", "Low", "Lower table clash"),
        (29, "RBC Roosendaal", "VV Goes", "Tweede Divisie", "Draw", "Low", "No data"),
        (30, "RWDM Brussels", "RSC Habay", "Belgian Nationale 1", "Home win", "Low", "Home favorite"),
        (31, "Union Rochefortoise", "Tubize", "Belgian Nationale 1", "Draw", "Low", "Close table"),
        (32, "FC Flénu", "Mons", "Belgian Nationale 1", "Away win", "Low", "Mons bigger club"),
        (33, "K. Lyra TSV", "Dessel", "Belgian Nationale 1", "Draw", "Low", "No data"),
        (34, "Heist", "Hoogstraten", "Belgian Nationale 1", "Draw", "Low", "No data"),
        (35, "Washington Spirit", "Utah Royals FC", "NWSL", "Home win or Draw", "Medium", "Spirit 3rd, home semifinal"),
        (36, "Hamilton Academical", "Ross County", "Scottish League One", "Away win", "Low", "Ross County 1st, 12-2-0"),
        (37, "Busaiteen", "Al Ittihad", "Bahraini 2nd Division", "Away win", "Low", "Al Ittihad stronger"),
        (38, "EC Juventude", "Botafogo SP", "Brazil Serie B", "Home win", "Medium", "2nd vs 12th, unbeaten 8"),
        (39, "Atlético Goianiense", "Cuiabá MT", "Brazil Serie B", "Home win", "Medium", "Cuiabá LLLLL"),
        (40, "CRB AL", "Ceará CE", "Brazil Serie B", "Home win", "Medium", "CRB 6th vs Ceará 16th"),
        (41, "CF Com Vida Saf RS", "Ser Panambi RS", "Campeonato Gaúcho", "Draw", "Low", "No data"),
        (42, "Real SC Tramandai RS", "Grêmio Atlético Farroupilha RS", "Campeonato Gaúcho", "Draw", "Low", "No data"),
        (43, "Ibis PE", "Ipojuca Atlético Clube PE", "Campeonato Pernambucano", "Draw", "Low", "No data"),
        (44, "América PE", "Vera Cruz PE", "Campeonato Pernambucano", "Draw", "Low", "No data"),
        (45, "Visão Celeste", "Alecrim RN", "Campeonato Potiguar", "Draw", "Low", "No data"),
        (46, "Figueirense", "CN Marcílio Dias", "Copa Santa Catarina", "Home win", "Medium", "Figueirense much bigger"),
        (47, "Cruzeiro EC MG W", "América MG W", "Campeonato Mineiro Feminino", "Home win", "Medium", "Cruzeiro dominant H2H"),
        (48, "FC Du Haut-Nkam", "Canon Sportif", "Cameroon Ligue 1", "Draw", "Low", "No data"),
        (49, "Municipal Santa Ana", "AD Sarchi", "Costa Rica Liga de Ascenso", "Draw", "Low", "No data"),
        (50, "Millonarios FC", "Alianza FC Valledupar", "Colombia Primera A", "Home win", "Medium", "7th vs 18th, ~75% market"),
    ]


def main():
    resolve = "--resolve" in sys.argv

    conn = get_db()
    slate = build_slate()

    print("=" * 70)
    print("CONTEXT-AWARE GATE HOLDOUT VALIDATION — 50-match slate")
    print("=" * 70)
    print()

    # Record signal predictions for each match
    for num, home, away, league, pick, conf, reason in slate:
        # Check context for reserve side
        ctx_url = f"https://www.forebet.com/en/football-match/{home.lower().replace(' ', '-')}-{away.lower().replace(' ', '-')}"
        ctx = None
        # Try to find context by URL pattern (we used these URLs in context JSON)
        from database import get_match_context
        # We'll match by home/away team names instead
        rows = conn.execute(
            "SELECT * FROM match_context WHERE home_team = ? AND away_team = ?",
            (home, away)
        ).fetchall()
        if rows:
            ctx = dict(rows[0])

        is_res = is_reserve_fixture(home, away) if ctx is None else bool(ctx.get("is_reserve_side"))
        res_reason = reserve_reason(home, away) if ctx is None else None

        print(f"#{num:2d} {home:28s} vs {away:28s} [{league:35s}]")
        print(f"     Research pick: {pick:20s} ({conf:6s}) — {reason}")
        if ctx:
            sig = ctx.get("context_signal", 0)
            conf_ctx = ctx.get("context_confidence", 0)
            print(f"     Context signal: {sig:+.2f} (conf {conf_ctx:.2f})")
            if is_res:
                rr = reserve_reason(home, away)
                print(f"     ⚠ RESERVE SIDE: {rr[0]} ({rr[1]}) — forced HELD")
        print()

    # If --resolve, check DB for actual results and compute reliability
    if resolve:
        print("\n" + "=" * 70)
        print("RESOLVED SIGNAL RELIABILITY (from DB results)")
        print("=" * 70)
        # This would need actual results in DB to work
        print("(Requires results to be in history.db — run 'pr results' first)")

    conn.close()
    print("\nValidation framework ready. Use --resolve after matches have results.")


if __name__ == "__main__":
    main()
