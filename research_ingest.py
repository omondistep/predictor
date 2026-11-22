#!/usr/bin/env python3
"""
Automated match-context research ingestion.

Fills the ``match_context`` table with motivation, reserve-side, cup stage,
pro/amateur, table position, and form data for upcoming fixtures.  This is the
data the model uses for context-aware gating and fast-track RELIABLE decisions.

Sources used (in priority order):
  1. history.db — resolved H2H and results already in the DB
  2. Forebet scrape — league table position, form, H2H from the match page
  3. Web research — manual entry via JSON for fixtures the scraper can't reach

Usage:
    python research_ingest.py --links links.txt          # ingest from link file
    python research_ingest.py --context context.json     # manual context JSON
    python research_ingest.py --url <single forebet url> # single fixture
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from database import get_db, save_match_context, get_match_context
from reserve_filter import is_reserve_fixture, reserve_reason

ROOT = Path(__file__).resolve().parent


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip()).strip()


def detect_competition_type(league: str) -> str:
    """Classify competition as league / cup / playoff / friendly / international."""
    t = (league or "").lower()
    if any(k in t for k in ("cup", "trophy", "challenge cup", "fa cup", "league cup")):
        return "cup"
    if any(k in t for k in ("playoff", "promotion", "relegation", "reduced")):
        return "playoff"
    if any(k in t for k in ("nations league", "euro", "world cup", "friendly",
                             "international", "qualifying")):
        return "international"
    return "league"


def detect_motivation(league: str, table_pos_home: int, table_pos_away: int,
                      table_pts_home: int, table_pts_away: int,
                      competition_type: str) -> str:
    """Infer motivation class from table position and competition type."""
    t = (league or "").lower()

    # Cup knockout matches are inherently high motivation
    if competition_type == "cup":
        return "high"

    # Playoff matches
    if competition_type == "playoff":
        return "high"

    # International fixtures
    if competition_type == "international":
        return "medium"

    # League matches: infer from table position
    # Bottom 3 vs top 3 = relegation battle / promotion chase = high
    # Mid-table with nothing to play for = low / dead_rubber
    if table_pos_home and table_pos_away:
        # Get total teams from context (rough estimate based on position)
        # If either side is in bottom 3 or top 3, it's high motivation
        if table_pos_home <= 3 or table_pos_away <= 3:
            return "high"
        if table_pos_home >= 18 or table_pos_away >= 18:  # relegation zone
            return "high"
        # Mid-table, nothing to play for
        if 8 <= table_pos_home <= 14 and 8 <= table_pos_away <= 14:
            return "low"

    return "medium"


def detect_pro_vs_amateur(home_team: str, away_team: str,
                          league: str) -> tuple:
    """Detect if this is a pro vs amateur matchup.

    Returns (pro_vs_amateur: bool, pro_side: 'home'|'away'|None).
    Heuristics:
      • League name contains 'national', 'regional', 'amateur', 'cup' with
        mixed-tier teams → likely pro vs amateur
      • One side is a well-known pro club, other is obscure → pro vs amateur
    """
    t = (league or "").lower()

    # Cup competitions often mix tiers
    if "cup" in t:
        # Heuristic: if one team name is much longer / more established
        # This is a weak signal; the real data comes from manual context
        return False, None

    # Reserve / B-team leagues are not pro vs amateur per se
    if "reserve" in t or "b " in t or " ii" in t:
        return False, None

    return False, None


def extract_context_from_forebet_data(data: dict) -> dict:
    """Extract match context from a Forebet scrape data dict.

    Reads whatever fields the scraper already populates: league, home/away
    team, positions, points, form, H2H, etc.  Missing fields are left None
    and filled later by manual context or web research.
    """
    ctx = {
        "home_team": data.get("home_team"),
        "away_team": data.get("away_team"),
        "league": data.get("league", ""),
        "forebet_url": data.get("forebet_url"),
        "competition_type": detect_competition_type(data.get("league", "")),
    }

    # Table data
    ctx["table_pos_home"] = data.get("home_pos")
    ctx["table_pos_away"] = data.get("away_pos")
    ctx["table_pts_home"] = data.get("home_pts")
    ctx["table_pts_away"] = data.get("away_pts")
    ctx["table_mp_home"] = data.get("home_games_played")
    ctx["table_mp_away"] = data.get("away_games_played")

    # Goal differential (from home/away avg goals for/against)
    h_gf = data.get("home_avg_goals_for")
    h_ga = data.get("home_avg_goals_against")
    a_gf = data.get("away_avg_goals_for")
    a_ga = data.get("away_avg_goals_against")
    if h_gf is not None and h_ga is not None and ctx.get("table_mp_home"):
        ctx["table_gd_home"] = round((h_gf - h_ga) * ctx["table_mp_home"])
    if a_gf is not None and a_ga is not None and ctx.get("table_mp_away"):
        ctx["table_gd_away"] = round((a_gf - a_ga) * ctx["table_mp_away"])

    # Form data from scraped form strings
    hf = data.get("home_form", "") or ""
    af = data.get("away_form", "") or ""
    if hf:
        ctx["form_home_w"] = hf.count("W")
        ctx["form_home_d"] = hf.count("D")
        ctx["form_home_l"] = hf.count("L")
    if af:
        ctx["form_away_w"] = af.count("W")
        ctx["form_away_d"] = af.count("D")
        ctx["form_away_l"] = af.count("L")

    # H2H from scraped data
    ctx["h2h_home_wins"] = data.get("h2h_home_wins", 0) or 0
    ctx["h2h_draws"] = data.get("h2h_draws", 0) or 0
    ctx["h2h_away_wins"] = data.get("h2h_away_wins", 0) or 0
    ctx["h2h_matches"] = data.get("h2h_matches", 0) or 0

    # Reserve detection
    if is_reserve_fixture(ctx["home_team"], ctx["away_team"]):
        ctx["is_reserve_side"] = 1
        reason = reserve_reason(ctx["home_team"], ctx["away_team"])
        if reason:
            ctx["reserve_side"] = reason[1]  # 'home' or 'away'

    # Pro vs amateur
    pv, pro_side = detect_pro_vs_amateur(
        ctx["home_team"], ctx["away_team"], ctx["league"])
    ctx["pro_vs_amateur"] = 1 if pv else 0
    ctx["pro_side"] = pro_side

    # Motivation
    ctx["motivation_class"] = detect_motivation(
        ctx["league"],
        ctx.get("table_pos_home") or 999,
        ctx.get("table_pos_away") or 999,
        ctx.get("table_pts_home") or 0,
        ctx.get("table_pts_away") or 0,
        ctx["competition_type"],
    )

    # Data quality: how many context fields we have
    filled = sum(1 for v in ctx.values() if v is not None and v != "" and v != 0)
    ctx["data_quality_score"] = min(filled / 20.0, 1.0)

    return ctx


def ingest_from_links_file(links_file: str, limit: int = None) -> int:
    """Read a links file and ingest context for each URL.

    For each URL, checks if context already exists; if not, creates a minimal
    context entry (league-based defaults) that can be enriched later by
    manual context or web research.
    """
    from predict import run_forebet_predictions  # noqa: F401  (lazy import)

    conn = get_db()
    count = 0
    with open(links_file) as f:
        urls = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    if limit:
        urls = urls[:limit]

    for url in urls:
        existing = get_match_context(forebet_url=url)
        if existing:
            continue

        # Minimal context from URL alone
        # We can't know teams without scraping, so create a placeholder
        # that will be filled when the match is predicted
        ctx = {
            "forebet_url": url,
            "league": "",
            "competition_type": "league",
            "motivation_class": "medium",
            "data_quality_score": 0.1,  # very low — no data yet
        }
        save_match_context(ctx)
        count += 1

    conn.close()
    print(f"Ingested {count} new context entries from {links_file}")
    return count


def ingest_from_context_json(context_file: str) -> int:
    """Load manual context from a JSON file and upsert into match_context.

    JSON format: list of objects with forebet_url + any context fields.
    Example:
      [
        {
          "forebet_url": "https://www.forebet.com/en/football-match/...",
          "home_team": "AFC Wimbledon",
          "away_team": "Blackpool",
          "league": "England League One",
          "competition_type": "league",
          "motivation_class": "high",
          "table_pos_home": 18,
          "table_pos_away": 19,
          "table_pts_home": 13,
          "table_pts_away": 12,
          ...
        }
      ]
    """
    with open(context_file) as f:
        contexts = json.load(f)

    if isinstance(contexts, dict):
        contexts = [contexts]

    count = 0
    for ctx in contexts:
        furl = ctx.get("forebet_url")
        if not furl:
            print(f"  Skipping entry without forebet_url: {ctx}")
            continue

        # Compute derived fields
        ctx["competition_type"] = ctx.get(
            "competition_type", detect_competition_type(ctx.get("league", "")))

        # Reserve detection
        if ctx.get("home_team") and ctx.get("away_team"):
            if is_reserve_fixture(ctx["home_team"], ctx["away_team"]):
                ctx["is_reserve_side"] = 1
                reason = reserve_reason(ctx["home_team"], ctx["away_team"])
                if reason:
                    ctx["reserve_side"] = reason[1]

        # Motivation if not provided
        if not ctx.get("motivation_class"):
            ctx["motivation_class"] = detect_motivation(
                ctx.get("league", ""),
                ctx.get("table_pos_home") or 999,
                ctx.get("table_pos_away") or 999,
                ctx.get("table_pts_home") or 0,
                ctx.get("table_pts_away") or 0,
                ctx["competition_type"],
            )

        # Data quality
        filled = sum(1 for v in ctx.values()
                     if v is not None and v != "" and v != 0 and v != False)
        ctx["data_quality_score"] = min(filled / 20.0, 1.0)

        save_match_context(ctx)
        count += 1

    print(f"Ingested {count} context entries from {context_file}")
    return count


def enrich_from_db_results(limit: int = 1000) -> int:
    """Enrich existing match_context rows with H2H and form data from
    resolved matches in history.db.

    For each context row with home/away teams and league, looks up resolved
    matches between those teams to fill H2H and recent form.
    """
    conn = get_db()
    rows = conn.execute("""
        SELECT mc.id, mc.forebet_url, mc.home_team, mc.away_team, mc.league
        FROM match_context mc
        WHERE (mc.h2h_matches IS NULL OR mc.h2h_matches = 0)
          AND mc.home_team IS NOT NULL AND mc.away_team IS NOT NULL
        LIMIT ?
    """, (limit,)).fetchall()

    count = 0
    for row in rows:
        mc_id, furl, home, away, league = row

        # H2H from resolved matches
        h2h = conn.execute("""
            SELECT home_team, away_team, actual_home_goals, actual_away_goals
            FROM matches
            WHERE ((home_team = ? AND away_team = ?)
                OR (home_team = ? AND away_team = ?))
              AND actual_home_goals IS NOT NULL
            ORDER BY match_date DESC LIMIT 10
        """, (home, away, away, home)).fetchall()

        if h2h:
            hw = d = aw = 0
            last = h2h[0]
            for r in h2h:
                is_home = (r["home_team"] == home)
                if is_home:
                    if r["actual_home_goals"] > r["actual_away_goals"]:
                        hw += 1
                    elif r["actual_home_goals"] == r["actual_away_goals"]:
                        d += 1
                    else:
                        aw += 1
                else:
                    if r["actual_away_goals"] > r["actual_home_goals"]:
                        hw += 1
                    elif r["actual_home_goals"] == r["actual_away_goals"]:
                        d += 1
                    else:
                        aw += 1

            h2h_last_result = "D"
            hg, ag = last["actual_home_goals"], last["actual_away_goals"]
            if last["home_team"] == home:
                if hg > ag: h2h_last_result = "W"
                elif hg < ag: h2h_last_result = "L"
            else:
                if ag > hg: h2h_last_result = "W"
                elif ag < hg: h2h_last_result = "L"

            conn.execute("""
                UPDATE match_context SET
                    h2h_home_wins = ?, h2h_draws = ?, h2h_away_wins = ?,
                    h2h_matches = ?, h2h_last_result = ?,
                    h2h_last_score = ?, last_updated = datetime('now')
                WHERE id = ?
            """, (hw, d, aw, len(h2h), h2h_last_result,
                  f"{hg}-{ag}", mc_id))
            count += 1

    conn.commit()
    conn.close()
    print(f"Enriched {count} context rows with H2H data from DB")
    return count


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Ingest match context for context-aware predictions")
    parser.add_argument("--links", help="Link file to ingest URLs from")
    parser.add_argument("--context", help="Manual context JSON file")
    parser.add_argument("--url", help="Single Forebet URL to create context for")
    parser.add_argument("--enrich-db", action="store_true",
                        help="Enrich existing contexts with DB H2H data")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max entries to process")
    args = parser.parse_args()

    if args.context:
        ingest_from_context_json(args.context)
    elif args.url:
        ctx = {"forebet_url": args.url, "competition_type": "league",
               "motivation_class": "medium", "data_quality_score": 0.1}
        save_match_context(ctx)
        print(f"Created context for {args.url}")
    elif args.links:
        ingest_from_links_file(args.links, args.limit)
    elif args.enrich_db:
        enrich_from_db_results(args.limit or 1000)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
