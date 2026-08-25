#!/usr/bin/env python3
"""
Red-flag detection — data-quality and model/market sanity checks.

Born from the 21/08/2026 slate where all 3 losing picks showed pre-kickoff
warning signs that were silently ignored:

  1. Implausible scraped stat (Corinthians U20 avg GA 0.1 vs venue GA 1.0)
     -> Poisson built its 68% home probability on a broken number.
  2. Anomalous odds (Qadisiya 4.50 / 1.40 / 4.50 — draw shortest price while
     both sides 4.5x) -> market priced a tight low-scoring game, contradicting
     the BTTS Yes pick.
  3. Model-market divergence (Rudes 1.48 vs our Poisson 76% away) -> when the
     model and the bookies point opposite ways that hard, the market usually
     knows something form stats don't.

Two entry points:
  sanitize_scraped_stats(data)          Fix glitches in-place at scrape time.
  detect_red_flags(data, ...)           Surface remaining anomalies as flags.
"""

from typing import Dict, List, Optional

# Plausibility bounds for per-game averages over a meaningful sample.
_GF_MIN, _GF_MAX = 0.15, 5.5
_GA_MIN, _GA_MAX = 0.30, 5.0
_MIN_GAMES_FOR_BOUNDS = 2

# Overall-vs-venue divergence above this means one of the two is unreliable.
_VENUE_DIVERGENCE = 1.00

# Early-season noise threshold: standings samples this small are unreliable.
_SMALL_SAMPLE_GP = 3

# |model_prob - implied_prob| beyond this on the picked outcome = fading market.
_DIVERGENCE = 0.25


def _plausible(val, lo, hi) -> bool:
    return val is not None and lo <= val <= hi


def sanitize_scraped_stats(data: Dict) -> List[str]:
    """Fix implausible goal-average values in-place using venue-specific data.

    Forebet standings parsing can grab the wrong column for a team (seen live:
    Corinthians U20 overall GA parsed as 0.1/game while their home-venue GA was
    1.0).  When an overall average is outside plausible bounds and the
    venue-specific equivalent looks sane, replace it and record the fix.

    Returns a list of human-readable fix descriptions (also stored in
    data["stat_fixes"] so downstream code/logs can show what was corrected).
    """
    fixes: List[str] = []
    sides = (
        ("home",
         "home_avg_goals_for", "home_avg_goals_against", "home_games_played",
         "home_home_avg_goals_for", "home_home_avg_goals_against"),
        ("away",
         "away_avg_goals_for", "away_avg_goals_against", "away_games_played",
         "away_away_avg_goals_for", "away_away_avg_goals_against"),
    )

    for name, gf_key, ga_key, gp_key, v_gf_key, v_ga_key in sides:
        team_label = f"{name} team"

        # ── Goals against ──
        ga = data.get(ga_key)
        gp = data.get(gp_key)
        if (ga is not None and (ga < _GA_MIN or ga > _GA_MAX)
                and (gp is None or gp >= _MIN_GAMES_FOR_BOUNDS)):
            venue = data.get(v_ga_key)
            if _plausible(venue, _GA_MIN, _GA_MAX):
                fixes.append(
                    f"{team_label}: implausible avg GA {ga} replaced "
                    f"with venue-specific {venue}")
                data[ga_key] = venue
            else:
                fixes.append(f"{team_label}: implausible avg GA {ga} nulled")
                data[ga_key] = None

        # ── Goals for ──
        gf = data.get(gf_key)
        if (gf is not None and (gf < _GF_MIN or gf > _GF_MAX)
                and (gp is None or gp >= _MIN_GAMES_FOR_BOUNDS)):
            venue = data.get(v_gf_key)
            if _plausible(venue, _GF_MIN, _GF_MAX):
                fixes.append(
                    f"{team_label}: implausible avg GF {gf} replaced "
                    f"with venue-specific {venue}")
                data[gf_key] = venue
            else:
                fixes.append(f"{team_label}: implausible avg GF {gf} nulled")
                data[gf_key] = None

    if fixes:
        existing = data.get("stat_fixes") or []
        data["stat_fixes"] = existing + fixes

    return fixes


def detect_red_flags(
    data: Dict,
    market: Optional[str] = None,
    pick: Optional[str] = None,
    model_prob: Optional[float] = None,
    pick_odds: Optional[float] = None,
) -> List[Dict[str, str]]:
    """Detect remaining data anomalies and model/market conflicts.

    Call AFTER sanitize_scraped_stats has run.  Returns a list of
    {"severity": "critical"|"high", "msg": str}.  Critical = probable data
    glitch (should gate confidence); high = genuine warning sign (should be
    prominent and shrink stake).
    """
    flags: List[Dict[str, str]] = []

    # ── 1. Stat sanity ────────────────────────────────────────────────────
    for side in ("home", "away"):
        gp = data.get(f"{side}_games_played")
        label = "Home" if side == "home" else "Away"
        if gp is not None and gp <= _SMALL_SAMPLE_GP:
            flags.append({
                "severity": "high",
                "msg": (f"{label} standings sample tiny ({gp} games) — "
                        f"form/goal stats are early-season noise"),
            })
        ogf, oga = data.get(f"{side}_avg_goals_for"), data.get(f"{side}_avg_goals_against")
        vgf = data.get(f"{'home_home' if side == 'home' else 'away_away'}_avg_goals_for")
        vga = data.get(f"{'home_home' if side == 'home' else 'away_away'}_avg_goals_against")
        for overall, venue, kind in ((ogf, vgf, "GF"), (oga, vga, "GA")):
            if overall is not None and venue is not None and gp and gp >= _SMALL_SAMPLE_GP \
                    and abs(overall - venue) > _VENUE_DIVERGENCE:
                flags.append({
                    "severity": "high",
                    "msg": (f"{label} overall avg {kind} {overall} diverges from "
                            f"venue-specific {venue} — one source unreliable"),
                })

    # ── 2. Odds anomalies ─────────────────────────────────────────────────
    oh, od, oa = data.get("odds_home"), data.get("odds_draw"), data.get("odds_away")
    if all(v is not None for v in (oh, od, oa)):
        if min(oh, od, oa) <= 1.01:
            flags.append({
                "severity": "critical",
                "msg": f"Impossible odds ({oh}/{od}/{oa}) — scrape artifact",
            })
        elif abs(oh - od) < 1e-9 or abs(od - oa) < 1e-9:
            flags.append({
                "severity": "critical",
                "msg": f"Duplicate odds prices ({oh}/{od}/{oa}) — likely bad parse",
            })
        else:
            overround = 1.0 / oh + 1.0 / od + 1.0 / oa
            if overround > 1.25 or overround < 1.02:
                flags.append({
                    "severity": "high",
                    "msg": (f"Abnormal book margin (implied sum {overround:.0%}) — "
                            f"odds unreliable"),
                })
            # Draw priced far shorter than both sides = market expects a tight,
            # low-scoring contest — an anti-BTTS / anti-goals signal.
            if od < min(oh, oa) and min(oh, oa) / od >= 3.0:
                flags.append({
                    "severity": "high",
                    "msg": (f"Draw heavily favoured ({od} vs {oh}/{oa}) — market "
                            f"prices a tight low-scoring game"),
                })
    elif oh is None and od is None and oa is None:
        pass  # missing odds handled elsewhere ("No attack/defense data" etc.)

    # ── 3. Model-vs-market divergence on the picked outcome ───────────────
    if market and pick and model_prob and pick_odds and pick_odds > 1.0:
        implied = 1.0 / pick_odds
        gap = model_prob - implied
        if abs(gap) >= _DIVERGENCE:
            direction = "AGAINST the market" if gap < 0 else "vs market consensus"
            flags.append({
                "severity": "high",
                "msg": (f"Model-market divergence on {market} {pick}: model "
                        f"{model_prob:.0%} vs market {implied:.0%} — betting "
                        f"{direction}"),
            })
        elif abs(gap) >= 0.35:
            flags.append({
                "severity": "high",
                "msg": (f"Huge model-market gap on {market} {pick}: "
                        f"{model_prob:.0%} vs {implied:.0%} — verify odds freshness"),
            })

    return flags
