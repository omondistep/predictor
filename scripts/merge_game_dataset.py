#!/usr/bin/env python3
"""Merge the rich game/ dataset into predictor's LEAGUE_PROFILES.

Reads 15,294 historical match records from /home/stdk/game/data/,
aggregates stats per league, and updates LEAGUE_PROFILES in predict.py
with real data — enhancing predictive power with actual match results.

Usage:
  python merge_game_dataset.py [--dry-run] [--min-matches N]
  python merge_game_dataset.py --dry-run --min-matches 10  (preview with threshold)
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GAME_DATA = Path("/home/stdk/game/data/historical_matches_combined.json")
PREDICT_PY = Path(__file__).parent / "predict.py"

# ---------------------------------------------------------------------------
# Known league patterns that map to existing predictor profiles
# Keyed by (country_prefix, league_keywords) for pattern matching
# ---------------------------------------------------------------------------

def profile_key_for(country: str, league: str) -> str | None:
    """Return a predictor profile key for a (country, league) pair.

    Returns None if the league should remain unmapped (goes to default).
    """
    c = country.lower().strip()
    l = league.lower().strip()
    both = f"{c} {l}"

    # ---- Reserve / Youth ----
    if any(kw in both for kw in [
        "reserve", "reserv", "u19", "u20", "u21", "u23", "primavera",
        "youth", "junior", "revelacao", "academy", "b team", "ii team"
    ]):
        return "reserve-leagues"

    # ---- Women's football ----
    if any(kw in both for kw in [
        "women", "(w)", " wfc", " w ", "nadeshiko", "mulan", "we league"
    ]):
        return "women-football"

    # ---- Specific known leagues that map to existing profiles ----
    # Only include entries that map to an actual profile key (not "default").
    # Leagues NOT in this dict will generate new profile keys if they have
    # enough matches, or fall into "default" otherwise.
    KNOWN = {
        # Sweden
        ("sweden", "allsvenskan"): "sweden-allsvenskan",
        ("sweden", "superettan"): "sweden-superettan",
        ("sweden", "ettan"): "sweden-ettan",
        # Finland
        ("finland", "veikkausliiga"): "finland-veikkausliiga",
        ("finland", "ykkonen"): "finland-ykkonen",
        ("finland", "kakkonen"): "finland-kakkonen",
        # Brazil top divisions
        ("brazil", "brasileiro serie a"): "brazil-serie-a",
        ("brazil", "brasileiro serie b"): "brazil-serie-b",
        ("brazil", "brasileiro serie c"): "brazil-serie-c",
        ("brazil", "brasileiro serie d"): "brazil-serie-d",
        # Brazil state championships → map to nearest Serie level
        ("brazil", "campeonato paulista"): "brazil-serie-a",
        ("brazil", "campeonato carioca"): "brazil-serie-a",
        ("brazil", "campeonato mineiro"): "brazil-serie-a",
        ("brazil", "campeonato gaúcho"): "brazil-serie-b",
        ("brazil", "campeonato paranaense"): "brazil-serie-b",
        ("brazil", "campeonato catarinense"): "brazil-serie-b",
        ("brazil", "campeonato baiano"): "brazil-serie-c",
        ("brazil", "campeonato pernambucano"): "brazil-serie-c",
        ("brazil", "campeonato cearense"): "brazil-serie-c",
        ("brazil", "campeonato goiano"): "brazil-serie-c",
        ("brazil", "campeonato matogrossense"): "brazil-serie-c",
        ("brazil", "campeonato paulista a2"): "brazil-serie-b",
        ("brazil", "campeonato paulista a3"): "brazil-serie-c",
        ("brazil", "campeonato sul-matogrossense"): "brazil-serie-d",
        ("brazil", "campeonato capixaba"): "brazil-serie-d",
        ("brazil", "campeonato paraibano"): "brazil-serie-d",
        ("brazil", "campeonato paraense"): "brazil-serie-d",
        ("brazil", "campeonato maranhense"): "brazil-serie-d",
        ("brazil", "campeonato piauiense"): "brazil-serie-d",
        ("brazil", "campeonato sergipano"): "brazil-serie-d",
        ("brazil", "campeonato alagoano"): "brazil-serie-d",
        ("brazil", "campeonato potiguar"): "brazil-serie-d",
        ("brazil", "campeonato rondoniense"): "brazil-serie-d",
        ("brazil", "campeonato roraimense"): "brazil-serie-d",
        ("brazil", "campeonato acreano"): "brazil-serie-d",
        ("brazil", "campeonato amazonense"): "brazil-serie-d",
        ("brazil", "campeonato amapaense"): "brazil-serie-d",
        ("brazil", "campeonato brasiliense"): "brazil-serie-d",
        ("brazil", "campeonato tocantinense"): "brazil-serie-d",
        ("brazil", "campeonato brasileiro u20"): "brazil-u20",
        # Argentina
        ("argentina", "liga profesional"): "argentina-b-nacional",
        ("argentina", "nacional b"): "argentina-b-nacional",
        ("argentina", "primera b metropolitana"): "argentina-primera-b",
        ("argentina", "primera c"): "argentina-primera-c",
        # Chile
        ("chile", "primera division"): "chile-primera",
        ("chile", "primera b"): "chile-primera-b",
        # Uruguay
        ("uruguay", "primera division"): "uruguay-primera",
        ("uruguay", "segunda division"): "uruguay-segunda",
        # Ecuador
        ("ecuador", "serie a"): "ecuador-serie-a",
        ("ecuador", "serie b"): "ecuador-serie-b",
        # Peru
        ("peru", "primera división"): "peru-primera",
        ("peru", "primera division"): "peru-primera",
        # Paraguay
        ("paraguay", "primera division"): "paraguay-primera",
        ("paraguay", "segunda division"): "paraguay-segunda",
        # Morocco
        ("morocco", "botola pro"): "morocco-botola",
        # Spain (lower tiers)
        ("spain", "segunda b"): "spain-segunda",
        ("spain", "tercera division"): "spain-segunda",
        ("spain", "primera division women"): "women-football",
        # USA
        ("usa", "usl championship"): "usl-championship",
        ("usa", "usl league one"): "usl-league-one",
        ("usa", "usl league two"): "usl-league-two",
        ("usa", "mls next pro"): "mls-next-pro",
        # NWSL
        ("usa", "national women"): "nwsl",
        ("usa", "nwsl"): "nwsl",
        # Austria
        ("austria", "landesliga"): "austria-landesliga",
        ("austria", "oberliga"): "austria-landesliga",
        ("austria", "regionalliga"): "austria-landesliga",
        # Iceland
        ("iceland", "league cup"): "iceland",
        ("iceland", "women"): "iceland-women",
        # Estonia
        ("estonia", "meistriliiga"): "estonia",
        # Georgia
        ("georgia", "erovnuli liga"): "georgia",
        # Lithuania
        ("lithuania", "super cup"): "lithuania",
        # Women's & reserve (specific leagues)
        ("mexico", "liga mx women"): "women-football",
        ("germany", "junioren bundesliga"): "reserve-leagues",
        ("germany", "dfb junioren pokal"): "reserve-leagues",
        ("germany", "bundesliga women"): "women-football",
        ("germany", "2. bundesliga women"): "women-football",
        ("italy", "serie a women"): "women-football",
        ("italy", "primavera 1"): "reserve-leagues",
        ("italy", "primavera 2"): "reserve-leagues",
        ("italy", "primavera 3"): "reserve-leagues",
        ("italy", "coppa italia women"): "women-football",
        ("italy", "coppa italia primavera"): "reserve-leagues",
        ("portugal", "liga revelacao"): "reserve-leagues",
        ("portugal", "liga revelacao u23"): "reserve-leagues",
        ("japan", "we league"): "women-football",
        ("japan", "nadeshiko"): "women-football",
        ("japan", "nadeshiko league 2"): "women-football",
        ("australia", "a-league women"): "women-football",
        ("czech republic", "u19 league"): "reserve-leagues",
        ("russia", "youth league"): "reserve-leagues",
        ("argentina", "reserve league"): "reserve-leagues",
        ("england", "premier league 2"): "reserve-leagues",
        ("england", "premier league cup"): "reserve-leagues",
        ("england", "efl trophy"): "reserve-leagues",
        ("netherlands", "knvb beker women"): "women-football",
        ("netherlands", "super league women"): "women-football",
        ("belgium", "super league women"): "women-football",
        ("switzerland", "super league women"): "women-football",
        ("switzerland", "nationalliga b women"): "women-football",
        ("spain", "primera division women"): "women-football",
        ("brazil", "campeonato brasileiro women"): "women-football",
        ("taiwan", "mulan football league women"): "women-football",
    }

    # Try exact match on normalized (country, league) tuple
    key = KNOWN.get((c, l))
    if key is not None:
        return key

    # Try partial match for leagues with " - " suffix (tercera groups, etc.)
    # For Spain's Tercera Division groups
    if "tercera division" in l:
        return "spain-segunda"
    if "segunda b" in l:
        return "spain-segunda"

    # Brazil state championships that aren't explicitly listed
    if c == "brazil" and "campeonato" in l:
        # Default Brazilian state champs → Serie D level
        return "brazil-serie-d"

    # Handle "?" country records — these are mostly AFC/CAF cups, friendlies, etc.
    if country == "?" or not country:
        return None

    # For any league not in KNOWN, return None to create a new profile key
    return None


def _slugify(text: str) -> str:
    """Convert text to a safe slug, handling special characters."""
    t = text.lower().strip()
    # Normalize unicode (ü→u, ö→o, ç→c, etc.)
    t = (t.replace("ü", "u").replace("ö", "o").replace("ä", "a")
          .replace("é", "e").replace("è", "e").replace("ê", "e")
          .replace("á", "a").replace("à", "a").replace("â", "a")
          .replace("í", "i").replace("ì", "i").replace("î", "i")
          .replace("ó", "o").replace("ò", "o").replace("ô", "o")
          .replace("ú", "u").replace("ù", "u").replace("û", "u")
          .replace("ñ", "n").replace("ç", "c").replace("ş", "s")
          .replace("ğ", "g").replace("ı", "i"))
    return re.sub(r'[^a-z0-9]+', '-', t).strip('-')


def make_profile_key(country: str, league: str) -> str:
    """Generate a new profile key from country + league name."""
    c_slug = _slugify(country)[:20]
    l_slug = _slugify(league)[:30]
    # Remove common words for brevity
    for w in ['division', 'league', 'football', 'premier', 'primera', 'primera',
              'professional', 'championship', 'campeonato']:
        l_slug = l_slug.replace(f'-{w}-', '-').replace(f'-{w}', '').replace(f'{w}-', '')
    l_slug = l_slug.strip('-')
    return f"{c_slug}-{l_slug}" if l_slug else c_slug


# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
def load_dataset() -> list:
    if not GAME_DATA.exists():
        print(f"ERROR: {GAME_DATA} not found")
        sys.exit(1)
    with open(GAME_DATA) as f:
        dataset = json.load(f)
    return [r for r in dataset if isinstance(r, dict) and "league_code" in r]


# ---------------------------------------------------------------------------
# Detect league using predictor's own detect_league (for short codes)
# ---------------------------------------------------------------------------
def load_detect_league():
    """Load detect_league from predict.py and return the function."""
    if not PREDICT_PY.exists():
        return None
    content = PREDICT_PY.read_text()
    fn_start = content.find("def detect_league")
    if fn_start < 0:
        return None
    fn_rest = content[fn_start:]
    fn_end = re.search(r"\n\ndef ", fn_rest)
    fn_code = fn_rest[:fn_end.start()] if fn_end else fn_rest
    ns = {}
    exec(compile(fn_code + "\ndetect_league_fn = detect_league", "detect.py", "exec"), ns)
    return ns.get("detect_league_fn")


# ---------------------------------------------------------------------------
# Aggregate stats by profile key
# ---------------------------------------------------------------------------
def _compute_stats(matches: list) -> dict:
    n = len(matches)
    total_goals = sum(m["home_goals"] + m["away_goals"] for m in matches)
    draws = sum(1 for m in matches if m["home_goals"] == m["away_goals"])
    home_wins = sum(1 for m in matches if m["home_goals"] > m["away_goals"])
    u25 = sum(1 for m in matches if m["home_goals"] + m["away_goals"] <= 2)
    btts_no = sum(1 for m in matches if m["home_goals"] == 0 or m["away_goals"] == 0)
    return {
        "matches": n,
        "avg_goals": round(total_goals / n, 2),
        "draw_rate": round(draws / n, 2),
        "home_win_rate": round(home_wins / n, 2),
        "u25_rate": round(u25 / n, 2),
        "btts_no_rate": round(btts_no / n, 2),
    }


def aggregate(dataset: list) -> dict:
    detect_league_fn = load_detect_league()

    # Group by resolved profile key
    by_profile = defaultdict(list)           # existing known profiles
    new_profile_groups = defaultdict(list)   # new candidate profiles (keyed by (country,league))
    default_contributions = []               # falls into "default"

    for r in dataset:
        home_score = r.get("home_score")
        away_score = r.get("away_score")
        if not r.get("has_result") or home_score is None or away_score is None:
            continue

        country = r.get("country", "?") or "?"
        league_name = r.get("league_name", "?") or "?"
        league_code = r.get("league_code", "")
        hs, aws = int(home_score), int(away_score)
        md = {"home_goals": hs, "away_goals": aws}

        # Strategy: detect_league(code) → profile_key_for(country, league)
        code_key = detect_league_fn(league_code) if detect_league_fn and league_code else None
        explicit_key = profile_key_for(country, league_name)

        if explicit_key is not None:
            # Known exact mapping (may be reserve, women, or an existing profile)
            by_profile[explicit_key].append(md)
        elif code_key and code_key != "default":
            # Mapped by short code (country prefix) to an existing profile
            by_profile[code_key].append(md)
        elif country == "?" or not country:
            # Unidentifiable → default
            default_contributions.append(md)
        else:
            # New league not yet in predictor profiles
            new_profile_groups[(country, league_name)].append(md)

    # Build aggregated dict
    aggregated = {}

    # Existing profiles
    for key, matches in by_profile.items():
        if matches:
            aggregated[key] = _compute_stats(matches)

    # New profiles — only if enough matches
    NEW_PROFILE_MIN = 30
    for (country, league_name), matches in sorted(new_profile_groups.items(),
                                                  key=lambda x: -len(x[1])):
        if len(matches) >= NEW_PROFILE_MIN:
            key = make_profile_key(country, league_name)
            # Avoid collision with existing key
            if key in aggregated:
                key = f"{key}-2"
            stats = _compute_stats(matches)
            stats["_country"] = country
            stats["_league"] = league_name
            aggregated[key] = stats
        else:
            default_contributions.extend(matches)

    # Default from all unmatched
    if default_contributions:
        aggregated["default"] = _compute_stats(default_contributions)

    new_count = sum(1 for k in aggregated if "_country" in aggregated[k])
    print(f"New profile keys created: {new_count}")
    return aggregated

    # Compute stats for each profile
    aggregated = {}
    for key, matches in list(by_profile.items()):
        n = len(matches)
        if n == 0:
            continue
        total_goals = sum(m["home_goals"] + m["away_goals"] for m in matches)
        draws = sum(1 for m in matches if m["home_goals"] == m["away_goals"])
        home_wins = sum(1 for m in matches if m["home_goals"] > m["away_goals"])
        u25 = sum(1 for m in matches if m["home_goals"] + m["away_goals"] <= 2)
        btts_no = sum(1 for m in matches if m["home_goals"] == 0 or m["away_goals"] == 0)

        aggregated[key] = {
            "matches": n,
            "avg_goals": round(total_goals / n, 2),
            "draw_rate": round(draws / n, 2),
            "home_win_rate": round(home_wins / n, 2),
            "u25_rate": round(u25 / n, 2),
            "btts_no_rate": round(btts_no / n, 2),
        }

    # Add new profiles (leagues with enough data to create a profile)
    new_profile_count = 0
    for (country, league_name), matches in sorted(new_profiles.items(), key=lambda x: -len(x[1])):
        n = len(matches)
        if n < 30:  # Minimum threshold for new profile
            default_contributions.extend(matches)
            continue
        total_goals = sum(m["home_goals"] + m["away_goals"] for m in matches)
        draws = sum(1 for m in matches if m["home_goals"] == m["away_goals"])
        home_wins = sum(1 for m in matches if m["home_goals"] > m["away_goals"])
        u25 = sum(1 for m in matches if m["home_goals"] + m["away_goals"] <= 2)
        btts_no = sum(1 for m in matches if m["home_goals"] == 0 or m["away_goals"] == 0)

        key = make_profile_key(country, league_name)
        # Avoid collision with existing keys
        if key in aggregated:
            key = f"{key}-2"

        aggregated[key] = {
            "matches": n,
            "avg_goals": round(total_goals / n, 2),
            "draw_rate": round(draws / n, 2),
            "home_win_rate": round(home_wins / n, 2),
            "u25_rate": round(u25 / n, 2),
            "btts_no_rate": round(btts_no / n, 2),
            "_country": country,
            "_league": league_name,
        }
        new_profile_count += 1

    # Compute "default" from all unmatched matches
    if default_contributions:
        n = len(default_contributions)
        total_goals = sum(m["home_goals"] + m["away_goals"] for m in default_contributions)
        draws = sum(1 for m in default_contributions if m["home_goals"] == m["away_goals"])
        home_wins = sum(1 for m in default_contributions if m["home_goals"] > m["away_goals"])
        u25 = sum(1 for m in default_contributions if m["home_goals"] + m["away_goals"] <= 2)
        btts_no = sum(1 for m in default_contributions if m["home_goals"] == 0 or m["away_goals"] == 0)

        aggregated["default"] = {
            "matches": n,
            "avg_goals": round(total_goals / n, 2),
            "draw_rate": round(draws / n, 2),
            "home_win_rate": round(home_wins / n, 2),
            "u25_rate": round(u25 / n, 2),
            "btts_no_rate": round(btts_no / n, 2),
        }

    print(f"\nNew profile keys created: {new_profile_count}")
    return aggregated


# ---------------------------------------------------------------------------
# Update LEAGUE_PROFILES in predict.py
# ---------------------------------------------------------------------------
def update_profiles(aggregated: dict, min_matches: int, dry_run: bool = False):
    if not PREDICT_PY.exists():
        print(f"ERROR: {PREDICT_PY} not found")
        return

    content = PREDICT_PY.read_text()
    original = content

    field_map = {
        "avg_goals": "avg_goals",
        "u25_rate": "u25_rate",
        "btts_no_rate": "btts_no_rate",
        "draw_rate": "draw_rate",
        "home_win_rate": "home_win_rate",
    }

    # Current profiles that should be updated (not skipped)
    skip_profiles = {"reserve-leagues", "iceland-women"}

    changed = 0
    added = 0

    for key, stats in sorted(aggregated.items()):
        if stats["matches"] < min_matches:
            continue
        if key == "default":
            continue  # Don't touch default
        if key in skip_profiles:
            continue

        profile_start = f'    "{key}":'
        idx = content.find(profile_start)

        if idx != -1:
            # Update existing profile
            line_end = content.find("\n", idx)
            line = content[idx:line_end]
            for stat_name, prof_key in field_map.items():
                old_val = stats[prof_key]
                pat = rf'("{prof_key}":\s*)[\d.]+'
                repl = rf"\g<1>{old_val}"
                new_line = re.sub(pat, repl, line)
                if new_line != line:
                    content = content[:idx] + content[idx:].replace(line, new_line, 1)
                    line = new_line
                    changed += 1
        else:
            # Add new profile entry before 'default'
            vol = 0.12
            if stats["avg_goals"] > 3.5:
                vol = 0.20
            elif stats["avg_goals"] < 2.0:
                vol = 0.08
            if stats.get("volatility"):
                vol = stats["volatility"]

            home_adv = 1.15
            if stats["home_win_rate"] > 0.55:
                home_adv = 1.20
            elif stats["home_win_rate"] < 0.40:
                home_adv = 1.10

            comment = ""
            if "_country" in stats:
                comment = f"  # {stats['_country']} - {stats['_league']}"

            new_entry = (
                f'    "{key}": '
                f'{{"avg_goals": {stats["avg_goals"]}, '
                f'"u25_rate": {stats["u25_rate"]}, '
                f'"btts_no_rate": {stats["btts_no_rate"]}, '
                f'"draw_rate": {stats["draw_rate"]}, '
                f'"home_win_rate": {stats["home_win_rate"]}, '
                f'"home_adv": {home_adv}, "volatility": {vol}}},{comment}\n'
            )
            default_idx = content.find('    "default":')
            if default_idx != -1:
                content = content[:default_idx] + new_entry + content[default_idx:]
                added += 1

    if dry_run:
        print(f"\n[DRY-RUN] Would update {changed} fields in existing profiles, add {added} new profiles.")
        return

    if content != original:
        PREDICT_PY.write_text(content)
        print(f"\nUpdated {PREDICT_PY.name}: {changed} field updates, {added} new profiles.")
    else:
        print("\nNo changes needed.")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def show_report(aggregated: dict, min_matches: int):
    print(f"\n{'Profile Key':45s} {'M':>5s} {'Gls':>6s} {'Draw':>6s} {'HW':>6s} {'U25':>6s} {'NoBTTS':>6s}  {'Note':>4s}")
    print("-" * 92)
    for key in sorted(aggregated):
        s = aggregated[key]
        if s["matches"] < min_matches:
            continue
        note = ""
        if key == "default":
            note = " ←"
        elif "_country" in s:
            note = " *"
        print(f"{key:45s} {s['matches']:5d} {s['avg_goals']:6.2f} {s['draw_rate']:6.2f} {s['home_win_rate']:6.2f} {s['u25_rate']:6.2f} {s['btts_no_rate']:6.2f}  {note:>4s}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Merge game/ dataset into predictor LEAGUE_PROFILES")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing")
    parser.add_argument("--min-matches", type=int, default=5, help="Minimum matches per profile (default: 5)")
    args = parser.parse_args()

    print("=" * 60)
    print("Merge Game Dataset → Predictor LEAGUE_PROFILES")
    print("=" * 60)

    dataset = load_dataset()
    print(f"Loaded {len(dataset)} match records from {GAME_DATA.name}")

    aggregated = aggregate(dataset)
    print(f"\nAggregated stats for {len(aggregated)} profile keys:")
    show_report(aggregated, args.min_matches)

    update_profiles(aggregated, args.min_matches, dry_run=args.dry_run)

    if args.dry_run:
        print("\nUse --dry-run to preview. Remove it to apply changes.")


if __name__ == "__main__":
    main()
