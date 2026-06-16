#!/usr/bin/env python3
"""
historical_calibrate — scrape past predictions-1x2 pages for results,
aggregate stats by league, and update LEAGUE_PROFILES in predict.py.

Scrapes each date from 2026-05-25 to 2026-06-14 from:
  https://www.forebet.com/en/football-predictions/predictions-1x2/YYYY-MM-DD

Extracts match scores and league info, aggregates, updates profiles.

Usage:
  python historical_calibrate.py [--dry-run] [--min-matches N] [--start DATE] [--end DATE]
"""
import re, json, sys, os, time, textwrap
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

PREDICT_PY = Path(__file__).parent / "predict.py"

# ---------------------------------------------------------------------------
# League detection (mirrors predict.py detect_league + cnfupdate.py)
# ---------------------------------------------------------------------------

def detect_league(text: str) -> str:
    t = text.lower()
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
    if "brazil" in t or "brasil" in t:
        if "u20" in t: return "brazil-u20"
        if "serie d" in t: return "brazil-serie-d"
        if "serie c" in t: return "brazil-serie-c"
        if "serie b" in t: return "brazil-serie-b"
        if "serie a" in t: return "brazil-serie-a"
    if "argentina" in t:
        if "b nacional" in t: return "argentina-b-nacional"
        if "primera b" in t: return "argentina-primera-b"
        if "primera c" in t: return "argentina-primera-c"
        if "federal a" in t: return "argentina-federal-a"
    if "chile" in t:
        if "primera b" in t: return "chile-primera-b"
        if "primera" in t: return "chile-primera"
    if "usa" in t or "usl" in t:
        if "championship" in t: return "usl-championship"
        if "league one" in t: return "usl-league-one"
        if "league two" in t: return "usl-league-two"
        if "mls next pro" in t or "mls" in t: return "mls-next-pro"
    if "nwsl" in t: return "nwsl"
    if "austria" in t: return "austria-landesliga"
    if "uruguay" in t:
        if "segunda" in t: return "uruguay-segunda"
        return "uruguay-primera"
    if "ecuador" in t:
        if "serie b" in t: return "ecuador-serie-b"
        return "ecuador-serie-a"
    if "peru" in t: return "peru-primera"
    if "paraguay" in t:
        if "segunda" in t: return "paraguay-segunda"
        return "paraguay-primera"
    if "sweden" in t or "sverige" in t:
        if "allsvenskan" in t: return "sweden-allsvenskan"
        if "superettan" in t: return "sweden-superettan"
        if "ettan" in t: return "sweden-ettan"
        return "sweden-division-2"
    if "finland" in t or "suomi" in t:
        if "veikkausliiga" in t: return "finland-veikkausliiga"
        if "ykkonen" in t: return "finland-ykkonen"
        return "finland-kakkonen"
    if "morocco" in t or "botola" in t: return "morocco-botola"
    if "spain" in t or "espana" in t:
        if "segunda" in t: return "spain-segunda"
    if (" w" in t or " women" in t or "(w)" in t):
        return "women-football"
    if "iceland" in t: return "iceland"
    if "estonia" in t: return "estonia"
    if "georgia" in t: return "georgia"
    if "lithuania" in t: return "lithuania"
    if "reserve" in t or "u21" in t or "u23" in t: return "reserve-leagues"
    return "default"


def resolve_league_from_onclick(onclick: str) -> str:
    onclick = onclick.strip()
    m = re.search(r"getstag\(this,\d+,'([^']*)','([^']*)'", onclick)
    if m:
        country = m.group(1)
        league = m.group(2)
        return detect_league(f"{country} {league}")
    m2 = re.search(r"getstag\(this,\d+,'([^']*)'", onclick)
    if m2:
        return detect_league(m2.group(1))
    return "default"


# ---------------------------------------------------------------------------
# Scrape a single date page
# ---------------------------------------------------------------------------

def scrape_date_page(date_str: str) -> list:
    url = f"https://www.forebet.com/en/football-predictions/predictions-1x2/{date_str}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            print(f"  [{date_str}] HTTP {r.status_code}")
            return []
    except Exception as e:
        print(f"  [{date_str}] Error: {e}")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    matches = []

    rows = soup.find_all("div", {"class": re.compile(r"rcnt tr_\d")})
    if not rows:
        rows = soup.find_all("div", {"class": "rcnt"})

    for row in rows:
        link_tag = row.find("a", {"class": "tnmscn"})
        if not link_tag:
            continue
        match_url = link_tag.get("href", "")
        if not match_url.startswith("http"):
            match_url = "https://www.forebet.com" + match_url

        score_el = row.find(["b", "span"], {"class": re.compile(r"l_scr")})
        if not score_el:
            continue
        score_text = score_el.get_text(strip=True)
        m = re.search(r"(\d+)\s*[-–:]\s*(\d+)", score_text)
        if not m:
            continue
        home_goals = int(m.group(1))
        away_goals = int(m.group(2))

        shortag = row.find(["div", "span"], {"class": re.compile(r"shortagDiv")})
        league_key = "default"
        if shortag:
            img = shortag.find("img")
            if img:
                onclick = img.get("onclick", "")
                if onclick:
                    league_key = resolve_league_from_onclick(onclick)

        matches.append({
            "url": match_url,
            "league_key": league_key,
            "date": date_str,
            "home_goals": home_goals,
            "away_goals": away_goals,
        })

    return matches


# ---------------------------------------------------------------------------
# Aggregate by league
# ---------------------------------------------------------------------------

def aggregate(results: list) -> dict:
    by_profile = defaultdict(list)
    for r in results:
        by_profile[r["league_key"]].append(r)

    aggregated = {}
    for key, matches in sorted(by_profile.items()):
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
    return aggregated


# ---------------------------------------------------------------------------
# Update LEAGUE_PROFILES in predict.py
# ---------------------------------------------------------------------------

def update_profiles(aggregated: dict, min_matches: int, dry_run: bool = False, no_update_existing: bool = False):
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

    changed = 0
    added = 0

    skip_profiles = {"default", "reserve-leagues"}

    for key, stats in aggregated.items():
        if stats["matches"] < min_matches:
            continue
        if key in skip_profiles:
            continue

        profile_start = f'    "{key}":'
        idx = content.find(profile_start)

        if idx != -1:
            if no_update_existing:
                continue
            line_end = content.find("\n", idx)
            line = content[idx:line_end]
            for stat_name, prof_key in field_map.items():
                old_val = stats[prof_key]
                pat = rf'("{prof_key}":\s*)[\d.]+'
                repl = rf'\g<1>{old_val}'
                new_line = re.sub(pat, repl, line)
                if new_line != line:
                    content = content[:idx] + content[idx:].replace(line, new_line, 1)
                    line = new_line
                    changed += 1
        else:
            vol = 0.08 if stats["draw_rate"] > 0.30 else 0.12
            home_adv = 1.15
            new_entry = (
                f'    "{key}": '
                f'{{"avg_goals": {stats["avg_goals"]}, '
                f'"u25_rate": {stats["u25_rate"]}, '
                f'"btts_no_rate": {stats["btts_no_rate"]}, '
                f'"draw_rate": {stats["draw_rate"]}, '
                f'"home_win_rate": {stats["home_win_rate"]}, '
                f'"home_adv": {home_adv}, "volatility": {vol}}},\n'
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
        print(f"\nUpdated {PREDICT_PY.name}: {changed} field changes, {added} new profiles added.")
    else:
        print("\nNo changes needed.")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def show_report(aggregated: dict, min_matches: int):
    print(f"\n{'Profile Key':35s} {'M':>4s} {'Gls':>5s} {'Draw':>5s} {'HW':>5s} {'U25':>5s} {'NoBTTS':>6s}")
    print("-" * 65)
    for key in sorted(aggregated):
        s = aggregated[key]
        if s["matches"] < min_matches:
            continue
        print(f"{key:35s} {s['matches']:4d} {s['avg_goals']:5.2f} {s['draw_rate']:5.2f} {s['home_win_rate']:5.2f} {s['u25_rate']:5.2f} {s['btts_no_rate']:6.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def ensure_alias():
    bindir = Path.home() / ".local" / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve()
    for name in ("histcal", "historical-calibrate"):
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Scrape historical predictions pages and calibrate profiles")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing")
    parser.add_argument("--min-matches", type=int, default=5, help="Minimum matches per league (default: 5)")
    parser.add_argument("--start", default="2026-05-25", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-06-14", help="End date (YYYY-MM-DD)")
    parser.add_argument("--no-update-existing", action="store_true", help="Don't update existing profiles, only add new ones")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between page requests in seconds (default: 1.0)")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    dates = []
    d = start
    while d <= end:
        dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)

    print(f"Scraping {len(dates)} date pages: {args.start} to {args.end}")
    print(f"Delay: {args.delay}s between requests")
    print(f"Minimum matches per league: {args.min_matches}")
    if args.dry_run:
        print("[DRY-RUN mode]")

    all_results = []
    for i, date_str in enumerate(dates):
        print(f"[{i+1}/{len(dates)}] {date_str} ...", end=" ", flush=True)
        matches = scrape_date_page(date_str)
        print(f"{len(matches)} matches")
        all_results.extend(matches)

        # Be respectful: delay between pages
        if i < len(dates) - 1:
            time.sleep(args.delay)

    print(f"\nTotal matches scraped: {len(all_results)}")

    if not all_results:
        print("No results found. Exiting.")
        return

    aggregated = aggregate(all_results)
    print(f"Unique profile keys: {len(aggregated)}")
    show_report(aggregated, args.min_matches)

    # Also show which leagues have too few matches
    skipped = {k: v for k, v in aggregated.items() if v["matches"] < args.min_matches}
    if skipped:
        print(f"\nLeagues with <{args.min_matches} matches (skipped from update):")
        for k, v in sorted(skipped.items(), key=lambda x: x[1]["matches"], reverse=True):
            print(f"  {k:35s} {v['matches']} matches")

    update_profiles(aggregated, args.min_matches, dry_run=args.dry_run, no_update_existing=args.no_update_existing)

    if not args.dry_run:
        ensure_alias()
        print("\nDone. Profiles updated in predict.py")


if __name__ == "__main__":
    main()
