#!/usr/bin/env python3
"""
cnfupdate — scrape remaining match URLs, aggregate stats by league, update profiles.

Usage:
  cnfupdate [--dry-run] [--min-matches N]

The alias `cnfupdate` points to this script.
"""
import re, json, sys, os, time, textwrap
from collections import defaultdict, Counter
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ── Config ─────────────────────────────────────
PREDICT_PY = Path(__file__).parent / "predict.py"
PLAYED_TXT = Path(__file__).parent / "played.txt"
RESULTS_JSON = Path(__file__).parent / "scraped_results.json"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Forebet diff_league → profile key mapping for leagues not covered by detect_league
MANUAL_MAP = {
    "Br4": "brazil-serie-d",
    "Br3": "brazil-serie-c",
    "Br2": "brazil-serie-b",
    "Br1": "brazil-serie-a",
    "Uy2": "uruguay-segunda",
    "Uy1": "uruguay-primera",
    "AtL": "austria-landesliga",
    "ArR": "reserve-leagues",
    # extra fallbacks for codes detect_league handles
}

# ── Detection (mirrors predict.py detect_league) ──

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
    # Colombia
    if t.startswith("co"):
        if "2" in t[:4] or "b" in t[:4]: return "colombia-b"
        return "colombia-a"
    # Mexico
    if t.startswith("mx"):
        if "w" in t[:4]: return "women-football"
        if "2" in t[:4]: return "mexico-liga-de-expansion-mx"
        if "3" in t[:4] or "4" in t[:4]: return "mexico-liga-serie-a"
        return "mexico-liga-mx"
    # Venezuela
    if t.startswith("ve"):
        if "2" in t[:4]: return "default"
        return "venezuela-primera"
    # Guatemala
    if t.startswith("gt"):
        if "1" in t[:4]: return "guatemala-liga-nacional"
        return "guatemala-primera"
    # El Salvador
    if t.startswith("sv"):
        return "el-salvador-primera"
    # Honduras
    if t.startswith("hn"):
        return "honduras-liga-nacional"
    # Nicaragua
    if t.startswith("ni"):
        return "nicaragua-primera"
    # Costa Rica
    if t.startswith("cr"):
        if "1" in t[:4]: return "default"
        return "costa-rica-liga-de-ascenso"
    # Panama
    if t.startswith("pa"):
        return "panama-football"
    # Libya
    if t.startswith("ly"):
        return "libya-premier"
    # Sudan
    if t.startswith("sd"):
        return "sudan-premier"
    # Syria
    if t.startswith("sy"):
        return "syria-premier"
    # DR Congo
    if t.startswith("cd"):
        return "dr-congo-ligue-1"
    # Saudi Arabia
    if t.startswith("sa"):
        if "1" in t[:4]: return "default"
        return "saudi-arabia-1st"
    # Turkey (Turkiye)
    if t.startswith("tr"):
        if "1" in t[:4]: return "default"
        if "2" in t[:4]: return "default"
        if "3" in t[:4]: return "turkiye-tff-3-lig"
        if "4" in t[:4]: return "turkiye-tff-3-lig"
        if "c" in t[-1:].lower(): return "default"
        return "turkiye-tff-3-lig"
    # Thailand
    if t.startswith("th"):
        if "1" in t[:4]: return "default"
        if "2" in t[:4]: return "default"
        if "3" in t[:4]: return "thailand-thai-3"
        if "c" in t[-1:].lower(): return "default"
        if "l" in t[-1:].lower(): return "default"
        return "thailand-thai-3"
    # Algeria
    if t.startswith("dz"):
        if "1" in t[:4]: return "default"
        return "algeria-ligue-2"
    # Full name checks
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
        if "mls" in t: return "mls-next-pro"
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
    # Colombia
    if "colombia" in t:
        if "primera b" in t or "segunda" in t: return "colombia-b"
        return "colombia-a"
    # Mexico
    if "mexico" in t or "mx" in t[:3]:
        if "women" in t: return "women-football"
        if "expansion" in t: return "mexico-liga-de-expansion-mx"
        if "premier" in t: return "mexico-liga-serie-a"
        if "liga mx" in t: return "mexico-liga-mx"
    # Venezuela
    if "venezuela" in t:
        if "segunda" in t: return "default"
        return "venezuela-primera"
    # Guatemala
    if "guatemala" in t:
        if "liga nacional" in t: return "guatemala-liga-nacional"
        if "primera" in t: return "guatemala-primera"
    # El Salvador
    if "el salvador" in t:
        return "el-salvador-primera"
    # Honduras
    if "honduras" in t:
        return "honduras-liga-nacional"
    # Nicaragua
    if "nicaragua" in t:
        return "nicaragua-primera"
    # Costa Rica
    if "costa rica" in t:
        if "ascenso" in t: return "costa-rica-liga-de-ascenso"
        return "default"
    # Panama
    if "panama" in t:
        return "panama-football"
    # Libya
    if "libya" in t:
        return "libya-premier"
    # Sudan
    if "sudan" in t:
        return "sudan-premier"
    # Syria
    if "syria" in t:
        return "syria-premier"
    # DR Congo
    if "dr congo" in t:
        return "dr-congo-ligue-1"
    # Saudi Arabia
    if "saudi" in t:
        if "professional" in t or "1st" in t: return "default"
        return "saudi-arabia-1st"
    # Turkey (Turkiye)
    if "turkiye" in t or "türkiye" in t or "turkey" in t:
        if "super lig" in t: return "default"
        if "1. lig" in t or "tff 1" in t: return "default"
        if "2. lig" in t or "tff 2" in t: return "default"
        if "3. lig" in t or "tff 3" in t: return "turkiye-tff-3-lig"
        if "kupasi" in t: return "default"
        return "turkiye-tff-3-lig"
    # Thailand
    if "thailand" in t or "thai" in t:
        if "premier" in t or "league 1" in t: return "default"
        if "league 2" in t: return "default"
        if "league 3" in t: return "thailand-thai-3"
        if "fa cup" in t or "league cup" in t: return "default"
        return "thailand-thai-3"
    # Algeria
    if "algeria" in t or "algerie" in t:
        if "ligue 1" in t: return "default"
        return "algeria-ligue-2"
    if "reserve" in t or "u21" in t or "u23" in t: return "reserve-leagues"
    return "default"


# ── Scrape ────────────────────────────────────

def scrape_one(url: str) -> dict | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        lel = soup.find(["span", "div"], {"class": "diff_league"})
        league = lel.get_text(strip=True) if lel else "UNKNOWN"
        lscr = soup.find(["b", "span"], {"class": re.compile(r"l_scr|l_score")})
        if not lscr:
            return None
        m = re.search(r"(\d+)\s*-\s*(\d+)", lscr.get_text(strip=True))
        if not m:
            return None
        return {
            "url": url,
            "league_code": league,
            "home_goals": int(m.group(1)),
            "away_goals": int(m.group(2)),
        }
    except Exception:
        return None


def do_scrape(dry_run: bool = False) -> list:
    results_path = RESULTS_JSON
    results = []
    done_urls = set()
    if results_path.exists():
        try:
            existing = json.loads(results_path.read_text())
            results = existing
            done_urls = {r["url"] for r in results}
        except (json.JSONDecodeError, Exception):
            pass

    urls = PLAYED_TXT.read_text().strip().splitlines()
    todo = [u for u in urls if u not in done_urls]
    if not todo:
        print(f"All {len(results)} URLs already scraped.")
        return results

    print(f"Scraping {len(todo)} remaining URLs (0.3s delay) ...")
    for i, url in enumerate(todo):
        data = scrape_one(url)
        if data:
            results.append(data)
        if (i + 1) % 20 == 0:
            if not dry_run:
                results_path.write_text(json.dumps(results, indent=2))
            print(f"  [{i+1}/{len(todo)}] ({len(results)} total)")
        sys.stdout.flush()
        time.sleep(0.3)

    if not dry_run:
        results_path.write_text(json.dumps(results, indent=2))
    print(f"Done. Total scraped: {len(results)}")
    return results


# ── Aggregate ─────────────────────────────────

def aggregate(results: list) -> dict:
    """Aggregate stats by profile key, return {profile_key: stats}."""
    by_profile = defaultdict(list)
    for r in results:
        key = detect_league(r["league_code"])
        by_profile[key].append(r)

    aggregated = {}
    for key, matches in sorted(by_profile.items()):
        n = len(matches)
        total_goals = sum(m["home_goals"] + m["away_goals"] for m in matches)
        draws = sum(1 for m in matches if m["home_goals"] == m["away_goals"])
        home_wins = sum(1 for m in matches if m["home_goals"] > m["away_goals"])
        u25 = sum(1 for m in matches if m["home_goals"] + m["away_goals"] <= 2)
        btts_no = sum(1 for m in matches if m["home_goals"] == 0 or m["away_goals"] == 0)
        aggregated[key] = {
            "matches": n,
            "avg_goals": round(total_goals / n, 2) if n else 0,
            "draw_rate": round(draws / n, 2) if n else 0,
            "home_win_rate": round(home_wins / n, 2) if n else 0,
            "u25_rate": round(u25 / n, 2) if n else 0,
            "btts_no_rate": round(btts_no / n, 2) if n else 0,
        }
    return aggregated


# ── Clean played.txt ──────────────────────────

def clean_played():
    """Remove processed and malformed URLs from played.txt."""
    if not PLAYED_TXT.exists():
        return
    urls = PLAYED_TXT.read_text().strip().splitlines()

    # Remove URLs that don't end with -<7 digits>
    valid_pat = re.compile(r"-\d{7}$")
    urls = [u for u in urls if valid_pat.search(u)]

    # Remove URLs already in scraped_results.json
    done_urls = set()
    if RESULTS_JSON.exists():
        try:
            done_urls = {r["url"] for r in json.loads(RESULTS_JSON.read_text())}
        except (json.JSONDecodeError, Exception):
            pass
    urls = [u for u in urls if u not in done_urls]

    PLAYED_TXT.write_text("\n".join(urls) + ("\n" if urls else ""))
    print(f"Trimmed {PLAYED_TXT.name} to {len(urls)} valid URLs.")


# ── Update profiles ───────────────────────────

def update_profiles(aggregated: dict, min_matches: int, dry_run: bool = False):
    """Read predict.py, update LEAGUE_PROFILES, write back."""
    if not PREDICT_PY.exists():
        print(f"ERROR: {PREDICT_PY} not found")
        return

    content = PREDICT_PY.read_text()
    original = content

    # Profile field mapping: stat name → profile key
    field_map = {
        "avg_goals": "avg_goals",
        "u25_rate": "u25_rate",
        "btts_no_rate": "btts_no_rate",
        "draw_rate": "draw_rate",
        "home_win_rate": "home_win_rate",
    }

    changed = 0
    added = 0

    for key, stats in aggregated.items():
        if stats["matches"] < min_matches:
            continue

        # Find existing profile line
        profile_start = f'    "{key}":'
        idx = content.find(profile_start)

        if idx != -1:
            # Update existing profile — find the dict end
            line_end = content.find("\n", idx)
            line = content[idx:line_end]
            # Replace values within the dict
            for stat_name, prof_key in field_map.items():
                old_val = stats[prof_key]
                # Build pattern to match "prof_key: <value>"
                pat = rf'("{prof_key}":\s*)[\d.]+'
                repl = rf'\g<1>{old_val}'
                new_line = re.sub(pat, repl, line)
                if new_line != line:
                    content = content[:idx] + content[idx:].replace(line, new_line, 1)
                    line = new_line
                    changed += 1
        else:
            # Add new profile entry before the last profile's closing brace
            # Insert before the "default" entry or before the closing }
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
            # Insert before the "default" line
            default_idx = content.find('    "default":')
            if default_idx != -1:
                content = content[:default_idx] + new_entry + content[default_idx:]
                added += 1

    if dry_run:
        print(f"[DRY-RUN] Would update {changed} fields in {len(aggregated)} profiles, add {added} new profiles.")
        return

    if content != original:
        PREDICT_PY.write_text(content)
        print(f"Updated {PREDICT_PY.name}: {changed} field changes, {added} new profiles added.")
    else:
        print("No changes needed.")


# ── Report ────────────────────────────────────

def show_report(aggregated: dict, min_matches: int):
    print(f"\n{'League':30s} {'M':>3s} {'Gls':>5s} {'Draw':>5s} {'HW':>5s} {'U25':>5s} {'NoBTTS':>6s}")
    print("-" * 60)
    for key in sorted(aggregated):
        s = aggregated[key]
        if s["matches"] < min_matches:
            continue
        print(f"{key:30s} {s['matches']:3d} {s['avg_goals']:5.2f} {s['draw_rate']:5.2f} {s['home_win_rate']:5.2f} {s['u25_rate']:5.2f} {s['btts_no_rate']:6.2f}")


# ── Alias setup ───────────────────────────────

def ensure_alias():
    bindir = Path.home() / ".local" / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    src = PREDICT_PY.parent / "cnfupdate.py"
    for name in ("cnfupdate", "cnf"):
        bin_path = bindir / name
        if bin_path.exists() and bin_path.samefile(src):
            continue
        try:
            if bin_path.exists() or bin_path.is_symlink():
                bin_path.unlink()
            bin_path.symlink_to(src.resolve())
            print(f"Alias created: {bin_path} -> {src.resolve()}")
        except Exception as e:
            print(f"Warning: could not create alias for {name}: {e}")


# ── Main ──────────────────────────────────────

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = [a for a in sys.argv[1:] if a.startswith("-")]

    dry_run = "--dry-run" in flags
    min_matches = 10
    for a in flags:
        if a.startswith("--min-matches="):
            min_matches = int(a.split("=")[1])

    print("=" * 50)
    print("cnfupdate: scrape → aggregate → update profiles")
    print("=" * 50)

    # 1. Scrape
    single_url = None
    for a in args:
        if a.startswith("http://") or a.startswith("https://"):
            single_url = a
            break

    if single_url:
        results = RESULTS_JSON
        existing = []
        if results.exists():
            try:
                existing = json.loads(results.read_text())
            except (json.JSONDecodeError, Exception):
                pass
        done_urls = {r["url"] for r in existing}
        if single_url not in done_urls:
            print(f"Scraping single URL ...")
            data = scrape_one(single_url)
            if data:
                existing.append(data)
                if not dry_run:
                    results.write_text(json.dumps(existing, indent=2))
                print(f"  Scraped: {data['home_goals']}-{data['away_goals']}")
            else:
                print("  Could not scrape URL (no score found).")
        else:
            print("  URL already scraped.")
        results = existing
    else:
        results = do_scrape(dry_run=dry_run)

    if not results:
        print("No results to process.")
        return

    # 2. Aggregate
    aggregated = aggregate(results)
    print(f"\nAggregated stats for {len(aggregated)} profile keys:")
    show_report(aggregated, min_matches)

    # 3. Update profiles
    print()
    update_profiles(aggregated, min_matches, dry_run=dry_run)

    # 4. Ensure alias
    if not dry_run:
        ensure_alias()

    # 5. Clear processed URLs from played.txt
    if not dry_run:
        clean_played()

    print("\nDone.")


if __name__ == "__main__":
    main()
