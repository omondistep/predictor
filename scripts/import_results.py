#!/usr/bin/env python3
"""
Import match results from betting data format → update league profiles.

Parses raw betting data (one block per match), maps teams to leagues,
aggregates stats, and updates LEAGUE_PROFILES in predict.py.

Usage:
  python3 import_results.py < data.txt
  python3 import_results.py results.txt
  python3 import_results.py --dry-run < data.txt
"""

import re, sys, json, os
from collections import defaultdict
from pathlib import Path

PREDICT_PY = Path(__file__).parent / "predict.py"
RESULTS_JSON = Path(__file__).parent / "scraped_results.json"

# ── Team → league mapping ─────────────────────────
# Add more entries here as you encounter new teams.
# Key is a substring to match in team name.
TEAM_LEAGUE = {
    # Sweden
    "BK":       "sweden-division-2",
    "FF":       "sweden-division-2",
    "IF":       "sweden-division-2",
    "IK":       "sweden-division-2",
    "SK":       "sweden-division-2",
    "Onsala":   "sweden-division-2",
    "Torslanda":"sweden-division-2",
    "Astorps":  "sweden-division-2",
    "Landvetter":"sweden-division-2",
    "Vastra Frolunda":"sweden-division-2",
    "Boljan":   "sweden-division-2",
    "Angby":    "sweden-division-2",
    "Skiljebo": "sweden-division-2",
    "Husqvarna":"sweden-division-2",
    "Vanersborgs":"sweden-division-2",
    "Norrby":   "sweden-ettan",
    "Oddevold": "sweden-superettan",
    # Norway
    "Skeid":    "default",
    "Rana":     "default",
    "Lysekloster":"default",
    "Notodden": "default",
    "Sotra":    "default",
    "Brattvag": "default",
    # Finland
    "Keski Uusimaa":"finland-kakkonen",
    "TPV":      "finland-kakkonen",
    # Finland women (exact match to avoid false positives)
    "Vifk Vaasa W":"women-football",
    "Ilves W":  "women-football",
    # China
    "Shenzhen Juniors":"default",
    "Guangxi Hengchen":"default",
    "Wuxi":     "default",
    "Meizhou":  "default",
    # Belarus
    "Gomel":    "default",
    "Niva Dolbizno":"default",
    # Estonia
    "Flora":    "estonia",
    "Tallinna Kalev":"estonia",
    # Australia
    "Canberra Juventus":"default",
    "Cooma Tigers":"default",
}

def detect_league(home: str, away: str) -> str:
    """Best-effort league detection from team names."""
    # Check for women's indicators - be specific
    def is_women(name):
        name_lower = name.lower()
        if name_lower.endswith(" w") or " wfc" in name_lower:
            return True
        if "(w)" in name_lower:
            return True
        if name_lower.endswith("(w)") or "women" in name_lower:
            return True
        return False
    if is_women(home) or is_women(away):
        return "women-football"
    # Check team names against mapping
    for team_substr, league in TEAM_LEAGUE.items():
        if team_substr in home or team_substr in away:
            return league
    return "default"


def parse_blocks(lines: list) -> list:
    """Parse betting data blocks. Returns list of dicts."""
    matches = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or not line.isdigit():
            i += 1
            continue
        seq = line
        i += 1
        if i >= len(lines): break
        mid = lines[i].strip()
        i += 1
        if i >= len(lines): break
        dt = lines[i].strip()
        i += 1
        if i >= len(lines): break
        teams_line = lines[i].strip()
        i += 1
        if i >= len(lines): break
        odds = lines[i].strip()
        i += 1
        if i >= len(lines): break
        market = lines[i].strip()
        i += 1
        if i >= len(lines): break
        selection = lines[i].strip()
        i += 1
        if i >= len(lines): break
        score_raw = lines[i].strip()
        i += 1

        # Parse teams
        sep = '–' if '–' in teams_line else ' - ' if ' - ' in teams_line else None
        if sep:
            home, away = [t.strip() for t in teams_line.split(sep, 1)]
        else:
            home, away = teams_line, ""

        # Parse score
        ft_score = score_raw.split('(')[0].strip() if '(' in score_raw else score_raw
        if ft_score and ft_score != '-':
            parts = ft_score.split(':')
            hg, ag = (int(parts[0]), int(parts[1])) if len(parts) == 2 else (None, None)
        else:
            hg, ag = None, None

        matches.append({
            'seq': seq, 'id': mid, 'datetime': dt,
            'home': home, 'away': away, 'odds': float(odds) if odds else 0,
            'market': market, 'selection': selection,
            'hg': hg, 'ag': ag,
        })
    return matches


def main():
    dry_run = '--dry-run' in sys.argv

    # Read input
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        with open(sys.argv[1]) as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    lines = text.strip().split('\n')
    matches = parse_blocks(lines)
    if not matches:
        print("No matches parsed.")
        return

    # Filter to matches with scores
    played = [m for m in matches if m['hg'] is not None]
    print(f"Parsed {len(matches)} matches, {len(played)} with scores.\n")

    # Group by league
    league_matches = defaultdict(list)
    for m in played:
        league = detect_league(m['home'], m['away'])
        m['league'] = league
        league_matches[league].append(m)

    # Compute stats per league
    stats = {}
    for league, ms in sorted(league_matches.items()):
        n = len(ms)
        total_goals = sum(m['hg'] + m['ag'] for m in ms)
        draws = sum(1 for m in ms if m['hg'] == m['ag'])
        home_wins = sum(1 for m in ms if m['hg'] > m['ag'])
        u25 = sum(1 for m in ms if m['hg'] + m['ag'] <= 2)
        btts_no = sum(1 for m in ms if m['hg'] == 0 or m['ag'] == 0)
        stats[league] = {
            'matches': n,
            'avg_goals': round(total_goals / n, 2) if n else 0,
            'draw_rate': round(draws / n, 2) if n else 0,
            'home_win_rate': round(home_wins / n, 2) if n else 0,
            'u25_rate': round(u25 / n, 2) if n else 0,
            'btts_no_rate': round(btts_no / n, 2) if n else 0,
        }

    # ── Report ──
    print(f"{'League':30s} {'M':>3s} {'Gls':>5s} {'Draw':>5s} {'HW':>5s} {'U25':>5s} {'NoBTTS':>6s}")
    print("-" * 60)
    for league in sorted(stats):
        s = stats[league]
        print(f"{league:30s} {s['matches']:3d} {s['avg_goals']:5.2f} {s['draw_rate']:5.2f} {s['home_win_rate']:5.2f} {s['u25_rate']:5.2f} {s['btts_no_rate']:6.2f}")

    # ── Performance by league ──
    print(f"\n{'='*60}")
    print("BETTING PERFORMANCE BY LEAGUE")
    print(f"{'League':30s} {'Bets':>5s} {'Won':>4s} {'Lost':>4s} {'Win%':>5s}")
    print("-" * 50)
    for league in sorted(league_matches):
        ms = league_matches[league]
        settled = []
        for m in ms:
            hg, ag = m['hg'], m['ag']
            total = hg + ag
            sel = m['selection']
            market = m['market']
            if 'Draw no bet' in market:
                if sel == m['home']:
                    if hg == ag: continue  # void
                    won = hg > ag
                else:
                    if hg == ag: continue
                    won = ag > hg
            elif market == '3 Way':
                if sel == m['home']:
                    won = hg > ag
                elif sel == m['away']:
                    won = ag > hg
                else:
                    won = hg == ag
            elif 'Full time result' in market:
                parts = sel.split(' and ')
                rp = parts[0].strip()
                rw = (hg > ag) if rp == '1' else (ag > hg) if rp == '2' else (hg == ag)
                ou = parts[1] if len(parts) > 1 else ''
                ot_match = re.search(r'[\d.]+', ou)
                ot = float(ot_match.group()) if ot_match else 0
                ow = total > ot if 'OVER' in ou.upper() else total <= ot
                won = rw and ow
            elif 'Over/Under' in market:
                thresh = float(sel.split()[-1])
                won = total > thresh if 'OVER' in sel.upper() else total <= thresh
            elif market == 'Double Chance':
                if sel == '1 OR X': won = hg >= ag
                elif sel == 'X OR 2': won = ag >= hg
                else: won = hg != ag
            else:
                continue
            settled.append(won)
        if settled:
            wins = sum(settled)
            print(f"{league:30s} {len(settled):5d} {wins:4d} {len(settled)-wins:4d} {100*wins//len(settled):4d}%")

    # ── Save to scraped_results.json ──
    if not dry_run and played:
        existing = []
        if RESULTS_JSON.exists():
            try:
                existing = json.loads(RESULTS_JSON.read_text())
            except (json.JSONDecodeError, Exception):
                pass
        seen_urls = {r["url"] for r in existing}
        existing_ids = set()
        for r in existing:
            url = r.get("url", "")
            m = re.search(r"/matches/([^/]+?)-(\d{7})$", url)
            if m:
                existing_ids.add(m.group(2))
        new_entries = []
        for m in played:
            if m["id"] not in existing_ids:
                league_code = {"sweden-division-2": "Se4", "estonia": "Ee1",
                               "finland-kakkonen": "Fi3", "women-football": "FiW",
                               "default": "XX"}.get(m.get("league", "default"), "XX")
                new_entries.append({
                    "url": f"https://www.forebet.com/en/football/matches/imported-{m['id']}",
                    "league_code": league_code,
                    "home_goals": m["hg"],
                    "away_goals": m["ag"],
                })
        if new_entries:
            all_entries = existing + new_entries
            RESULTS_JSON.write_text(json.dumps(all_entries, indent=2))
            print(f"\n✓ Saved {len(new_entries)} results to {RESULTS_JSON.name}")

    # ── Update profiles ──
    print(f"\n{'='*60}")
    if dry_run:
        print("[DRY-RUN] Would update profiles in predict.py:")
    else:
        print("Updating LEAGUE_PROFILES in predict.py ...")

    content = PREDICT_PY.read_text()
    original = content
    changed = 0

    field_map = {
        "avg_goals": "avg_goals",
        "u25_rate": "u25_rate",
        "btts_no_rate": "btts_no_rate",
        "draw_rate": "draw_rate",
        "home_win_rate": "home_win_rate",
    }

    for key, s in stats.items():
        min_match = 8 if key == "default" else 3
        if s['matches'] < min_match:
            print(f"  Skipping {key}: only {s['matches']} matches (< {min_match} minimum)")
            continue

        profile_start = f'    "{key}":'
        idx = content.find(profile_start)

        if idx != -1:
            # Update existing profile
            line_end = content.find("\n", idx)
            line = content[idx:line_end]
            for stat_name, prof_key in field_map.items():
                old_val = s[prof_key]
                pat = rf'("{prof_key}":\s*)[\d.]+'
                repl = rf'\g<1>{old_val}'
                new_line = re.sub(pat, repl, line)
                if new_line != line:
                    content = content[:idx] + content[idx:].replace(line, new_line, 1)
                    line = new_line
                    changed += 1
            print(f"  Updated {key}: {s['matches']} matches → avg_goals={s['avg_goals']}, draw={s['draw_rate']}, hw={s['home_win_rate']}, u25={s['u25_rate']}, nobtts={s['btts_no_rate']}")
        else:
            # Add new profile
            vol = 0.08 if s['draw_rate'] > 0.30 else 0.12
            home_adv = 1.15
            new_entry = (
                f'    "{key}": '
                f'{{"avg_goals": {s["avg_goals"]}, '
                f'"u25_rate": {s["u25_rate"]}, '
                f'"btts_no_rate": {s["btts_no_rate"]}, '
                f'"draw_rate": {s["draw_rate"]}, '
                f'"home_win_rate": {s["home_win_rate"]}, '
                f'"home_adv": {home_adv}, "volatility": {vol}}},\n'
            )
            default_idx = content.find('    "default":')
            if default_idx != -1:
                content = content[:default_idx] + new_entry + content[default_idx:]
                changed += 1
                print(f"  Added new profile {key}")
            else:
                print(f"  Could not add {key}: no 'default' anchor found")

    if dry_run:
        print(f"\n[DRY-RUN] Would make {changed} changes.")
        return

    if content != original:
        PREDICT_PY.write_text(content)
        print(f"\n✓ Updated {PREDICT_PY.name}: {changed} changes.")
    else:
        print("\nNo changes needed.")


if __name__ == "__main__":
    main()
