#!/usr/bin/env python3
"""Rebuild data/leagues_db.json from a saved list of Forebet URLs.

Reads a file with one Forebet URL per line (e.g. leagues.html), keeps the
existing per-league records verbatim, and appends every league URL belonging
to one of the countries already tracked in the DB. New entries get a
deterministic league_code, ISO country_code, match_count 0 and source
"leagues.html".

Usage:
    python scripts/rebuild_leagues_db.py [urls_file] [out_file]
"""
import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

DEFAULT_URLS = BASE / "leagues.html"
DEFAULT_OUT = BASE / "data" / "leagues_db.json"

SECTIONS = {
    "football-tips-and-predictions-for-paraguay": ("Paraguay", "py"),
    "tips-and-predictions-for-australia": ("Australia", "au"),
    "football-tips-and-predictions-for-india": ("India", "in"),
    "football-tips-and-predictions-for-uzbekistan": ("Uzbekistan", "uz"),
    "football-tips-and-predictions-for-south-korea": ("South Korea", "kr"),
    "football-tips-and-predictions-for-kazakhstan": ("Kazakhstan", "kz"),
    "football-tips-and-predictions-for-czech-rep": ("Czech Republic", "cz"),
    "football-tips-and-predictions-for-finland": ("Finland", "fi"),
    "football-tips-and-predictions-for-romania": ("Romania", "ro"),
    "football-tips-and-predictions-for-denmark": ("Denmark", "dk"),
    "football-tips-and-predictions-for-russia": ("Russia", "ru"),
    "football-tips-and-predictions-for-sweden": ("Sweden", "se"),
    "football-tips-and-predictions-for-switzerland": ("Switzerland", "ch"),
    "football-tips-and-predictions-for-argentina": ("Argentina", "ar"),
    "football-tips-and-predictions-for-brazil": ("Brazil", "br"),
    "football-tips-and-predictions-for-iceland": ("Iceland", "is"),
    "predictions-lebanon": ("Lebanon", "lb"),
    "football-tips-and-predictions-for-ireland": ("Ireland", "ie"),
    "football-tips-and-predictions-for-norway": ("Norway", "no"),
}

LINK_RE = re.compile(r"https://www\.forebet\.com/en/([^/]+)/([^/]+)/?$")


def prettify(slug: str) -> str:
    parts = []
    for i, tok in enumerate(slug.split("-")):
        if not tok:
            continue
        if tok.isdigit():
            parts.append(tok + "." if i == 0 else tok)
        elif tok == "u20":
            parts.append("U20")
        elif len(tok) == 1:
            parts.append(tok.upper())
        else:
            parts.append(tok[:1].upper() + tok[1:])
    return " ".join(parts)


def main() -> None:
    urls_file = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_URLS
    out_file = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT

    db_path = BASE / "data" / "leagues_db.json"
    db = json.loads(db_path.read_text())

    found = {}
    for line in urls_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        m = LINK_RE.match(line)
        if not m:
            continue
        section = m.group(1).lower()
        if section not in SECTIONS:
            continue
        country, cc = SECTIONS[section]
        league_path = f"{m.group(1)}/{m.group(2)}"
        found.setdefault(country, []).append((league_path, prettify(m.group(2))))

    existing_paths = {v.get("league_url_path") for v in db.values()}
    used_codes = set(db.keys())

    new_entries = []
    for country in sorted(found):
        for i, (path, league) in enumerate(sorted(found[country], key=lambda x: x[1]), 1):
            if path in existing_paths:
                continue
            cc = SECTIONS[next(s for s, (c, _) in SECTIONS.items() if c == country)][1]
            code = f"{cc}{i:02d}"
            while code in used_codes:
                code += "x"
            used_codes.add(code)
            new_entries.append({
                "league_code": code,
                "country": country,
                "league": league,
                "league_url_path": path,
                "country_code": cc,
                "match_count": 0,
                "teams": {},
                "source": "leagues.html",
            })

    for entry in new_entries:
        db[entry["league_code"]] = entry

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(db, indent=2) + "\n")

    print(f"Existing kept: {len(db) - len(new_entries)}")
    print(f"Added from {urls_file.name}: {len(new_entries)}")
    print(f"Total leagues: {len(db)}")
    per_country = {}
    for v in db.values():
        per_country.setdefault(v["country"], 0)
        per_country[v["country"]] += 1
    for c in sorted(per_country, key=lambda c: -per_country[c]):
        print(f"  {c:16s} {per_country[c]:3d}")
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
