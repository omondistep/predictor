#!/usr/bin/env python3
"""Scrape match URLs from upcoming.txt, extract scores, update DB + scraped_results.json."""
import re
import json
import time
import sys
import requests
from bs4 import BeautifulSoup
from database import update_result, get_db

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

UPCOMING_FILE = "upcoming.txt"
RESULTS_FILE = "scraped_results.json"
DELAY = 0.3
BATCH_SIZE = 50
BATCH_DELAY = 1.0


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
        score_text = lscr.get_text(strip=True)
        m = re.search(r"(\d+)\s*-\s*(\d+)", score_text)
        if not m:
            return None

        home_goals, away_goals = int(m.group(1)), int(m.group(2))
        return {
            "url": url,
            "league_code": league,
            "home_goals": home_goals,
            "away_goals": away_goals,
        }
    except Exception as e:
        print(f"  [ER] {e}", file=sys.stderr)
        return None


def main():
    # Load upcoming URLs (filter to match pages)
    with open(UPCOMING_FILE) as f:
        all_urls = [line.strip() for line in f if "/matches/" in line]
    print(f"Match URLs in {UPCOMING_FILE}: {len(all_urls)}", file=sys.stderr)

    # Load existing results
    results = []
    done_urls = set()
    try:
        with open(RESULTS_FILE) as f:
            existing = json.load(f)
            results = existing
            done_urls = {r["url"] for r in results}
        print(f"Already scraped: {len(done_urls)}", file=sys.stderr)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Pre-build URL → match_id lookup from DB
    conn = get_db()
    url_to_id = {}
    for row in conn.execute("SELECT id, forebet_url FROM matches WHERE forebet_url IS NOT NULL"):
        url_to_id[row["forebet_url"]] = row["id"]
    conn.close()
    print(f"DB has {len(url_to_id)} matches with URLs", file=sys.stderr)

    todo = [u for u in all_urls if u not in done_urls]
    print(f"Remaining to scrape: {len(todo)}", file=sys.stderr)

    if not todo:
        print("Nothing to scrape.", file=sys.stderr)
        return

    scraped_count = 0
    db_updated_count = 0
    new_results = []

    for i, url in enumerate(todo):
        data = scrape_one(url)
        if data:
            new_results.append(data)
            results.append(data)
            scraped_count += 1

            # Update DB if this URL is known
            match_id = url_to_id.get(url)
            if match_id:
                try:
                    update_result(match_id, data["home_goals"], data["away_goals"])
                    db_updated_count += 1
                    print(f"  [OK] DB updated: {data['home_goals']}-{data['away_goals']} ({match_id})", file=sys.stderr)
                except Exception as e:
                    print(f"  [DB] update failed for {match_id}: {e}", file=sys.stderr)
            else:
                print(f"  [OK] Scraped but not in DB: {data['home_goals']}-{data['away_goals']}", file=sys.stderr)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(todo)}]", file=sys.stderr, flush=True)
            json.dump(results, open(RESULTS_FILE, "w"), indent=2)

        time.sleep(DELAY)
        if (i + 1) % BATCH_SIZE == 0:
            time.sleep(BATCH_DELAY)

    json.dump(results, open(RESULTS_FILE, "w"), indent=2)
    print(f"\nDone. Scraped: {scraped_count}, DB updated: {db_updated_count}, Total in file: {len(results)}", file=sys.stderr)


if __name__ == "__main__":
    main()
