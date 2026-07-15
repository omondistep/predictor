#!/usr/bin/env python3
"""Scrape played match URLs to extract scores and league codes."""
import re
import json
import time
import sys
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

URLS_FILE = "played.txt"
RESULTS_FILE = "scraped_results.json"
BATCH_SIZE = 50
DELAY = 0.3  # seconds between requests
BATCH_DELAY = 1.0  # extra delay between batches

def scrape_one(url: str) -> dict | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")

        # League code
        lel = soup.find(["span", "div"], {"class": "diff_league"})
        league = lel.get_text(strip=True) if lel else "UNKNOWN"

        # Score
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
    urls = open(URLS_FILE).read().strip().splitlines()
    print(f"Total URLs: {len(urls)}", file=sys.stderr)

    # Load existing results if resuming
    results = []
    done_urls = set()
    try:
        with open(RESULTS_FILE) as f:
            existing = json.load(f)
            results = existing
            done_urls = {r["url"] for r in results}
        print(f"Resuming with {len(done_urls)} already done", file=sys.stderr)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    todo = [u for u in urls if u not in done_urls]
    print(f"Remaining: {len(todo)}", file=sys.stderr)

    for i, url in enumerate(todo):
        data = scrape_one(url)
        if data:
            results.append(data)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(todo)}]", file=sys.stderr, flush=True)
            json.dump(results, open(RESULTS_FILE, "w"), indent=2)

        time.sleep(DELAY)
        if (i + 1) % BATCH_SIZE == 0:
            time.sleep(BATCH_DELAY)

    json.dump(results, open(RESULTS_FILE, "w"), indent=2)
    print(f"\nDone. Scraped {len(results)} results.", file=sys.stderr)

if __name__ == "__main__":
    main()
