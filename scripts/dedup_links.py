"""
Remove already-processed URLs from links.txt by checking history.db.

New (unprocessed) URLs stay in links.txt.
Already-processed URLs are appended to played.txt.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "history.db"
LINKS_TXT = Path(__file__).parent / "links.txt"
PLAYED_TXT = Path(__file__).parent / "played.txt"


def main():
    if not DB_PATH.exists():
        print("history.db not found — nothing to deduplicate against.")
        return

    if not LINKS_TXT.exists():
        print("links.txt not found.")
        return

    # Read all URLs from links.txt
    with open(LINKS_TXT) as f:
        all_urls = [line.strip() for line in f if line.strip()]

    if not all_urls:
        print("links.txt is empty.")
        return

    # Query DB for already-processed URLs
    conn = sqlite3.connect(str(DB_PATH))
    existing = {
        row[0]
        for row in conn.execute("SELECT forebet_url FROM matches WHERE forebet_url IS NOT NULL")
    }
    conn.close()

    new_urls = [u for u in all_urls if u not in existing]
    old_urls = [u for u in all_urls if u in existing]

    # Write new URLs back to links.txt
    with open(LINKS_TXT, "w") as f:
        for url in new_urls:
            f.write(url + "\n")

    # Append already-processed URLs to played.txt
    if old_urls:
        with open(PLAYED_TXT, "a") as f:
            for url in old_urls:
                f.write(url + "\n")

    total = len(all_urls)
    print(f"Total URLs: {total}")
    print(f"New (kept in links.txt): {len(new_urls)}")
    print(f"Already in DB (moved to played.txt): {len(old_urls)}")


if __name__ == "__main__":
    main()
