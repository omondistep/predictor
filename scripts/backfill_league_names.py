"""Backfill cleaned league names into history.db.

Forebet match pages frequently expose the two teams' standings text
("10th place Paraguay Primera Division 6th place") instead of the
competition name. Storing that verbatim put ~75% of resolved rows into
an unusable league bucket, which also starves any per-league expert
layer (it cannot route a match whose league is a standings table).

This script applies ``league_names.clean_league_name`` to every distinct
``matches.league`` value and rewrites the column in place. It only
touches resolved rows (actual_home_goals IS NOT NULL) and leaves
already-clean names untouched.

Usage:
    .venv/bin/python scripts/backfill_league_names.py        # dry-run (report only)
    .venv/bin/python scripts/backfill_league_names.py --apply  # write to DB

Safety: back up history.db before running --apply.
"""
import argparse
import sqlite3
import sys

ROOT = __import__("os").path.dirname(__import__("os").path.dirname(
    __import__("os").path.abspath(__file__)))
sys.path.insert(0, ROOT)

from league_names import clean_league_name, looks_like_standings  # noqa: E402

DB = ROOT + "/history.db"


def polluted(league: str) -> bool:
    """True when a league value still reads like standings text."""
    return bool(league) and looks_like_standings(league)


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill cleaned league names")
    ap.add_argument("--apply", action="store_true",
                    help="write changes to history.db (default: dry-run)")
    args = ap.parse_args()

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    total = conn.execute(
        "SELECT COUNT(*) FROM matches WHERE actual_home_goals IS NOT NULL"
    ).fetchone()[0]
    currently_polluted = conn.execute(
        "SELECT COUNT(*) FROM matches WHERE actual_home_goals IS NOT NULL "
        "AND league LIKE '%place%'"
    ).fetchone()[0]

    # Distinct league strings -> cleaned form
    distinct = conn.execute(
        "SELECT DISTINCT league FROM matches WHERE actual_home_goals IS NOT NULL "
        "AND league IS NOT NULL AND league <> ''"
    ).fetchall()

    changes = []  # (original, cleaned, n_rows)
    cleaned_polluted = 0
    still_polluted = 0
    distinct_cleaned = 0
    distinct_still_polluted = 0

    for row in distinct:
        orig = row["league"]
        cleaned = clean_league_name(orig)
        n = conn.execute(
            "SELECT COUNT(*) FROM matches WHERE actual_home_goals IS NOT NULL "
            "AND league = ?", (orig,)
        ).fetchone()[0]
        if cleaned != orig:
            changes.append((orig, cleaned, n))
            distinct_cleaned += 1
            if polluted(orig):
                cleaned_polluted += n
            if polluted(cleaned):
                still_polluted += n
                distinct_still_polluted += 1
        else:
            if polluted(orig):
                still_polluted += n
                distinct_still_polluted += 1

    print(f"resolved matches total : {total}")
    print(f"currently polluted     : {currently_polluted} "
          f"({100*currently_polluted/total:.1f}%)")
    print(f"distinct league strings: {len(distinct)}")
    print(f"distinct to be cleaned : {distinct_cleaned}")
    print(f"rows to be cleaned     : {sum(c[2] for c in changes)} "
          f"({100*sum(c[2] for c in changes)/total:.1f}%)")
    print(f"distinct still polluted: {distinct_still_polluted}")
    print(f"rows still polluted    : {still_polluted} "
          f"({100*still_polluted/total:.1f}%)")

    if args.apply:
        for orig, cleaned, _n in changes:
            conn.execute(
                "UPDATE matches SET league = ? WHERE league = ? "
                "AND actual_home_goals IS NOT NULL", (cleaned, orig))
        conn.commit()
        # Re-measure after write
        after = conn.execute(
            "SELECT COUNT(*) FROM matches WHERE actual_home_goals IS NOT NULL "
            "AND league LIKE '%place%'"
        ).fetchone()[0]
        print(f"\napplied: {len(changes)} distinct values updated")
        print(f"polluted rows after backfill: {after} "
              f"({100*after/total:.1f}%)")
    else:
        print("\ndry-run: pass --apply to write changes")
        print("\nsample changes:")
        for orig, cleaned, n in changes[:15]:
            print(f"  {n:5d}  {orig!r} -> {cleaned!r}")

    conn.close()


if __name__ == "__main__":
    main()
