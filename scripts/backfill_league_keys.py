"""Backfill cleaned league keys into component_accuracy and calibration_log.

Both tables were written with polluted league names (e.g. "10th place England
Premier League 6th place") before the label fix. This script:

1. component_accuracy: cleans each league name via clean_league_name(), converts
   to the detect_league() key (e.g. "england-premier"), and merges duplicate
   (component, league, market) rows by summing total/correct. This makes
   get_dynamic_weights() — which queries by detect_league key — actually find
   per-league data.

2. calibration_log: cleans each league name in place (append-only audit log,
   unique rows — no merge needed).

Usage:
    .venv/bin/python scripts/backfill_league_keys.py        # dry-run (report only)
    .venv/bin/python scripts/backfill_league_keys.py --apply  # write to DB
"""
import argparse
import sqlite3
import sys
from collections import defaultdict

ROOT = __import__("os").path.dirname(__import__("os").path.dirname(
    __import__("os").path.abspath(__file__)))
sys.path.insert(0, ROOT)

from league_names import clean_league_name  # noqa: E402
from predict import detect_league  # noqa: E402

DB = ROOT + "/history.db"


def to_key(league: str) -> str:
    """Clean a raw league name, then map to its detect_league key."""
    cleaned = clean_league_name(league) or league or "unknown"
    return detect_league(cleaned)


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill cleaned league keys")
    ap.add_argument("--apply", action="store_true",
                    help="write changes to history.db (default: dry-run)")
    args = ap.parse_args()

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    # ── component_accuracy ──
    rows = conn.execute("""
        SELECT component, league, market, SUM(total) as total, SUM(correct) as correct
        FROM component_accuracy
        GROUP BY component, league, market
    """).fetchall()

    # Aggregate by (component, detect_league_key, market)
    merged = defaultdict(lambda: {"total": 0, "correct": 0})
    polluted_rows = 0
    for r in rows:
        key = to_key(r["league"])
        if key != (r["league"] or ""):
            polluted_rows += 1
        d = merged[(r["component"], key, r["market"])]
        d["total"] += r["total"]
        d["correct"] += r["correct"]

    n_distinct_before = len(rows)
    n_distinct_after = len(merged)
    n_polluted_rows = polluted_rows

    # ── calibration_log ──
    cl_rows = conn.execute("""
        SELECT id, league FROM calibration_log WHERE league IS NOT NULL AND league <> ''
    """).fetchall()
    cl_changes = []
    for r in cl_rows:
        cleaned = clean_league_name(r["league"])
        if cleaned and cleaned != r["league"]:
            cl_changes.append((cleaned, r["id"]))

    print("=== component_accuracy ===")
    print(f"distinct (component,league,market) before: {n_distinct_before}")
    print(f"distinct after clean+merge:                {n_distinct_after}")
    print(f"rows with polluted league names:           {n_polluted_rows}")
    print()
    print("=== calibration_log ===")
    print(f"rows to clean: {len(cl_changes)}")
    print()
    print("sample component_accuracy merges:")
    shown = 0
    for (comp, key, mkt), d in merged.items():
        # find an original polluted source for display
        orig = None
        for r in rows:
            if r["component"] == comp and r["market"] == mkt and to_key(r["league"]) == key:
                orig = r["league"]; break
        if orig and orig != key:
            print(f"  {comp:10s} {mkt:10s} {orig!r:55s} -> {key!r} (total={d['total']}, correct={d['correct']})")
            shown += 1
            if shown >= 10:
                break

    if args.apply:
        # Rebuild component_accuracy
        conn.execute("DELETE FROM component_accuracy")
        for (comp, key, mkt), d in merged.items():
            conn.execute("""
                INSERT INTO component_accuracy (component, league, market, total, correct, last_updated)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
            """, (comp, key, mkt, d["total"], d["correct"]))

        # Clean calibration_log in place
        for cleaned, rid in cl_changes:
            conn.execute("UPDATE calibration_log SET league = ? WHERE id = ?", (cleaned, rid))

        conn.commit()

        # Re-measure
        ca_polluted = conn.execute("SELECT COUNT(*) FROM component_accuracy WHERE league LIKE '%place%'").fetchone()[0]
        cl_polluted = conn.execute("SELECT COUNT(*) FROM calibration_log WHERE league LIKE '%place%'").fetchone()[0]
        print(f"\napplied: component_accuracy rebuilt ({n_distinct_after} rows)")
        print(f"         calibration_log cleaned ({len(cl_changes)} rows)")
        print(f"component_accuracy still polluted: {ca_polluted}")
        print(f"calibration_log still polluted:    {cl_polluted}")
    else:
        print("dry-run: pass --apply to write changes")

    conn.close()


if __name__ == "__main__":
    main()
