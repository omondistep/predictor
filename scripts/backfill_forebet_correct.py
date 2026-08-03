#!/usr/bin/env python3
"""Backfill forebet_correct in calibration_log.

The BTTS "Yes"/"No" rows were scored with _prediction_correct() before it
understood BTTS, so they were always recorded as incorrect. Recompute every
forebet_correct value from the actual score (via the matches table) using the
fixed scoring function.
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database import _prediction_correct, DB_PATH

conn = sqlite3.connect(DB_PATH)

rows = conn.execute("""
    SELECT c.id, c.market, c.forebet_pred,
           m.actual_home_goals, m.actual_away_goals,
           c.forebet_correct
    FROM calibration_log c
    LEFT JOIN matches m ON c.match_id = m.id
    WHERE c.forebet_pred IS NOT NULL
""").fetchall()

updated = 0
unchanged = 0
skipped = 0
for r in rows:
    rid, market, fbp, hg, ag, old = r
    if hg is None or ag is None:
        skipped += 1
        continue
    new = int(_prediction_correct(fbp, hg, ag))
    if new != old:
        conn.execute("UPDATE calibration_log SET forebet_correct = ? WHERE id = ?", (new, rid))
        updated += 1
    else:
        unchanged += 1

conn.commit()
conn.close()

print(f"rows scanned: {len(rows)}")
print(f"  updated:   {updated}")
print(f"  unchanged: {unchanged}")
print(f"  skipped (no score): {skipped}")
