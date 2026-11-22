# Historical Calibration

Scrapes past Forebet predictions-1x2 date pages for match results, aggregates actual stats by league, and updates `LEAGUE_PROFILES` in `predict.py` to improve model accuracy.

## How it works

1. Fetches each date page from `https://www.forebet.com/en/football-predictions/predictions-1x2/YYYY-MM-DD`
2. Extracts match scores (`l_scr`) and league info (from the `shortagDiv` country/league tags)
3. Maps each match to a profile key via `detect_league()`
4. Aggregates actual stats per league: avg goals, draw rate, home win rate, U2.5 rate, BTTS No rate
5. Updates the matching profile in `predict.py` (or adds a new one if it doesn't exist)

## Usage

```bash
# Full run (scrapes all dates in range, updates profiles)
python historical_calibrate.py

# Preview only — no changes written
python historical_calibrate.py --dry-run

# Custom date range
python historical_calibrate.py --start 2026-06-01 --end 2026-06-14

# Only add new profiles, don't touch existing ones
python historical_calibrate.py --no-update-existing

# Stricter minimum matches (default: 5)
python historical_calibrate.py --min-matches 10

# Faster/slower scraping (default: 1.0s between pages)
python historical_calibrate.py --delay 0.5
```

## Aliases

```bash
histcal          # quick alias
historical-calibrate  # full name
```

The script auto-creates symlinks in `~/.local/bin/` on first run.

## Profiles skipped from update

- `default` — catch-all for unrecognised leagues, would skew the generic fallback
- `reserve-leagues` — catch-all for youth/reserve sides, too heterogeneous

## Data sources

Each predictions-1x2 date page lists all matches scheduled for that day with Forebet's predictions. After the matches are played, the same page shows actual scores. The script scrapes these scores post-event, so the data reflects **real results**, not predictions.
