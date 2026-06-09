FOOTBALL MATCH PREDICTOR — INSTRUCTIONS
=========================================

1. WHAT THIS APP DOES
----------------------
Analyses football matches using rule-based logic (league trends, odds,
form, standings, H2H) and stores everything in SQLite for post-game
review and model calibration.

Two data sources:
  * Forebet match URLs → deep data (form, standings, H2H, odds)
  * Raw odds text file     → odds-only analysis (legacy mode, use predict_odds.py.bak)

2. HOW TO USE (Forebet mode — default)
----------------------------------------

  Step 1 — Collect match URLs from https://www.forebet.com/en/football/matches/
  Step 2 — Save URLs to a text file (one per line)
  Step 3 — Run:

      predict /path/to/links.txt

  Output: predictions saved to history.db + printed to terminal.

  Options:
    --high-only      Show only Near Certain and High confidence picks
    --no-reasoning   Hide reasoning lines
    --no-compare     Hide Forebet prediction comparison
    --json           Output as JSON (for scripting)

  Examples:
    predict links.txt
    predict links.txt --high-only
    predict links.txt --json > predictions.json

3. POST-MATCH REVIEW
---------------------
  After matches finish, record actual results:

    predict --review

  This shows each unreviewed match and asks for the score (e.g. "2-1").
  Press Enter to skip a match.

  Auto-review (re-scrapes Forebet for the score):

    predict --review --auto

  Once reviewed, the result is stored in history.db and the calibration
  log is updated.

4. MODEL CALIBRATION
---------------------
  View accuracy stats by confidence tier and league:

    predict --calibrate

  Shows:
    - Overall accuracy (our model vs Forebet)
    - Accuracy breakdown by confidence level
    - Accuracy breakdown by league
    - Calibration suggestions (leagues where profile needs adjustment)

  Use this after reviewing several matches to identify weak spots.

5. LEGACY ODDS MODE
--------------------
  The original odds-based analysis (v1) is still available in the backup:

    python3 predict_odds.py.bak /path/to/matches.txt

  Input format (one match block):

    USA - USL, Championship (1)
    31/05/26 - 03:00 | ID: 3718
     Home Team Name
     Away Team Name
    2.33        (Home Win)
    3.50        (Draw)
    2.47        (Away Win)
    1.37        (DC 1X)
    1.41        (DC 12)
    1.23        (DC X2)
    1.60        (Over 2.5)
    2.20        (Under 2.5)
    1.54        (BTTS Yes)
    2.22        (BTTS No)

6. HOW THE ANALYSIS WORKS (Forebet mode)
-----------------------------------------
  For each URL, the app:

  1. Scrapes: teams, date/time, form (W/D/L strings), league standings,
     head-to-head results, probability percentages, match odds.

  2. Runs rule-based analysis using:
     - Odds structure (1X2 value detection)
     - Form comparison (PPG from recent results)
     - League position gap
     - H2H history patterns
     - League profile stats (avg goals, scoring rates)

  3. Confidence levels:
     - Near Certain: odds < 1.25
     - High: odds < 1.50 or strong league pattern
     - Medium-High: draw value with tight odds
     - Medium: clear lean but no extreme signal

  4. Stores everything in history.db for later review + calibration.

7. FILES
---------
  ~/predictor/predict.py         Main CLI (Forebet mode)
  ~/predictor/database.py         SQLite operations
  ~/predictor/forebet_scraper.py  Forebet page scraper
  ~/predictor/history.db          Prediction database
  ~/predictor/predict_odds.py.bak Legacy odds-based mode (v1)
  ~/predictor/raw_sample.txt      Sample Forebet URLs for testing
  ~/predictor/INSTRUCTIONS.md     This file

8. LEAGUE PROFILES
-------------------
  BRAZIL: Serie A, B, C, D
  USA: USL Championship, USL League One, USL League Two,
       MLS Next Pro, NWSL
  ARGENTINA: B Nacional, Primera B, Primera C, Federal A
  CHILE: Primera, Primera B
  URUGUAY: Primera, Segunda
  ECUADOR: Serie A, Serie B
  PERU: Primera
  PARAGUAY: Primera, Segunda
  SPAIN: Segunda
  default: neutral profile

  Profiles store: avg_goals, u25_rate, btts_no_rate, draw_rate, home_win_rate.

9. QUICK START
---------------
  # Save predictions from a link file
  predict ~/predictor/raw_sample.txt

  # After games are played, review results
  predict --review

  # Auto-review (re-scrape Forebet for scores)
  predict --review --auto

  # Check accuracy
  predict --calibrate
