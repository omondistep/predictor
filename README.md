# Football Match Predictor

Rule-based terminal app for football match analysis and predictions.
Uses the same analytical strategy as the manual workflow: league trends,
odds structure, home/away splits, and market value.

## Quick Start

```bash
# Paste match data directly
python predict.py --paste

# Pipe data from a file
cat matches.txt | python predict.py

# Read from file argument
python predict.py matches.txt

# Show only high-confidence predictions
python predict.py matches.txt --high-only

# Export as JSON
python predict.py matches.txt --json > predictions.json
```

## Input Format

The app expects match data in the same format used in the original analysis:

```
USA - USL, Championship (1)
31/05/26 - 03:00 | ID: 3718
 Home Team Name
 Away Team Name
2.33
3.50
2.47
1.37
1.41
1.23
1.60
2.20
1.54
2.22
```

Odds columns: Home Win, Draw, Away Win, 1X, 12, X2, Over 2.5, Under 2.5, BTTS Yes, BTTS No

## Options

| Flag | Description |
|------|-------------|
| `file` | Path to file with match data |
| `--paste` | Paste match data interactively |
| `--json` | Output as JSON (for scripting) |
| `--high-only` | Show only High / Near Certain picks |
| `--scrape` | Attempt web scraping for form data (experimental) |
| `--no-reasoning` | Hide reasoning lines |

## Analysis Engine

The rule engine evaluates multiple signals:

1. **1X2 Market** — Strong favorite detection (odds < 1.40), clear favorite (< 1.60), even match → draw
2. **Over/Under 2.5** — Market expectation + league trend override
3. **BTTS** — Market expectation + league scoring patterns
4. **League Profiles** — Hardcoded statistical profiles for 30+ leagues (avg goals, U2.5 rate, BTTS No rate, draw rate)
5. **Confidence Scoring** — Near Certain (odds < 1.25), High (odds < 1.50 or strong league pattern), Medium, Low

## League Coverage

Includes statistical profiles for: Brazil Série A/B/C/D, USL Championship/League One/Two,
MLS Next Pro, NWSL, Chile Primera/Primera B, Argentina B Nacional/Primera B/C/Federal A,
Uruguay Primera/Segunda, Ecuador Serie A/B, Peru Primera, Paraguay Primera/Segunda,
Canada Premier, Spain Segunda/Tercera, and more.

## Installation

```bash
pip install -r requirements.txt
```
