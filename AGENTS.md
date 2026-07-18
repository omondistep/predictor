# Predictor Instructions

## Quick Commands

| Command | Description |
|---------|-------------|
| `pr <file>` | Run predictions for matches in a link file |
| `pr results` | Update existing reports with match results (no duplicates) |

## Workflow

### 1. Predict matches
```bash
pr links.txt          # Run all links
pr data/links.txt     # Run all links (explicit path)
pr upcoming.txt       # Run upcoming matches
```

### 2. Update results (after matches finish)
```bash
pr results            # Scrape Forebet for scores, update HTML reports + index
```

This avoids re-running `pr links.txt` which would create duplicate entries.

### 3. View reports
- Reports are saved in `predictions/` as numbered HTML files (001.html, 002.html...)
- `predictions/index.html` shows the last 100 reports with accuracy stats
- Reports auto-open in browser after generation

## Aliases

| Alias | What it does |
|-------|--------------|
| `pr` | Runs predict.py with venv Python |
| `pr-results` | Runs update_results.py to update existing reports |

## How `pr results` works

1. Scans all HTML reports in `predictions/` for matches without results
2. Extracts Forebet URLs from each report
3. Re-scrapes those pages for updated scores
4. Updates the DB with actual scores
5. Injects `RESULT:` lines into existing HTML files
6. Regenerates `index.html`

Run this periodically after matches finish instead of re-running predictions.
