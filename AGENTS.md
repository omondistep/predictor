# Predictor Instructions

## Core Commands

| Command | Description |
|---------|-------------|
| `pr <file>` | Run predictions for matches in a link file |
| `pr <url>` | Predict a single match URL |
| `pr results` | Scrape Forebet for scores, update DB + HTML reports |
| `pr --today` | Auto-fetch today's + tonight's (00:00–06:00) matches per league → `links/today.html` |
| `pr --weekday` | Auto-fetch all Mon–Fri matches (this week if mid-week, next week if weekend) via Forebet's AJAX per day → `links/weekday.html` |
| `pr --high-only <file>` | Only show High / Near Certain picks |
| `pr --no-ml <file>` | Classic Poisson only (no ML ensemble) |
| `pr --json <file>` | JSON output |
| `pr --train-weights` | Train ensemble weights from historical predictions |

## Workflow

### 1. Predict matches
```bash
pr links.txt          # Run all links
pr data/links.txt     # Run all links (explicit path)
pr upcoming.txt       # Run upcoming matches
```

### 2. Update results (after matches finish)
```bash
pr results            # Scrape Forebet for scores → update DB + HTML → auto-learn
```

### 3. Learn from results (manual, if needed)
```bash
pr --calibrate            # Show accuracy stats per confidence level and league
```

### 4. View reports
- Report saved to `predictions/latest.html` (overwritten each run, old versions in `predictions/archive/`)
- `predictions/best.html` — matches where the default pick is highlighted yellow (it's a consensus pick or the best combined pick); auto-opened in browser
- `predictions/consistent.html` — matches where a consensus pick (both models ≥69.5%) is consistent with the best combined pick (e.g. consensus BTTS Yes + best Over 1.5)

## How It All Fits Together

```
pr links.txt               → Predictions saved to history.db + HTML reports
                           → data/pending_results.json updated with new matches
                           → Cron scheduled to retrain model in 18h

pr results                 → Reads data/pending_results.json
                           → Scrapes Forebet for finished scores
                           → Updates history.db + HTML files
                           → Removes completed entries from pending file
                           → Auto-learn: analyze bias + retrain ML if 50+ new examples

Cron (scheduled by pr)     → Runs auto_retrain() after 18h
                           → Retrains the ML model on latest data
```

**Key point:** `pr results` does NOT trigger learning. It only updates data. Learning happens via `pr --learn-calibration` or the cron job scheduled by `pr links.txt`.

## Aliases

| Alias | What it does |
|-------|--------------|
| `pr` | Runs predict.py with venv Python |

## Key Files

| File | Purpose |
|------|---------|
| `history.db` | All predictions + results + calibration data |
| `data/pending_results.json` | Matches still needing result updates |
| `data/links.txt` | Forebet URLs to predict |
| `predictions/` | HTML reports (numbered) |
| `calibration_params.json` | Isotonic regression calibration params |
| `ml_models/ml_predictor/` | Trained ML model components |

## Replicate to a New Machine (Linux / macOS / Windows)

The full system ships in git — DB, calibration params and trained ML models —
plus a snapshot of regenerated state in `seed/data/`. To stand up an identical
copy:

```bash
git clone <repo-url> predictor
cd predictor
bash scripts/install.sh        # Linux / macOS (delegates to install.ps1 from Git-Bash)
# or on Windows (cmd / PowerShell):
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
```

The installer:
1. Creates `.venv` and installs `requirements.txt` (includes xgboost, lightgbm, cloudscraper)
2. Restores `data/{pending_results,leagues_db,predictions}.json` from `seed/data/`
3. Verifies ML models load from git (only retrains if broken/missing)
4. Installs the `pr` command — `~/.local/bin/pr` symlink on Linux/macOS,
   `pr.cmd` + `pr.ps1` shims in `%USERPROFILE%\.local\bin` on Windows

**Keep snapshots fresh after big state changes:**
```bash
cp data/pending_results.json data/leagues_db.json data/predictions.json seed/data/
git add seed/data/ history.db ml_models/ && git commit
```

## Auto-Retrain (Cron)

A cron job is automatically scheduled when you run predictions. It retrains the model 18 hours later.

```bash
crontab -l                    # View scheduled jobs
tail -f /tmp/retrain.log      # Watch retrain logs
```

## DB Quick Stats

```bash
sqlite3 history.db "SELECT COUNT(*) as total, SUM(CASE WHEN actual_home_goals IS NOT NULL THEN 1 ELSE 0 END) as with_results FROM matches;"
```
