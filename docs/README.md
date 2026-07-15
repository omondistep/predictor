# Football Match Predictor

Rule-based + ML-enhanced football match predictor. Scrapes Forebet match data,
runs Poisson and ensemble ML models, and surfaces value picks with confidence
ratings. Saves everything to SQLite for post-match calibration.

## Quick Start

### Linux/macOS (Arch/Debian)

```bash
git clone https://github.com/omondistep/predictor.git
cd predictor
./install.sh         # sets up venv, deps, symlink, trains ML model
export PATH="$PATH:$HOME/.local/bin"

# Then just use `pr` anywhere:
pr https://www.forebet.com/en/football/matches/...
pr links.txt
pr --calibrate
pr --review
```

### Windows (PowerShell)

```powershell
# Clone
git clone https://github.com/omondistep/predictor.git
cd predictor

# Setup
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python ml_model.py --train
$env:Path += ";$pwd"

# Usage
.\pr.ps1 https://www.forebet.com/en/football/matches/...
.\pr.ps1 links.txt

# Or directly
python predict.py https://...
python predict.py --ml links.txt

# Permanent alias (add to your $PROFILE):
# New-Alias pr .\pr.ps1
```

## Commands

| Command | Description |
|---------|-------------|
| `pr <url>` | Scrape + predict (ML-enhanced ensemble) |
| `pr links.txt` | Same, reading URLs from a file |
| `pr --no-ml <url>` | Classic Poisson-only prediction |
| `pr --review` | Auto-review past predictions from DB (fetches results from Forebet, fallback to manual) |
| `pr --review results.txt` | Review URLs from a file — fetch each, extract score, update DB |
| `pr --calibrate` | Show accuracy stats per confidence level and league |
| `pr --learn <url>` | Scrape a Forebet results-list page to auto-update match results |
| `pr --high-only` | Show only High / Near Certain picks |
| `pr --json` | JSON output (for scripting) |
| `pr --no-reasoning` | Hide reasoning lines |
| `pr --no-compare` | Skip Forebet comparison display |
| `pr --help` | Show all options |

## ML Model Management

```bash
python ml_model.py --train       # Train RandomForest + GradientBoosting (14K+ examples)
python ml_model.py --analyze     # Show feature importance (mutual information)
python ml_model.py --predict     # Test prediction on sample data
python ml_model.py --load        # Load existing model and test
```

The ML model is trained on 14,385 game dataset records + reviewed history.db
matches. Feature extraction covers form, position, goal averages, H2H, odds,
Forebet probabilities, and league profiles. Training outputs 5 components
(scaler, RF 1X2, GB 1X2, RF O/U, GB O/U) to `ml_models/ml_predictor/`.

## Prediction Pipeline

1. **Scrape** — Forebet match page → form, positions, odds, H2H, league
2. **Detect league** — Match `detect_league()` → pick statistical profile
3. **Compute expected goals** — Enhanced Poisson with attack/defense strength (Dixon-Coles style), form/position adjustments, volatility regression
4. **Ensemble** — Blend Poisson (60%) + RandomForest + GradientBoosting (40%) + Forebet (25%) into final 1X2, O/U, BTTS probabilities
5. **Value detection** — Compare model probability vs odds-implied probability → confidence rating (Near Certain / High / Medium-High / Medium / Low)
6. **Save** — Match data + predictions → `history.db`

## League Coverage

62 league profiles including: Brazil Série A/B/C/D, Argentina B Nacional/Primera B/C/Federal A,
Chile Primera/Primera B, USL Championship/League One/Two, MLS Next Pro, NWSL,
Uruguay Primera/Segunda, Ecuador Serie A/B, Peru Primera, Paraguay Primera/Segunda,
Sweden Allsvenskan/Superettan/Ettan/Div 2, Finland Veikkausliiga/Ykkonen/Kakkonen,
Morocco Botola, Mexico Liga MX/Expansion MX, Colombia Primera A/B,
Venezuela Primera, plus 21 newly merged leagues from game dataset (DR Congo,
Libya, Sudan, Saudi 1st, Turkiye 3. Lig, Thailand 3, Algeria Ligue 2, etc.).

## Database

- `history.db` — All predictions + scraped data + reviewed results
- `predict.py --review` — Auto-fetches actual scores from Forebet (past matches only)
- `predict.py --calibrate` — Accuracy breakdown by confidence level and league
- `predict.py --learn <results_url>` — Batch-import results from Forebet results pages

## Usage
# Analyze bias and retrain if needed
python predict.py --learn-calibration

# Full calibration quality report
python predict.py --calibration-report

# Force retrain ML models
python predict.py --force-retrain

# Normal prediction (auto-runs calibration check)
python predict.py links.txt

## Files

| File | Purpose |
|------|---------|
| `predict.py` | Main CLI — scraping, analysis, DB, review, calibration |
| `ml_model.py` | ML training + enhanced Poisson + hybrid ensemble |
| `database.py` | SQLite operations |
| `forebet_scraper.py` | Forebet HTML scraper |
| `historical_calibrate.py` | Historical data collection |
| `cnfupdate.py` | Confidence tuning |
| `merge_game_dataset.py` | Merged 15K game/ records into league profiles |
| `ml_models/ml_predictor/` | Trained model components (RF, GB, scaler) |


File	Change
auto_learn.py	New — 329-line automated learning pipeline
predict.py	Added --auto-learn CLI arg, enhanced _maybe_auto_calibrate() to scrape results
aliases.sh	Added auto-learn, learn-status, learn-daemon aliases
install_auto_learn	New — systemd/cron scheduling installer
Here's a summary of what was implemented for continuous automated learning:
What was built
auto_learn.py — Continuous Learning Pipeline
Orchestrates the full learning cycle:
1. Result scraping — automatically fetches actual scores from Forebet for past unreviewed predictions
2. DB update — stores results, marks as reviewed, logs calibration entries
3. Calibration analysis — detects bias per league/market/probability-bucket via calibration_learner.py
4. Auto-retrain — triggers ML model retraining when sufficient new data accumulates (50+ new examples)
Integration into predict.py
- Added --auto-learn flag to the CLI
- Auto-learn now runs automatically after every prediction run — _maybe_auto_calibrate() scrapes results for past matches, analyzes bias, and retrains if needed
- So anytime you run pr links.txt, the system silently learns from past outcomes
Scheduling (for full automation)
- install_auto_learn.sh — interactive installer for systemd timer (daily) or cron (twice daily)
- Daemon mode — python auto_learn.py --daemon runs continuously with configurable interval
- Status command — python auto_learn.py --status shows learning status, data volumes, accuracy
How it works
pr links.txt          → predicts matches, then auto-learns from past outcomes
pr --auto-learn       → explicitly runs the full learning pipeline
auto_learn.py         → standalone pipeline (for cron/systemd)
auto_learn.py --status → dashboard of learning progress
auto_learn.py --daemon → runs continuously as a background process
