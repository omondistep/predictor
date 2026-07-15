# Predictor Commands

## Retraining Model

### Run Retraining (Isotonic Regression)
```bash
cd /home/stdk/predictor
.venv/bin/python3 -c "from calibration_learner import retrain_from_results; retrain_from_results(force=True)"
```

### View Calibration Parameters
```bash
cat /home/stdk/predictor/calibration_params.json | python3 -m json.tool
```

### View Market Accuracy
```bash
cd /home/stdk/predictor
.venv/bin/python3 -c "
from database import get_market_accuracy
for m in get_market_accuracy():
    if m['market']:
        print(f\"{m['market']:6s}: {m['total']:4d} total, {m['correct']:4d} correct, {m['accuracy']:5.1f}%\")
"
```

## Automatic Retraining (Cron)

**Schedule:** 8:30 AM EAT (GMT+3) daily

**Installed cron job:**
```bash
30 5 * * * cd /home/stdk/predictor && .venv/bin/python3 -c "from calibration_learner import retrain_from_results; retrain_from_results(force=True)" >> /tmp/retrain.log 2>&1
```

**Verify:** `crontab -l`

**View logs:** `tail -f /tmp/retrain.log`

## Predictions

### Run on URL
```bash
cd /home/stdk/predictor
echo "https://www.forebet.com/en/football/matches/your-match-url" > /tmp/test.txt
.venv/bin/python3 predict.py /tmp/test.txt --no-ml
```

### Run on File
```bash
cd /home/stdk/predictor
.venv/bin/python3 predict.py data/urls.txt --no-ml
```

### View Reports
```bash
ls /home/stdk/predictor/predictions/
xdg-open /home/stdk/predictor/predictions/index.html
```

## Database

### Stats
```bash
cd /home/stdk/predictor
.venv/bin/python3 -c "
import sqlite3
conn = sqlite3.connect('history.db')
print(f'Matches: {conn.execute(\"SELECT COUNT(*) FROM matches\").fetchone()[0]}')
print(f'Results: {conn.execute(\"SELECT COUNT(*) FROM matches WHERE actual_home_goals IS NOT NULL\").fetchone()[0]}')
print(f'Calibration: {conn.execute(\"SELECT COUNT(*) FROM calibration_log\").fetchone()[0]}')
"
```

### Export Calibration Data
```bash
cd /home/stdk/predictor
.venv/bin/python3 -c "
import sqlite3, csv
conn = sqlite3.connect('history.db')
rows = conn.execute('SELECT market, odds, correct, created_at FROM calibration_log WHERE market IS NOT NULL').fetchall()
with open('data/calibration_data.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['market', 'odds', 'correct', 'created_at'])
    w.writerows(rows)
print(f'Exported {len(rows)} records')
"
```

## Directory Structure

```
predictor/
├── predict.py              # Main entry point
├── database.py             # Database operations
├── forebet_scraper.py      # Web scraper
├── calibration_learner.py  # ML calibration
├── ml_model.py             # ML model
├── requirements.txt        # Dependencies
├── history.db              # SQLite database
├── pr                      # Shell wrapper
├── docs/                   # Documentation
│   ├── command.md
│   ├── COMMANDS.md
│   ├── HISTORICAL_CALIBRATE.md
│   ├── INSTRUCTIONS.md
│   └── README.md
├── scripts/                # Utility scripts
│   ├── install.sh
│   ├── run_pipeline.sh
│   ├── auto_learn.py
│   └── ...
├── data/                   # Data files
│   ├── links.txt
│   ├── upcoming.txt
│   └── ...
├── predictions/            # HTML reports
│   ├── index.html
│   ├── 001.html
│   └── ...
├── ml_models/              # Saved ML models
└── backups/                # Backup files
```

## Troubleshooting

**scikit-learn not installed:**
```bash
.venv/bin/pip install scikit-learn
```

**View errors:**
```bash
grep -i error /tmp/retrain.log
```
