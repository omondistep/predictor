# Football Predictor — Command Reference

## Core Prediction
| Command | Action |
|---|---|
| `pr links.txt` | Predict all matches in file, save to DB, auto-learn |
| `pr https://...` | Predict a single match URL |
| `pr --high-only links.txt` | Show only High / Near Certain picks |
| `pr --json links.txt` | JSON output |
| `pr --no-ml links.txt` | Classic Poisson only (no ML ensemble) |

## Continuous Learning (automated)
| Command | Action |
|---|---|
| `pr --auto-learn` | Full pipeline: scrape results → calibrate → retrain |
| `pr links.txt` | Also runs auto-learn silently after predictions |
| `auto_learn.py` | Same as `pr --auto-learn` (standalone) |
| `auto_learn.py --days-back 365 --max-matches 2000` | Bulk learn: scrape all past results (for initial backlog) |
| `auto_learn.py --scrape-only` | Just scrape results, skip calibration |
| `auto_learn.py --calibrate-only` | Just bias analysis + ML retrain |
| `auto_learn.py --daemon --daemon-interval 43200` | Run as daemon every 12h |
| `auto_learn.py --status` | Dashboard: data volumes, accuracy, retrain status |

## Calibration & Review
| Command | Action |
|---|---|
| `pr --calibrate` | Show accuracy stats, bias corrections, calibration report |
| `pr --calibration-report` | Detailed calibration quality (Brier score, reliability) |
| `pr --learn-calibration` | Analyze bias + auto-retrain ML if needed |
| `pr --force-retrain` | Force full ML model retrain |
| `pr --review` | Interactive review (manual score entry) |
| `pr --review played.txt` | Auto-review URLs from file |
| `scrape_results.py` | Batch scrape `played.txt` URLs → `scraped_results.json` |
| `update_db_from_results.py` | Import `scraped_results.json` into DB |

## Scheduling (full automation)
| Command | Action |
|---|---|
| `./install_auto_learn.sh` | Interactive: choose systemd or cron |
| `./install_auto_learn.sh --systemd` | Install systemd daily timer |
| `./install_auto_learn.sh --cron` | Install crontab (6am/6pm) |
| `./install_auto_learn.sh --status` | Show current schedule + learning status |
| `./install_auto_learn.sh --remove` | Remove all scheduling |

## Development
| Command | Action |
|---|---|
| `ml_model.py --train` | Retrain ML models from game data + history.db |
| `ml_model.py --load` | Load existing model and test |
| `cnfupdate.py` | Update league profiles from played matches |
| `historical_calibrate.py` | Batch league profile updates from past results |
| `pr --calibrate` | View league accuracy + suggested profile tweaks |

## Aliases (source aliases.sh or use pr wrapper)
| Alias | Maps to |
|---|---|
| `predict` | `python3 predict.py` |
| `predict-calibrate` | `python3 predict.py --calibrate` |
| `predict-review` | `python3 predict.py --review` |
| `auto-learn` | `python3 auto_learn.py` |
| `learn-status` | `python3 auto_learn.py --status` |
| `learn-daemon` | `python3 auto_learn.py --daemon` |
| `scrape-results` | `python3 scrape_results.py` |

## Typical Workflow
```
# 1. Predict upcoming matches
pr links.txt

# 2. Initial bulk learning (after many predictions have actual results)
auto_learn.py --days-back 365 --max-matches 2000

# 3. Check learning status
learn-status

# 4. Set up automated daily learning
./install_auto_learn.sh --systemd

# 5. Check calibration after enough data
predict-calibrate
```
pr links.txt && auto_learn.py --days-back 365 --max-matches 2000
 python3 auto_learn.py --days-back 14




 ./run_pipeline.sh 
