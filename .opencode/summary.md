# Session Summary

## Objective
- Produce a read-only, line-referenced map of the football betting prediction system at `/home/stdk/predictor`: which markets are predicted, where each probability/pick/confidence/stake is derived, how Poisson+ML probabilities are merged, what is stored per match, how calibration is applied, whether first-half markets exist, how HTML/best-pick reports are generated, and how EV/value picks work.
- Do NOT modify anything. No git changes, no code edits, no commands that mutate state.

## Important Details
- Read-only research; 7 specific questions to answer with `file:line` citations.
- Main files: `predict.py` (6886 lines), `forebet_scraper.py` (~1700), `database.py` (1691), `ml_model.py` (~3300), `calibration_learner.py` (~2700), `synthesis.py`.
- Markets tracked in code: **1X2, O/U (1.5/2.5/3.5/4.5/5.5), BTTS, DNB, DC** (per `_MARKET_THRESHOLDS` predict.py:471-474 and `_pick_odds` predict.py:3293-3312).
- Recent git state: `c123fbd Fix market selection bias, cap low-reliability confidence, retrain ML`; working tree modified: `AGENTS.md`, `forebet_scraper.py`, `history.db`, `links/*.html`, `ml_model.py`, `ml_models/` joblib files.
- `calibration_params.json` top-level keys: `last_retrained`, `markets`; markets present: `1X2`, `O/U`, `BTTS`, `DNB`, `DC` (old format with `X_train/y_train/n_samples`; `apply_calibration` also supports `X_thresholds/y_thresholds`).
- Verdict on Q5 first-half: **no first-half markets exist** — only HT *score* data is parsed/stored/displayed; no ht 1X2 / ht O/U / ht BTTS predictions are generated anywhere.

## Work State
### Completed

**Q1 — Which markets & where probabilities are derived:**
- **Poisson core** (predict.py): `poisson_prob` 1238, `poisson_cdf` 1244, `prob_exact_score` 1249, `prob_home_win` 1253, `prob_draw` 1264, `prob_away_win` 1271, `prob_over` 1275 (`1 - poisson_cdf(total, int(threshold))`), `_ou_exp` 1281, `prob_btts` 1300.
- **1X2**: probabilities blended from ensemble at 3205-3232 (with calibration applied at 3272-3274); pick+confidence threshold logic 3364-3469 in `analyze_from_data` (predict.py:2820). 
- **O/U 1.5/2.5/3.5**: Poisson Over prob blended with ML signal (threshold-dependent ML weight: 0.35/0.15/0.25 at 3762-3768), monotonicity enforced 3770-3771, value-vs-base-rate direction pick 3776-3790, confidence mapping 3792-3801; over/under gates and venue stats caps 3803-3905.
- **O/U 4.5/5.5**: Over-side only, value-gated 3907-3928.
- **DNB**: derived from 1X2 (p_home/(p_home+p_draw) normalized at 3699-3706) — volatility-gated, capped at Medium-High by design, separate odds floors 1.40/1.45 (3691-3697).
- **DC**: 1X/X2/12 derived from 1X2 sums (3714-3750); "12" only in low-draw leagues (p_draw<0.22).
- **BTTS**: blended Poisson + Forebet match BTTS% + team rates (3946-3958), common-opponent boost (3960-3968), ML agreement gate `ml_btts_yes >= 0.52` for Yes (3986-4003), strict No gate (value>0.14 & p>=0.58, Medium cap) 4004-4011.
- **Correct score** (`prob_exact_score`, 1249): only used as `score_lean` estimate (4251-4255), displayed via `our_score_lean` (6050) — NOT a betting market, no stake.

**Q2 — How Poisson+ML probabilities merge into a best pick:**
- `ml_model.py`: `predict_from_row` 1175-1205 → `ml_prob_home/draw/away/under/over`; `poisson_predict` 1434 → `prob_home`, `prob_over` (keys at 1563-1566); `ensemble_predict` 1593-1790: Poisson + ML + Forebet weighted blend; default per-market `MARKET_WEIGHT_PROFILES` 1632; `get_dynamic_weights` (database.py:1173-1212, defaults ml 0.25/poisson 0.35/forebet 0.25/default 0.15) blended 80/20 at 1634-1637; `p_h` 1721-1726, `p_over` 1742-1745; output keys `prob_home`/`prob_over` at 1776-1779.
- O/U 2.5 disagreement reconciliation: ML direction wins when ML≥52% (predict.py:4018-4052, flip logic). BTTS ML agreement gate at 3972-3987.
- **Aggregation**: every market's pick is appended via local `add()` (3332-3362) which runs `_value_adjust` (3317) odds-floor capping, league-market accuracy skip/damp, and records `model_prob`, `implied_prob`, `value_ratio`. Then:
  - Coverage mapping 4068-4078 (DC=2, O/U/BTTS=3, 1X2/DNB=1).
  - Holistic synthesis re-ranks candidates by `decision_value` (4080-4103, `synthesis.py::synthesize`); league-market accuracy boosts/penalties 4108-4125.
  - Primary pick = top non-always-show candidate (4136-4137).
  - Kelly stake sizing = Quarter-Kelly, 5% max (4417-4426), scaled by league-market accuracy (4428-4433), red-flag haircuts ×0.75/×0.5 (4435-4439); `_kelly_stake` key 4454, saved to `our_stake` (6051).
  - Post-selection adjustment gates: BTTS-No fallback (4139-4150), disagreement veto capping 1X2 High→Medium/Low when both ML and Forebet oppose (4152-4174), league-reliability gate capping/suppressing picks (4176-4200), red-flag gate capping confidence (4202-4228).

**Q3 — DB fields stored:**
- `matches` table schema: database.py:25-119 — odds 47-53, forebet 56-61, `our_prediction`/`our_confidence`/`our_score_lean`/`our_stake`/`our_market` 64-68, `method_used` 70-80 (poisson_prob_home/draw/away, ml_prob_home/draw/away, forebet_prob_home/draw/away at 71-80), actuals 83-86, `ht_home_goals`/`ht_away_goals` 89-90, injuries 93-104, FBref xG 107-115, `created_at`/`reviewed_at` 117-118.
- **Per match DB write** (`run_forebet_predictions`): `db_data` built at predict.py:6046-6073 → `save_prediction` (database.py:360, INSERT OR REPLACE keyed by `forebet_url`). `our_stake` = `pred.get("_kelly_stake", 0.0)` (6051); HT goals saved at 6159-6160. `method_used` = `"+"`.join(method_parts) (6053).
- **What is NOT stored**: `prob_over`/`prob_btts` columns do not exist (grep confirms); per-market `all_picks` model probabilities are stored in separate `market_picks` table (via `store_market_results`, database.py:982-1100 with odds_map 997-1007).
- `get_base_rates` database.py:921 (per-league empirical rates with blend toward global), `pick_base_rate` 964-979 maps `(market,pick)` → rates (used at O/U 3776-3777, BTTS 3981-3982).
- League/market per-match result tracking: `record_ml_league_result` (6119-6122), `record_model_market_result` (6124-6136).

**Q4 — Calibration scope:**
- `apply_calibration` implementation: calibration_learner.py:494-529 (reads `calibration_params.json` 506; for `1X2` returns raw prob unchanged 521-530 — isotonic would be flat; O/U/BTTS get isotonic reconstruction 532+).
- **But predict.py only calls `apply_calibration("1X2", p_home, ...)` at 3272-3274** — which is a **no-op** (returns raw clusters). So isotonic O/U/BTTS/DNB/DC calibration exists in `calibration_params.json` but is **never invoked** in the analyze path.
- Actual 1X2 adjustments in predict.py come from: `_MARKET_THRESHOLDS["1X2"]` (scaled at 3388-3393 via ML agreement), draw-specific thresholds 3395-3403, draw-proximity penalty 3415-3425, volatility capping 3427-3435, league-reliability damping 3437-3453, and `LeaguePredictor._ld`/hard-league levels.
- `_auto_calibrate_thresholds` (805-883): reads `calibration_log` grouped by market+confidence; **only tightens** (raises) thresholds when actual_pct < target-0.03, using step_map (high:0.08, nc:0.04, mh:0.03, medium:0.02) + cap_map; hierarchy validation 868-878.
- `_value_adjust` (3317-3330): odds-floor based confidence capping (ODDS_FLOORS 3315: NC 1.10, High 1.18, MH 1.28, Medium 1.50), skip Medium if below floor.
- `apply_bias_correction` calibration_learner.py:233-261, `apply_all_bias_corrections` 264-282 — but grep shows these are **not imported/used** in predict.py.
- `pr --learn-calibration`/`--auto-learn` (main:6808-6827) run `run_calibration_learning`/`auto_retrain` — separate training pipeline, not at prediction time.
- `_model_market_strengths` (710-787): per-market per-model accuracy from `model_market_accuracy` table with fallbacks (calibration_log.forebet_correct 747-758, ml_league_accuracy 759-771), credibility blend K=30 (725-726).

**Q5 — First-half / HT markets:**
- **No HT markets predicted.** Grep of `ht_` shows only: scraper data keys, H2H stats, DB columns, result display, and `ht_home_goals`/`ht_away_goals` written to results (predict.py:6159-6160).
- `forebet_scraper.py`: default keys 120-121; `_parse_ht_home` 534 / `_parse_ht_away` 543 parse `"X-Y"` strings; `_parse_score` 272-294 gets HT from `span.ht_scr` (288-294) for finished matches; `ht_goals_for/against` in H2H rows 445-446, 461-462, 477-478, 634-647.
- So the system could compute HT stats for a match but never emits an HT prediction market.

**Q6 — HTML/card generation & best/consistent/high80 reports:**
- `_write_html` (4503) — confidence filter 4509; `_fb_1x2_comparison` 4485-4500; `_comparison_data` 4659; `_picks_consistent` 4838 (`_PICK_IMPLIES`); `_has_yellow_pick` 4845-4861 (conf ≥ Medium AND model-agreed or best_alt==default key); `_consensus_consistent_with_best` 4863-4874; `_high80_data`/`_high80_banner` 4882-4943.
- Reports written at 5463-5512: `latest.html` (archived to `archive/{ts}.html` 5456-5463), `best.html` 5466-5473, `consistent.html` 5475-5482, `high80.html` 5484-5491; auto-opened in browser 5493-5500. `_update_index` 5515+ builds `index.html`.

**Q7 — EV/value detection:**
- EV/value concepts via: `_value_adjust` (3317) + ODDS_FLOORS (3315) + implied prob vs model prob `value_ratio` (3354-3361).
- Base-rate-relative value (not 50%): O/U `value_o/value_u = p - pick_base_rate(...)` (3776-3790); BTTS `value_yes/value_no` (3981-3982) with thresholds.
- Kelly stake: edge = (model_prob/implied_prob)−1; Kelly fraction, quarter-Kelly, 5% cap (4417-4426); scissors by league accuracy + red flags.
- Red-flag detection `detect_red_flags` (4207-4212) vetoes/caps confidence.
- O/U 2.5 ML disagreement flip (4018-4052) is a value-preserving direction reconciliation.

### Active
- Complete. All 7 questions researched with `file:line` citations.

### Blocked
- (none)

## Next Move
- (none — research complete). If user wants, verify `_ml_analysis` per-market picks feed into `_comparison_data` (4659) for the "FB vs ML" comparison used by best.html, and re-run `pr results` — but no code changes were requested.

## Relevant Files
- `/home/stdk/predictor/predict.py`: main analysis (`analyze_from_data` 2820), HTML (`_write_html` 4503), runner/save (`run_forebet_predictions` 5925, db keys 6046-6073).
- `/home/stdk/predictor/ml_model.py`: `poisson_predict` 1434, `ensemble_predict` 1593, `predict_from_row` 1175, Platt-scaled calibrated models (cal_*_1x2/ou) 883-900, 1048-1072.
- `/home/stdk/predictor/calibration_learner.py`: `apply_calibration` 494, bias corrections 233/264, `_auto_calibrate_thresholds` 805, retraining 355/398.
- `/home/stdk/predictor/calibration_params.json`: 0.79MB; markets 1X2/O/U/BTTS/DNB/DC (old format).
- `/home/stdk/predictor/database.py`: schema 25-119, `get_base_rates` 921, `pick_base_rate` 964, `store_market_results` 982, `get_dynamic_weights` 1173.
- `/home/stdk/predictor/forebet_scraper.py`: `_parse_score`/HT 272-294, `_parse_ht_*` 534/543, `_parse_ou_btts` 1014, `_parse_probabilities` 1299, `_parse_odds` 1376.
- `/home/stdk/predictor/synthesis.py`: `synthesize`, `decision_value` fusion (no line refs yet read — not needed for Q7 since predict.py:4080-4103 shows the call).
- `/home/stdk/predictor/history.db`: 36MB SQLite storing matches, calibration, market picks, per-model accuracy.