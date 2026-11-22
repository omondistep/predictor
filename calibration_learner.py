"""
Calibration Learner — ML component that learns from prediction vs actual outcomes.

Analyzes every reviewed prediction across all markets (1X2, O/U, BTTS),
computes bias corrections per league/market/probability-bucket, and
supports automatic model retraining when sufficient new data accumulates.

Key capabilities:
  - Calibration curve analysis (predicted prob vs actual hit rate)
  - Bias correction storage and retrieval per league/market/bucket
  - Probability recalibration using Platt scaling parameters
  - Auto-retrain trigger detection based on data volume and drift
  - Calibration quality reporting (Brier score, log loss, reliability diagrams)
"""

import json
import math
import os
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

from database import (
    get_db, save_calibration_bias, get_calibration_biases,
    get_calibration_data_for_retraining, log_retrain,
)

BASE = Path(__file__).parent
MODELS_DIR = BASE / "ml_models"

# Cache for calibration_params.json (mtime-checked) — apply_outcome_calibration
# runs several times per match, and re-reading the JSON each time is wasteful.
_PARAMS_CACHE = {"mtime": None, "outcomes": {}}


def _resolve_result(row: dict) -> Optional[tuple]:
    """Extract actual (home_goals, away_goals) from a row."""
    hg = row.get("actual_home_goals")
    ag = row.get("actual_away_goals")
    if hg is not None and ag is not None:
        return int(hg), int(ag)
    return None


def _outcome_from_score(hg: int, ag: int) -> str:
    if hg > ag:
        return "Home win"
    elif ag > hg:
        return "Away win"
    return "Draw"


def _ou_result(hg: int, ag: int, threshold: float = 2.5) -> str:
    total = hg + ag
    return "Over" if total > threshold else "Under"


def _btts_result(hg: int, ag: int) -> str:
    return "Yes" if hg > 0 and ag > 0 else "No"


def _bucket_prob(prob: float) -> str:
    """Map a probability to a 10% bucket label."""
    bucket = min(int(prob * 10), 9)
    low = bucket * 10
    high = (bucket + 1) * 10
    return f"{low}-{high}%"


def _platt_params_from_logits(pos_logits: np.ndarray, neg_logits: np.ndarray,
                               y_true: np.ndarray) -> tuple:
    """Fit Platt scaling parameters (A, B) using logits and binary labels.

    P(y=1|x) = 1 / (1 + exp(A * f(x) + B))
    where f(x) is the logit (raw model score before sigmoid).
    """
    from sklearn.linear_model import LogisticRegression
    logits = np.concatenate([pos_logits, neg_logits])
    labels = np.concatenate([np.ones_like(pos_logits), np.zeros_like(neg_logits)])
    lr = LogisticRegression(C=1e10, solver="lbfgs")
    lr.fit(logits.reshape(-1, 1), labels)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


# ---------------------------------------------------------------------------
# Calibration analysis
# ---------------------------------------------------------------------------

def analyze_calibration(min_samples: int = 8, days_back: int = 365):
    """Analyze all reviewed predictions and store bias corrections.

    For each league/market/probability-bucket, compares the mean predicted
    probability against the actual hit rate and stores the bias.

    Buckets are stored from `min_samples` (default 8) observations — small
    leagues legitimately produce thin buckets, and dropping them entirely is
    what left e.g. Paraguay/Ecuador/Kazakhstan O-U buckets uncorrected. The
    application side (apply_bias_correction) shrinks each stored bias by
    n/(n+K) so thin buckets move probabilities only slightly.
    """
    conn = get_db()
    cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    rows = conn.execute("""
        SELECT m.id, m.league, c.market, c.our_prediction,
               m.actual_home_goals, m.actual_away_goals,
               c.confidence, c.correct, c.model_prob,
               m.poisson_prob_home, m.poisson_prob_draw, m.poisson_prob_away,
               m.ml_prob_home, m.ml_prob_draw, m.ml_prob_away,
               m.forebet_prob_home, m.forebet_prob_draw, m.forebet_prob_away,
               m.odds_over25, m.odds_under25,
               m.reviewed_at
        FROM calibration_log c
        JOIN matches m ON m.id = c.match_id
        WHERE m.reviewed = 1
          AND m.actual_home_goals IS NOT NULL
          AND (m.reviewed_at IS NULL OR m.reviewed_at >= ?)
        ORDER BY m.reviewed_at
    """, (cutoff,)).fetchall()
    conn.close()

    if not rows:
        print("[calibrate] No reviewed matches to analyze.")
        return

    # Group data for calibration curve computation
    buckets_1x2 = defaultdict(lambda: {"preds": [], "actuals": [], "count": 0})
    buckets_ou = defaultdict(lambda: {"preds": [], "actuals": [], "count": 0})
    buckets_btts = defaultdict(lambda: {"preds": [], "actuals": [], "count": 0})

    corrected_count = 0

    for row in rows:
        r = dict(row)
        result = _resolve_result(r)
        if not result:
            continue
        hg, ag = result
        league = r["league"] or "unknown"
        market = r["market"] or ""

        if market == "1X2":
            pred = r["our_prediction"] or ""
            # Get the model probability for the predicted outcome
            prob_home = r.get("poisson_prob_home") or 0
            prob_draw = r.get("poisson_prob_draw") or 0
            prob_away = r.get("poisson_prob_away") or 0
            if pred == "Home win":
                model_prob = prob_home
                actual_correct = 1 if hg > ag else 0
            elif pred == "Draw":
                model_prob = prob_draw
                actual_correct = 1 if hg == ag else 0
            elif pred == "Away win":
                model_prob = prob_away
                actual_correct = 1 if ag > hg else 0
            else:
                continue

            if model_prob and model_prob > 0:
                bucket_key = (league, market, pred, _bucket_prob(model_prob))
                buckets_1x2[bucket_key]["preds"].append(model_prob)
                buckets_1x2[bucket_key]["actuals"].append(actual_correct)
                buckets_1x2[bucket_key]["count"] += 1

        elif market == "O/U":
            pred = r["our_prediction"] or ""
            # Determine which threshold (default 2.5)
            thresh = 2.5
            m_thresh = re.search(r"(\d+\.?\d*)", pred)
            if m_thresh:
                thresh = float(m_thresh.group(1))
            actual_ou = "Over" if (hg + ag) > thresh else "Under"
            actual_correct = 1 if (pred.startswith("Over") and actual_ou == "Over") or \
                                   (pred.startswith("Under") and actual_ou == "Under") else 0
            # Use the model_prob stored in calibration_log (from predict.py),
            # which is the actual Poisson/ML probability for this pick.
            # Fall back to odds-implied probability only when missing.
            model_prob = r.get("model_prob")
            if not model_prob or model_prob <= 0:
                odds_o25 = r.get("odds_over25")
                odds_u25 = r.get("odds_under25")
                if pred.startswith("Over") and odds_o25 and odds_o25 > 1:
                    model_prob = 1.0 / odds_o25
                elif pred.startswith("Under") and odds_u25 and odds_u25 > 1:
                    model_prob = 1.0 / odds_u25
                else:
                    model_prob = 0.5

            if model_prob > 0:
                bucket_key = (league, market, f"Over{int(thresh)}", _bucket_prob(model_prob))
                buckets_ou[bucket_key]["preds"].append(model_prob)
                buckets_ou[bucket_key]["actuals"].append(actual_correct)
                buckets_ou[bucket_key]["count"] += 1

        elif market == "BTTS":
            pred = r["our_prediction"] or ""
            actual_btts = "Yes" if hg > 0 and ag > 0 else "No"
            actual_correct = 1 if pred == actual_btts else 0
            # Use the model_prob stored in calibration_log (from predict.py),
            # which is the blended Poisson/Forebet probability for this pick.
            model_prob = r.get("model_prob")
            if not model_prob or model_prob <= 0:
                model_prob = 0.5
            if model_prob > 0:
                bucket_key = (league, market, pred, _bucket_prob(model_prob))
                buckets_btts[bucket_key]["preds"].append(model_prob)
                buckets_btts[bucket_key]["actuals"].append(actual_correct)
                buckets_btts[bucket_key]["count"] += 1

    # Compute and store bias corrections
    all_buckets = [buckets_1x2, buckets_ou, buckets_btts]
    for bucket_dict in all_buckets:
        for (league, market, threshold, bucket), data in bucket_dict.items():
            n = data["count"]
            if n < min_samples:
                continue
            predicted_mean = float(np.mean(data["preds"]))
            actual_mean = float(np.mean(data["actuals"]))
            bias = save_calibration_bias(league, market, threshold, bucket,
                                         predicted_mean, actual_mean, n)
            corrected_count += 1
            if abs(bias) > 0.05:
                direction = "overconfident" if bias < 0 else "underconfident"
                print(f"  [{league}] {market} {threshold} @ {bucket}: "
                      f"pred={predicted_mean:.2f} actual={actual_mean:.2f} "
                      f"({direction} by {abs(bias):.1%}) [{n} samples]")

    print(f"[calibrate] Analyzed {corrected_count} bias buckets across "
          f"{sum(len(d) for d in all_buckets)} total entries")


# ---------------------------------------------------------------------------
# Bias correction application
# ---------------------------------------------------------------------------

def _norm_ou_threshold(threshold: str) -> str:
    """Normalise an O/U threshold label to the storage key form.

    analyze_calibration stores O/U buckets keyed by line, side-agnostic:
    "Over3" means the 3.5 line (parsed via int()). Application sites pass
    pick labels ("Over 3.5", "Under 3.5") or the bare side ("Over"/"Under",
    which in predict.py's main blend always means the 2.5 line). Without
    this normalisation those lookups never matched, so per-league O/U bias
    corrections were stored but never applied.
    """
    if not threshold:
        return threshold
    if threshold in ("Over", "Under"):
        return "Over2"  # main blended O/U is the 2.5 line
    m = re.match(r"(?:Over|Under)\s*(\d+\.?\d*)$", threshold.strip(), re.IGNORECASE)
    if m:
        try:
            return f"Over{int(float(m.group(1)))}"
        except ValueError:
            return threshold
    return threshold


def apply_bias_correction(league: str, market: str, threshold: str,
                          prob: float, min_samples: int = 8,
                          max_bias: float = 0.40,
                          shrink_k: float = 15.0) -> float:
    """Apply a learned bias correction to a probability.

    Looks up the calibration bias for the given league/market/threshold/bucket
    and adjusts the probability accordingly. Corrections backed by fewer than
    `min_samples` reviewed examples are ignored; everything else is shrunk
    toward zero by the empirical-Bayes factor n/(n + shrink_k), so a 10-sample
    bucket moves the probability about 40% as far as its raw bias suggests and
    a 100-sample bucket about 87%. The applied magnitude is additionally capped
    at `max_bias` so even thick-sample noise cannot destroy a probability.
    """
    if prob <= 0 or prob >= 1:
        return prob
    bucket = _bucket_prob(prob)
    if market == "O/U":
        threshold = _norm_ou_threshold(threshold)
    biases = get_calibration_biases(league=league, market=market, min_samples=min_samples)

    # Find the closest matching bucket
    best_bias = 0.0
    best_n = 0
    best_match = None
    for b in biases:
        if b["threshold"] == threshold and b["bucket"] == bucket:
            best_bias = b["bias"]
            best_n = b.get("sample_count") or 0
            best_match = b
            break

    if best_match is None:
        return prob

    # Empirical-Bayes shrinkage: thin buckets are mostly noise
    if shrink_k > 0 and best_n > 0:
        best_bias *= best_n / (best_n + shrink_k)

    # Cap the correction magnitude: a |bias| > max_bias on a modest sample is
    # more likely noise than signal, so don't let it swallow the base model.
    best_bias = max(-max_bias, min(max_bias, best_bias))

    # Apply bias correction: if bias = actual - predicted,
    # corrected_prob = prob + bias
    corrected = prob + best_bias
    corrected = max(0.01, min(0.99, corrected))
    return corrected


def apply_all_bias_corrections(league: str, probs_1x2: dict, probs_ou: dict,
                                min_samples: int = 8,
                                max_bias: float = 0.40) -> tuple:
    """Apply bias corrections to all 1X2 and O/U probabilities."""
    corrected_1x2 = {}
    for outcome, prob in probs_1x2.items():
        corrected_1x2[outcome] = apply_bias_correction(
            league, "1X2", outcome, prob, min_samples, max_bias
        )

    corrected_ou = {}
    for label, prob in probs_ou.items():
        corrected_ou[label] = apply_bias_correction(
            league, "O/U", label, prob, min_samples, max_bias
        )

    # Renormalize 1X2
    total_1x2 = sum(corrected_1x2.values())
    if total_1x2 > 0:
        for k in corrected_1x2:
            corrected_1x2[k] /= total_1x2

    # Renormalize O/U
    total_ou = sum(corrected_ou.values())
    if total_ou > 0:
        for k in corrected_ou:
            corrected_ou[k] /= total_ou

    return corrected_1x2, corrected_ou


# ---------------------------------------------------------------------------
# Platt scaling for ML model probability calibration
# ---------------------------------------------------------------------------

def compute_platt_calibration(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    """Compute Platt scaling parameters to calibrate probabilities.

    Fits a logistic regression on logit-transformed probabilities.
    Returns {'a': slope, 'b': intercept} for the calibration function:
        P_calibrated = 1 / (1 + exp(a * logit(p) + b))
    where logit(p) = ln(p / (1-p))
    """
    eps = 1e-10
    p = np.clip(probabilities, eps, 1 - eps)
    logits = np.log(p / (1 - p))
    a, b = _platt_params_from_logits(
        logits[labels == 1],
        logits[labels == 0],
        labels
    )
    return {"a": a, "b": b}


def apply_platt_calibration(probabilities: np.ndarray, params: dict) -> np.ndarray:
    """Apply Platt scaling to uncalibrated probabilities."""
    eps = 1e-10
    p = np.clip(probabilities, eps, 1 - eps)
    logits = np.log(p / (1 - p))
    calibrated = 1.0 / (1.0 + np.exp(params["a"] * logits + params["b"]))
    return np.clip(calibrated, eps, 1 - eps)


# ---------------------------------------------------------------------------
# Auto-retrain trigger
# ---------------------------------------------------------------------------

def check_retrain_needed(min_new_examples: int = 50, max_days_since_retrain: int = 90) -> bool:
    """Check if the ML model should be retrained based on new calibration data."""
    stats = get_calibration_data_for_retraining()
    total = stats["total_calibration_entries"]
    recent_30d = stats["recent_30d"]
    last_examples = stats["last_retrain_examples"] or 0

    # Trigger if enough new data since last retrain
    new_since_last = total - last_examples
    if new_since_last >= min_new_examples:
        print(f"[calibrate] Retrain triggered: {new_since_last} new examples "
              f"since last retrain ({last_examples} total)")
        return True

    # Trigger if recent accuracy has dropped significantly
    for market_acc in stats["accuracy_by_market"]:
        if market_acc["total"] >= 30 and market_acc["pct"] < 40.0:
            print(f"[calibrate] Retrain triggered: {market_acc['market']} "
                  f"accuracy dropped to {market_acc['pct']}% "
                  f"({market_acc['total']} samples)")
            return True

    return False


def auto_retrain(force: bool = False):
    """Retrain ML models if conditions are met. Returns True if retrained."""
    try:
        from ml_model import train, MLPredictor
    except ImportError:
        print("[calibrate] Could not import ml_model for retraining")
        return False

    stats = get_calibration_data_for_retraining()
    total_before = stats["total_calibration_entries"]

    if not force and not check_retrain_needed():
        print("[calibrate] No retrain needed")
        return False

    # Load current model to get baseline accuracy
    old_model = MLPredictor.load(auto_train=False)
    acc_1x2_before = old_model.accuracy_1x2 if old_model else 0.0
    acc_ou_before = old_model.accuracy_ou if old_model else 0.0
    examples_before = old_model.training_examples if old_model else 0

    print(f"[calibrate] Retraining ML model (had {examples_before} examples)...")
    new_model = train(force=True)

    if new_model:
        log_retrain(
            triggered_by="auto" if not force else "manual",
            examples_before=examples_before,
            examples_after=new_model.training_examples,
            acc_1x2_before=acc_1x2_before,
            acc_1x2_after=new_model.accuracy_1x2,
            acc_ou_before=acc_ou_before,
            acc_ou_after=new_model.accuracy_ou,
        )
        print(f"[calibrate] Retrained: {examples_before} -> {new_model.training_examples} examples, "
              f"1X2: {acc_1x2_before:.3f} -> {new_model.accuracy_1x2:.3f}, "
              f"O/U: {acc_ou_before:.3f} -> {new_model.accuracy_ou:.3f}")
        return True

    print("[calibrate] Retraining failed")
    return False


def retrain_from_results(force: bool = False):
    """Retrain calibration using isotonic regression on historical results.

    This function:
    1. Fetches all historical predictions with results
    2. For each market (1X2, O/U, BTTS, DNB, DC), fits a picked-side isotonic
       regression (legacy single curve)
    3. Fits PER-OUTCOME maps — one curve per outcome, trained on every match,
       not only picked-side rows:
         - 1X2: Home / Draw / Away, from matches.poisson_prob_* (the final
           stated 1X2 probabilities stored per fixture)
         - O/U: one Over curve per line (1.5/2.5/3.5), reconstructed from
           calibration_log pick-side rows as (P(Over), 1{over})
         - BTTS: one Yes curve (No is its complement)
       The single-curve approach collapsed 1X2 to ~0.33 because it pooled
       outcomes with different base rates; per-outcome maps keep the relative
       ordering and are renormalized at apply time.
    4. Saves calibration parameters to calibration_params.json

    Returns True if retrained successfully.
    """
    from sklearn.isotonic import IsotonicRegression
    import json

    print("[calibrate] Starting isotonic regression retraining...")

    # Get calibration data for each market
    markets = ["1X2", "O/U", "BTTS", "DNB", "DC"]
    calibration_data = {}

    for market in markets:
        conn = get_db()
        rows = conn.execute("""
            SELECT model_prob, odds, correct
            FROM calibration_log
            WHERE market = ? AND correct IS NOT NULL
            ORDER BY created_at ASC
        """, (market,)).fetchall()
        conn.close()

        if len(rows) < 200:  # Need sufficient samples for reliable isotonic regression
            print(f"  [SKIP] {market}: insufficient data ({len(rows)} samples)")
            continue

        # Use model_prob when available, fall back to implied probability from odds
        X_list = []
        y_list = []
        for r in rows:
            if r["model_prob"] is not None:
                X_list.append(r["model_prob"])
            elif r["odds"] is not None and r["odds"] > 1.0:
                X_list.append(1.0 / r["odds"])
            else:
                continue  # Skip rows with neither model_prob nor odds
            y_list.append(r["correct"])

        if len(X_list) < 10:
            print(f"  [SKIP] {market}: insufficient usable data ({len(X_list)} samples with prob/odds)")
            continue

        X = np.array(X_list)
        y = np.array(y_list)

        # Fit isotonic regression
        try:
            ir = IsotonicRegression(increasing='auto', out_of_bounds='clip')
            ir.fit(X, y)

            # Store ONLY fitted parameters (thresholds + mapped values), not raw data
            calibration_data[market] = {
                "X_thresholds": ir.X_thresholds_.tolist(),
                "y_thresholds": ir.y_thresholds_.tolist(),
                "n_samples": len(X_list),
            }

            # Calculate accuracy improvement
            raw_accuracy = np.mean(y)
            calibrated_predictions = ir.predict(X)
            calibrated_accuracy = np.mean((calibrated_predictions > 0.5) == y)

            print(f"  [OK] {market}: {len(X_list)} samples, "
                  f"raw acc: {raw_accuracy:.1%}, calibrated acc: {calibrated_accuracy:.1%}")

        except Exception as e:
            print(f"  [ERR] {market}: {e}")
            continue

    # ── Per-outcome maps ──
    outcome_data = _fit_outcome_maps()

    if not calibration_data and not outcome_data:
        print("[calibrate] No markets with sufficient data for retraining")
        return False

    # Save calibration parameters (legacy per-market + per-outcome)
    params_path = BASE / "calibration_params.json"
    # Preserve any existing outcome maps if this fit produced none
    if outcome_data or not params_path.exists():
        outcomes_to_save = outcome_data
    else:
        try:
            outcomes_to_save = json.loads(params_path.read_text()).get("outcomes", {})
        except Exception:
            outcomes_to_save = {}
    params = {
        "last_retrained": datetime.now().isoformat(),
        "markets": calibration_data,
        "outcomes": outcomes_to_save,
    }

    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    _PARAMS_CACHE.clear()
    _PARAMS_CACHE.update({"mtime": None, "outcomes": {}})

    print(f"[calibrate] Saved calibration parameters to {params_path}")
    print(f"[calibrate] Retrained {len(calibration_data)} markets, "
          f"{sum(len(v) for v in outcomes_to_save.values())} outcome maps")

    return True


def _fit_isotonic_map(X, y, min_n: int = 200, min_pos: int = 50, min_neg: int = 50):
    """Fit one isotonic map; returns threshold dict or None."""
    from sklearn.isotonic import IsotonicRegression
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(X) < min_n or int(y.sum()) < min_pos or int(len(y) - y.sum()) < min_neg:
        return None
    try:
        # increasing=True: a higher stated probability must never map to a
        # lower realised frequency. 'auto' can flip to decreasing on flat
        # slices, which would invert the calibration.
        ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
        ir.fit(X, y)
        return {
            "X_thresholds": ir.X_thresholds_.tolist(),
            "y_thresholds": ir.y_thresholds_.tolist(),
            "n_samples": int(len(X)),
        }
    except Exception:
        return None


def _fit_outcome_maps(days_back: int = 730) -> dict:
    """Fit per-outcome isotonic maps on ALL resolved predictions.

    1X2 uses the final stated probabilities stored per match
    (matches.poisson_prob_home/draw/away — despite the column name these hold
    the blended ex-ante probabilities predict.py saves). O/U and BTTS are
    reconstructed from calibration_log pick-side rows into (P(outcome),
    1{outcome}) pairs, so both sides of each market contribute.
    """
    cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    outcomes: dict = {"1X2": {}, "O/U": {}, "BTTS": {}}

    # ── 1X2: full stated distribution per match ──
    conn = get_db()
    rows = conn.execute("""
        SELECT poisson_prob_home, poisson_prob_draw, poisson_prob_away,
               actual_home_goals, actual_away_goals
        FROM matches
        WHERE reviewed = 1
          AND actual_home_goals IS NOT NULL
          AND actual_away_goals IS NOT NULL
          AND poisson_prob_home IS NOT NULL
          AND poisson_prob_draw IS NOT NULL
          AND poisson_prob_away IS NOT NULL
          AND (reviewed_at IS NULL OR reviewed_at >= ?)
    """, (cutoff,)).fetchall()
    conn.close()

    if rows:
        ph = np.array([r["poisson_prob_home"] for r in rows])
        pd_ = np.array([r["poisson_prob_draw"] for r in rows])
        pa = np.array([r["poisson_prob_away"] for r in rows])
        hg = np.array([r["actual_home_goals"] for r in rows])
        ag = np.array([r["actual_away_goals"] for r in rows])
        y_home = (hg > ag).astype(int)
        y_draw = (hg == ag).astype(int)
        y_away = (hg < ag).astype(int)
        for outcome, X, y in (("Home win", ph, y_home),
                              ("Draw", pd_, y_draw),
                              ("Away win", pa, y_away)):
            m = _fit_isotonic_map(X, y, min_n=500)
            if m:
                outcomes["1X2"][outcome] = m
                print(f"  [OK] 1X2 {outcome}: n={m['n_samples']}")
            else:
                print(f"  [SKIP] 1X2 {outcome}: insufficient data")
    else:
        print("  [SKIP] 1X2 outcomes: no stored stated probabilities")

    # ── O/U: per-line Over curves from pick-side rows ──
    conn = get_db()
    ou_rows = conn.execute("""
        SELECT our_prediction, model_prob, correct
        FROM calibration_log
        WHERE market = 'O/U' AND correct IS NOT NULL AND model_prob IS NOT NULL
    """).fetchall()
    conn.close()

    line_pairs: dict = {}  # "Over 2.5" -> ([p_over...], [y_over...])
    for r in ou_rows:
        pred = r["our_prediction"] or ""
        mp = r["model_prob"]
        if not pred or mp is None or mp <= 0 or mp >= 1:
            continue
        m_thresh = re.search(r"(\d+\.?\d*)", pred)
        if not m_thresh:
            continue
        try:
            thresh = float(m_thresh.group(1))
        except ValueError:
            continue
        if thresh not in (1.5, 2.5, 3.5):
            continue
        key = f"Over {thresh}"
        if pred.startswith("Over"):
            p_over, y_over = mp, int(r["correct"])
        elif pred.startswith("Under"):
            p_over, y_over = 1.0 - mp, 1 - int(r["correct"])
        else:
            continue
        pair = line_pairs.setdefault(key, ([], []))
        pair[0].append(p_over)
        pair[1].append(y_over)

    for key, (Xs, ys) in sorted(line_pairs.items()):
        m = _fit_isotonic_map(Xs, ys, min_n=200)
        if m:
            outcomes["O/U"][key] = m
            print(f"  [OK] O/U {key}: n={m['n_samples']}")
        else:
            print(f"  [SKIP] O/U {key}: insufficient data ({len(Xs)})")

    # ── BTTS: one Yes curve (No is the complement) ──
    conn = get_db()
    btts_rows = conn.execute("""
        SELECT our_prediction, model_prob, correct
        FROM calibration_log
        WHERE market = 'BTTS' AND correct IS NOT NULL AND model_prob IS NOT NULL
    """).fetchall()
    conn.close()

    Xs, ys = [], []
    for r in btts_rows:
        pred = r["our_prediction"] or ""
        mp = r["model_prob"]
        if mp is None or mp <= 0 or mp >= 1:
            continue
        if pred == "Yes":
            Xs.append(mp)
            ys.append(int(r["correct"]))
        elif pred == "No":
            Xs.append(1.0 - mp)
            ys.append(1 - int(r["correct"]))
    m = _fit_isotonic_map(Xs, ys, min_n=200)
    if m:
        outcomes["BTTS"]["Yes"] = m
        print(f"  [OK] BTTS Yes: n={m['n_samples']}")
    else:
        print(f"  [SKIP] BTTS Yes: insufficient data ({len(Xs)})")

    # Drop empty market sections
    return {k: v for k, v in outcomes.items() if v}


def apply_calibration(market: str, raw_prob: float) -> float:
    """Apply isotonic regression calibration to a raw probability.
    
    Args:
        market: Market type (1X2, O/U, BTTS, DNB, DC)
        raw_prob: Raw model probability (0-1)
    
    Returns:
        Calibrated probability (0-1)
    """
    import json
    
    params_path = BASE / "calibration_params.json"
    if not params_path.exists():
        return raw_prob
    
    try:
        with open(params_path) as f:
            params = json.load(f)
    except:
        return raw_prob
    
    if market not in params.get("markets", {}):
        return raw_prob
    
    market_data = params["markets"][market]

    # 1X2 must NOT be isotonic-calibrated per outcome: each of Home/Draw/Away
    # is trained as an independent binary problem against a ~1/3 base rate, so
    # the regression collapses to a flat line (~0.33) and destroys the relative
    # ordering of the three outcomes (they all renormalize to 33%). The Poisson/
    # ensemble probabilities already sum to 1 and carry the real separation, so
    # for 1X2 we keep the raw probability and let the 1X2 re-normalization in
    # predict.py preserve it. O/U and BTTS are genuine binary markets and keep
    # their calibration.
    if market == "1X2":
        return raw_prob

    # Reconstruct isotonic regression from saved thresholds
    from sklearn.isotonic import IsotonicRegression

    # Support both old format (X_train/y_train) and new format (X_thresholds/y_thresholds)
    if "X_thresholds" in market_data:
        X_thresholds = np.array(market_data["X_thresholds"])
        y_thresholds = np.array(market_data["y_thresholds"])
    elif "X_train" in market_data and "y_train" in market_data:
        # Old format: re-fit from raw data
        X = np.array(market_data["X_train"])
        y = np.array(market_data["y_train"])
        try:
            ir = IsotonicRegression(increasing='auto', out_of_bounds='clip')
            ir.fit(X, y)
            X_thresholds = ir.X_thresholds_
            y_thresholds = ir.y_thresholds_
        except:
            return raw_prob
    else:
        return raw_prob

    try:
        # Linear interpolation over the stored thresholds. Rehydrating an
        # IsotonicRegression from thresholds alone fails under sklearn >= 1.6
        # (predict needs fit-time attributes like X_min_), and the previous
        # bare except silently returned the raw probability.
        calibrated = np.interp(raw_prob, X_thresholds, y_thresholds)
        return float(np.clip(calibrated, 0.01, 0.99))
    except:
        return raw_prob


# ---------------------------------------------------------------------------
# Per-outcome calibration
# ---------------------------------------------------------------------------
#
# One isotonic curve per OUTCOME, fit on every resolved match (not only
# picked-side rows). A single pooled curve per market cannot express that a
# stated 30% Draw lands ~25% of the time while a stated 30% Away lands ~33% —
# and the old 1X2 attempt at per-outcome calibration collapsed to ~0.33
# because each curve was fit one-vs-rest on picked-side rows. Fitting each
# outcome against its own empirical frequency over the full distribution, then
# renormalising, keeps the Home/Draw/Away (and Over/Under) separation intact.

def _load_outcome_maps() -> dict:
    """Load per-outcome maps from calibration_params.json (mtime-cached)."""
    params_path = BASE / "calibration_params.json"
    try:
        mtime = params_path.stat().st_mtime
    except OSError:
        return {}
    if _PARAMS_CACHE.get("mtime") == mtime:
        return _PARAMS_CACHE.get("outcomes") or {}
    try:
        import json as _json
        data = _json.loads(params_path.read_text())
        outcomes = data.get("outcomes") or {}
    except Exception:
        outcomes = {}
    _PARAMS_CACHE["mtime"] = mtime
    _PARAMS_CACHE["outcomes"] = outcomes
    return outcomes


def _apply_isotonic_map(prob: float, map_data: dict) -> float:
    """Apply one stored isotonic map (thresholds format) to a probability.

    Uses plain linear interpolation over the stored thresholds — sklearn's
    IsotonicRegression.predict requires attributes set during fit (X_min_ etc.)
    that a threshold-only reconstruction lacks, so rehydrating the estimator
    silently fails under sklearn >= 1.6 and the bare except returns the raw
    probability. The thresholds ARE the piecewise-linear curve, so interp is
    both correct and dependency-free.
    """
    if not map_data or prob is None or prob <= 0 or prob >= 1:
        return prob
    try:
        xs = np.asarray(map_data["X_thresholds"], dtype=float)
        ys = np.asarray(map_data["y_thresholds"], dtype=float)
        if len(xs) == 0 or len(xs) != len(ys):
            return prob
        return float(np.clip(np.interp(prob, xs, ys), 0.01, 0.99))
    except Exception:
        return prob


def apply_outcome_calibration(market: str, probs: dict) -> dict:
    """Calibrate each outcome with its own isotonic map.

    market: "1X2" | "O/U" | "BTTS"
    probs:
      1X2  — {"Home win": p, "Draw": p, "Away win": p}
      O/U  — {"Over 1.5": p, "Under 1.5": p} (line inferred from key; bare
             "Over"/"Under" means the 2.5 line). Only the Over side carries a
             map; Under is its complement.
      BTTS — {"Yes": p, "No": p}; Yes carries the map, No is its complement.

    Returns a new dict with calibrated probabilities renormalised to sum to 1.
    Missing maps leave the input untouched. Callers should fall back to the
    raw probabilities when the returned dict is unchanged.
    """
    outcomes = _load_outcome_maps()
    maps = outcomes.get(market)
    if not maps or not probs:
        return dict(probs)

    if market == "1X2":
        calibrated = {}
        for outcome in ("Home win", "Draw", "Away win"):
            p = probs.get(outcome)
            m = maps.get(outcome)
            if p is None:
                calibrated[outcome] = 0.0
            elif m:
                calibrated[outcome] = _apply_isotonic_map(p, m)
            else:
                calibrated[outcome] = p
        total = sum(calibrated.values())
        if total > 0:
            return {k: v / total for k, v in calibrated.items()}
        return dict(probs)

    if market == "O/U":
        # Find the Over side and its line
        over_key = next((k for k in probs if k.startswith("Over")), None)
        under_key = next((k for k in probs if k.startswith("Under")), None)
        if over_key is None:
            return dict(probs)
        p_over_raw = probs[over_key]
        line_key = over_key if over_key in maps else (
            "Over 2.5" if over_key == "Over" else over_key)
        m = maps.get(line_key)
        if m and p_over_raw is not None and 0 < p_over_raw < 1:
            p_over = _apply_isotonic_map(p_over_raw, m)
        else:
            p_over = p_over_raw
        out = dict(probs)
        out[over_key] = p_over
        if under_key is not None:
            out[under_key] = 1.0 - p_over
        return out

    if market == "BTTS":
        p_yes_raw = probs.get("Yes")
        m = maps.get("Yes")
        out = dict(probs)
        if m and p_yes_raw is not None and 0 < p_yes_raw < 1:
            out["Yes"] = _apply_isotonic_map(p_yes_raw, m)
            out["No"] = 1.0 - out["Yes"]
        return out

    return dict(probs)


# ---------------------------------------------------------------------------
# Calibration quality metrics
# ---------------------------------------------------------------------------

def brier_score(probabilities: np.ndarray, actuals: np.ndarray) -> float:
    """Compute the Brier score (mean squared error between prob and outcome)."""
    return float(np.mean((probabilities - actuals) ** 2))


def reliability_curve(probabilities: np.ndarray, actuals: np.ndarray,
                      n_bins: int = 10) -> dict:
    """Compute reliability curve data for a calibration plot.

    Returns dict with bin_centers, predicted_mean, actual_mean, counts.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    pred_mean = np.zeros(n_bins)
    actual_mean = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (probabilities >= bins[i]) & (probabilities < bins[i + 1])
        if i == n_bins - 1:
            mask = (probabilities >= bins[i]) & (probabilities <= bins[i + 1])
        count = int(mask.sum())
        counts[i] = count
        if count > 0:
            pred_mean[i] = float(np.mean(probabilities[mask]))
            actual_mean[i] = float(np.mean(actuals[mask]))

    return {
        "bin_centers": bin_centers.tolist(),
        "predicted_mean": pred_mean.tolist(),
        "actual_mean": actual_mean.tolist(),
        "counts": counts.tolist(),
    }


def calibration_report(min_samples: int = 10, verbose: bool = True):
    """Generate a comprehensive calibration quality report."""
    conn = get_db()
    rows = conn.execute("""
        SELECT c.market, c.our_prediction, c.correct, c.model_prob,
               m.league, m.poisson_prob_home, m.poisson_prob_draw, m.poisson_prob_away
        FROM calibration_log c
        JOIN matches m ON m.id = c.match_id
        WHERE c.correct IS NOT NULL
          AND m.poisson_prob_home IS NOT NULL
    """).fetchall()
    conn.close()

    if not rows:
        print("[calibrate] No data for calibration report")
        return {}

    report = {}
    for market in ("1X2", "O/U", "BTTS"):
        market_rows = [r for r in rows if r["market"] == market]
        if len(market_rows) < min_samples:
            continue

        probs = []
        corrects = []
        for r in market_rows:
            pred = r["our_prediction"] or ""
            if market == "1X2":
                pmap = {"Home win": r["poisson_prob_home"],
                        "Draw": r["poisson_prob_draw"],
                        "Away win": r["poisson_prob_away"]}
                p = pmap.get(pred)
            else:
                # Use stored model_prob from calibration_log
                p = r["model_prob"]
                if not p or p <= 0:
                    p = 0.5
            if p and p > 0:
                probs.append(p)
                corrects.append(r["correct"])

        if len(probs) < min_samples:
            continue

        probs_arr = np.array(probs, dtype=np.float32)
        corrects_arr = np.array(corrects, dtype=np.float32)
        bs = brier_score(probs_arr, corrects_arr)
        rel = reliability_curve(probs_arr, corrects_arr, n_bins=10)

        report[market] = {
            "samples": len(probs),
            "brier_score": round(bs, 4),
            "accuracy": float(np.mean(corrects_arr)),
            "mean_predicted": float(np.mean(probs_arr)),
            "reliability": rel,
        }

        if verbose:
            print(f"\n{'='*60}")
            print(f"CALIBRATION QUALITY: {market}")
            print(f"{'='*60}")
            print(f"  Samples:        {report[market]['samples']}")
            print(f"  Accuracy:       {report[market]['accuracy']:.1%}")
            print(f"  Mean predicted: {report[market]['mean_predicted']:.1%}")
            print(f"  Brier score:    {report[market]['brier_score']:.4f}")
            print(f"  {'Bin':<10} {'Predicted':<12} {'Actual':<12} {'Count':<8}")
            print(f"  {'-'*42}")
            for i in range(10):
                if rel["counts"][i] > 0:
                    print(f"  {rel['bin_centers'][i]:<10.2f} "
                          f"{rel['predicted_mean'][i]:<12.3f} "
                          f"{rel['actual_mean'][i]:<12.3f} "
                          f"{int(rel['counts'][i]):<8}")

    return report


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_calibration_learning(analyze: bool = True, retrain: bool = False,
                              report: bool = False, force_retrain: bool = False):
    """Run the full calibration learning pipeline."""
    print("=" * 60)
    print("CALIBRATION LEARNING ENGINE")
    print("=" * 60)

    if analyze:
        print("\n[1/3] Analyzing prediction bias...")
        analyze_calibration(min_samples=10)

    if retrain or force_retrain:
        print("\n[2/3] Checking retrain conditions...")
        did_retrain = auto_retrain(force=force_retrain)
        if did_retrain:
            print("  Model retrained successfully")
        else:
            print("  No retrain performed")
        # Per-outcome isotonic maps (1X2 Home/Draw/Away, O/U per-line, BTTS)
        # are refit on every learning pass — they are cheap and always benefit
        # from the latest results.
        try:
            retrain_from_results()
        except Exception as e:
            print(f"  Outcome-map refit failed: {e}")

    if report:
        print("\n[3/3] Generating calibration report...")
        calibration_report(verbose=True)

    print("\n" + "=" * 60)
    print("Calibration learning complete")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calibration Learner")
    parser.add_argument("--analyze", action="store_true", help="Analyze prediction bias")
    parser.add_argument("--retrain", action="store_true", help="Auto-retrain if needed")
    parser.add_argument("--force-retrain", action="store_true", help="Force model retraining")
    parser.add_argument("--report", action="store_true", help="Generate calibration quality report")
    args = parser.parse_args()

    run_calibration_learning(
        analyze=args.analyze,
        retrain=args.retrain,
        report=args.report,
        force_retrain=args.force_retrain,
    )
