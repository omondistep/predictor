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

def analyze_calibration(min_samples: int = 10, days_back: int = 365):
    """Analyze all reviewed predictions and store bias corrections.

    For each league/market/probability-bucket, compares the mean predicted
    probability against the actual hit rate and stores the bias.
    """
    conn = get_db()
    cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    rows = conn.execute("""
        SELECT m.id, m.league, c.market, c.our_prediction,
               m.actual_home_goals, m.actual_away_goals,
               c.confidence, c.correct,
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
            # We don't store model prob for O/U directly in DB, compute from odds
            actual_ou = "Over" if (hg + ag) > thresh else "Under"
            actual_correct = 1 if (pred.startswith("Over") and actual_ou == "Over") or \
                                   (pred.startswith("Under") and actual_ou == "Under") else 0
            # Estimate implied probability from odds as proxy for model prob
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

def apply_bias_correction(league: str, market: str, threshold: str,
                          prob: float, min_samples: int = 10) -> float:
    """Apply a learned bias correction to a probability.

    Looks up the calibration bias for the given league/market/threshold/bucket
    and adjusts the probability accordingly.
    """
    if prob <= 0 or prob >= 1:
        return prob
    bucket = _bucket_prob(prob)
    biases = get_calibration_biases(league=league, market=market, min_samples=min_samples)

    # Find the closest matching bucket
    best_bias = 0.0
    best_match = None
    for b in biases:
        if b["threshold"] == threshold and b["bucket"] == bucket:
            best_bias = b["bias"]
            best_match = b
            break

    if best_match is None:
        return prob

    # Apply bias correction: if bias = actual - predicted,
    # corrected_prob = prob + bias
    corrected = prob + best_bias
    corrected = max(0.01, min(0.99, corrected))
    return corrected


def apply_all_bias_corrections(league: str, probs_1x2: dict, probs_ou: dict,
                                min_samples: int = 10) -> tuple:
    """Apply bias corrections to all 1X2 and O/U probabilities."""
    corrected_1x2 = {}
    for outcome, prob in probs_1x2.items():
        corrected_1x2[outcome] = apply_bias_correction(
            league, "1X2", outcome, prob, min_samples
        )

    corrected_ou = {}
    for label, prob in probs_ou.items():
        corrected_ou[label] = apply_bias_correction(
            league, "O/U", label, prob, min_samples
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
        SELECT c.market, c.our_prediction, c.correct,
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
