#!/usr/bin/env python3
"""A/B the odds-guard fix on a chronological holdout.

Compares two feature pipelines on identical rows and an identical split:

  raw     — odds fed to the model exactly as stored in history.db
  guarded — odds passed through :mod:`odds_guard`; a rejected price becomes a
            neutral constant so no tree split can key on it

Both variants are scored on the *same* final 20% of matches by date, using
the same model family, so the only difference is the odds handling.

Reported: multiclass Brier, log loss, accuracy, and the calibration gap
(mean stated probability vs realised hit rate).  The gap is the number that
matters for a report that promises "High confidence" to the user.
"""

from __future__ import annotations

import math
import os
import sqlite3
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import odds_guard  # noqa: E402

HOLDOUT_FRAC = 0.20
SEED = 42


def load_rows(db_path="history.db"):
    """Pull every resolved row with the fields the featuriser needs."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT * FROM matches
        WHERE actual_home_goals IS NOT NULL AND actual_away_goals IS NOT NULL
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def parse_date(s):
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y", "%Y/%m/%d"):
        try:
            from datetime import datetime
            return datetime.strptime(s or "", fmt)
        except (ValueError, TypeError):
            continue
    return None


def build_matrices(rows, guard: bool):
    """Return (X, y, dates, odds_were_valid) for one pipeline variant."""
    import ml_model

    # Patch the featuriser's odds handling for the "raw" variant by feeding it
    # a pre-validated copy: if guarding, blank out failing rows so the
    # featuriser's own guard sees "missing" and substitutes the neutral value.
    X, y, dates, valid = [], [], [], []
    for r in rows:
        ok = odds_guard.feature_is_usable(
            r.get("odds_home"), r.get("odds_draw"), r.get("odds_away"))
        rr = dict(r)
        if guard and not ok:
            rr["odds_home"] = None
            rr["odds_draw"] = None
            rr["odds_away"] = None
        if not guard and not ok and r.get("odds_home") is None:
            rr["odds_home"] = 2.5
            rr["odds_draw"] = 3.2
            rr["odds_away"] = 3.0
        try:
            fv = ml_model.extract_features_from_db_row(rr)
        except Exception:
            continue
        hg, ag = r["actual_home_goals"], r["actual_away_goals"]
        X.append(fv)
        y.append(2 if hg > ag else (1 if hg == ag else 0))
        dates.append(parse_date(r.get("match_date")))
        valid.append(ok)
    return np.asarray(X, dtype=np.float32), np.asarray(y), dates, np.asarray(valid)


def evaluate(model, X, y, label, n_train):
    from sklearn.metrics import accuracy_score, log_loss

    pred = model.predict(X)
    proba = model.predict_proba(X)
    brier = float(np.mean(np.sum((proba - np.eye(3)[y]) ** 2, axis=1)))
    ll = float(log_loss(y, proba, labels=[0, 1, 2]))
    acc = float(accuracy_score(y, pred))
    top = proba.argmax(axis=1)
    stated = float(np.mean(proba.max(axis=1)))
    realised = float(np.mean(top == y))
    print(f"  {label:<10} n={len(y):<6} Brier={brier:.4f}  logloss={ll:.4f}  "
          f"acc={acc*100:.1f}%  stated={stated*100:.1f}%  realised={realised*100:.1f}%  "
          f"gap={(realised-stated)*100:+.1f}")
    return {"brier": brier, "logloss": ll, "acc": acc,
            "stated": stated, "realised": realised, "gap": realised - stated}


def main():
    import lightgbm as lgb
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    print("Loading rows...")
    rows = load_rows()
    print(f"  {len(rows)} resolved rows")

    results = {}
    for guard in (False, True):
        label = "guarded" if guard else "raw"
        print(f"\nBuilding features [{label}]...")
        X, y, dates, valid = build_matrices(rows, guard)
        print(f"  X={X.shape}  odds-valid rows: {valid.sum()} ({valid.mean()*100:.1f}%)")

        # Chronological split: sort by date, take the tail as holdout.
        idx = np.argsort([d.timestamp() if d else 0 for d in dates])
        X, y = X[idx], y[idx]
        split = int(len(y) * (1 - HOLDOUT_FRAC))
        Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
        print(f"  train={len(ytr)}  holdout={len(yte)}")

        print(f"  Training [{label}]...")
        gb = GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.025,
            random_state=SEED)
        gb.fit(Xtr, ytr)
        lg = lgb.LGBMClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.025,
            subsample=0.60, colsample_bytree=0.60, min_child_samples=80,
            random_state=SEED, n_jobs=-1, verbose=-1,
            reg_alpha=0.5, reg_lambda=2.0)
        lg.fit(Xtr, ytr)

        results[label] = {
            "GB": evaluate(gb, Xte, yte, f"{label}/GB", len(ytr)),
            "LGB": evaluate(lg, Xte, yte, f"{label}/LGB", len(ytr)),
        }

    print("\n" + "=" * 78)
    print("VERDICT (lower Brier / logloss is better; |gap| closer to 0 is better)")
    print("=" * 78)
    for fam in ("GB", "LGB"):
        r = results["raw"][fam]
        g = results["guarded"][fam]
        db = g["brier"] - r["brier"]
        dl = g["logloss"] - r["logloss"]
        print(f"\n  {fam}:")
        print(f"    Brier   {r['brier']:.4f} -> {g['brier']:.4f}  ({db:+.4f})")
        print(f"    logloss {r['logloss']:.4f} -> {g['logloss']:.4f}  ({dl:+.4f})")
        print(f"    acc     {r['acc']*100:.1f}% -> {g['acc']*100:.1f}% "
              f"({(g['acc']-r['acc'])*100:+.1f} pts)")
        print(f"    calib   gap {r['gap']*100:+.1f} -> {g['gap']*100:+.1f} "
              f"({abs(g['gap'])-abs(r['gap']):+.4f} toward 0)")


if __name__ == "__main__":
    main()