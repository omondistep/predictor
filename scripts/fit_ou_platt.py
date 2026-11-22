#!/usr/bin/env python3
"""Fit the out-of-sample calibration map for the ML Over 2.5 probability.

The ML over-probability is fitted on history it has already seen, so it cannot
be calibrated in-sample. This script trains a fresh model on matches before a
cutoff, scores the ex-ante matches after it, and fits a two-parameter Platt map
logit(p_cal) = a + b * logit(p_raw) on those out-of-sample predictions.

Output: ml_models/ml_predictor/ou_platt.json, read by MLPredictor.predict_from_row.
The map is a bias correction. Re-run it after material retrains.

Usage:  .venv/bin/python scripts/fit_ou_platt.py [--cutoff 2026-09-01]
"""
import argparse
import datetime as dt
import json
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import ml_model as M  # noqa: E402

DB = ROOT / "history.db"
OUT = ROOT / "ml_models" / "ml_predictor" / "ou_platt.json"


def _day(s):
    return M._elo_parse_date(s)


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2026-09-01")
    args = ap.parse_args()
    cutoff = dt.datetime.strptime(args.cutoff, "%Y-%m-%d")

    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM matches WHERE actual_home_goals IS NOT NULL "
        "AND actual_away_goals IS NOT NULL")]
    reviewed = {r["id"] for r in conn.execute("SELECT id FROM matches WHERE reviewed = 1")}
    conn.close()

    elo_pre = M.elo_pre_match_for_rows(rows)  # ratings from every settled result

    def attach(r):
        pv = elo_pre.get(r["id"])
        if pv is not None:
            r["_elo_home_pre"], r["_elo_away_pre"] = pv
        return r

    # Training: the production filter (reviewed, no post-kickoff rows, one
    # row per fixture), restricted to matches before the cutoff.
    train, seen = [], set()
    for r in sorted(rows, key=lambda x: (x.get("created_at") or "", x["id"])):
        if r["id"] not in reviewed or _day(r["match_date"]) >= cutoff:
            continue
        if M._created_after_kickoff(r):
            continue
        k = M._elo_fixture_key(r)
        if k in seen:
            continue
        seen.add(k)
        train.append(attach(dict(r)))

    # Test: ex-ante matches on or after the cutoff, one row per fixture.
    test, seen = [], set()
    for r in sorted(rows, key=lambda x: (x.get("created_at") or "", x["id"])):
        if _day(r["match_date"]) < cutoff or _day(r["match_date"]).year == 1900:
            continue
        if M._created_after_kickoff(r):
            continue
        k = M._elo_fixture_key(r)
        if k in seen:
            continue
        seen.add(k)
        test.append(attach(dict(r)))

    print(f"train={len(train)} test(ex-ante)={len(test)} cutoff={args.cutoff}")

    def xy(rs):
        # Rows whose features cannot be built are skipped, as in production
        # (load_training_data wraps each row in try/except).
        ok = []
        feats = []
        for r in rs:
            try:
                feats.append(M.extract_features_from_db_row(r))
                ok.append(r)
            except Exception:
                continue
        rs = ok
        X = np.vstack(feats).astype(np.float32)
        hs = np.array([r["actual_home_goals"] for r in rs])
        aw = np.array([r["actual_away_goals"] for r in rs])
        y1 = np.where(hs > aw, 2, np.where(hs == aw, 1, 0)).astype(np.int32)
        y2 = ((hs + aw) > 2).astype(np.int32)
        return X, y1, y2

    Xtr, y1tr, y2tr = xy(train)
    Xte, _, y2te = xy(test)
    sel = M.select_features(Xtr, y1tr, y2tr, max_features=50, min_independent=30)
    ml = M.MLPredictor()
    ml.train(Xtr, y1tr, y2tr, sample_weights=None, calibration_method="sigmoid",
             dates=None, feature_indices=sel)
    p_raw = ml.predict_proba_ou(Xte[:, sel])[:, 1]

    lg = _logit(p_raw).reshape(-1, 1)
    cv = np.zeros(len(y2te))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(lg):
        clf = LogisticRegression(C=1e6).fit(lg[tr], y2te[tr])
        cv[te] = clf.predict_proba(lg[te])[:, 1]
    final = LogisticRegression(C=1e6).fit(lg, y2te)
    a = float(final.intercept_[0])
    b = float(final.coef_[0][0])

    report = {
        "a": a, "b": b,
        "fitted_on": {"cutoff": args.cutoff, "train_rows": len(train),
                      "test_rows": len(test), "test_over_rate": round(float(y2te.mean()), 4)},
        "cv_logloss_raw": round(_ll(p_raw, y2te), 4),
        "cv_logloss_platt": round(_ll(cv, y2te), 4),
        "constant_base_logloss": round(_ll(np.full(len(y2te), y2te.mean()), y2te), 4),
        "fitted_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
