#!/usr/bin/env python3
"""Walk-forward validation of the per-league statistical experts.

Fits experts only on the past, then measures whether the expert-adjusted
probabilities improve pick accuracy on held-out future matches.

The expert's claim: in league L, scale the model's edge over the base rate by
``effective_mult = 1 + confidence * (multiplier - 1)`` where ``multiplier`` is
the shrinkage-weighted ratio of my accuracy to Forebet's accuracy in that
league.  A specialist that only works in-sample is fitting noise -- this
validator checks whether the edge survives on matches it was never fitted on.

Usage:
    .venv/bin/python scripts/validate_league_experts.py
    .venv/bin/python scripts/validate_league_experts.py --folds 4 --min-resolved 200
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predict import detect_league  # noqa: E402

DB = Path(__file__).resolve().parent.parent / "history.db"

CAL_PRIOR_WEIGHT = 100  # shrinkage: pseudo-observations of "multiplier = 1.0"
MIN_TEST_N = 30  # a league needs this many test matches to be judged


def result(h, a):
    if h is None or a is None:
        return None
    if h > a:
        return "Home win"
    if h == a:
        return "Draw"
    return "Away win"


def load_rows(conn, since=None, until=None):
    """Load resolved 1X2 matches with component probabilities, ordered by date."""
    where = [
        "actual_home_goals IS NOT NULL",
        "actual_away_goals IS NOT NULL",
        "our_prediction IN ('Home win','Draw','Away win')",
        "ml_prob_home IS NOT NULL",
        "poisson_prob_home IS NOT NULL",
    ]
    args = []
    if since:
        where.append("created_at >= ?")
        args.append(since)
    if until:
        where.append("created_at < ?")
        args.append(until)
    rows = conn.execute(
        f"""
        SELECT our_prediction, created_at, ml_prob_home, ml_prob_draw,
               ml_prob_away, poisson_prob_home, poisson_prob_draw,
               poisson_prob_away, actual_home_goals, actual_away_goals,
               league
          FROM matches WHERE {' AND '.join(where)}
        ORDER BY created_at
        """,
        args,
    ).fetchall()
    return [dict(r) for r in rows]


def fit_experts(rows, min_resolved):
    """Build per-league experts from fit rows (past only)."""
    groups = defaultdict(list)
    for r in rows:
        key = detect_league(r["league"] or "")
        groups[key].append(r)

    experts = {}
    for key, lr in groups.items():
        n = len(lr)
        if n < min_resolved:
            continue

        my_hits = my_tot = fb_hits = fb_tot = 0
        base_home = base_draw = base_away = 0
        for r in lr:
            act = result(r["actual_home_goals"], r["actual_away_goals"])
            if act is None:
                continue
            if act == "Home win":
                base_home += 1
            elif act == "Draw":
                base_draw += 1
            else:
                base_away += 1
            if r["our_prediction"] == act:
                my_hits += 1
            my_tot += 1
            # Forebet prediction not stored as probability; infer from
            # forebet_prob columns (absent) — use our_prediction vs actual
            # as the "my" baseline and compare against a naive Forebet proxy:
            # Forebet's pick is its highest probability. We don't store it,
            # so use the home/draw/away prior from base rates as the FB proxy.
            # (This is conservative: we validate the multiplier concept, not
            # exact Forebet comparison.)

        n_valid = base_home + base_draw + base_away
        if n_valid == 0:
            continue
        base = {
            "home": base_home / n_valid,
            "draw": base_draw / n_valid,
            "away": base_away / n_valid,
        }
        my_acc = my_hits / my_tot if my_tot else 0.0

        # Blind-pick baseline for this league: always pick the most common
        # outcome.  The repo's rule: the model must beat picking blind before
        # it gets any edge amplification.  multiplier = my_acc / base_rate_fav.
        base_rate_fav = max(base.values())
        rel_skill = my_acc / base_rate_fav if base_rate_fav > 0 else 1.0
        w = n / (n + CAL_PRIOR_WEIGHT)
        multiplier = w * rel_skill + (1.0 - w) * 1.0
        confidence = w

        experts[key] = {
            "multiplier": multiplier,
            "confidence": confidence,
            "base": base,
            "n": n,
            "my_acc": my_acc,
            "base_rate_fav": base_rate_fav,
        }
    return experts


def apply_expert(probs, base, expert):
    """Apply expert calibration to a probability dict, return adjusted dict."""
    mult = expert["multiplier"]
    conf = expert["confidence"]
    effective_mult = 1.0 + conf * (mult - 1.0)
    if abs(effective_mult - 1.0) < 1e-6:
        return probs
    adj = {}
    for outcome in ("Home win", "Draw", "Away win"):
        key = {"Home win": "home", "Draw": "draw", "Away win": "away"}[outcome]
        b = base[key]
        adj[outcome] = max(0.0, min(1.0, b + (probs[outcome] - b) * effective_mult))
    total = sum(adj.values())
    if total <= 0:
        return probs
    return {k: v / total for k, v in adj.items()}


def blend_probs(r):
    """Blend ML and Poisson probabilities (equal weight) as the global baseline."""
    return {
        "Home win": (r["ml_prob_home"] + r["poisson_prob_home"]) / 2,
        "Draw": (r["ml_prob_draw"] + r["poisson_prob_draw"]) / 2,
        "Away win": (r["ml_prob_away"] + r["poisson_prob_away"]) / 2,
    }


def evaluate(test_rows, experts):
    """Compare expert-adjusted vs global accuracy on test rows.

    Returns (global_correct, global_total, expert_correct, expert_total,
    per_league_stats).
    """
    g_correct = g_total = e_correct = e_total = 0
    per_league = defaultdict(lambda: {"g_correct": 0, "g_total": 0,
                                       "e_correct": 0, "e_total": 0})

    for r in test_rows:
        act = result(r["actual_home_goals"], r["actual_away_goals"])
        if act is None:
            continue
        key = detect_league(r["league"] or "")
        probs = blend_probs(r)
        base = experts[key]["base"] if key in experts else None

        # Global pick: max blended probability
        global_pick = max(probs, key=probs.get)

        if key in experts:
            adjusted = apply_expert(probs, base, experts[key])
            expert_pick = max(adjusted, key=adjusted.get)
            e_correct += 1 if expert_pick == act else 0
            e_total += 1
            per_league[key]["e_correct"] += 1 if expert_pick == act else 0
            per_league[key]["e_total"] += 1
        else:
            # No expert: global pick is the only option
            e_correct += 1 if global_pick == act else 0
            e_total += 1
            per_league[key]["e_correct"] += 1 if global_pick == act else 0
            per_league[key]["e_total"] += 1

        g_correct += 1 if global_pick == act else 0
        g_total += 1
        per_league[key]["g_correct"] += 1 if global_pick == act else 0
        per_league[key]["g_total"] += 1

    return g_correct, g_total, e_correct, e_total, per_league


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=4,
                    help="number of sequential walk-forward folds")
    ap.add_argument("--min-resolved", type=int, default=200,
                    help="minimum resolved matches for a league expert")
    ap.add_argument("--min-test-n", type=int, default=MIN_TEST_N,
                    help="minimum test matches for a league to be judged")
    args = ap.parse_args()

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    all_rows = load_rows(conn)
    conn.close()

    total = len(all_rows)
    print(f"resolved 1X2 matches with component probabilities: {total:,}")
    if total < 400:
        print("Not enough history to validate.")
        return 1

    dates = [r["created_at"] for r in all_rows]
    step = len(dates) // (args.folds + 1)
    if step < 100:
        print(f"History too short for {args.folds} folds ({total} matches).")
        return 1

    print(f"Expert min resolved: {args.min_resolved}, min test n: {args.min_test_n}\n")

    all_g_correct = all_g_total = all_e_correct = all_e_total = 0
    fold_results = []

    for f in range(args.folds):
        cut = dates[(f + 1) * step]
        fit_rows = [r for r in all_rows if r["created_at"] < cut]
        test_rows = [r for r in all_rows if r["created_at"] >= cut]
        if len(fit_rows) < 200 or len(test_rows) < 200:
            continue

        experts = fit_experts(fit_rows, args.min_resolved)
        n_experts = len(experts)

        g_c, g_t, e_c, e_t, per_league = evaluate(test_rows, experts)

        # Only judge leagues with enough test samples
        judged_leagues = {
            k: v for k, v in per_league.items() if v["e_total"] >= args.min_test_n
        }
        expert_leagues_tested = [
            k for k in judged_leagues if k in experts
        ]

        g_acc = g_c / g_t if g_t else 0
        e_acc = e_c / e_t if e_t else 0
        diff = (e_acc - g_acc) * 100

        fold_results.append({
            "fold": f + 1,
            "cut": cut[:10],
            "fit_n": len(fit_rows),
            "test_n": len(test_rows),
            "n_experts": n_experts,
            "expert_leagues_tested": len(expert_leagues_tested),
            "global_acc": g_acc,
            "expert_acc": e_acc,
            "diff_pts": diff,
        })

        print(f"--- fold {f+1}: fit {len(fit_rows):,}, test {len(test_rows):,}, "
              f"{n_experts} experts ---")
        print(f"  global accuracy : {g_acc*100:5.1f}% ({g_c}/{g_t})")
        print(f"  expert accuracy : {e_acc*100:5.1f}% ({e_c}/{e_t})")
        print(f"  difference      : {diff:+5.1f} pts  "
              f"{'expert wins' if diff > 0 else 'global wins'}")
        print(f"  leagues tested  : {len(expert_leagues_tested)} with >= {args.min_test_n} test matches")

        # Per-league breakdown for expert leagues
        for k in sorted(expert_leagues_tested,
                        key=lambda kk: -per_league[kk]["e_total"]):
            v = per_league[k]
            ga = v["g_correct"] / v["g_total"] if v["g_total"] else 0
            ea = v["e_correct"] / v["e_total"] if v["e_total"] else 0
            d = (ea - ga) * 100
            exp = experts[k]
            print(f"    {k:28s} n={v['e_total']:4d} "
                  f"global={ga*100:5.1f}% expert={ea*100:5.1f}% "
                  f"diff={d:+5.1f} mult={exp['multiplier']:.2f}")

        all_g_correct += g_c
        all_g_total += g_t
        all_e_correct += e_c
        all_e_total += e_t

    print("\n" + "=" * 66)
    g_acc = all_g_correct / all_g_total if all_g_total else 0
    e_acc = all_e_correct / all_e_total if all_e_total else 0
    diff = (e_acc - g_acc) * 100
    print(f"total test matches : {all_g_total:,}")
    print(f"global accuracy    : {g_acc*100:5.1f}% ({all_g_correct}/{all_g_total})")
    print(f"expert accuracy    : {e_acc*100:5.1f}% ({all_e_correct}/{all_e_total})")
    print(f"difference         : {diff:+5.1f} pts")
    print("=" * 66)

    # Per-fold summary
    diffs = [fr["diff_pts"] for fr in fold_results]
    folds_won = sum(1 for d in diffs if d > 0)
    print(f"\nfolds won by expert : {folds_won}/{len(fold_results)}")
    print(f"mean diff           : {sum(diffs)/len(diffs):+5.1f} pts" if diffs else "")

    if diff > 0:
        print("\nVERDICT: league experts improve accuracy out of sample — deploy.")
        return 0
    print("\nVERDICT: league experts do not improve accuracy out of sample — "
          "keep global model only.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
