#!/usr/bin/env python3
"""Walk-forward validation of external signals (LLM agents, tipsters, etc.).

Fits nothing — purely measures.  For each fold, takes the signal predictions
recorded in signal_log during the fitting window, refreshes signal_accuracy,
and measures out-of-sample accuracy on the test window.

An external signal is only stakeable if its measured hit rate beats its base
rate with statistical significance (same gate as the internal model).  This
validator answers: does the signal's edge survive on matches it was never
fitted on?

Usage:
    .venv/bin/python scripts/validate_signals.py
    .venv/bin/python scripts/validate_signals.py --signal-source "llm-agent-premier-league"
    .venv/bin/python scripts/validate_signals.py --folds 4 --min-test-n 30
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database import (  # noqa: E402
    get_db,
    get_signal_accuracy,
    refresh_signal_accuracy,
    compute_signal_reliability,
)
from predict import detect_league  # noqa: E402

DB = Path(__file__).resolve().parent.parent / "history.db"

MIN_SAMPLE = 25  # gate's min sample floor
MIN_TEST_N = 30  # a signal needs this many test matches to be judged


def load_signals(conn, since=None, until=None, signal_source=None):
    """Load resolved signal predictions from signal_log, ordered by date."""
    where = ["actual_result IS NOT NULL", "correct IS NOT NULL"]
    args = []
    if signal_source:
        where.append("signal_source = ?")
        args.append(signal_source)
    if since:
        where.append("created_at >= ?")
        args.append(since)
    if until:
        where.append("created_at < ?")
        args.append(until)

    rows = conn.execute(
        f"""
        SELECT signal_source, league, market, pick, confidence,
               correct, created_at, match_id
        FROM signal_log
        WHERE {' AND '.join(where)}
        ORDER BY created_at
        """,
        args,
    ).fetchall()
    return [dict(r) for r in rows]


def evaluate_signals(signals):
    """Compute accuracy stats for a list of signal dicts.

    Returns dict keyed by (signal_source, league, market, pick, confidence)
    with total, correct, accuracy, base_rate, edge_pts.
    """
    from database import pick_base_rate, get_base_rates

    groups = defaultdict(lambda: {"total": 0, "correct": 0})
    for s in signals:
        key = (s["signal_source"], s["league"], s["market"], s["pick"], s["confidence"])
        groups[key]["total"] += 1
        groups[key]["correct"] += 1 if s["correct"] else 0

    results = {}
    for key, v in groups.items():
        signal_source, league, market, pick, confidence = key
        total = v["total"]
        correct = v["correct"]
        acc = correct / total if total else 0.0
        base = pick_base_rate(get_base_rates(league or ""), market, pick) or 0.40
        edge = (acc - base) * 100
        results[key] = {
            "signal_source": signal_source,
            "league": league,
            "market": market,
            "pick": pick,
            "confidence": confidence,
            "total": total,
            "correct": correct,
            "accuracy": acc,
            "base_rate": base,
            "edge_pts": edge,
        }
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--signal-source", default=None,
                    help="validate only this signal source (default: all)")
    ap.add_argument("--folds", type=int, default=4,
                    help="number of sequential walk-forward folds (default 4)")
    ap.add_argument("--min-test-n", type=int, default=MIN_TEST_N,
                    help="minimum test matches for a bucket to be judged")
    args = ap.parse_args()

    conn = get_db()
    all_signals = load_signals(conn, signal_source=args.signal_source)
    conn.close()

    total = len(all_signals)
    print(f"resolved signal predictions: {total:,}")
    if total < 100:
        print("Not enough resolved signal history to validate.")
        print("  Record some signals first: run predictions with data['_external_signals'] set,")
        print("  then run 'pr results' to resolve them against actual results.")
        return 1

    # Group by signal source
    sources = defaultdict(list)
    for s in all_signals:
        sources[s["signal_source"]].append(s)

    for source, sigs in sources.items():
        print(f"\n{'='*66}")
        print(f"Signal source: {source}  ({len(sigs):,} resolved predictions)")
        print(f"{'='*66}")

        dates = [s["created_at"] for s in sigs]
        dates.sort()
        step = len(dates) // (args.folds + 1)
        if step < 10:
            print(f"  Too few samples for {args.folds} folds ({len(sigs)} predictions).")
            continue

        all_oos_edges = []
        all_insample_edges = []
        buckets_judged = 0
        buckets_kept = 0

        for f in range(args.folds):
            cut = dates[(f + 1) * step]
            fit_sigs = [s for s in sigs if s["created_at"] < cut]
            test_sigs = [s for s in sigs if s["created_at"] >= cut]
            if len(fit_sigs) < 10 or len(test_sigs) < 10:
                continue

            fit_stats = evaluate_signals(fit_sigs)
            test_stats = evaluate_signals(test_sigs)

            print(f"\n  --- fold {f+1}: fit {len(fit_sigs)}, test {len(test_sigs)} ---")

            # For each bucket approved in-fit, check out-of-sample
            for key, fit_v in sorted(fit_stats.items(), key=lambda kv: -kv[1]["edge_pts"]):
                n_fit = fit_v["total"]
                if n_fit < MIN_SAMPLE:
                    continue
                test_v = test_stats.get(key)
                if not test_v or test_v["total"] < args.min_test_n:
                    continue

                buckets_judged += 1
                kept = test_v["edge_pts"] > 0
                if kept:
                    buckets_kept += 1

                all_insample_edges.append(fit_v["edge_pts"])
                all_oos_edges.append(test_v["edge_pts"])

                label = f"{key[1]:24s}|{key[3]:12s}|{key[4]}"
                print(f"    {label:<44} fit_edge={fit_v['edge_pts']:+7.1f} "
                      f"test_n={test_v['total']:4d} test_acc={test_v['accuracy']*100:5.1f}% "
                      f"test_base={test_v['base_rate']*100:5.1f}% "
                      f"test_edge={test_v['edge_pts']:+7.1f}  "
                      f"{'holds' if kept else 'EDGE GONE'}")

        if not all_oos_edges:
            print(f"\n  No buckets had enough out-of-sample samples to judge.")
            continue

        mi = sum(all_insample_edges) / len(all_insample_edges)
        mo = sum(all_oos_edges) / len(all_oos_edges)
        decay = (mo - mi) / mi * 100 if mi else 0.0

        print(f"\n  buckets judged : {buckets_judged}")
        print(f"  kept edge      : {buckets_kept}/{buckets_judged}")
        print(f"  mean edge in-sample : {mi:+.1f} pts")
        print(f"  mean edge held out  : {mo:+.1f} pts")
        print(f"  edge retained  : {decay:+.0f}%")

        if mo <= 0:
            print(f"  VERDICT: {source} does not generalise — edges don't survive OOS.")
        elif mo < mi * 0.5:
            print(f"  VERDICT: {source} edges decay sharply OOS — treat as optimistic.")
        else:
            print(f"  VERDICT: {source} generalises — edges survive OOS measurement.")

    # Overall reliability summary (all time)
    print(f"\n{'='*66}")
    print("ALL-TIME RELIABILITY SUMMARY")
    print(f"{'='*66}")
    for source in sources:
        rel = compute_signal_reliability(source)
        print(f"  {source:36s} "
              f"REL={rel['reliable']:3d} WATCH={rel['watch']:3d} HELD={rel['held']:3d} "
              f"(total buckets={rel['total_buckets']})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
