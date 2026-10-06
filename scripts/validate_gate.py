#!/usr/bin/env python3
"""Walk-forward validation of the reliability gate.

The gate in ``scripts/audit_reliability.py`` answers "does this pick beat what
you'd get by picking blind?".  Fitting that gate on all of history and then
reporting its own numbers proves nothing -- every gate looks good on the data
that produced it.  This script is the check that matters:

  1. cut history in half at a date
  2. build the gate on the first half only
  3. measure what those approved buckets actually did on the second half

A gate that only works in-sample is fitting noise, and this is where that
shows up.  It is deliberately a separate tool rather than part of the ledger
build: a validation number that gets recomputed every time you tweak the gate
stops being a check and becomes part of the thing being checked.

Usage:
    .venv/bin/python scripts/validate_gate.py
    .venv/bin/python scripts/validate_gate.py --folds 4
    .venv/bin/python scripts/validate_gate.py --min-test-n 30
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import audit_reliability as A  # noqa: E402


def shrink(hits: int, n: int, base: float) -> float:
    return (hits + A.K * base) / (n + A.K) if n else base


def load(conn: sqlite3.Connection, since=None, until=None):
    where = ["actual_home_goals IS NOT NULL", "our_prediction IS NOT NULL"]
    args: list = []
    if since:
        where.append("created_at >= ?")
        args.append(since)
    if until:
        where.append("created_at < ?")
        args.append(until)
    rows = conn.execute(
        # Column names must stay unaliased: dedupe_fixtures() reads
        # forebet_url / home_team / away_team / match_date / created_at by name,
        # and an alias makes every row resolve to the same empty fixture key —
        # which silently collapses the whole population to one match.
        f"SELECT our_prediction, our_confidence, our_market, created_at,"
        f"       forebet_url, home_team, away_team, match_date,"
        f"       actual_home_goals hg, actual_away_goals ag,"
        f"       ht_home_goals, ht_away_goals"
        f"  FROM matches WHERE {' AND '.join(where)}", args).fetchall()
    # Collapse duplicate fixtures exactly as the ledger's measure() does, so the
    # validator never judges a gate built from a different population.
    return A.dedupe_fixtures(rows)


def base_rates(rows):
    """Unconditional selection frequencies, per market, over this population.

    Market-aware for the same reason the ledger is: a first-half Draw must be
    measured against first-half results, or the validator would endorse the very
    pooling the ledger was fixed to remove.
    """
    ft = [(r["hg"], r["ag"]) for r in rows]
    ht = [(r["ht_home_goals"], r["ht_away_goals"]) for r in rows
          if r["ht_home_goals"] is not None and r["ht_away_goals"] is not None]
    decisive = [(h, a) for h, a in ft if h != a]

    out = {}
    for market, sels in A.MARKET_SELECTIONS.items():
        if market == "?":
            sel_list = [s for s in A.SELECTIONS if s not in ("Home", "Away")]
            pop = ft
        elif A.period_of(market) == A.HT:
            sel_list, pop = sels, ht
        elif market == "DNB":
            sel_list, pop = sels, decisive
        else:
            sel_list, pop = sels, ft
        n = len(pop)
        for sel in sel_list:
            hits = sum(1 for h, a in pop if A.resolve_selection(sel, h, a))
            out[(market, sel)] = hits / n if n else A.DEFAULT_BASE
    return out, len(ft)


def buckets_for(rows):
    acc = defaultdict(lambda: {"n": 0, "hit": 0})
    for r in rows:
        market = A.market_of(r)
        hg, ag = A.scoreline_for(r, market)
        if hg is None or ag is None:
            continue  # no half-time score for a half-time market
        hit = A.resolve_selection(r["our_prediction"], hg, ag)
        if hit is None:
            continue
        b = acc[(market, r["our_prediction"], r["our_confidence"] or "Unknown")]
        b["n"] += 1
        b["hit"] += 1 if hit else 0
    return acc


def approve(acc, base, min_samples, min_edge, use_wilson=True):
    """Buckets the gate would publish, given a population to fit on."""
    out = {}
    for (market, pred, conf), b in acc.items():
        if b["n"] < min_samples:
            continue
        b_rate = base.get((market, pred), A.DEFAULT_BASE)
        edge = shrink(b["hit"], b["n"], b_rate) - b_rate
        if edge * 100 < min_edge:
            continue
        if use_wilson and A.wilson_lower_bound(b["hit"], b["n"]) <= b_rate:
            continue
        out[(market, pred, conf)] = edge
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="history.db")
    ap.add_argument("--folds", type=int, default=2,
                    help="number of sequential walk-forward folds (default 2: fit "
                         "on the first half, then refit on the first two thirds "
                         "— each fold is fitted only on its own past)")
    ap.add_argument("--min-samples", type=int, default=A.MIN_SAMPLE)
    ap.add_argument("--min-edge", type=float, default=2.0)
    ap.add_argument("--min-test-n", type=int, default=30,
                    help="a bucket needs this many second-half samples to be judged")
    ap.add_argument("--no-wilson", action="store_true",
                    help="skip the significance floor (shows what it is buying)")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    total = conn.execute("""
        SELECT COUNT(*) FROM matches
        WHERE actual_home_goals IS NOT NULL AND our_prediction IS NOT NULL
    """).fetchone()[0]
    if total < 400:
        print(f"Not enough resolved history to validate ({total} matches).")
        return 1

    dates = [r[0] for r in conn.execute("""
        SELECT created_at FROM matches
        WHERE actual_home_goals IS NOT NULL AND our_prediction IS NOT NULL
        ORDER BY created_at
    """)]
    step = len(dates) // (args.folds + 1)
    if step < 200:
        print(f"History too short for {args.folds} folds ({total} matches).")
        return 1

    wilson = not args.no_wilson
    print(f"Walk-forward validation of the reliability gate — {total:,} resolved matches")
    print(f"Significance floor: {'ON (Wilson one-sided 95%)' if wilson else 'OFF'}\n")

    all_insample, all_oos, kept, tested = [], [], 0, 0

    for f in range(args.folds):
        cut = dates[(f + 1) * step]
        fit_rows = load(conn, until=cut)
        test_rows = load(conn, since=cut)
        if len(fit_rows) < 200 or len(test_rows) < 200:
            continue

        # Fit on the past only. Base rates come from the fitting population so
        # no future information leaks into the yardstick.
        base_fit, n_fit = base_rates(fit_rows)
        approved = approve(buckets_for(fit_rows), base_fit,
                           args.min_samples, args.min_edge, wilson)
        if not approved:
            print(f"fold {f+1}: nothing approved on the fitting half — skipping")
            continue

        base_test, n_test = base_rates(test_rows)
        test = buckets_for(test_rows)

        print(f"--- fold {f+1}: fit on {n_fit:,} matches (< {cut[:10]}), "
              f"test on {n_test:,} ---")
        print(f"{'bucket':<40}{'fit edge':>9}{'test n':>8}{'test hit':>10}"
              f"{'test base':>10}{'test edge':>10}  verdict")
        fold_i, fold_o = [], []
        for key in sorted(approved, key=lambda k: -approved[k]):
            market, pred, conf = key
            t = test.get(key)
            if not t or t["n"] < args.min_test_n:
                continue
            b_rate = base_test.get((market, pred), A.DEFAULT_BASE)
            edge = shrink(t["hit"], t["n"], b_rate) - b_rate
            ok = edge * 100 >= args.min_edge
            fold_i.append(approved[key])
            fold_o.append(edge)
            tested += 1
            kept += 1 if ok else 0
            label = f"{market}|{pred}|{conf}"
            print(f"{label:<40}{approved[key]*100:+8.1f}{t['n']:>8}"
                  f"{t['hit']/t['n']*100:9.1f}%{b_rate*100:9.1f}%"
                  f"{edge*100:+9.1f}  {'holds' if ok else 'EDGE GONE'}")
        untested = len(approved) - len(fold_o)
        print(f"  {len(approved)} approved, {len(fold_o)} testable, "
              f"{untested} below --min-test-n\n")
        all_insample += fold_i
        all_oos += fold_o

    if not all_oos:
        print("No bucket had enough second-half samples to judge. Widen the window "
              "or lower --min-test-n.")
        return 1

    mi = sum(all_insample) / len(all_insample) * 100
    mo = sum(all_oos) / len(all_oos) * 100
    print("=" * 66)
    print(f"buckets judged      : {tested}")
    print(f"kept their edge     : {kept}/{tested}")
    print(f"mean edge in-sample : {mi:+.1f} pts")
    print(f"mean edge held out  : {mo:+.1f} pts")
    decay = (mo - mi) / mi * 100 if mi else 0.0
    print(f"edge retained       : {decay:+.0f}%")
    print("=" * 66)

    if mo <= 0:
        print("\nVERDICT: the gate does not generalise. Its edges do not survive "
              "being measured on unseen matches.")
        return 2
    if mo < mi * 0.5:
        print("\nVERDICT: edges decay sharply out of sample. Treat the gate as "
              "optimistic and keep the significance floor.")
        return 2
    print("\nVERDICT: the gate generalises. Its edges survive being measured on "
          "matches it was not fitted on.")
    return 0


if __name__ == "__main__":
    sys.exit(main())