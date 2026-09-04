#!/usr/bin/env python3
"""FAST backtest: re-predict finished matches for A/B validation.
Patches _auto_calibrate_thresholds to no-op and builds a fresh data dict per
match so results are reproducible run-to-run."""
import sys, os, argparse, time
sys.path.insert(0, "/home/stdk/predictor")
os.chdir("/home/stdk/predictor")

from database import init_db, get_db


def check_correct(pick, market, hg, ag):
    if hg is None or ag is None:
        return None
    actual = "Home win" if hg > ag else "Away win" if hg < ag else "Draw"
    if market == "1X2":
        return pick == actual
    elif market == "O/U":
        total = hg + ag
        if "Over" in pick:
            return total > float(pick.split()[-1])
        elif "Under" in pick:
            return total <= float(pick.split()[-1])
    elif market == "BTTS":
        both = hg > 0 and ag > 0
        return (pick == "Yes" and both) or (pick == "No" and not both)
    elif market == "DNB":
        if actual == "Draw":
            return None
        return (pick == "Home" and actual == "Home win") or (pick == "Away" and actual == "Away win")
    elif market == "DC":
        if pick == "1X": return actual in ("Home win", "Draw")
        elif pick == "X2": return actual in ("Away win", "Draw")
        elif pick == "12": return actual in ("Home win", "Away win")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="28/08/2026,27/08/2026,26/08/2026,31/07/2026,30/07/2026,29/07/2026")
    ap.add_argument("--limit", type=int, default=2000)
    args = ap.parse_args()

    import predict
    predict._auto_calibrate_thresholds = lambda: None

    init_db()
    conn = get_db()
    dates = [d.strip() for d in args.date.split(",")]
    qmarks = ",".join("?" * len(dates))
    rows = conn.execute(f"""
        SELECT * FROM matches
        WHERE actual_result IS NOT NULL AND match_date IN ({qmarks})
        ORDER BY id
    """, dates).fetchall()
    col_names = [c["name"] for c in conn.execute("PRAGMA table_info(matches)").fetchall()]
    rows = rows[:args.limit]
    print(f"Re-predicting {len(rows)} matches from {len(dates)} dates (fast mode)...", flush=True)

    total = 0
    old_c = new_c = 0
    by_market = {}
    by_conf = {}
    switches = 0
    examples = []
    t0 = time.time()

    for row_vals in rows:
        row = dict(zip(col_names, row_vals))
        hg = row.get("actual_home_goals")
        ag = row.get("actual_away_goals")
        if hg is None or ag is None:
            continue
        data = {}
        for k, v in row.items():
            if v is not None and k not in ("actual_home_goals", "actual_away_goals", "actual_result", "our_prediction", "our_market", "our_confidence", "method_used", "id"):
                data[k] = v
        for k in ('odds_home', 'odds_draw', 'odds_away', 'odds_over25', 'odds_under25',
                  'forebet_home_pct', 'forebet_draw_pct', 'forebet_away_pct', 'forebet_over25_pct'):
            if data.get(k) is not None:
                try:
                    data[k] = float(data[k])
                except:
                    pass

        op = row.get("our_prediction", "?")
        om = row.get("our_market", "?")
        oc = row.get("our_confidence", "?")

        try:
            pred = predict.analyze_from_data(data, use_ml=True)
        except Exception as e:
            print(f"  ERROR {row['home_team']}: {e}", flush=True)
            continue
        np_ = pred.get("pick", "?")
        nm = pred.get("market", "?")
        nc = pred.get("confidence", "?")

        old_ok = check_correct(op, om, hg, ag)
        new_ok = check_correct(np_, nm, hg, ag)
        if old_ok is None and new_ok is None:
            continue

        total += 1
        if old_ok:
            old_c += 1
        if new_ok:
            new_c += 1

        by_market.setdefault(nm, [0, 0])
        if new_ok is not None:
            by_market[nm][0] += 1
            if new_ok:
                by_market[nm][1] += 1
        by_conf.setdefault(nc, [0, 0])
        if new_ok is not None:
            by_conf[nc][0] += 1
            if new_ok:
                by_conf[nc][1] += 1

        if om != nm:
            switches += 1
        if old_ok != new_ok and len(examples) < 40:
            examples.append(f"  {row['home_team'][:16]:16s} {hg}-{ag} vs {row['away_team'][:16]:16s} || "
                            f"Old {om}({op[:10]}){('✓' if old_ok else '✗' if old_ok is not None else '—')} "
                            f"-> New {nm}({np_[:10]}){('✓' if new_ok else '✗' if new_ok is not None else '—')}")

    dt = time.time() - t0
    print(f"\n{'='*66}")
    print(f"MATCHES: {total}   ({dt:.0f}s, {total/dt:.1f}/s)")
    print(f"OLD: {old_c}/{total} = {old_c/total*100:.1f}%")
    print(f"NEW: {new_c}/{total} = {new_c/total*100:.1f}%")
    print(f"Market switches: {switches}/{total}")
    print(f"\nNEW BY MARKET:")
    for m, (t, c) in sorted(by_market.items()):
        print(f"  {m:6s}: {c}/{t} = {c/t*100:.1f}%")
    print(f"\nNEW BY CONFIDENCE:")
    for m, (t, c) in sorted(by_conf.items()):
        print(f"  {m:14s}: {c}/{t} = {c/t*100:.1f}%")
    print(f"\nCHANGES (first 40):")
    for e in examples:
        print(e)
    conn.close()


if __name__ == "__main__":
    main()