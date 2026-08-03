#!/usr/bin/env python3
"""Backtest model accuracy on a sample of past (settled) matches.

Commands:
  scrape   Re-scrape a stratified sample of settled matches -> JSONL cache
  eval     Run THIS codebase's analyzer over the cache -> CSV + quick stats
  report   Aggregate one or more eval CSVs into a comparison table

The eval pass is codebase-relative: run it from the checkout whose logic you
want to test (e.g. a git worktree at HEAD for the old model, or the working
tree for the new model). Optional overrides isolate the ML/calibration inputs:

  ML_MODELS_DIR=<dir>   point at a different set of ML joblibs (env var)
  --calibration <path>  use a different calibration_params.json

Usage examples:
  scripts/backtest_models.py scrape data/backtest_sample.jsonl
  scripts/backtest_models.py eval data/backtest_sample.jsonl /tmp/bt_old.csv
  ML_MODELS_DIR=/tmp/old_ml/ml_models scripts/backtest_models.py eval \
      data/backtest_sample.jsonl /tmp/bt_new_logic.csv \
      --calibration /tmp/old_model/calibration_params.json
  scripts/backtest_models.py eval data/backtest_sample.jsonl /tmp/bt_new_full.csv
  scripts/backtest_models.py report \
      --csv old=/tmp/bt_old.csv --csv new_logic=/tmp/bt_new_logic.csv \
      --csv new_full=/tmp/bt_new_full.csv
"""

import argparse
import csv
import json
import random
import re
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONF_BUCKETS = ["Near Certain", "High", "Medium-High", "Medium"]
MARKETS = ["1X2", "O/U", "BTTS", "DC", "DNB"]


def actual_result(hg, ag):
    if hg is None or ag is None:
        return None
    if hg > ag:
        return "Home win"
    if hg < ag:
        return "Away win"
    return "Draw"


def score_pick(market, pick, hg, ag):
    """Return 'hit' | 'miss' | 'void' | '' (unscoreable)."""
    if hg is None or ag is None or not pick:
        return ""
    market = (market or "").upper()
    if market == "1X2":
        return "hit" if pick == actual_result(hg, ag) else "miss"
    if market == "O/U":
        m = re.match(r"^(Over|Under)\s+([0-9]+(?:\.[0-9]+)?)", pick)
        if not m:
            return ""
        side, thresh = m.group(1), float(m.group(2))
        total = hg + ag
        if side == "Over":
            return "hit" if total > thresh else "miss"
        return "hit" if total < thresh else "miss"
    if market == "BTTS":
        both = hg > 0 and ag > 0
        if pick == "Yes":
            return "hit" if both else "miss"
        if pick == "No":
            return "hit" if not both else "miss"
        return ""
    if market == "DC":
        if pick == "1X":
            return "hit" if hg >= ag else "miss"
        if pick == "X2":
            return "hit" if hg <= ag else "miss"
        if pick == "12":
            return "hit" if hg != ag else "miss"
        return ""
    if market == "DNB":
        if hg == ag:
            return "void"
        if pick == "Home":
            return "hit" if hg > ag else "miss"
        if pick == "Away":
            return "hit" if ag > hg else "miss"
        return ""
    return ""


# ---------------------------------------------------------------------------
# scrape
# ---------------------------------------------------------------------------

def cmd_scrape(args):
    import cloudscraper
    from forebet_scraper import scrape_url

    if not args.no_shared_session:
        # Reuse one cloudscraper session for all fetches (forebet_scraper
        # creates a fresh one per call otherwise).
        _shared = cloudscraper.create_scraper()
        import forebet_scraper as fs
        fs.cloudscraper.create_scraper = lambda *a, **k: _shared

    conn = sqlite3.connect(str(ROOT / "history.db"))
    rows = conn.execute(
        "SELECT id, forebet_url, our_market, actual_home_goals, actual_away_goals, "
        "match_date FROM matches WHERE actual_home_goals IS NOT NULL "
        "AND forebet_url LIKE '%/football/matches/%' AND our_market IS NOT NULL "
        "ORDER BY id"
    ).fetchall()
    non_ou = [r for r in rows if (r[2] or "").upper() != "O/U"]
    ou = [r for r in rows if (r[2] or "").upper() == "O/U"]

    rng = random.Random(args.seed)
    rng.shuffle(non_ou)
    rng.shuffle(ou)
    sample = non_ou[: args.max_non_ou] + ou[: args.n_ou]
    print(f"Pool: non-O/U {len(non_ou)}, O/U {len(ou)} -> sample {len(sample)}",
          file=sys.stderr)

    out = []
    n_fail = 0
    t0 = time.time()
    for i, (mid, url, market, hg, ag, date) in enumerate(sample, 1):
        data = None
        for attempt in range(args.retries + 1):
            try:
                data = scrape_url(url)
                break
            except Exception as e:
                if attempt < args.retries:
                    time.sleep(2)
                else:
                    sys.stderr.write(f"[{i}/{len(sample)}] FAIL {url}: {e}\n")
                    n_fail += 1
        if not data or not data.get("home_team"):
            continue
        out.append({
            "id": mid, "url": url, "hg": hg, "ag": ag,
            "stored_market": market, "date": date, "data": data,
        })
        if i % 10 == 0:
            el = time.time() - t0
            sys.stderr.write(
                f"[{i}/{len(sample)}] ok={len(out)} fail={n_fail} "
                f"elapsed={el:.0f}s avg={(el / i):.1f}s\n")
            sys.stderr.flush()
        time.sleep(args.delay)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    print(f"Scraped {len(out)}/{len(sample)} matches -> {args.out}")
    print(f"  stored-market mix: "
          f"{dict((k, sum(1 for r in out if (r['stored_market'] or '') == k)) for k in ['1X2','BTTS','DC','DNB','O/U'])}")


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

def cmd_eval(args):
    if args.calibration:
        import calibration_learner
        calibration_learner.params_path = Path(args.calibration)

    from predict import analyze_from_data

    recs = [json.loads(line) for line in open(args.sample) if line.strip()]
    rows = []
    n_err = 0
    for rec in recs:
        row = {
            "id": rec.get("id"), "url": rec.get("url"),
            "league": rec["data"].get("league", ""),
            "home_team": rec["data"].get("home_team", ""),
            "away_team": rec["data"].get("away_team", ""),
            "market": "", "pick": "", "confidence": "",
            "hg": rec.get("hg"), "ag": rec.get("ag"),
            "score": "", "error": "",
        }
        try:
            pred = analyze_from_data(rec["data"], use_ml=not args.classic)
            row["market"] = pred.get("market", "") or ""
            row["pick"] = pred.get("pick", "") or ""
            row["confidence"] = pred.get("confidence", "") or ""
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            n_err += 1
        row["score"] = score_pick(row["market"], row["pick"], row["hg"], row["ag"])
        rows.append(row)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "id", "url", "league", "home_team", "away_team", "market",
            "pick", "confidence", "hg", "ag", "score", "error"])
        w.writeheader()
        w.writerows(rows)

    print(f"Evaluated {len(rows)} matches (errors={n_err}) -> {args.out}")
    _print_eval_summary(rows)


def _print_eval_summary(rows):
    scored = [r for r in rows if r["score"] in ("hit", "miss")]
    hits = sum(1 for r in scored if r["score"] == "hit")
    if scored:
        print(f"  OVERALL accuracy: {hits}/{len(scored)} = {hits / len(scored):.1%}")
    for mk in MARKETS:
        m = [r for r in scored if (r["market"] or "").upper() == mk]
        if not m:
            continue
        h = sum(1 for r in m if r["score"] == "hit")
        print(f"  {mk:5s}: {h}/{len(m)} = {h / len(m):.1%}")
    for cb in CONF_BUCKETS:
        m = [r for r in scored if r["confidence"] == cb]
        if not m:
            continue
        h = sum(1 for r in m if r["score"] == "hit")
        print(f"  conf {cb:12s}: {h}/{len(m)} = {h / len(m):.1%}")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _acc(pairs):
    sc = [p for p in pairs if p[1] in ("hit", "miss")]
    if not sc:
        return None, 0
    hits = sum(1 for p in sc if p[1] == "hit")
    return hits / len(sc), len(sc)


def _fmt(acc, n):
    return "  -  " if acc is None else f"{acc:.1%} ({n})"


def cmd_report(args):
    data = {}
    for spec in args.csv:
        label, _, path = spec.partition("=")
        path = path or label
        label = label if _ != "" else Path(path).stem
        with open(path, newline="") as f:
            data[label] = list(csv.DictReader(f))

    labels = list(data.keys())

    def scored(rows):
        return [(r, r["score"]) for r in rows if r["score"] in ("hit", "miss")]

    print("=" * 70)
    print("ACCURACY COMPARISON")
    print("=" * 70)
    print(f"{'':16s}" + "".join(f"{l:>18s}" for l in labels))
    print(f"{'overall':16s}" + "".join(
        f"{_fmt(*_acc(scored(data[l]))):>18s}" for l in labels))
    for mk in MARKETS:
        row = []
        for l in labels:
            pairs = [(r, r["score"]) for r in data[l]
                     if (r["market"] or "").upper() == mk and r["score"] in ("hit", "miss")]
            row.append(_fmt(*_acc(pairs)))
        print(f"{mk:16s}" + "".join(f"{v:>18s}" for v in row))
    for cb in CONF_BUCKETS:
        row = []
        for l in labels:
            pairs = [(r, r["score"]) for r in data[l]
                     if r["confidence"] == cb and r["score"] in ("hit", "miss")]
            row.append(_fmt(*_acc(pairs)))
        print(f"conf {cb:10s}" + "".join(f"{v:>18s}" for v in row))

    print()
    print("MARKET DISTRIBUTION (primary pick)")
    print(f"{'':16s}" + "".join(f"{l:>18s}" for l in labels))
    for mk in MARKETS:
        print(f"{mk:16s}" + "".join(
            f"{sum(1 for r in data[l] if (r['market'] or '').upper() == mk):>18d}"
            for l in labels))

    print()
    print("O/U OVER vs UNDER")
    for l in labels:
        rows = data[l]
        ov = [r for r in rows if r["market"] == "O/U" and str(r["pick"]).startswith("Over")]
        un = [r for r in rows if r["market"] == "O/U" and str(r["pick"]).startswith("Under")]
        ovn = len([r for r in ov if r["score"] in ("hit", "miss")])
        unh = len([r for r in un if r["score"] in ("hit", "miss")])
        ova = sum(1 for r in ov if r["score"] == "hit") / ovn if ovn else None
        una = sum(1 for r in un if r["score"] == "hit") / unh if unh else None
        u35 = sum(1 for r in un if r["pick"] == "Under 3.5")
        print(f"  {l:14s} Over n={len(ov):4d} acc={_fmt(ova, ovn)} | "
              f"Under n={len(un):4d} acc={_fmt(una, unh)} | Under-3.5 picks: {u35}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    sys.path.insert(0, str(ROOT))
    p = argparse.ArgumentParser(prog="backtest_models")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scrape", help="Re-scrape sample of settled matches")
    s.add_argument("out", type=Path)
    s.add_argument("--n-ou", type=int, default=200, help="random O/U matches to add")
    s.add_argument("--max-non-ou", type=int, default=10 ** 9,
                   help="cap on non-O/U matches")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--delay", type=float, default=0.3, help="seconds between fetches")
    s.add_argument("--retries", type=int, default=1)
    s.add_argument("--no-shared-session", action="store_true")
    s.set_defaults(fn=cmd_scrape)

    e = sub.add_parser("eval", help="Run this codebase's analyzer over a sample")
    e.add_argument("sample", type=Path)
    e.add_argument("out", type=Path)
    e.add_argument("--calibration", default=None,
                   help="calibration_params.json path override")
    e.add_argument("--classic", action="store_true", help="use_ml=False")
    e.set_defaults(fn=cmd_eval)

    r = sub.add_parser("report", help="Compare eval CSVs")
    r.add_argument("--csv", action="append", required=True,
                   help="label=path (repeatable)")
    r.set_defaults(fn=cmd_report)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
