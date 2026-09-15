#!/usr/bin/env python3
"""
Bettor's toolbox — data extraction + staking math for the opencode Bettor agent.

Commands:
  bettor.py audit [--json]
      Read-only weakness report straight from history.db:
      confidence calibration, worst/best leagues per market, model-vs-model
      accuracy, stored bias corrections, and kelly ROI (if any).

  bettor.py card <predictions.json | links.txt> [--json] [--all]
      Build the bettor's own betting card on top of a `pr --json` prediction
      dump (or a links file, which is shelled out to `pr --json` first).
      Computes, per candidate pick:
        * fair (de-vigged) probability from bookmaker odds
        * edge %   = model_prob - fair_prob
        * EV %     = model_prob * odds - 1
        * stake %  = Quarter-Kelly capped at 5%, scaled by the league's
                     historical accuracy in that market (STUD/neutral/WEAK/COLD)
      Only EV>0 picks are printed unless --all is passed.

  bettor.py learn
      Convenience: run the standard local learning step
      (pr --learn-calibration: bias analysis + ML retrain). Run `pr results`
      first to scrape actual scores.

Exit code is 0 on success. Read-only on the DB (never writes history.db).
"""

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DB = REPO / "history.db"

NON_ML = "--no-ml"

# Target calibration per confidence bucket (expected hit rate).
CONF_TARGETS = {
    "Near Certain": 0.70,
    "High": 0.60,
    "Medium-High": 0.50,
    "Medium": 0.40,
    "Low": 0.30,
}

# Odds fields available per match in the `pr --json` dump.
MARKET_ODDS = {
    "1X2": {
        "Home win": "odds_home",
        "Draw": "odds_draw",
        "Away win": "odds_away",
    },
    "O/U": {
        "Over 2.5": "odds_over25",
        "Under 2.5": "odds_under25",
        "Over 1.5": "odds_over15",
        "Under 1.5": "odds_under15",
        "Over 3.5": "odds_over35",
        "Under 3.5": "odds_under35",
    },
    "BTTS": {
        "Yes": "odds_btts_yes",
        "No": "odds_btts_no",
    },
}

# Markets/picks that have no direct bag/key in the dump (no EV math possible).
NO_ODDS_MARKETS = {"DNB", "DC", "1H"}

MIN_LEAGUE_SAMPLES = 25


def _conn():
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    return conn


def _try(query, args=()):
    try:
        with _conn() as c:
            return [dict(r) for r in c.execute(query, args).fetchall()]
    except sqlite3.DatabaseError as e:
        return {"_error": str(e)}


def _pct(n, d):
    return round(100.0 * n / d, 1) if d else 0.0


# ─────────────────────────────────────────────
# audit
# ─────────────────────────────────────────────

def audit():
    report = {}

    # A. Confidence calibration
    rows = _try(
        "SELECT confidence, COUNT(*) n, SUM(correct) correct "
        "FROM calibration_log GROUP BY confidence ORDER BY n DESC"
    )
    if isinstance(rows, dict):
        report["confidence"] = rows
    else:
        buckets = []
        for r in rows:
            pct = _pct(r["correct"], r["n"])
            target = CONF_TARGETS.get(r["confidence"])
            status = "ok"
            if target is not None:
                if pct < (target - 0.03) * 100:
                    status = "OVERCONFIDENT (actual below target)"
                elif pct > (target + 0.10) * 100:
                    status = "underconfident (actual well above target)"
            buckets.append({**r, "pct": pct, "target_pct": round(target * 100, 1) if target else None, "status": status})
        report["confidence"] = buckets

    # B. Worst / best leagues per market
    rows = _try(
        "SELECT m.league, c.market, COUNT(*) n, SUM(c.correct) correct "
        "FROM calibration_log c JOIN matches m ON m.id = c.match_id "
        "GROUP BY m.league, c.market "
        "HAVING COUNT(*) >= ? ORDER BY (SUM(c.correct)*1.0/COUNT(*)) ASC",
        (MIN_LEAGUE_SAMPLES,),
    )
    if isinstance(rows, dict):
        report["league_market"] = rows
    else:
        worst, best = {}, {}
        for r in rows:
            pct = _pct(r["correct"], r["n"])
            r = {**r, "pct": pct}
            worst.setdefault(r["market"], []).append(r)
            best.setdefault(r["market"], []).append(r)
        report["league_market_worst"] = {m: v[:12] for m, v in worst.items()}
        report["league_market_best"] = {m: sorted(v, key=lambda x: -x["pct"])[:3] for m, v in best.items()}
        # Overall league rating (all markets pooled)
        report["worst_leagues_overall"] = [{**r, "pct": _pct(r["correct"], r["n"])} for r in rows[:12]]

    # C. Model comparison
    mm = _try("SELECT * FROM model_market_accuracy ORDER BY market, model")
    report["model_vs_forebet"] = mm
    ml = _try(
        "SELECT league, market, ml_correct, ml_total, poisson_correct, poisson_total "
        "FROM ml_league_accuracy WHERE ml_total >= 20 ORDER BY (ml_correct*1.0/ml_total) ASC"
    )
    if not isinstance(ml, dict):
        report["ml_vs_poisson_worst"] = [
            {**r, "ml_pct": _pct(r["ml_correct"], r["ml_total"]),
             "poisson_pct": _pct(r["poisson_correct"], r["poisson_total"]),
             "gap": _pct(r["ml_correct"], r["ml_total"]) - _pct(r["poisson_correct"], r["poisson_total"])}
            for r in ml[:10]
        ]

    # D. Stored bias corrections (largest |bias|)
    b = _try(
        "SELECT league, market, threshold, bucket, bias, sample_count "
        "FROM calibration_bias WHERE sample_count >= 10 ORDER BY ABS(bias) DESC LIMIT 15"
    )
    report["bias_corrections"] = b

    # E. Kelly ROI
    k = _try("SELECT COUNT(*) n, ROUND(SUM((CASE WHEN result=1 THEN odds-1 ELSE -1 END)*stake),2) pnl, "
             "ROUND(SUM(stake),2) staked FROM kelly_log")
    if isinstance(k, dict):
        report["kelly"] = {"error": k["_error"]}
    elif k and k[0]["n"]:
        roi = round(100.0 * k[0]["pnl"] / k[0]["staked"], 1) if k[0]["staked"] else 0.0
        report["kelly"] = {**k[0], "roi_pct": roi}
    else:
        report["kelly"] = {"n": 0}

    return report


def print_audit(report):
    L = lambda s: print(s)

    L("=" * 72)
    L("SHARP'S WEAKNESS REPORT  (history.db)")
    L("=" * 72)

    L("\n[A] CONFIDENCE CALIBRATION")
    L(f"{'Confidence':<14}{'n':>8}{'hit%':>8}{'target':>9}  status")
    L("-" * 60)
    for b in report.get("confidence", []):
        if "error" in b:
            L(f"  db error: {b['error']}")
            break
        tgt = f"{b['target_pct']:.0f}%" if b["target_pct"] is not None else "—"
        L(f"{b['confidence']:<14}{b['n']:>8}{b['pct']:>8}{tgt:>9}  {b['status']}")

    L("\n[B] WORST LEAGUES PER MARKET  (pooled; top 5 shown, min 25 samples)")
    wm = report.get("league_market_worst", {})
    if isinstance(wm, dict) and wm:
        for mkt, rows in list(wm.items())[:5]:
            L(f"\n  -- {mkt} --")
            for r in rows[:5]:
                L(f"  {r['league'][:34]:<36}{r['n']:>5}  {r['pct']:>5.1f}%")

    L("\n[C] MODEL COMPARISON")
    mm = report.get("model_vs_forebet", [])
    if isinstance(mm, list):
        L(f"{'model':<5}{'market':<7}{'n':>6}{'hit%':>8}")
        L("-" * 40)
        for r in mm:
            L(f"{r['model']:<5}{r['market']:<7}{r['total']:>6}{_pct(r['correct'], r['total']):>8}")
        worst_ml = report.get("ml_vs_poisson_worst", [])
        if worst_ml:
            L("\n  ML vs Poisson (leagues where ML trails, worst gap):")
            for r in worst_ml:
                L(f"  {r['league'][:32]:<34} ML {r['ml_pct']:>5.1f}% vs Poisson {r['poisson_pct']:>5.1f}%  (n={r['ml_total']})")

    L("\n[D] ACTIVE BIAS CORRECTIONS (|bias| top, n>=10)")
    for b in report.get("bias_corrections", []):
        if "error" in b:
            L("  db error")
            break
        L(f"  {b['league'][:22]:<24}{b['market']:<6}{b['bucket']:<12}{b['bias']:+.3f}  n={b['sample_count']}")

    L("\n[E] KELLY LOG")
    k = report.get("kelly", {})
    if k.get("n"):
        L(f"  bets={k['n']}  staked={k['staked']}  pnl={k['pnl']}  ROI={k['roi_pct']}%")
    else:
        L("  empty (no staking history yet)")

    L("\n[DONE]")


def cmd_audit(args):
    report = audit()
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_audit(report)


# ─────────────────────────────────────────────
# card
# ─────────────────────────────────────────────

def _run_pr_json(links_path):
    py = REPO / ".venv" / "bin" / "python"
    if not py.exists():
        py = Path(sys.executable)
    cmd = [str(py), str(REPO / "predict.py"), "--json", str(links_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        print(f"pr --json failed:\n{proc.stderr[-2000:]}", file=sys.stderr)
        sys.exit(2)
    return json.loads(proc.stdout)


def _load_matches(path):
    p = Path(path)
    if p.exists() and p.suffix.lower() == ".json":
        return json.loads(p.read_text())
    return _run_pr_json(p)


def _devig(odds):
    """Proportionally remove bookmaker margin; returns {key: fair_prob}."""
    imp = {k: 1.0 / v for k, v in odds.items() if v and v > 1.0}
    s = sum(imp.values())
    if s <= 0:
        return {}
    return {k: v / s for k, v in imp.items()}


def _league_market_accuracy():
    acc = {}
    rows = _try(
        "SELECT m.league, c.market, COUNT(*) n, SUM(c.correct) correct "
        "FROM calibration_log c JOIN matches m ON m.id=c.match_id "
        "GROUP BY m.league, c.market HAVING COUNT(*) >= 20"
    )
    if isinstance(rows, list):
        for r in rows:
            acc.setdefault(r["market"], {})[r["league"]] = r["correct"] / r["n"]
    return acc


def _rel_factor(acc, league, market):
    pct = acc.get(market, {}).get(league)
    if pct is None:
        return 1.0, "unknown"
    if pct >= 0.58:
        return 1.0, "STUD"
    if pct >= 0.50:
        return 1.0, "ok"
    if pct >= 0.40:
        return 0.5, "WEAK"
    return 0.0, "COLD"


def _stake(odds, prob, kelly_frac, cap):
    b = odds - 1
    if b <= 0 or prob <= 0:
        return 0.0
    return round(min((odds * prob - 1) / b * kelly_frac, cap), 3)


def cmd_card(args):
    matches = _load_matches(args.file)
    acc = _league_market_accuracy()
    picks = []

    for m in matches:
        league = m.get("league", "?")
        home, away = m.get("home", "?"), m.get("away", "?")
        label = f"{home} vs {away}"

        odds = {k: m.get(k) for k in ("odds_home", "odds_draw", "odds_away",
                                      "odds_over25", "odds_under25",
                                      "odds_btts_yes", "odds_btts_no")}
        odds = {k: v for k, v in odds.items() if isinstance(v, (int, float)) and v > 1.0}

        # fair probabilities per market from the odds we DO have
        fair = {}
        for mkt, mapping in MARKET_ODDS.items():
            present = {pick: odds[odkey] for pick, odkey in mapping.items() if odkey in odds}
            if len(present) >= 1:
                fair[mkt] = _devig(present)

        candidates = m.get("all_picks") or []
        for c in candidates:
            mkt = c.get("market")
            pick = c.get("pick")
            prob = c.get("model_prob")
            if not mkt or not pick or prob is None:
                continue

            entry = {
                "league": league, "match": label, "date": m.get("date", ""),
                "market": mkt, "pick": pick, "confidence": c.get("confidence"),
                "model_prob": round(prob, 4),
                "is_primary": (m.get("pick") == pick and m.get("market") == mkt),
            }

            if mkt in NO_ODDS_MARKETS:
                entry["note"] = "no bookmaker odds in feed"
                picks.append(entry)
                continue

            odkey_map = MARKET_ODDS.get(mkt, {})
            odkey = odkey_map.get(pick)
            bp = odds.get(odkey) if odkey else None
            if bp is None:
                entry["note"] = f"no odds for threshold ({pick})"
                picks.append(entry)
                continue

            fprob = (fair.get(mkt) or {}).get(pick)
            fair_p = fprob if fprob else 1.0 / bp
            ev = prob * bp - 1
            edge = prob - fair_p
            rel, grade = _rel_factor(acc, league, mkt)
            stake = _stake(bp, prob, 0.25, 0.05) * rel
            entry.update({
                "odds": bp, "fair_prob": round(fair_p, 4),
                "edge_pct": round(edge * 100, 2),
                "ev_pct": round(ev * 100, 2),
                "stake_pct": round(stake * 100, 2),
                "reliability": grade,
            })
            picks.append(entry)

    value = [p for p in picks if "ev_pct" in p and p["ev_pct"] > 0]
    value.sort(key=lambda p: -p["ev_pct"])

    if args.json:
        out = {"value_picks": value, "all_candidates": picks}
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    print("=" * 78)
    print("BETTOR'S CARD  (Quarter-Kelly, max 5%, scaled by league reliability)")
    print("=" * 78)
    if not value:
        print("\n  No +EV picks in this slate. The sharp move is to BET NOTHING.")
    else:
        hdr = f"{'LEAGUE':<24}{'MATCH':<26}{'MARKET/PICK':<22}{'EV%':>6}{'EDGE%':>7}{'ODDS':>6}{'STAKE%':>7} {'NOTE'}"
        print("\n" + hdr)
        print("-" * 78)
        for p in value:
            mkt = f"{p['market']}:{p['pick']}"
            note = "" if p["reliability"] in ("STUD", "ok") else f"[{p['reliability']}]"
            print(
                f"{p['league'][:23]:<24}{p['match'][:25]:<26}{mkt[:21]:<22}"
                f"{p['ev_pct']:>6}{p['edge_pct']:>7}{p['odds']:>6}{p['stake_pct']:>7} {note}"
            )
    if args.all:
        passed = {id(p) for p in value}
        losers = [p for p in picks if id(p) not in passed and "note" not in p]
        if losers:
            print(f"\n{'All candidates (incl. -EV, for reference)':-^78}")
            for p in losers[:15]:
                mkt = f"{p['market']}:{p['pick']}"
                print(f"{p['league'][:23]:<24}{p['match'][:25]:<26}{mkt[:21]:<22}{p['ev_pct']:>6}{p['edge_pct']:>7}{p['odds']:>6}")
    print()


def cmd_learn(args):
    py = REPO / ".venv" / "bin" / "python"
    if not py.exists():
        py = Path(sys.executable)
    print("Running local learning step (bias analysis + ML retrain)...")
    rc = subprocess.run([str(py), str(REPO / "predict.py"), "--learn-calibration"]).returncode
    sys.exit(rc)


def main():
    ap = argparse.ArgumentParser(description="Bettor's toolbox for the opencode Bettor agent")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit", help="weakness report from history.db")
    a.add_argument("--json", action="store_true")
    a.set_defaults(fn=cmd_audit)

    c = sub.add_parser("card", help="build the bettor's card from pr --json output or a links file")
    c.add_argument("file")
    c.add_argument("--json", action="store_true")
    c.add_argument("--all", action="store_true", help="also print -EV candidates")
    c.set_defaults(fn=cmd_card)

    l = sub.add_parser("learn", help="run local calibration learning (bias + retrain)")
    l.set_defaults(fn=cmd_learn)

    args = ap.parse_args()
    if args.cmd == "card" and not args.file:
        ap.error("card needs a predictions.json or a links file")
    args.fn(args)


if __name__ == "__main__":
    main()