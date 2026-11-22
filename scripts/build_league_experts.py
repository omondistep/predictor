"""Build ``data/league_experts.json``: per-league statistical specialists.

Each league with enough resolved history gets an expert entry:

* base rates      — what lands blind in this league (home/draw/away, O/U 2.5,
  BTTS), blended toward the global rate for small samples
* component accuracy — how often ML / Poisson / Forebet / my pick are correct
  in this league, used to set dynamic ensemble weights
* calibration multiplier — realised / stated accuracy for this league, shrunk
  toward 1.0 when the sample is thin; applied at prediction time to scale the
  global calibrated probability

Only leagues meeting the gate's sample floor (>= MIN_RESOLVED resolved
matches) get an expert. Thin leagues fall back to the global model — an
unmeasured league is not a proven one.

Usage:
    .venv/bin/python scripts/build_league_experts.py        # write ledger
    .venv/bin/python scripts/build_league_experts.py --min-resolved 300
"""
import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from predict import detect_league  # noqa: E402

DB = ROOT / "history.db"
OUT = ROOT / "data" / "league_experts.json"

# Gate: a league needs this many resolved matches to get its own expert.
DEFAULT_MIN_RESOLVED = 200

# Shrinkage prior for the calibration multiplier: blend the league's measured
# multiplier toward 1.0 (no adjustment) based on sample size.
CAL_PRIOR_WEIGHT = 100  # pseudo-observations of "multiplier = 1.0"


def _result(home_goals, away_goals):
    if home_goals is None or away_goals is None:
        return None
    if home_goals > away_goals:
        return "Home win"
    if home_goals == away_goals:
        return "Draw"
    return "Away win"


_FB_MAP = {"1": "Home win", "X": "Draw", "2": "Away win"}


def _normalize_forebet(pred):
    if pred is None:
        return None
    return _FB_MAP.get(str(pred).strip(), str(pred).strip())


def _is_1x2_prediction(pred):
    return pred in ("Home win", "Draw", "Away win")


def _over25(home_goals, away_goals):
    if home_goals is None or away_goals is None:
        return None
    return (home_goals + away_goals) > 2.5


def _btts(home_goals, away_goals):
    if home_goals is None or away_goals is None:
        return None
    return home_goals > 0 and away_goals > 0


def build(min_resolved: int) -> dict:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    # All resolved matches with predictions + actuals.
    rows = conn.execute("""
        SELECT league, our_prediction, forebet_pred,
               ml_prob_home, ml_prob_draw, ml_prob_away,
               poisson_prob_home, poisson_prob_draw, poisson_prob_away,
               forebet_prob_home, forebet_prob_draw, forebet_prob_away,
               actual_home_goals, actual_away_goals
        FROM matches
        WHERE actual_home_goals IS NOT NULL AND actual_away_goals IS NOT NULL
    """).fetchall()
    conn.close()

    # Group by detect_league key.
    groups = defaultdict(list)
    for r in rows:
        key = detect_league(r["league"] or "")
        groups[key].append(r)

    global_home = sum(1 for r in rows if _result(r["actual_home_goals"], r["actual_away_goals"]) == "Home win") / len(rows)
    global_draw = sum(1 for r in rows if _result(r["actual_home_goals"], r["actual_away_goals"]) == "Draw") / len(rows)
    global_away = sum(1 for r in rows if _result(r["actual_home_goals"], r["actual_away_goals"]) == "Away win") / len(rows)
    global_over25 = sum(1 for r in rows if _over25(r["actual_home_goals"], r["actual_away_goals"])) / len(rows)
    global_btts = sum(1 for r in rows if _btts(r["actual_home_goals"], r["actual_away_goals"])) / len(rows)

    experts = {}
    for key, league_rows in sorted(groups.items()):
        n = len(league_rows)
        if n < min_resolved:
            continue

        def rate(fn):
            vals = [fn(r["actual_home_goals"], r["actual_away_goals"]) for r in league_rows]
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else None

        home_rate = rate(lambda h, a: _result(h, a) == "Home win")
        draw_rate = rate(lambda h, a: _result(h, a) == "Draw")
        away_rate = rate(lambda h, a: _result(h, a) == "Away win")
        over25_rate = rate(_over25)
        btts_rate = rate(_btts)

        def acc(pred_col, is_forebet=False):
            """Accuracy of the pick derived from pred_col vs actual.

            Only counts 1X2 predictions ('Home win'/'Draw'/'Away win'); market
            picks like 'Over 1.5' are skipped. Forebet's '1'/'X'/'2' codes are
            normalised to match.
            """
            hits = 0; total = 0
            for r in league_rows:
                actual = _result(r["actual_home_goals"], r["actual_away_goals"])
                pred = r[pred_col]
                if is_forebet:
                    pred = _normalize_forebet(pred)
                if actual is None or pred is None:
                    continue
                if not _is_1x2_prediction(pred) or not _is_1x2_prediction(actual):
                    continue
                total += 1
                if pred == actual:
                    hits += 1
            return (hits / total if total else None, total)

        my_acc, my_n = acc("our_prediction", is_forebet=False)
        fb_acc, fb_n = acc("forebet_pred", is_forebet=True)

        # Component accuracies from probability columns (argmax).
        def prob_acc(ph, pd, pa):
            hits = 0; total = 0
            for r in league_rows:
                actual = _result(r["actual_home_goals"], r["actual_away_goals"])
                if actual is None:
                    continue
                p = {"Home win": ph(r), "Draw": pd(r), "Away win": pa(r)}
                if any(v is None for v in p.values()):
                    continue
                best = max(p, key=p.get)
                total += 1
                if best == actual:
                    hits += 1
            return hits / total if total else None

        ml_acc = prob_acc(lambda r: r["ml_prob_home"], lambda r: r["ml_prob_draw"], lambda r: r["ml_prob_away"])
        pois_acc = prob_acc(lambda r: r["poisson_prob_home"], lambda r: r["poisson_prob_draw"], lambda r: r["poisson_prob_away"])
        fb_prob_acc = prob_acc(lambda r: r["forebet_prob_home"], lambda r: r["forebet_prob_draw"], lambda r: r["forebet_prob_away"])

        # Calibration multiplier: my accuracy vs forebet accuracy (relative
        # skill in this league). Shrink toward 1.0 for thin samples.
        rel_skill = None
        if my_acc is not None and fb_acc is not None and fb_acc > 0:
            rel_skill = my_acc / fb_acc
        # Shrink: weight = n / (n + prior); multiplier = w*rel + (1-w)*1.0
        cal_mult = 1.0
        cal_confidence = 0.0
        if rel_skill is not None:
            w = n / (n + CAL_PRIOR_WEIGHT)
            cal_mult = w * rel_skill + (1.0 - w) * 1.0
            cal_confidence = w  # how much we trust the league-specific value

        # Component weights: weight each component by its accuracy, only where
        # it has enough samples in this league.
        comps = {}
        for name, acc_val, n_comp in (("ml", ml_acc, n), ("poisson", pois_acc, n),
                                       ("forebet", fb_prob_acc, n), ("default", my_acc, my_n)):
            if acc_val is not None:
                comps[name] = {"accuracy": round(acc_val, 4), "samples": n_comp}

        total_acc = sum(c["accuracy"] for c in comps.values())
        weights = {}
        if total_acc > 0:
            weights = {name: round(c["accuracy"] / total_acc, 4) for name, c in comps.items()}
        else:
            weights = {"ml": 0.25, "poisson": 0.35, "forebet": 0.25, "default": 0.15}

        experts[key] = {
            "resolved_matches": n,
            "base_rates": {
                "home_win_rate": round(home_rate, 4) if home_rate is not None else None,
                "draw_rate": round(draw_rate, 4) if draw_rate is not None else None,
                "away_win_rate": round(away_rate, 4) if away_rate is not None else None,
                "over_25_rate": round(over25_rate, 4) if over25_rate is not None else None,
                "btts_yes_rate": round(btts_rate, 4) if btts_rate is not None else None,
            },
            "global_base_rates": {
                "home_win_rate": round(global_home, 4),
                "draw_rate": round(global_draw, 4),
                "away_win_rate": round(global_away, 4),
                "over_25_rate": round(global_over25, 4),
                "btts_yes_rate": round(global_btts, 4),
            },
            "component_accuracy": comps,
            "component_weights": weights,
            "calibration": {
                "my_accuracy": round(my_acc, 4) if my_acc is not None else None,
                "my_samples": my_n,
                "forebet_accuracy": round(fb_acc, 4) if fb_acc is not None else None,
                "relative_skill": round(rel_skill, 4) if rel_skill is not None else None,
                "multiplier": round(cal_mult, 4),
                "confidence": round(cal_confidence, 4),
            },
        }

    ledger = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "min_resolved": min_resolved,
        "total_leagues": len(experts),
        "total_resolved_matches": len(rows),
        "experts": experts,
    }
    return ledger


def main() -> None:
    ap = argparse.ArgumentParser(description="Build per-league expert ledger")
    ap.add_argument("--min-resolved", type=int, default=DEFAULT_MIN_RESOLVED,
                    help="minimum resolved matches for a league expert (default 200)")
    args = ap.parse_args()

    ledger = build(args.min_resolved)

    # Backup any existing ledger.
    if OUT.exists():
        bak = OUT.with_suffix(".json.bak")
        if bak.exists():
            bak.unlink()
        OUT.replace(bak)

    OUT.write_text(json.dumps(ledger, indent=2))
    print(f"wrote {OUT}")
    print(f"leagues with experts: {ledger['total_leagues']}")
    print(f"total resolved matches: {ledger['total_resolved_matches']}")
    print()
    print("Top experts by resolved matches:")
    for key, exp in sorted(ledger["experts"].items(),
                           key=lambda kv: -kv[1]["resolved_matches"])[:12]:
        cal = exp["calibration"]
        print(f"  {key:28s} n={exp['resolved_matches']:5d}  "
              f"my_acc={cal['my_accuracy']} fb_acc={cal['forebet_accuracy']}  "
              f"cal_mult={cal['multiplier']} (conf={cal['confidence']})")


if __name__ == "__main__":
    main()
