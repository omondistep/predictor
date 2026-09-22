"""
predict2.py — pr2 entry point

Identical to predict.py in every way except analysis uses the new
7-step pipeline (pipeline.py) instead of the monolithic analyze_from_data().

All scraping, DB saving, HTML writing, and CLI flags are inherited from
predict.py so pr2 supports the full pr command surface:

    pr2 links.txt
    pr2 links/today.html
    pr2 --today
    pr2 --high-only links.txt
    pr2 --json links.txt
    pr2 results
    etc.

The pipeline prints a step-by-step trace for every match so you can watch
the chain of reasoning in real time.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ── Bootstrap: make sure we run from the repo root ──
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── Import the full predict.py public surface ──
# Everything from predict.py is available; we only override the
# analyze_from_data call inside the prediction loop.
import predict as _predict
from predict import (
    log,
    init_db,
    ensure_alias,
    scrape_and_save,
    _write_html,
    _update_index,
    schedule_retrain,
    CONF_RANK,
)

# ── The new pipeline ──
from pipeline import run_pipeline


def run_pipeline_predictions(
    links_path: str,
    high_only: bool = False,
    json_out: bool = False,
    compare_forebet: bool = True,
    use_ml: bool = True,
    single_link: bool = False,
) -> None:
    """Read Forebet links, scrape, run 7-step pipeline, store, and output predictions.

    Mirrors run_forebet_predictions() from predict.py but calls run_pipeline()
    instead of analyze_from_data().
    """
    import json
    from forebet_scraper import scrape_injuries, get_team_injury_summary
    from database import save_prediction, get_db

    with open(links_path) as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if not urls:
        log("No URLs found in " + links_path)
        return

    match_urls = [u for u in urls if "/football/matches/" in u]
    skipped = len(urls) - len(match_urls)
    if skipped:
        log(f"[Skipped {skipped} preview/other URLs — only match pages used]")

    log(f"\n══ pr2: step-by-step prediction pipeline ══")
    log(f"Processing {len(match_urls)} Forebet match links...\n")

    log("Scraping injury data...")
    injuries = scrape_injuries()
    if injuries:
        log(f"  Found injury data for {len(injuries)} teams")
    else:
        log("  No injury data available")

    BATCH_SIZE = 50
    results = []
    batch_num = 0
    _pred_dir = Path(__file__).parent / "predictions"
    _pred_dir.mkdir(exist_ok=True)

    def _flush_batch(results_so_far: list, batch_index: int, is_final: bool = False) -> None:
        if not results_so_far or single_link or json_out:
            return
        batch_label = f"batch_{batch_index:03d}"
        batch_path = _pred_dir / f"{batch_label}.html"
        _write_html(
            results_so_far, match_urls, compare_forebet, high_only,
            _save_to=batch_path,
            _title_suffix=f" — Batch {batch_index} ({len(results_so_far)} matches, pr2)",
        )
        log(f"  ✓ Flushed {batch_path.name}  ({len(results_so_far)} matches so far)")
        _latest = _pred_dir / "latest.html"
        if not is_final:
            _write_html(
                results_so_far, match_urls, compare_forebet, high_only,
                _save_to=_latest,
                _title_suffix=f" (pr2 — in progress {len(results_so_far)}/{len(match_urls)})",
            )

    for i, url in enumerate(match_urls, 1):
        log(f"\n{'─'*60}")
        log(f"[{i}/{len(match_urls)}] {url}")
        data = scrape_and_save(url)
        if not data.get("home_team"):
            if data.get("_blocked"):
                log("  [Skipped — Cloudflare rate-limited; wait and re-run]")
            else:
                log("  [Skipped — no data]")
            continue

        # Inject injury data
        home_team = data.get("home_team", "")
        away_team = data.get("away_team", "")
        if injuries:
            _hs = get_team_injury_summary(injuries, home_team) or {}
            _as = get_team_injury_summary(injuries, away_team) or {}
            for _side, _s in [("home", _hs), ("away", _as)]:
                for _f in ("total_injured", "forwards_out", "midfielders_out",
                           "defenders_out", "keepers_out", "key_players_out", "suspended"):
                    data[f"{_side}_{_f}"] = _s.get(_f, 0)

        # ── Run the 7-step pipeline ──
        pred = run_pipeline(data, use_ml=use_ml, verbose=True)

        # ── Minimum odds check (same as predict.py) ──
        min_odds = 1.10
        if pred["confidence"] == "Near Certain":
            min_odds = 1.08
        elif pred["confidence"] == "High":
            min_odds = 1.15
        elif pred["confidence"] == "Medium-High":
            min_odds = 1.25
        elif pred["confidence"] == "Medium":
            min_odds = 1.50

        pick = pred.get("pick", "")
        market = pred.get("market", "")
        pick_odds_val = None
        if market == "1X2":
            pick_odds_val = {"Home win": data.get("odds_home"),
                             "Draw": data.get("odds_draw"),
                             "Away win": data.get("odds_away")}.get(pick)
        elif market == "O/U":
            if "Over" in pick:
                pick_odds_val = data.get("odds_over25")
            elif "Under" in pick:
                pick_odds_val = data.get("odds_under25")

        if pick_odds_val and pick_odds_val <= min_odds:
            conf_rank = CONF_RANK.get(pred["confidence"], 99)
            if conf_rank < CONF_RANK.get("Medium-High", 2):
                pred["confidence"] = "Medium-High"
                log(f"  [odds] Downgraded {pick} (odds {pick_odds_val:.2f} < min {min_odds:.2f})")

        # ── Result detection (if already finished) ──
        actual_hg = data.get("actual_home_goals")
        actual_ag = data.get("actual_away_goals")
        actual_outcome = None
        correct_pick = None
        if actual_hg is not None and actual_ag is not None:
            actual_outcome = (
                "Home win" if actual_hg > actual_ag else
                "Away win" if actual_hg < actual_ag else "Draw"
            )
            our_market = pred.get("market", "")
            our_pick = pred["pick"]
            if our_market == "1X2":
                correct_pick = (our_pick == actual_outcome)
            elif our_market == "O/U":
                total_g = actual_hg + actual_ag
                if "Over" in our_pick:
                    correct_pick = total_g > float(our_pick.split()[-1])
                elif "Under" in our_pick:
                    correct_pick = total_g <= float(our_pick.split()[-1])
            elif our_market == "BTTS":
                both = actual_hg > 0 and actual_ag > 0
                correct_pick = (our_pick == "Yes" and both) or (our_pick == "No" and not both)
            elif our_market == "DNB":
                correct_pick = None if actual_outcome == "Draw" else (
                    (our_pick == "Home" and actual_outcome == "Home win") or
                    (our_pick == "Away" and actual_outcome == "Away win")
                )
            elif our_market == "DC":
                if our_pick == "1X": correct_pick = actual_outcome in ("Home win", "Draw")
                elif our_pick == "X2": correct_pick = actual_outcome in ("Away win", "Draw")
                elif our_pick == "12": correct_pick = actual_outcome in ("Home win", "Away win")

        # ── Save to DB ──
        poisson_probs = pred.get("_poisson_probs", (None, None, None))
        ml_pred_data = pred.get("_ml") or {}
        home_summary = get_team_injury_summary(injuries, home_team) if injuries else {}
        away_summary = get_team_injury_summary(injuries, away_team) if injuries else {}

        db_data = {
            **data,
            "our_prediction": pred["pick"],
            "our_confidence": pred["confidence"],
            "our_score_lean": pred["score_lean"],
            "our_stake": pred.get("_kelly_stake", 0.0),
            "our_market": pred.get("market", ""),
            "method_used": pred.get("_method", "") + "+pipeline",
            "poisson_prob_home": poisson_probs[0] if poisson_probs else None,
            "poisson_prob_draw": poisson_probs[1] if poisson_probs else None,
            "poisson_prob_away": poisson_probs[2] if poisson_probs else None,
            "ml_prob_home": ml_pred_data.get("ml_prob_home"),
            "ml_prob_draw": ml_pred_data.get("ml_prob_draw"),
            "ml_prob_away": ml_pred_data.get("ml_prob_away"),
            "home_injured_total": home_summary.get("total_injured", 0),
            "home_forwards_out": home_summary.get("forwards_out", 0),
            "home_midfielders_out": home_summary.get("midfielders_out", 0),
            "home_defenders_out": home_summary.get("defenders_out", 0),
            "home_key_players_out": home_summary.get("key_players_out", 0),
            "home_suspended": home_summary.get("suspended", 0),
            "away_injured_total": away_summary.get("total_injured", 0),
            "away_forwards_out": away_summary.get("forwards_out", 0),
            "away_midfielders_out": away_summary.get("midfielders_out", 0),
            "away_defenders_out": away_summary.get("defenders_out", 0),
            "away_key_players_out": away_summary.get("key_players_out", 0),
            "away_suspended": away_summary.get("suspended", 0),
        }
        try:
            save_prediction(db_data)
        except Exception as exc:
            log(f"  [DB] Save error: {exc}")

        # ── Assemble result record for HTML writer ──
        # Must match the exact key names that _write_html() expects.
        home_summary = get_team_injury_summary(injuries, home_team) if injuries else {}
        away_summary = get_team_injury_summary(injuries, away_team) if injuries else {}

        r = {
            "url": url,
            "home": home_team,
            "away": away_team,
            "league": data.get("league", ""),
            "date": data.get("match_date", ""),
            "time": data.get("match_time", ""),
            **pred,  # spreads pick, market, confidence, score_lean, reasoning, etc.
            "forebet": data.get("forebet_pred", ""),
            "forebet_pct": (
                data.get("forebet_home_pct"),
                data.get("forebet_draw_pct"),
                data.get("forebet_away_pct"),
            ),
            "forebet_over25_pct": data.get("forebet_over25_pct"),
            "forebet_btts_yes_pct": data.get("forebet_btts_yes_pct"),
            # Match result (finished games)
            "actual_home_goals": actual_hg,
            "actual_away_goals": actual_ag,
            "actual_outcome": actual_outcome,
            "correct_pick": correct_pick,
            "ht_home_goals": data.get("ht_home_goals"),
            "ht_away_goals": data.get("ht_away_goals"),
            # Injury data
            "home_injured": home_summary.get("total_injured", 0),
            "away_injured": away_summary.get("total_injured", 0),
            "home_key_out": home_summary.get("key_players_out", 0),
            "away_key_out": away_summary.get("key_players_out", 0),
            # Raw data for table display
            "home_form": data.get("home_form", ""),
            "away_form": data.get("away_form", ""),
            "home_pos": data.get("home_pos"),
            "away_pos": data.get("away_pos"),
            "odds_home": data.get("odds_home"),
            "odds_draw": data.get("odds_draw"),
            "odds_away": data.get("odds_away"),
            "odds_over25": data.get("odds_over25"),
            "odds_under25": data.get("odds_under25"),
            "odds_btts_yes": data.get("odds_btts_yes"),
            "odds_btts_no": data.get("odds_btts_no"),
            "h2h_matches": data.get("h2h_matches", 0),
            "h2h_home_wins": data.get("h2h_home_wins", 0),
            "h2h_draws": data.get("h2h_draws", 0),
            "h2h_away_wins": data.get("h2h_away_wins", 0),
            "h2h_goals_for": data.get("h2h_goals_for", 0),
            "h2h_goals_against": data.get("h2h_goals_against", 0),
            "h2h_avg_total_goals": data.get("h2h_avg_total_goals", 0),
            "home_home_avg_goals_for": data.get("home_home_avg_goals_for"),
            "home_home_avg_goals_against": data.get("home_home_avg_goals_against"),
            "away_away_avg_goals_for": data.get("away_away_avg_goals_for"),
            "away_away_avg_goals_against": data.get("away_away_avg_goals_against"),
            "home_home_matches": data.get("home_home_matches"),
            "away_away_matches": data.get("away_away_matches"),
            "home_over15_pct": data.get("home_over15_pct"),
            "home_under15_pct": data.get("home_under15_pct"),
            "away_over15_pct": data.get("away_over15_pct"),
            "away_under15_pct": data.get("away_under15_pct"),
            "home_over25_pct": data.get("home_over25_pct"),
            "home_under25_pct": data.get("home_under25_pct"),
            "away_over25_pct": data.get("away_over25_pct"),
            "away_under25_pct": data.get("away_under25_pct"),
            "home_over35_pct": data.get("home_over35_pct"),
            "home_under35_pct": data.get("home_under35_pct"),
            "away_over35_pct": data.get("away_over35_pct"),
            "away_under35_pct": data.get("away_under35_pct"),
            "home_btts_yes_pct": data.get("home_btts_yes_pct"),
            "home_btts_no_pct": data.get("home_btts_no_pct"),
            "away_btts_yes_pct": data.get("away_btts_yes_pct"),
            "away_btts_no_pct": data.get("away_btts_no_pct"),
            "home_scored_pct": data.get("home_scored_pct"),
            "home_conceded_pct": data.get("home_conceded_pct"),
            "away_scored_pct": data.get("away_scored_pct"),
            "away_conceded_pct": data.get("away_conceded_pct"),
            "home_total_shots_pg": data.get("home_total_shots_pg"),
            "home_shots_ontarget_pct": data.get("home_shots_ontarget_pct"),
            "away_total_shots_pg": data.get("away_total_shots_pg"),
            "away_shots_ontarget_pct": data.get("away_shots_ontarget_pct"),
            "home_clean_sheets_pct": data.get("home_clean_sheets_pct"),
            "away_clean_sheets_pct": data.get("away_clean_sheets_pct"),
            "match_id": None,
            "stat_fixes": data.get("stat_fixes") or [],
        }
        results.append(r)
        # Terminal summary line
        kelly_pct = pred.get("_kelly_stake", 0)
        kelly_interp = _predict._kelly_interpretation(kelly_pct * 100) if kelly_pct else "—"
        gates = pred.get("_pipeline", {}).get("gates_fired", [])
        gate_str = f"  🔒 {', '.join(gates)}" if gates else ""
        log(
            f"  → {market}: {pick}  ({pred['confidence']})  "
            f"score {pred['score_lean']}  Kelly {kelly_pct:.1%} ({kelly_interp})"
            + gate_str
        )

        # ── Batch flush ──
        if not json_out and len(results) % BATCH_SIZE == 0:
            batch_num += 1
            _flush_batch(results, batch_num)

    if not results:
        log("\nNo predictions generated.")
        return

    # ── Final output ──
    if json_out:
        import json as _json
        out = []
        for r in results:
            out.append({
                "home_team": r["home_team"],
                "away_team": r["away_team"],
                "league": r["league"],
                "match_date": r["match_date"],
                "pick": r["pick"],
                "market": r["market"],
                "confidence": r["confidence"],
                "score_lean": r["score_lean"],
                "kelly_stake": r["kelly_stake"],
                "model_prob": r["model_prob"],
                "url": r["url"],
                "pipeline": r.get("_pipeline", {}),
            })
        print(_json.dumps(out, indent=2))
        return

    # HTML output
    _latest = _pred_dir / "latest.html"
    _write_html(
        results, match_urls, compare_forebet, high_only,
        _save_to=_latest,
        _title_suffix=" (pr2 — step-by-step pipeline)",
    )
    _update_index(_pred_dir, _latest.stem)
    log(f"\n✓ {len(results)} predictions written → {_latest}")

    # Auto-schedule model retrain (same as pr)
    try:
        schedule_retrain()
    except Exception:
        pass


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(
        description="pr2 — Step-by-step football prediction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  predict2.py links.txt         Predict from a links file (step-by-step trace)
  predict2.py results           Update DB with match results (delegates to pr)
  predict2.py --today           Fetch today's matches → links/today.html → predict
  predict2.py --weekday         Fetch weekday matches → predict

Options:
  --high-only    Show only High / Near Certain predictions
  --json         JSON output (includes per-match pipeline breakdown)
  --no-compare   Skip Forebet comparison
  --no-ml        Classic Poisson only (no ML ensemble)
        """
    )
    parser.add_argument("file", nargs="?", help="File with Forebet URLs (txt or html)")
    parser.add_argument("--high-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-compare", action="store_true")
    parser.add_argument("--no-ml", "--classic", action="store_true")
    parser.add_argument("--today", action="store_true",
                        help="Fetch today's matches then predict")
    parser.add_argument("--weekday", action="store_true",
                        help="Fetch weekday matches then predict")

    args = parser.parse_args()

    ensure_alias()
    init_db()

    # ── results → delegate to the standard update_results ──
    if args.file and args.file.lower() == "results":
        from update_results import main as update_results_main
        update_results_main()
        return

    # ── --today: fetch links then run ──
    if args.today:
        from predict import fetch_today_links
        links_file = fetch_today_links()
        if not links_file:
            log("No links fetched for today.")
            return
        run_pipeline_predictions(
            links_file,
            high_only=args.high_only,
            json_out=args.json,
            compare_forebet=not args.no_compare,
            use_ml=not args.no_ml,
        )
        return

    # ── --weekday: fetch links then run ──
    if args.weekday:
        from predict import fetch_weekday_links
        links_file = fetch_weekday_links()
        if not links_file:
            log("No weekday links fetched.")
            return
        run_pipeline_predictions(
            links_file,
            high_only=args.high_only,
            json_out=args.json,
            compare_forebet=not args.no_compare,
            use_ml=not args.no_ml,
        )
        return

    # ── Single URL passed directly ──
    if args.file and (
        args.file.startswith("http://") or args.file.startswith("https://")
    ):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tf:
            tf.write(args.file + "\n")
            tmp_path = tf.name
        run_pipeline_predictions(
            tmp_path,
            high_only=args.high_only,
            json_out=args.json,
            compare_forebet=not args.no_compare,
            use_ml=not args.no_ml,
            single_link=True,
        )
        os.unlink(tmp_path)
        return

    # ── File of links ──
    if args.file:
        if not Path(args.file).exists():
            log(f"File not found: {args.file}")
            sys.exit(1)
        run_pipeline_predictions(
            args.file,
            high_only=args.high_only,
            json_out=args.json,
            compare_forebet=not args.no_compare,
            use_ml=not args.no_ml,
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
