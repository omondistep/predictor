#!/usr/bin/env python3
"""
Auto-Learn Pipeline — continuous learning from played match outcomes.

Orchestrates the full learning cycle without manual intervention:
  1. Result scraping — fetch actual scores from Forebet for past predictions
  2. DB update — store results, mark as reviewed, log calibration entries
  3. Calibration analysis — detect bias per league/market/probability-bucket
  4. Model retraining — trigger ML model retrain when sufficient new data
  5. League profile update — adjust league parameters from actual outcomes

Usage:
  python auto_learn.py                           Run full pipeline
  python auto_learn.py --scrape-only             Only scrape & update results
  python auto_learn.py --calibrate-only          Only analyze bias & retrain
  python auto_learn.py --status                  Show learning status / data volumes
  python auto_learn.py --daemon                  Run continuously with interval
  python auto_learn.py --daemon-interval 3600    Daemon check interval in seconds
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

BASE = Path(__file__).parent

from database import (
    get_db, init_db, get_unreviewed_matches, update_result,
    get_calibration_summary, get_calibration_data_for_retraining,
)
from forebet_scraper import ForebetScraper


def _extract_result_from_forebet(soup) -> tuple | None:
    """Extract final score from Forebet soup. Returns (h, a) or None."""
    if not soup:
        return None
    import re
    from collections import Counter

    candidates = []
    for div in soup.find_all("div"):
        text = div.get_text(strip=True)
        m = re.match(r"^(\d+)\s*[-–:]\s*(\d+)$", text)
        if m:
            h, a = int(m.group(1)), int(m.group(2))
            if h <= 10 and a <= 10 and h + a <= 15 and not (h == 0 and a == 0):
                candidates.append((h, a))

    h1 = soup.find("h1")
    if h1:
        m = re.search(r"(\d+)\s*[-–:]\s*(\d+)", h1.get_text())
        if m:
            h, a = int(m.group(1)), int(m.group(2))
            if h <= 10 and a <= 10 and h + a <= 15:
                candidates.append((h, a))

    tables = soup.find_all("table", {"class": "stat-content"})
    for table in tables:
        rows = table.find_all("tr")
        for row in rows[:5]:
            cells = row.find_all("td")
            if len(cells) >= 3:
                m = re.search(r"(\d+)\s*[-–:]\s*(\d+)", cells[-1].get_text())
                if m:
                    h, a = int(m.group(1)), int(m.group(2))
                    if h <= 10 and a <= 10 and h + a <= 15:
                        candidates.append((h, a))

    if not candidates:
        return None
    best = Counter(candidates).most_common(1)[0][0]
    return best


def step_scrape_results(days_back: int = 14, delay: float = 0.3,
                         max_matches: int = 500) -> dict:
    """Scrape actual scores for unreviewed predictions from Forebet."""
    init_db()
    cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%d/%m/%Y")
    # Convert DD/MM/YYYY cutoff to YYYYMMDD integer for proper numeric comparison
    cutoff_parts = cutoff.split("/")
    cutoff_int = int(cutoff_parts[2]) * 10000 + int(cutoff_parts[1]) * 100 + int(cutoff_parts[0])
    conn = get_db()
    rows = conn.execute("""
        SELECT id, forebet_url, home_team, away_team, match_date
        FROM matches
        WHERE reviewed = 0
          AND forebet_url IS NOT NULL
          AND match_date IS NOT NULL
          AND CAST(SUBSTR(match_date, 7, 4) AS INTEGER) * 10000
              + CAST(SUBSTR(match_date, 4, 2) AS INTEGER) * 100
              + CAST(SUBSTR(match_date, 1, 2) AS INTEGER) < ?
        ORDER BY match_date DESC
        LIMIT ?
    """, (cutoff_int, max_matches)).fetchall()
    conn.close()

    pending = [dict(r) for r in rows]
    if not pending:
        return {"status": "no_pending", "updated": 0, "errors": 0, "total": 0}

    import time as _time
    updated = 0
    errors = 0
    no_score = 0

    for m in pending:
        scraper = ForebetScraper(m["forebet_url"])
        if scraper.fetch():
            score = _extract_result_from_forebet(scraper.soup)
            if score:
                try:
                    update_result(m["id"], score[0], score[1])
                    updated += 1
                except Exception:
                    errors += 1
            else:
                no_score += 1
        else:
            errors += 1
        _time.sleep(delay)

    return {
        "status": "done",
        "pending": len(pending),
        "updated": updated,
        "no_score": no_score,
        "errors": errors,
        "total": len(pending),
    }


def step_scrape_date_pages(days_back: int = 14, delay: float = 1.0,
                           min_matches: int = 5, dry_run: bool = False) -> dict:
    """Scrape ALL matches from Forebet date pages using AJAX endpoint.

    Unlike step_scrape_results() which only scrapes individual match pages
    for matches already in our DB, this scrapes entire date pages to discover
    ALL played matches and update league profiles.
    """
    try:
        from historical_calibrate import scrape_date_page, aggregate, update_profiles
    except ImportError:
        return {"status": "no_historical_calibrate_module", "updated": False}

    from datetime import timedelta

    dates = []
    d = datetime.now() - timedelta(days=days_back)
    end = datetime.now() - timedelta(days=1)  # yesterday
    while d <= end:
        dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)

    print(f"  Scraping {len(dates)} date pages ({dates[0]} to {dates[-1]})")

    all_results = []
    for i, date_str in enumerate(dates):
        matches = scrape_date_page(date_str)
        all_results.extend(matches)
        if i < len(dates) - 1:
            time.sleep(delay)

    print(f"  Total matches scraped: {len(all_results)}")

    if not all_results:
        return {"status": "no_matches", "updated": False, "total_matches": 0}

    aggregated = aggregate(all_results)
    update_profiles(aggregated, min_matches, dry_run=dry_run)

    return {
        "status": "done",
        "updated": not dry_run,
        "total_matches": len(all_results),
        "leagues": len(aggregated),
        "dry_run": dry_run,
    }


def step_calibrate() -> dict:
    """Run calibration analysis and auto-retrain if conditions are met."""
    try:
        from calibration_learner import analyze_calibration, auto_retrain, check_retrain_needed
    except ImportError:
        return {"status": "no_calibration_module", "analyzed": False, "retrained": False}

    analyze_calibration(min_samples=10)
    retrain_needed = check_retrain_needed()
    retrained = False
    if retrain_needed:
        print("  Retrain condition met. Retraining ML model...")
        retrained = auto_retrain(force=False)

    return {
        "status": "done",
        "analyzed": True,
        "retrain_needed": retrain_needed,
        "retrained": retrained,
    }


def show_status():
    """Display current learning status and data volumes."""
    init_db()
    summary = get_calibration_summary()
    retrain_data = get_calibration_data_for_retraining()

    conn = get_db()
    unreviewed = conn.execute("""
        SELECT COUNT(*) as cnt FROM matches
        WHERE reviewed = 0 AND match_date IS NOT NULL
    """).fetchone()["cnt"]

    unreviewed_past = conn.execute("""
        SELECT COUNT(*) as cnt FROM matches
        WHERE reviewed = 0
          AND match_date IS NOT NULL
          AND CAST(SUBSTR(match_date, 7, 4) AS INTEGER) * 10000
              + CAST(SUBSTR(match_date, 4, 2) AS INTEGER) * 100
              + CAST(SUBSTR(match_date, 1, 2) AS INTEGER) < ?
    """, (int(datetime.now().strftime("%Y")) * 10000 + int(datetime.now().strftime("%m")) * 100 + int(datetime.now().strftime("%d")),)).fetchone()["cnt"]

    total_preds = conn.execute("SELECT COUNT(*) as cnt FROM matches").fetchone()["cnt"]
    conn.close()

    print("=" * 55)
    print("CONTINUOUS LEARNING — STATUS")
    print("=" * 55)
    print(f"Total predictions in DB:     {total_preds}")
    print(f"Reviewed (calibrated):       {summary['total']}")
    print(f"Unreviewed (total):          {unreviewed}")
    print(f"Unreviewed (past, ready):    {unreviewed_past}")
    print(f"Our accuracy:                {summary['our_correct']}/{summary['total']} ({summary['our_pct']}%)")
    print(f"Forebet accuracy:            {summary['fb_correct']}/{summary['total']} ({summary['fb_pct']}%)")
    print(f"Calibration log entries:     {retrain_data['total_calibration_entries']}")
    print(f"New entries (last 30d):      {retrain_data['recent_30d']}")
    print(f"Training examples (model):   {retrain_data['last_retrain_examples']}")
    print(f"Last retrain:                {retrain_data['last_retrain_time'] or 'never'}")

    if retrain_data.get("accuracy_by_market"):
        print(f"\nAccuracy by market:")
        for m in retrain_data["accuracy_by_market"]:
            tag = " ✓" if (m["pct"] or 0) >= 50 else " ⚠"
            print(f"  {tag} {m['market']:<10}: {m['pct']}%  ({m['correct']}/{m['total']})")

    return {
        "total_predictions": total_preds,
        "reviewed": summary,
        "retrain_data": retrain_data,
        "unreviewed": unreviewed,
        "unreviewed_past": unreviewed_past,
    }


def run_full_pipeline(scrape_only: bool = False, calibrate_only: bool = False,
                      days_back: int = 14, delay: float = 0.3,
                      max_matches: int = 500, verbose: bool = True):
    """Run the complete automated learning pipeline."""
    print("=" * 55)
    print("AUTO-LEARN PIPELINE")
    print("=" * 55)

    results = {}
    status = "ok"

    if calibrate_only:
        print("\n[1/1] Calibration analysis & model retraining...")
        cal = step_calibrate()
        results["calibration"] = cal
        if not cal["analyzed"]:
            status = "warning"
        print(f"  Analyzed bias: {cal['analyzed']}, Retrained: {cal['retrained']}")
    else:
        print(f"\n[1/3] Scraping individual match results for past {days_back} days...")
        scrape = step_scrape_results(days_back=days_back, delay=delay, max_matches=max_matches)
        results["scrape"] = scrape
        if scrape["updated"] > 0:
            print(f"  Updated {scrape['updated']}/{scrape['total']} matches "
                  f"(no_score: {scrape['no_score']}, errors: {scrape['errors']})")
        else:
            print(f"  No results to update ({scrape['status']})")

        print(f"\n[2/3] Scraping ALL matches from Forebet date pages (league profiles)...")
        dp = step_scrape_date_pages(days_back=days_back, delay=delay, dry_run=False)
        results["date_pages"] = dp
        if dp["updated"]:
            print(f"  Scraped {dp['total_matches']} matches across {dp['leagues']} leagues")
        else:
            print(f"  Date page scraping: {dp['status']}")

        print("\n[3/3] Calibration analysis & model retraining...")
        cal = step_calibrate()
        results["calibration"] = cal
        if not cal["analyzed"]:
            status = "warning"
        print(f"  Analyzed bias: {cal['analyzed']}, Retrained: {cal['retrained']}")

    if status == "warning":
        print(f"\n{'=' * 55}")
        print("Pipeline completed with warnings (some optional components missing)")
    else:
        print(f"\n{'=' * 55}")
        print("Pipeline complete. Model continues learning.")
    print("=" * 55)

    results["status"] = status
    return results


def run_daemon(interval_seconds: int = 3600, days_back: int = 14,
               delay: float = 0.3, max_matches: int = 500):
    """Run the learning pipeline in a loop at the specified interval."""
    print(f"Auto-Learn Daemon starting (interval={interval_seconds}s)")
    print(f"Press Ctrl+C to stop.\n")

    while True:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Learning cycle...")
        try:
            run_full_pipeline(days_back=days_back, delay=delay, max_matches=max_matches, verbose=False)
        except KeyboardInterrupt:
            print("\nDaemon stopped by user.")
            break
        except Exception as e:
            print(f"  Cycle failed: {e}")
        print(f"\n  Next cycle in {interval_seconds}s...")
        try:
            time.sleep(interval_seconds)
        except KeyboardInterrupt:
            print("\nDaemon stopped by user.")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Continuous Learning Pipeline for Football Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  auto_learn.py                           Run full pipeline
  auto_learn.py --scrape-only             Only scrape & update results
  auto_learn.py --calibrate-only          Only analyze bias & retrain
  auto_learn.py --status                  Show learning status / data volumes
  auto_learn.py --daemon                  Run continuously with interval

Examples:
  auto_learn.py --daemon --daemon-interval 43200   Run twice daily
  auto_learn.py --days-back 30 --max-matches 200   Collect 30 days of results
        """,
    )
    parser.add_argument("--scrape-only", action="store_true", help="Only scrape results, skip calibration")
    parser.add_argument("--calibrate-only", action="store_true", help="Only run calibration & retrain")
    parser.add_argument("--status", action="store_true", help="Show learning status")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon with interval")
    parser.add_argument("--daemon-interval", type=int, default=3600,
                        help="Daemon check interval in seconds (default: 3600)")
    parser.add_argument("--days-back", type=int, default=14,
                        help="How many days back to look for results (default: 14)")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Delay between scrape requests (default: 0.3s)")
    parser.add_argument("--max-matches", type=int, default=500,
                        help="Max matches to scrape per cycle (default: 500)")

    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.daemon:
        run_daemon(
            interval_seconds=args.daemon_interval,
            days_back=args.days_back,
            delay=args.delay,
            max_matches=args.max_matches,
        )
        return

    run_full_pipeline(
        scrape_only=args.scrape_only,
        calibrate_only=args.calibrate_only,
        days_back=args.days_back,
        delay=args.delay,
        max_matches=args.max_matches,
    )


if __name__ == "__main__":
    main()
