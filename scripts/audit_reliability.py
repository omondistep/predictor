#!/usr/bin/env python3
"""Build the reliability ledger that gates the single canonical report.

What it does
------------
1. Stamps ``matches.odds_status`` / ``matches.odds_overround`` using
   :mod:`odds_guard`, so every consumer knows whether a stored price is usable.
2. Measures, from resolved history only, how well each *kind of selection*
   actually lands — grouped by (selection, declared confidence) and, where the
   sample allows, by league.
3. Writes ``data/reliability.json``: the allow/deny list the report generator
   consults before it prints a pick.

Why grouping matters
--------------------
The declared confidence labels are not calibrated (audit Oct 2026: "High"
realised 59.7%, "Medium-High" 53.0%, "Medium" 43.7%, "Low" 29.4% — while
"High" was labelled for probabilities up to 80%). A report that trusts the
label will therefore publish selections that are wrong far more often than the
label implies.  The ledger replaces the label with the *measured* hit rate.

Odds-derived numbers are deliberately excluded from the hit-rate gate: only
2.8% of stored triplets pass :mod:`odds_guard`, so any ROI computed from them
would be fiction.  Where a price is usable we do report ROI, flagged as such.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import odds_guard  # noqa: E402

DB_PATH = "history.db"
OUT_PATH = os.path.join("data", "reliability.json")

# A bucket must clear this many resolved matches before it may gate output.
MIN_SAMPLE = 200
# Shrinkage credibility pseudo-count: a bucket with few samples is pulled
# toward the base rate rather than trusted at face value.
K = 120
# Fallback used only when a selection has too little history to measure even an
# unconditional rate.  Should essentially never trigger.
DEFAULT_BASE = 0.40

# --------------------------------------------------------------------------
# Markets and periods
# --------------------------------------------------------------------------
#
# A stored pick is only comparable to history if it is scored against the same
# part of the match.  ``our_market`` records which market a pick came from, and
# this table records which scoreline that market is decided by.
#
# This distinction is not cosmetic.  A first-half Draw lands when the sides are
# level after 45 minutes -- measured base rate 39.5%, not the 24.2% a
# full-time Draw lands at.  Scoring one against the other made ``Draw`` look
# like a 32.8% bucket when the first-half market it actually came from is a
# different bet, and it published 91 first-half picks whose "reliability" was
# borrowed from full-time history.
FT = "FT"
HT = "HT"
MARKET_PERIOD = {
    "1X2": FT, "O/U": FT, "DC": FT, "BTTS": FT, "DNB": FT,
    "1H": HT,
    # Legacy rows predate the our_market column.  None of them carry HT scores
    # and every label they use is a full-time market label, so they are read as
    # full-time; ``unknown_period_rows`` in the ledger records how many were
    # inferred rather than tagged.
    None: FT,
}
# Canonical spelling for the "no market recorded" case, so it can be used as a
# dict key and written into ledger keys without special-casing None.
MARKET_UNKNOWN = "?"
UNKNOWN_MARKET_PERIOD = FT

# Which selections can legitimately appear in each market.  Used to score only
# real combinations and to keep a stray label from creating a phantom bucket.
MARKET_SELECTIONS = {
    "1X2": ("Home win", "Draw", "Away win"),
    "DC": ("1X", "X2", "12"),
    "O/U": ("Over 1.5", "Under 1.5", "Over 2.5", "Under 2.5",
            "Over 3.5", "Under 3.5", "Over 4.5", "Under 4.5",
            "Over 5.5", "Under 5.5"),
    "BTTS": ("Yes", "No"),
    # Draw-no-bet: a draw pushes the stake back, so it is neither a hit nor a
    # miss and is excluded from both numerator and denominator.
    "DNB": ("Home", "Away"),
    "1H": ("Home win", "Draw", "Away win"),
}

# Every selection the gate can score, and the base rates behind them.
#
# Base rates are MEASURED, not assumed.  They are the unconditional frequency
# with which each selection lands across *all* resolved matches in history --
# not across the matches the model happened to predict, which would bake the
# model's own selection bias straight into the yardstick it is measured
# against.  See compute_base_rates().
#
# "Over 4.5" / "Over 5.5" and the DNB labels were missing here, which meant
# 554 resolved over-4.5 picks and 764 draw-no-bet picks were silently dropped
# by resolve_selection() returning None -- they were shown by the report with
# no measured reliability behind them at all.
SELECTIONS = (
    "Home win", "Draw", "Away win",
    "1X", "X2", "12",
    "Home", "Away",
    "Over 1.5", "Under 1.5",
    "Over 2.5", "Under 2.5",
    "Over 3.5", "Under 3.5",
    "Over 4.5", "Under 4.5",
    "Over 5.5", "Under 5.5",
    "Yes", "No",
)

# Exact complements: these two land on precisely one of every match, so their
# rates must sum to 1.0.  Not used to derive rates — both members share the
# same denominator, so one can never be measurable while the other is not.
# They exist to self-check resolve_selection(), since a drift between the two
# tables would silently corrupt every edge figure in the gate.
#
# DNB is deliberately absent: a draw pushes on *both* Home and Away, so the two
# do not partition the match and their rates sum to less than 1.0 by exactly
# the draw rate.
COMPLEMENTS = {
    "Over 1.5": "Under 1.5", "Under 1.5": "Over 1.5",
    "Over 2.5": "Under 2.5", "Under 2.5": "Over 2.5",
    "Over 3.5": "Under 3.5", "Under 3.5": "Over 3.5",
    "Over 4.5": "Under 4.5", "Under 4.5": "Over 4.5",
    "Over 5.5": "Under 5.5", "Under 5.5": "Over 5.5",
    "Yes": "No", "No": "Yes",
}

# Confidence labels, ordered strongest first.  Used for display only — the
# gate keys off measured hit rate, never the label.
CONF_ORDER = ["High", "Medium-High", "Medium", "Low"]


# --------------------------------------------------------------------------
# Selection resolution
# --------------------------------------------------------------------------

def market_of(row) -> str:
    """Canonical market label for a row, normalising NULL to ``?``."""
    raw = _get(row, "our_market", None)
    if raw is None or str(raw).strip() == "":
        return MARKET_UNKNOWN
    return str(raw).strip()


def period_of(market: str) -> str:
    """Which scoreline decides this market: full-time or half-time."""
    return MARKET_PERIOD.get(market, UNKNOWN_MARKET_PERIOD)


def bucket_key(market: str, pred: str, conf: str) -> str:
    """Ledger key for one (market, selection, confidence) bucket.

    The market is part of the key because the selection label alone is
    ambiguous: ``Draw`` is both a full-time selection and a first-half one,
    decided by different scorelines against base rates 24.2% and 39.5%
    apart.  Keying on the label alone pooled the two and let first-half picks
    inherit full-time reliability.
    """
    return f"{market}|{pred}|{conf}"


def base_key(market: str, pred: str) -> str:
    """Ledger key for a measured base rate of one selection within a market."""
    return f"{market}|{pred}"


def resolve_selection(pred: str, hg: int, ag: int):
    """Map a stored ``our_prediction`` label + scoreline to hit/miss.

    Returns True/False, or None when the label cannot be scored against this
    scoreline.  None covers three distinct cases, all of which mean "this row
    contributes to no bucket":

    * the label is not a recognised market (a bug guard -- see SELECTIONS);
    * the scoreline is missing or is for the wrong period (the caller must pass
      the scoreline the market is decided by);
    * the pick pushed, which for draw-no-bet is a draw.

    ``hg``/``ag`` must already be the scoreline that decides the market: the
    half-time columns for a first-half pick, full-time otherwise.  Callers get
    this from :func:`scoreline_for`.
    """
    if hg is None or ag is None:
        return None
    total = hg + ag
    if pred == "Home win":
        return hg > ag
    if pred == "Away win":
        return ag > hg
    if pred == "Draw":
        return hg == ag
    # Draw-no-bet: the draw refunds the stake, so it is neither a hit nor a
    # miss.  Returning None keeps the row out of numerator *and* denominator,
    # which is the only correct treatment -- counting it as a miss would make
    # DNB look worse than it is, and counting it as a hit would make it look
    # better than it is.  Tested before the Home/Away branches below, or a
    # drawn DNB pick would register as a miss on both sides.
    if pred in ("Home", "Away"):
        if hg == ag:
            return None
        return (hg > ag) if pred == "Home" else (ag > hg)
    if pred == "1X":
        return hg >= ag
    if pred == "X2":
        return ag >= hg
    if pred == "12":
        return hg != ag
    if pred == "Over 1.5":
        return total >= 2
    if pred == "Under 1.5":
        return total <= 1
    if pred == "Over 2.5":
        return total >= 3
    if pred == "Under 2.5":
        return total <= 2
    if pred == "Over 3.5":
        return total >= 4
    if pred == "Under 3.5":
        return total <= 3
    if pred == "Over 4.5":
        return total >= 5
    if pred == "Under 4.5":
        return total <= 4
    if pred == "Over 5.5":
        return total >= 6
    if pred == "Under 5.5":
        return total <= 5
    if pred == "Yes":
        return hg > 0 and ag > 0
    if pred == "No":
        return hg == 0 or ag == 0
    return None


def scoreline_for(row, market: str):
    """The (home, away) goals that decide ``market`` for this row, or (None, None).

    Half-time markets are decided by the half-time columns, and those columns
    are missing for most historical first-half rows -- only 54 of 668 resolved
    ``1H`` rows have them.  Rather than silently substitute full-time goals
    (which scores a first-half draw as if it were a full-time draw, inflating
    ``Draw`` by 8 points), a row without its deciding scoreline is skipped and
    counted.
    """
    if period_of(market) == HT:
        return _get(row, "ht_home_goals"), _get(row, "ht_away_goals")
    return _get(row, "hg"), _get(row, "ag")


def price_for(pred: str, r: sqlite3.Row):
    """Bookmaker price for a selection, or None when unavailable/unvalidated."""
    if pred == "Home win":
        return r["odds_home"]
    if pred == "Away win":
        return r["odds_away"]
    if pred == "Draw":
        return r["odds_draw"]
    if pred == "Over 2.5":
        return r["odds_over25"]
    if pred == "Under 2.5":
        return r["odds_under25"]
    if pred == "Yes":
        return r["odds_btts_yes"]
    if pred == "No":
        return r["odds_btts_no"]
    # 1.5/3.5 goals and the double-chance lines have no stored price, so no
    # edge can ever be quoted for them from history.
    return None


# One-sided 95% confidence multiplier. 1.645 ≈ the 95th percentile of the
# standard normal, i.e. a one-sided 95% interval.
Z_ONE_SIDED_95 = 1.645


def wilson_lower_bound(hits: int, n: int, z: float = Z_ONE_SIDED_95) -> float:
    """Lower limit of a one-sided Wilson score interval on a hit rate.

    Used as a significance floor.  The gate tests 54 (selection, confidence)
    buckets, so at a plain 2pt threshold some buckets publish purely because
    they were the luckiest of 54 draws.  Requiring the *lower* bound to clear
    the base rate asks a stricter question: is the edge large enough that it
    would survive re-measuring on fresh matches?

    Wilson rather than a normal approximation because hit rates sit near 1.0
    for selections like Over 1.5, where the Wald interval misbehaves.
    """
    if n <= 0:
        return 0.0
    p = hits / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - margin)


# --------------------------------------------------------------------------
# Base rates
# --------------------------------------------------------------------------

def compute_base_rates(conn: sqlite3.Connection, since_days: int = 0) -> dict:
    """Measure the unconditional frequency of every selection, per market.

    The gate asks "does this pick beat what you'd get by picking blind?".  That
    yardstick has to be the selection's *unconditional* landing rate over every
    resolved match, so it is computed over all resolved matches here rather
    than from the ``totals`` accumulator in measure() -- the latter only covers
    matches the model actually predicted, which would tilt the baseline toward
    the model's own habits.

    Rates are keyed by market as well as selection, and each is measured over
    the population its market is actually decided by:

    * half-time markets over resolved matches that *have* half-time scores;
    * draw-no-bet over decisive matches only, since a draw pushes the stake
      back and belongs to neither outcome;
    * everything else over all resolved matches.

    Measuring a first-half draw against full-time results (the old behaviour)
    compared it to a 24.2% base rate when the market's own base rate is 39.5%.
    That understates the market by 15 points and let first-half buckets publish
    on full-time evidence.

    Same time window as measure() so base rate and hit rate are drawn from one
    population.  Rates are measured whenever any history exists; the thin-
    population case is labelled as such rather than silently swapped for a
    constant, and DEFAULT_BASE is reserved for an empty database.

    Returns {"rates": {"<market>|<selection>": {...}}, ...}.
    """
    where = ["actual_home_goals IS NOT NULL", "actual_away_goals IS NOT NULL"]
    args: list = []
    if since_days:
        where.append("created_at >= date('now', ?)")
        args.append(f"-{int(since_days)} days")

    scores = conn.execute(f"""
        SELECT id, created_at, forebet_url, home_team, away_team, match_date,
               actual_home_goals hg, actual_away_goals ag,
               ht_home_goals, ht_away_goals
        FROM matches WHERE {' AND '.join(where)}
    """, args).fetchall()
    # Same fixture counted once as in measure(): a double-stored match would
    # otherwise sit in the numerator of the base rate twice and in the
    # denominator of the hit rate once, biasing every edge figure.
    scores = dedupe_fixtures(scores)

    # Population per market-period, computed once.
    ft_rows = [(r["hg"], r["ag"]) for r in scores]
    ht_rows = [(r["ht_home_goals"], r["ht_away_goals"]) for r in scores
               if r["ht_home_goals"] is not None and r["ht_away_goals"] is not None]
    # A push is not an outcome, so it leaves both sides of the ratio.
    decisive_rows = [(hg, ag) for hg, ag in ft_rows if hg != ag]

    if not ft_rows:
        return {
            "rates": {base_key(m, sel): {"rate": DEFAULT_BASE, "n": 0, "hits": 0,
                                         "source": "default fallback — no resolved matches"}
                      for m, sels in MARKET_SELECTIONS.items() for sel in sels},
            "population_n": 0,
            "self_check": "skipped — no resolved matches",
            "window_days": int(since_days) if since_days else 0,
        }

    rates = {}
    for market, sels in MARKET_SELECTIONS.items():
        if market == "?":
            # Untagged legacy rows: every selection any full-time market can
            # produce, measured on the full-time population.
            sel_list = [s for s in SELECTIONS if s not in ("Home", "Away")]
            pop = ft_rows
        elif period_of(market) == HT:
            sel_list, pop = sels, ht_rows
        elif market == "DNB":
            sel_list, pop = sels, decisive_rows
        else:
            sel_list, pop = sels, ft_rows

        n = len(pop)
        for sel in sel_list:
            hits = sum(1 for hg, ag in pop if resolve_selection(sel, hg, ag))
            rates[base_key(market, sel)] = {
                "rate": (hits / n) if n else DEFAULT_BASE,
                "n": n,
                "hits": hits,
                "source": "measured" if n >= MIN_SAMPLE
                          else f"measured on a thin population ({n} matches)",
            }

    # Self-check: exact complements must partition every match. A breach means
    # resolve_selection() and the complement table have drifted apart, which
    # would silently corrupt every edge figure below.
    breaches = []
    for market, sels in MARKET_SELECTIONS.items():
        if market in ("?", "DNB"):
            continue  # no complement pairs, or a third outcome (the push)
        for sel in sels:
            partner = COMPLEMENTS.get(sel)
            if not partner:
                continue
            a = rates[base_key(market, sel)]["rate"]
            b = rates[base_key(market, partner)]["rate"]
            if abs(a + b - 1.0) > 1e-9:
                breaches.append(f"{market}:{sel}+{partner}={a + b:.6f}")

    return {
        "rates": rates,
        "population_n": len(ft_rows),
        "populations": {
            "full_time": len(ft_rows),
            "half_time": len(ht_rows),
            "decisive_full_time": len(decisive_rows),
        },
        "self_check": "ok" if not breaches else "FAILED: " + "; ".join(breaches),
        "window_days": int(since_days) if since_days else 0,
    }


def base_rate_of(base_rates: dict, market: str, pred: str) -> float:
    """Measured base rate for a selection within a market, or DEFAULT_BASE."""
    entry = (base_rates or {}).get("rates", {}).get(base_key(market, pred))
    if entry:
        return float(entry["rate"])
    # Fall back to the untagged full-time rate for the same label, so a market
    # that was never recorded is still measured against something honest rather
    # than the DEFAULT_BASE guess.
    entry = (base_rates or {}).get("rates", {}).get(base_key(MARKET_UNKNOWN, pred))
    return float(entry["rate"]) if entry else DEFAULT_BASE


# --------------------------------------------------------------------------
# Odds stamping
# --------------------------------------------------------------------------

def scan_odds(conn: sqlite3.Connection) -> Tuple[dict, list]:
    """Classify every stored 1X2 price. Read-only.

    Returns ``(counts, updates)`` where ``counts`` maps each ``odds_guard``
    verdict reason to how many rows carry it, and ``updates`` is the row list
    :func:`stamp_odds` writes back.  Split out so that a build run with
    ``--no-stamp`` still reports odds integrity: writing it into the ledger is
    what makes the report's "why most expected values are absent" note appear,
    and the ledger used to ship with an empty ``odds_integrity`` precisely
    because stamping had been skipped.
    """
    counts = defaultdict(int)
    upd = []
    for r in conn.execute("""
        SELECT id, odds_home, odds_draw, odds_away,
               forebet_home_pct, forebet_draw_pct, forebet_away_pct
        FROM matches
        WHERE odds_home IS NOT NULL OR odds_draw IS NOT NULL OR odds_away IS NOT NULL
    """):
        verdict = odds_guard.validate_1x2(
            r["odds_home"], r["odds_draw"], r["odds_away"],
            r["forebet_home_pct"], r["forebet_draw_pct"], r["forebet_away_pct"])
        counts[verdict.reason] += 1
        upd.append((verdict.reason, verdict.overround, r["id"]))
    return dict(counts), upd


def stamp_odds(conn: sqlite3.Connection) -> dict:
    """Fill matches.odds_status / matches.odds_overround. Idempotent."""
    existing = {r[1] for r in conn.execute("PRAGMA table_info(matches)")}
    for col, decl in (("odds_status", "TEXT"), ("odds_overround", "REAL")):
        if col not in existing:
            conn.execute(f"ALTER TABLE matches ADD COLUMN {col} {decl}")
    conn.commit()

    counts, upd = scan_odds(conn)
    conn.executemany(
        "UPDATE matches SET odds_status = ?, odds_overround = ? WHERE id = ?", upd)
    conn.commit()
    return counts


# --------------------------------------------------------------------------
# Reliability measurement
# --------------------------------------------------------------------------

def _get(row, key, default=None):
    """Read a column from a sqlite3.Row or a plain dict, tolerating absence."""
    try:
        v = row[key]
    except (KeyError, IndexError):
        return default
    return default if v is None else v


def fixture_key(row) -> str:
    """Canonical identity for a fixture, so one match counts once.

    The same Forebet fixture can be stored several times: its URL appears with
    and without percent-encoding (``...-def-belgrano-%28vr%29-2443574`` vs
    ``...-def-belgrano-(vr)-2443574``), and the model re-predicted some matches
    on a later run with a different pick. 152 such groups exist, and 26 of them
    disagree on the pick outright.

    Counting all of them inflates ``n`` on a bucket that only ever saw 152
    distinct matches — which matters twice over now the gate has a significance
    floor, since a duplicated observation makes thin evidence look conclusive.
    The trailing Forebet match id is the stable identifier; teams plus date are
    the fallback for rows that predate it.
    """
    url = _get(row, "forebet_url", "") or ""
    m = re.search(r"-(\d{5,})\s*$", url)
    if m:
        return f"fb:{m.group(1)}"
    return f"td:{(_get(row, 'home_team', '') or '').strip().lower()}|" \
           f"{(_get(row, 'away_team', '') or '').strip().lower()}|" \
           f"{_get(row, 'match_date', '') or ''}"


def dedupe_fixtures(rows) -> list:
    """Keep the earliest row per fixture.

    Earliest, not latest: the first call is the one made before the result was
    known, so it cannot have been chosen with hindsight. Dropping later rows
    rather than the original keeps the row that was actually published first.
    """
    seen = set()
    keep = []
    for r in sorted(rows, key=lambda x: (_get(x, "created_at", "") or "",
                                         _get(x, "id", 0) or 0)):
        k = fixture_key(r)
        if k in seen:
            continue
        seen.add(k)
        keep.append(r)
    return keep


def measure(conn: sqlite3.Connection, since_days: int = 0,
            base_rates: dict | None = None) -> dict:
    """Build per-market / per-selection / per-confidence reliability stats.

    ``base_rates`` comes from compute_base_rates(); it is computed here if not
    supplied so callers cannot accidentally score against assumed rates.
    """
    if base_rates is None:
        base_rates = compute_base_rates(conn, since_days)

    where = ["actual_home_goals IS NOT NULL", "our_prediction IS NOT NULL"]
    args: list = []
    if since_days:
        where.append("created_at >= date('now', ?)")
        args.append(f"-{int(since_days)} days")

    rows = conn.execute(f"""
        SELECT id, created_at, forebet_url, home_team, away_team, match_date,
               our_market, our_prediction, our_confidence, league,
               actual_home_goals hg, actual_away_goals ag,
               ht_home_goals, ht_away_goals,
               odds_home, odds_draw, odds_away, odds_over25, odds_under25,
               odds_btts_yes, odds_btts_no, odds_status
        FROM matches
        WHERE {' AND '.join(where)}
    """, args).fetchall()
    rows = dedupe_fixtures(rows)

    # (market, selection, confidence) -> accumulator
    buckets = defaultdict(lambda: {
        "n": 0, "hit": 0, "stakes": 0, "returns": 0.0, "priced": 0})
    # (market, selection) -> accumulator
    totals = defaultdict(lambda: {"n": 0, "hit": 0})
    # (league, market, selection) -> accumulator
    league_buckets = defaultdict(lambda: {"n": 0, "hit": 0})
    # Rows we refused to score, and why. Surfaced in the ledger so a gap in the
    # data is visible rather than showing up as mysterious thin buckets.
    skipped = defaultdict(int)

    for r in rows:
        market = market_of(r)
        pred = r["our_prediction"]
        hg, ag = scoreline_for(r, market)
        if hg is None or ag is None:
            # A half-time market with no half-time score. Scoring it on
            # full-time goals is what produced the bogus first-half buckets.
            skipped["missing_deciding_scoreline"] += 1
            continue
        hit = resolve_selection(pred, hg, ag)
        if hit is None:
            if pred in MARKET_SELECTIONS.get(market, ()):
                skipped["push_or_unscorable"] += 1  # e.g. a draw on draw-no-bet
            else:
                skipped["label_not_in_market"] += 1
            continue
        conf = r["our_confidence"] or "Unknown"

        b = buckets[(market, pred, conf)]
        b["n"] += 1
        b["hit"] += 1 if hit else 0
        totals[(market, pred)]["n"] += 1
        totals[(market, pred)]["hit"] += 1 if hit else 0

        # ROI only from rows whose prices passed validation.
        if r["odds_status"] == "ok":
            price = price_for(pred, r)
            if price and price > 1.01:
                b["priced"] += 1
                b["stakes"] += 1.0
                if hit:
                    b["returns"] += (price - 1.0)

        if pred in ("Home win", "Away win", "Draw", "Home", "Away",
                    "Over 2.5", "Under 2.5", "Yes", "No"):
            lb = league_buckets[(r["league"] or "Unknown", market, pred)]
            lb["n"] += 1
            lb["hit"] += 1 if hit else 0

    def shrink(hits: int, n: int, base: float) -> float:
        """Credibility-weighted hit rate: pulls thin samples to the base rate."""
        if n <= 0:
            return base
        return (hits + K * base) / (n + K)

    out_buckets = {}
    for (market, pred, conf), b in buckets.items():
        base = base_rate_of(base_rates, market, pred)
        rate = b["hit"] / b["n"] if b["n"] else base
        shrunk = shrink(b["hit"], b["n"], base)
        entry = {
            "market": market,
            "period": period_of(market),
            "selection": pred,
            "confidence": conf,
            "n": b["n"],
            "hits": b["hit"],
            "hit_rate": round(rate, 4),
            "expected": round(base, 4),
            "shrunk_rate": round(shrunk, 4),
            "edge_pts": round((shrunk - base) * 100, 2),
            "priced_n": b["priced"],
        }
        if b["priced"] >= 50:
            entry["roi"] = round(b["returns"] / b["stakes"] * 100, 2)
        out_buckets[bucket_key(market, pred, conf)] = entry

    out_leagues = {}
    for (league, market, pred), lb in league_buckets.items():
        if lb["n"] < 60:
            continue
        base = base_rate_of(base_rates, market, pred)
        out_leagues[f"{league}|{market}|{pred}"] = {
            "market": market,
            "selection": pred,
            "n": lb["n"],
            "hit_rate": round(lb["hit"] / lb["n"], 4),
            "shrunk_rate": round(shrink(lb["hit"], lb["n"], base), 4),
            "edge_pts": round((shrink(lb["hit"], lb["n"], base) - base) * 100, 2),
        }

    out_totals = {}
    for (market, pred), t in totals.items():
        if t["n"] < MIN_SAMPLE:
            continue
        base = base_rate_of(base_rates, market, pred)
        out_totals[base_key(market, pred)] = {
            "market": market,
            "period": period_of(market),
            "selection": pred,
            "n": t["n"],
            "hit_rate": round(t["hit"] / t["n"], 4),
            "expected": round(base, 4),
            # shrunk_rate must be stored, not just folded into edge_pts:
            # classify_pick() in reliable_report.py reads it to display the
            # rate it expects the selection to land. Omitting it raised a
            # KeyError that aborted the whole report write.
            "shrunk_rate": round(shrink(t["hit"], t["n"], base), 4),
            "edge_pts": round((shrink(t["hit"], t["n"], base) - base) * 100, 2),
        }

    unknown_period = sum(1 for r in rows if market_of(r) == MARKET_UNKNOWN)
    return {"buckets": out_buckets, "leagues": out_leagues, "totals": out_totals,
            "resolved_rows": len(rows), "base_rates": base_rates,
            "scored_rows": sum(b["n"] for b in buckets.values()),
            "unscored_rows": dict(skipped),
            "unknown_market_rows": unknown_period}


# --------------------------------------------------------------------------
# Gate construction
# --------------------------------------------------------------------------

def build_gate(measured: dict, min_samples: int, min_edge_pts: float,
               min_hit: float = 0.0) -> dict:
    """Turn measurements into an explicit publish / hold decision table.

    A bucket is publishable when it has enough history *and* its
    credibility-shrunk hit rate beats the unconditional base rate of that
    selection by ``min_edge_pts``.  The base-rate comparison is the honest
    test: "Over 1.5 at 88%" is not a signal, it is simply the definition of
    Over 1.5 landing.

    Significance is measured **relative to that same base rate**: the one-sided
    95% Wilson lower bound on the raw hit rate must clear the base rate by the
    full ``min_edge_pts``, not merely by a sliver.  Asking the lower bound to
    clear the bar — rather than asking it to clear zero — is what makes this a
    per-selection test instead of a fixed threshold on how often a pick wins:
    "Over 3.5 at 35%" and "1X2 Home at 55%" are judged against their own base
    rates, so no market is advantaged by having a naturally high hit rate.

    It replaces an earlier absolute floor (``min_hit >= 0.55`` on that lower
    bound), which required every published pick to win more than half the time
    regardless of market.  That constant suppressed 89 of 100 buckets, and a
    walk-forward showed it bought nothing the relative test does not already
    catch: with the floor removed and the bar left as "clears base at all",
    ``O/U|Under 1.5|Medium`` (+0.6pt lower-bound margin) and
    ``O/U|Over 3.5|Low`` (+0.9pt) both published and both then failed
    out-of-sample.  Requiring the full 2pt margin removes exactly those two and
    keeps the seven genuinely significant buckets the floor was holding down.
    ``min_hit`` remains available as an *optional* absolute floor (0 disables,
    which is the default) for anyone who wants to reason about stakeability
    that way.

    ROI is reported for reference only and is **never** part of the gate.
    Historical ROI from stored odds is not trustworthy: the odds columns
    failed validation on 87.7% of rows, and the surviving subset is biased
    toward matches where the book's favourite already matched the model's
    pick.  That bias is what inflates the raw figures to absurd levels
    (+145% on "Away win"), so a report that quotes them would be lying.
    The gate therefore rests only on hit rate vs base rate, neither of which
    touches the odds columns.
    """
    gate = {}
    for key, e in measured["buckets"].items():
        wilson_lo = wilson_lower_bound(e["hits"], e["n"])
        # Three independent conditions must all hold before a bucket may publish:
        #   1. enough history and a credibility-shrunk edge over the base rate
        #   2. statistical significance *at the same bar* — the lower bound of a
        #      one-sided 95% confidence interval on the raw hit rate must clear
        #      the base rate by the full min_edge_pts
        #   3. (optional, off by default) an absolute floor on that lower bound
        # (2) is what stops a 54-way search from promoting buckets that merely
        # got lucky; see wilson_lower_bound().  Because the margin is measured
        # against each selection's own base rate, a pick that beats a 23% base
        # by a sliver is held for the same reason a pick that beats a 66% base
        # by a sliver is held — no absolute constant decides it.
        margin_pts = (wilson_lo - e["expected"]) * 100
        publishable = (e["n"] >= min_samples
                       and e["edge_pts"] >= min_edge_pts
                       and margin_pts >= min_edge_pts
                       and wilson_lo >= min_hit)
        if e["n"] < min_samples:
            reason = f"only {e['n']} samples (need {min_samples})"
        elif e["edge_pts"] < min_edge_pts:
            reason = (f"shrunk rate {e['shrunk_rate']*100:.1f}% does not beat "
                      f"{e['expected']*100:.0f}% base by {min_edge_pts:g} pts")
        elif margin_pts < min_edge_pts:
            reason = (f"edge not significant — 95% lower bound "
                      f"{wilson_lo*100:.1f}% clears {e['expected']*100:.0f}% base "
                      f"by only {margin_pts:.1f} pts (need {min_edge_pts:g})")
        elif wilson_lo < min_hit:
            reason = (f"below the absolute floor — 95% lower bound "
                      f"{wilson_lo*100:.1f}% is under the {min_hit*100:.0f}% "
                      f"minimum for a pick that usually wins")
        else:
            reason = ("enough history + beats base rate + edge is statistically "
                      "significant at the same bar")
        entry = {
            "market": e["market"],
            "period": e["period"],
            "selection": e["selection"],
            "confidence": e["confidence"],
            "n": e["n"],
            "hits": e["hits"],
            "hit_rate": e["hit_rate"],
            "expected": e["expected"],
            "shrunk_rate": e["shrunk_rate"],
            "edge_pts": e["edge_pts"],
            "wilson_lower": round(wilson_lo, 4),
            "priced_n": e["priced_n"],
            "publish": bool(publishable),
            "reason": reason,
        }
        if e["priced_n"] >= 50:
            entry["roi"] = e["roi"]
            entry["roi_trustworthy"] = False
            entry["roi_note"] = (
                "computed only on the odds-validated subset, which is biased "
                "toward matches where the book already agreed with the model; "
                "treat as indicative, not as an expected return"
            )
        gate[key] = entry
    return gate


# --------------------------------------------------------------------------
# Calibration curve
# --------------------------------------------------------------------------

def calibration(conn: sqlite3.Connection) -> dict:
    """Reliability curve of the model's own stated probability vs realised.

    Two sources are averaged into the "stated" number the reports display:
    the Poisson and ML components, which is what the ensemble blends.  The
    result is the shrinkage factor applied to every displayed probability.
    """
    rows = conn.execute("""
        SELECT actual_home_goals hg, actual_away_goals ag,
               poisson_prob_home ph, poisson_prob_draw pd, poisson_prob_away pa,
               ml_prob_home mh, ml_prob_draw md, ml_prob_away ma
        FROM matches
        WHERE actual_home_goals IS NOT NULL
          AND poisson_prob_home IS NOT NULL AND ml_prob_home IS NOT NULL
    """).fetchall()

    buckets = defaultdict(lambda: {"n": 0, "hit": 0})
    for r in rows:
        ph, pd_, pa = r["ph"], r["pd"], r["pa"]
        mh, md, ma = r["mh"], r["md"], r["ma"]
        if None in (ph, pd_, pa, mh, md, ma):
            continue
        sp = ph + pd_ + pa
        sm = mh + md + ma
        if not (0.9 <= sp <= 1.1) or not (0.9 <= sm <= 1.1):
            continue
        ps = [0.5 * (ph / sp) + 0.5 * (mh / sm),
              0.5 * (pd_ / sp) + 0.5 * (md / sm),
              0.5 * (pa / sp) + 0.5 * (ma / sm)]
        total = sum(ps)
        if total <= 0:
            continue
        ps = [p / total for p in ps]
        actual = 0 if r["hg"] > r["ag"] else (2 if r["hg"] < r["ag"] else 1)
        top = max(range(3), key=lambda i: ps[i])
        b = min(int(ps[top] * 10), 9)
        buckets[b]["n"] += 1
        buckets[b]["hit"] += 1 if top == actual else 0

    curve = []
    tw = th = 0
    for b in sorted(buckets):
        n, h = buckets[b]["n"], buckets[b]["hit"]
        if n < 100:
            continue
        stated = (b + 1) / 10.0
        realised = h / n
        curve.append({
            "bucket": f"{b/10:.1f}-{(b+1)/10:.1f}",
            "n": n, "stated": round(stated, 3), "realised": round(realised, 3),
            "gap": round(realised - stated, 3),
            "multiplier": round(realised / stated, 3) if stated else 1.0,
        })
        tw += n
        th += n * realised

    overall_stated = sum(c["stated"] * c["n"] for c in curve) / tw if tw else 0.0
    overall_realised = th / tw if tw else 0.0
    return {
        "curve": curve,
        "n": tw,
        "mean_stated": round(overall_stated, 4),
        "mean_realised": round(overall_realised, 4),
        "global_multiplier": round(overall_realised / overall_stated, 4)
                            if overall_stated else 1.0,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--days", type=int, default=0,
                    help="restrict history to the last N days (0 = all)")
    ap.add_argument("--min-samples", type=int, default=MIN_SAMPLE)
    ap.add_argument("--min-edge", type=float, default=2.0,
                    help="minimum shrunk edge over base rate, in points")
    ap.add_argument("--min-hit", type=float, default=0.0,
                    help="optional ABSOLUTE floor on the one-sided 95%% lower "
                         "bound of the hit rate (0 = off, the default). The "
                         "gate's own significance bar is relative to each "
                         "selection's base rate and needs no constant here")
    ap.add_argument("--no-stamp", action="store_true",
                    help="skip writing odds_status back to the DB")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    odds_summary = {}
    if args.no_stamp:
        # Read-only: the ledger still has to carry the counts, because the
        # report keys its odds-integrity note on them.  Skipping the write must
        # not mean shipping an empty object.
        odds_summary, _ = scan_odds(conn)
        print("Odds integrity (scanned, not written — --no-stamp)...")
    else:
        print("Stamping odds integrity...")
        odds_summary = stamp_odds(conn)
    for reason, count in sorted(odds_summary.items(), key=lambda x: -x[1]):
        print(f"  {reason:<22} {count:>7}")
    total = sum(odds_summary.values())
    ok = odds_summary.get("ok", 0)
    print(f"  {'-> usable':<22} {ok}/{total} = {ok/max(total,1)*100:.1f}%")

    print("\nMeasuring reliability...")
    measured = measure(conn, since_days=args.days)
    br = measured["base_rates"]
    calib = calibration(conn)
    conn.close()

    gate = build_gate(measured, args.min_samples, args.min_edge, args.min_hit)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "history_days": args.days or None,
        "policy": {
            "min_samples": args.min_samples,
            "min_edge_pts": args.min_edge,
            "min_hit_rate": args.min_hit,
            "credibility_k": K,
            "base_rates": {
                "source": "measured from history: unconditional frequency over "
                          "all resolved matches, same time window as the hit rates",
                "population_n": br["population_n"],
                "window_days": br["window_days"],
                "self_check": br["self_check"],
                "rates": {s: round(v["rate"], 4) for s, v in br["rates"].items()},
                "detail": {s: {"rate": round(v["rate"], 4), "n": v["n"],
                               "hits": v["hits"], "source": v["source"]}
                           for s, v in br["rates"].items()},
            },
        },
        "odds_integrity": odds_summary,
        "odds_favourite_agreement": round(odds_guard.favourite_agreement_rate(args.db)["rate"], 4),
        "calibration": calib,
        "gate": gate,
        "leagues": measured["leagues"],
        "totals": measured["totals"],
        "resolved_rows": measured["resolved_rows"],
        "scored_rows": measured.get("scored_rows"),
        "unscored_rows": measured.get("unscored_rows", {}),
        "unknown_market_rows": measured.get("unknown_market_rows", 0),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nWrote {args.out}")

    print(f"\nBase rates (full-time population: {br['population_n']} resolved matches"
          f", half-time population: {br.get('populations', {}).get('half_time', 0)}"
          f"{', last ' + str(br['window_days']) + ' days' if br['window_days'] else ', all time'}):")
    for market, sels in MARKET_SELECTIONS.items():
        print(f"  [{market}]")
        for sel in sels:
            v = br["rates"].get(base_key(market, sel)) or {"rate": DEFAULT_BASE,
                                                           "n": 0, "hits": 0,
                                                           "source": "not measured"}
            note = "" if v["source"] == "measured" else f"  <- {v['source']}"
            print(f"    {sel:<12} {v['rate']*100:5.1f}%  ({v['hits']}/{v['n']}){note}")
    print(f"  complement self-check: {br['self_check']}")

    unscored = measured.get("unscored_rows", {})
    if unscored:
        print("\nRows deliberately not scored (they belong to no bucket):")
        for reason, count in sorted(unscored.items(), key=lambda x: -x[1]):
            print(f"  {reason:<32} {count:>7}")
    if measured.get("unknown_market_rows"):
        print(f"  {'legacy rows with no market tag':<32} "
              f"{measured['unknown_market_rows']:>7}  (scored as full-time)")

    pub = [k for k, v in gate.items() if v["publish"]]
    print(f"\nGate: {len(pub)}/{len(gate)} (market, selection, confidence) buckets publishable")
    for key in sorted(pub, key=lambda k: -gate[k]["shrunk_rate"]):
        v = gate[key]
        print(f"  {key:<42} n={v['n']:<5} hit={v['hit_rate']*100:5.1f}% "
              f"(base {v['expected']*100:.0f}%, shrunk {v['shrunk_rate']*100:.1f}%)")

    print(f"\nCalibration: stated {calib['mean_stated']*100:.1f}% -> "
          f"realised {calib['mean_realised']*100:.1f}% "
          f"(global multiplier {calib['global_multiplier']:.3f})")
    for c in calib["curve"]:
        print(f"  {c['bucket']}  n={c['n']:<6} stated {c['stated']*100:5.1f}%  "
              f"realised {c['realised']*100:5.1f}%  x{c['multiplier']:.3f}")


if __name__ == "__main__":
    main()