"""The single canonical prediction report: ``predictions/picks.html``.

Why one file
------------
The pipeline used to emit seven HTML reports (``latest``, ``best``,
``consistent``, ``high80``, ``bankers``, ``bettor`` plus a ``batch_NNN`` every
50 matches).  None of them answered the only question that matters: *given how
well this kind of pick has actually landed, which of today's selections should
I trust?*  Every one of them printed the model's own stated probability, which
an Oct 2026 audit showed runs 3-13 points optimistic depending on the bucket,
and the declared confidence labels were worse still — "High" realised 59.7%,
"Medium-High" 53.0%, "Medium" 43.7%, "Low" 29.4%.

This module produces one file that answers the question directly.  Every pick
is looked up in the reliability ledger (``data/reliability.json``, built by
``scripts/audit_reliability.py``) and placed in one of three tiers:

``RELIABLE``
    At least ``min_samples`` resolved historical matches of this exact
    (selection, confidence) pair, whose credibility-shrunk hit rate beats the
    *unconditional base rate* for that selection.  This is the bar that
    matters: "Over 1.5 lands 88% of the time" is not a signal, it is the
    definition of Over 1.5.

``WATCH``
    Enough history to measure, but no edge over the base rate — or an edge but
    too few samples to trust.  Shown, clearly marked, not recommended.

``HELD``
    Suppressed.  No history, or the measured rate is at or below what the
    market gets for free.

Probabilities are displayed after applying the calibration multiplier measured
in the ledger, so the number on screen is the one history says to expect.

An expected value is printed **only** for a price that passes
:mod:`odds_guard`: 88% of the odds stored in ``history.db`` are unusable, so an
edge figure computed from them would be fiction.  A rejected price is reported
as rejected, never as a number.

Each fixture also carries a *best bet*: the best selection across every
market the fixture is priced in, not just the model's primary pick and not
necessarily 1/X/2.  Only a positive-EV candidate qualifies at all, and among
those the tier decides (RELIABLE over WATCH over HELD) with EV choosing
inside one tier — EV is ``model × price − 1``, which grows exactly where the
model disagrees with the market, and measured history says that disagreement
is the model's error.  When nothing prices, the ranking falls back to this
match's *model lift over base rate* — calibrated probability minus the
selection's unconditional base rate — never to the ledger's bucket edge,
which is one number per bucket and therefore ranks buckets rather than
matches.  When nothing prices *positive*, the cell prints "no +EV selection"
rather than a negative number under a Best bet heading.
"""

from __future__ import annotations

import html
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import odds_guard

from reserve_filter import reserve_reason

ROOT = Path(__file__).resolve().parent
LEDGER_PATH = ROOT / "data" / "reliability.json"

# A ledger older than this is rebuilt rather than trusted.
LEDGER_MAX_AGE_DAYS = 7

TIER_RELIABLE = "RELIABLE"
TIER_WATCH = "WATCH"
TIER_HELD = "HELD"

# How far the model's probability may sit from the de-vigged market before the
# Best bet cell prints the gap as a warning.  Measured over 3,146 odds-valid
# resolved rows: the market's error stays within ±2.5pts of the realised
# outcome in every disagreement bin while the model's error grows with the gap
# (up to +22pts), so a wide gap is disagreement to disclose, not value to sell.
MODEL_MARKET_GAP_PTS = 10.0

_LEDGER_CACHE: Optional[dict] = None


# --------------------------------------------------------------------------
# Ledger
# --------------------------------------------------------------------------

def load_ledger(path: Path = LEDGER_PATH, max_age_days: int = LEDGER_MAX_AGE_DAYS,
                allow_stale: bool = True) -> Optional[dict]:
    """Load the reliability ledger, orNone when absent/unusable.

    Returns ``None`` if the ledger is missing, malformed, or older than
    ``max_age_days`` (unless ``allow_stale``).  Callers should degrade to an
    un-gated report rather than invent reliability numbers.
    """
    global _LEDGER_CACHE
    if _LEDGER_CACHE is not None and path == LEDGER_PATH:
        return _LEDGER_CACHE

    if not path.exists():
        if not allow_stale:
            return None
        return None

    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None

    gen = data.get("generated_at")
    if gen and not allow_stale:
        try:
            age = datetime.now() - datetime.fromisoformat(gen)
            if age > timedelta(days=max_age_days):
                return None
        except ValueError:
            pass
    elif gen and max_age_days:
        try:
            age = datetime.now() - datetime.fromisoformat(gen)
            if age > timedelta(days=max_age_days):
                data["_stale"] = True
                data["_age_days"] = age.days
        except ValueError:
            pass

    if path == LEDGER_PATH:
        _LEDGER_CACHE = data
    return data


def ledger_is_stale(ledger: Optional[dict]) -> bool:
    return bool(ledger and ledger.get("_stale"))


def rebuild_ledger(db_path: str = "history.db") -> bool:
    """Re-run the audit so the ledger matches the current history. Returns success."""
    import subprocess
    import sys

    script = ROOT / "scripts" / "audit_reliability.py"
    if not script.exists():
        return False
    try:
        subprocess.run(
            [sys.executable, str(script), "--db", db_path],
            cwd=str(ROOT), check=True, capture_output=True, timeout=1800)
    except (subprocess.SubprocessError, OSError):
        return False
    global _LEDGER_CACHE
    _LEDGER_CACHE = None
    return True


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def calibrated(prob: Optional[float], ledger: Optional[dict]) -> Optional[float]:
    """Apply the ledger's measured calibration multiplier to a probability.

    The ensemble was measured over-confident (stated 52.8% vs realised 49.1%
    across 16,179 resolved matches, global multiplier 0.931).  Multiplying here
    means the report quotes what history says to expect.  Multipliers vary by
    bucket, so the nearest calibration point is used when the curve is present.
    """
    if prob is None:
        return None
    if not ledger:
        return prob
    curve = (ledger.get("calibration") or {}).get("curve") or []
    if not curve:
        mult = (ledger.get("calibration") or {}).get("global_multiplier", 1.0)
        return _clamp(prob * mult)
    best, best_d = 1.0, 1e9
    for point in curve:
        lo, hi = (float(x) for x in point["bucket"].split("-"))
        mid = (lo + hi) / 2
        d = abs(prob - mid)
        if d < best_d:
            best, best_d = point.get("multiplier", 1.0), d
    return _clamp(prob * best)


def _clamp(p: float, lo: float = 0.01, hi: float = 0.97) -> float:
    return max(lo, min(hi, p))


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

def _key(pick: str, confidence: str, market: Optional[str] = None,
         ledger: Optional[dict] = None) -> str:
    """Ledger key for a pick.

    The market is part of the key whenever the ledger carries market-aware keys.
    It matters because selection labels are ambiguous across markets: ``Draw``
    is both a full-time selection and a first-half one, decided by different
    scorelines against base rates 24.1% and 39.9% apart.  Keying on the label
    alone let 91 first-half picks inherit full-time reliability and publish as
    RELIABLE on evidence that never described them.

    ``ledger`` is passed explicitly rather than read from the module cache:
    callers classify against a ledger they have in hand, and silently keying off
    a different cached one returns HELD for every pick.
    """
    if not _ledger_is_market_aware(ledger):
        return f"{pick}|{confidence}"
    return f"{_norm_market(market)}|{pick}|{confidence}"


def _norm_market(market: Optional[str]) -> str:
    if market is None or not str(market).strip():
        return "?"
    return str(market).strip()


def _ledger_is_market_aware(ledger: Optional[dict]) -> bool:
    """True when the ledger's gates carry the market in their key.

    Lets the report read a ledger written before this change without silently
    mismatching every key, and then say so loudly instead.
    """
    if not ledger:
        return False
    gate = ledger.get("gate") or {}
    if not gate:
        return False
    sample = next(iter(gate.values()))
    if isinstance(sample, dict) and sample.get("market"):
        return True
    return any(k.count("|") >= 2 for k in gate)


def _aggregate_gate(ledger: dict, pick: str, market: Optional[str]) -> Optional[dict]:
    """Sum every confidence of one selection out of the gate.

    Used when a candidate has no confidence of its own — the derived side of a
    two-way market, or a pipeline that emitted ``None`` — so the exact
    ``(market, selection, confidence)`` key cannot exist.  The shrinkage is the
    ledger's own credibility formula, ``(hits + k · base) / (n + k)`` with
    ``policy.credibility_k``, so an aggregated figure stays on the same scale
    as the per-confidence ones beside it.

    Returns None when the selection has no gate presence at all, which is the
    genuine "nothing is known" case.
    """
    gate = ledger.get("gate") or {}
    prefix = (f"{_norm_market(market)}|{pick}|"
              if _ledger_is_market_aware(ledger) else f"{pick}|")
    hits = n = 0
    base = None
    for key, e in gate.items():
        if not key.startswith(prefix) or not isinstance(e, dict):
            continue
        n += e.get("n") or 0
        hits += e.get("hits") or 0
        if base is None:
            base = e.get("expected")
    if n <= 0 or not isinstance(base, (int, float)):
        return None
    k = (ledger.get("policy") or {}).get("credibility_k") or 120
    shrunk = (hits + k * base) / (n + k)
    return {
        "hit_rate": hits / n,
        "n": n,
        "expected": base,
        "shrunk_rate": shrunk,
        "edge_pts": round((shrunk - base) * 100, 2),
        "reason": (f"no pick at this confidence — {n:,} resolved matches "
                   f"aggregated across every confidence of {pick}"),
    }


def classify_pick(pick: str, confidence: str, ledger: Optional[dict],
                  market: Optional[str] = None) -> dict:
    """Look one pick up in the ledger and decide its tier.

    Returns a dict with ``tier``, ``hit_rate``, ``n``, ``expected``,
    ``shrunk_rate``, ``edge_pts`` and a human-readable ``reason``.  When there
    is no ledger the pick is never promoted to RELIABLE — an unmeasured pick
    cannot be a reliable one.

    ``market`` selects the right base rate and hit rate for picks whose label is
    shared across markets (first-half versus full-time ``Draw``).
    """
    out = {
        "tier": TIER_HELD,
        "hit_rate": None, "n": 0, "expected": None,
        "shrunk_rate": None, "edge_pts": None,
        "reason": "no reliability data available for this selection",
    }
    if not ledger:
        return out

    gate = ledger.get("gate") or {}
    entry = gate.get(_key(pick, confidence, market, ledger))
    if not entry:
        # Fall back to the selection's total across all confidences, so a pick
        # with no history at its own confidence still shows what is known.
        # The totals table is keyed "<market>|<selection>".
        totals = ledger.get("totals") or {}
        if _ledger_is_market_aware(ledger):
            total = totals.get(f"{_norm_market(market)}|{pick}")
        else:
            total = totals.get(pick)
        if total:
            # Display fields come through .get(): a ledger written by an older
            # audit_reliability.py (totals lacked shrunk_rate) must degrade a
            # row's numbers, not abort the whole report write — one absent key
            # once discarded every pick of a 722-match run.
            out.update({
                "hit_rate": total.get("hit_rate"), "n": total.get("n", 0),
                "expected": total.get("expected"),
                "shrunk_rate": total.get("shrunk_rate"),
                "edge_pts": total.get("edge_pts"),
                "reason": f"no pick at this confidence in history "
                          f"({total.get('n', 0)} resolved matches at other confidences)",
            })
            return out
        # The totals accumulator only covers the selections that have appeared
        # as a primary prediction, so it silently omits whole selections —
        # Over 2.5 and BTTS No are absent from a ledger that has thousands of
        # resolved matches for both.  Sum the gate's own per-confidence buckets
        # instead; reporting "0 matches" for those would be a factually wrong
        # answer to "how much is known here".
        agg = _aggregate_gate(ledger, pick, market)
        if agg:
            out.update(agg)
            if agg["n"] >= (ledger.get("policy") or {}).get("min_samples", 200):
                out["tier"] = TIER_WATCH
            return out
        return out

    # Same rule as the totals fallback: read defensively. A missing field can
    # cost a row its numbers, but it must never cost the run its report.
    out.update({
        "hit_rate": entry.get("hit_rate"), "n": entry.get("n", 0),
        "expected": entry.get("expected"),
        "shrunk_rate": entry.get("shrunk_rate"),
        "edge_pts": entry.get("edge_pts"),
        "reason": entry.get("reason", "no reason recorded in ledger"),
    })
    if entry.get("publish"):
        out["tier"] = TIER_RELIABLE
    elif out["n"] >= (ledger.get("policy") or {}).get("min_samples", 200):
        out["tier"] = TIER_WATCH
    else:
        out["tier"] = TIER_HELD
    return out


def _match_prob(result: dict, pick: str) -> Optional[float]:
    """Best available probability for a named pick on a result dict.

    Falls back to the Poisson triple so a pick still gets a number when the
    specialised key (``_p_under25`` and friends) is missing.
    """
    direct = {
        "Home win": result.get("_p_home"),
        "Away win": result.get("_p_away"),
        "Draw": result.get("_p_draw"),
        "Over 2.5": result.get("_p_over25"),
        "Under 2.5": result.get("_p_under25"),
        "Yes": result.get("_p_btts_yes"),
        "No": result.get("_p_btts_no"),
        "1X": result.get("_p_1x"),
        "X2": result.get("_p_x2"),
        "12": result.get("_p_12"),
    }.get(pick)
    if direct is not None:
        return direct

    # 1X2 from the Poisson triple (ordered home, draw, away).
    triple = result.get("_poisson_probs")
    if pick in ("Home win", "Draw", "Away win") and triple and all(triple):
        try:
            h, d, a = (float(x) for x in triple)
        except (TypeError, ValueError):
            return None
        total = h + d + a
        if total <= 0:
            return None
        return {"Home win": h, "Draw": d, "Away win": a}[pick] / total

    return _from_poisson(result, _predicate_for(pick))


def _predicate_for(pick: str):
    """Ground-truth test for a selection, or None if we cannot evaluate it.

    Double chances are included because they are ordinary functions of the
    scoreline: 1X is "not an away win", X2 is "not a home win", 12 is "not a
    draw". Leaving them out meant a 1X pick with no ``_p_1x`` fell through to
    _from_poisson with a None predicate and took the entire report down with it.
    """
    return {
        "Over 0.5": lambda h, a: h + a >= 1,
        "Over 1.5": lambda h, a: h + a >= 2,
        "Under 1.5": lambda h, a: h + a <= 1,
        "Over 2.5": lambda h, a: h + a >= 3,
        "Under 2.5": lambda h, a: h + a <= 2,
        "Over 3.5": lambda h, a: h + a >= 4,
        "Under 3.5": lambda h, a: h + a <= 3,
        "Yes": lambda h, a: h > 0 and a > 0,
        "No": lambda h, a: h == 0 or a == 0,
        "1X": lambda h, a: h >= a,
        "X2": lambda h, a: h <= a,
        "12": lambda h, a: h != a,
    }.get(pick)


def _from_poisson(result: dict, predicate) -> Optional[float]:
    """Compute an over/under probability from the expected-goals pair."""
    if predicate is None:
        return None
    exp = result.get("_exp_goals")
    if not exp:
        return None
    try:
        lh, la = float(exp[0]), float(exp[1])
    except (TypeError, ValueError, IndexError):
        return None
    if lh <= 0 or la <= 0:
        return None

    def pois(lam: float, k: int) -> float:
        import math
        return math.exp(-lam) * lam ** k / math.factorial(k)

    cap = 12
    ph = [pois(lh, k) for k in range(cap)]
    pa = [pois(la, k) for k in range(cap)]
    total = 0.0
    for i in range(cap):
        for j in range(cap):
            if predicate(i, j):
                total += ph[i] * pa[j]
    return _clamp(total)


def _flag_text(flag) -> str:
    """Red flags and warnings arrive as dicts (``{'severity','msg'}``) or strings."""
    if isinstance(flag, dict):
        msg = flag.get("msg") or flag.get("message") or flag.get("text") or ""
        return str(msg)
    return str(flag)


def _pick_text(candidate: dict) -> str:
    """Human label for a candidate pick."""
    if not isinstance(candidate, dict):
        return str(candidate)
    pick = candidate.get("pick") or ""
    conf = candidate.get("confidence") or ""
    return f"{pick} ({conf})" if conf else str(pick)


# ─────────────────────────────────────────────────────────────────────────────
# Best bet: rank every market on the fixture, not just the primary pick
# ─────────────────────────────────────────────────────────────────────────────

_OU_THR = {"0.5": "05", "1.5": "15", "2.5": "25",
           "3.5": "35", "4.5": "45", "5.5": "55"}

_TIER_RANK = {TIER_RELIABLE: 2, TIER_WATCH: 1, TIER_HELD: 0}


def _group_sides(market: str, pick: str) -> Optional[List[str]]:
    """Every selection that shares one priceable group with ``pick``.

    The three groups are the 1X2 triplet, the two BTTS outcomes, and the two
    sides of a single goal line.  Nothing outside these is completable: a
    derived side has to be exhaustive or the book's margin cannot be removed
    from it later.
    """
    if market == "1X2":
        return ["Home win", "Draw", "Away win"]
    if market == "BTTS":
        return ["Yes", "No"]
    if market == "O/U":
        m = re.match(r"^(Over|Under)\s+([0-9.]+)$", (pick or "").strip())
        if m:
            return [f"Over {m.group(2)}", f"Under {m.group(2)}"]
    return None


def _candidate_universe(res: dict) -> List[dict]:
    """Every distinct ``(market, pick)`` this fixture actually offers.

    Three sources, in order:

    1. the synthesis shortlist (``all_picks``) — the ranked picks the pipeline
       led with;
    2. the ML analysis list (``_ml_analysis.all_picks``) — where the goal lines
       at every threshold live;
    3. **the sides the pipeline did not emit.**  The shortlist carries only the
       side it prefers, so a fixture that leads with ``BTTS No`` offers no
       ``BTTS Yes`` at all — yet BTTS Yes is exactly what a reader wants priced
       when the model is wrong, and the 1X2 sides it did not pick are the ones
       with the long prices.  Filling each group out (probability from the
       direct key, else ``1 - p`` of a two-way sibling) is what lets the ranking
       compare *every* side against the book rather than only the side the model
       happened to lead with.  A derived side inherits its group's confidence:
       the conviction belongs to the market, not to which side of it lands.

    Reading only source 1 is what confined the old report to whatever the model
    had already chosen to lead with.
    """
    seen, out = set(), []
    for src in (res.get("all_picks"),
                (res.get("_ml_analysis") or {}).get("all_picks")):
        for c in src or []:
            if not isinstance(c, dict):
                continue
            market, pick = c.get("market"), c.get("pick")
            if not market or not pick or (market, pick) in seen:
                continue
            seen.add((market, pick))
            out.append(c)

    derived = []
    for c in out:
        market = c.get("market") or ""
        sides = _group_sides(market, c.get("pick") or "")
        if not sides:
            continue
        conf = c.get("confidence")
        sibling = c.get("model_prob")
        sibling_ok = (isinstance(sibling, (int, float))
                      and not isinstance(sibling, bool) and 0.0 < sibling < 1.0)
        for side in sides:
            if (market, side) in seen:
                continue
            prob = _match_prob(res, side)
            if not isinstance(prob, (int, float)) or isinstance(prob, bool):
                # A three-way market cannot be completed by subtraction — only
                # the two-way groups can.
                if len(sides) == 2 and sibling_ok:
                    prob = 1.0 - sibling
                else:
                    continue
            if not (0.0 < prob < 1.0):
                continue
            seen.add((market, side))
            derived.append({"market": market, "pick": side,
                            "confidence": conf, "model_prob": prob,
                            "_derived": True})
    return out + derived


def _price_group(res: dict, market: str, pick: str):
    """``(prices, verdict)`` for the priceable group ``pick`` belongs to.

    ``prices`` is ``[(selection, odds), ...]`` in canonical order: the set the
    book's margin is removed over.  ``verdict`` is what :mod:`odds_guard` says
    about that group.

    The two failure modes are kept apart on purpose.  ``verdict is None`` means
    *no price exists* — the feed carries no ``odds_over15`` column, so nobody
    could quote an EV even if they wanted to.  A verdict with ``ok=False`` means
    *a price exists and was rejected* — the row is corrupted (duplicate,
    favourite mismatch, blown overround) and must be reported as unpriceable
    rather than silently dropped, because a dropped price looks identical to a
    missing one and hides the integrity problem.
    """
    if market == "1X2":
        sel = [("Home win", "odds_home"), ("Draw", "odds_draw"),
               ("Away win", "odds_away")]
    elif market == "BTTS":
        sel = [("Yes", "odds_btts_yes"), ("No", "odds_btts_no")]
    elif market == "O/U":
        m = re.match(r"(Over|Under)\s+([0-9.]+)$", (pick or "").strip())
        thr = _OU_THR.get(m.group(2)) if m else None
        if not thr:
            return [], None
        sel = [(f"Over {m.group(2)}", f"odds_over{thr}"),
               (f"Under {m.group(2)}", f"odds_under{thr}")]
    else:
        return [], None

    prices = []
    for selection, key in sel:
        v = res.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 1.0:
            prices.append((selection, float(v)))
    if pick not in {s for s, _ in prices}:
        return prices, None

    if market == "1X2":
        if len(prices) != 3:
            return prices, odds_guard.OddsVerdict(False, "incomplete 1X2 triplet")
        fb = res.get("forebet_pct")
        if not (isinstance(fb, (list, tuple)) and len(fb) == 3
                and all(isinstance(x, (int, float)) for x in fb)):
            # Without Forebet's own percentages the favourite-agreement test
            # cannot run, and that test is the only one that catches a
            # structurally plausible triplet assigned to the wrong outcomes.
            return prices, odds_guard.OddsVerdict(
                False, "no Forebet percentages to verify the price against")
        return prices, odds_guard.validate_1x2(res.get("odds_home"),
                                               res.get("odds_draw"),
                                               res.get("odds_away"),
                                               fb[0], fb[1], fb[2])

    if len(prices) != len(sel):
        return prices, odds_guard.OddsVerdict(False, "missing")
    # Name the selection's own price first so its fair probability lands on it.
    own = dict(prices)[pick]
    other = next(v for s, v in prices if s != pick)
    return prices, odds_guard.validate_ou(own, other)


def _fair_prob(prices, verdict, market: str, pick: str) -> Optional[float]:
    """De-vigged price-implied probability for ``pick``, or None."""
    if not (verdict is not None and verdict.ok and verdict.devig):
        return None
    if market == "1X2":
        idx = ("Home win", "Draw", "Away win").index(pick)
    else:
        idx = 0  # validate_ou was handed (own, other)
    if idx >= len(verdict.devig):
        return None
    return float(verdict.devig[idx])


def best_candidate(res: dict, ledger: Optional[dict],
                   primary: Optional[tuple] = None) -> Optional[dict]:
    """The best bet on this fixture across **every** market.

    ``primary`` is the ``(market, pick)`` the row shows in its Selection column.
    When a candidate *is* that pick, its probability is taken from the same
    source the Selection column uses (``_match_prob``), so the EV quoted for a
    pick can never disagree with the probability printed beside it.  Every other
    candidate uses the probability the pipeline assigned to it in ``all_picks``,
    which is what its own ranking is built on.

    "Best" is deliberately not "the model's primary pick", and it is not
    restricted to 1/X/2 — a BTTS price and a goal-line price and a 1X2 price are
    alternatives on the same match, and only one of them is the value.

    Ranking is **tier first, then money**, applied in two steps:

    0. **+EV is a precondition.**  A candidate that prices at ≤0 is not a bet
       (house rule: no bet without +EV), so it never outranks a positive one
       no matter how good its tier is.
    1. Among the candidates that price positive, the tier (measured history:
       RELIABLE over WATCH over HELD) is the first key, in both rank orders
       below.  EV is only compared *within* one tier.  This matters because EV
       here is ``model probability × price − 1``: it grows exactly when the
       model disagrees with the market, and measured history says that
       disagreement is the model's own error (on odds-valid rows the de-vigged
       market sits within ±2.5pts of the realised outcome in every disagreement
       bin while the model is off by up to 22pts).  So a +100% EV on a bucket
       that never beats its base rate ranks below a +20% EV on one that does.
    2. Inside the chosen tier the comparison is, in order of availability:
       **expected value** (calibrated model probability × a price that passed
       :mod:`odds_guard`, minus one) whenever anything prices, else **model
       lift over base rate** (``model probability − the selection's
       unconditional base rate``).  Lift is used only when *nothing* prices (no
       O/U 1.5 column in the feed, or every price failed validation) and never
       the ledger's bucket edge: the bucket edge is identical for every pick
       drawn from the same bucket, so ranking on it selects the historically
       juiciest *bucket* and says nothing about the match in front of the
       reader.

    Two outputs are deliberately *not* recommendations, and carry flags the
    cell renders as such:

    * ``no_value`` — something prices, but every priced candidate is ≤0 EV.
      The cell prints "no +EV selection" instead of presenting the least bad
      negative number as a best bet.
    * ``gap_warning`` — the candidate sits more than 10pts from the de-vigged
      market.  The cell prints the gap, because that distance is the model
      disagreeing with a price history has measured as well calibrated.

    The result carries its own ``tier`` so a reader sees how good the price is
    *and* how often this kind of pick has historically landed.  HELD results
    additionally carry ``n`` and render "no measured evidence (n=…)".

    Returns None when there is no candidate with a probability to rank.
    """
    entries = []
    for c in _candidate_universe(res):
        market, pick = c.get("market"), c.get("pick")
        conf = c.get("confidence") or ""
        if primary is not None and (market, pick) == primary:
            stated = _match_prob(res, pick)
            if not isinstance(stated, (int, float)) or isinstance(stated, bool):
                stated = c.get("model_prob")
        else:
            stated = c.get("model_prob")
            if not isinstance(stated, (int, float)) or isinstance(stated, bool):
                stated = _match_prob(res, pick)
        verdict = classify_pick(pick, conf, ledger, market)
        entry = {
            "market": market, "pick": pick, "confidence": conf,
            "stated_prob": stated if isinstance(stated, (int, float)) else None,
            "shown_prob": calibrated(stated, ledger) if isinstance(stated, (int, float)) else None,
            "tier": verdict["tier"],
            "hit_rate": verdict.get("hit_rate"),
            "n": verdict.get("n", 0),
            "expected": verdict.get("expected"),
            "shrunk_rate": verdict.get("shrunk_rate"),
            "edge_pts": verdict.get("edge_pts"),
            "reason": verdict.get("reason"),
            "odds": None, "ev_pct": None, "price_edge_pct": None,
            "fair_prob": None, "gap_warning": None, "no_value": False,
            "note": "", "odds_rejected": False,
        }
        if entry["shown_prob"] is not None and entry["expected"] is not None:
            entry["lift_pts"] = round((entry["shown_prob"] - entry["expected"]) * 100, 2)
        else:
            entry["lift_pts"] = None
        prices, odv = _price_group(res, market, pick)
        if odv is None:
            if prices:
                entry["note"] = f"no price at this threshold ({', '.join(s for s, _ in prices)} priced)"
        elif not odv.ok:
            entry["odds_rejected"] = True
            entry["note"] = f"odds failed validation ({odv.reason}) — not priced"
        elif entry["shown_prob"] is not None:
            fair = _fair_prob(prices, odv, market, pick)
            own = dict(prices).get(pick)
            if fair is not None and own:
                entry["odds"] = own
                entry["fair_prob"] = fair
                entry["ev_pct"] = round((entry["shown_prob"] * own - 1.0) * 100, 2)
                entry["price_edge_pct"] = round((entry["shown_prob"] - fair) * 100, 2)
                # A model probability far from the de-vigged market is the
                # model disagreeing with a price history measures as well
                # calibrated — that distance is error, not value, so say so
                # rather than letting the size of the EV speak for itself.
                if abs(entry["price_edge_pct"]) > MODEL_MARKET_GAP_PTS:
                    entry["gap_warning"] = (
                        f"model {_pct(entry['shown_prob'])} vs market {_pct(fair)} "
                        f"({_signed(entry['price_edge_pct'], 0)}pt) — gaps this wide "
                        "have measured as model error, not edge")
        entries.append(entry)

    if not entries:
        return None

    # Tier first, then money.  Two hard rules, in this order:
    #   * +EV is a precondition — a candidate that prices at ≤0 is not a bet,
    #     so it never outranks a positive one regardless of its tier.
    #   * among the candidates that do price positive, the tier decides
    #     (RELIABLE over WATCH over HELD) and EV only chooses *within* the
    #     winning tier.  The tier is measured history; EV is the model's
    #     opinion of a price, and history says that opinion is where the model
    #     is most wrong.
    priced = [e for e in entries if e["ev_pct"] is not None]
    if priced:
        positive = [e for e in priced if e["ev_pct"] > 0]
        if positive:
            top = max(_TIER_RANK[e["tier"]] for e in positive)
            pool = [e for e in positive if _TIER_RANK[e["tier"]] == top]
            best = max(pool, key=lambda e: (e["ev_pct"], e["shrunk_rate"] or 0.0))
            best["ranked_by"] = "EV"
            return best
        # Nothing on this fixture prices positive.  Rank the least-bad
        # candidate so the cell can still show what was available, but flag
        # it: a negative EV printed under a "Best bet" heading reads as a
        # recommendation, which is the one thing it must not.
        best = max(priced, key=lambda e: (e["ev_pct"], _TIER_RANK[e["tier"]],
                                          e["shrunk_rate"] or 0.0))
        best["ranked_by"] = "EV"
        best["no_value"] = True
        return best

    # Nothing prices.  Distinguish the two ways that can happen: every price in
    # the feed failed odds_guard (a real integrity problem the reader should
    # know about), versus the feed simply carries no column for these lines.
    rejected = any(e["odds_rejected"] for e in entries)
    # Rank by this fixture's lift over the selection's base rate, never by the
    # bucket edge: that number is one per bucket, so choosing on it selects the
    # historically juiciest *bucket* and says nothing about the match in front
    # of the reader.  Same tier-first order as the priced path — lift is also a
    # model number, so it only ever ranks inside one tier.
    measured = [e for e in entries if e["lift_pts"] is not None]
    if not measured:
        measured = [e for e in entries if e["shown_prob"] is not None]
    if not measured:
        return None
    best = max(measured, key=lambda e: (_TIER_RANK[e["tier"]],
                                        e["lift_pts"] if e["lift_pts"] is not None else -99.0,
                                        e["shrunk_rate"] or 0.0))
    best["ranked_by"] = "model lift"
    if not best["note"] and rejected:
        best["note"] = "stored prices failed odds_guard — not priced"
    return best


def _fixture_id(home: str, away: str, kickoff: str) -> Optional[str]:
    """Identity for de-duplicating a slate, or None when it cannot be told apart.

    The same fixture can be predicted twice in one run (its URL appears with
    and without percent-encoding), which printed it twice in the report --
    Real Zaragoza v CD Teruel appeared as two identical RELIABLE rows.
    ``audit_reliability.fixture_key`` does the same job in history using the
    Forebet match id, which a result dict does not carry.

    Returns None when the fixture cannot be identified, because the alternative
    is worse than the bug being fixed: rows missing a team name or a kickoff
    would all collapse into a single "fixture" and the report would show one
    match instead of many.
    """
    h, a = (home or "").strip().casefold(), (away or "").strip().casefold()
    if not h or not a or h == "?" or a == "?":
        return None
    return "|".join([h, a, (kickoff or "").strip()])


def build_rows(results: Iterable[dict], ledger: Optional[dict]) -> List[dict]:
    """Turn prediction result dicts into report rows, tiered by reliability."""
    rows = []
    seen_fixtures: set = set()
    for res in results:
        pick = res.get("pick") or "—"
        market = res.get("market") or "—"
        conf = res.get("confidence") or "Unknown"
        home = res.get("home") or res.get("home_team") or "?"
        away = res.get("away") or res.get("away_team") or "?"
        kickoff = res.get("date") or res.get("match_date", "")

        verdict = classify_pick(pick, conf, ledger, market)
        stated = _match_prob(res, pick)
        shown = calibrated(stated, ledger)

        # Reserve / second / youth sides are suppressed regardless of what the
        # ledger says. The gate is measured on senior-team football; a reserve
        # fixture reuses the senior side's strength inputs, so a publish verdict
        # there describes a match the model never actually priced.
        reserve = reserve_reason(home, away)
        if reserve:
            team, side = reserve
            verdict = dict(verdict)
            verdict["tier"] = TIER_HELD
            verdict["reason"] = (
                f"{side} side {team!r} is a reserve / second team — the model's "
                f"inputs describe the senior side, not who is playing")

        fid = _fixture_id(home, away, kickoff)
        if fid is not None:
            if fid in seen_fixtures:
                continue
            seen_fixtures.add(fid)

        # Candidate list repeats the primary pick at index 0; skip anything
        # that is not a genuinely different selection.  Drawn from the full
        # universe rather than the synthesis shortlist so a goal line or a BTTS
        # call the shortlist did not lead with is still counted as an option.
        alternatives = []
        for c in _candidate_universe(res):
            alt_pick = c.get("pick") or ""
            alt_market = c.get("market") or market
            if not alt_pick or (alt_pick == pick and alt_market == market):
                continue
            alt = classify_pick(alt_pick, c.get("confidence") or "", ledger, alt_market)
            if reserve:
                alt["tier"] = TIER_HELD
            alternatives.append({"pick": alt_pick,
                                 "confidence": c.get("confidence") or "",
                                 "market": alt_market,
                                 "tier": alt["tier"]})
        # Promote what clears the gate before truncating, so the three shown
        # are not simply the first three considered.
        alternatives.sort(key=lambda a: -_TIER_RANK[a["tier"]])

        # The fixture's recommendation, ranked across markets by EV (or by the
        # model's lift over base rate when nothing prices).  A reserve fixture
        # never gets one: the measured rates behind the tier describe
        # senior-team football.
        best = best_candidate(res, ledger, primary=(market, pick))
        if best is not None:
            best = dict(best)
            best["is_primary"] = (best["pick"] == pick
                                  and best["market"] == market)
            if reserve:
                best["tier"] = TIER_HELD

        rows.append({
            # The pipeline's result dicts use `home`/`away`/`date`; the DB rows
            # use the longer names, so accept both.
            "home": home,
            "away": away,
            "league": res.get("league", ""),
            "kickoff": kickoff,
            "time": res.get("time", ""),
            "market": market,
            "pick": pick,
            "confidence": conf,
            "stated_prob": stated,
            "shown_prob": shown,
            "score_lean": res.get("score_lean", ""),
            "verdict": verdict,
            "tier": verdict["tier"],
            "reserve_side": reserve,
            "red_flags": [_flag_text(f) for f in (res.get("_red_flags") or [])],
            "warnings": [_flag_text(f) for f in (res.get("_warnings") or [])],
            "best": best,
            "alternatives": alternatives[:3],
        })
    order = {TIER_RELIABLE: 0, TIER_WATCH: 1, TIER_HELD: 2}
    rows.sort(key=lambda r: (order[r["tier"]],
                             -(r["shown_prob"] or 0),
                             -(r["verdict"]["shrunk_rate"] or 0)))
    return rows


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

_CSS = """
:root{--bg:#0b1220;--panel:#111c2e;--line:#1e293b;--fg:#e2e8f0;--dim:#94a3b8;
--reliable:#22c55e;--watch:#f59e0b;--held:#64748b;--accent:#60a5fa;--neg:#f87171;}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
background:var(--bg);color:var(--fg);padding:24px;line-height:1.5}
.wrap{max-width:1400px;margin:0 auto}
h1{font-size:1.5rem;margin-bottom:2px}
h2{font-size:1.05rem;margin:28px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}
.sub{color:var(--dim);font-size:.85rem;margin-bottom:18px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
.card .k{font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--dim)}
.card .v{font-size:1.7rem;font-weight:600;margin-top:4px}
.card .n{font-size:.75rem;color:var(--dim);margin-top:2px}
.v.reliable{color:var(--reliable)} .v.watch{color:var(--watch)} .v.held{color:var(--held)}
table{width:100%;border-collapse:collapse;background:var(--panel);
border:1px solid var(--line);border-radius:10px;overflow:hidden;font-size:.86rem}
th{background:#0f1a2b;color:var(--dim);font-weight:500;text-align:left;
padding:9px 10px;font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;
position:sticky;top:0}
td{padding:9px 10px;border-top:1px solid var(--line);vertical-align:top}
tr:hover td{background:#16233a}
.badge{display:inline-block;padding:2px 8px;border-radius:99px;font-size:.68rem;
font-weight:600;letter-spacing:.04em}
.b-reliable{background:rgba(34,197,94,.15);color:var(--reliable);border:1px solid rgba(34,197,94,.4)}
.b-watch{background:rgba(245,158,11,.15);color:var(--watch);border:1px solid rgba(245,158,11,.4)}
.b-held{background:rgba(100,116,139,.15);color:var(--held);border:1px solid rgba(100,116,139,.4)}
.mono{font-variant-numeric:tabular-nums;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.pos{color:var(--reliable)} .neg{color:#ef4444}
.muted{color:var(--dim)} .tiny{font-size:.75rem}
.bar{height:5px;background:#1e293b;border-radius:3px;overflow:hidden;min-width:52px}
.bar>i{display:block;height:100%;border-radius:3px}
.note{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--accent);
border-radius:8px;padding:12px 14px;font-size:.82rem;color:var(--dim);margin:14px 0}
.note b{color:var(--fg)}
.flag{display:inline-block;background:rgba(239,68,68,.14);color:#fca5a5;
border:1px solid rgba(239,68,68,.35);border-radius:5px;padding:1px 6px;
font-size:.68rem;margin:2px 3px 0 0}
details{margin-top:10px} summary{cursor:pointer;color:var(--accent);font-size:.85rem;
padding:6px 0;user-select:none}
.foot{margin-top:26px;padding-top:14px;border-top:1px solid var(--line);
color:var(--dim);font-size:.75rem}
.scroll{max-height:none;overflow:visible}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:.8rem;color:var(--dim);margin:10px 0}
"""


def _pct(x: Optional[float], digits: int = 1) -> str:
    return "—" if x is None else f"{x*100:.{digits}f}%"


def _pts(x: Optional[float]) -> str:
    if x is None:
        return '<span class="muted">—</span>'
    cls = "pos" if x > 0 else "neg" if x < 0 else "muted"
    return f'<span class="{cls}">{x:+.1f} pts</span>'


def _signed(x: Optional[float], digits: int = 1) -> str:
    """Signed percentage for an EV or price edge, or an em dash when absent."""
    if x is None:
        return '<span class="muted">—</span>'
    cls = "pos" if x > 0 else "neg" if x < 0 else "muted"
    return f'<span class="{cls}">{x:+.{digits}f}%</span>'


def _bar(rate: Optional[float], expected: Optional[float]) -> str:
    if rate is None:
        return ""
    pct = min(100.0, rate * 100)
    colour = "var(--reliable)" if (expected is not None and rate > expected) else "var(--watch)"
    return (f'<div class="bar" title="hit rate vs {expected*100:.0f}% base rate">'
            f'<i style="width:{pct:.0f}%;background:{colour}"></i></div>')


def render_html(rows: List[dict], ledger: Optional[dict], meta: Optional[dict] = None) -> str:
    """Render the whole report as one self-contained HTML document."""
    meta = meta or {}
    e = html.escape

    reliable = [r for r in rows if r["tier"] == TIER_RELIABLE]
    watch = [r for r in rows if r["tier"] == TIER_WATCH]
    held = [r for r in rows if r["tier"] == TIER_HELD]

    # Expected hit rate of the published card, weighted by measured history.
    num = den = 0.0
    for r in reliable:
        sr = r["verdict"]["shrunk_rate"]
        n = r["verdict"]["n"]
        if sr:
            num += sr * n
            den += n
    card_rate = (num / den) if den else None
    card_base = None
    if den:
        b = den = 0.0
        for r in reliable:
            exp = r["verdict"]["expected"]
            n = r["verdict"]["n"]
            if exp:
                b += exp * n
                den += n
        card_base = (b / den) if den else None

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    stale = ledger_is_stale(ledger)
    gen = (ledger or {}).get("generated_at", "never")
    res_rows = (ledger or {}).get("resolved_rows", 0)
    calib = (ledger or {}).get("calibration") or {}
    odds = (ledger or {}).get("odds_integrity") or {}
    odds_ok = odds.get("ok", 0)
    odds_total = sum(v for k, v in odds.items() if k != "ok") + odds_ok
    agree = (ledger or {}).get("odds_favourite_agreement")
    base_meta = ((ledger or {}).get("policy") or {}).get("base_rates") or {}
    base_detail = base_meta.get("detail") or {}

    parts: List[str] = []
    A = parts.append

    A('<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">')
    A('<meta name="viewport" content="width=device-width,initial-scale=1.0">')
    A(f'<title>Reliable Picks — {e(meta.get("label", "run"))}</title>')
    A(f"<style>{_CSS}</style></head><body><div class='wrap'>")

    A("<h1>Reliable Picks</h1>")
    A(f'<p class="sub">Generated {stamp}'
      f'{" &middot; " + e(str(meta.get("source", ""))) if meta.get("source") else ""}'
      f' &middot; reliability ledger built {e(str(gen))} from {res_rows:,} resolved matches</p>')

    if stale:
        age = (ledger or {}).get("_age_days")
        A(f'<div class="note"><b>Ledger is stale</b> ({e(str(age))} days old). '
          'Rerun <code>.venv/bin/python scripts/audit_reliability.py</code> before trusting these tiers.</div>')
    if not ledger:
        A('<div class="note"><b>No reliability ledger.</b> Run '
          '<code>.venv/bin/python scripts/audit_reliability.py</code>. Until then every pick is '
          'marked HELD — an unmeasured pick cannot be called reliable.</div>')

    # ── Headline numbers ──
    A('<div class="cards">')
    A(f'<div class="card"><div class="k">Reliable</div>'
      f'<div class="v reliable">{len(reliable)}</div>'
      f'<div class="n">of {len(rows)} matches cleared the gate</div></div>')
    A(f'<div class="card"><div class="k">Watch</div>'
      f'<div class="v watch">{len(watch)}</div>'
      f'<div class="n">measurable, no edge over base rate</div></div>')
    A(f'<div class="card"><div class="k">Held</div>'
      f'<div class="v held">{len(held)}</div>'
      f'<div class="n">suppressed — no proven edge</div></div>')
    A(f'<div class="card"><div class="k">Card hit rate</div>'
      f'<div class="v">{_pct(card_rate)}</div>'
      f'<div class="n">measured history{" vs " + _pct(card_base) + " base" if card_base else ""}</div></div>')
    mult = calib.get("global_multiplier")
    A(f'<div class="card"><div class="k">Calibration</div>'
      f'<div class="v">{_pct(calib.get("mean_realised"))}</div>'
      f'<div class="n">realised vs {_pct(calib.get("mean_stated"))} stated'
      f'{" (×%.3f)" % mult if mult else ""}</div></div>')
    A(f'<div class="card"><div class="k">Usable odds</div>'
      f'<div class="v">{_pct((odds_ok/odds_total) if odds_total else None, 0)}</div>'
      f'<div class="n">{"stored in history.db — see note" if (odds_total and odds_ok < odds_total*0.5) else "of stored prices"}</div></div>')
    A("</div>")

    A('<div class="legend">'
      '<span><span class="badge b-reliable">RELIABLE</span> beaten base rate on '
      f'&ge;{(ledger or {}).get("policy",{}).get("min_samples","—")} historical matches</span>'
      '<span><span class="badge b-watch">WATCH</span> measured, but no edge</span>'
      '<span><span class="badge b-held">HELD</span> not recommended</span></div>')
    A('<div class="note"><b>Best bet is ranked across markets, not handed down '
      'from the model&rsquo;s pick.</b> For each fixture it takes every selection '
      'the feed prices &mdash; 1/X/2, both goal lines, both BTTS outcomes &mdash; '
      'and ranks them in two steps. <b>Only positive expected value qualifies</b> '
      '(<i>calibrated model probability &times; validated price &minus; 1</i>); '
      'a price is used only if it passes <code>odds_guard</code>, a rejected '
      'price is shown as rejected, never as a number. <b>Then the tier decides, '
      'not the size of the EV</b> &mdash; RELIABLE beats WATCH beats HELD, and '
      'EV only chooses within one tier, because EV grows exactly when the model '
      'disagrees with the market and measured history says that disagreement is '
      'the model&rsquo;s own error. Where nothing prices &mdash; no column for '
      'the line, or every price rejected &mdash; the ranking falls back to '
      '<b>model lift over base rate</b>: this match&rsquo;s calibrated '
      'probability minus what that selection lands on blind. Lift is '
      'deliberately per-match; the ledger&rsquo;s bucket edge could not be used '
      'here because it is one number per bucket, so ranking on it would name the '
      'historically juiciest <i>bucket</i> rather than the best fixture (see the '
      'bucket-edge note below). Two cell states are warnings, not picks: '
      '<b style="color:var(--neg)">no +EV selection</b> means every price on the '
      'fixture came back at or below break-even (the least-bad price is shown '
      'and marked <i>pass</i>), and a HELD badge carries '
      '<i>no measured evidence (n=…)</i> naming the sample the pick would be '
      'judged on. A best bet sitting more than 10pts from the de-vigged market '
      'prints that gap alongside its EV.</div>')

    # The edge column is the most misreadable number on the page: it is a
    # property of the (market, selection, confidence) bucket, so every pick in a
    # bucket shows the same value and sorting by it ranks buckets, not matches.
    A('<div class="note"><b>Read &ldquo;bucket edge&rdquo; as a property of the '
      'bucket, not of the match.</b> It is the historical edge of that '
      '<i>(market, selection, confidence)</i> combination over its own base rate, '
      'and every pick drawn from the same bucket shows an identical figure. '
      'Ranking a slate by it ranks the buckets and says nothing about which '
      'fixture is best &mdash; rank by <b>model prob</b>, which is the only '
      'per-match number here.</div>')

    if (odds_total and agree is not None and agree < 0.6):
        A(f'<div class="note"><b>Where the expected values come from &mdash; '
          f'not from stored odds.</b> Of the {odds_total:,} prices stored in '
          f'history.db only {odds_ok:,} ({_pct(odds_ok/odds_total,0)}) pass integrity checks, and '
          'the stored 1X2 favourite agrees with Forebet&rsquo;s independent percentage only '
          f'{_pct(agree)} of the time, so no ROI or EV on this page is derived from them. Every '
          'fixture is priced instead from its own live feed prices, validated one by one: an EV '
          'is printed only where <code>odds_guard</code> accepted the price, and where nothing '
          'did, the best-bet column falls back to model lift over base rate and says so. Measured '
          '<i>hit rates</i> remain the only number the history itself supports.</div>')

    # ── Reliable card ──
    A("<h2>Recommended card</h2>")
    if reliable:
        A(_table(reliable))
    else:
        A('<div class="note">Nothing cleared the gate in this run. That is a legitimate '
          'result: the historical ledger does not currently support any of these selections '
          'beating its base rate. Do not force picks.</div>')

    # ── Watch list ──
    if watch:
        A("<h2>Watch list — measured, but no proven edge</h2>")
        A(_table(watch))

    # ── Held ──
    if held and meta.get("show_held", True):
        A("<h2>Suppressed</h2>")
        A(_table(held, compact=True))

    # ── Calibration appendix ──
    curve = calib.get("curve") or []
    if curve:
        A("<h2>Calibration</h2>")
        A('<details open><summary>Model&rsquo;s stated probability vs what actually happened</summary>')
        A("<table><tr><th>Stated bucket</th><th class='mono'>n</th>"
          "<th>Stated</th><th>Realised</th><th>Gap</th><th>Applied multiplier</th></tr>")
        for c in curve:
            gap = c["gap"]
            cls = "pos" if gap > 0 else "neg" if gap < 0 else "muted"
            A(f"<tr><td class='mono'>{e(c['bucket'])}</td><td class='mono'>{c['n']:,}</td>"
              f"<td class='mono'>{_pct(c['stated'])}</td>"
              f"<td class='mono'>{_pct(c['realised'])}</td>"
              f"<td class='mono'><span class='{cls}'>{gap*100:+.1f} pts</span></td>"
              f"<td class='mono'>&times;{c['multiplier']:.3f}</td></tr>")
        A("</table><p class='tiny muted' style='padding:8px 2px'>A negative gap means the model "
          "overstated its confidence. Every probability in this report has been multiplied down "
          "by the factor shown.</p></details>")

    # ── Base rates appendix ──
    # The yardstick every tier is judged against. Shown in full so a tier can
    # be checked by hand rather than taken on trust.
    if base_detail:
        pop = base_meta.get("population_n") or 0
        win = base_meta.get("window_days") or 0
        scope = f"last {win} days" if win else "all time"
        A("<h2>Base rates</h2>")
        A(f"<details open><summary>The yardstick: how often each selection lands "
          f"with no model at all ({pop:,} resolved matches, {scope})</summary>")
        A("<table><tr><th>Selection</th><th>Base rate</th><th class='mono'>Hits</th>"
          "<th>Source</th></tr>")
        for sel, v in sorted(base_detail.items(), key=lambda kv: -kv[1]["rate"]):
            src = v.get("source", "")
            note = "" if src == "measured" else f"<div class='tiny muted'>{e(src)}</div>"
            A(f"<tr><td class='mono'>{e(sel)}</td>"
              f"<td class='mono'>{_pct(v['rate'])}</td>"
              f"<td class='mono'>{v.get('hits', 0):,}/{v.get('n', 0):,}</td>"
              f"<td>{note or 'measured'}</td></tr>")
        A("</table><p class='tiny muted' style='padding:8px 2px'>A pick is only "
          "recommended when its measured hit rate beats this number. &ldquo;Over 1.5 "
          "lands 88% of the time&rdquo; is not an edge &mdash; Over 1.5 lands 77% of the "
          "time with no model involved. Rates are measured over <em>all</em> resolved "
          "matches, not just the ones the model predicted, so the model is never "
          "graded against its own habits.</p>")
        chk = base_meta.get("self_check")
        if chk and chk != "ok":
            A(f"<div class='note'><b>Base-rate self-check failed</b> {e(str(chk))}</div>")
        A("</details>")

    # ── Odds integrity appendix ──
    if odds:
        A("<h2>Data integrity</h2>")
        A("<details><summary>Odds validation of stored history</summary><table>")
        A("<tr><th>Verdict</th><th class='mono'>Rows</th><th>Meaning</th></tr>")
        meaning = {
            "ok": "passed all checks",
            "duplicate": "two prices identical — legacy parser read the predicted-outcome price twice",
            "favourite_mismatch": "cheapest price is not the outcome Forebet ranks first",
            "overround": "implied margin outside a plausible 1-8%",
            "range": "price outside plausible bounds",
            "missing": "no price stored",
        }
        for reason, count in sorted(odds.items(), key=lambda kv: -kv[1]):
            A(f"<tr><td class='mono'>{e(reason)}</td><td class='mono'>{count:,}</td>"
              f"<td class='tiny muted'>{e(meaning.get(reason, ''))}</td></tr>")
        A("</table><p class='tiny muted' style='padding:8px 2px'>Rejected prices are treated as "
          "missing by the model, so no tree can fit them. Forebet&rsquo;s own percentages are "
          "unaffected and remain the most reliable single signal in the data.</p></details>")

    # ── Gate appendix ──
    gate = (ledger or {}).get("gate") or {}
    if gate:
        A("<h2>Why the gate looks like this</h2>")
        A("<details><summary>Every (selection, confidence) bucket and its verdict</summary>")
        A("<table><tr><th>Selection</th><th>Confidence</th><th class='mono'>n</th>"
          "<th>Hit rate</th><th>Base rate</th><th>Shrunk</th><th>Edge</th>"
          "<th>95% lower bound</th><th>Decision</th></tr>")
        for key in sorted(gate, key=lambda k: (not gate[k]["publish"],
                                              -(gate[k]["shrunk_rate"] or 0))):
            g = gate[key]
            pick, _, conf = key.partition("|")
            badge = ("b-reliable" if g["publish"]
                     else "b-watch" if g["n"] >= 200 else "b-held")
            wl = g.get("wilson_lower")
            wl_cell = (f"<td class='mono'>{_pct(wl)}</td>" if wl is not None
                       else "<td class='mono muted'>&mdash;</td>")
            A(f"<tr><td>{e(pick)}</td><td class='tiny muted'>{e(conf)}</td>"
              f"<td class='mono'>{g['n']:,}</td>"
              f"<td class='mono'>{_pct(g['hit_rate'])}</td>"
              f"<td class='mono muted'>{_pct(g['expected'])}</td>"
              f"<td class='mono'>{_pct(g['shrunk_rate'])}</td>"
              f"<td class='mono'>{_pts(g['edge_pts'])}</td>"
              f"{wl_cell}"
              f"<td><span class='badge {badge}'>{'PUBLISH' if g['publish'] else 'HOLD'}</span>"
              f"<div class='tiny muted'>{e(g['reason'])}</div></td></tr>")
        A("</table><p class='tiny muted' style='padding:8px 2px'>The 95% lower "
          "bound is the bottom of a one-sided confidence interval on the hit rate. "
          "A bucket only publishes when that bound clears the base rate, so a "
          "thin edge measured on a small sample cannot pass on luck alone &mdash; "
          "which matters when 54 buckets are tested at once.</p></details>")

    src = meta.get("source", "")
    A(f'<div class="foot">One report, gated on measured history. '
      f'Ledger: <code>data/reliability.json</code>'
      f'{" &middot; source: " + e(str(src)) if src else ""}. '
      'Rebuild with <code>.venv/bin/python scripts/audit_reliability.py</code>.</div>')

    A("</div></body></html>")
    return "".join(parts)


def _table(rows: List[dict], compact: bool = False) -> str:
    e = html.escape
    out = ["<table><tr><th>Match</th><th>League</th><th>Selection</th>",
           "<th title='Best selection on this fixture across every market the feed prices — tier first (RELIABLE over WATCH over HELD), then expected value within that tier, and only if the EV is positive; a fixture where nothing prices positive says \"no +EV selection\" instead. Not restricted to 1/X/2 — a BTTS or goal-line price can be the value even when the model leads with something else. Shown on suppressed rows too: a fixture whose primary is HELD can still carry a selection that clears the gate.'>Best bet</th>",
           "<th>Conf.</th><th>Model prob</th><th>Measured hit rate</th>",
           "<th>Base</th><th title='The historical edge of this BUCKET over its base rate. It is identical for every pick in the same bucket, so it says nothing about how good this particular match is — rank by model probability.'>Bucket edge</th><th>Tier</th>"]
    if not compact:
        out.append("<th>Notes</th>")
    out.append("</tr>")

    for r in rows:
        v = r["verdict"]
        badge = {"RELIABLE": "b-reliable", "WATCH": "b-watch", "HELD": "b-held"}[r["tier"]]
        shown = _pct(r["shown_prob"])
        stated = _pct(r["stated_prob"])
        prob_cell = (f"<b>{shown}</b>"
                     + (f"<div class='tiny muted'>raw {stated}</div>"
                        if r["stated_prob"] and abs((r["shown_prob"] or 0) - (r["stated_prob"] or 0)) > 0.005
                        else ""))
        rate_cell = (f"<b>{_pct(v['hit_rate'])}</b>{_bar(v['hit_rate'], v['expected'])}"
                     f"<div class='tiny muted'>{v['n']:,} matches</div>"
                     if v["hit_rate"] is not None else "<span class='muted'>no history</span>")
        notes = ""
        if not compact:
            bits = []
            if r.get("reserve_side"):
                team, side = r["reserve_side"]
                bits.append(f"<span class='flag'>reserve side: {e(str(side))} "
                            f"{e(str(team))}</span>")
            if r["score_lean"]:
                bits.append(f"lean {e(r['score_lean'])}")
            for flag in (r["red_flags"] or [])[:2]:
                txt = e(flag[:110])
                bits.append(f"<span class='flag'>{txt}</span>")
            alts = [a for a in r["alternatives"] if a["tier"] == TIER_RELIABLE]
            if alts:
                bits.append("also clears gate: "
                            + e(", ".join(_pick_text(a) for a in alts)))
            notes = ("<div class='tiny muted'>" + " &middot; ".join(bits) + "</div>"
                     if bits else "")
        ko = " ".join(x for x in (r.get("kickoff"), r.get("time")) if x)
        out.append(
            f"<tr><td><b>{e(r['home'])}</b> v {e(r['away'])}"
            + (f"<div class='tiny muted'>{e(ko)}</div>" if ko else "") + "</td>"
            f"<td class='tiny muted'>{e(str(r['league']))}</td>"
            f"<td><b>{e(r['pick'])}</b><div class='tiny muted'>{e(r['market'])}</div></td>"
            f"<td>{_best_cell(r.get('best'), e)}</td>"
            f"<td class='tiny'>{e(r['confidence'])}</td>"
            f"<td class='mono'>{prob_cell}</td>"
            f"<td class='mono'>{rate_cell}</td>"
            f"<td class='mono muted'>{_pct(v['expected'])}</td>"
            f"<td class='mono'>{_pts(v['edge_pts'])}</td>"
            f"<td><span class='badge {badge}'>{r['tier']}</span>"
            f"<div class='tiny muted'>{e(v['reason'])}</div></td>"
            f"{'<td>' + notes + '</td>' if not compact else ''}</tr>")
    out.append("</table>")
    return "".join(out)


def _best_cell(best: Optional[dict], esc) -> str:
    """The fixture's Best bet column: selection, price, EV and its own tier.

    Three states are printed as such rather than as a recommendation:

    * ``no_value`` — nothing on the fixture prices positive, so the cell leads
      with "no +EV selection" and shows the least-bad price underneath.  A
      negative EV under a *Best bet* heading reads as a bet, which is the one
      thing it must not.
    * HELD tier — the selection has no measured history backing it, so the
      sample it would be judged on is named inline ("no measured evidence
      (n=…)") instead of leaving the badge to carry the whole warning.
    * ``gap_warning`` — the model sits far from the de-vigged market; the gap
      is printed beside the EV so the size of the number cannot be mistaken
      for confidence.
    """
    if not best:
        return "<span class='muted'>no candidate to rank</span>"
    badge = {"RELIABLE": "b-reliable", "WATCH": "b-watch",
             "HELD": "b-held"}[best["tier"]]
    conf = best.get("confidence") or ""
    where = (esc(best["market"]) + (f" · {esc(conf)}" if conf else ""))

    if best.get("no_value"):
        parts = [f"<b style='color:var(--neg)'>no +EV selection</b>",
                 f"<div class='tiny muted'>{esc(best['pick'])} · {where}</div>"]
        if best.get("odds") is not None:
            parts.append(f"<div class='mono tiny'>best available @ {best['odds']:.2f} · "
                         f"EV {_signed(best.get('ev_pct'))} — pass</div>")
        parts.append(f"<div><span class='badge {badge}'>{best['tier']}</span></div>")
        return "".join(parts)

    parts = [f"<b>{esc(best['pick'])}</b>", f"<div class='tiny muted'>{where}</div>"]
    if best.get("odds") is not None:
        price = (f"<div class='mono tiny'>@ {best['odds']:.2f} · EV "
                 f"{_signed(best.get('ev_pct'))}</div>")
        parts.append(price)
    elif best.get("note"):
        parts.append(f"<div class='tiny muted'>{esc(best['note'])}</div>")
    parts.append(f"<div><span class='badge {badge}'>{best['tier']}</span></div>")
    if best["tier"] == TIER_HELD:
        n = best.get("n") or 0
        ev_txt = f"no measured evidence (n={n})" if n else "no measured history"
        parts.append(f"<div class='tiny muted'>{ev_txt}</div>")
    if best.get("gap_warning"):
        parts.append(f"<div class='tiny muted'>{esc(best['gap_warning'])}</div>")
    if not best.get("is_primary"):
        parts.append("<div class='tiny' style='color:var(--accent)'>"
                     "value sits outside the model pick</div>")
    if best.get("ranked_by") == "model lift":
        if best.get("lift_pts") is not None:
            parts.append(f"<div class='mono tiny'>lift {_pts(best['lift_pts'])}</div>")
        parts.append("<div class='tiny muted'>ranked by model lift over base rate</div>")
    elif best.get("ranked_by") and best["ranked_by"] != "EV":
        parts.append(f"<div class='tiny muted'>ranked by {esc(best['ranked_by'])}</div>")
    return "".join(parts)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def write_report(results: Iterable[dict], out_path: Path, label: str = "run",
                 source: str = "", show_held: bool = True,
                 auto_rebuild: bool = True) -> Path:
    """Build the ledger if needed, classify the results, write the HTML.

    ``results`` is the list of prediction dicts the pipeline already produces.
    Returns the written path.
    """
    ledger = load_ledger()
    stale = ledger_is_stale(ledger)
    if (not ledger or stale) and auto_rebuild:
        rebuild_ledger()
        ledger = load_ledger()

    rows = build_rows(results, ledger)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_html(rows, ledger, {
        "label": label, "source": source, "show_held": show_held,
    }))
    write_slip(rows, out_path.parent)
    write_index(out_path.parent)
    return out_path


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Predictions</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       background:#0f172a; color:#e2e8f0; padding:20px; }
h1 { font-size:1.4rem; margin-bottom:4px; }
.sub { color:#94a3b8; font-size:0.85rem; margin-bottom:22px; max-width:62ch; line-height:1.5; }
.card { display:block; max-width:34rem; padding:16px 18px; border:1px solid #334155;
        border-left:4px solid #22c55e; border-radius:8px; text-decoration:none;
        color:inherit; background:#1e293b; }
.card:hover { border-color:#22c55e; }
.card b { display:block; font-size:1.05rem; margin-bottom:5px; color:#f1f5f9; }
.card span { color:#94a3b8; font-size:0.82rem; line-height:1.5; display:block; }
code { background:#1e293b; border:1px solid #334155; border-radius:4px;
       padding:1px 5px; font-size:0.85em; }
</style>
</head>
<body>
<h1>&#9917; Predictions</h1>
<p class="sub">Every prediction run writes three files: <code>picks.html</code>
(the gated report), the full card view (<code>latest.html</code>, or
<code>&lt;Team&gt;.html</code> for a single match) and <code>bettor.html</code>
(the staking card). picks.html is gated on measured historical performance, not
on the model's own stated probability: every pick is shown with how often that
exact selection has actually landed, and picks with no edge over their base rate
are not recommended.</p>

<a class="card" href="picks.html">
  <b>picks.html &mdash; the report</b>
  <span>RELIABLE / WATCH / HELD tiers, calibration-adjusted probabilities,
  measured hit rate vs base rate per pick, plus a best bet ranked across
  markets by validated EV.</span>
</a>

<a class="card" href="latest.html" style="margin-top:12px;">
  <b>latest.html &mdash; the full card view</b>
  <span>Every fixture as a card with confidence, reasoning, filters and the
  model's stated probabilities (single-match runs write this as
  &lt;Team&gt;.html).</span>
</a>

<a class="card" href="bettor.html" style="margin-top:12px;">
  <b>bettor.html &mdash; the staking card</b>
  <span>De-vigged edge, EV% and Quarter-Kelly stakes scaled by league
  reliability.</span>
</a>

<a class="card" href="slip.html" style="margin-top:12px;">
  <b>slip.html &mdash; consensus slip</b>
  <span>Only picks where picks.html (RELIABLE gate) and latest.html
  (model synthesis) independently agree on the same market and side.</span>
</a>

<p class="sub" style="margin-top:22px">The remaining per-angle views
(<code>best</code>, <code>consistent</code>, <code>high80</code>,
<code>bankers</code>, <code>batch_NNN</code>) each reprint the model's stated
probability with no statement of how often that kind of pick actually lands.
They are not written by default &mdash; regenerate them with
<code>pr --all-reports links.txt</code> if you want them.</p>
</body>
</html>
"""


def write_index(pred_dir: Path) -> Path:
    """Point ``predictions/index.html`` at the one report that matters.

    predictions/ is gitignored, so this is generated alongside the report
    rather than committed -- a fresh clone gets it on its first run.
    """
    p = pred_dir / "index.html"
    p.write_text(INDEX_HTML)
    return p


SLIP_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Consensus Slip</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#0b1220; color:#e2e8f0; padding:24px; }
.wrap { max-width:900px; margin:0 auto; }
h1 { font-size:1.5rem; margin-bottom:4px; }
.sub { color:#94a3b8; font-size:.85rem; margin-bottom:20px; }
.legend { display:flex; gap:16px; flex-wrap:wrap; font-size:.8rem; color:#94a3b8; margin-bottom:16px; }
.badge { display:inline-block; padding:2px 8px; border-radius:99px; font-size:.68rem; font-weight:600; letter-spacing:.04em; }
.b-rel { background:rgba(34,197,94,.15); color:#22c55e; border:1px solid rgba(34,197,94,.4); }
.b-agree { background:rgba(96,165,250,.15); color:#60a5fa; border:1px solid rgba(96,165,250,.4); }
.note { background:#111c2e; border:1px solid #1e293b; border-left:3px solid #60a5fa; border-radius:8px; padding:14px 16px; font-size:.84rem; color:#94a3b8; margin-bottom:18px; }
.note b { color:#e2e8f0; }
table { width:100%; border-collapse:collapse; background:#111c2e; border:1px solid #1e293b; border-radius:10px; overflow:hidden; font-size:.88rem; }
th { background:#0f1a2b; color:#94a3b8; font-weight:500; text-align:left; padding:10px 12px; font-size:.72rem; text-transform:uppercase; letter-spacing:.05em; }
td { padding:10px 12px; border-top:1px solid #1e293b; }
tr:hover td { background:#16233a; }
.pick { color:#22c55e; font-weight:700; }
.odds { color:#facc15; font-weight:700; font-variant-numeric:tabular-nums; }
.lift { color:#60a5fa; font-weight:600; font-variant-numeric:tabular-nums; }
.prob { font-variant-numeric:tabular-nums; }
.conf { display:inline-block; padding:2px 8px; border-radius:4px; font-size:.72rem; font-weight:600; color:#fff; }
.c-mh { background:#eab308; } .c-m { background:#f97316; } .c-h { background:#3b82f6; }
.foot { margin-top:24px; padding-top:14px; border-top:1px solid #1e293b; color:#64748b; font-size:.75rem; }
.empty { background:#111c2e; border:1px solid #1e293b; border-radius:8px; padding:24px; text-align:center; color:#94a3b8; }
</style>
</head>
<body>
<div class="wrap">
<h1>&#127919; Consensus Slip</h1>
<p class="sub">Only picks where <b>picks.html</b> (RELIABLE gate) and <b>latest.html</b> (model synthesis) independently agree on the same market and side</p>
<div class="legend">
  <span><span class="badge b-rel">RELIABLE</span> &ge;200 historical matches, beats base rate by &ge;2pt</span>
  <span><span class="badge b-agree">CONSENSUS</span> both reports pick same market + side</span>
</div>
<div class="note"><b>What counts as agreement:</b> the gated report (<code>picks.html</code>) and the raw model report (<code>latest.html</code>) must independently land on the <b>same market and same selection</b> for the same fixture. Only RELIABLE-tier picks are included.</div>
__TABLE__
<div class="foot">Generated from picks.html RELIABLE tier where the best bet matches the model's primary pick. Model probabilities are calibration-adjusted. Base rates from data/reliability.json.</div>
</div>
</body>
</html>
"""


def write_slip(rows: List[dict], out_dir: Path) -> Path:
    """Write ``predictions/slip.html``: RELIABLE picks where both reports agree.

    A row qualifies when it is RELIABLE tier, has a best bet that is not
    ``no_value``, and that best bet matches the row's primary market+pick
    (which is what latest.html shows).
    """
    agree = []
    for r in rows:
        if r["tier"] != TIER_RELIABLE:
            continue
        best = r.get("best")
        if not best or best.get("no_value"):
            continue
        # best bet must match primary pick (= what latest.html shows)
        if not best.get("is_primary"):
            continue
        agree.append(r)

    if not agree:
        table = '<div class="empty">No consensus picks this run &mdash; the two reports did not agree on any RELIABLE fixture.</div>'
    else:
        conf_cls = {"High": "c-h", "Medium-High": "c-mh", "Medium": "c-m", "Low": ""}
        rows_html = []
        for i, r in enumerate(agree, 1):
            best = r["best"]
            odds = f'{best["odds"]:.2f}' if best.get("odds") is not None else "—"
            prob = f'{best["shown_prob"]:.1%}' if best.get("shown_prob") is not None else "—"
            base = f'{best["expected"]:.1%}' if best.get("expected") is not None else "—"
            lift = f'+{best["lift_pts"]:.1f}pt' if best.get("lift_pts") is not None else "—"
            conf = best.get("confidence") or r["confidence"]
            cc = conf_cls.get(conf, "")
            conf_html = f'<span class="conf {cc}">{html.escape(conf)}</span>' if cc else html.escape(conf)
            rows_html.append(
                f'<tr><td>{i}</td>'
                f'<td><b>{html.escape(r["home"])} v {html.escape(r["away"])}</b></td>'
                f'<td class="pick">{html.escape(best["pick"])}</td>'
                f'<td>{html.escape(best["market"])}</td>'
                f'<td class="odds">{odds}</td>'
                f'<td class="prob">{prob}</td>'
                f'<td class="prob">{base}</td>'
                f'<td class="lift">{lift}</td>'
                f'<td>{conf_html}</td>'
                f'<td>{html.escape(r["kickoff"] or "")}</td></tr>'
            )
        table = (
            '<table><tr><th>#</th><th>Match</th><th>Pick</th><th>Market</th>'
            '<th>Odds</th><th>Model</th><th>Base</th><th>Lift</th>'
            '<th>Conf</th><th>Kickoff</th></tr>'
            + "".join(rows_html) + "</table>"
        )

    out = out_dir / "slip.html"
    out.write_text(SLIP_HTML.replace("__TABLE__", table))
    return out


def summarise(rows: List[dict]) -> str:
    """One-line console summary for the end of a prediction run."""
    counts = {TIER_RELIABLE: 0, TIER_WATCH: 0, TIER_HELD: 0}
    for r in rows:
        counts[r["tier"]] += 1
    return (f"{counts[TIER_RELIABLE]} reliable / {counts[TIER_WATCH]} watch / "
            f"{counts[TIER_HELD]} held")


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--rebuild":
        rebuild_ledger()
        led = load_ledger()
        print(f"ledger: {len((led or {}).get('gate', {}))} buckets, "
              f"{(led or {}).get('resolved_rows', 0)} resolved rows")