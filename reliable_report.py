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

No odds-derived EV is ever printed: only12% of the odds stored in
``history.db`` survive validation, so any edge figure computed from them would
be fiction.  See :mod:`odds_guard`.
"""

from __future__ import annotations

import html
import json
import os
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
            out.update({
                "hit_rate": total["hit_rate"], "n": total["n"],
                "expected": total["expected"], "shrunk_rate": total["shrunk_rate"],
                "edge_pts": total["edge_pts"],
                "reason": f"no pick at this confidence in history "
                          f"({total['n']} resolved matches at other confidences)",
            })
        return out

    out.update({
        "hit_rate": entry["hit_rate"], "n": entry["n"],
        "expected": entry["expected"], "shrunk_rate": entry["shrunk_rate"],
        "edge_pts": entry["edge_pts"], "reason": entry["reason"],
    })
    if entry["publish"]:
        out["tier"] = TIER_RELIABLE
    elif entry["n"] >= (ledger.get("policy") or {}).get("min_samples", 200):
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
        # that is not a genuinely different selection.
        alternatives = []
        for c in (res.get("all_picks") or [])[1:6]:
            if not isinstance(c, dict):
                continue
            alt_pick = c.get("pick") or ""
            if not alt_pick or alt_pick == pick:
                continue
            alt_market = c.get("market") or market
            alt = classify_pick(alt_pick, c.get("confidence") or "", ledger, alt_market)
            if reserve:
                alt["tier"] = TIER_HELD
            alternatives.append({"pick": alt_pick,
                                 "confidence": c.get("confidence") or "",
                                 "tier": alt["tier"]})

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
--reliable:#22c55e;--watch:#f59e0b;--held:#64748b;--accent:#60a5fa;}
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
      f'<div class="n">{"no EV quoted — see note" if (odds_total and odds_ok < odds_total*0.5) else "of stored prices"}</div></div>')
    A("</div>")

    A('<div class="legend">'
      '<span><span class="badge b-reliable">RELIABLE</span> beaten base rate on '
      f'&ge;{(ledger or {}).get("policy",{}).get("min_samples","—")} historical matches</span>'
      '<span><span class="badge b-watch">WATCH</span> measured, but no edge</span>'
      '<span><span class="badge b-held">HELD</span> not recommended</span></div>')

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
        A(f'<div class="note"><b>Why no expected value is shown.</b> Only '
          f'{odds_ok:,} of {odds_total:,} stored prices ({_pct(odds_ok/odds_total,0)}) pass integrity '
          f'checks, and the stored 1X2 favourite agrees with Forebet&rsquo;s independent '
          f'percentage only {_pct(agree)} of the time. Any ROI or EV derived from these prices '
          'would be fiction, so this report quotes measured <i>hit rates</i> instead — the only '
          'thing the history can support.</div>')

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
<p class="sub">A prediction run writes exactly one report. It is gated on measured
historical performance, not on the model's own stated probability: every pick is
shown with how often that exact selection has actually landed, and picks with no
edge over their base rate are not recommended.</p>

<a class="card" href="picks.html">
  <b>picks.html &mdash; the report</b>
  <span>RELIABLE / WATCH / HELD tiers, calibration-adjusted probabilities,
  measured hit rate vs base rate per pick. No EV or ROI.</span>
</a>

<p class="sub" style="margin-top:22px">The older per-angle views
(<code>latest</code>, <code>best</code>, <code>consistent</code>,
<code>high80</code>, <code>bankers</code>, <code>bettor</code>,
<code>batch_NNN</code>) each reprint the model's stated probability with no
statement of how often that kind of pick actually lands. They are no longer
written by default &mdash; regenerate them with
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