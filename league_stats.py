"""Exact, market-aware league reliability measurements.

The old predictor asked ``calibration_log WHERE league LIKE 'England%'`` and
compared the pooled hit rate with universal 65/80% thresholds.  That query was
wrong in three independent ways:

* ``England`` (and Spain, Italy, Germany, ...) is a country, not a competition.
  The pool mixed the Premier League with FA Cup, National League, women's and
  reserve competitions.
* ``calibration_log`` has one row per *market prediction*, not one row per
  match.  1,883 rows are 1,883 scored picks, not 1,883 games.
* A Draw pick lands about 24% of the time and Under 5.5 about 91%.  Comparing
  both with the same absolute accuracy threshold is meaningless.

This module is the single source for league reliability.  It keys on the exact
normalised competition, measures each market against the base rate of the
*selection actually picked*, and reports edges in percentage points.  Unknown
competitions stay unknown; a thin sample never earns a full stake or a "reliable"
badge by borrowing another league's history.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path

from league_names import clean_league_name

MIN_SAMPLES = 25
CREDIBILITY_K = 120.0
MIN_EDGE_PTS = 2.0
Z_ONE_SIDED_95 = 1.645

_REPO = Path(__file__).resolve().parent
_LEDGER_PATH = _REPO / "data" / "reliability.json"

_FALLBACK_BASE_RATES = {
    "1X2|Home win": 0.430, "1X2|Draw": 0.242, "1X2|Away win": 0.328,
    "DC|1X": 0.672, "DC|X2": 0.567, "DC|12": 0.757,
    "O/U|Over 1.5": 0.771, "O/U|Under 1.5": 0.229,
    "O/U|Over 2.5": 0.547, "O/U|Under 2.5": 0.453,
    "O/U|Over 3.5": 0.336, "O/U|Under 3.5": 0.664,
    "O/U|Over 4.5": 0.186, "O/U|Under 4.5": 0.814,
    "O/U|Over 5.5": 0.092, "O/U|Under 5.5": 0.908,
    "BTTS|Yes": 0.542, "BTTS|No": 0.458,
    "DNB|Home": 0.572, "DNB|Away": 0.428,
    "1H|Home win": 0.330, "1H|Draw": 0.397, "1H|Away win": 0.273,
}

# Scraper crumbs such as "Us4 Peoria Sueno 17/07/2026" or
# "ArC Racing Club Defensa" are match rows, not competitions.  They used to be
# swept into a country prefix and then blamed on every league in that country.
_JUNK_RE = re.compile(
    r"\d{1,2}/\d{1,2}/\d{4}|\bvs\.?\b|\bspieltag\b|\bfixture\b|\bround\b",
    re.I,
)
_CODE_LEAF_RE = re.compile(r"^[A-Z][a-z]{0,2}(?:\d|[A-Z])\b")

_BASE_CACHE = None
_INDEX_CACHE = None


# ---------------------------------------------------------------------------
# Normalisation and base rates
# ---------------------------------------------------------------------------

def base_rates() -> dict:
    """Measured unconditional rates, loaded once from the reliability ledger."""
    global _BASE_CACHE
    if _BASE_CACHE is not None:
        return _BASE_CACHE
    rates = dict(_FALLBACK_BASE_RATES)
    try:
        ledger = json.loads(_LEDGER_PATH.read_text())
        loaded = ledger["policy"]["base_rates"]["rates"]
        if isinstance(loaded, dict) and loaded:
            rates.update({k: float(v) for k, v in loaded.items()})
    except Exception:
        pass
    _BASE_CACHE = rates
    return rates


def league_key(label: str) -> str | None:
    """Exact competition key, or ``None`` for unusable / match-row labels.

    Punctuation, diacritics and standings noise are folded so the feed's
    ``"Spain — Segunda Division"`` lands on the stored ``"Spain Segunda
    Division"``.  Country/division are deliberately preserved: ``England
    Premier League`` must never resolve to the same key as ``England FA Cup``.
    """
    if not label:
        return None
    original = re.sub(r"\s+", " ", str(label)).strip()
    if not original or _JUNK_RE.search(original):
        return None
    # A code plus team names is a mis-scraped fixture row, e.g.
    # "Br1 Botafogo Santos".  Bare codes ("Br1") are kept for legacy rows.
    if _CODE_LEAF_RE.match(original) and len(original.split()) > 1:
        return None

    cleaned = clean_league_name(original)
    if not cleaned or _JUNK_RE.search(cleaned):
        return None
    cleaned = re.sub(r"^\s*in\s+", "", cleaned, flags=re.I)
    folded = unicodedata.normalize("NFKD", cleaned)
    folded = "".join(c for c in folded if not unicodedata.combining(c)).lower()
    folded = folded.replace("ı", "i").replace("ø", "o")
    folded = re.sub(r"[^a-z0-9]+", " ", folded)
    folded = re.sub(r"\s+", " ", folded).strip()

    folded = folded.replace("turkiye", "turkey").replace("turk", "turk")
    folded = folded.replace("espana", "spain")
    folded = re.sub(r"\bsuper\s+(?:lig|ligi)\b", "super league", folded)
    folded = re.sub(r"\s*-\s*", " ", folded)
    folded = re.sub(r"\s+", " ", folded).strip()
    return folded or None


def league_aliases(label: str) -> list[str]:
    """Exact key first, then spelling-only variants.  Never a country roll-up."""
    key = league_key(label)
    if not key:
        return []
    aliases = [key]
    for variant in (
        key.replace("turkiye", "turkey"),
        re.sub(r"\bsuper\s+(?:lig|ligi)\b", "super league", key),
        key.replace("espana", "spain"),
    ):
        if variant and variant not in aliases:
            aliases.append(variant)
    return aliases


def selection_base(market: str, pick: str) -> float | None:
    """Unconditional base rate for the selection actually picked."""
    market = (market or "").strip()
    pick = (pick or "").strip()
    rates = base_rates()

    if market in ("1X2", "1H"):
        low = pick.lower()
        if low in ("home", "home win", "1"):
            return rates["1X2|Home win" if market == "1X2" else "1H|Home win"]
        if low in ("draw", "x"):
            return rates["1X2|Draw" if market == "1X2" else "1H|Draw"]
        if low in ("away", "away win", "2"):
            return rates["1X2|Away win" if market == "1X2" else "1H|Away win"]
        return None

    if market == "DNB":
        low = pick.lower()
        if low in ("home", "home win", "1"):
            return rates["DNB|Home"]
        if low in ("away", "away win", "2"):
            return rates["DNB|Away"]
        return None

    if market == "DC":
        key = pick.upper()
        return rates.get(f"DC|{key}")

    if market == "BTTS":
        low = pick.lower()
        if low in ("yes", "btts yes", "gg"):
            return rates["BTTS|Yes"]
        if low in ("no", "btts no", "ng"):
            return rates["BTTS|No"]
        return None

    if market == "O/U":
        low = pick.lower()
        side = "Over" if low.startswith("over") else "Under" if low.startswith("under") else None
        if side is None:
            return None
        m = re.search(r"(\d+(?:\.\d+)?)", pick)
        line = float(m.group(1)) if m else 2.5
        # Storage keys are side-agnostic line labels: "Over3" means 3.5.
        if line.is_integer():
            line += 0.5
        line = max(1.5, min(5.5, line))
        label = f"{line:.1f}".rstrip("0").rstrip(".")
        return rates.get(f"O/U|{side} {label}")

    return None


# ---------------------------------------------------------------------------
# Measured index
# ---------------------------------------------------------------------------

def _fixture_key(match_id, url) -> str:
    """Collapse duplicate stored fixtures by trailing Forebet match id."""
    ids = re.findall(r"-(\d+)\b", str(url or "").split("?")[0])
    return ids[-1] if ids else f"id:{match_id}"


def _build_index() -> dict:
    """key -> market -> list of scored predictions (deduplicated by fixture)."""
    try:
        from database import get_db
        conn = get_db()
        rows = conn.execute(
            "SELECT c.league AS cleague, m.league AS mleague, c.market AS market, "
            "c.our_prediction AS pick, c.correct AS correct, c.created_at AS created_at, "
            "c.match_id AS match_id, m.forebet_url AS url "
            "FROM calibration_log c LEFT JOIN matches m ON m.id = c.match_id "
            "WHERE c.correct IS NOT NULL"
        ).fetchall()
        conn.close()
    except Exception:
        rows = []

    groups: dict[str, dict[str, list]] = {}
    seen = set()
    for r in rows:
        market = r["market"] or "?"
        base = selection_base(market, r["pick"] or "")
        if base is None:
            continue
        fixture = (fixture_key, market) if (fixture_key := _fixture_key(r["match_id"], r["url"])) else None
        if fixture in seen:
            continue
        seen.add(fixture)
        record = {
            "n": 1,
            "hits": 1 if r["correct"] else 0,
            "base": base,
            "created_at": str(r["created_at"] or "")[:10],
            "pick": r["pick"] or "",
        }
        # One prediction may carry two labels for the same competition (the
        # calibration row and its joined match row).  Index it once per key so
        # an exact league never gets its own pick counted twice.
        keys = []
        for label in (r["cleague"], r["mleague"]):
            for key in league_aliases(label):
                if key not in keys:
                    keys.append(key)
        for key in keys:
            groups.setdefault(key, {}).setdefault(market, []).append(record)
    return groups


def _index() -> dict:
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        _INDEX_CACHE = _build_index()
    return _INDEX_CACHE


def invalidate_cache() -> None:
    """Re-read history after a results update (mostly useful in long-lived tools)."""
    global _INDEX_CACHE, _BASE_CACHE
    _INDEX_CACHE = None
    _BASE_CACHE = None


def _records(label: str, market: str | None = None, days: int | None = None) -> tuple[list, str | None]:
    groups = _index()
    for key in league_aliases(label):
        if market is None:
            recs = [r for rows in groups.get(key, {}).values() for r in rows]
        else:
            recs = list(groups.get(key, {}).get(market, []))
        if days:
            from datetime import datetime, timedelta
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            recs = [r for r in recs if r["created_at"] >= cutoff]
        if recs:
            return recs, key
    return [], None


def _wilson_lower(hits: int, n: int) -> float:
    if n <= 0:
        return 0.0
    p = hits / n
    z = Z_ONE_SIDED_95
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - margin)


def _summary(records: list, key: str | None, market: str | None) -> dict:
    n = len(records)
    if not n:
        return {
            "market": market, "league_key": key, "n": 0, "correct": 0,
            "pct": 0.0, "base_pct": 0.0, "edge_pts": 0.0,
            "shrunk_pct": 0.0, "wilson_lo_pct": 0.0, "grade": "unknown",
            "measured": False,
        }
    hits = sum(r["hits"] for r in records)
    base = sum(r["base"] for r in records) / n
    pct = 100.0 * hits / n
    edge = pct - 100.0 * base
    shrunk = 100.0 * (hits + CREDIBILITY_K * base) / (n + CREDIBILITY_K)
    wilson = 100.0 * _wilson_lower(hits, n)
    if n < MIN_SAMPLES:
        grade = "unknown"
    elif edge >= MIN_EDGE_PTS and wilson > 100.0 * base:
        grade = "reliable"
    elif edge >= -MIN_EDGE_PTS:
        grade = "at_base"
    else:
        grade = "weak"
    return {
        "market": market, "league_key": key, "n": n, "correct": hits,
        "pct": round(pct, 1), "base_pct": round(100.0 * base, 1),
        "edge_pts": round(edge, 1), "shrunk_pct": round(shrunk, 1),
        "wilson_lo_pct": round(wilson, 1), "grade": grade, "measured": True,
    }


def market_stat(label: str, market: str, days: int | None = None) -> dict:
    """Base-relative reliability for one exact competition and market."""
    records, key = _records(label, market, days)
    return _summary(records, key, market)


def league_stat(label: str, days: int | None = None) -> dict:
    """Base-relative reliability across all markets for one exact competition."""
    records, key = _records(label, None, days)
    return _summary(records, key, None)


def all_league_stats() -> dict[str, dict]:
    """Base-relative stats for every exact competition in the measured index."""
    out = {}
    for key, markets in _index().items():
        records = [r for rows in markets.values() for r in rows]
        out[key] = _summary(records, key, None)
    return out


def all_market_stats() -> dict[str, dict[str, dict]]:
    """Exact competition key -> market -> base-relative stat."""
    return {
        key: {market: _summary(records, key, market)
              for market, records in markets.items()}
        for key, markets in _index().items()
    }


def market_breakdown(label: str) -> dict[str, dict]:
    """Stats for every market measured in this exact competition."""
    key = league_key(label)
    if not key:
        return {}
    out = {}
    for market, records in _index().get(key, {}).items():
        out[market] = _summary(records, key, market)
    return out


def reliability_factor(label: str, market: str | None = None) -> tuple[float, str]:
    """Conservative damping factor used by the prediction pipeline.

    A competition at its own base rate needs no damping; a measured negative
    edge does.  An unmeasured competition stays at the old cautious 0.7 — it is
    unknown, not proven.
    """
    stat = market_stat(label, market) if market else league_stat(label)
    if stat["grade"] in ("reliable", "at_base"):
        return 1.0, stat["grade"]
    if stat["grade"] == "weak":
        return 0.7, "weak"
    return 0.7, "unknown"


def difficulty(label: str) -> dict:
    """Replacement for the old absolute-threshold league difficulty bucket."""
    stat = league_stat(label)
    n = stat["n"]
    edge = stat["edge_pts"]
    key = stat["league_key"] or "unknown"
    if stat["grade"] == "reliable":
        level = "easy"
        reason = (f"Historically beats its own base rate by {edge:+.1f}pts "
                  f"({stat['pct']:.0f}% on {n} scored picks, {key})")
    elif stat["grade"] == "at_base":
        level = "medium"
        reason = (f"Measured at its own base rate ({edge:+.1f}pts, {n} scored "
                  f"picks, {key})")
    elif stat["grade"] == "weak":
        level = "hard"
        reason = (f"Measured below its own base rate ({edge:+.1f}pts, {n} "
                  f"scored picks, {key}) — bet with caution")
    else:
        level = "medium"
        reason = (f"Insufficient exact-league data ({n} scored picks, {key}); "
                  f"not judged by a country-wide pool")
    return {
        "level": level,
        "accuracy": stat["pct"],
        "raw_pct": stat["pct"],
        "base_pct": stat["base_pct"],
        "edge_pts": edge,
        "matches": n,
        "reason": reason,
        "grade": stat["grade"],
        "league_key": key,
    }
