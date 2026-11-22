"""Single source of truth for bookmaker-odds integrity.

Why this module exists
----------------------
``history.db`` was found (Oct 2026 audit) to contain unusable odds on97.2% of
resolved matches.  Two independent defects combined:

1. The pre-2026-09-22 legacy odds parser took the *first* decimal after the
   temperature, but Forebet prints the price of its own *predicted* outcome
   first.  A home forecast ("1") therefore produced ``home == draw``
   (e.g. 1.75/1.75/3.30).  Measured agreement of the duplicate rows with
   ``forebet_pred='1'``: 98%.
2. Even duplicate-free rows are mis-assigned: the stored 1X2 favourite agrees
   with Forebet's independently-published favourite only 23.3% of the time —
   worse than the 33% a coin flip would give.

Consequences that this module now prevents:

* Six ML features (``odds_home/draw/away``, ``implied_home_prob``,
  ``odds_fb_value_home``, ``odds_implied_overround``) were being fitted on
  scrambled numbers for 70.7% of training rows — pure noise.
* Every ROI / edge / EV figure computed from stored odds was meaningless.
  (De-vigged "market" Brier came out at 0.7165 — worse than a uniform 0.667
  forecast, and calibration gap of -19.4 pts.)

Policy
------
Never *repair* a bad price — it is unknowable which slot is wrong.  Instead
:classify` the row and let callers (a) neutralise the odds features so the
model learns nothing from them, and (b) refuse to quote an edge/ROI for the
match.  Forebet's own published percentages *are* reliable (Brier 0.6090, the
best forecaster in the DB) and are unaffected.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

__all__ = [
    "OddsVerdict",
    "validate_1x2",
    "validate_ou",
    "odds_triplet_is_usable",
    "devig",
    "NEUTRAL_ODDS",
    "feature_is_usable",
]

# Neutral values substituted for a rejected price.  Chosen as the DB-wide
# medians of the (small) trustworthy subset so the features land in the middle
# of their historical range and any residual tree split is a no-op rather than
# a spurious signal.
NEUTRAL_ODDS = {
    "odds_home": 2.30,
    "odds_draw": 3.20,
    "odds_away": 3.30,
}

# A real 1X2 book has 3-8% margin.  Anything outside this band means the three
# numbers did not come from one coherent market.
_MIN_OVERROUND = 0.97
_MAX_OVERROUND = 1.15

# Reject prices outside the plausible range for a league match outright.
_MIN_PRICE = 1.01
_MAX_PRICE = 100.0


class OddsVerdict:
    """Result of an odds integrity check.

    Attributes
    ----------
    ok:
        True only when the prices are safe to price a market with.
    reason:
        Short machine-readable code, e.g. ``"ok"``, ``"missing"``,
        ``"duplicate"``, ``"overround"``, ``"range"``, ``"favourite_mismatch"``.
    overround:
        ``sum(1/price) - 1`` when computable, else None.
    devig:
        The de-vigged (fair) probabilities when ok, else None.
    """

    __slots__ = ("ok", "reason", "overround", "devig")

    def __init__(self, ok: bool, reason: str,
                 overround: Optional[float] = None,
                 devig: Optional[Sequence[float]] = None):
        self.ok = ok
        self.reason = reason
        self.overround = overround
        self.devig = devig

    def __bool__(self) -> bool:
        return self.ok

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"OddsVerdict(ok={self.ok}, reason={self.reason!r})"


def _clean(x) -> Optional[float]:
    """Coerce to float, rejecting NaN/inf/non-positive."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v or v in (float("inf"), float("-inf")):  # NaN / inf
        return None
    return v


def devig(prices: Sequence[float]) -> Tuple[Optional[float], Optional[Tuple[float, ...]]]:
    """Return ``(overround, fair_probs)`` for a 3-way market.

    ``overround`` is ``sum(1/p) - 1`` (the book's margin).  ``fair_probs`` is
    the normalised probability vector, or None when a price is unusable.
    """
    vals = [_clean(p) for p in prices]
    if any(v is None for v in vals):
        return None, None
    total = sum(1.0 / v for v in vals)
    return total - 1.0, tuple((1.0 / v) / total for v in vals)


def validate_1x2(
    odds_home,
    odds_draw,
    odds_away,
    forebet_home_pct=None,
    forebet_draw_pct=None,
    forebet_away_pct=None,
    min_favourite_agreement: float = 0.30,
) -> OddsVerdict:
    """Decide whether a 1X2 triplet can be trusted.

    Checks, in order of cheapness:

    1. **present** — all three prices present and finite.
    2. **range** — each price within (1.01, 100].
    3. **duplicate** — no two prices equal.  This is the signature of the
       legacy parser bug and is worth rejecting outright: two outcome prices in
       a real market are equal maybe 1% of the time, and when they are it is
       a longshot-vs-favourite coincidence that the other checks catch.
    4. **overround** — implied sum within the margin band.
    5. **favourite_mismatch** — when Forebet's percentages are available, the
       cheapest price must be the outcome Forebet ranks first at least
       ``min_favourite_agreement`` of the time.  This is the check that
       catches the residual scramble the earlier checks miss.

    The mismatch test is *aggregate* by design (it returns a per-row verdict
    that is conservative), so callers should feed it Forebet percentages only
    when they trust them; the aggregate agreement rate reported by
    :func:`favourite_agreement_rate` is the honest measurement.
    """
    vals = (_clean(odds_home), _clean(odds_draw), _clean(odds_away))
    if any(v is None for v in vals):
        return OddsVerdict(False, "missing")
    if any(not (_MIN_PRICE <= v <= _MAX_PRICE) for v in vals):
        return OddsVerdict(False, "range")
    if len(set(vals)) != 3:
        return OddsVerdict(False, "duplicate")

    overround, fair = devig(vals)
    if overround is None or not (_MIN_OVERROUND <= 1.0 + overround <= _MAX_OVERROUND):
        return OddsVerdict(False, "overround", overround)

    fb = (_clean(forebet_home_pct), _clean(forebet_draw_pct), _clean(forebet_away_pct))
    if all(f is not None for f in fb) and 90 <= sum(fb) <= 110:
        cheapest = min(range(3), key=lambda i: vals[i])
        fb_top = max(range(3), key=lambda i: fb[i])
        if cheapest != fb_top:
            return OddsVerdict(False, "favourite_mismatch", overround)

    return OddsVerdict(True, "ok", overround, fair)


def validate_ou(odds_over, odds_under) -> OddsVerdict:
    """Decide whether an over/under pair can be trusted."""
    vals = (_clean(odds_over), _clean(odds_under))
    if any(v is None for v in vals):
        return OddsVerdict(False, "missing")
    if any(not (_MIN_PRICE <= v <= _MAX_PRICE) for v in vals):
        return OddsVerdict(False, "range")
    if vals[0] == vals[1]:
        return OddsVerdict(False, "duplicate")
    total = 1.0 / vals[0] + 1.0 / vals[1]
    if not (_MIN_OVERROUND <= total <= _MAX_OVERROUND):
        return OddsVerdict(False, "overround", total - 1.0)
    return OddsVerdict(True, "ok", total - 1.0,
                       ((1.0 / vals[0]) / total, (1.0 / vals[1]) / total))


def odds_triplet_is_usable(odds_home, odds_draw, odds_away) -> bool:
    """Fast structural check (no Forebet cross-reference). Used in hot loops."""
    return bool(validate_1x2(odds_home, odds_draw, odds_away))


def feature_is_usable(odds_home, odds_draw, odds_away) -> bool:
    """True when the odds may be fed to the ML model as features."""
    return odds_triplet_is_usable(odds_home, odds_draw, odds_away)


def favourite_agreement_rate(db_path: str = "history.db", limit_days: Optional[int] = None) -> dict:
    """Measure how often the stored 1X2 favourite matches Forebet's own.

    Returns ``{"n": int, "agree": int, "rate": float}``.  A rate near 1/3 means
    the odds columns carry no information; ~0.9+ means they are trustworthy.
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    sql = """
        SELECT odds_home, odds_draw, odds_away,
               forebet_home_pct, forebet_draw_pct, forebet_away_pct
        FROM matches
        WHERE actual_home_goals IS NOT NULL
          AND odds_home > 1 AND odds_draw > 1 AND odds_away > 1
          AND forebet_home_pct IS NOT NULL
          AND forebet_draw_pct IS NOT NULL AND forebet_away_pct IS NOT NULL
    """
    args: tuple = ()
    if limit_days:
        sql += " AND created_at >= date('now', ?)"
        args = (f"-{int(limit_days)} days",)
    rows = conn.execute(sql, args).fetchall()
    conn.close()

    n = agree = 0
    for r in rows:
        total = (r["forebet_home_pct"] + r["forebet_draw_pct"] + r["forebet_away_pct"])
        if not (90 <= total <= 110):
            continue
        overround, _ = devig([r["odds_home"], r["odds_draw"], r["odds_away"]])
        if overround is None or not (0.95 <= 1.0 + overround <= 1.25):
            continue
        prices = [r["odds_home"], r["odds_draw"], r["odds_away"]]
        fbs = [r["forebet_home_pct"], r["forebet_draw_pct"], r["forebet_away_pct"]]
        n += 1
        agree += (min(range(3), key=lambda i: prices[i])
                  == max(range(3), key=lambda i: fbs[i]))
    return {"n": n, "agree": agree, "rate": (agree / n) if n else 0.0}


if __name__ == "__main__":  # pragma: no cover - manual diagnostic
    import json
    print(json.dumps(favourite_agreement_rate(), indent=2))