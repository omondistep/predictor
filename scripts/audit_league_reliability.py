#!/usr/bin/env python3
"""Per-league diagnosis of the old "Unreliable league" warnings.

The predictor used to label a fixture hard when
``calibration_log.league LIKE '<prefix>%'`` had a raw hit rate below 65%.  This
script explains every such warning in a saved report:

* what the old warning actually measured (pooled countries, markets, picks);
* what the exact competition measures now (base-relative edge per market);
* why the old label failed for that competition;
* which correction now applies to it.

Usage:
    .venv/bin/python scripts/audit_league_reliability.py [report.html]

With no argument, the newest predictions/*.html containing the old warning is
used.  Writes reports/unreliable_league_audit.md and
data/unreliable_league_audit.json.  Read-only on history.db.
"""
from __future__ import annotations

import argparse
import collections
import html
import json
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DB = ROOT / "history.db"
REPORT_DIR = ROOT / "reports"
DATA_DIR = ROOT / "data"

import league_stats  # noqa: E402
from predict import _resolve_league_prefix  # noqa: E402

_WARNING_RE = re.compile(r"Unreliable league \((\d+)% accuracy, (\d+) matches\)")
_CARD_RE = re.compile(r'<div class="card" data-match-id=')
_META_RE = re.compile(r'<div class="card-meta">(.*?)</div>', re.S)

_SPECIAL_RE = re.compile(
    r"\b(cup|trophy|women|womens|u19|u20|u21|u23|primavera|reserve|youth|junior|juniors)\b"
    r"|\b(ii|iii|2|3)\b|\b(3\.|4\.)|third|fourth|amateur|regionalliga|"
    r"oberliga|landesliga|verbandsliga|serie d|tercera|segunda b|national league|"
    r"division 2|league 2|league 3|npl|isthmian|non ?league",
    re.I,
)


def parse_warnings(path: Path) -> list[dict]:
    """Return unique (league, shown pct, shown n) warnings from a card report."""
    text = path.read_text(encoding="utf-8", errors="replace")
    out = {}
    for block in _CARD_RE.split(text):
        meta = _META_RE.search(block)
        warning = _WARNING_RE.search(block)
        if not (meta and warning):
            continue
        league = html.unescape(re.sub(r"<[^>]+>", "", meta.group(1))).strip()
        league = league.split("·")[0].strip()
        if not league:
            continue
        key = (league, int(warning.group(1)), int(warning.group(2)))
        out[key] = out.get(key, 0) + 1
    return [{"league": k[0], "shown_pct": k[1], "shown_n": k[2], "cards": n}
            for k, n in sorted(out.items())]


def oldest_pool_info(prefix: str) -> dict:
    """What the old prefix query actually pooled, with examples."""
    if not prefix:
        return {"leagues": 0, "markets": 0, "picks": 0, "first_tokens": [],
                "special_labels": [], "examples": []}
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT league, market, COUNT(*) n FROM calibration_log "
        "WHERE league LIKE ? GROUP BY league, market ORDER BY n DESC",
        (f"{prefix}%",)).fetchall()
    conn.close()
    labels = sorted({r[0] for r in rows if r[0]})
    return {
        "leagues": len(labels),
        "markets": len({r[1] for r in rows}),
        "picks": sum(r[2] for r in rows),
        "first_tokens": sorted({label.split()[0] for label in labels if label.split()}),
        "special_labels": [label for label in labels if _SPECIAL_RE.search(label)],
        "examples": [r[0] for r in rows[:5]],
    }


def causes_for(row: dict, exact: dict, pool: dict) -> list[str]:
    causes = []
    label = row["league"]
    if exact["n"] < league_stats.MIN_SAMPLES:
        causes.append("thin-exact-sample")
    if pool["leagues"] > 1:
        causes.append("prefix-pooling")
    if len(pool["first_tokens"]) > 1:
        causes.append("country-code-collision")
    if pool["markets"] > 1:
        causes.append("market-pooling")
    if pool["special_labels"] or _SPECIAL_RE.search(label):
        causes.append("competition-type-mixing")
    weak_markets = []
    for market, st in league_stats.market_breakdown(label).items():
        if st["grade"] == "weak":
            weak_markets.append(market)
    if weak_markets:
        causes.append(f"genuine-negative-edge:{','.join(weak_markets)}")
    if not causes:
        causes.append("threshold-artefact")
    return causes


CAUSE_DETAIL = {
    "prefix-pooling": (
        "The old LIKE query pooled several competitions under one country/code "
        "prefix; the warning described none of them accurately.",
        "Resolve the exact competition key and measure it alone; never inherit a "
        "country-wide sample.",
    ),
    "country-code-collision": (
        "The short prefix also matched unrelated countries or codes (for example "
        "'Au' pulls Australia and Austria, 'Ch' pulls Switzerland, China and "
        "Chile), so the warning was often attributed to the wrong country.",
        "Replace prefix inference with an explicit normalized competition key; "
        "never resolve a league from its first word.",
    ),
    "market-pooling": (
        "1X2 (~35% base), O/U (~55%), DC (~76%) and 1H (~40%) were averaged "
        "behind one 65% cut-off.",
        "Score each market against the base rate of the selection actually picked; "
        "a negative edge in one market no longer condemns the others.",
    ),
    "competition-type-mixing": (
        "Cups, women's, youth/reserve and lower-tier competitions were pooled "
        "with senior league football.",
        "Keep those competitions as separate exact keys; do not let their "
        "goal/draw profile suppress a senior league or vice versa.",
    ),
    "thin-exact-sample": (
        "The exact competition has fewer than 25 scored picks; the large old "
        "sample came from other leagues or other markets.",
        "Label it unknown: cautious confidence/stake haircut, no 'unreliable' "
        "verdict and no country-pool rescue.",
    ),
    "threshold-artefact": (
        "The raw hit rate was below 65%, but the picks sit at or above their "
        "own measured base rate.",
        "Remove the warning and leave probabilities untouched; the model is not "
        "demonstrably wrong here.",
    ),
}

_FACTOR_FIX = (
    "Measured negative edge: damp O/U and BTTS toward base, cap confidence and "
    "Quarter-Kelly stake for this competition only."
)


def correction_text(causes: list[str], exact: dict) -> str:
    bits = []
    for cause in causes:
        if cause.startswith("genuine-negative-edge:"):
            markets = cause.split(":", 1)[1]
            bits.append(f"Weak markets: {markets}. {_FACTOR_FIX}")
        elif cause in CAUSE_DETAIL:
            bits.append(CAUSE_DETAIL[cause][1])
    return " ".join(dict.fromkeys(bits))


def why_text(causes: list[str], row: dict, pool: dict, exact: dict) -> str:
    bits = []
    for cause in causes:
        if cause.startswith("genuine-negative-edge:"):
            bits.append("measured below its own base rate in the named markets")
        elif cause in CAUSE_DETAIL:
            bits.append(CAUSE_DETAIL[cause][0])
    return " ".join(dict.fromkeys(bits))


def build_rows(warnings: list[dict]) -> list[dict]:
    rows = []
    for row in warnings:
        league = row["league"]
        prefix = _resolve_league_prefix(league)
        pool = oldest_pool_info(prefix)
        exact = league_stats.league_stat(league)
        one_x2 = league_stats.market_stat(league, "1X2")
        ou = league_stats.market_stat(league, "O/U")
        btts = league_stats.market_stat(league, "BTTS")
        causes = causes_for(row, exact, pool)
        rows.append({
            **row,
            "prefix": prefix,
            "pool": pool,
            "exact": exact,
            "1X2": one_x2,
            "O/U": ou,
            "BTTS": btts,
            "causes": causes,
            "why": why_text(causes, row, pool, exact),
            "correction": correction_text(causes, exact),
        })
    return rows


def _pct(value):
    return "—" if value is None else f"{value:.1f}%"


def _edge(stat):
    return "—" if not stat["n"] else f"{stat['edge_pts']:+.1f}"


def write_markdown(rows: list[dict], source: Path) -> Path:
    counts = collections.Counter()
    for r in rows:
        for cause in r["causes"]:
            counts[cause.split(":", 1)[0]] += 1
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "unreliable_league_audit.md"
    lines = [
        "# Unreliable-league audit",
        "",
        f"Generated {datetime.now().isoformat(timespec='seconds')} from "
        f"`{source.relative_to(ROOT)}` and `history.db`.",
        "",
        "The old warning was `Unreliable league (P% accuracy, N matches)`. "
        "N was a count of **scored market picks**, not matches, and the prefix "
        "query could pull in several competitions and markets at once. Every "
        "row below now carries the exact competition's base-relative edge.",
        "",
        "## Failure classes",
        "",
        "| Cause | Competitions | Meaning |",
        "|---|---:|---|",
    ]
    for cause, n in counts.most_common():
        if cause in CAUSE_DETAIL:
            meaning = CAUSE_DETAIL[cause][0]
        elif cause == "genuine-negative-edge":
            meaning = "Measured below the base rate of the picked selection in at least one market."
        else:
            meaning = cause
        lines.append(f"| `{cause}` | {n} | {meaning} |")
    lines.extend([
        "",
        "## Per-competition diagnosis and correction",
        "",
        "| Competition | Old warning | Old pool | Exact picks | 1X2 edge | O/U edge | BTTS edge | Cause | Why it failed | Correction |",
        "|---|---|---|---:|---:|---:|---:|---|---|---|",
    ])
    for r in rows:
        old = f"{r['shown_pct']}% / n={r['shown_n']}"
        codes = ", ".join(r["pool"]["first_tokens"][:4]) or "—"
        examples = ", ".join(r["pool"]["examples"][:2]) or "—"
        pool = (f"prefix {r['prefix']}: {r['pool']['leagues']} league(s), "
                f"{r['pool']['markets']} markets, {r['pool']['picks']} picks; "
                f"codes {codes}; e.g. {examples}").replace("|", "\\|")
        lines.append(
            "| {league} | {old} | {pool} | {n} | {e1} | {eo} | {eb} | {causes} | {why} | {fix} |".format(
                league=r["league"].replace("|", "\\|"),
                old=old, pool=pool, n=r["exact"]["n"],
                e1=_edge(r["1X2"]), eo=_edge(r["O/U"]), eb=_edge(r["BTTS"]),
                causes=", ".join(c.split(":", 1)[0] for c in r["causes"]),
                why=r["why"].replace("|", "\\|"),
                fix=r["correction"].replace("|", "\\|"),
            )
        )
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", help="HTML report containing old warnings")
    args = parser.parse_args()

    if args.report:
        source = Path(args.report).resolve()
    else:
        candidates = [p for p in (ROOT / "predictions").glob("*.html")
                      if "Unreliable league" in p.read_text(encoding="utf-8", errors="replace")]
        if not candidates:
            raise SystemExit("no report containing 'Unreliable league' was found")
        source = max(candidates, key=lambda p: p.stat().st_mtime)

    warnings = parse_warnings(source)
    if not warnings:
        raise SystemExit(f"no 'Unreliable league' warnings found in {source}")
    rows = build_rows(warnings)
    md = write_markdown(rows, source)
    json_path = DATA_DIR / "unreliable_league_audit.json"
    json_path.write_text(json.dumps({
        "source": str(source.relative_to(ROOT)),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "warnings": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(rows)} warned competitions diagnosed")
    print(f"  {md.relative_to(ROOT)}")
    print(f"  {json_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
