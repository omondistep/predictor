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

  bettor.py slip <slip.txt> [--preds <pr-dump.json>] [--no-model] [--json] [--html]
      Leg-by-leg verdict for a multi-bet slip file, one leg per line:
          Team A - Team B, 3-Way Home @2.03
          Team C - Team D, Over 2.5 @1.48   (odds optional)
      Every leg is resolved to a pipeline fixture (pending_results.json +
      history.db, or scraped via pr --json unless --no-model), priced with
      the same de-vig / EV / Quarter-Kelly math as `card`, then classified:
          BET     EV >= +3%, conf >= Medium, league not COLD, no red flags
          PASS    analysed but no edge / Low conf / COLD league / red flags
          KILL    model's pick conflicts with the slip, EV <= -5%,
                  already finished
          NO-DATA no fixture or no model probability — unverifiable
      Prints the verdict table, an action ticket (BET legs only) and a
      research checklist for the agent's web layer (venue, injuries,
      phantom fixtures stay human).  --html writes predictions/verdict.html.

  bettor.py research <slip.txt> [--preds <pr-dump.json>] [--no-model]
                              [--json] [--apply <dir>] [--template]
      The opencode-powered research loop.  Without --apply it prints a
      research AGENDA (one item per leg needing human web checks) plus the
      evidence schema each leg expects.  opencode then runs the web search
      itself and writes one data/research/<slug>.json per fixture.  With
      --apply <dir> it loads those evidence files back, merges them into the
      pricing loop (key-absence prob haircuts, venue fixtures, red-flag
      clearing) and prints the same verdict table + action ticket as `slip`.
      Research can only downgrade or confirm a model pick, never invent edge.

Exit code is 0 on success. Read-only on the DB (never writes history.db).
"""

import argparse
import json
import re
import sqlite3
import subprocess
import sys
from functools import lru_cache
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import odds_guard  # noqa: E402

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

# Country tokens the league-family roll-up hoists to the front of a key so
# scraper variants ('in Bundesliga (Germany)...') merge into one family.
_COUNTRIES = (
    "germany", "austria", "france", "italy", "spain", "england", "portugal",
    "netherlands", "denmark", "switzerland", "turkey", "brazil", "argentina",
    "china", "sweden", "norway", "poland", "greece", "belgium", "serbia",
    "scotland", "ireland", "romania", "ukraine", "colombia", "ecuador", "peru",
    "chile", "uruguay", "paraguay", "bolivia", "mexico", "united states",
    "south korea", "japan", "india", "australia", "slovenia", "slovakia",
    "czech republic", "hungary", "croatia", "russia", "kazakhstan", "belarus",
    "morocco", "tunisia", "algeria", "egypt", "south africa",
)

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

# ── Research layer (opencode does the web search) ──────────────────────────
# Evidence files live in data/research/<slug>.json.  The agent (opencode)
# writes them after a real web search; `research --apply` merges them back.
RESEARCH_DIR = REPO / "data" / "research"

# prob haircut per confirmed absence, by playing role.  Annualised as a
# fraction of the side's win prob / the scoring side of an O/U or BTTS bet.
# Hatched to be conservative: the pipeline itself moves ~3-5 goal-percent,
# this is a fraction of the *probability* that the pick already carries.
ROLE_WEIGHTS = {
    "forward": 0.05,       # top scorer out — the market barely priced it in
    "midfielder": 0.03,
    "defender": 0.035,
    "keeper": 0.06,
    "key": 0.04,           # club communicates "key man absent"
    "unknown": 0.03,
}
# A star absence beyond this many points total is suspicious — cap it.
ABSENCE_CAP = 0.10

# Venue haircut when the fixture is neutral (not played at the side's home).
NEUTRAL_VENUE_CUT = 0.04

# Stakes below this % of bankroll are dust — rounding and model error swamp
# the edge at that size, so a real card leaves them unbet.  They stay visible
# in all_candidates (annotated) for audit.
DUST_STAKE_PCT = 0.5

# Must be present in an evidence file before it is trusted at all.
EVIDENCE_REQUIRED = {
    "fixture": ["real", "competition"],
    "findings": ["summary"],
}
_ALWAYS_ALLOWED_EVIDENCE = {
    "sources", "date_checked", "looked_at_odds", "confidence_in_research",
    "flags",
}

# Shape for an evidence file at data/research/<slug>.json.  `research` emits
# this skeleton + a worklist; opencode fills it in after a real web search.
_EVIDENCE_TEMPLATE = {
    "slug": "",
    "fixture": {
        "real": True,
        "competition": "",
        "date": "",
        "venue": "home",            # home | neutral | away
    },
    "findings": {
        "summary": "",              # required — what the web check established
        "absences": [               # optional — confirmed absences
            {"team": "home", "role": "forward", "player": ""},
        ],
        "red_flags": [],            # optional — extra problems found
    },
    "sources": [],                  # urls actually consulted
    "date_checked": "",
    "looked_at_odds": True,
    "confidence_in_research": "medium",   # high | medium | low
    "flags": [],
}


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
    import sys as _sys
    print(f"[book] Scraping {links_path} — this takes ~1-2 min per 10 matches...", file=_sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        print(f"pr --json failed:\n{proc.stderr[-2000:]}", file=_sys.stderr)
        sys.exit(2)
    return _load_json_robust(proc.stdout)


def _load_json_robust(text):
    """Parse JSON even when predict.py prints [calibrate] noise to stdout
    before the actual JSON array."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pos = text.find("[")
        while pos >= 0:
            try:
                return json.loads(text[pos:])
            except json.JSONDecodeError:
                pos = text.find("[", pos + 1)
        raise


def _load_matches(path):
    p = Path(path)
    if p.exists() and p.suffix.lower() == ".json":
        return _load_json_robust(p.read_text())
    return _run_pr_json(p)


def _devig(odds):
    """Proportionally remove bookmaker margin; returns {key: fair_prob}."""
    imp = {k: 1.0 / v for k, v in odds.items() if v and v > 1.0}
    s = sum(imp.values())
    if s <= 0:
        return {}
    return {k: v / s for k, v in imp.items()}


def _market_prices_trustworthy(odds, market):
    """True when the prices backing ``market`` survive :mod:`odds_guard`.

    An Oct 2026 audit found 87.7% of stored prices unusable (duplicated from
    the legacy parser, or mis-assigned so the favourite did not match
    Forebet's own ranking).  Quoting edge or EV from those prices produces
    numbers that look impressive and mean nothing — the raw figures ran to
    +145% ROI, which is not a market, it is a bug.  When validation fails the
    card reports the pick but never prices it.
    """
    mapping = MARKET_ODDS.get(market) or {}
    present = {pick: odds[odkey] for pick, odkey in mapping.items() if odds.get(odkey)}
    if not present:
        return False, "no prices"
    if market == "1X2":
        fb = _fb_pcts()
        if not any(isinstance(p, (int, float)) for p in fb):
            # The favourite-agreement test is the ONLY check that catches a
            # structurally plausible triplet assigned to the wrong outcomes --
            # the failure mode behind 5,666 stored rows.  Without Forebet's
            # own percentages it cannot run, so a scrambled triplet would sail
            # through and get quoted as +272% EV.  This card publishes money,
            # so an unverifiable price is treated as no price at all.
            return False, "no forecet percentages to verify the price against"
        v = odds_guard.validate_1x2(
            odds.get("odds_home"), odds.get("odds_draw"), odds.get("odds_away"),
            fb[0], fb[1], fb[2])
        return bool(v), v.reason
    picks = list(present)
    first = odds[mapping[picks[0]]]
    second = odds[mapping[picks[1]]] if len(picks) > 1 else None
    v = odds_guard.validate_ou(first, second)
    return bool(v), v.reason


def _fb_pcts():
    """Forebet percentages if the caller has set them, else Nones.

    Kept as a hook so the favourite-agreement cross-check (the only test that
    catches the mis-assignment the structural checks miss) stays optional.
    """
    return (_FB_PCTS.get("home"), _FB_PCTS.get("draw"), _FB_PCTS.get("away"))


_FB_PCTS = {}


def _league_market_accuracy():
    """Reliability per league family, sampled across ALL seasons.

    League labels drift between scraper variants and seasons ('Türkiye Süper
    Lig' vs 'Turkey Super League', 'China Super League' vs '10th place China
    Super League 5th place').  Grouping by the exact stored label splits every
    real league into crumbs that never reach n>=20 and get misreported as
    'unverified'.  Roll up each label to a stable family key first, then apply
    the sample floor to the family total."""
    acc = {}
    fam = {}
    rows = _try(
        "SELECT m.league, c.market, COUNT(*) n, SUM(c.correct) correct "
        "FROM calibration_log c JOIN matches m ON m.id=c.match_id "
        "GROUP BY m.league, c.market"
    )
    if isinstance(rows, list):
        for r in rows:
            family = _league_family(r["league"])
            if family is None:
                continue
            k = (r["market"], family)
            n0, c0 = fam.get(k, (0, 0))
            fam[k] = (n0 + r["n"], c0 + (r["correct"] or 0))
    for (market, family), (n, correct) in fam.items():
        if n >= 20:
            acc.setdefault(market, {})[family] = correct / n
    return acc


def _league_family(league):
    """Canonical cross-season league key, or None for junk.

    Normalises country/league spelling variants (Türkiye/Turkey, 'Super Lig'
    vs 'Super League'), strips Forebet place tokens and any 'in ... ... '
    cup-glue the scraper can dump into the league column, and drops rows that
    still carry dates or team names."""
    lg = _league_key_clean(league)
    if lg is None:
        return None
    out = lg.lower().strip()
    out = re.sub(r"^in\s+regionalliga.*?öfb cup\s+", "", out)
    out = re.sub(r"\s+(champions league|europa league|conference league|öfb cup|dfb pokal|dfb junioren pokal|copa del rey|coppa italia|french cup|us open cup|africa cup|greek cup)\b.*", "", out)
    out = out.replace("türkiye", "turkey").replace("türk", "turk")
    out = out.replace("usa", "united states")
    # Bare Bundesliga/2. Bundesliga rows (DFB-cup glue) are German by default
    if out in ("bundesliga", "in bundesliga"):
        out = "germany bundesliga"
    elif out == "2. bundesliga":
        out = "germany 2. bundesliga"
    # 'Super Liga/Lig/Süper' are the same competition family as 'Super League'
    out = re.sub(r"\bsuper\s+(liga|lig|ligi|süper)\b", "super league", out)
    out = re.sub(r"\bsüper lig\b", "super league", out)
    out = re.sub(r"\bhighlight.*", "", out)
    # Drop the 'Women' suffix from pair labels that also cover men's teams
    out = re.sub(r"\s+women\b", "", out)
    # Bring a '(country)' / trailing country token to the FRONT so cup-glued
    # variants ('in Bundesliga (Germany) ...', '2. Bundesliga Germany ...')
    # roll into the same family as the plain league rows.
    for country in _COUNTRIES:
        c2 = country.replace(" ", "-")
        out = re.sub(r"\(" + re.escape(country) + r"\)", country, out)
        out = re.sub(r"\(" + re.escape(c2) + r"\)", country, out)
        if re.search(r"\b" + re.escape(country) + r"\b", out):
            out = re.sub(r"\b" + re.escape(country) + r"\b", " ", out, count=1)
            out = (country + " " + re.sub(r"\s+", " ", out).strip()).strip()
            break
    # Strip scoop 'in' tokens the scraper can emit between country and league
    out = re.sub(r"\s+in\s+", " ", out)
    if re.match(r"^in\s+", out):
        rest = re.sub(r"^in\s+", "", out)
        if not rest.startswith(("regionalliga", "oberliga", "landesliga", "verbandsliga")):
            out = rest
    out = re.sub(r"\s+", " ", out).strip()
    if not out or len(out) < 3:
        return None
    if re.search(r"\d{1,2}/\d{1,2}/\d{4}", out):
        return None
    if " vs " in out:
        return None
    # Crumbs that are really a competition match pairing, not a league
    if "spieltag" in out or "round" in out or "fixture" in out or "super cup" in out:
        return None
    return out


def _league_key_clean(league):
    """Normalise a league key for accuracy lookups, or drop it when the
    stored league field is junk (scraper glues team names / dates into the
    league column, e.g. 'Br1 Botafogo Santos 17/07/2026')."""
    lg = _clean_league(league)
    if not lg:
        return None
    if re.search(r"\d{1,2}/\d{1,2}/\d{4}", lg):
        return None
    if " vs " in lg.lower():
        return None
    return lg


def _clean_league(league):
    """Strip Forebet's league-position prefixes so the league matches up with
    history.db league names (and league-accuracy lookups). The feed can carry
    BOTH teams' 'Nth place' tokens in the league string."""
    return re.sub(r"\b\d+(?:st|nd|rd|th) place\b", "", league or "").strip()


def _clean_team(name, date):
    """The feed sometimes glues the match date/time onto the away team name
    (scraper quirk). Drop everything from the first date token onward."""
    if not name:
        return name or "?"
    name = re.split(r"\d{2}/\d{2}/\d{4}", name)[0]
    return name.strip() or "?"


def _rel_factor(acc, league, market):
    pct = acc.get(market, {}).get(_league_family(league))
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


def build_card(matches):
    """Compute the bettor's card from a list of match dicts.

    Accepts either a `pr --json` prediction dump or the in-memory ``results``
    list of a prediction run (same shape).  Per candidate pick this computes:

      * fair (de-vigged) probability from bookmaker odds
      * edge %   = model_prob - fair_prob
      * EV %     = model_prob * odds - 1
      * stake %  = Quarter-Kelly capped at 5%, scaled by the league's
                   historical accuracy in that market (STUD/ok/WEAK/COLD)

    Returns {"value_picks": [...] only EV>0, "all_candidates": [...],
             "skipped_finished": n}.
    """
    acc = _league_market_accuracy()
    picks = []
    skipped_finished = 0

    for m in matches:
        league = _clean_league(m.get("league", "?"))
        home = _clean_team(m.get("home"), m.get("date"))
        away = _clean_team(m.get("away"), m.get("date"))
        label = f"{home} vs {away}"

        if m.get("actual_home_goals") is not None:
            skipped_finished += 1
            continue

        odds = {k: m.get(k) for k in ("odds_home", "odds_draw", "odds_away",
                                      "odds_over25", "odds_under25",
                                      "odds_btts_yes", "odds_btts_no")}
        odds = {k: v for k, v in odds.items() if isinstance(v, (int, float)) and v > 1.0}

        # Feed Forebet's own percentages to the odds validator: the
        # favourite-agreement test is the only check that catches prices which
        # are structurally plausible but assigned to the wrong outcomes.
        _FB_PCTS.clear()
        _pcts = m.get("forebet_pct")
        if isinstance(_pcts, (list, tuple)) and len(_pcts) == 3:
            _FB_PCTS.update({"home": _pcts[0], "draw": _pcts[1], "away": _pcts[2]})

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

            # Never price a market off prices that failed integrity checks.
            trusted, why = _market_prices_trustworthy(odds, mkt)
            if not trusted:
                entry["note"] = f"odds failed validation ({why}) — not priced"
                entry["odds_untrusted"] = True
                picks.append(entry)
                continue

            fprob = (fair.get(mkt) or {}).get(pick)
            fair_p = fprob if fprob else 1.0 / bp
            ev = prob * bp - 1
            edge = prob - fair_p
            rel, grade = _rel_factor(acc, league, mkt)
            stake = _stake(bp, prob, 0.25, 0.05) * rel
            note = ""
            if bp >= 4.0:
                note = "LONGSHOT"
            entry.update({
                "odds": bp, "fair_prob": round(fair_p, 4),
                "edge_pct": round(edge * 100, 2),
                "ev_pct": round(ev * 100, 2),
                "stake_pct": round(stake * 100, 2),
                "reliability": grade,
                "note": note,
            })
            picks.append(entry)

    value = [p for p in picks if "ev_pct" in p and p["ev_pct"] > 0]
    value.sort(key=lambda p: -p["ev_pct"])

    # ── make the ticket actually bettable ──────────────────────────────
    # (1) Drop legs the staking model prices at 0.0% — COLD-league scaling
    #     and negligible edge both mean "don't bet", not "bet nothing on".
    #     They stay visible in all_candidates for reference.
    # (2) One single per match — the best +EV leg.  Listing mutually
    #     exclusive outcomes of one fixture as separate singles (e.g. both
    #     1X2 Draw and 1X2 Away win) is contradictory and double-counts the
    #     same bankroll, and same-match legs are correlated bets, not
    #     independent ones.  Shadowed legs stay visible, annotated.
    # (3) Dust stakes (<DUST_STAKE_PCT) are below a rational minimum bet
    #     size — at 0.1-0.4% the rounding and model error swamp the edge, so
    #     the match is left unbet (its best +EV leg was dust).  Stays visible
    #     in all_candidates, annotated.
    # (4) A slate of simultaneous singles must never demand more than the
    #     bankroll it stakes against: cap the sum at 100% by scaling
    #     relative sizes down (keeps sizing ratios intact), then shave off
    #     any overage the per-leg rounding re-creates so the real total is
    #     never above 100%.
    _ticket = {}
    dropped_dust = 0
    for p in value:
        if p["stake_pct"] <= 0:
            continue
        prev = _ticket.get(p["match"])
        if prev is None:
            if p["stake_pct"] < DUST_STAKE_PCT:
                dropped_dust += 1
                note = (f"dust stake {p['stake_pct']:.1f}% (<{DUST_STAKE_PCT:.0f}%) — "
                        "below rational minimum bet size, left unbet")
                p["note"] = " | ".join(x for x in (p.get("note"), note) if x)
                continue
            _ticket[p["match"]] = p
        else:
            note = (f"shadowed by {prev['market']}:{prev['pick']} "
                    f"(best +EV on match; single-bet card keeps one leg per match)")
            p["note"] = " | ".join(x for x in (p.get("note"), note) if x)
    value = sorted(_ticket.values(), key=lambda p: -p["ev_pct"])

    scaled = False
    total_stake = sum(p["stake_pct"] for p in value)
    if total_stake > 100.0:
        factor = 100.0 / total_stake
        for p in value:
            p["stake_pct"] = round(p["stake_pct"] * factor, 2)
        scaled = True
    over = sum(p["stake_pct"] for p in value) - 100.0
    if over > 1e-9:
        over = round(over, 2)
        for p in sorted(value, key=lambda x: -x["stake_pct"]):
            if over <= 0:
                break
            avail = p["stake_pct"] - DUST_STAKE_PCT
            if avail <= 0:
                continue
            cut = min(avail, over)
            p["stake_pct"] = round(p["stake_pct"] - cut, 2)
            over = round(over - cut, 2)
    total_stake = round(sum(p["stake_pct"] for p in value), 2)
    shadowed = sum(1 for p in picks if "shadowed by" in (p.get("note") or ""))

    return {"value_picks": value, "all_candidates": picks,
            "skipped_finished": skipped_finished,
            "total_stake_pct": total_stake,
            "scaled_down": scaled, "shadowed": shadowed,
            "dropped_dust": dropped_dust}


def cmd_card(args):
    matches = _load_matches(args.file)
    card = build_card(matches)
    picks = card["all_candidates"]
    value = card["value_picks"]
    skipped_finished = card["skipped_finished"]

    if args.json:
        out = {"value_picks": value, "all_candidates": picks, "skipped_finished": skipped_finished,
               "total_stake_pct": card.get("total_stake_pct"),
               "scaled_down": card.get("scaled_down"), "shadowed": card.get("shadowed"),
               "dropped_dust": card.get("dropped_dust")}
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    print("=" * 78)
    print("BETTOR'S CARD  (Quarter-Kelly, max 5%, scaled by league reliability)")
    print("=" * 78)
    if skipped_finished:
        print(f"  ({skipped_finished} finished match(es) excluded — no betting hindsight)")
    if not value:
        print("\n  No +EV picks in this slate. The sharp move is to BET NOTHING.")
    else:
        hdr = f"{'LEAGUE':<24}{'MATCH':<26}{'MARKET/PICK':<22}{'EV%':>6}{'EDGE%':>7}{'ODDS':>6}{'STAKE%':>7} {'NOTE'}"
        print("\n" + hdr)
        print("-" * 78)
        for p in value:
            mkt = f"{p['market']}:{p['pick']}"
            notes = " ".join(x for x in (p.get("note"), f"[{p['reliability']}]" if p["reliability"] not in ("STUD", "ok") else "") if x)
            print(
                f"{p['league'][:23]:<24}{p['match'][:25]:<26}{mkt[:21]:<22}"
                f"{p['ev_pct']:>6}{p['edge_pct']:>7}{p['odds']:>6}{p['stake_pct']:>7} {notes}"
            )
        nticket = sum(p["stake_pct"] for p in value)
        print(f"\n  Ticket: {len(value)} single(s) — one/best +EV leg per match, "
              f"total {nticket:.1f}% of bankroll"
              + (" (scaled down to the 100% cap)" if card.get("scaled_down") else "")
              + f" | {card.get('shadowed', 0)} same-match leg(s) shadowed"
              + (f" | {card.get('dropped_dust', 0)} dust leg(s) (<{DUST_STAKE_PCT:.0f}%) left unbet"
                 if card.get("dropped_dust") else ""))
    if args.all:
        passed = {id(p) for p in value}
        losers = [p for p in picks if id(p) not in passed and "note" not in p]
        if losers:
            print(f"\n{'All candidates (incl. -EV, for reference)':-^78}")
            for p in losers[:15]:
                mkt = f"{p['market']}:{p['pick']}"
                notes = " ".join(x for x in (p.get("note"), f"[{p['reliability']}]" if p["reliability"] not in ("STUD", "ok") else "") if x)
                print(f"{p['league'][:23]:<24}{p['match'][:25]:<26}{mkt[:21]:<22}{p['ev_pct']:>6}{p['edge_pct']:>7}{p['odds']:>6} {notes}")
    print()


# ─────────────────────────────────────────────
# slip  — leg-by-leg verdict for a multi-bet
# ─────────────────────────────────────────────

_PAGE_STYLE = """
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#0f172a; color:#e2e8f0; padding:20px; }
h1 { font-size:1.4rem; margin-bottom:4px; }
h2 { font-size:1.1rem; margin:16px 0 8px; color:#94a3b8; }
.sub { color:#94a3b8; font-size:0.85rem; margin-bottom:20px; }
.stats { display:flex; gap:16px; flex-wrap:wrap; margin-bottom:20px; }
.stat { background:#1e293b; padding:10px 16px; border-radius:8px; font-size:0.85rem; }
.stat span { font-weight:700; font-size:1.1rem; }
table { width:100%; border-collapse:collapse; margin:8px 0; }
th, td { text-align:left; padding:8px 12px; }
th { color:#94a3b8; font-weight:500; border-bottom:1px solid #334155; }
tr:hover { background:#1e293b; }
.conf-badge { font-size:0.75rem; padding:2px 8px; border-radius:4px; color:#fff; font-weight:600; white-space:nowrap; }
a { color:#60a5fa; text-decoration:none; }
a:hover { text-decoration:underline; }
details { margin-top:18px; background:#0b1425; border:1px solid #1e293b; border-radius:8px; padding:12px 14px; }
summary { cursor:pointer; color:#60a5fa; font-weight:500; }
.v-badge { font-size:0.75rem; padding:2px 10px; border-radius:4px; color:#fff; font-weight:700; }
.v-BET { background:#16a34a; } .v-PASS { background:#64748b; }
.v-KILL { background:#dc2626; } .v-NO-DATA { background:#d97706; }
@media (max-width:700px) { body { padding:10px; } .stats { flex-direction:column; } }
"""

# Slip pick tokens: "3-Way Home @2.03", "Over 2.5 @1.48", "BTTS Yes", ...
_PICK_RE = re.compile(
    r"(?:(?:3-Way|Winner)\s+)?(?:Home|Draw|Away)(?:\s+win)?"
    r"|(?:Over|Under)\s+\d+(?:\.\d+)?"
    r"|BTTS\s+(?:Yes|No)"
    r"|Double\s+Chance\s+(?:1X|12|X2|1|2|X)",
    re.IGNORECASE,
)

_THR_KEYS = {"1.5": "15", "2.5": "25", "3.5": "35"}

# Reserve / second / youth team detection lives in reserve_filter at the repo
# root so predictions/picks.html applies exactly the same rule.  It used to be
# private to this file, which let the report publish picks on Betis III,
# UD Almeria B and Pontevedra II -- precisely the fixtures where the model's
# senior-team inputs do not describe who is actually playing.  The names below
# stay bound here for backwards compatibility.
from reserve_filter import (  # noqa: E402
    _RESERVE_RE, _SENIOR_CLUB_ALLOW, reserve_reason, is_reserve_fixture,
)

# Only these flags block a +EV leg from becoming a BET.  Other flags
# ("de-vigged line", "prob via complement line") are methodology notes —
# they document how the numbers were made, they do not kill the bet.
RED_FLAGS = ("reserve/second team", "league unverified (n<20)", "LONGSHOT")


def _canon_pick(txt):
    """Canonicalise a slip pick token to the (market, pick) vocabulary the
    pr --json dump uses, e.g. ("1X2", "Home win"), ("O/U", "Over 2.5")."""
    t = txt.strip().lower()
    if t.startswith(("over", "under")):
        thr = re.search(r"\d+(?:\.\d+)?", t).group(0)
        return "O/U", f"{'Over' if t.startswith('over') else 'Under'} {thr}"
    if t.startswith("btts"):
        return "BTTS", "Yes" if "yes" in t else "No"
    if t.startswith("double chance"):
        side = re.sub(r"double chance\s*", "", t).upper()
        return "DC", {"1X": "Home or Draw", "12": "Home or Away",
                      "X2": "Draw or Away", "1": "Home", "2": "Away"}.get(side, side)
    if re.search(r"home|draw|away", t):
        side = re.search(r"home|draw|away", t).group(0)
        return "1X2", {"home": "Home win", "draw": "Draw",
                       "away": "Away win"}[side]
    return None, None


def _parse_slip(text):
    """Parse a multi-bet slip file into legs — one per line.

      Team A - Team B, 3-Way Home @2.03      (odds optional)
      Team C - Team D, Over 2.5 @1.48
      Team E - Team F, BTTS Yes
      # comment lines are ignored
    """
    legs = []
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.search(r"@(\d+(?:\.\d+)?)", line)
        odds = float(m.group(1)) if m else None
        if odds is not None:
            line = line[:m.start()].strip(" |,;:-–—")
        pm = _PICK_RE.search(line)
        if not pm:
            legs.append({"leg": n, "home": line, "away": "", "market": None,
                         "pick": None, "odds": odds, "raw": raw.strip()})
            continue
        pick_txt = pm.group(0).strip()
        market, pick = _canon_pick(pick_txt)
        teams = line[:pm.start()].strip(" |,;:-–—")
        sep = re.search(r"\s+[-–—]\s+|,?\s+vs\.?\s+", teams, re.IGNORECASE)
        home = teams[:sep.start()].strip() if sep else teams
        away = teams[sep.end():].strip() if sep else ""
        legs.append({"leg": n, "home": home, "away": away, "market": market,
                     "pick": pick, "odds": odds, "raw": raw.strip()})
    return legs


@lru_cache(maxsize=65536)
def _norm_team(name):
    s = (name or "").lower()
    # scraper glues the kickoff date/time onto team names
    # ("New Mexico United17/09/2026 02:30 GMT") — drop it before anything else
    s = re.split(r"\d{1,2}/\d{1,2}/\d{4}", s)[0]
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(
        r"\b(fc|sc|cf|af|ac|cd|ud|sd|as|ss|us|nk|fk|vfb|bsc|deportivo|deportes|"
        r"club|association|reserves)\b", " ", s)
    return " ".join(t for t in s.split() if len(t) > 1)


def _team_sim(a, b):
    ta, tb = _norm_team(a).split(), _norm_team(b).split()
    if not ta or not tb:
        return 0.0
    return len(set(ta) & set(tb)) / max(len(set(ta)), len(set(tb)))


def _fixture_score(leg, cand):
    sh = _team_sim(leg["home"], cand.get("home_team") or cand.get("home") or "")
    sa = _team_sim(leg["away"], cand.get("away_team") or cand.get("away") or "")
    return 0.6 * sh + 0.4 * sa


def _resolve_fixtures(legs, preds=None):
    """Attach the best-matching pipeline fixture to each leg.

    Fixture pool: data/pending_results.json (upcoming scraped slate),
    history.db matches already predicted but not finished, and (optionally)
    a pr --json dump (--preds or a fresh scrape).  A leg needs a name-score
    >= 0.60 to count as resolved; anything below that is unverifiable."""
    pending = []
    try:
        pending = json.loads((REPO / "data" / "pending_results.json").read_text())
    except Exception:
        pass
    pool = list(pending)
    if isinstance(preds, list):
        pool += [{"home_team": p.get("home"), "away_team": p.get("away"),
                  "forebet_url": p.get("forebet_url"), "_preds": p} for p in preds]
    for leg in legs:
        for name in (leg["home"], leg["away"]):
            toks = [t for t in re.split(r"\W+", (name or "").lower()) if len(t) > 3]
            for tok in toks[-1:]:
                rows = _try(
                    "SELECT * FROM matches WHERE actual_home_goals IS NULL "
                    "AND (home_team LIKE ? OR away_team LIKE ?) "
                    "ORDER BY id DESC LIMIT 10", (f"%{tok}%", f"%{tok}%"))
                if isinstance(rows, list):
                    pool += [dict(r, _row=True) for r in rows]
    out = []
    for leg in legs:
        best, bs = None, 0.6
        for cand in pool:
            # prefer a fresh pr dump, then a DB row, over a bare pending entry
            bonus = (0.05 if cand.get("_preds") else 0.0) + (0.02 if cand.get("_row") else 0.0)
            s = _fixture_score(leg, cand) + bonus
            if s > bs:
                # both team names must actually match — a strong home match
                # with a zero away match is a different fixture
                sh = _team_sim(leg["home"], cand.get("home_team") or cand.get("home") or "")
                sa = _team_sim(leg["away"], cand.get("away_team") or cand.get("away") or "")
                if sh >= 0.4 and sa >= 0.4:
                    best, bs = cand, s
        out.append((leg, best, round(bs, 2)))
    return out


def _fixture_preds(fx):
    return fx.get("_preds") if fx else None


def _fixture_primary(fx):
    src = fx.get("_preds") or fx
    return (src.get("pick") or src.get("our_prediction"),
            src.get("market") or src.get("our_market"))


def _fixture_primary_conf(fx):
    """Confidence of the model's primary pick, or None when unavailable."""
    src = fx.get("_preds") or fx
    conf = src.get("confidence") or src.get("our_confidence")
    if conf:
        return conf
    # Fall back to the top candidate's confidence when the headline field
    # is absent but a ranked pick list survived.
    cands = _fixture_candidates(fx)
    if cands:
        return cands[0].get("confidence")
    return None


def _fixture_candidates(fx):
    """Candidate picks with model_prob for a resolved fixture.

    Merges, in order of preference:
      1) the pr dump's per-market synthesis candidates (all_picks)
      2) the ML analysis market candidates (_ml_analysis.all_picks — the
         O/U lines at every threshold live here)
      3) the raw ensemble probs broadcast from _ml (1X2 + the 2.5 line)
    DB rows fall back to their stored ML 1X2 probs."""
    src = fx.get("_preds") or fx
    merged = {}
    for c in (src.get("all_picks") or []):
        if c.get("market") and c.get("pick") and c.get("model_prob") is not None:
            merged.setdefault((c["market"], c["pick"]), c)
    for c in ((src.get("_ml_analysis") or {}).get("all_picks") or []):
        if c.get("market") and c.get("pick") and c.get("model_prob") is not None:
            merged.setdefault((c["market"], c["pick"]), c)
    ml = src.get("_ml") or {}
    for pick, key in (("Home win", "ml_prob_home"), ("Draw", "ml_prob_draw"),
                      ("Away win", "ml_prob_away")):
        if isinstance(ml.get(key), (int, float)):
            merged.setdefault(("1X2", pick),
                              {"market": "1X2", "pick": pick,
                               "model_prob": ml[key], "confidence": None})
    for pick, key in (("Over 2.5", "ml_prob_over"), ("Under 2.5", "ml_prob_under")):
        if isinstance(ml.get(key), (int, float)):
            merged.setdefault(("O/U", pick),
                              {"market": "O/U", "pick": pick,
                               "model_prob": ml[key], "confidence": None})
    if merged:
        return list(merged.values())
    if fx.get("_row"):
        out = []
        for side, key in (("Home win", "ml_prob_home"), ("Draw", "ml_prob_draw"),
                          ("Away win", "ml_prob_away")):
            v = fx.get(key)
            if isinstance(v, (int, float)):
                out.append({"market": "1X2", "pick": side, "model_prob": v,
                            "confidence": fx.get("our_confidence")})
        return out
    return []


def _fixture_odds(fx):
    src = fx.get("_preds") or fx
    return {k: v for k, v in src.items()
            if k.startswith("odds_") and isinstance(v, (int, float)) and v > 1.0}


def _market_odds_map(market, pick):
    if market == "1X2":
        return {"Home win": "odds_home", "Draw": "odds_draw",
                "Away win": "odds_away"}
    if market == "BTTS":
        return {"Yes": "odds_btts_yes", "No": "odds_btts_no"}
    if market == "O/U":
        m = re.search(r"\d+(?:\.\d+)?", pick)
        thr = m.group(0) if m else "2.5"
        k = _THR_KEYS.get(thr)
        if k:
            return {f"Over {thr}": f"odds_over{k}", f"Under {thr}": f"odds_under{k}"}
    return {}


def _model_prob_for(leg, cands):
    """model_prob for the slip pick; also tries the complementary line
    (O/U and BTTS are binary — P(Over) = 1 - P(Under))."""
    for c in cands:
        if c["market"] == leg["market"] and c["pick"] == leg["pick"]:
            return c["model_prob"], c, False
    comp = None
    if leg["market"] == "O/U":
        m = re.match(r"(Over|Under)\s+(\d+(?:\.\d+)?)", leg["pick"])
        if m:
            comp = f"{'Under' if m.group(1) == 'Over' else 'Over'} {m.group(2)}"
    elif leg["market"] == "BTTS":
        comp = "No" if leg["pick"] == "Yes" else "Yes"
    if comp:
        for c in cands:
            if (c["market"] == leg["market"] and c["pick"] == comp
                    and 0 < c["model_prob"] < 1):
                pc = dict(c, model_prob=1.0 - c["model_prob"])
                return pc["model_prob"], pc, True
    return None, None, False


def _dir_of(pick):
    p = (pick or "").upper()
    for d in ("HOME", "AWAY", "DRAW"):
        if d in p:
            return d
    return None


def _analyze_leg(leg, fx, acc):
    """Price one slip leg: de-vigged fair prob, EV, Quarter-Kelly stake and
    a verdict (BET / PASS / KILL / NO-DATA) with a human-readable why.

    Decisive verdicts only come from pipeline evidence — anything that
    needs outside knowledge (venue, injuries, phantom fixtures) is flagged
    for the agent's web layer instead."""
    e = {"leg": leg["leg"],
         "match": f"{leg['home']} vs {leg['away']}" if leg["away"] else leg["home"],
         "market": leg["market"], "pick": leg["pick"], "slip_odds": leg["odds"],
         "raw": leg["raw"], "league": None, "date": None,
         "model_prob": None, "fair_prob": None, "odds": None,
         "edge_pct": None, "ev_pct": None, "confidence": None,
         "stake_pct": 0.0, "reliability": None, "verdict": "NO-DATA",
         "why": "", "flags": []}
    if leg["market"] is None:
        e["why"] = "unrecognised pick token — check the slip line"
        return e
    if fx is None:
        e["verdict"] = "NO-DATA"
        e["why"] = "fixture not found in pipeline — unverifiable (phantom or un-scraped league)"
        return e

    src = fx.get("_preds") or fx
    e["league"] = _clean_league(src.get("league") or fx.get("league"))
    e["date"] = src.get("date") or fx.get("match_date")
    if src.get("actual_home_goals") is not None:
        e["verdict"] = "KILL"
        e["why"] = "already finished — not a bet"
        return e

    # direction conflict against the model's primary pick — decisive even
    # when no probability survived into the pipeline.  Only KILL when the
    # primary carry genuine conviction: a Low/Medium primary is "no signal"
    # (research/web layer decides), so conflicting legs degrade to PASS, not
    # KILL.  Medium-High+ primaries veto (they are the calibrated picks).
    prim_pick, prim_mkt = _fixture_primary(fx)
    prim_conf = _fixture_primary_conf(fx)
    _prim_vetoes = prim_conf in ("Medium-High", "High", "Near Certain", None)
    d1 = _dir_of(prim_pick) if prim_pick and prim_mkt in ("1X2", "1H", "2H") else None
    d2 = _dir_of(leg["pick"]) if leg["market"] == "1X2" else None
    if d1 and d2 and d1 != d2:
        if _prim_vetoes:
            e["verdict"] = "KILL"
            e["why"] = f"model's pick is {prim_pick} ({prim_mkt}), slip has {leg['pick']}"
        else:
            e["verdict"] = "PASS"
            e["why"] = (f"conflicts {prim_pick} ({prim_mkt}, {prim_conf or 'conf Low'}) "
                        f"— Low/Medium primary does not veto, research decides")
            e["flags"].append("conflicts low-conf model primary")
        return e
    if prim_pick and prim_mkt == "O/U" and leg["market"] == "O/U":
        m1 = re.match(r"(Over|Under)", prim_pick)
        m2 = re.match(r"(Over|Under)", leg["pick"])
        if m1 and m2 and m1.group(1) != m2.group(1):
            if _prim_vetoes:
                e["verdict"] = "KILL"
                e["why"] = f"model leans {prim_pick}, slip has {leg['pick']}"
            else:
                e["verdict"] = "PASS"
                e["why"] = (f"conflicts {prim_pick} ({prim_mkt}, {prim_conf or 'conf Low'}) "
                            f"— Low/Medium primary does not veto, research decides")
                e["flags"].append("conflicts low-conf model primary")
            return e

    odds = _fixture_odds(fx)
    cands = _fixture_candidates(fx)
    prob, cand, inferred = _model_prob_for(leg, cands)
    if prob is None:
        e["why"] = "no model probability in pipeline"
        if fx.get("_row") and not fx.get("_preds"):
            e["why"] += " (DB row stores 1X2 only — rerun with --preds or live scrape for O/U/BTTS)"
        elif not odds:
            e["why"] += " and no feed odds — cannot price"
        return e
    e["model_prob"] = round(prob, 4)
    e["confidence"] = cand.get("confidence")
    if inferred:
        e["flags"].append("prob via complement line")

    mkt_map = _market_odds_map(leg["market"], leg["pick"])
    bp = leg["odds"]
    if bp is None and mkt_map:
        bp = odds.get(mkt_map.get(leg["pick"]))
    if not bp:
        e["why"] = "no odds for this pick — cannot price EV"
        return e
    bp = float(bp)
    e["odds"] = round(bp, 2)

    present = {p: odds.get(k) for p, k in mkt_map.items() if odds.get(k)}
    fair = _devig(present).get(leg["pick"]) if len(present) >= 2 else None
    if fair is None:
        fair = 1.0 / bp
    else:
        e["flags"].append("de-vigged line")
    e["fair_prob"] = round(fair, 4)

    ev = prob * bp - 1
    edge = prob - fair
    e["edge_pct"], e["ev_pct"] = round(edge * 100, 2), round(ev * 100, 2)

    rel, grade = _rel_factor(acc, e["league"], leg["market"])
    e["reliability"] = grade
    e["_rel"] = rel
    e["stake_pct"] = round(_stake(bp, prob, 0.25, 0.05) * rel * 100, 2)

    both = f"{leg['home']} {leg['away']}"
    if is_reserve_fixture(leg.get("home", ""), leg.get("away", "")):
        e["flags"].append("reserve/second team")
    if not e["league"] or _league_family(e["league"]) not in (acc.get(leg["market"]) or {}):
        e["flags"].append("league unverified (n<20)")
    if bp >= 4.0:
        e["flags"].append("LONGSHOT")
    for rf in (src.get("_red_flags") or []):
        msg = rf.get("msg") if isinstance(rf, dict) else str(rf)
        if msg:
            e["flags"].append("pipeline red flag: " + msg[:44])

    # direction conflict was already checked earlier — what remains is pricing
    if ev <= -0.05:
        e["verdict"] = "KILL"
        e["why"] = (f"bad price — EV {ev*100:+.0f}% "
                    f"(model {prob*100:.0f}% vs fair {fair*100:.0f}%)")
    elif ev >= 0.03 and cand.get("confidence") not in (None, "Low") and grade != "COLD":
        reds = [f for f in e["flags"] if f in RED_FLAGS or f.startswith("pipeline red flag:")]
        if reds:
            e["verdict"] = "PASS"
            e["why"] = f"+EV but red flags: {', '.join(reds)} (needs agent web check)"
        else:
            e["verdict"] = "BET"
            e["why"] = f"model {prob*100:.0f}% vs fair {fair*100:.0f}% — EV {ev*100:+.1f}%"
    else:
        bits = [f"EV {ev*100:+.1f}% below +3% bar"]
        if cand.get("confidence") in (None, "Low"):
            bits.append("conf Low")
        if grade == "COLD":
            bits.append("COLD league")
        e["verdict"] = "PASS"
        e["why"] = ", ".join(bits)
    return e


# ─────────────────────────────────────────────
# research layer (opencode is the research agent)
# ─────────────────────────────────────────────

def _evidence_valid(ev):
    """An evidence file is trusted only when it carries the mandatory fields
    (real fixture + a findings summary) and no unknown top-level keys."""
    fx = ev.get("fixture") or {}
    fnd = ev.get("findings") or {}
    if not isinstance(fx.get("real"), bool):
        return False
    if fx.get("real") is True and not fx.get("competition"):
        return False
    if not fnd.get("summary"):
        return False
    allowed = set(EVIDENCE_REQUIRED) | {"fixture", "findings"} | _ALWAYS_ALLOWED_EVIDENCE
    for k in ev:
        if k != "slug" and k not in allowed:
            return False
    return True


def _slug_of(entry):
    m = entry.get("match") or ""
    m = re.sub(r"[^a-z0-9]+", "-", m.lower())
    s = f"{m}-{str(entry.get('date') or '').replace('/','-')}"
    return s.strip("-")


def _evidence_path(slug):
    return RESEARCH_DIR / f"{slug}.json"


def _read_evidence(slug, strict=True):
    p = _evidence_path(slug)
    if not p.exists():
        return None
    try:
        ev = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return {"_error": "unreadable JSON"}
    if strict and not _evidence_valid(ev):
        return {"_error": f"missing required evidence fields ({EVIDENCE_REQUIRED})"}
    return ev


def _evidence_haircut(ev, entry):
    """Fraction of the pick's prob that the evidence removes (0 = none).

    Never boosts: only side-directional absences (1X2 Home/Away), scorer
    absences on Over/Yes, and a neutral venue for Home picks can cut."""
    if not ev or ev.get("_error"):
        return 0.0, [], "no usable evidence"
    fx = ev.get("fixture") or {}
    fnd = ev.get("findings") or {}
    notes = []
    absent = fnd.get("absences") or []
    mkt = entry.get("market")
    pick = (entry.get("pick") or "").upper()

    sum_w = 0.0
    if mkt == "1X2":
        team = "home" if "HOME" in pick else ("away" if "AWAY" in pick else None)
        if not team:
            return 0.0, notes, "draw pick — absences favour the draw (never boost)"
        for ab in (a for a in absent if (a.get("team") or "") == team):
            w = ROLE_WEIGHTS.get(ab.get("role") or "unknown", ROLE_WEIGHTS["unknown"])
            sum_w += w
            notes.append(f"{team} {'—'.join(x for x in (ab.get('player'), ab.get('role')) if x)} out")
    elif mkt == "O/U":
        ov = pick.split(None, 1)[0].upper()
        if ov not in ("OVER", "YES"):
            return 0.0, notes, f"{ov} pick — absences usually help, never boost"
        for ab in absent:
            role = ab.get("role") or "unknown"
            if role in ("forward", "midfielder", "unknown"):
                sum_w += ROLE_WEIGHTS.get(role, ROLE_WEIGHTS["unknown"]) * 0.5
                notes.append(f"{ab.get('team')} {ab.get('player') or role} out (scorer)")
    elif mkt == "BTTS":
        if pick != "YES":
            return 0.0, notes, "BTTS No — absences usually help, never boost"
        for ab in absent:
            role = ab.get("role") or "unknown"
            if role in ("forward", "midfielder", "unknown"):
                sum_w += ROLE_WEIGHTS.get(role, ROLE_WEIGHTS["unknown"]) * 0.5
                notes.append(f"{ab.get('team')} {ab.get('player') or role} out (scorer)")
    else:
        return 0.0, notes, f"{mkt} outside research haircut scope"

    cap_notes = []
    if sum_w > ABSENCE_CAP:
        cap_notes.append(f"absence haircut capped at {ABSENCE_CAP:.2f}")
        sum_w = ABSENCE_CAP

    rel_cut = 0.0
    if mkt == "1X2" and "HOME" in pick:
        venue = (fx.get("venue") or "home").lower()
        if venue == "neutral":
            rel_cut += NEUTRAL_VENUE_CUT
            notes.append("neutral venue — home edge off")
    rel_cut += sum_w
    return rel_cut, notes, " | ".join(cap_notes)


def _apply_research_to_prob(prob, rel_cut):
    if prob is None:
        return None, 0.0
    if rel_cut <= 0:
        return prob, rel_cut
    return max(prob * (1.0 - rel_cut), 0.05), rel_cut


def _merge_research(entry, ev):
    """Re-price a leg after research evidence.  Research never flips a NO-DATA
    or KILL leg to BET — it can only confirm, downgrade, or kill."""
    if entry.get("verdict") in ("KILL", "NO-DATA"):
        e = dict(entry)
        if ev and not ev.get("_error") and ev.get("fixture", {}).get("real") is False:
            e["verdict"] = "KILL"
            e["why"] = "phantom fixture confirmed — not a real match"
            e["flags"] = e["flags"] + ["research: phantom fixture"]
        elif ev and ev.get("_error"):
            e["flags"] = e["flags"] + [f"research error: {ev['_error']}"]
        return e

    e = dict(entry)
    if ev is None:
        return e
    if ev.get("_error"):
        e["flags"] = e["flags"] + [f"research error: {ev['_error']}"]
        return e
    fx = ev.get("fixture") or {}
    if fx.get("real") is False:
        e["verdict"] = "KILL"
        e["why"] = "phantom fixture confirmed — not a real match"
        e["flags"] = e["flags"] + ["research: phantom fixture"]
        return e

    rel_cut, notes, _cap = _evidence_haircut(ev, entry)
    new_prob, applied = _apply_research_to_prob(entry.get("model_prob"), rel_cut)
    if not applied:
        e["flags"] = e["flags"] + ["research: " + n for n in notes] if notes else e["flags"]
        if entry.get("verdict") == "BET":
            e["why"] += " · research confirmed"
        return e

    e["model_prob"] = round(new_prob, 4)
    e["flags"] = e["flags"] + ["research: " + n for n in notes]
    if e.get("fair_prob"):
        e["fair_prob"] = round(min(new_prob, e["fair_prob"]), 4)
    if e.get("odds"):
        ev_pct = round((new_prob * e["odds"] - 1) * 100, 2)
        e["ev_pct"] = ev_pct
        e["edge_pct"] = round((new_prob - e["fair_prob"]) * 100, 2)
        e["stake_pct"] = round(_stake(e["odds"], new_prob, 0.25, 0.05)
                               * e.get("_rel", 1.0) * 100, 2)
        reason = f"research haircut — prob {entry['model_prob']*100:.0f}% → {new_prob*100:.0f}%: {', '.join(notes)}"
        if ev_pct / 100 <= -0.05:
            e["verdict"] = "KILL"
            e["why"] = f"bad price after research — EV {ev_pct:+.0f}%"
        elif ev_pct / 100 >= 0.03 and entry.get("confidence") not in (None, "Low") \
                and entry.get("reliability") != "COLD":
            if any(f in RED_FLAGS or f.startswith("pipeline red flag:") for f in e["flags"]):
                e["verdict"] = "PASS"
                e["why"] = f"+EV but red flags persist after research ({reason})"
            else:
                e["verdict"] = "BET"
                e["why"] = f"{reason} — still {ev_pct:+.1f}% EV"
        else:
            e["verdict"] = "PASS"
            e["why"] = f"{reason} — dropped below +3% bar (EV {ev_pct:+.1f}%)"
    return e


def worklist_for(entries):
    """Research agenda for opencode: per leg, what to verify on the web and
    the evidence file slug to write to.  KILL legs are skipped."""
    out = []
    for e in entries:
        if e.get("verdict") == "KILL":
            continue
        slug = _slug_of(e)
        tasks = []
        if e.get("verdict") == "NO-DATA":
            tasks = [
                "verify this fixture is REAL (competition + date + teams)",
                "find bookmaker odds for the slip pick (or nearest line)",
                "note the venue (home / neutral / away)",
            ]
        elif e.get("verdict") == "PASS":
            reds = [f for f in e.get("flags", [])
                    if f in RED_FLAGS or f.startswith("pipeline red flag:")]
            tasks.append("check whether flagged red flags are real (reserve team, unverified league, longshot)")
            mkt, pick = e.get("market"), (e.get("pick") or "").upper()
            if "HOME" in pick or "AWAY" in pick:
                side = "home" if "HOME" in pick else "away"
                tasks.append(f"confirm {side} team absences (role + player) and whether venue is neutral")
            elif mkt in ("O/U", "BTTS") and (pick.startswith("OVER") or pick == "YES"):
                tasks.append("confirm scorer absences (forwards/mids) on BOTH teams")
        else:  # BET
            mkt, pick = e.get("market"), (e.get("pick") or "").upper()
            if "HOME" in pick or "AWAY" in pick:
                side = "home" if "HOME" in pick else "away"
                tasks.append(f"confirm {side} team absences (role + player) and that venue is truly their home")
            elif mkt in ("O/U", "BTTS") and (pick.startswith("OVER") or pick == "YES"):
                tasks.append("confirm scorer absences (forwards/mids) on BOTH teams")
            tasks.append("sanity-check the odds still match the slip")
        out.append({
            "slug": slug,
            "leg": e["leg"], "match": e["match"],
            "pick": f"{e.get('market')}:{e.get('pick')}" if e.get("market") else "",
            "verdict": e.get("verdict"), "why": e.get("why", ""),
            "tasks": tasks,
        })
    return out


def cmd_research(args):
    """research <slip> [--preds] [--no-model] [--json]

    Default: resolve the slip, print a research agenda, and write the an
    evidence skeleton file for every leg that needs a web check.  opencode
    runs the web searches, fills each data/research/<slug>.json, then runs
    `research --apply <slip>` to merge the evidence back into the verdict."""
    p = Path(args.file)
    if not p.exists():
        sys.exit(f"slip file not found: {p}")
    legs = _parse_slip(p.read_text(encoding="utf-8", errors="replace"))
    if not legs:
        sys.exit("no legs parsed from slip file")
    acc = _league_market_accuracy()

    preds_dump = None
    if args.preds:
        pp = Path(args.preds)
        if pp.exists():
            preds_dump = _load_json_robust(pp.read_text())
    resolved = _resolve_fixtures(legs, preds=preds_dump)

    if not args.no_model and preds_dump is None:
        need = [fx for leg, fx, _s in resolved
                if fx and fx.get("forebet_url") and fx.get("actual_home_goals") is None
                and not fx.get("_preds") and not _fixture_candidates(fx)]
        if need:
            links_file = REPO / "data" / "slip_links.txt"
            links_file.write_text("\n".join(sorted({fx["forebet_url"] for fx in need})) + "\n")
            try:
                dump = _run_pr_json(str(links_file))
                resolved = _resolve_fixtures(legs, preds=dump)
            except SystemExit:
                print("[research] pr scrape failed — falling back to DB data only",
                      file=sys.stderr)

    entries = [_analyze_leg(leg, fx, acc) for leg, fx, _s in resolved]
    for (leg, fx, score), e in zip(resolved, entries):
        e["match_score"] = score

    counts = {"BET": 0, "PASS": 0, "NO-DATA": 0, "KILL": 0}
    for e in entries:
        counts[e["verdict"]] += 1

    if not args.apply:
        worklist = worklist_for(entries)
        RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
        for w in worklist:
            skel = dict(_EVIDENCE_TEMPLATE)
            skel["slug"] = w["slug"]
            fx = skel["fixture"]
            mm = re.match(r"(.+?)(?:\s+vs\.?\s+|\s+-\s+)(.*)", w["match"])
            if mm:
                fx["_teams"] = [mm.group(1).strip(), mm.group(2).strip()]
            path = _evidence_path(w["slug"])
            if not path.exists():
                path.write_text(json.dumps(skel, indent=2, ensure_ascii=False))
        if args.json:
            print(json.dumps({"legs": entries, "summary": {"total": len(entries), **counts},
                              "worklist": worklist}, indent=2, ensure_ascii=False))
            return
        print("RESEARCH AGENDA — fill data/research/<slug>.json, then run "
              "`research --apply <slip>`")
        print("=" * 100)
        for w in worklist:
            print(f"\n[{w['leg']}] {w['match']:<36} {w['pick']:<16} "
                  f"{w['verdict']:<10} {w['why'][:52]}")
            for t in w["tasks"]:
                print(f"  ☐ {t}")
            print(f"  → {_evidence_path(w['slug'])}")
        print("\nLegs skipped (KILL — no research needed): "
              f"{counts['KILL']}  |  already finished: n/a")
        return

    # ── --apply: merge saved evidence back into the verdict ──
    merged = [_merge_research(e, _read_evidence(_slug_of(e))) for e in entries]
    m_counts = {"BET": 0, "PASS": 0, "NO-DATA": 0, "KILL": 0}
    for e in merged:
        m_counts[e["verdict"]] += 1
    m_bets = sorted([e for e in merged if e["verdict"] == "BET"],
                    key=lambda e: -e["ev_pct"])
    m_total = sum(e["stake_pct"] for e in m_bets)

    if args.json:
        print(json.dumps({"legs": merged, "summary": {"total": len(merged), **m_counts},
                          "action": m_bets, "total_stake_pct": round(m_total, 2)},
                         indent=2, ensure_ascii=False))
        return

    _print_slip_verdict(merged, m_bets, m_counts,
                        title="SLIP AFTER RESEARCH",
                        header_extra=f"{counts['BET']}→{m_counts['BET']} BET   "
                                     f"{counts['KILL']}→{m_counts['KILL']} KILL   "
                                     f"after evidence merge")


def cmd_slip(args):
    p = Path(args.file)
    if not p.exists():
        sys.exit(f"slip file not found: {p}")
    legs = _parse_slip(p.read_text(encoding="utf-8", errors="replace"))
    if not legs:
        sys.exit("no legs parsed from slip file")
    acc = _league_market_accuracy()

    preds_dump = None
    if args.preds:
        pp = Path(args.preds)
        if pp.exists():
            preds_dump = _load_json_robust(pp.read_text())
    resolved = _resolve_fixtures(legs, preds=preds_dump)

    # scrape model data for resolved, unfinished fixtures that carry no
    # candidate probabilities yet (bare pending entries, or DB rows without
    # stored probs — O/U/BTTS candidates only live in the pr dump)
    if not args.no_model and preds_dump is None:
        need = [fx for leg, fx, _s in resolved
                if fx and fx.get("forebet_url") and fx.get("actual_home_goals") is None
                and not fx.get("_preds") and not _fixture_candidates(fx)]
        if need:
            links_file = REPO / "data" / "slip_links.txt"
            links_file.write_text("\n".join(sorted({fx["forebet_url"] for fx in need})) + "\n")
            try:
                dump = _run_pr_json(str(links_file))
                resolved = _resolve_fixtures(legs, preds=dump)
            except SystemExit:
                print("[slip] pr scrape failed — falling back to DB data only",
                      file=sys.stderr)

    entries = [_analyze_leg(leg, fx, acc) for leg, fx, _s in resolved]
    for (leg, fx, score), e in zip(resolved, entries):
        e["match_score"] = score

    counts = {"BET": 0, "PASS": 0, "NO-DATA": 0, "KILL": 0}
    for e in entries:
        counts[e["verdict"]] += 1
    bets = sorted([e for e in entries if e["verdict"] == "BET"],
                  key=lambda e: -e["ev_pct"])
    total_stake = sum(e["stake_pct"] for e in bets)

    if args.json:
        print(json.dumps({
            "legs": entries,
            "summary": {"total": len(entries), **counts},
            "action": bets,
            "total_stake_pct": round(total_stake, 2),
        }, indent=2, ensure_ascii=False))
        return

    _print_slip_verdict(entries, bets, counts, title="SLIP VERDICT")

    if args.html:
        out_file = REPO / "predictions" / "verdict.html"
        out_file.write_text(render_slip_html(entries, bets, counts))
        print(f"\n[slip] wrote {out_file}")


def _print_slip_verdict(entries, bets, counts, title="SLIP VERDICT",
                        header_extra=None):
    total_stake = sum(e["stake_pct"] for e in bets)
    print("=" * 100)
    suffix = f"  |  {header_extra}" if header_extra else ""
    print(f"{title}  — {len(entries)} legs{suffix}")
    print(f"{counts['BET']} BET   {counts['PASS']} PASS   "
          f"{counts['NO-DATA']} NO-DATA   {counts['KILL']} KILL")
    print("=" * 100)
    print(f"{'#':<3}{'MATCH':<32}{'PICK':<17}{'ODDS':>6}{'MODEL':>7}{'FAIR':>7}"
          f"{'EDGE%':>7}{'EV%':>7}{'CONF':<11}{'VERDICT':<9}WHY")
    print("-" * 100)
    for e in entries:
        disp_pick = f"{e['market']}:{e['pick']}" if e["market"] else "—"
        odds = f"{e['odds']:.2f}" if e["odds"] else " — "
        mp = f"{e['model_prob']*100:4.0f}%" if e["model_prob"] is not None else "  — "
        fp = f"{e['fair_prob']*100:4.0f}%" if e["fair_prob"] is not None else "  — "
        ep = f"{e['edge_pct']:+5.1f}" if e["edge_pct"] is not None else "  — "
        ev = f"{e['ev_pct']:+5.1f}" if e["ev_pct"] is not None else "  — "
        conf = (e["confidence"] or "—")[:10]
        why = e["why"].replace("\n", " ")[:44]
        print(f"{e['leg']:<3}{e['match'][:31]:<32}{disp_pick[:16]:<17}{odds:>6}"
              f"{mp:>7}{fp:>7}{ep:>7}{ev:>7}{conf:<11}{e['verdict']:<9}{why}")
    print("-" * 100)

    if not bets:
        print("\n  No leg clears +EV — the sharp move is to BET NOTHING.")
    else:
        print("\nACTION TICKET (singles, Quarter-Kelly cap 5%, league-scaled):")
        for e in bets:
            print(f"  {e['match'][:30]:<30} {e['market']}:{e['pick']:<14} "
                  f"@{e['odds']:<7} EV {e['ev_pct']:+5.1f}%  "
                  f"stake {e['stake_pct']:.1f}%  [{e['confidence']}]  {e['why'][:38]}")
        print(f"\n  → TOTAL STAKE {total_stake:.1f}% of bankroll across "
              f"{len(bets)} single(s).")

    needs = [e for e in entries
             if e["verdict"] == "NO-DATA"
             or any(f in RED_FLAGS or f.startswith("pipeline red flag:") for f in e["flags"])]
    if needs:
        print("\nNEEDS AGENT WEB CHECK (venue / injuries / phantom fixtures):")
        for e in needs:
            reds = [f for f in e["flags"] if f in RED_FLAGS or f.startswith("pipeline red flag:")]
            why = e["why"] if not reds else ", ".join(reds)
            print(f"  [{e['leg']}] {e['match'][:36]:<36} {why[:60]}")


def render_slip_html(entries, bets, counts, title="Slip Verdict"):
    now = __import__("datetime").datetime.now().strftime("%d/%m/%Y %H:%M")
    conf_colors = {
        "Near Certain": "#22c55e", "High": "#3b82f6",
        "Medium-High": "#eab308", "Medium": "#f97316", "Low": "#ef4444",
    }

    def conf_badge(c):
        c = c or "?"
        color = conf_colors.get(c, "#64748b")
        return f'<span class="conf-badge" style="background:{color}">{escape(c)}</span>'

    rows = ""
    for e in entries:
        odds = f"{e['odds']:.2f}" if e["odds"] else "—"
        mp = f"{e['model_prob']*100:.0f}%" if e["model_prob"] is not None else "—"
        fp = f"{e['fair_prob']*100:.0f}%" if e["fair_prob"] is not None else "—"
        ep = f"{e['edge_pct']:+.1f}%" if e["edge_pct"] is not None else "—"
        ev = f"{e['ev_pct']:+.1f}%" if e["ev_pct"] is not None else "—"
        ev_style = 'color:#22c55e;font-weight:700;' if (e["ev_pct"] or 0) > 0 else 'color:#94a3b8;'
        badge = f'<span class="v-badge v-{e["verdict"]}">{e["verdict"]}</span>'
        pick = f"{escape(str(e['market']))} <strong>{escape(str(e['pick']))}</strong>" if e["market"] else "—"
        flags = " · ".join(escape(f) for f in e["flags"])
        why = escape(e["why"]) + (f' <span style="color:#d97706;">({flags})</span>' if flags else "")
        rows += (
            f'<tr><td style="color:#94a3b8;">{e["leg"]}</td>'
            f'<td><strong>{escape(e["match"])}</strong></td>'
            f'<td>{pick}</td><td>{odds}</td><td>{mp}</td>'
            f'<td style="color:#94a3b8;">{fp}</td><td style="color:#60a5fa;">{ep}</td>'
            f'<td style="{ev_style}">{ev}</td>'
            f'<td>{conf_badge(e["confidence"])}</td><td>{badge}</td>'
            f'<td style="color:#94a3b8;">{why}</td></tr>\n')

    if bets:
        tickets = "".join(
            f'<tr><td><strong>{escape(b["match"])}</strong></td>'
            f'<td>{escape(b["market"])} <strong>{escape(b["pick"])}</strong></td>'
            f'<td>{b["odds"]:.2f}</td><td>{b["stake_pct"]:.1f}%</td></tr>\n'
            for b in bets)
        total = sum(b["stake_pct"] for b in bets)
        ticket_html = (
            f'<div style="background:#052e16;border:1px solid #22c55e;border-radius:8px;'
            f'padding:12px 14px;margin:0 0 16px;">'
            f'<h2 style="margin:0 0 8px;">🎯 Action Ticket <span style="font-weight:400;'
            f'font-size:0.8rem;color:#94a3b8;">&middot; {len(bets)} single(s) &middot; '
            f'sum &asymp; {total:.1f}% of bankroll</span></h2>'
            f'<table style="width:100%;font-size:0.84rem;">'
            f'<tr><th>Match</th><th>Bet</th><th>Odds</th><th>Stake</th></tr>{tickets}</table>'
            f'</div>')
    else:
        ticket_html = (
            f'<div style="background:#1e293b;border:1px solid #ef4444;border-radius:8px;'
            f'padding:16px 20px;margin:0 0 16px;">'
            f'<h2 style="margin:0 0 4px;color:#f87171;">No leg clears +EV.</h2>'
            f'<p style="color:#94a3b8;font-size:0.88rem;margin:0;">'
            f'The sharp move is to <strong style="color:#fff;">bet nothing</strong> — '
            f'no leg cleared positive expected value after de-vigging, or every '
            f'+EV leg carries red flags.</p></div>')

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{escape(title)} — Multi-bet Verdict</title>
<style>
{_PAGE_STYLE}
</style>
</head>
<body>
<h1>🎯 {escape(title)}</h1>
<p class="sub">Leg-by-leg multi-bet verdict — model_prob vs de-vigged fair odds &middot; edge &middot; EV &middot; Quarter-Kelly stakes &middot; generated {now}</p>
<div class="stats">
  <div class="stat">Legs <span>{len(entries)}</span></div>
  <div class="stat">BET <span style="color:#22c55e;">{counts["BET"]}</span></div>
  <div class="stat">PASS <span style="color:#94a3b8;">{counts["PASS"]}</span></div>
  <div class="stat">NO-DATA <span style="color:#d97706;">{counts["NO-DATA"]}</span></div>
  <div class="stat">KILL <span style="color:#f87171;">{counts["KILL"]}</span></div>
</div>
{ticket_html}
<table>
  <tr><th>#</th><th>Match</th><th>Bet</th><th>Odds</th><th>Model</th><th>De-vig</th><th>Edge</th><th>EV</th><th>Conf</th><th>Verdict</th><th>Why</th></tr>
{rows}
</table>
<p class="sub" style="margin-top:16px;">Method: legs resolved against the pipeline (pending_results.json / history.db / pr dump); fair probs by proportional de-vig when the full line is known, else 1/odds; EV = model_prob × odds − 1; stake = Quarter-Kelly capped at 5%, scaled by league reliability in that market. KILL = model conflict, EV ≤ −5%, or already finished. NO-DATA = not verifiable by the pipeline — needs the agent's web layer (venue, injuries, phantom fixtures).</p>
</body>
</html>
"""


def render_card_html(card, title_suffix="", title="Bettor's Card"):
    """Render the bettor's card as a standalone dark-theme HTML page.

    Mirrors the predictions/*.html report style (background #0f172a, #1e293b
    cards, green accent #22c55e) so the agent's staking card is one more
    deliverable at the end of a prediction run.  The +EV table is the card
    itself; a <details> block lists every candidate (incl. -EV and no-odds)
    so the reasoning is auditable."""
    value = [p for p in (card.get("value_picks") or [])]
    candidates = card.get("all_candidates") or []
    skipped = card.get("skipped_finished") or 0
    now = __import__("datetime").datetime.now().strftime("%d/%m/%Y %H:%M")

    conf_colors = {
        "Near Certain": "#22c55e", "High": "#3b82f6",
        "Medium-High": "#eab308", "Medium": "#f97316", "Low": "#ef4444",
    }
    rel_badges = {
        "STUD": ('<span style="color:#22c55e;font-weight:700;">STUD</span>', "1.0"),
        "ok": ('<span style="color:#94a3b8;">ok</span>', "1.0"),
        "WEAK": ('<span style="color:#facc15;font-weight:700;">WEAK</span>', "0.5"),
        "COLD": ('<span style="color:#ef4444;font-weight:700;">COLD</span>', "0.0"),
    }

    def conf_badge(c):
        c = c or "?"
        color = conf_colors.get(c, "#64748b")
        return f'<span class="conf-badge" style="background:{color}">{escape(c)}</span>'

    # Exact total — the ticket table prints stakes at 2dp so the caption and
    # the printed stakes sum to the same number (1dp display rounding drifts
    # the implied total by several % on a big slate).
    total_stake = sum(p.get("stake_pct", 0.0) for p in value)
    rows = ""
    for p in value:
        rel, _ = rel_badges.get(p.get("reliability", ""), ("", ""))
        notes = " ".join(x for x in (p.get("note"), rel) if x)
        mkt = p.get("market", "?")
        pick = p.get("pick", "?")
        rows += (
            f'<tr>'
            f'<td style="color:#94a3b8;">{escape(str(p.get("league", ""))[:30])}</td>'
            f'<td><strong>{escape(p.get("match", ""))}</strong></td>'
            f'<td>{escape(mkt)} <strong>{escape(pick)}</strong></td>'
            f'<td>{conf_badge(p.get("confidence"))}</td>'
            f'<td>{p.get("model_prob", 0) * 100:.0f}%</td>'
            f'<td style="color:#94a3b8;">{p.get("fair_prob", 0) * 100:.0f}%</td>'
            f'<td style="color:#60a5fa;">{p.get("edge_pct", 0):+.1f}%</td>'
            f'<td style="color:#22c55e;font-weight:700;">{p.get("ev_pct", 0):+.1f}%</td>'
            f'<td>{p.get("odds", 0):.2f}</td>'
            f'<td style="color:#facc15;font-weight:700;">{p.get("stake_pct", 0):.1f}%</td>'
            f'<td style="color:#94a3b8;">{escape(notes)}</td>'
            f'</tr>\n'
        )

    if value:
        tickets = "".join(
            f'<tr><td><strong>{escape(p.get("match", ""))}</strong></td>'
            f'<td>{escape(p.get("market", ""))} <strong>{escape(p.get("pick", ""))}</strong></td>'
            f'<td>{p.get("odds", 0):.2f}</td>'
            f'<td>{p.get("stake_pct", 0):.2f}%</td></tr>\n'
            for p in value)
        scaled_note = ""
        if card.get("scaled_down"):
            scaled_note = ('<div style="margin-top:8px;color:#eab308;font-size:0.8rem;">'
                           'Sum pushed past 100% of bankroll — stakes scaled down '
                           'proportionally to the cap (sizing ratios kept).</div>')
        dust_note = ""
        if card.get("dropped_dust"):
            dust_note = (f" &middot; {card['dropped_dust']} dust leg(s) "
                         f"(&lt;{DUST_STAKE_PCT:.0f}%) left unbet")
        ticket_html = (
            f'<div style="background:#052e16;border:1px solid #22c55e;border-radius:8px;'
            f'padding:12px 14px;margin:0 0 16px;">'
            f'<h2 style="margin:0 0 8px;">🎯 Action Ticket <span style="font-weight:400;'
            f'font-size:0.8rem;color:#94a3b8;">&middot; one/best +EV single per match '
            f'(0%-stake legs dropped){dust_note} &middot; total &asymp; {total_stake:.1f}% '
            f'of bankroll'
            f'{", capped &le;100%" if card.get("scaled_down") else ""}</span></h2>'
            f'<table style="width:100%;font-size:0.84rem;">'
            f'<tr><th>Match</th><th>Bet</th><th>Odds</th><th>Stake</th></tr>{tickets}</table>'
            f'{scaled_note}</div>')
    else:
        ticket_html = (
            f'<div style="background:#1e293b;border:1px solid #ef4444;border-radius:8px;'
            f'padding:16px 20px;margin:0 0 16px;">'
            f'<h2 style="margin:0 0 4px;color:#f87171;">No +EV picks in this slate.</h2>'
            f'<p style="color:#94a3b8;font-size:0.88rem;margin:0;">'
            f'The sharp move is to <strong style="color:#fff;">bet nothing</strong> — '
            f'no candidate cleared positive expected value after de-vigging bookmaker odds.</p>'
            f'</div>')

    # ── reference block: every candidate (incl. -EV / no-odds) ──
    ref_rows = ""
    for p in sorted(candidates, key=lambda x: -x.get("ev_pct", -999)):
        if "ev_pct" in p:
            ev = f'{p["ev_pct"]:+.1f}%'
            ev_style = 'color:#22c55e;' if p["ev_pct"] > 0 else 'color:#ef4444;'
            ref_rows += (
                f'<tr><td style="color:#94a3b8;">{escape(str(p.get("league", ""))[:28])}</td>'
                f'<td>{escape(p.get("match", ""))}</td>'
                f'<td>{escape(p.get("market", ""))} <strong>{escape(p.get("pick", ""))}</strong></td>'
                f'<td>{conf_badge(p.get("confidence"))}</td>'
                f'<td style="{ev_style}">{ev}</td>'
                f'<td>{p.get("odds", "--") if "odds" in p else "--"}</td>'
                f'<td style="color:#94a3b8;">{escape(p.get("note", ""))}</td></tr>\n')
        else:
            ref_rows += (
                f'<tr><td style="color:#94a3b8;">{escape(str(p.get("league", ""))[:28])}</td>'
                f'<td>{escape(p.get("match", ""))}</td>'
                f'<td>{escape(p.get("market", ""))} <strong>{escape(p.get("pick", ""))}</strong></td>'
                f'<td>{conf_badge(p.get("confidence"))}</td>'
                f'<td style="color:#64748b;">—</td>'
                f'<td>—</td>'
                f'<td style="color:#94a3b8;">{escape(p.get("note", ""))}</td></tr>\n')
    skipped_note = ""
    if skipped:
        skipped_note = (f'<p class="sub">{skipped} finished match(es) excluded — '
                        f'no betting hindsight.</p>')

    title_esc = escape(title)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title_esc} — Agent's Betting Card</title>
<style>
{_PAGE_STYLE}
</style>
</head>
<body>
<h1>🎯 {title_esc}{escape(title_suffix)}</h1>
<p class="sub">The agent's staking card — model_prob vs de-vigged fair odds &middot; edge &middot; EV &middot; Quarter-Kelly stakes (max 5%, scaled by league reliability) &middot; generated {now}</p>
{skipped_note}
<div class="stats">
  <div class="stat">+EV picks <span>{len(value)}</span></div>
  <div class="stat">Total stake <span>{total_stake:.1f}%</span></div>
  <div class="stat">Candidates <span>{len(candidates)}</span></div>
</div>
{ticket_html}
<table>
  <tr><th>League</th><th>Match</th><th>Bet</th><th>Conf</th><th>Model</th><th>De-vig</th><th>Edge</th><th>EV</th><th>Odds</th><th>Stake</th><th>Note</th></tr>
{rows}
</table>
<details>
<summary>All candidates ({len(candidates)}) — including -EV and no-odds, for reference</summary>
<table>
  <tr><th>League</th><th>Match</th><th>Bet</th><th>Conf</th><th>EV</th><th>Odds</th><th>Note</th></tr>
{ref_rows}
</table>
</details>
<p class="sub" style="margin-top:16px;">Method: fair probs by proportional de-vig of bookmaker odds; edge = model_prob − fair_prob; EV = model_prob × odds − 1; stake = Quarter-Kelly, capped at 5% of bankroll, multiplied by league reliability in that market (STUD 1.0 / ok 1.0 / WEAK 0.5 / COLD 0.0). No bet without stated edge. The ticket keeps the single best +EV leg per match; legs priced at 0.0% (COLD leagues / negligible edge) are excluded, and a slate sum past 100% of bankroll is scaled down to the cap so the card never over-exposes one bankroll.</p>
</body>
</html>
"""


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

    s = sub.add_parser("slip", help="leg-by-leg verdict + action ticket for a multi-bet slip file")
    s.add_argument("file")
    s.add_argument("--preds", help="use a pr --json dump instead of scraping (path to json)")
    s.add_argument("--no-model", action="store_true",
                   help="never shell out to pr — DB/pending data only")
    s.add_argument("--json", action="store_true")
    s.add_argument("--html", action="store_true",
                   help="also write predictions/verdict.html")
    s.set_defaults(fn=cmd_slip)

    r = sub.add_parser("research",
                       help="emit a web-research agenda for a slip, then --apply the saved evidence back into the verdict")
    r.add_argument("file")
    r.add_argument("--apply", action="store_true",
                   help="merge data/research/<slug>.json evidence back into the verdict")
    r.add_argument("--preds", help="use a pr --json dump instead of scraping (path to json)")
    r.add_argument("--no-model", action="store_true",
                   help="never shell out to pr — DB/pending data only")
    r.add_argument("--json", action="store_true")
    r.set_defaults(fn=cmd_research)

    l = sub.add_parser("learn", help="run local calibration learning (bias + retrain)")
    l.set_defaults(fn=cmd_learn)

    args = ap.parse_args()
    if args.cmd == "card" and not args.file:
        ap.error("card needs a predictions.json or a links file")
    if args.cmd == "research" and not args.file:
        ap.error("research needs a slip file")
    args.fn(args)


if __name__ == "__main__":
    main()