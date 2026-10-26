"""
pipeline.py — Step-by-step prediction pipeline (pr2)

Replaces the monolithic analyze_from_data() with 7 explicit steps, each
returning a typed dict that the next step consumes.  Every intermediate
result is logged so you can see exactly what changed and why.

Steps
-----
1. DATA AUDIT        — completeness score, flag thin/missing fields
2. LEAGUE CONTEXT    — historical accuracy, draw rate, home advantage
3. EXPECTED GOALS    — Dixon-Coles + form + injuries → locked (exp_h, exp_a)
4. BASE PROBABILITIES — 3 independent vectors: Poisson, ML, Forebet
5. PROBABILITY FUSION — blend vectors, apply bias corrections, league damping
6. PICK SELECTION    — rank all markets, apply red-flag + league-market gate
7. KELLY / STAKE     — de-vig edge, Quarter-Kelly capped at 5%

Entry point
-----------
    from pipeline import run_pipeline
    result = run_pipeline(data, use_ml=True, verbose=True)

result is drop-in compatible with the dict returned by analyze_from_data().
"""

from __future__ import annotations

import math
from typing import Any

# ---------------------------------------------------------------------------
# Helpers shared with predict.py — import them rather than duplicate
# ---------------------------------------------------------------------------
import predict as _p  # noqa: E402 – the main module


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Data Audit
# ──────────────────────────────────────────────────────────────────────────────

def step1_data_audit(data: dict) -> dict:
    """Assess data completeness and quality.

    Returns
    -------
    dict with keys:
        quality_score   float 0-1  (1.0 = all fields present)
        warnings        list[str]  human-readable issues
        thin_sample     bool       True when form < 3 games or H2H < 3
        gate            str        "ok" | "caution" | "skip"
        fixes           list[str]  stat corrections applied (sanitize_scraped_stats)
    """
    from red_flags import sanitize_scraped_stats

    fixes = sanitize_scraped_stats(data)

    warnings: list[str] = list(fixes)
    score = 1.0

    # Form length
    hf = data.get("home_form") or ""
    af = data.get("away_form") or ""
    hf_len = sum(1 for c in hf if c in "WDL")
    af_len = sum(1 for c in af if c in "WDL")
    if hf_len < 3:
        warnings.append(f"Home form thin: only {hf_len} games")
        score -= 0.10
    if af_len < 3:
        warnings.append(f"Away form thin: only {af_len} games")
        score -= 0.10

    # H2H
    hm = data.get("h2h_matches") or 0
    if hm < 3:
        warnings.append(f"H2H thin: only {hm} meetings")
        score -= 0.08

    # Attack / defence averages
    if not data.get("home_avg_goals_for"):
        warnings.append("Missing home avg goals for")
        score -= 0.08
    if not data.get("away_avg_goals_for"):
        warnings.append("Missing away avg goals for")
        score -= 0.08

    # League position
    if not data.get("home_pos") or not data.get("away_pos"):
        warnings.append("No league position data")
        score -= 0.05

    # Venue stats
    if not data.get("home_home_avg_goals_for"):
        warnings.append("No home-venue attack data")
        score -= 0.04
    if not data.get("away_away_avg_goals_for"):
        warnings.append("No away-venue attack data")
        score -= 0.04

    # Shots data
    if not data.get("home_shots_ontarget_pct"):
        warnings.append("No shots-on-target data")
        score -= 0.03

    # Forebet data
    if not data.get("forebet_home_pct") and not data.get("forebet_away_pct"):
        warnings.append("No Forebet probability data")
        score -= 0.06

    # Odds
    if not data.get("odds_home"):
        warnings.append("No bookmaker odds available")
        score -= 0.05

    score = max(0.0, min(1.0, round(score, 3)))
    thin_sample = hf_len < 3 or af_len < 3 or hm < 3

    if score < 0.35:
        gate = "skip"
    elif score < 0.60 or thin_sample:
        gate = "caution"
    else:
        gate = "ok"

    return {
        "quality_score": score,
        "warnings": warnings,
        "thin_sample": thin_sample,
        "gate": gate,
        "fixes": fixes,
        "hf_len": hf_len,
        "af_len": af_len,
        "h2h_matches": hm,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: League Context
# ──────────────────────────────────────────────────────────────────────────────

def step2_league_context(data: dict) -> dict:
    """Resolve league profile and historical accuracy.

    Returns
    -------
    dict with keys:
        league_key          str
        profile             dict   (avg_goals, home_adv, volatility, …)
        difficulty          dict   (level, accuracy, matches, reason)
        league_reliability  float  0.45 | 0.70 | 1.0
        draw_base_rate      float  league draw % from DB (or 0.28 fallback)
        market_accuracy     dict   {market: {total, correct, pct}} last 7 days
        is_hard             bool
        reasoning           list[str]
    """
    league = data.get("league", "")
    league_key = _p.detect_league(league)
    profile = _p.get_profile(league_key)

    difficulty = _p._get_league_difficulty(league)
    acc = difficulty.get("accuracy") or 0

    if acc >= 65:
        reliability = 1.0
    elif acc >= 45:
        reliability = 0.70
    elif acc > 0:
        reliability = 0.45
    else:
        reliability = 0.70  # unknown

    # Per-market accuracy (last 7 days)
    market_acc: dict[str, dict] = {}
    for mk in ("1X2", "O/U", "BTTS"):
        a = _p._get_league_market_accuracy(league, mk)
        if a.get("total", 0) >= 5:
            market_acc[mk] = a

    # Draw base rate from DB
    from database import get_base_rates, pick_base_rate
    base_rates = get_base_rates(league)
    draw_br = pick_base_rate(base_rates, "1X2", "Draw") if base_rates else 0.28

    reasoning: list[str] = []
    if difficulty.get("level") == "hard":
        reasoning.append(f"⚠ {difficulty.get('reason', 'Unreliable league')}")
    for mk, a in market_acc.items():
        if a.get("pct", 100) < 40:
            reasoning.append(
                f"⚠ {league_key} {mk} accuracy only {a['pct']:.0f}% "
                f"({a['total']} recent picks) — confidence capped"
            )

    derby = _p._detect_derby(data)
    if derby["is_derby"]:
        reasoning.append(derby["warning"])

    return {
        "league_key": league_key,
        "profile": profile,
        "difficulty": difficulty,
        "league_reliability": reliability,
        "draw_base_rate": draw_br,
        "market_accuracy": market_acc,
        "is_hard": difficulty.get("level") == "hard",
        "derby": derby,
        "reasoning": reasoning,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Expected Goals  (locked after this step — no further mutation)
# ──────────────────────────────────────────────────────────────────────────────

def step3_expected_goals(data: dict, ctx: dict) -> dict:
    """Compute locked (exp_h, exp_a) using Dixon-Coles + form + injuries.

    Runs every adjustment in a deterministic order so no downstream step
    can silently re-mutate the base expected goals.

    Returns
    -------
    dict with keys:
        exp_h, exp_a        float  final locked values
        exp_total           float
        method              list[str]  which sub-models fired
        adjustments         list[str]  human-readable shifts applied
    """
    from ml_model import poisson_predict

    profile = ctx["profile"]
    method: list[str] = []
    adjustments: list[str] = []

    # ── Base: Dixon-Coles enhanced Poisson ──
    try:
        enhanced = poisson_predict(data, profile, use_dixon_coles=True)
        exp_h = enhanced["exp_home_goals"]
        exp_a = enhanced["exp_away_goals"]
        method.append("dc-poisson")
    except Exception:
        exp_h, exp_a = _p.estimate_goals(data, profile)
        method.append("simple-poisson")

    base_exp_h, base_exp_a = exp_h, exp_a  # for logging shifts

    # ── Common-opponent strength ──
    _cos = _p._common_opponent_strength(data)
    if _cos.get("reason"):
        exp_h = max(0.1, exp_h * _cos["h_mult"])
        exp_a = max(0.1, exp_a * _cos["a_mult"])
        adjustments.append(f"Common-opponent: {_cos['reason']}")
        method.append("common-opp")

    # ── Form signal (additive shift on xG) ──
    form_analysis = _p.build_form_analysis(data)
    fsig = form_analysis.get("signal", 0.0)

    ml_model_tmp = None
    try:
        ml_model_tmp = _p._load_ml_model()
    except Exception:
        pass
    ml_signal_weights = _p._get_ml_signal_weights(ml_model_tmp) if ml_model_tmp else {}
    _form_weight = ml_signal_weights.get("form", 1.0)
    if ctx["derby"]["is_derby"]:
        _form_weight *= 0.6

    if abs(fsig) >= 0.05:
        shift = fsig * 0.08 * _form_weight
        exp_h_before = exp_h
        exp_a_before = exp_a
        exp_h = max(0.05, exp_h - shift)
        exp_a = max(0.05, exp_a + shift)
        adjustments.append(
            f"Form signal {fsig:+.2f}: "
            f"home {exp_h_before:.2f}→{exp_h:.2f}, "
            f"away {exp_a_before:.2f}→{exp_a:.2f}"
        )
        method.append("form")

    # ── Injury adjustment ──
    exp_h, exp_a, _inj_reason = _p._injury_goal_adjustment(data, exp_h, exp_a)
    if _inj_reason:
        adjustments.append(f"Injuries: {_inj_reason} → home={exp_h:.2f}, away={exp_a:.2f}")

    exp_h = max(0.1, exp_h)
    exp_a = max(0.1, exp_a)

    return {
        "exp_h": exp_h,
        "exp_a": exp_a,
        "exp_total": exp_h + exp_a,
        "method": method,
        "adjustments": adjustments,
        "form_analysis": form_analysis,
        "fsig": fsig,
        "ml_signal_weights": ml_signal_weights,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Base Probabilities  (3 independent vectors)
# ──────────────────────────────────────────────────────────────────────────────

def step4_base_probabilities(data: dict, ctx: dict, xg: dict, use_ml: bool = True) -> dict:
    """Compute three independent probability vectors.

    Vectors
    -------
    poisson_vec   pure Poisson from locked (exp_h, exp_a)
    ml_vec        ML ensemble (RF + XGB + LGB + GB)
    forebet_vec   scraped Forebet percentages (normalised)

    Returns
    -------
    dict with keys:
        poisson_vec, ml_vec, forebet_vec   each a dict of Home/Draw/Away/Over/Under
        agreement_score   float 0-1  (1 = all 3 point to same winner)
        ml_available      bool
        forebet_available bool
        ml_model          loaded model object or None
        ensemble_result   raw ensemble output dict (for O/U etc)
        reasoning         list[str]
    """
    exp_h = xg["exp_h"]
    exp_a = xg["exp_a"]
    profile = ctx["profile"]
    reasoning: list[str] = []

    # ── Vector 1: Poisson ──
    p_h = _p.prob_home_win(exp_h, exp_a)
    p_d = _p.prob_draw(exp_h, exp_a)
    p_a = _p.prob_away_win(exp_h, exp_a)
    p_ov, p_un = _p._ou_exp(exp_h, exp_a)
    p_btts = _p.prob_btts(exp_h, exp_a)

    poisson_vec = {
        "Home win": p_h, "Draw": p_d, "Away win": p_a,
        "Over 2.5": p_ov, "Under 2.5": p_un,
        "BTTS Yes": p_btts, "BTTS No": 1.0 - p_btts,
    }

    # ── Vector 2: ML ensemble ──
    ml_available = False
    ml_model = None
    ensemble_result: dict = {}

    if use_ml:
        try:
            ml_model = _p._load_ml_model()
        except Exception:
            ml_model = None

    if ml_model:
        try:
            from ml_model import ensemble_predict
            dyn_weights = _p._get_dynamic_weights(ctx["league_key"])
            ensemble_result = ensemble_predict(
                data, profile, ml_model,
                dynamic_weights=dyn_weights,
                league=ctx["league_key"],
                market="1X2",
            )
            ml_available = True
        except Exception as exc:
            reasoning.append(f"ML ensemble failed: {exc}")

    if ml_available:
        ml_vec = {
            "Home win": ensemble_result.get("prob_home", p_h),
            "Draw":     ensemble_result.get("prob_draw", p_d),
            "Away win": ensemble_result.get("prob_away", p_a),
            "Over 2.5": ensemble_result.get("prob_over", p_ov),
            "Under 2.5": ensemble_result.get("prob_under", p_un),
            "BTTS Yes": ensemble_result.get("prob_btts_yes", p_btts),
            "BTTS No":  1.0 - ensemble_result.get("prob_btts_yes", p_btts),
        }
    else:
        ml_vec = dict(poisson_vec)  # fall back to Poisson
        if use_ml:
            reasoning.append("ML unavailable — using Poisson as ML vector")

    # ── Vector 3: Forebet ──
    fb_h_raw = (data.get("forebet_home_pct") or 0) / 100.0
    fb_d_raw = (data.get("forebet_draw_pct") or 0) / 100.0
    fb_a_raw = (data.get("forebet_away_pct") or 0) / 100.0
    fb_ov_raw = (data.get("forebet_over25_pct") or 0) / 100.0
    fb_btts_raw = (data.get("forebet_btts_yes_pct") or 0) / 100.0
    fb_total = fb_h_raw + fb_d_raw + fb_a_raw
    forebet_available = fb_total > 0

    if forebet_available:
        fb_h_n = fb_h_raw / fb_total
        fb_d_n = fb_d_raw / fb_total
        fb_a_n = fb_a_raw / fb_total
        forebet_vec = {
            "Home win": fb_h_n, "Draw": fb_d_n, "Away win": fb_a_n,
            "Over 2.5": fb_ov_raw, "Under 2.5": 1.0 - fb_ov_raw,
            "BTTS Yes": fb_btts_raw, "BTTS No": 1.0 - fb_btts_raw,
        }
    else:
        forebet_vec = dict(poisson_vec)
        reasoning.append("No Forebet data — only Poisson + ML vectors available")

    # ── Agreement score ──
    _tops = []
    for vec, name in [(poisson_vec, "Poisson"), (ml_vec, "ML"), (forebet_vec, "Forebet")]:
        t = max(["Home win", "Draw", "Away win"], key=lambda k: vec.get(k, 0))
        _tops.append(t)

    if len(set(_tops)) == 1:
        agreement = 1.0
        reasoning.append(f"✓ All 3 models agree: {_tops[0]}")
    elif len(set(_tops)) == 2:
        agreement = 0.67
        majority = max(set(_tops), key=_tops.count)
        minority = [t for t in _tops if t != majority][0]
        reasoning.append(
            f"⚠ 2/3 models agree ({majority}); "
            f"{'Forebet' if forebet_vec.get(minority, 0) > poisson_vec.get(minority, 0) else 'ML'} "
            f"favours {minority}"
        )
    else:
        agreement = 0.0
        reasoning.append(
            f"⚠ 3-way split: Poisson={_tops[0]}, ML={_tops[1]}, Forebet={_tops[2]} "
            f"— confidence will be capped at Medium"
        )

    return {
        "poisson_vec": poisson_vec,
        "ml_vec": ml_vec,
        "forebet_vec": forebet_vec,
        "agreement_score": agreement,
        "ml_available": ml_available,
        "forebet_available": forebet_available,
        "ml_model": ml_model,
        "ensemble_result": ensemble_result,
        "reasoning": reasoning,
        "tops": _tops,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Step 5: Probability Fusion
# ──────────────────────────────────────────────────────────────────────────────

def step5_fusion(data: dict, ctx: dict, xg: dict, vecs: dict) -> dict:
    """Fuse the three vectors into a single calibrated probability set.

    Weighting
    ---------
    ML and Forebet get higher weight when both are available.
    Agreement score amplifies the top-pick probability.
    Bias corrections from calibration_params.json are applied.
    League reliability dampens O/U / BTTS when the league is unreliable.

    Returns
    -------
    dict with keys:
        p_home, p_draw, p_away, p_over25, p_under25, p_btts_yes, p_btts_no
        p_1x, p_x2, p_12     (double-chance)
        method_parts          list[str]
        reasoning             list[str]
    """
    pv = vecs["poisson_vec"]
    mv = vecs["ml_vec"]
    fv = vecs["forebet_vec"]
    agreement = vecs["agreement_score"]
    reliability = ctx["league_reliability"]
    league_key = ctx["league_key"]
    reasoning: list[str] = []
    method_parts: list[str] = []

    # ── Determine blend weights ──
    if vecs["ml_available"] and vecs["forebet_available"]:
        w_p, w_m, w_f = 0.30, 0.45, 0.25
        method_parts.append("ml+fb+poisson")
    elif vecs["ml_available"]:
        w_p, w_m, w_f = 0.40, 0.60, 0.0
        method_parts.append("ml+poisson")
    elif vecs["forebet_available"]:
        w_p, w_m, w_f = 0.55, 0.0, 0.45
        method_parts.append("fb+poisson")
    else:
        w_p, w_m, w_f = 1.0, 0.0, 0.0
        method_parts.append("poisson-only")
        reasoning.append("⚠ Poisson-only — ML and Forebet unavailable")

    def _blend(key: str) -> float:
        return w_p * pv.get(key, 0) + w_m * mv.get(key, 0) + w_f * fv.get(key, 0)

    p_home = _blend("Home win")
    p_draw = _blend("Draw")
    p_away = _blend("Away win")
    p_over = _blend("Over 2.5")
    p_under = _blend("Under 2.5")
    p_btts_y = _blend("BTTS Yes")
    p_btts_n = _blend("BTTS No")

    # Normalise 1X2
    total = p_home + p_draw + p_away
    if total > 0:
        p_home /= total
        p_draw /= total
        p_away /= total

    # ── Apply bias corrections ──
    try:
        p_home, p_draw, p_away, p_over, p_under = _p._apply_bias_corrections(
            league_key, p_home, p_draw, p_away, p_over, p_under
        )
        method_parts.append("bias-corrected")
    except Exception:
        pass

    # ── Calibrate via per-outcome isotonic maps ──
    # 1X2 maps are fit on every match (unbiased) so they apply to the blended
    # probabilities directly; BTTS is conservative (reduces measured BTTS-No
    # overconfidence). O/U per-line maps are pick-conditioned and are applied
    # at emission in predict.py, not to the internal signal.
    try:
        p_home, p_draw, p_away = _p._outcome_calibrate_1x2(p_home, p_draw, p_away)
        _p_y_cal = _p._outcome_calibrate_btts(p_btts_y)
        if _p_y_cal is not None:
            p_btts_y, p_btts_n = _p_y_cal, 1.0 - _p_y_cal
        method_parts.append("outcome-isotonic")
    except Exception:
        pass

    # ── Agreement concordance boost ──
    if agreement == 1.0:
        top = max(["Home win", "Draw", "Away win"],
                  key=lambda k: {"Home win": p_home, "Draw": p_draw, "Away win": p_away}[k])
        boost = 0.04
        if top == "Home win":
            p_home = min(p_home + boost, 0.92)
        elif top == "Draw":
            p_draw = min(p_draw + boost, 0.92)
        else:
            p_away = min(p_away + boost, 0.92)
        total = p_home + p_draw + p_away
        p_home /= total; p_draw /= total; p_away /= total
        reasoning.append(f"+{boost:.0%} concordance boost → {top}")
        method_parts.append("concordance")

    # ── Away form boost (prevents systematic under-prediction of away wins) ──
    fsig = xg["fsig"]
    if fsig > 0.20:
        away_boost = min(fsig * 0.10, 0.08)
        p_away = min(p_away + away_boost, 0.65)
        total = p_home + p_draw + p_away
        p_home /= total; p_draw /= total; p_away /= total
        reasoning.append(f"Away form boost +{away_boost:.2f} (fsig={fsig:+.2f})")

    # ── League reliability damping on O/U + BTTS ──
    if reliability < 1.0:
        neutral_ou = 0.50
        p_over = p_over * reliability + neutral_ou * (1 - reliability)
        p_under = p_under * reliability + neutral_ou * (1 - reliability)
        neutral_btts = 0.50
        p_btts_y = p_btts_y * reliability + neutral_btts * (1 - reliability)
        p_btts_n = p_btts_n * reliability + neutral_btts * (1 - reliability)
        reasoning.append(
            f"League reliability {reliability:.0%}: O/U + BTTS probabilities dampened"
        )

    # ── Double-chance ──
    p_1x = p_home + p_draw
    p_x2 = p_away + p_draw
    p_12 = p_home + p_away

    return {
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "p_over25": p_over,
        "p_under25": p_under,
        "p_btts_yes": p_btts_y,
        "p_btts_no": p_btts_n,
        "p_1x": p_1x,
        "p_x2": p_x2,
        "p_12": p_12,
        "method_parts": method_parts,
        "reasoning": reasoning,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Step 6: Pick Selection + Gating
# ──────────────────────────────────────────────────────────────────────────────

def step6_pick_selection(data: dict, ctx: dict, audit: dict,
                         xg: dict, vecs: dict, fused: dict) -> dict:
    """Build all market candidates, apply gates, and select the top pick.

    Gates (in order)
    ----------------
    1. Red-flag critical → confidence cap
    2. 3-way model disagreement (vecs.agreement_score == 0) → cap at Medium
    3. League-market historical accuracy < 40% → cap at Medium-Low, reduce Kelly
    4. Data quality gate == "skip" → output Low confidence

    Returns
    -------
    dict with keys:
        primary     dict  (market, pick, confidence, model_prob, implied_prob, …)
        candidates  list[dict]
        reasoning   list[str]
        warnings    list[str]
        gates_fired list[str]
    """
    from red_flags import detect_red_flags

    exp_h = xg["exp_h"]
    exp_a = xg["exp_a"]
    p_home = fused["p_home"]
    p_draw = fused["p_draw"]
    p_away = fused["p_away"]
    p_over = fused["p_over25"]
    p_under = fused["p_under25"]
    p_btts_y = fused["p_btts_yes"]
    p_btts_n = fused["p_btts_no"]
    p_1x = fused["p_1x"]
    p_x2 = fused["p_x2"]
    p_12 = fused["p_12"]
    profile = ctx["profile"]

    reasoning: list[str] = list(ctx["reasoning"])
    warnings: list[str] = list(audit["warnings"])
    gates_fired: list[str] = []

    # ── Red flags ──
    _top_1x2_pick = max(
        [("Home win", p_home), ("Draw", p_draw), ("Away win", p_away)],
        key=lambda x: x[1]
    )[0]
    _top_1x2_prob = max(p_home, p_draw, p_away)
    _top_odds = data.get("odds_home") if _top_1x2_pick == "Home win" else (
        data.get("odds_draw") if _top_1x2_pick == "Draw" else data.get("odds_away")
    )
    rf_all = detect_red_flags(
        data,
        market="1X2",
        pick=_top_1x2_pick,
        model_prob=_top_1x2_prob,
        pick_odds=_top_odds,
    )
    crit_flags = [f for f in rf_all if f.get("severity") == "critical"]
    high_flags = [f for f in rf_all if f.get("severity") == "high"]
    for f in crit_flags + high_flags:
        warnings.append(("🚩 " if f["severity"] == "critical" else "⚠ ") + f["msg"])

    # ── Build candidates (same logic as analyze_from_data) ──
    def _odds(market: str, pick: str) -> float | None:
        """Return bookmaker odds for a market/pick from the scraped data."""
        odds_h   = data.get("odds_home")
        odds_d   = data.get("odds_draw")
        odds_a   = data.get("odds_away")
        odds_o25 = data.get("odds_over25")
        odds_u25 = data.get("odds_under25")
        odds_btts_y = data.get("odds_btts_yes")
        odds_btts_n = data.get("odds_btts_no")
        if market == "1X2":
            return {"Home win": odds_h, "Draw": odds_d, "Away win": odds_a}.get(pick)
        if market == "DNB":
            return {"Home": odds_h, "Away": odds_a}.get(pick)
        if market == "DC":
            if pick == "1X" and odds_h and odds_d:
                return 1.0 / (1.0/odds_h + 1.0/odds_d)
            if pick == "X2" and odds_a and odds_d:
                return 1.0 / (1.0/odds_a + 1.0/odds_d)
            if pick == "12" and odds_h and odds_a:
                return 1.0 / (1.0/odds_h + 1.0/odds_a)
        if market == "O/U":
            return {"Over 2.5": odds_o25, "Under 2.5": odds_u25}.get(pick)
        if market == "BTTS":
            return {"Yes": odds_btts_y, "No": odds_btts_n}.get(pick)
        return None

    def _implied(odds_val):
        if odds_val and odds_val > 1:
            return 1.0 / odds_val
        return None

    def _edge(prob, imp):
        if prob and imp:
            return prob - imp
        return None

    raw_candidates = [
        {"market": "1X2",  "pick": "Home win",   "prob": p_home},
        {"market": "1X2",  "pick": "Draw",        "prob": p_draw},
        {"market": "1X2",  "pick": "Away win",    "prob": p_away},
        {"market": "O/U",  "pick": "Over 2.5",    "prob": p_over},
        {"market": "O/U",  "pick": "Under 2.5",   "prob": p_under},
        {"market": "BTTS", "pick": "Yes",          "prob": p_btts_y},
        {"market": "BTTS", "pick": "No",           "prob": p_btts_n},
        {"market": "DC",   "pick": "1X",           "prob": p_1x},
        {"market": "DC",   "pick": "X2",           "prob": p_x2},
        {"market": "DC",   "pick": "12",           "prob": p_12},
    ]

    candidates = []
    for c in raw_candidates:
        odds_val = _odds(c["market"], c["pick"])
        imp = _implied(odds_val)
        edge = _edge(c["prob"], imp)
        conf = _p._value_to_conf(
            edge if edge else 0,
            odds_val if odds_val else 2.0,
        )
        candidates.append({
            "market": c["market"],
            "pick": c["pick"],
            "model_prob": c["prob"],
            "implied_prob": imp,
            "odds": odds_val,
            "edge": edge,
            "confidence": conf,
        })

    # ── Holistic synthesis re-ranking ──
    _synth_rationale = ""
    try:
        from synthesis import MatchContext as SynthCtx, synthesize, build_synthesis_rationale, context_from_pred

        _fsig = xg.get("fsig", 0.0)
        _form_clearly = abs(_fsig) >= 0.20
        _top_pick_1x2 = max(
            [("Home win", p_home), ("Draw", p_draw), ("Away win", p_away)],
            key=lambda x: x[1]
        )[0]
        _vol = ctx["profile"].get("volatility", 0.1)

        _synth_ctx = context_from_pred(
            pred={},
            data=data,
            vol=_vol,
            form_signal=_fsig,
            trans_signal=0.0,
            draw_tendency=(p_draw >= p_home and p_draw >= p_away),
            draw_factors=0,
            top_pick=_top_pick_1x2,
            margin=max(p_home, p_draw, p_away) - sorted([p_home, p_draw, p_away])[-2],
            league_reliability=ctx["league_reliability"],
            ml_dir=None,
            draw_bias_suppressed=_form_clearly,
            away_win_boosted=(_fsig > 0.20),
        )
        _synth_ctx.p_home = p_home
        _synth_ctx.p_draw = p_draw
        _synth_ctx.p_away = p_away
        _synth_ctx.exp_h = xg["exp_h"]
        _synth_ctx.exp_a = xg["exp_a"]
        _synth_ctx.p_over25 = p_over
        _synth_ctx.p_under25 = p_under
        _synth_ctx.p_btts_yes = p_btts_y
        _synth_ctx.p_btts_no = p_btts_n
        _synth_ctx.p_1x = p_1x
        _synth_ctx.p_x2 = p_x2
        _synth_ctx.p_12 = p_12

        candidates = synthesize(_synth_ctx, candidates)
        _synth_rationale = build_synthesis_rationale(_synth_ctx, candidates, candidates[0])

        # Boost/penalise candidates by league-market accuracy
        mkt_acc_all = ctx["market_accuracy"]
        for c in candidates:
            _acc = mkt_acc_all.get(c.get("market", ""), {})
            if _acc.get("total", 0) >= 5 and _acc.get("pct", 0) > 0:
                dv = c.get("decision_value", 0)
                if _acc["pct"] >= 70:
                    c["decision_value"] = dv + 0.05
                elif _acc["pct"] >= 60:
                    c["decision_value"] = dv + 0.03
                elif _acc["pct"] < 45:
                    c["decision_value"] = dv - 0.03
        candidates.sort(key=lambda c: c.get("decision_value", 0), reverse=True)

        if _synth_rationale:
            reasoning.append("── Synthesis ──")
            reasoning.append(_synth_rationale)
    except Exception:
        # Synthesis is optional — don't break on failure
        pass

    # Sort by model_prob descending if synthesis didn't re-rank
    candidates.sort(key=lambda c: c.get("decision_value", c["model_prob"]), reverse=True)
    primary = candidates[0]

    # ── GATE 1: Data quality too low ──
    if audit["gate"] == "skip":
        primary = dict(primary)
        primary["confidence"] = "Low"
        gates_fired.append("DATA_QUALITY_SKIP")
        reasoning.append("⚠ Gate 1 fired: data quality too low → confidence capped at Low")

    # ── GATE 2: 3-way model disagreement ──
    _CONF_LADDER = ["Near Certain", "High", "Medium-High", "Medium", "Medium-Low", "Low"]
    if vecs["agreement_score"] == 0.0:
        _cur = primary.get("confidence", "Medium")
        _rank = _CONF_LADDER.index(_cur) if _cur in _CONF_LADDER else 3
        _new = _CONF_LADDER[min(_rank + 1, len(_CONF_LADDER) - 1)]
        primary = dict(primary)
        primary["confidence"] = _new
        gates_fired.append("MODEL_DISAGREEMENT")
        reasoning.append(
            f"⚠ Gate 2 fired: 3-way model split → confidence capped "
            f"{_cur} → {_new}"
        )

    # ── GATE 3: League-market historical accuracy < 40% ──
    mkt = primary.get("market", "1X2")
    mkt_acc = ctx["market_accuracy"].get(mkt, {})
    if mkt_acc.get("total", 0) >= 5 and mkt_acc.get("pct", 100) < 40:
        _cur = primary.get("confidence", "Medium")
        _rank = _CONF_LADDER.index(_cur) if _cur in _CONF_LADDER else 3
        _new = _CONF_LADDER[min(_rank + 2, len(_CONF_LADDER) - 1)]
        primary = dict(primary)
        primary["confidence"] = _new
        gates_fired.append(f"LEAGUE_MARKET_ACCURACY_{mkt}_{mkt_acc['pct']:.0f}pct")
        reasoning.append(
            f"⚠ Gate 3 fired: {ctx['league_key']} {mkt} only "
            f"{mkt_acc['pct']:.0f}% accurate ({mkt_acc['total']} picks) "
            f"→ confidence {_cur} → {_new}"
        )

    # ── GATE 4: Red-flag critical ──
    if crit_flags:
        _cur = primary.get("confidence", "Medium")
        _rank = _CONF_LADDER.index(_cur) if _cur in _CONF_LADDER else 3
        _new = _CONF_LADDER[min(_rank + 1, len(_CONF_LADDER) - 1)]
        primary = dict(primary)
        primary["confidence"] = _new
        gates_fired.append("RED_FLAG_CRITICAL")
        reasoning.append(
            f"⚠ Gate 4 fired: critical red flag → confidence "
            f"{_cur} → {_new}"
        )

    # Summary of gates
    if gates_fired:
        reasoning.append(f"Gates fired: {', '.join(gates_fired)}")

    # Correct-score lean
    cs_h = max(0, min(int(round(xg["exp_h"])), 5))
    cs_a = max(0, min(int(round(xg["exp_a"])), 5))

    return {
        "primary": primary,
        "candidates": candidates,
        "reasoning": reasoning,
        "warnings": warnings,
        "gates_fired": gates_fired,
        "crit_flags": crit_flags,
        "high_flags": high_flags,
        "correct_score": f"{cs_h}-{cs_a}",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Step 7: Kelly / Stake
# ──────────────────────────────────────────────────────────────────────────────

def step7_kelly(ctx: dict, pick6: dict) -> dict:
    """Compute Quarter-Kelly stake, applying league-accuracy and red-flag haircuts.

    Returns
    -------
    dict with keys:
        kelly_stake     float  0-0.05
        edge            float  model_prob - implied_prob
        interpretation  str    human-readable label
        reasoning       list[str]
    """
    primary = pick6["primary"]
    model_prob = primary.get("model_prob")
    implied_prob = primary.get("implied_prob")
    odds_val = primary.get("odds")
    reasoning: list[str] = []

    kelly_stake = 0.0
    edge = 0.0

    if model_prob and implied_prob and implied_prob > 0 and odds_val and odds_val > 1.0:
        edge = model_prob - implied_prob
        if edge > 0:
            kelly_fraction = edge / (odds_val - 1)
            kelly_stake = min(kelly_fraction * 0.25, 0.05)  # Quarter-Kelly, 5% cap

    # ── League-market accuracy haircut ──
    mkt = primary.get("market", "1X2")
    mkt_acc = ctx["market_accuracy"].get(mkt, {})
    if mkt_acc.get("total", 0) >= 5 and mkt_acc.get("pct", 0) > 0:
        acc_factor = mkt_acc["pct"] / 100.0
        old_stake = kelly_stake
        kelly_stake = round(kelly_stake * acc_factor, 4)
        reasoning.append(
            f"League-market accuracy haircut {acc_factor:.0%} "
            f"({mkt} {mkt_acc['pct']:.0f}%): "
            f"{old_stake:.1%} → {kelly_stake:.1%}"
        )

    # ── Red-flag haircuts ──
    if pick6["high_flags"]:
        kelly_stake = round(kelly_stake * 0.75, 4)
        reasoning.append(f"High red-flag haircut 25%: stake → {kelly_stake:.1%}")
    if pick6["crit_flags"]:
        kelly_stake = round(kelly_stake * 0.50, 4)
        reasoning.append(f"Critical red-flag haircut 50%: stake → {kelly_stake:.1%}")

    kelly_stake = round(kelly_stake, 4)
    interp = _p._kelly_interpretation(kelly_stake * 100)

    return {
        "kelly_stake": kelly_stake,
        "edge": edge,
        "interpretation": interp,
        "reasoning": reasoning,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(data: dict, use_ml: bool = True, verbose: bool = True) -> dict:
    """Run all 7 steps and return a result dict compatible with analyze_from_data().

    If verbose=True, each step prints a brief summary line prefixed with the
    step number so you can watch the chain execute in real time.
    """

    def _log(msg: str):
        if verbose:
            print(f"    {msg}", flush=True)

    home = data.get("home_team", "?")
    away = data.get("away_team", "?")

    _log(f"▶ Step 1/7  Data audit  ({home} vs {away})")
    audit = step1_data_audit(data)
    _log(f"    quality={audit['quality_score']:.2f}  gate={audit['gate']}"
         + (f"  warnings: {'; '.join(audit['warnings'][:3])}" if audit["warnings"] else ""))

    _log("▶ Step 2/7  League context")
    ctx = step2_league_context(data)
    _log(f"    league={ctx['league_key']}  reliability={ctx['league_reliability']:.0%}"
         + (f"  ⚠ {ctx['difficulty']['reason']}" if ctx["is_hard"] else ""))
    if ctx["market_accuracy"]:
        for mk, a in ctx["market_accuracy"].items():
            flag = " ⚠ LOW" if a.get("pct", 100) < 40 else ""
            _log(f"    {mk} accuracy (7d): {a['pct']:.0f}% ({a['total']} picks){flag}")

    _log("▶ Step 3/7  Expected goals")
    xg = step3_expected_goals(data, ctx)
    _log(f"    exp_h={xg['exp_h']:.2f}  exp_a={xg['exp_a']:.2f}  total={xg['exp_total']:.2f}")
    for adj in xg["adjustments"]:
        _log(f"    ↳ {adj}")

    _log("▶ Step 4/7  Base probabilities  (3 independent vectors)")
    vecs = step4_base_probabilities(data, ctx, xg, use_ml=use_ml)
    pv = vecs["poisson_vec"]
    mv = vecs["ml_vec"]
    fv = vecs["forebet_vec"]
    _log(f"    Poisson:  H={pv['Home win']:.1%}  D={pv['Draw']:.1%}  A={pv['Away win']:.1%}"
         f"  O2.5={pv['Over 2.5']:.1%}")
    _log(f"    ML:       H={mv['Home win']:.1%}  D={mv['Draw']:.1%}  A={mv['Away win']:.1%}"
         f"  O2.5={mv['Over 2.5']:.1%}")
    _log(f"    Forebet:  H={fv['Home win']:.1%}  D={fv['Draw']:.1%}  A={fv['Away win']:.1%}"
         f"  O2.5={fv['Over 2.5']:.1%}")
    _log(f"    Agreement: {vecs['agreement_score']:.0%}  — {', '.join(vecs['reasoning'][:1])}")

    _log("▶ Step 5/7  Probability fusion")
    fused = step5_fusion(data, ctx, xg, vecs)
    _log(f"    Fused:  H={fused['p_home']:.1%}  D={fused['p_draw']:.1%}  "
         f"A={fused['p_away']:.1%}  O2.5={fused['p_over25']:.1%}")
    for r in fused["reasoning"]:
        _log(f"    ↳ {r}")

    _log("▶ Step 6/7  Pick selection + gating")
    pick6 = step6_pick_selection(data, ctx, audit, xg, vecs, fused)
    p = pick6["primary"]
    _log(f"    Pick: {p['market']} {p['pick']}  conf={p['confidence']}"
         f"  prob={p.get('model_prob', 0):.1%}  edge={p.get('edge') or 0:+.1%}")
    if pick6["gates_fired"]:
        _log(f"    ⚠ Gates fired: {', '.join(pick6['gates_fired'])}")

    _log("▶ Step 7/7  Kelly / stake")
    kelly = step7_kelly(ctx, pick6)
    _log(f"    Kelly={kelly['kelly_stake']:.1%}  edge={kelly['edge']:+.1%}  "
         f"({kelly['interpretation']})")

    # ──────────────────────────────────────────────────────────────────────
    # Assemble full reasoning for display
    # ──────────────────────────────────────────────────────────────────────
    reasoning_lines: list[str] = []

    # Step 1
    if audit["warnings"]:
        reasoning_lines.append("── Data Quality ──")
        reasoning_lines.extend(audit["warnings"][:6])

    # Step 2
    if ctx["reasoning"]:
        reasoning_lines.append("── League Context ──")
        reasoning_lines.extend(ctx["reasoning"])

    # Step 3
    if xg["adjustments"]:
        reasoning_lines.append("── Expected Goals ──")
        reasoning_lines.append(
            f"exp_h={xg['exp_h']:.2f}  exp_a={xg['exp_a']:.2f}  "
            f"total={xg['exp_total']:.2f}"
        )
        reasoning_lines.extend(xg["adjustments"])

    # Step 4 — vector comparison
    reasoning_lines.append("── Model Vectors ──")
    reasoning_lines.append(
        f"Poisson H={pv['Home win']:.1%} D={pv['Draw']:.1%} A={pv['Away win']:.1%}  "
        f"O2.5={pv['Over 2.5']:.1%}"
    )
    reasoning_lines.append(
        f"ML      H={mv['Home win']:.1%} D={mv['Draw']:.1%} A={mv['Away win']:.1%}  "
        f"O2.5={mv['Over 2.5']:.1%}"
    )
    reasoning_lines.append(
        f"Forebet H={fv['Home win']:.1%} D={fv['Draw']:.1%} A={fv['Away win']:.1%}  "
        f"O2.5={fv['Over 2.5']:.1%}"
    )
    reasoning_lines.extend(vecs["reasoning"])

    # Step 5
    if fused["reasoning"]:
        reasoning_lines.append("── Fusion ──")
        reasoning_lines.extend(fused["reasoning"])

    # Step 6
    if pick6["reasoning"]:
        reasoning_lines.append("── Pick Selection ──")
        reasoning_lines.extend(pick6["reasoning"])

    # Step 7
    if kelly["reasoning"]:
        reasoning_lines.append("── Stake ──")
        reasoning_lines.extend(kelly["reasoning"])

    # Form analysis detail
    fa = xg.get("form_analysis", {})
    if fa.get("reasoning"):
        reasoning_lines.append("")
        reasoning_lines.append("── Form Analysis ──")
        reasoning_lines.extend(fa["reasoning"])

    # ──────────────────────────────────────────────────────────────────────
    # Build picks_summary for HTML display (★ = primary)
    # ──────────────────────────────────────────────────────────────────────
    picks_summary = []
    for c in pick6["candidates"]:
        star = "★" if c == pick6["primary"] else " "
        picks_summary.append(f"{star}{c['market']}: {c['pick']} ({c['confidence']})")

    primary = pick6["primary"]

    # ──────────────────────────────────────────────────────────────────────
    # Return dict — drop-in compatible with analyze_from_data()
    # ──────────────────────────────────────────────────────────────────────
    return {
        # Core fields used by run_forebet_predictions / HTML writer
        "pick": primary["pick"],
        "market": primary["market"],
        "confidence": primary["confidence"],
        "all_picks": pick6["candidates"],
        "picks_summary": picks_summary,
        "score_lean": pick6["correct_score"],
        "reasoning": reasoning_lines,
        "_exp_goals": (xg["exp_h"], xg["exp_a"]),
        "_volatility": ctx["profile"].get("volatility", 0.1),
        "_method": "+".join(fused["method_parts"]),
        "_kelly_stake": kelly["kelly_stake"],
        "_model_prob": primary.get("model_prob"),
        "_implied_prob": primary.get("implied_prob"),
        "_odds": primary.get("odds"),
        "_poisson_probs": (fused["p_home"], fused["p_draw"], fused["p_away"]),
        "_p_over25": fused["p_over25"],
        "_p_under25": fused["p_under25"],
        "_p_btts_yes": fused["p_btts_yes"],
        "_p_btts_no": fused["p_btts_no"],
        "_p_1x": fused["p_1x"],
        "_p_x2": fused["p_x2"],
        "_p_12": fused["p_12"],
        "_ml": vecs["ensemble_result"].get("_ml") if vecs["ensemble_result"] else None,
        "_disagree_vetoed": False,
        # Pipeline-specific fields (extra transparency)
        "_pipeline": {
            "audit": audit,
            "league_ctx": ctx,
            "xg": {k: v for k, v in xg.items() if k != "form_analysis"},
            "agreement": vecs["agreement_score"],
            "gates_fired": pick6["gates_fired"],
            "kelly_detail": kelly,
        },
        "supporting_markets": [],
        # Expose poisson probs for method-tracking in DB
        "poisson_prob_home": fused["p_home"],
        "poisson_prob_draw": fused["p_draw"],
        "poisson_prob_away": fused["p_away"],
        "ml_prob_home": vecs["ml_vec"].get("Home win"),
        "ml_prob_draw": vecs["ml_vec"].get("Draw"),
        "ml_prob_away": vecs["ml_vec"].get("Away win"),
        "method_used": "+".join(fused["method_parts"]),
    }
