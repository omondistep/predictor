"""Holistic prediction synthesis.

Instead of ranking candidate picks by an isolated per-market confidence score,
this module fuses ALL available signals for a match into a single composite
"decision value" per candidate. The goal is to treat the match as a whole:

  - model probability (calibrated, sums to 1)
  - edge vs the bookmaker market (value)
  - agreement across independent components (Poisson / ML / Forebet / form /
    transitivity / H2H) -- conviction rises when they agree, falls when split
  - uncertainty (volatility, data-quality, sample size) -- shrinks conviction
  - draw tendency -- discounts 1X2 side picks, favours Draw / coverage
  - coverage -- goal-based markets span more outcomes (mild tie-breaker only)

The output is a re-ranked candidate list plus a plain-language synthesis
rationale that explains *why* the top pick wins by combining the signals,
rather than listing them separately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from database import get_base_rates, pick_base_rate


# Confidence -> numeric 0..1 (higher = more confident)
CONF_VALUE = {
    "Near Certain": 1.0,
    "High": 0.8,
    "Medium-High": 0.6,
    "Medium": 0.4,
    "Low": 0.15,
}

# Outcome coverage: how many of the 3 base 1X2 results a pick spans.
COVERAGE = {
    ("DC", "1X"): 2, ("DC", "X2"): 2, ("DC", "12"): 2,
    ("O/U", "Over 1.5"): 3, ("O/U", "Under 1.5"): 3,
    ("O/U", "Over 2.5"): 3, ("O/U", "Under 2.5"): 3,
    ("O/U", "Over 3.5"): 3, ("O/U", "Under 3.5"): 3,
    ("BTTS", "Yes"): 3, ("BTTS", "No"): 3,
    ("DNB", "Home"): 1, ("DNB", "Away"): 1,
    ("1X2", "Home win"): 1, ("1X2", "Draw"): 1, ("1X2", "Away win"): 1,
}


@dataclass
class MatchContext:
    """Everything the synthesis layer needs about one match."""
    p_home: float = 0.0
    p_draw: float = 0.0
    p_away: float = 0.0
    exp_h: float = 0.0
    exp_a: float = 0.0
    vol: float = 0.1
    form_signal: float = 0.0          # negative favours home, positive favours away
    trans_signal: float = 0.0         # same sign convention as form_signal
    draw_tendency: bool = False
    draw_factors: int = 0
    h2h_matches: int = 0
    h2h_avg_goals: float = 0.0
    top_pick: str = ""                # model's raw top 1X2 outcome
    margin: float = 0.0               # top_prob - second_prob
    league_reliability: float = 1.0
    base_rates: dict = field(default_factory=dict)  # empirical per-league outcome rates
    warnings: list = field(default_factory=list)
    # component agreement signals (each -1..+1; sign = home/away direction,
    # magnitude = strength). None means component unavailable.
    poisson_dir: Optional[float] = None
    ml_dir: Optional[float] = None
    forebet_dir: Optional[float] = None
    # market odds (decimal); None if unavailable
    odds_home: Optional[float] = None
    odds_draw: Optional[float] = None
    odds_away: Optional[float] = None
    # Forebet win percentages (almost always present even when decimal odds are
    # missing) — used to derive an effective price when real odds are absent.
    fb_home_pct: Optional[float] = None
    fb_draw_pct: Optional[float] = None
    fb_away_pct: Optional[float] = None
    # Draw bias suppression and away win boost flags
    draw_bias_suppressed: bool = False
    away_win_boosted: bool = False


@dataclass
class SynthCandidate:
    market: str
    pick: str
    confidence: str
    model_prob: Optional[float] = None
    reason: str = ""
    coverage: int = 1
    always_show: bool = False
    # filled by synthesis
    decision_value: float = 0.0
    components: dict = field(default_factory=dict)
    synth_note: str = ""


# ---------------------------------------------------------------------------
# Component agreement
# ---------------------------------------------------------------------------
def _dir_for_1x2(ctx: MatchContext) -> Optional[float]:
    """Net 1X2 direction from the model probs: +1 = strong away, -1 = strong home."""
    if ctx.p_home + ctx.p_draw + ctx.p_away <= 0:
        return None
    return (ctx.p_away - ctx.p_home) / max(ctx.p_home + ctx.p_draw + ctx.p_away, 1e-6)


def _forebet_dir(ctx: MatchContext) -> Optional[float]:
    if not ctx.odds_home or not ctx.odds_draw or not ctx.odds_away:
        return None
    ih, idr, ia = 1 / ctx.odds_home, 1 / ctx.odds_draw, 1 / ctx.odds_away
    tot = ih + idr + ia
    if tot <= 0:
        return None
    return (ia - ih) / tot


def component_agreement(ctx: MatchContext):
    """Return a -1..+1 consensus signal and the number of agreeing components.

    Aggregates Poisson (model probs), ML direction, Forebet direction and
    transitivity/form into one signed consensus. Magnitude reflects how tightly
    the independent sources agree.
    """
    comps = []
    d = _dir_for_1x2(ctx)
    if d is not None:
        comps.append(("model", d))
    fb = _forebet_dir(ctx)
    if fb is not None:
        comps.append(("forebet", fb))
    if ctx.ml_dir is not None:
        comps.append(("ml", ctx.ml_dir))
    if ctx.trans_signal != 0:
        comps.append(("trans", ctx.trans_signal))
    if ctx.form_signal != 0:
        comps.append(("form", ctx.form_signal))

    if not comps:
        return 0.0, 0

    vals = [max(-1.0, min(1.0, v)) for _, v in comps]
    mean_dir = sum(vals) / len(vals)
    if len(vals) > 1:
        spread = (max(vals) - min(vals)) / 2.0
    else:
        spread = 0.0
    agreement = (1.0 - spread) * (1.0 if abs(mean_dir) > 1e-6 else 0.0)
    source_factor = min(1.0, 0.5 + 0.25 * len(vals))
    consensus = mean_dir * agreement * source_factor
    return max(-1.0, min(1.0, consensus)), len(comps)


# ---------------------------------------------------------------------------
# Edge vs market
# ---------------------------------------------------------------------------
# When real decimal odds are missing, derive an effective price from Forebet's
# win percentages (almost always present) by stripping the bookmaker vig.
VIG_HAIRCUT = 0.92  # keep ~8% margin out of the derived price


def _derive_odds_from_fb(pct_home, pct_draw, pct_away, pick):
    table = {"Home win": pct_home, "Draw": pct_draw, "Away win": pct_away}
    pct = table.get(pick)
    if not pct or pct <= 0:
        return None
    implied = pct / 100.0
    if implied >= 1.0:
        return None
    return round(1.0 / (implied / VIG_HAIRCUT), 3)


def edge_for(ctx: MatchContext, market: str, pick: str) -> Optional[float]:
    """Positive when model prob exceeds bookmaker-implied prob.

    Uses real decimal odds when available; otherwise derives an effective price
    from Forebet win percentages so the value term still fires on matches that
    lack market odds.
    """
    if market != "1X2":
        return None
    odds = {"Home win": ctx.odds_home, "Draw": ctx.odds_draw, "Away win": ctx.odds_away}.get(pick)
    if odds is None or odds <= 1.0:
        odds = _derive_odds_from_fb(ctx.fb_home_pct, ctx.fb_draw_pct, ctx.fb_away_pct, pick)
    prob = {"Home win": ctx.p_home, "Draw": ctx.p_draw, "Away win": ctx.p_away}.get(pick)
    if odds is None or prob is None or odds <= 1.0:
        return None
    implied = 1.0 / odds
    return prob - implied


# ---------------------------------------------------------------------------
# Uncertainty penalty
# ---------------------------------------------------------------------------
def uncertainty(ctx: MatchContext) -> float:
    """0 = certain, 1 = maximally uncertain. Shrinks conviction."""
    u = 0.0
    u += min(1.0, ctx.vol / 0.30) * 0.5
    u += (1.0 - ctx.league_reliability) * 0.3
    u += min(0.2, 0.04 * len(ctx.warnings))
    if ctx.h2h_matches and ctx.h2h_matches < 3:
        u += 0.05
    return min(1.0, u)


# ---------------------------------------------------------------------------
# Core synthesis
# ---------------------------------------------------------------------------
def synthesize(ctx: MatchContext, candidates: list, ml_only: bool = False) -> list:
    """Re-rank candidates by a holistic decision value.

    Returns the candidate dicts (mutated) sorted best-first, each carrying
    `decision_value`, `components` and `synth_note`.
    
    When ml_only=True, skips the 1X2 side-pick bonus so the ML model
    picks independently based on probability and coverage.
    """
    consensus, n_sources = component_agreement(ctx)
    unc = uncertainty(ctx)
    conv_base = 1.0 - unc

    drawish = ctx.draw_tendency or (ctx.p_draw >= ctx.p_home and ctx.p_draw >= ctx.p_away)

    results = []
    for c in candidates:
        market = c.get("market")
        pick = c.get("pick")
        prob = c.get("model_prob")
        cov = COVERAGE.get((market, pick), 1)

        comp = {}

        if prob is None:
            prob = CONF_VALUE.get(c.get("confidence", "Low"), 0.15)
        # Score a pick by how far it exceeds its market's base rate, not its
        # raw probability. Wide markets (Under 3.5) are naturally ~80% likely;
        # crediting that raw probability lets them dominate every ranking.
        base = pick_base_rate(ctx.base_rates, market, pick)
        prob_component = max(0.0, prob - base)

        # DC normalization: DC "1X" covers 2 outcomes so its raw prob is
        # inherently higher than 1X2 "Home win" (1 outcome). Normalize by
        # the probability mass of the covered outcomes to make cross-market
        # comparison fair.
        if market == "DC" and prob is not None:
            if pick == "1X":
                dc_mass = ctx.p_home + ctx.p_draw
            elif pick == "X2":
                dc_mass = ctx.p_draw + ctx.p_away
            elif pick == "12":
                dc_mass = ctx.p_home + ctx.p_away
            else:
                dc_mass = prob
            if dc_mass > 0:
                # Use the normalized probability (prob / mass of covered outcomes)
                # so DC "1X" at 0.73 over mass 0.73 becomes 1.0, comparable to
                # 1X2 "Home win" at 0.45
                normalized_prob = prob / dc_mass
                prob_component = max(0.0, normalized_prob - base)

        comp["prob"] = round(prob_component, 3)

        edge = edge_for(ctx, market, pick)
        comp["edge"] = round(edge, 3) if edge is not None else None

        if market == "1X2":
            if pick == "Home win":
                dir_align = -consensus
            elif pick == "Away win":
                dir_align = consensus
            else:
                # Draw = "no clear side". A draw is the weakest, hardest-to-predict
                # outcome; forcing align=1.0 on drawish matches let weak 1X2 Draw
                # picks systematically outrank strong, well-calibrated O/U Under
                # picks (1X2 overall ~36%, O/U ~67%). Cap it so draws must earn
                # their rank via real probability/edge, not a free conviction boost.
                dir_align = 0.2 * (1.0 - abs(consensus))
        else:
            dir_align = consensus * 0.3
        comp["align"] = round(dir_align, 3)

        comp["conv"] = round(conv_base, 3)

        value = (
            prob_component * 0.50
            + max(0.0, edge or 0.0) * 0.50
            + dir_align * 0.12 * conv_base
        )
        value += CONF_VALUE.get(c.get("confidence", "Low"), 0.15) * 0.12

        # Market-reliability rebalance. Measured calibration accuracies (from
        # settled history) are: DC ~85%, O/U ~67%, DNB ~56%, BTTS ~53%,
        # 1X2 ~36% (worst of all). The old code gave 1X2 the LARGEST bonus,
        # boosting its weakest market the most. Flip that: reward the strong
        # markets (O/U, DC) when they carry genuine value, and shrink the
        # artificial 1X2 side-pick boost so near-random 1X2 picks don't
        # outrank well-calibrated O/U/DC picks.

        # Shrunk 1X2 side-pick boost: 1X2 is the weakest market, so it gets the
        # smallest conviction bonus, not the largest (was +0.15/+0.08).
        if not ml_only and market == "1X2" and pick in ("Home win", "Away win"):
            if prob and prob >= 0.52 and abs(dir_align) >= 0.35:
                value += 0.06  # strong side pick bonus (reduced from 0.15)
            elif prob and prob >= 0.46 and abs(dir_align) >= 0.25:
                value += 0.03  # moderate side pick bonus (reduced from 0.08)

        # Strong-market preference: reward O/U and DC picks that actually beat
        # their league base rate (genuine value), so the proven-strong markets
        # win ties and near-ties against 1X2. Gated on prob_component>0 and a
        # non-trivial probability so we never surface a value-less O/U pick.
        if market == "O/U" and prob_component > 0.02 and c.get("confidence") != "Low":
            value += 0.07
        elif market == "DC" and prob_component > 0.05 and c.get("confidence") != "Low":
            value += 0.05

        if drawish and market == "1X2" and pick != "Draw":
            value *= 0.88  # reduced from 0.80 to let strong 1X2 side picks compete with DC
        # NOTE: removed the old "drawish and pick == Draw: value *= 1.12" boost —
        # it double-counted with the (already-capped) Draw alignment bonus and
        # inflated 1X2 Draw above calibrated O/U picks, a proven weak-market bias.
        if drawish and market == "O/U" and "Under" in pick:
            value *= 1.05

        value = max(0.0, value)
        c["decision_value"] = value
        c["components"] = comp
        c["coverage"] = cov
        results.append(c)

    results.sort(key=lambda c: c["decision_value"], reverse=True)
    return results


def build_synthesis_rationale(ctx: MatchContext, ranked: list, top: dict) -> str:
    """Plain-language explanation of the holistic decision."""
    consensus, n_sources = component_agreement(ctx)
    unc = uncertainty(ctx)
    bits = []

    direction = ("home" if consensus < -0.05 else "away" if consensus > 0.05 else "balanced")
    if consensus != 0 or n_sources:
        bits.append(
            f"Component consensus favours {direction} "
            f"(agreement {consensus:+.2f} across {n_sources} sources)"
        )

    edge = edge_for(ctx, top.get("market", ""), top.get("pick", ""))
    if edge is not None and edge > 0.02:
        bits.append(f"Model edge vs market +{edge:.0%} (value present)")
    elif edge is not None and edge < -0.02:
        bits.append(f"Market favours this more than model ({edge:.0%}) -- limited value")

    # Explain why O/U won over 1X2 picks when they differ
    top_market = top.get("market", "")
    if top_market == "O/U" and direction in ("home", "away"):
        # Find the best 1X2 pick
        best_1x2 = None
        for c in ranked:
            if c.get("market") == "1X2" and c.get("pick") in ("Home win", "Away win"):
                best_1x2 = c
                break
        if best_1x2:
            best_1x2_dv = best_1x2.get("decision_value", 0)
            top_dv = top.get("decision_value", 0)
            if top_dv > best_1x2_dv:
                bits.append(
                    f"O/U pick scored higher than 1X2 {best_1x2['pick']} "
                    f"(coverage {top.get('coverage', 1)} vs {best_1x2.get('coverage', 1)} outcomes)"
                )

    # Enhanced draw tendency analysis with suppression awareness
    if ctx.draw_tendency or ctx.p_draw >= max(ctx.p_home, ctx.p_away):
        if ctx.draw_bias_suppressed:
            bits.append(f"Draw bias suppressed (form/trans favor clear side)")
        elif ctx.draw_factors >= 5:
            bits.append(f"Strong draw signal ({ctx.draw_factors} factors) -- side picks heavily discounted")
        elif ctx.draw_factors >= 3:
            bits.append(f"Moderate draw signal ({ctx.draw_factors} factors) -- side picks discounted")
        else:
            bits.append("Draw tendency detected -- side picks discounted")

    # Away win probability analysis with boost awareness
    if ctx.away_win_boosted:
        bits.append(f"Away win probability ({ctx.p_away:.0%}) boosted by form/trans signals")
    elif ctx.p_away > ctx.p_home and consensus > 0.1:
        bits.append(f"Away win probability ({ctx.p_away:.0%}) favoured by component consensus")

    if unc >= 0.5:
        bits.append(f"High uncertainty (vol {ctx.vol:.2f}, {len(ctx.warnings)} data warnings) -- conviction reduced")

    if top.get("market") == "1X2":
        bits.append(f"Model: H {ctx.p_home:.0%} / D {ctx.p_draw:.0%} / A {ctx.p_away:.0%}")
    else:
        bits.append(f"Expected goals {ctx.exp_h:.1f}-{ctx.exp_a:.1f} (total {ctx.exp_h+ctx.exp_a:.1f})")

    return "; ".join(bits)


# ---------------------------------------------------------------------------
# Convenience: build context from a prediction dict produced by predict.py
# ---------------------------------------------------------------------------
def context_from_pred(pred: dict, data: dict, vol: float = 0.1,
                      form_signal: float = 0.0, trans_signal: float = 0.0,
                      draw_tendency: bool = False, draw_factors: int = 0,
                      top_pick: str = "", margin: float = 0.0,
                      league_reliability: float = 1.0,
                      ml_dir: Optional[float] = None,
                      draw_bias_suppressed: bool = False,
                      away_win_boosted: bool = False) -> MatchContext:
    ph, pd, pa = pred.get("_poisson_probs", (0.0, 0.0, 0.0))
    eh, ea = pred.get("_exp_goals", (0.0, 0.0))
    return MatchContext(
        p_home=ph, p_draw=pd, p_away=pa,
        exp_h=eh, exp_a=ea,
        vol=vol, form_signal=form_signal, trans_signal=trans_signal,
        draw_tendency=draw_tendency, draw_factors=draw_factors,
        h2h_matches=data.get("h2h_matches", 0) or 0,
        h2h_avg_goals=data.get("h2h_avg_total_goals", 0) or 0.0,
        top_pick=top_pick, margin=margin,
        league_reliability=league_reliability,
        base_rates=get_base_rates(data.get("league", "")),
        warnings=pred.get("_warnings", []),
        odds_home=data.get("odds_home"), odds_draw=data.get("odds_draw"),
        odds_away=data.get("odds_away"),
        fb_home_pct=data.get("forebet_home_pct"),
        fb_draw_pct=data.get("forebet_draw_pct"),
        fb_away_pct=data.get("forebet_away_pct"),
        ml_dir=ml_dir,
        draw_bias_suppressed=draw_bias_suppressed,
        away_win_boosted=away_win_boosted,
    )
