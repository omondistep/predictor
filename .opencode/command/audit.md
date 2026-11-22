---
description: Full model audit — confidence calibration, worst leagues per market, per-component accuracy, bias corrections. Use to find where predictions leak money.
agent: bettor
---

Run a complete weakness audit of the prediction model and report it back.

$ARGUMENTS

Follow the Bettor's four-step discipline (GATHER → DIAGNOSE → SELECT/IMPROVE →
VERIFY). For this task, stop at DIAGNOSE unless I ask for a fix.

Steps, in order:

1. **GATHER**
   - `cd /home/sondiek/Documents/GitHub/predictor`
   - `.venv/bin/python scripts/bettor.py audit`
   - `pr --calibrate`
   - `pr --calibration-report`
   - Cross-check with direct SQL on `history.db` (confidence buckets, per-league
     per-market accuracy with `HAVING COUNT(*) >= 30`, `model_market_accuracy`,
     `calibration_bias`, `ml_league_accuracy`).

2. **DIAGNOSE** — synthesize into a ranked Weakness Report:
   - **Overconfident buckets**: confidence levels whose actual hit rate trails
     the target by 3+ points.
   - **Anti-leagues**: leagues/markets under ~50% (e.g. FA Cup O/U, K League 2
     1X2) — the model leaks money there.
   - **Sabotaging components**: where ML or a sub-model undercuts the blend
     (e.g. ML trailing Poisson badly in specific leagues/markets).
   - **Dangerous corrections**: stored bias corrections with |bias| > 0.5 on
     small samples (n < 15) — are they noise? Should they be floored?
   - **Structural gaps**: dead calibration paths (e.g. 1X2 `apply_calibration`
     is a no-op), threshold caps that look stale.
   - Each weakness gets: the number (actual %, n, market, league), a cause
     hypothesis with a `file:line` reference, and a suggested fix.

3. End with a short verdict: the top 3 things that are costing the most edge
   and how you would fix each — no code changes unless I ask.

Respect the house rules: no hand-waving, sample sizes below 25-30 are noise,
never commit.