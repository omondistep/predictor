---
description: Implement a data-backed prediction improvement (threshold/gate/profile/weight), then verify it against history. Use for "improve the model", "fix this weakness", "tune X".
agent: bettor
---

Improve the prediction pipeline to close a specific, evidence-backed weakness.

$ARGUMENTS

`$ARGUMENTS` = which weakness to attack (e.g. "FA Cup O/U overconfidence",
"1X2 in brazil-serie-b", "ML sabotages O/U 2.5", or "1X2 calibration no-op").
If empty, run `/audit` first and pick the highest-leak item.

Discipline: a change is only "done" when the before/after evidence proves it.

1. **Confirm the weakness with numbers**
   - Latest audit + DB queries + `pr --calibrate`. Write down the baseline:
     league, market, bucket, n, current hit rate / bias. If n < 30, say the
     weakness is too thin to act on and stop.

2. **Find the root cause** in code (verify line numbers before editing):
   - Per-market thresholds: `_MARKET_THRESHOLDS` predict.py:471
   - Odds floors / value gate: `ODDS_FLOORS` predict.py:3464, `_value_adjust` predict.py:3466
   - League reliability damping: `_model_market_strengths` predict.py:859
   - Auto-threshold learning: `_auto_calibrate_thresholds` predict.py:954
   - League profiles (avg goals / volatility / rates): `LEAGUE_PROFILES` predict.py:89
   - Synthesis re-rank: `synthesis.py` `synthesize` :215
   - Calibration no-op path: `apply_calibration("1X2", ...)` call site in predict.py
   - Weight training: `pr --train-weights` (database.py `train_weights_from_history`)
   - Component gating (ML/BTTS/O-U agreement): predict.py:2560-2639

3. **Make the smallest fix** that a controlled test can prove. Favor:
   - tightening a threshold / gate, adding a min-sample floor to a bias
     correction, damping a league profile, or widening/narrowing a cap —
     over inventing new machinery. Keep changes surgical and commented-free
     (match repo style; do not add comments unless asked).

4. **VERIFY — non-negotiable**
   - Re-run the audit and the targeted SQL query. The change must move the
     metric in the expected direction on the historical data, not just in
     today's fixtures.
   - Optional but ideal: re-train (`pr --learn-calibration`), then re-predict a
     held-out day (e.g. `pr test.txt --json`) and compare hit rate before/after.
   - If it doesn't help, revert it. Be honest: patching for one slate is noise.

5. **Report** concisely:
   - the weakness, root cause (`file:line`), the change made,
   - before → after numbers (n, hit rate / bias / ROI),
   - risk note (what could it break) and a one-line verdict.

House rules: no commit unless asked; do not edit what you cannot verify; a
single-slate patch is not an improvement.