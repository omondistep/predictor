# Predictor Instructions

## Bettor Agent (opencode)

A sharp, self-improving betting agent ships in this repo (`.opencode/agent/bettor.md`).
It learns from the model, audits weaknesses, improves the pipeline, and produces
its own staking card. Select it in opencode, or use its commands:

| Command | What it does |
|---------|--------------|
| `/bettor audit` | Full weakness report (confidence calibration, anti-leagues, model-vs-model, bias corrections) |
| `/bettor book <slate>` | Bettor's own card: de-vigged edge, EV%, Quarter-Kelly stakes, league reliability |
| `/bettor improve <weakness>` | Implement a data-backed fix + verify against history |

Supporting toolbox (stdlib, read-only on `history.db`):
```bash
.venv/bin/python scripts/bettor.py audit            # weakness report
.venv/bin/python scripts/bettor.py audit --json     # machine-readable
.venv/bin/python scripts/bettor.py card <preds.json | links.txt>   # betting card
.venv/bin/python scripts/bettor.py slip <slip.txt>                 # multi-bet leg verdict (BET/PASS/KILL/NO-DATA) + card
.venv/bin/python scripts/bettor.py research <slip.txt>            # web-check agenda + evidence skeletons
.venv/bin/python scripts/bettor.py research <slip.txt> --apply    # merge evidence → re-priced verdict
.venv/bin/python scripts/bettor.py learn            # bias analysis + ML retrain
```
House rules the agent follows: no bet without +EV (odds de-vigged), ignore
samples < 25–30, Quarter-Kelly capped at 5%, scrap picks with red flags, never
commit unless asked. An **unmeasured league is not a proven one**: `unknown`
accuracy earns the same 0.5 stake haircut as a measured-but-weak league, and a
ticket that would exceed 100% of the bankroll has its **lowest-EV legs cut**
rather than having every stake scaled down to fill the cap.

**Two house gates keep the worst-priced legs off the ticket** (both stay
visible in the card's candidate list, annotated, never staked):
reserve/second-team fixtures are never bet, and a leg whose
(market, selection, confidence) bucket is **HELD** by the picks gate
(`data/reliability.json`, same ledger `picks.html` uses) is never bet — only
RELIABLE/WATCH buckets are stakeable.

**No EV is quoted for a market whose prices fail `odds_guard` validation.** 88%
of stored odds are unusable, so a card that prices off them is quoting
fiction. Such picks are listed but annotated `odds failed validation — not
priced`.

The reserve/second-team scanner (`_RESERVE_RE` in `scripts/bettor.py`) treats
tokens like "II/III/B/2/U20/U21" in team names as B-teams. Senior clubs with a
Roman numeral in their official name (e.g. Willem II) are allow-listed in
`_SENIOR_CLUB_ALLOW` so they are not wrongly flagged; genuine reserve sides
(Juventus II, KRC Genk II, Zenit-2) still get blocked.

## Report Files: `predictions/picks.html` + card view + bettor card

A prediction run writes **three** HTML files:

| File | What it is |
|------|------------|
| `predictions/picks.html` | The gated report (see below) — the one to read |
| `predictions/latest.html` (slate) or `predictions/<Team>.html` (single match) | The classic card view: every fixture as a card with confidence, reasoning and filters |
| `predictions/bettor.html` | The staking card: de-vigged edge, EV%, Quarter-Kelly stakes |

Everything else in `predictions/` is opt-in via `pr --all-reports`.

The gate in `picks.html` works as follows.

It is gated on measured historical performance rather than on the model's own
stated probability. Each pick is looked up in `data/reliability.json` and
placed in one of three tiers:

| Tier | Meaning |
|------|---------|
| **RELIABLE** | ≥200 resolved historical matches of this exact (selection, confidence) pair, its credibility-shrunk hit rate beats the selection's unconditional base rate, and the 95% lower bound on that rate clears the same base rate by the full `min_edge_pts` |
| **WATCH** | Enough history to measure, but no edge over the base rate — shown, not recommended |
| **HELD** | Suppressed. No history, or the measured rate is no better than the market gets for free |

The base-rate test is the whole point. "Over 1.5 lands 88% of the time" is not
a signal — it is the definition of Over 1.5. What matters is whether the model
beats what you'd get by picking blind.

### Every fixture carries a best bet across all markets

Each row prints a **Best bet** cell: the best selection on that match drawn
from *every* market the feed prices — 1/X/2, both BTTS outcomes, the goal
lines — so the recommendation is not obliged to be the model's primary pick and
is not obliged to be 1/X/2. Two rank orders, in priority:

0. **+EV is a precondition.** A candidate that prices at or below zero is not a
   bet. When nothing on the fixture prices positive the cell prints
   **no +EV selection** and shows the least-bad price underneath as *pass* —
   a negative EV under a Best bet heading reads as a bet, which is the one
   thing it must not.
1. **Tier first, EV inside one tier.** Among the candidates that price
   positive, RELIABLE beats WATCH beats HELD, and `calibrated model
   probability × validated price − 1` only chooses *within* the winning tier.
   The reason is in the next section: EV is where the model's opinion of a
   price diverges from the market, and history measures that divergence as
   model error. A +100% EV on a bucket that never clears its base rate ranks
   below a +20% EV on one that does. The price itself must pass `odds_guard`
   or the candidate is not priced at all; a rejected price is printed as
   rejected, never as a number (88% of stored odds fail, so most rows say so).
2. **Model lift over base rate** — this fixture's calibrated probability minus
   what that selection lands on blind, used when nothing prices. Same
   tier-first order. It must be lift, not the ledger's bucket edge: the bucket
   edge is one number per bucket, so ranking on it names the historically
   juiciest *bucket* rather than the best match — which is exactly what the
   Bucket-edge column warns readers not to do.

The cell also carries the selection's own tier, so a fat EV on a bucket that
never clears its base rate is visibly not a recommendation. Two cell states are
warnings rather than picks: a HELD badge is accompanied by
`no measured evidence (n=…)` naming the sample the selection would be judged
on, and a best bet sitting more than 10pts from the de-vigged market prints
that gap next to its EV. Reserve fixtures never get a best bet: their measured
rates describe senior-team football.

### Expected value is model disagreement, not edge

Measured on 3,146 odds-valid resolved rows (9,438 side observations), binned by
how far the model sits from the de-vigged price:

| model − market | n | model says | market says | realised | model err | market err |
|---|---|---|---|---|---|---|
| −20pt | 614 | 30.5% | 51.5% | 53.4% | −22.9 | −2.0 |
| 0 | 4,221 | 29.5% | 29.5% | 30.3% | −0.8 | −0.8 |
| +20pt | 692 | 59.9% | 38.3% | 39.3% | **+20.6** | −1.0 |

Restricted to longshots (fair ≤25%, the odds range the big EVs come from): model
34.5%, market 13.6%, realised **12.7%** — market off 0.9pt, model off 21.7pt.
Where the model and the book disagree, reality follows the book; the model's
error scales with the gap. So `EV = model × price − 1` measures how wrong the
model is, not what the market missed: a reported `+144% EV` on a longshot
corresponds to roughly −6% at the true probability. This is why the best bet
ranks on tier first, why a gap over 10pts is printed next to the EV, and why
the largest EVs on a page are the least trustworthy numbers on it.

Followed through at real prices (vig included), the model's own confident 1X2
picks do not clear the market either: Home n=750 ROI −5.2%, Away n=445 −2.5%,
Draw n=689 +9.2% (the last drawn from the odds-valid subset, which AGENTS
flags as biased). The measured hit-rate gate — a bucket beating *its own base
rate* — remains the only number the history itself supports.

### The ledger key includes the market

A bucket is `"<market>|<selection>|<confidence>"` — `1X2|Home win|Medium-High`,
`O/U|Over 3.5|Medium`, `1H|Draw|Medium-High`. The market is part of the key
because "Draw lands 24% of the time at full time" and "Draw lands 40% of the
time at half time" are different facts, and pooling them measured each against
a base rate it had no business beating.

The gate scores a market the way the market is actually settled:

- **1H uses `ht_home_goals` / `ht_away_goals`.** Rows without HT scores are
  skipped, never filled in from the full-time result.
- **DNB draws are pushes** — excluded from both the hit numerator and the
  denominator, which is what "not a loss" means.
- Rows written before `our_market` existed are `NULL`; they are scored as
  full-time and counted in `unknown_market_rows` rather than dropped.

**Half-time gating is currently inert, and that is the correct outcome.**
19,905 fixtures have full-time scores but only 1,228 carry HT scores, so every
`1H` bucket is HELD. Do not "fix" this by borrowing full-time goals — that
would publish a bucket measured against the wrong scoreline. It resolves on its
own as `pr results` collects HT scores.

### Reserve sides can never be RELIABLE

`reserve_filter.py` is the single authority on whether a side is a B-team, and
both `reliable_report.py` and `scripts/bettor.py` call it. A fixture with a
reserve side is forced to HELD and names the side, because the model's inputs
for a B-team describe the wrong players — measured performance on those rows
says nothing about the senior side it is named after.

Matching team names is a minefield. The marker is usually a Roman numeral or a
`B` (`Betis III`, `UD Almería B`, `Barcelona U19`, `Real Madrid Castilla`), and
`Willem II`, `FCSB`, `Kilb` and `NK Ivancica Iv.` are all senior clubs that an
earlier pattern wrongly flagged. Eastern-European second teams carry a hyphen
and a single digit — `Rubin-2 Kazan`, `Rodina-2 Moscow`, `Zenit-2` — and that
digit must be **one** digit, which is what separates them from senior clubs
named after a founding year (`SC Weiche-08`, `PK-35 Vantaa`, `TP-47 Tornio`).
Both directions are pinned by tests in `scripts/test_regressions.py`; re-scan
the full team-name population before changing the pattern.

### Base rates are measured, not assumed

`compute_base_rates()` in `scripts/audit_reliability.py` derives each base rate
from **every resolved match in history**, not just the ones the model predicted
— conditioning on the model's own picks would tilt the yardstick toward the
model's habits. Rates use the same time window as the hit rates they judge.

Two properties worth keeping:

- **Thin populations are still measured**, and labelled `measured on a thin
  population (n matches)`. A ±5pt base rate with visible provenance beats an
  invented constant. `DEFAULT_BASE = 0.40` is reserved for an empty database.
- **A self-check asserts every complement pair sums to 1.0** (`Over 2.5` /
  `Under 2.5`, `Yes` / `No`, …). A breach means `resolve_selection()` and the
  `COMPLEMENTS` table have drifted apart, which would silently corrupt every
  edge figure in the gate.

Measured over 19,780 resolved matches (all time):

| Selection | Base | Selection | Base | Selection | Base |
|-----------|------|-----------|------|-----------|------|
| Home win | 43.0% | X2 | 57.0% | Over 2.5 | 54.7% |
| Draw | 24.2% | 12 | 75.8% | Under 2.5 | 45.3% |
| Away win | 32.8% | Over 1.5 | 77.1% | Over 3.5 | 33.6% |
| 1X | 67.2% | Under 1.5 | 22.9% | Under 3.5 | 66.4% |

Correcting these from hand-picked constants moved the gate from 14 to 20
publishable buckets. The largest corrections were `Over 3.5` (42% → 33.6%),
`Under 3.5` (72% → 66.4%) and `Under 1.5` (27% → 22.9%) — the old constants
were badly wrong, and had been suppressing real edges on those lines.

Rebuild the ledger (auto-runs when missing or >7 days stale):

```bash
.venv/bin/python scripts/audit_reliability.py      # stamp odds + measure + write ledger
.venv/bin/python scripts/audit_reliability.py --days 365   # last year only
```

### The gate asks a *relative* significance question, not a fixed one

A bucket publishes when its credibility-shrunk hit rate beats the base rate
**and** the one-sided 95% Wilson lower bound of that rate clears that same base
rate by the full `min_edge_pts` — the bar the shrunk edge itself has to meet
(`Z_ONE_SIDED_95 = 1.645` in `scripts/audit_reliability.py`).

Both conditions are measured against *the selection's own base rate*, which is
the whole design: `Over 3.5` at 35% and `1X2 Home` at 55% are judged by the
same rule even though one can never win half its matches. There is no absolute
"must win more often than it loses" constant anywhere in the gate.

There used to be one — `min_hit_rate = 0.55` on the Wilson lower bound — and it
held 89 of 100 buckets off the page. Four-fold walk-forward comparison is why
it went: with the floor, 10 buckets judged, 8/10 kept their edge, +6.5 pts
out-of-sample (−44% shrinkage); with significance reduced to "clears base at
all", 28 judged / 23 kept — but `O/U|Under 1.5|Medium` (+0.6pt margin) and
`O/U|Over 3.5|Low` (+0.9pt) then published *and* failed out-of-sample. Asking
for the full 2pt margin keeps those two out while releasing seven genuinely
significant buckets the fixed floor was suppressing, giving **17 publishable
buckets today**. `--min-hit` survives as an opt-in absolute floor (default
`0.0`) for anyone who wants to reason about stakeability that way.

Without any significance test the gate tests its buckets at a flat threshold,
so some publish purely because they were the luckiest of the draws. Asking
whether the *lower* bound clears the bar is the stricter question: is the edge
big enough to survive being re-measured on fresh matches?

Validate the gate out-of-sample before trusting it:

```bash
.venv/bin/python scripts/validate_gate.py --folds 4
```

Each fold fits the gate only on its own past, then scores the approved buckets
on matches the fit never saw. Current results (relative 2pt bar, no absolute
floor):

| Folds | Judged | Kept edge | In-sample | Out-of-sample | Shrinkage |
|-------|--------|-----------|-----------|--------------|-----------|
| 2 | 11 | 10/11 | +10.9 pts | +10.1 pts | −7% |
| 3 | 18 | 16/18 | +10.9 pts | +9.4 pts | −14% |
| 4 | 26 | 23/26 | +10.8 pts | +8.9 pts | −18% |

The old absolute floor scored −26% / −21% / −20% on the same folds with a
third of the coverage. Edges still shrink out-of-sample — that shrinkage is the
honest number, and it is why the gate publishes a subset rather than everything
that beat its base rate in-sample.

### Fixtures are counted once

The same Forebet match can be stored two or three times: its URL turns up with
and without percent-encoding (`...-def-belgrano-%28vr%29-2443574` vs
`...-def-belgrano-(vr)-2443574`), and some fixtures were re-predicted on a later
run with a different pick. 152 such groups exist.

`fixture_key()` / `dedupe_fixtures()` collapse them by trailing Forebet match
id, keeping the earliest row, and both `measure()` and `compute_base_rates()`
apply it. Without it a double-stored match sits in the base-rate numerator twice
and the hit-rate denominator once, biasing every edge figure — and it inflates
`n` on a bucket that only ever saw one fixture, which the significance floor
would then read as strong evidence. It moves the base rates by <0.1pt and the
publishable set not at all (19/54 either way); it is here so the numbers mean
what they say.

Legacy per-angle reports (`best`, `consistent`, `high80`, `bankers`,
`batch_NNN`) are still available behind `--all-reports`. They each reprint the
model's stated probability with no statement of how often that kind of pick
actually lands, which is why they read as confident and are often wrong:

```bash
pr --all-reports links.txt
```

## Data Integrity: stored odds are mostly unusable

`odds_guard.py` is the single source of truth for whether a price can be
trusted. An Oct 2026 audit of 19,778 resolved matches found **87.7% of stored
prices unusable**:

- **duplicate** (9,393 rows) — the pre-2026-09-22 legacy parser read the price
  of Forebet's own predicted outcome first, so a home forecast produced
  `home == draw` (1.75/1.75/3.30). 98% of duplicate rows have
  `forebet_pred='1'`.
- **favourite_mismatch** (5,666 rows) — the cheapest price is not the outcome
  Forebet ranks first. On duplicate-free rows the stored favourite agrees with
  Forebet's independent percentage only 23% of the time, worse than chance.
- **ok** (2,110 rows, 12.3%)

Consequences already handled:

- Six ML features (`odds_home/draw/away`, `implied_home_prob`,
  `odds_fb_value_home`, `odds_implied_overround`) were fitting scrambled
  numbers on 70.7% of training rows. `extract_features_from_db_row` now
  substitutes a neutral constant when validation fails, so no tree can split
  on a rejected price.
- No report or tool quotes an EV/ROI from stored odds. `scripts/bettor.py`
  refuses to price a market whose prices fail validation. Historical ROI on the
  surviving subset ran to +145%, which is not a market, it is a bug.

**Forebet's own percentages are reliable** and remain a strong signal. They are
unaffected by any of this.

Never repair a rejected price — which slot is wrong is unknowable. Classify the
row and treat the price as missing.

## Calibration

The ensemble is systematically overconfident. Measured over 16,179 resolved
matches: stated 52.8% vs realised 49.1% (global multiplier **0.931**), varying
by bucket (0.904 at the 40–50% bucket up to 0.960 at 70–80%).

`reliable_report.py` multiplies every displayed probability by the measured
factor for its bucket, so the number on screen is the one history says to
expect. Declared confidence labels are *not* reliable and are never used to
gate output — realised accuracy is High 59.7%, Medium-High 53.0%, Medium 43.7%,
Low 29.4%.

## Recent-performance fixes (Oct 2026)

Measured on ex-ante 1X2 picks, Jul–Oct 2026 (9,145 rows). Created-after-kickoff
rows are excluded from the measurement but not yet from ML training.

- **1X2 Draw is never the primary pick.** Draw picks hit 25.6% against a 24.1%
  base rate; the ML's Home/Away call on the same rows hit 43.4%. A Draw primary
  is demoted to the model's favoured side at Low confidence (predict.py,
  "1X2 Draw is never the primary pick"). The ledger already withholds every
  1X2 Draw bucket, so this changes the raw pick, not the published report.
- **Half-time goals are never ML features.** `ht_home_goals` / `ht_away_goals`
  are written after the match; they were in the deployed feature set and are
  always unknown at prediction time. Indices 61–63 are neutralised and excluded
  by `POST_MATCH_FEATURES` in `ml_model.py`.
- **Pre-match Elo features** (`elo_diff`, `elo_home`, `elo_away`, indices
  128–130) come from `elo_pass()`, which rates only earlier results. Gain is
  marginal: a LightGBM walk-forward (Jul–Oct) showed −0.0072 log-loss; the
  production-model holdout (train <Sep 1, test ex-ante Sep–Oct) showed no
  measurable change.
- Known open items: `O/U Over 4.5` hits 17.2% against an 18.4% base; 1H Draw
  is at base rate but has only 86 scored half-time rows to judge from.

## Core Commands

| Command | Description |
|---------|-------------|
| `pr <file>` | Run predictions for matches in a link file |
| `pr <url>` | Predict a single match URL |
| `pr results` | Scrape Forebet for scores, update DB + HTML reports |
| `pr --all-reports <file>` | Also write the legacy secondary reports |
| `pr --today` | Auto-fetch today's + tonight's (00:00–06:00) matches per league → `links/today.html` |
| `pr links/today.html` | Same thing — refetches the slate first if it was built on an earlier date |
| `pr --no-refresh <file>` | Predict a generated slate exactly as it sits on disk |
| `pr --weekday` | Auto-fetch all Mon–Fri matches (this week if mid-week, next week if weekend) via Forebet's AJAX per day → `links/weekday.html` |
| `pr --high-only <file>` | Only show High / Near Certain picks |
| `pr --no-ml <file>` | Classic Poisson only (no ML ensemble) |
| `pr --json <file>` | JSON output |
| `pr --workers N <file>` | Concurrent scrapes (default 4). Scraping is 93% of run time |
| `pr --analysis pipeline <file>` | Seven-step analyzer from `pipeline.py` (see "One CLI, two backends") |
| `pr --train-weights` | Train ensemble weights from historical predictions |

## Generated slates refresh themselves

`links/today.html` and `links/weekday.html` carry the `.html` extension for
historical reasons but hold one URL per line, and they go stale silently:
yesterday's `today.html` is a list of matches that have already been played, so
`pr links/today.html` "succeeds" while predicting nothing bettable.

Two mechanisms prevent that wasted run:

- `_write_links_file()` stamps each generated file with
  `# generated <ISO timestamp> — N match URL(s)`. The reader already skipped
  `#` lines, so this costs nothing, and `_links_generated_at()` falls back to
  file mtime for older hand-made files.
- `_slate_needs_refresh()` treats a `today` slate as stale the moment the
  calendar rolls over, and a `weekday` slate as stale past 12 hours. When stale,
  `pr <file>` re-fetches before predicting and says so. `--no-refresh` opts out.

Independently, a run whose fixtures are *all* already finished now ends with a
plain warning rather than a silent success. That combination is what made the
bettor card report "no bets" on a slate that had been overtaken by time.

## Long Slates: incremental output and `--workers`

`links/weekday.html` holds ~946 matches. Serially that is ~45 minutes of
apparent hang. Two mechanisms fix it:

**Live output.** A run rewrites the single `predictions/picks.html` every 50
matches (`PUBLISH_EVERY`). Open it while the run continues; it is always a
whole-file snapshot, not an append. Nothing else is written *mid-run* — no
`batch_NNN.html` files, which is what the old batcher produced (46MB across 15
near-duplicate files). The card view and `bettor.html` are written once, at the
end of the run.

**Overlapped fetching.** Measured per match on live fixtures: 2.5s scraping
against 0.23s of analysis, so scraping is 93% of wall clock and it is spent
waiting on Forebet. `--workers` overlaps those fetches while analysis stays
serial in the consumer thread (SQLite is a single writer, and 0.23s is not
worth parallelising). Prefetch preserves input order, and a worker that
explodes yields a per-match error rather than killing the run.

Forebet throttles by **request volume per IP**, so the gain shrinks the longer
you run and eventually inverts. Measured on never-before-fetched fixtures, 40+
each, run long enough to reach steady state:

| Workers | Per match | Throughput |
|---------|-----------|-----------|
| 1 | 2.50s | 24.0/min |
| 2 | **1.71s** | **35.1/min** ← default |
| 4 | 1.89s | 31.7/min |
| 6 | 9.4s | throttled |
| 8 | 6.57s | throttled |

**Measure long, not short.** An earlier 8-fixture burst test put 6 workers at
1.00s/match and made it look like the sweet spot. It was an artefact: bursts
hide the throttle. A real 1,088-match run at 6 workers averaged 9.4s a match —
three times *worse* than serial. A speedup measured over 20 matches tells you
nothing about a 1,000-match run, which is the only case where speed matters.
Yields were identical at 1, 2 and 4 workers.

## One CLI, two backends

`predict2.py` was deleted. It was a hand-copied 497-line fork of the prediction
loop and it was silently broken: unlike `predict.py` it never called
`add_pending_result()`, so **none of its 2,183 predictions were ever queued for
result scraping and not one was ever resolved**. Its accuracy could not be
measured, and it had already drifted (no `--all-reports`, no report gating,
broken `--json` keys).

The analyzer it used was worth keeping, so it now lives behind one flag:

```bash
pr --analysis pipeline <file>     # or equivalently: pr2 <file>
```

Both backends share the loop, the DB columns and the report, which is the point:
`method_used` carries a `+pipeline` suffix on pipeline rows, so once results
come back the two are directly comparable. `pr2` remains as a two-line shim for
muscle memory — it now forwards to `pr --analysis pipeline` and, unlike the
fork, its picks get scored.

## Workflow

### 1. Predict matches
```bash
pr links.txt          # Run all links
pr data/links.txt     # Run all links (explicit path)
pr upcoming.txt       # Run upcoming matches
```

### 2. Update results (after matches finish)
```bash
pr results            # Scrape Forebet for scores → update DB + HTML → auto-learn
```

### 3. Learn from results (manual, if needed)
```bash
pr --calibrate            # Show accuracy stats per confidence level and league
```

### 4. View reports
- `predictions/picks.html` — **the gated report.** Open this one.
- `predictions/latest.html` — the full card view (single-match runs write
  `predictions/<Team>.html` instead, same format).
- `predictions/bettor.html` — the staking card.
- The remaining legacy views (`batch_NNN`, `best`, `consistent`, `high80`,
  `bankers`) are written only with `pr --all-reports`.

## How It All Fits Together

```
pr links.txt               → Predictions saved to history.db + HTML reports
                           → data/pending_results.json updated with new matches
                           → Cron scheduled to retrain model in 18h

pr results                 → Reads data/pending_results.json
                           → Scrapes Forebet for finished scores
                           → Updates history.db + HTML files
                           → Removes completed entries from pending file
                           → Auto-learn: analyze bias + retrain ML if 50+ new examples

Cron (scheduled by pr)     → Runs auto_retrain() after 18h
                           → Retrains the ML model on latest data
```

**Key point:** `pr results` does NOT trigger learning. It only updates data. Learning happens via `pr --learn-calibration` or the cron job scheduled by `pr links.txt`.

## Aliases

| Alias | What it does |
|-------|--------------|
| `pr` | Runs predict.py with venv Python |
| `output` | Lists (and opens in file manager) every HTML file in `predictions/` |

## Key Files

| File | Purpose |
|------|---------|
| `history.db` | All predictions + results + calibration data |
| `data/pending_results.json` | Matches still needing result updates |
| `data/links.txt` | Forebet URLs to predict |
| `predictions/` | HTML reports (numbered) |
| `calibration_params.json` | Isotonic regression calibration params |
| `ml_models/ml_predictor/` | Trained ML model components |
| `odds_guard.py` | Odds-integrity validator — the only authority on whether a price can be trusted |
| `reliable_report.py` | Builds `predictions/picks.html`, the single gated report |
| `data/reliability.json` | The reliability ledger: measured hit rate + base rate per (selection, confidence), calibration curve, odds-integrity counts |
| `scripts/audit_reliability.py` | Rebuilds the ledger (stamp odds → measure → gate → write JSON) |
| `scripts/validate_gate.py` | Walk-forward validation of the gate (fit on the past, score on unseen matches) |
| `scripts/test_regressions.py` | 16 offline regression tests — odds parser, prefetch ordering, Wilson math |
| `pipeline.py` | Seven-step analyzer behind `pr --analysis pipeline` |

## Replicate to a New Machine (Linux / macOS / Windows)

The full system ships in git — DB, calibration params and trained ML models —
plus a snapshot of regenerated state in `seed/data/`. To stand up an identical
copy:

```bash
git clone <repo-url> predictor
cd predictor
bash scripts/install.sh        # Linux / macOS (delegates to install.ps1 from Git-Bash)
# or on Windows (cmd / PowerShell):
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
```

The installer:
1. Creates `.venv` and installs `requirements.txt` (includes xgboost, lightgbm, curl_cffi)
2. Restores `data/{pending_results,leagues_db,predictions}.json` from `seed/data/`
3. Verifies ML models load from git (only retrains if broken/missing)
4. Installs the `pr` command — `~/.local/bin/pr` symlink on Linux/macOS,
   `pr.cmd` + `pr.ps1` shims in `%USERPROFILE%\.local\bin` on Windows

**Keep snapshots fresh after big state changes:**
```bash
cp data/pending_results.json data/leagues_db.json data/predictions.json seed/data/
git add seed/data/ history.db ml_models/ && git commit
```

## Auto-Retrain (Cron)

A cron job is automatically scheduled when you run predictions. It retrains the model 18 hours later.

```bash
crontab -l                    # View scheduled jobs
tail -f /tmp/retrain.log      # Watch retrain logs
```

## DB Quick Stats

```bash
sqlite3 history.db "SELECT COUNT(*) as total, SUM(CASE WHEN actual_home_goals IS NOT NULL THEN 1 ELSE 0 END) as with_results FROM matches;"
```
