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
commit unless asked.

**No EV is quoted for a market whose prices fail `odds_guard` validation.** 88%
of stored odds are unusable, so a card that prices off them is quoting
fiction. Such picks are listed but annotated `odds failed validation — not
priced`.

The reserve/second-team scanner (`_RESERVE_RE` in `scripts/bettor.py`) treats
tokens like "II/III/B/2/U20/U21" in team names as B-teams. Senior clubs with a
Roman numeral in their official name (e.g. Willem II) are allow-listed in
`_SENIOR_CLUB_ALLOW` so they are not wrongly flagged; genuine reserve sides
(Juventus II, KRC Genk II, Zenit-2) still get blocked.

## One Report: `predictions/picks.html`

A prediction run writes **exactly one** HTML file: `predictions/picks.html`.
Open it; ignore anything else in `predictions/`.

It is gated on measured historical performance rather than on the model's own
stated probability. Each pick is looked up in `data/reliability.json` and
placed in one of three tiers:

| Tier | Meaning |
|------|---------|
| **RELIABLE** | ≥200 resolved historical matches of this exact (selection, confidence) pair, and its credibility-shrunk hit rate beats the selection's unconditional base rate |
| **WATCH** | Enough history to measure, but no edge over the base rate — shown, not recommended |
| **HELD** | Suppressed. No history, or the measured rate is no better than the market gets for free |

The base-rate test is the whole point. "Over 1.5 lands 88% of the time" is not
a signal — it is the definition of Over 1.5. What matters is whether the model
beats what you'd get by picking blind.

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

### The gate also demands *significant* edges, not just positive ones

A bucket publishes when its credibility-shrunk hit rate beats the base rate
**and** the one-sided 95% Wilson lower bound of that rate beats the base rate
too (`Z_ONE_SIDED_95 = 1.645` in `scripts/audit_reliability.py`).

Without it the gate tests 54 buckets at a flat threshold, so some publish purely
because they were the luckiest of 54 draws. Asking whether the *lower* bound
clears the bar is a stricter question: is the edge big enough to survive being
re-measured on fresh matches? It costs one bucket today — `Over 3.5|Low` drops
out — and `Draw|Low` survives only because 2,326 matches back it.

Validate the gate out-of-sample before trusting it:

```bash
.venv/bin/python scripts/validate_gate.py --folds 4
```

Each fold fits the gate only on its own past, then scores the approved buckets
on matches the fit never saw. Current results, with the significance floor on:

| Folds | Judged | Kept edge | In-sample | Out-of-sample | Shrinkage |
|-------|--------|-----------|-----------|--------------|-----------|
| 2 | 10 | 9/10 | +10.4 pts | +7.7 pts | −26% |
| 3 | 20 | 18/20 | +9.9 pts | +7.8 pts | −21% |
| 4 | 25 | 23/25 | +10.4 pts | +8.3 pts | −20% |

Edges roughly halve out-of-sample — that shrinkage is the honest number, and it
is why the gate publishes a subset rather than everything that beat its base
rate in-sample.

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

Legacy per-angle reports (`latest`, `best`, `consistent`, `high80`,
`bankers`, `bettor`, `batch_NNN`) are still available behind `--all-reports`.
They each reprint the model's stated probability with no statement of how often
that kind of pick actually lands, which is why they read as confident and are
often wrong:

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
whole-file snapshot, not an append. Nothing else is written — no
`batch_NNN.html` files, which is what the old batcher produced (46MB across 15
near-duplicate files).

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
- `predictions/picks.html` — **the only report.** Open this one.
- The legacy views (`latest`, `batch_NNN`, `best`, `consistent`, `high80`)
  are written only with `pr --all-reports`.

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
