---
description: Build the bettor's own match-prediction card for a slate — edge vs de-vigged odds, EV, Quarter-Kelly stakes, league reliability. 
agent: bettor
---

Build today's or this slate's betting card — your own call on top of the model,
not just a dump of `pr`.

$ARGUMENTS

`$ARGUMENTS` is either a links file (e.g. `links/today.html`, `upcoming.txt`),
a `pr --json` dump path, or empty (auto-fetch today's slate).

Steps:

1. **Get the slate**
   - If a path is given and ends in `.json`, use it directly.
   - If a path is given, pass it to `pr <path> --json > /tmp/preds.json`.
     (Link files like `links/today.html` / `links/tod.txt` work — `pr` picks
     out the `/football/matches/` URLs.)
   - If nothing is given, run `pr --today` then use the resulting links file.
   - If a single URL is the argument, wrap it in a temp txt file and run `pr`.

2. **Compute the card**
   - `cd /home/sondiek/Documents/GitHub/predictor`
   - `.venv/bin/python scripts/bettor.py card /tmp/preds.json`
   - (This de-vigs the odds, computes edge% and EV% per candidate pick,
     applies Quarter-Kelly capped at 5%, and scales stakes by each league's
     historical hit rate in that market.)

3. **Overlay your judgment** — the part only you can do:
   - Sanity-check the model's direction on the +EV picks; reject any where the
     reasoning doesn't hold or the pick is backed by a single flaky component.
   - Kill picks in red-flag situations: heavy injuries/key players out, dirty
     stats, absurd volatility leagues, unknown tiny cups with no history.
   - Boost note-worthy consensus picks (both models + you agree).
   - If the card is empty or all picks are thin (+EV < 4%), say so plainly.

4. **Print the final card** in self-contained form:
   ```
   SLATE: <date + leagues covered>
   BANK: 100 units (assume)
   #  LEAGUE | MATCH | MARKET:PICK | CONFIDENCE | EV% | EDGE% | ODDS | STAKE
   ...
   OVERALL CALL: <total units risked> | <your confidence in the slate>
   ```
   One-line "why" per top 3 picks. Order by EV.

House rules: no bet without stated edge; batch odds margins are de-vigged by
`bettor.py card`; never commit; if the math says pass, print "Bet nothing."