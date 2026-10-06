#!/usr/bin/env python3
"""Canonical reserve / second-team detection, shared by every consumer.

One definition of "is this a B-team?", so the betting card and the report
cannot disagree.  The reserve regex used to live only in ``scripts/bettor.py``,
which meant ``predictions/picks.html`` had no reserve filtering at all -- and
the reliability gate happily published picks on Betis III, UD Almeria B,
Pontevedra II and Puerto Cabello II.  Those are exactly the fixtures where the
model is least trustworthy: its team-strength inputs come from the senior side,
so a reserve match is scored as though the first team were playing.

Being a reserve side is not a prediction about the match, it is a statement that
the inputs the model used do not describe the teams on the pitch.  So the honest
response is to suppress the pick, not to discount it.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# "FC B", "Barinas II", "Atletico III", "U21", "Real Madrid Castilla"-style
# reserves.  \b boundaries keep "Willem II" from matching unless allow-listed.
#
# A bare "IV" is deliberately absent from this looser pattern: case-insensitively
# it matches "Iv." -- the Croatian abbreviation for Ivan -- and so flagged three
# senior clubs ("NK Ivancica Iv.", "Tekstilshchik Iv.").  Reserve sides labelled
# IV do occur, and are still caught, but only by the stricter end-of-name token
# check in _RESERVE_TOKEN_RE below.
_RESERVE_RE = re.compile(
    r"\b(?:II|III|B|Castilla|B\.\s*U\s*2[0-3]|U\s*(?:1[6-9]|2[0-3])|Reserves?)\b"
    # Eastern-European second teams: "Rubin-2 Kazan", "Rodina-2 Moscow",
    # "Zenit-2". A SINGLE digit only, which is what separates a second team from
    # a founding year: second teams are numbered 2 or 3, whereas "SC Weiche-08",
    # "PK-35 Vantaa" and "TP-47 Tornio" are all senior clubs whose name carries
    # the year they were founded. Matching one or two digits flagged those three.
    r"|-\s*\d\b",
    re.IGNORECASE,
)

# Senior clubs whose official name contains a token that otherwise reads as a
# reserve suffix -- a Roman numeral in the club name proper, not a B-team.
# Only genuine cases belong here; widening it silently un-blocks real reserves.
# Kept deliberately short. "Willem II" is the near-universal case across the
# leagues this scraper covers; every entry added must be a club whose official
# name really does carry the numeral, or reserve filtering quietly stops working.
_SENIOR_CLUB_ALLOW = (
    "willem ii",
)

# Standalone-letter / numeral side names that mean the same thing.  Kept
# separate from the regex above so a single club word ("B") cannot disable it
# for a match where the other side is not a reserve.
#
# The token must END the name and be preceded by the start of the string or a
# separator.  Allowing a bare letter prefix glued to the token looked reasonable
# and was wrong: it matched "FCSB" -- a senior Romanian club -- and mangled
# scrape artefacts such as "Kilb", because the "B" was simply the last letter.
_RESERVE_TOKEN_RE = re.compile(
    r"(?:^|[\s.\-_])(?:ii|iii|iv|b|castilla|u1[6-9]|u2[0-3])$",
    re.IGNORECASE,
)


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip())


def is_reserve_side(team: str) -> bool:
    """True when this single team name denotes a reserve / second / youth side."""
    n = _norm(team)
    if not n:
        return False
    if any(a in n.casefold() for a in _SENIOR_CLUB_ALLOW if a):
        return False
    # A side literally named "B" / "II" is a reserve whatever else it says,
    # as is a side whose name *ends* in a separated reserve token.
    if _RESERVE_TOKEN_RE.search(n):
        return True
    return bool(_RESERVE_RE.search(n))


def is_reserve_fixture(home: str, away: str) -> bool:
    """True when either side is a reserve, second, or youth team.

    Both names are considered together so the senior-club allow-list can rescue
    a fixture whose *other* side happens to contain a numeral-looking token --
    but only when the numeral belongs to a club whose real name carries it.
    """
    pair = f"{_norm(home)} {_norm(away)}"
    allow = any(a in pair.casefold() for a in _SENIOR_CLUB_ALLOW if a)
    if allow:
        return False
    return is_reserve_side(home) or is_reserve_side(away)


def reserve_reason(home: str, away: str) -> Optional[Tuple[str, str]]:
    """Which side triggered the reserve flag, as (side, 'home'|'away').

    ``None`` when neither side is a reserve.  Kept separate from the boolean so
    the report can name the offending team instead of saying "something is
    wrong with this fixture".
    """
    for team, side in ((home, "home"), (away, "away")):
        if is_reserve_side(team):
            return team, side
    return None


# Backwards-compatible aliases for the names scripts/bettor.py exports.
RESERVE_RE = _RESERVE_RE
SENIOR_CLUB_ALLOW = _SENIOR_CLUB_ALLOW