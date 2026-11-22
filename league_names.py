"""Clean Forebet competition labels into usable league names.

Forebet match pages frequently expose the two teams' *standings* text instead of
the competition name, e.g. ``"10th place Paraguay Primera Division 6th place"``.
That string is not a league; it is ``<home position> place <Country> <League>
<away position> place``. Storing it verbatim put ~75% of history.db's resolved
rows into an unusable league bucket, which also starves any per-league expert
layer (it cannot route a match whose league is a standings table).

This module extracts the competition name. Anything it cannot confidently clean
is returned unchanged so callers can treat "still looks like standings" as
unknown rather than inventing a league.
"""

import re

# "10th place" / "2nd place" / "23rd place" / "1st place"
_POS = r"\d+(?:st|nd|rd|th)"

# Primary shape: "<home standings> <Country> <League> <away standings>".
_PRIMARY = re.compile(
    rf"^\s*{_POS}\s+place\s+(.*?)\s+{_POS}\s+place\s*$", re.I
)
# A leading standings phrase, optionally "Nth place in <competition>".
_LEADING = re.compile(rf"^\s*{_POS}\s+place\s+(?:in\s+)?", re.I)
# A trailing standings phrase, optionally "Nth place in <competition>".
# "place" may be at end of string (no trailing whitespace), so the whitespace
# after it is optional.
_TRAILING = re.compile(rf"\s+{_POS}\s+place(?:\s+(?:in\s+)?)?.*$", re.I)


def looks_like_standings(text: str) -> bool:
    """True when *text* contains a "<position> place" standings marker."""
    return bool(text) and bool(re.search(rf"\b{_POS}\s+place\b", text, re.I))


def clean_league_name(text: str) -> str:
    """Return the competition name embedded in a Forebet league label.

    ``"10th place Paraguay Primera Division 6th place"`` -> ``"Paraguay
    Primera Division"``. Text without a standings marker is returned stripped
    and otherwise untouched. When the middle still carries standings phrases
    (cup pages where each side's own league is named), those are stripped too;
    if the result cannot be cleaned confidently the original is returned.
    """
    if not text:
        return ""
    s = re.sub(r"\s+", " ", str(text)).strip()
    if not looks_like_standings(s):
        return s

    m = _PRIMARY.match(s)
    if m:
        candidate = _strip_markers(m.group(1))
        if candidate and not looks_like_standings(candidate):
            return candidate

    candidate = _strip_markers(s)
    if candidate and not looks_like_standings(candidate):
        return candidate

    return s  # cannot clean confidently — caller treats this as unknown


def _strip_markers(text: str) -> str:
    """Remove leading/trailing standings phrases and a dangling leading 'in'."""
    s = _LEADING.sub("", text or "")
    s = _TRAILING.sub("", s)
    s = re.sub(r"^\s*in\s+", "", s, flags=re.I)
    return s.strip()
