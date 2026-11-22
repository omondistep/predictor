#!/usr/bin/env python3
"""Filter predictions/latest.html into predictions/output.html.

Keeps cards that satisfy the "filtered output" criteria derived from the
successful betslip analysis:

  - Medium-High confidence
  - Forebet agrees with our 1X2 pick
  - League 7d accuracy > 50%
  - Primary pick is a favourite (odds < 2.0)
  - Forebet O1.5 >= 56% (best-effort, when parseable)

Usage:
    python scripts/filter_output.py [latest.html] [output.html]
Defaults to predictions/latest.html -> predictions/output.html.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from report_paths import stamped_path

PRED_DIR = Path(__file__).resolve().parent.parent / "predictions"


def _attr(block, name):
    m = re.search(rf'data-{name}="([^"]*)"', block)
    return m.group(1) if m else None


def _league7d_pct(block):
    m = re.search(r'League \(7d\):.*?<strong>(\d+)%</strong>', block, re.S)
    return int(m.group(1)) if m else None


def _primary_odds(block):
    market = _attr(block, "market")
    pick = _attr(block, "pick") or ""
    tables = re.findall(r'<table[^>]*>(.*?)</table>', block, re.S)
    if not tables:
        return None
    # 1X2 form table: Home/Draw/Away rows, odds in last numeric <td>
    first = tables[0]
    odds = {}
    for row in re.finditer(r'<tr>\s*<th>(Home|Draw|Away)</th>(.*?)</tr>', first, re.S):
        side = row.group(1)
        tds = re.findall(r'<td>([^<]*)</td>', row.group(2))
        val = None
        for td in reversed(tds):
            td = td.strip()
            if re.fullmatch(r'\d+\.\d+', td):
                val = float(td)
                break
        odds[side] = val
    if market == "1X2":
        if "Home" in pick:
            return odds.get("Home")
        if "Away" in pick:
            return odds.get("Away")
        if "Draw" in pick:
            return odds.get("Draw")
    if market == "O/U" and len(tables) >= 2:
        m = re.search(r'O/U\s*2\.5</th>\s*<td>([\d.]+)/([\d.]+)</td>', tables[1])
        if m:
            if "Over" in pick:
                return float(m.group(1))
            if "Under" in pick:
                return float(m.group(2))
    return None


def _o15(block):
    # Forebet O15 from venue-stats line: "O15: 44% (Forebet)"
    m = re.search(r'O15:\s*(\d+)%\s*\(Forebet\)', block)
    return int(m.group(1)) if m else None


def _keep(block):
    if "Medium-High" not in block:
        return False
    if "Forebet agrees" not in block:
        return False
    l7 = _league7d_pct(block)
    if l7 is None or l7 <= 50:
        return False
    po = _primary_odds(block)
    if po is None or not (po < 2.0):
        return False
    o15 = _o15(block)
    if o15 is not None and o15 < 56:
        return False
    return True


def main(src=None, dst=None):
    if src is None:
        src = Path(sys.argv[1]) if len(sys.argv) > 1 else PRED_DIR / "latest.html"
    else:
        src = Path(src)
    if dst is None:
        dst = Path(sys.argv[2]) if len(sys.argv) > 2 else PRED_DIR / "output.html"
    else:
        dst = Path(dst)
    if not src.exists():
        print(f"Source not found: {src}")
        sys.exit(1)

    html = src.read_text()
    card_starts = [(m.start(), m.group(1))
                   for m in re.finditer(r'<div class="card" data-match-id="([^"]*)"', html)]

    blocks = []
    for i, (start, fid) in enumerate(card_starts):
        end = card_starts[i + 1][0] if i + 1 < len(card_starts) else html.find("<script>", start)
        blocks.append(html[start:end])

    kept = [b for b in blocks if _keep(b)]
    print(f"Filtered {len(kept)}/{len(blocks)} cards")

    first_start = card_starts[0][0] if card_starts else 0
    head = html[:first_start]
    script_idx = html.find("<script>", card_starts[-1][0]) if card_starts else len(html)
    tail = html[script_idx:]

    import datetime
    title = f"Predictions (Filtered: fav odds<2, Forebet agrees, League7d>50%, Medium-High, O1.5>=56%) — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    head = re.sub(r"<title>.*?</title>", f"<title>{title}</title>", head, count=1)

    banner = (
        f'<h1>Predictions Report (Filtered)</h1>\n'
        f'<p class="sub">Filtered output — {len(kept)} picks meeting criteria: '
        f'favorite odds (&lt;2.0), model agrees with Forebet, league 7d accuracy &gt;50%, '
        f'Medium-High confidence, Forebet O1.5 &ge;56%. Generated from {src.name}.</p>\n'
        f'<div class="stats"><div class="stat">Picks <span>{len(kept)}</span></div></div>\n'
    )

    out = head + banner + "".join(kept) + "\n" + tail
    dst = stamped_path(dst)
    dst.write_text(out)
    print(f"Wrote {dst} ({len(out)} bytes)")


if __name__ == "__main__":
    main()
