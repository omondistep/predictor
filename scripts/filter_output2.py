#!/usr/bin/env python3
"""Filter predictions/latest.html -> predictions/output2.html.

Keeps cards where:
  - League (7d) badge is GREEN (hot rating, >=65% accuracy)
  - Forebet agrees with our 1X2 pick (🤝)

Usage:
    python scripts/filter_output2.py [latest.html] [output.html]
Defaults to predictions/latest.html -> predictions/output2.html.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from report_paths import stamped_path

PRED_DIR = Path(__file__).resolve().parent.parent / "predictions"


def main(src=None, dst=None):
    if src is None:
        src = Path(sys.argv[1]) if len(sys.argv) > 1 else PRED_DIR / "latest.html"
    else:
        src = Path(src)
    if dst is None:
        dst = Path(sys.argv[2]) if len(sys.argv) > 2 else PRED_DIR / "output2.html"
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

    kept = []
    for b in blocks:
        # green league 7d badge
        green = ("rgba(34,197,94,0.15)" in b) and ("League (7d)" in b)
        agrees = "Forebet agrees" in b
        if green and agrees:
            kept.append(b)
    print(f"Filtered {len(kept)}/{len(blocks)} cards")

    first_start = card_starts[0][0] if card_starts else 0
    head = html[:first_start]
    script_idx = html.find("<script>", card_starts[-1][0]) if card_starts else len(html)
    tail = html[script_idx:]

    import datetime
    title = f"Predictions (Filtered2: green League7d + Forebet agrees) — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    head = re.sub(r"<title>.*?</title>", f"<title>{title}</title>", head, count=1)

    banner = (
        f'<h1>Predictions Report (Filtered 2)</h1>\n'
        f'<p class="sub">Filtered output — {len(kept)} picks where League (7d) is GREEN (hot) '
        f'and Forebet agrees. Generated from {src.name}.</p>\n'
        f'<div class="stats"><div class="stat">Picks <span>{len(kept)}</span></div></div>\n'
    )

    out = head + banner + "".join(kept) + "\n" + tail
    dst = stamped_path(dst)
    dst.write_text(out)
    print(f"Wrote {dst} ({len(out)} bytes)")


if __name__ == "__main__":
    main()
