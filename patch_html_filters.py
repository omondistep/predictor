#!/usr/bin/env python3
"""Patch existing prediction HTML files to add filter controls and data attributes."""

import re
import sys
from pathlib import Path


FILTER_CSS = """
select { cursor:pointer; }
"""

FILTER_UI = """<div style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:14px;align-items:center;">
<input type="text" id="matchSearch" placeholder="Search teams…" style="flex:1;min-width:180px;max-width:400px;padding:8px 12px;border-radius:6px;border:1px solid #334155;background:#1e293b;color:#e2e8f0;font-size:0.9rem;outline:none;">
<select id="filterExpTotal" style="padding:8px 12px;border-radius:6px;border:1px solid #334155;background:#1e293b;color:#e2e8f0;font-size:0.9rem;outline:none;">
  <option value="">Exp Goals: Any</option>
  <option value="2.0">Under 2.0</option>
  <option value="2.5">Under 2.5</option>
  <option value="3.0">Under 3.0</option>
  <option value="3.5">Under 3.5</option>
  <option value="4.0">Under 4.0</option>
</select>
<select id="filterOutcome" style="padding:8px 12px;border-radius:6px;border:1px solid #334155;background:#1e293b;color:#e2e8f0;font-size:0.9rem;outline:none;">
  <option value="">1X2: Any</option>
  <option value="Home win">Home Win</option>
  <option value="Draw">Draw</option>
  <option value="Away win">Away Win</option>
</select>
<select id="filterConf" style="padding:8px 12px;border-radius:6px;border:1px solid #334155;background:#1e293b;color:#e2e8f0;font-size:0.9rem;outline:none;">
  <option value="">Confidence: Any</option>
  <option value="Near Certain">Near Certain</option>
  <option value="High">High</option>
  <option value="Medium-High">Medium-High</option>
  <option value="Medium">Medium</option>
  <option value="Low">Low</option>
</select>
<select id="filterMarket" style="padding:8px 12px;border-radius:6px;border:1px solid #334155;background:#1e293b;color:#e2e8f0;font-size:0.9rem;outline:none;">
  <option value="">Market: Any</option>
  <option value="1X2">1X2</option>
  <option value="O/U">O/U</option>
  <option value="BTTS">BTTS</option>
  <option value="DC">DC</option>
  <option value="DNB">DNB</option>
</select>
</div>"""

FILTER_JS = """<script>
const cards = document.querySelectorAll('.card');
const searchInput = document.getElementById('matchSearch');
const filterExp = document.getElementById('filterExpTotal');
const filterOutcome = document.getElementById('filterOutcome');
const filterConf = document.getElementById('filterConf');
const filterMarket = document.getElementById('filterMarket');
const matchCount = document.getElementById('matchCount');

function applyFilters() {
  const q = (searchInput.value || '').toLowerCase().trim();
  const maxExp = filterExp.value ? parseFloat(filterExp.value) : null;
  const outcome = filterOutcome.value;
  const conf = filterConf.value;
  const market = filterMarket.value;
  let visible = 0;
  cards.forEach(card => {
    let show = true;
    if (q) {
      const teams = card.querySelector('.teams');
      if (!teams || !q.split(/\\s+/).every(w => teams.textContent.toLowerCase().includes(w))) show = false;
    }
    if (show && maxExp !== null) {
      const t = parseFloat(card.dataset.expTotal);
      if (isNaN(t) || t >= maxExp) show = false;
    }
    if (show && outcome) {
      if (card.dataset.pick !== outcome) show = false;
    }
    if (show && conf) {
      const badge = card.querySelector('.conf-badge');
      if (!badge || !badge.textContent.includes(conf)) show = false;
    }
    if (show && market) {
      if (card.dataset.market !== market) show = false;
    }
    card.style.display = show ? '' : 'none';
    if (show) visible++;
  });
  if (matchCount) matchCount.textContent = visible + ' / ' + cards.length;
}

[searchInput, filterExp, filterOutcome, filterConf, filterMarket].forEach(el => {
  if (el) el.addEventListener(el.tagName === 'INPUT' ? 'input' : 'change', applyFilters);
});
</script>"""


def extract_card_data(card_section):
    """Extract data attributes from a card section (from opening to next card)."""
    data = {}

    # Extract expected goals from "Exp: <span...>X.X-Y.Y</span>" or "Exp: X.X-Y.Y"
    exp_match = re.search(r'Exp:.*?(\d+\.\d+)-(\d+\.\d+)', card_section)
    if exp_match:
        eh, ea = float(exp_match.group(1)), float(exp_match.group(2))
        data['exp-home'] = f"{eh:.2f}"
        data['exp-away'] = f"{ea:.2f}"
        data['exp-total'] = f"{eh + ea:.2f}"
    else:
        data['exp-home'] = ""
        data['exp-away'] = ""
        data['exp-total'] = ""

    # Extract primary market from pick-line: <strong>Pick</strong> (Market)
    pick_match = re.search(
        r'<div class="pick-line"(?:\s[^>]*)?><strong>(.*?)</strong>\s*\((\w+/\w+|\w+)\)',
        card_section
    )
    if pick_match:
        data['pick'] = pick_match.group(1)
        data['market'] = pick_match.group(2)
    else:
        # Fallback: try to find market from the picks table
        market_match = re.search(r'<tr><td>(1X2|O/U|BTTS|DC|DNB)</td><td>(.*?)</td>', card_section)
        if market_match:
            data['market'] = market_match.group(1)
            data['pick'] = market_match.group(2)
        else:
            data['market'] = ""
            data['pick'] = ""

    return data


def patch_html(html_content):
    """Add filter controls and data attributes to HTML content."""
    # Check if already patched (has filter UI and properly placed data attributes)
    if 'filterExpTotal' in html_content and 'data-exp-home="' in html_content:
        # Check that data attributes are on card divs, not on wrong elements
        if re.search(r'<div class="card"[^>]*data-exp-home="', html_content):
            print("  Already patched, skipping.")
            return html_content

    # Add CSS for select elements
    if 'select { cursor:pointer; }' not in html_content:
        html_content = html_content.replace(
            'canvas { max-height:300px; }\n</style>',
            'canvas { max-height:300px; }\nselect { cursor:pointer; }\n</style>'
        )

    # Find all card positions and extract data for each
    card_positions = []
    for m in re.finditer(r'<div class="card" data-match-id="([^"]*)"', html_content):
        card_positions.append((m.start(), m.group(1)))

    # Process cards in reverse order so positions don't shift
    for i in range(len(card_positions) - 1, -1, -1):
        start_pos = card_positions[i][0]
        match_id = card_positions[i][1]
        
        # Find the end of this card (next card or </body>)
        if i + 1 < len(card_positions):
            end_pos = card_positions[i + 1][0]
        else:
            end_pos = html_content.find('</body>', start_pos)
            if end_pos == -1:
                end_pos = len(html_content)
        
        card_section = html_content[start_pos:end_pos]
        attrs = extract_card_data(card_section)
        
        # Find the opening tag's > position
        tag_match = re.match(r'<div class="card" data-match-id="[^"]*"[^>]*>', html_content[start_pos:])
        if tag_match:
            tag_end = start_pos + tag_match.end()
            data_attrs = f' data-exp-home="{attrs["exp-home"]}" data-exp-away="{attrs["exp-away"]}" data-exp-total="{attrs["exp-total"]}" data-market="{attrs["market"]}" data-pick="{attrs["pick"]}"'
            # Insert data attributes before the closing >
            html_content = html_content[:tag_end-1] + data_attrs + html_content[tag_end-1:]

    # Replace single search input with filter bar
    old_search = re.search(
        r'<input type="text" id="matchSearch"[^>]*>',
        html_content
    )
    if old_search:
        html_content = html_content.replace(old_search.group(0), FILTER_UI)

    # Replace existing filter script with new comprehensive one
    old_script = re.search(
        r'<script>\s*const cards = document\.querySelectorAll\(\'\.card\'\);.*?</script>',
        html_content,
        re.DOTALL
    )
    if old_script:
        html_content = html_content.replace(old_script.group(0), FILTER_JS)

    return html_content


def main():
    pred_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("predictions")
    if not pred_dir.exists():
        print(f"Directory not found: {pred_dir}")
        sys.exit(1)

    html_files = sorted(pred_dir.glob("*.html"))
    patched = 0
    for f in html_files:
        if f.name == "index.html":
            continue
        print(f"Patching {f.name}...")
        content = f.read_text()
        new_content = patch_html(content)
        if new_content != content:
            f.write_text(new_content)
            patched += 1
            print(f"  Patched.")
        else:
            print(f"  Skipped (already patched or no cards found).")

    print(f"\nDone. Patched {patched} files.")


if __name__ == "__main__":
    main()
