"""
Forebet match page scraper.

Extracts match data, form, standings, H2H, odds, and predictions
from forebet.com match pages for analysis.
"""

import re
import requests
from bs4 import BeautifulSoup, Tag
from typing import Optional
from datetime import datetime

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


class ForebetScraper:
    """Scrape structured match data from a Forebet match URL."""

    def __init__(self, url: str, timeout: int = 20):
        self.url = url
        self.timeout = timeout
        self.soup = None
        self.data = {
            "forebet_url": url,
            "home_team": "",
            "away_team": "",
            "league": "",
            "match_date": "",
            "match_time": "",
            "home_form": "",
            "away_form": "",
            "home_pos": None,
            "away_pos": None,
            "home_pts": None,
            "away_pts": None,
            "home_games_played": None,
            "away_games_played": None,
            "h2h_home_wins": 0,
            "h2h_draws": 0,
            "h2h_away_wins": 0,
            "h2h_matches": 0,
            "home_avg_goals_for": None,
            "home_avg_goals_against": None,
            "away_avg_goals_for": None,
            "away_avg_goals_against": None,
            "odds_home": None, "odds_draw": None, "odds_away": None,
            "odds_over25": None, "odds_under25": None,
            "odds_btts_yes": None, "odds_btts_no": None,
            "forebet_pred": "",
            "forebet_home_pct": None, "forebet_draw_pct": None, "forebet_away_pct": None,
            "forebet_over25_pct": None, "forebet_btts_yes_pct": None,
        }

    def fetch(self) -> bool:
        """Fetch and parse the Forebet page. Returns True on success."""
        try:
            resp = requests.get(self.url, headers=HEADERS, timeout=self.timeout)
            if resp.status_code != 200:
                return False
            self.soup = BeautifulSoup(resp.text, "html.parser")
            return True
        except Exception as e:
            print(f"  [scraper] Error fetching {self.url}: {e}")
            return False

    def parse(self) -> dict:
        """Parse all data from the fetched page. Returns data dict."""
        if not self.soup:
            return self.data

        self._parse_header()
        self._parse_date_time()
        self._parse_form()
        self._parse_standings()
        self._parse_h2h()
        self._parse_probabilities()
        self._parse_odds()
        return self.data

    def _parse_header(self):
        """Extract teams and league from page header/breadcrumb."""
        # Try h1
        h1 = self.soup.find("h1")
        if h1:
            text = h1.get_text(strip=True)
            # English: "Team A VS Team B", Spanish: "Team A - Team B"
            parts = re.split(r"\s*(?:[Vv][Ss]|-)\s*", text)
            if len(parts) == 2:
                self.data["home_team"] = parts[0].strip()
                self.data["away_team"] = parts[1].strip()

        # Try breadcrumb for league
        bread = self.soup.find("div", {"class": "breadcrumb"})
        if bread:
            links = bread.find_all("a")
            # Usually Football > Country > League > Match
            parts = []
            for link in links:
                txt = link.get_text(strip=True)
                if txt and len(txt) > 2 and txt.lower() not in ("football", "forebet", "home", "predictions"):
                    if "match" not in txt.lower():
                        parts.append(txt)
            if len(parts) >= 2:
                # Combine Country + League (e.g. "Brazil Serie A")
                self.data["league"] = f"{parts[0]} {parts[1]}"
            elif parts:
                self.data["league"] = parts[0]

        # Alternative: find league via the rcnt div first text
        rcnt = self.soup.find("div", {"class": "rcnt"})
        if rcnt and (not self.data.get("league") or len(self.data["league"]) < 4):
            text = rcnt.get_text(" ", strip=True)
            # First few words are usually the league
            words = text.split()[:4]
            league = " ".join(words)
            if league and len(league) > 2:
                self.data["league"] = league

    def _parse_date_time(self):
        """Extract match date and time."""
        date_span = self.soup.find("span", {"class": "date_bah"})
        if date_span:
            text = date_span.get_text(strip=True)
            parts = text.split()
            if len(parts) >= 2:
                self.data["match_date"] = parts[0]
                self.data["match_time"] = parts[1]
            elif len(parts) == 1:
                # Could be just a date
                if "/" in text:
                    self.data["match_date"] = text

    def _parse_form(self):
        """Extract form strings (W/D/L) for both teams."""
        form_divs = self.soup.find_all("div", {"class": "prformcont"})
        for i, div in enumerate(form_divs):
            form_str = div.get_text(strip=True)
            # Normalize: remove extra spaces, keep letters
            form_str = " ".join(form_str.split())
            if i == 0:
                self.data["home_form"] = form_str
            elif i == 1:
                self.data["away_form"] = form_str

    def _parse_standings(self):
        """Extract league standings for both teams."""
        tables = self.soup.find_all("table", {"class": "standings"})
        for table in tables:
            rows = table.find_all("tr")
            data_row_idx = 0
            for row in rows:
                cells = row.find_all("td")
                if len(cells) < 2:
                    continue
                first_text = cells[0].get_text(strip=True) if cells else ""
                if "PTS" in first_text.upper() or "GP" in first_text.upper():
                    continue
                team_cell = cells[1].get_text(strip=True)
                pts = self._safe_float(cells[2].get_text(strip=True)) if len(cells) > 2 else None
                gp = self._safe_int(cells[3].get_text(strip=True)) if len(cells) > 3 else None
                gf = self._safe_int(cells[7].get_text(strip=True)) if len(cells) > 7 else None
                ga = self._safe_int(cells[8].get_text(strip=True)) if len(cells) > 8 else None
                data_row_idx += 1

                pos_text = row.get("data-pos", "")
                pos = self._safe_int(pos_text) or data_row_idx

                home_match = self._team_match(team_cell, self.data["home_team"])
                away_match = self._team_match(team_cell, self.data["away_team"])

                if home_match and self.data["home_pos"] is None:
                    self.data["home_pos"] = pos
                    self.data["home_pts"] = pts
                    self.data["home_games_played"] = gp
                    if gf is not None and ga is not None and gp and gp > 0:
                        self.data["home_avg_goals_for"] = round(gf / gp, 2)
                        self.data["home_avg_goals_against"] = round(ga / gp, 2)
                if away_match and self.data["away_pos"] is None:
                    self.data["away_pos"] = pos
                    self.data["away_pts"] = pts
                    self.data["away_games_played"] = gp
                    if gf is not None and ga is not None and gp and gp > 0:
                        self.data["away_avg_goals_for"] = round(gf / gp, 2)
                        self.data["away_avg_goals_against"] = round(ga / gp, 2)

    def _parse_h2h(self):
        """Extract head-to-head results."""
        tables = self.soup.find_all("table", {"class": "stat-content"})
        for table in tables:
            header = table.find("th") or table.find("td", {"class": "stat_header"})
            if header and "head to head" in header.get_text(strip=True).lower():
                rows = table.find_all("tr")
                home_wins = away_wins = draws = 0
                match_count = 0
                for row in rows[1:]:  # Skip header
                    cells = row.find_all("td")
                    if len(cells) < 3:
                        continue
                    # Last cell usually has the score or result
                    result_cell = cells[-1].get_text(strip=True)
                    # Look for score pattern like "2 - 1"
                    score_match = re.search(r"(\d+)\s*-\s*(\d+)", result_cell)
                    if score_match:
                        match_count += 1
                        h = int(score_match.group(1))
                        a = int(score_match.group(2))
                        if h > a:
                            home_wins += 1
                        elif a > h:
                            away_wins += 1
                        else:
                            draws += 1
                self.data["h2h_home_wins"] = home_wins
                self.data["h2h_draws"] = draws
                self.data["h2h_away_wins"] = away_wins
                self.data["h2h_matches"] = match_count

    def _parse_probabilities(self):
        """Extract Forebet's probability percentages and prediction."""
        fprc_divs = self.soup.find_all("div", {"class": "fprc"})
        for div in fprc_divs:
            text = div.get_text(" ", strip=True)
            # Lines separated by newlines in the div
            lines = text.split("\n")
            clean = " ".join(l.strip() for l in lines if l.strip())

            # 1X2 probabilities: "Prob. % 1 X 2 33 44 23" or "Probabilidad % 1 X 2 22 46 32"
            m1 = re.search(r"(?:Prob\.|Probabilidad)\s*%\s*1\s*X\s*2\s*(\d{1,3})\s*(\d{1,3})\s*(\d{1,3})", clean)
            if m1:
                self.data["forebet_home_pct"] = int(m1.group(1))
                self.data["forebet_draw_pct"] = int(m1.group(2))
                self.data["forebet_away_pct"] = int(m1.group(3))
                # Determine Forebet's prediction
                pcts = [
                    (int(m1.group(1)), "1"),
                    (int(m1.group(2)), "X"),
                    (int(m1.group(3)), "2"),
                ]
                max_pct = max(pcts, key=lambda x: x[0])
                self.data["forebet_pred"] = max_pct[1]
                continue

            # Over/Under 2.5: "Prob. % Under/Over 2.5 48 52" or "Probabilidad % Menos/Más 2.5"
            m2 = re.search(r"(?:Prob\.|Probabilidad)\s*%\s*(?:Under/Over|Menos/Más)\s*2\.5\s*(\d+)\s*(\d+)", clean)
            if m2:
                self.data["forebet_over25_pct"] = int(m2.group(2))
                continue

            # BTTS: "Prob. % No Yes 41 59" or "Probabilidad % No Sí 41 59"
            m3 = re.search(r"(?:Prob\.|Probabilidad)\s*%\s*No\s*(?:Yes|Sí)\s*(\d+)\s*(\d+)", clean)
            if m3:
                self.data["forebet_btts_yes_pct"] = int(m3.group(2))
                continue

    def _parse_odds(self):
        """Extract match odds from the page.

        Forebet has multiple .rcnt divs — one per market (1X2, O/U, BTTS, etc).
        The first div is always the 1X2 market.
        """
        rcnt_divs = self.soup.find_all("div", {"class": "rcnt"})

        for idx, div in enumerate(rcnt_divs):
            text = div.get_text(" ", strip=True)
            temp_match = re.search(r"(\d+)°", text)
            if not temp_match:
                continue
            after_temp = text[temp_match.end():].strip()
            odds_nums_str = re.findall(r"\d+\.\d+", after_temp)
            odds_nums = [float(n) for n in odds_nums_str]

            # First rcnt div = 1X2 market (3 values: home, draw, away)
            if idx == 0:
                # Also parse Forebet probabilities and prediction from before temp
                before_temp = text[:temp_match.start()].strip()
                # Pattern: "... <home%> <draw%> <away%> <1|X|2> <score>..."
                fb_match = re.search(r"(\d{1,3})\s+(\d{1,3})\s+(\d{1,3})\s+([1X2])\s+\d+\s*-\s*\d+", before_temp)
                if fb_match:
                    self.data["forebet_home_pct"] = int(fb_match.group(1))
                    self.data["forebet_draw_pct"] = int(fb_match.group(2))
                    self.data["forebet_away_pct"] = int(fb_match.group(3))
                    self.data["forebet_pred"] = fb_match.group(4)

                if len(odds_nums) >= 3:
                    self.data["odds_home"] = odds_nums[0]
                    self.data["odds_draw"] = odds_nums[1]
                    self.data["odds_away"] = odds_nums[2]
                    if len(odds_nums) >= 5:
                        self.data["odds_over25"] = odds_nums[3]
                        self.data["odds_under25"] = odds_nums[4]

            # Parse other markets from their respective divs
            txt_lower = text.lower()

            if "both to score" in txt_lower or "btts" in txt_lower:
                if len(odds_nums) >= 2:
                    self.data["odds_btts_yes"] = odds_nums[1]
                    self.data["odds_btts_no"] = odds_nums[0]

            if "under/over" in txt_lower or "over/under" in txt_lower:
                if len(odds_nums) >= 2 and self.data["odds_over25"] is None:
                    self.data["odds_over25"] = odds_nums[0]
                    self.data["odds_under25"] = odds_nums[1]

    # ── Helpers ──

    @staticmethod
    def _safe_float(val) -> Optional[float]:
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_int(val) -> Optional[int]:
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _team_match(scraped_name: str, our_name: str) -> bool:
        """Check if a scraped team name matches our stored team name."""
        if not scraped_name or not our_name:
            return False
        s = scraped_name.lower().strip()
        o = our_name.lower().strip()
        # Direct match
        if s == o:
            return True
        # One contains the other
        if len(s) > 2 and len(o) > 2:
            if s in o or o in s:
                return True
        # First word match (for multi-word names)
        s_words = s.split()
        o_words = o.split()
        if s_words and o_words and s_words[0] == o_words[0]:
            return True
        return False

    def scrape(self) -> dict:
        """Convenience: fetch + parse in one call."""
        if self.fetch():
            return self.parse()
        return self.data


def scrape_results_list(url: str) -> list:
    """Scrape a list of results from a Forebet page.
    Returns list of dicts: {"url": str, "home_goals": int, "away_goals": int}
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        
        results = []
        # Matches are usually in div blocks with class 'rcnt'
        rows = soup.find_all("div", {"class": "rcnt"})
        
        if not rows:
            # Fallback to general predict-row or tr tags
            rows = soup.find_all(["tr", "div"], {"class": re.compile(r"(tr_\d+|predict-row)")})

        for row in rows:
            # Look for match link - tnmscn is the common class for the team name link
            link_tag = row.find("a", {"class": "tnmscn"}) or row.find("a", href=re.compile(r"/(?:football/matches/|predictions-tips-)"))
            if not link_tag:
                continue
            
            match_url = link_tag["href"]
            if not match_url.startswith("http"):
                match_url = "https://www.forebet.com" + match_url
            
            # Look for actual score - l_scr is the bold score in the results table
            score_cell = row.find(["b", "span"], {"class": re.compile(r"(l_scr|l_score|lscr_main|res_sc)")})
            
            if score_cell:
                score_text = score_cell.get_text(strip=True)
            else:
                # Fallback: look for text pattern in the whole row
                score_text = row.get_text(" ", strip=True)

            # Match pattern "X - Y" or "X-Y" or "X:Y"
            m = re.search(r"(\d+)\s*[-–:]\s*(\d+)", score_text)
            if m:
                results.append({
                    "url": match_url,
                    "home_goals": int(m.group(1)),
                    "away_goals": int(m.group(2))
                })
        
        # Deduplicate results by URL
        seen_urls = set()
        final_results = []
        for r in results:
            if r["url"] not in seen_urls:
                final_results.append(r)
                seen_urls.add(r["url"])
        
        return final_results
    except Exception as e:
        print(f"  [scraper] Error scraping results list: {e}")
        return []


def scrape_url(url: str) -> dict:
    """Scrape a single Forebet URL and return structured data."""
    scraper = ForebetScraper(url)
    return scraper.scrape()


def scrape_and_save(url: str) -> dict:
    """Scrape a URL and print progress (to stderr)."""
    import sys
    print(f"  Scraping: {url.split('/')[-1][:50]}...", end=" ", file=sys.stderr, flush=True)
    data = scrape_url(url)
    if data.get("home_team"):
        print(f"{data['home_team'][:20]} vs {data['away_team'][:20]}", file=sys.stderr)
    else:
        print("no data", file=sys.stderr)
    return data
