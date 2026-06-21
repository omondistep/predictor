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
            "h2h_goals_for": 0,
            "h2h_goals_against": 0,
            "h2h_avg_total_goals": 0,
            "h2h_weighted_form": 0.5,
            "h2h_details": [],
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
            "home_home_avg_goals_for": None,
            "home_home_avg_goals_against": None,
            "away_away_avg_goals_for": None,
            "away_away_avg_goals_against": None,
            "home_over15_pct": None, "home_under15_pct": None,
            "home_over25_pct": None, "home_under25_pct": None,
            "home_over35_pct": None, "home_under35_pct": None,
            "home_btts_yes_pct": None, "home_btts_no_pct": None,
            "away_over15_pct": None, "away_under15_pct": None,
            "away_over25_pct": None, "away_under25_pct": None,
            "away_over35_pct": None, "away_under35_pct": None,
            "away_btts_yes_pct": None, "away_btts_no_pct": None,
            "home_scored_pct": None, "home_conceded_pct": None,
            "away_scored_pct": None, "away_conceded_pct": None,
            "home_total_shots_pg": None,
            "home_shots_ontarget_pct": None,
            "away_total_shots_pg": None,
            "away_shots_ontarget_pct": None,
            "home_clean_sheets_pct": None,
            "away_clean_sheets_pct": None,
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
        self._parse_venue_matches()
        self._parse_ou_btts()
        self._parse_shots()
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
        """Extract head-to-head results.

        Parses the condensed hidd_stat text and the structured st_row
        elements inside the "Head to head" section.  Falls back to the
        old table.stat-content approach when the new layout is absent.
        """
        # --- Try the current div-based layout first ---
        h2h_div = None
        for mt in self.soup.find_all("div", class_="moduletable"):
            mptlt = mt.find("div", class_="mptlt")
            if mptlt and "head to head" in mptlt.get_text(strip=True).lower():
                td = mt.find_parent("td", class_="floatLeft")
                if td is None:
                    td = mt.parent  # fallback
                h2h_div = td
                break

        if h2h_div is not None:
            # Prefer the condensed hidd_stat text – it contains every match.
            hidd = h2h_div.find("div", class_="hidd_stat")
            if hidd:
                raw = hidd.get_text(" ", strip=True)
                self._parse_h2h_from_text(raw)

        # If hidd_stat was empty, try parsing from the visual st_row structure
        if self.data.get("h2h_matches", 0) == 0:
            self._parse_h2h_from_strows()

        # Last resort: old table.stat-content layout
        if self.data.get("h2h_matches", 0) == 0:
            self._parse_h2h_fallback()

    def _parse_h2h_from_text(self, raw):
        """Parse hidd_stat condensed text into structured H2H data.

        Format per match:
          dd/mm YYYY HomeTeam home - away (ht_home - ht_away) AwayTeam ClCode

        Competition codes seen: Cl1, Cl2, ClC, etc.
        """
        home_team = self.data.get("home_team", "")
        away_team = self.data.get("away_team", "")
        home_wins = away_wins = draws = 0
        match_count = 0
        goals_for = 0
        goals_against = 0
        details = []

        pattern = (
            r"(\d{2}/\d{2})\s+(\d{4})\s+"
            r"(.+?)\s+"
            r"(\d+)\s*-\s*(\d+)\s*"
            r"\(([^)]*)\)\s+"
            r"(.+?)\s+"
            r"([A-Z][a-z]\w*)"
        )
        for m in re.finditer(pattern, raw):
            date_str = f"{m.group(1)}/{m.group(2)}"
            h_name = m.group(3).strip()
            hg = int(m.group(4))
            ag = int(m.group(5))
            ht_str = m.group(6).strip()
            a_name = m.group(7).strip()
            comp = m.group(8).strip()

            match_count += 1
            if hg > ag:
                home_wins += 1
            elif ag > hg:
                away_wins += 1
            else:
                draws += 1

            # Determine perspective: is home/away team in H2H match our target team?
            h_prefix = h_name.lower()[:max(4, len(h_name)//2)]
            a_prefix = a_name.lower()[:max(4, len(a_name)//2)]
            ours = home_team.lower()[:max(4, len(home_team)//2)] if home_team else ""
            theirs = away_team.lower()[:max(4, len(away_team)//2)] if away_team else ""

            h_is_us = ours and h_prefix.startswith(ours)
            h_is_them = theirs and h_prefix.startswith(theirs)
            a_is_us = ours and a_prefix.startswith(ours)
            a_is_them = theirs and a_prefix.startswith(theirs)

            # Resolve conflicts: prefer the match that identifies one side clearly
            if h_is_us and not h_is_them:
                # We were the home team in this H2H match
                goals_for += hg
                goals_against += ag
                details.append({
                    "date": date_str,
                    "opponent": a_name,
                    "goals_for": hg,
                    "goals_against": ag,
                    "total_goals": hg + ag,
                    "ht_goals_for": self._parse_ht_home(ht_str),
                    "ht_goals_against": self._parse_ht_away(ht_str),
                    "venue": "home",
                    "competition": comp,
                    "result": "W" if hg > ag else ("L" if ag > hg else "D"),
                })
            elif a_is_us and not a_is_them:
                # We were the away team in this H2H match
                goals_for += ag
                goals_against += hg
                details.append({
                    "date": date_str,
                    "opponent": h_name,
                    "goals_for": ag,
                    "goals_against": hg,
                    "total_goals": hg + ag,
                    "ht_goals_for": self._parse_ht_away(ht_str),
                    "ht_goals_against": self._parse_ht_home(ht_str),
                    "venue": "away",
                    "competition": comp,
                    "result": "W" if ag > hg else ("L" if hg > ag else "D"),
                })
            elif h_is_them and not h_is_us:
                # We are the away team; opponent was home
                goals_for += ag
                goals_against += hg
                details.append({
                    "date": date_str,
                    "opponent": h_name,
                    "goals_for": ag,
                    "goals_against": hg,
                    "total_goals": hg + ag,
                    "ht_goals_for": self._parse_ht_away(ht_str),
                    "ht_goals_against": self._parse_ht_home(ht_str),
                    "venue": "away",
                    "competition": comp,
                    "result": "W" if ag > hg else ("L" if hg > ag else "D"),
                })
            else:
                # Can't confidently identify – generic perspective (first team)
                goals_for += hg
                goals_against += ag
                details.append({
                    "date": date_str,
                    "opponent": a_name,
                    "goals_for": hg,
                    "goals_against": ag,
                    "total_goals": hg + ag,
                    "ht_goals_for": 0,
                    "ht_goals_against": 0,
                    "venue": "neutral",
                    "competition": comp,
                    "result": "W" if hg > ag else ("L" if ag > hg else "D"),
                })

        self.data["h2h_home_wins"] = home_wins
        self.data["h2h_draws"] = draws
        self.data["h2h_away_wins"] = away_wins
        self.data["h2h_matches"] = match_count
        self.data["h2h_goals_for"] = goals_for
        self.data["h2h_goals_against"] = goals_against
        self.data["h2h_avg_total_goals"] = round(
            (goals_for + goals_against) / match_count, 2
        ) if match_count > 0 else 0
        self.data["h2h_details"] = details

        # Compute recency-weighted H2H form (last 3 matches weighted 2x)
        if details:
            recent = details[-3:]
            old = details[:-3]
            rw_pts = sum(
                (3 if d["result"] == "W" else 1 if d["result"] == "D" else 0) * 2
                for d in recent
            )
            ow_pts = sum(
                (3 if d["result"] == "W" else 1 if d["result"] == "D" else 0)
                for d in old
            )
            max_rw = len(recent) * 3 * 2
            max_ow = len(old) * 3
            total_pts = rw_pts + ow_pts
            total_max = max_rw + max_ow
            self.data["h2h_weighted_form"] = round(
                total_pts / total_max, 2
            ) if total_max > 0 else 0.5
        else:
            self.data["h2h_weighted_form"] = 0.5

    @staticmethod
    def _parse_ht_home(ht_str):
        if ht_str and "-" in ht_str:
            parts = ht_str.split("-")
            val = parts[0].strip()
            if val.isdigit():
                return int(val)
        return 0

    @staticmethod
    def _parse_ht_away(ht_str):
        if ht_str and "-" in ht_str:
            parts = ht_str.split("-")
            if len(parts) > 1:
                val = parts[1].strip()
                if val.isdigit():
                    return int(val)
        return 0

    def _parse_h2h_from_strows(self):
        """Parse H2H from visual st_row elements when hidd_stat is empty.

        Some pages (e.g. Cameroon Elite Two) have empty hidd_stat but
        the data is present in st_row divs inside the head-to-head section.
        st_ateam and st_ltag are siblings of st_row, not children.
        """
        home_team = self.data.get("home_team", "")
        away_team = self.data.get("away_team", "")
        home_wins = away_wins = draws = 0
        match_count = 0
        goals_for = 0
        goals_against = 0
        details = []

        for mt in self.soup.find_all("div", class_="moduletable"):
            mptlt = mt.find("div", class_="mptlt")
            if not mptlt or "head to head" not in mptlt.get_text(strip=True).lower():
                continue
            # Collect all st_row divs within this module
            rows = mt.find_all("div", class_=lambda c: c and "st_row" in c.split())
            for row in rows:
                date_div = row.find("div", class_="st_date")
                if not date_div:
                    continue
                date_parts = date_div.find_all("div")
                if len(date_parts) < 2:
                    continue
                date_str = f"{date_parts[0].get_text(strip=True)}/{date_parts[1].get_text(strip=True)}"

                hteam_div = row.find("div", class_="st_hteam")
                if not hteam_div:
                    continue
                h_name = hteam_div.get_text(strip=True)

                res_span = row.find("span", class_="st_res")
                if not res_span:
                    continue
                score_match = re.search(r"(\d+)\s*-\s*(\d+)", res_span.get_text(strip=True))
                if not score_match:
                    continue
                hg = int(score_match.group(1))
                ag = int(score_match.group(2))

                ht_span = row.find("span", class_="st_htscr")
                ht_str = ht_span.get_text(strip=True) if ht_span else ""

                # st_ateam and st_ltag are next siblings of st_row
                a_name = away_team
                comp = ""
                for ns in row.find_next_siblings():
                    if "st_ateam" in ns.get("class", []):
                        a_name = ns.get_text(strip=True)
                    elif "st_ltag" in ns.get("class", []):
                        comp = ns.get_text(strip=True)
                    # Stop at next st_row or module boundary
                    if "st_row" in ns.get("class", []) or "moduletable" in ns.get("class", []):
                        break

                # Determine perspective: is h_name our home team or the away opponent?
                h_prefix = h_name.lower()[:max(4, len(h_name)//2)]
                ours = home_team.lower()[:max(4, len(home_team)//2)] if home_team else ""
                theirs = away_team.lower()[:max(4, len(away_team)//2)] if away_team else ""

                h_is_us = ours and h_prefix.startswith(ours)
                h_is_them = theirs and h_prefix.startswith(theirs)

                match_count += 1
                if hg > ag:
                    home_wins += 1
                elif ag > hg:
                    away_wins += 1
                else:
                    draws += 1

                if h_is_us:
                    goals_for += hg
                    goals_against += ag
                    details.append({
                        "date": date_str, "opponent": a_name,
                        "goals_for": hg, "goals_against": ag,
                        "total_goals": hg + ag,
                        "ht_goals_for": self._parse_ht_home(ht_str),
                        "ht_goals_against": self._parse_ht_away(ht_str),
                        "venue": "home", "competition": comp,
                        "result": "W" if hg > ag else ("L" if ag > hg else "D"),
                    })
                elif h_is_them:
                    goals_for += ag
                    goals_against += hg
                    details.append({
                        "date": date_str, "opponent": h_name,
                        "goals_for": ag, "goals_against": hg,
                        "total_goals": hg + ag,
                        "ht_goals_for": self._parse_ht_away(ht_str),
                        "ht_goals_against": self._parse_ht_home(ht_str),
                        "venue": "away", "competition": comp,
                        "result": "W" if ag > hg else ("L" if hg > ag else "D"),
                    })
                else:
                    goals_for += hg
                    goals_against += ag
                    details.append({
                        "date": date_str, "opponent": a_name,
                        "goals_for": hg, "goals_against": ag,
                        "total_goals": hg + ag,
                        "ht_goals_for": 0, "ht_goals_against": 0,
                        "venue": "neutral", "competition": comp,
                        "result": "W" if hg > ag else ("L" if ag > hg else "D"),
                    })

            break  # Only first H2H module

        if match_count > 0:
            self.data["h2h_home_wins"] = home_wins
            self.data["h2h_draws"] = draws
            self.data["h2h_away_wins"] = away_wins
            self.data["h2h_matches"] = match_count
            self.data["h2h_goals_for"] = goals_for
            self.data["h2h_goals_against"] = goals_against
            self.data["h2h_avg_total_goals"] = round(
                (goals_for + goals_against) / match_count, 2
            )
            self.data["h2h_details"] = details
            # Recency-weighted form
            if details:
                recent = details[-3:]
                old = details[:-3]
                rw_pts = sum((3 if d["result"] == "W" else 1 if d["result"] == "D" else 0) * 2 for d in recent)
                ow_pts = sum((3 if d["result"] == "W" else 1 if d["result"] == "D" else 0) for d in old)
                total_pts = rw_pts + ow_pts
                total_max = len(recent) * 6 + len(old) * 3
                self.data["h2h_weighted_form"] = round(total_pts / total_max, 2) if total_max > 0 else 0.5

    def _parse_h2h_fallback(self):
        """Fallback parser using old table.stat-content layout."""
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
                    result_cell = cells[-1].get_text(strip=True)
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

    def _parse_venue_matches(self):
        """Parse venue-specific match data from hidd_stat sections.

        Extracts goal averages for home-team-at-home and away-team-away
        from the 'CDH home matches' and 'PUM away matches' blocks.
        """
        for hidd in self.soup.find_all("div", class_="hidd_stat"):
            prev = hidd.find_previous("div", class_="mptlt")
            if not prev:
                continue
            label = prev.get_text(" ", strip=True).lower()

            if "home matches" in label:
                raw = hidd.get_text(" ", strip=True)
                goals_for = goals_against = matches = 0
                for m in re.finditer(
                    r"(\d{2}/\d{2})\s+(\d{4})\s+"
                    r"(.+?)\s+(\d+)\s*-\s*(\d+)\s*\(([^)]*)\)\s+(.+?)\s+([A-Z][a-z]\w*)",
                    raw,
                ):
                    matches += 1
                    goals_for += int(m.group(4))
                    goals_against += int(m.group(5))
                if matches > 0:
                    self.data["home_home_avg_goals_for"] = round(goals_for / matches, 2)
                    self.data["home_home_avg_goals_against"] = round(goals_against / matches, 2)

            elif "away matches" in label:
                raw = hidd.get_text(" ", strip=True)
                goals_for = goals_against = matches = 0
                for m in re.finditer(
                    r"(\d{2}/\d{2})\s+(\d{4})\s+"
                    r"(.+?)\s+(\d+)\s*-\s*(\d+)\s*\(([^)]*)\)\s+(.+?)\s+([A-Z][a-z]\w*)",
                    raw,
                ):
                    matches += 1
                    goals_for += int(m.group(5))
                    goals_against += int(m.group(4))
                if matches > 0:
                    self.data["away_away_avg_goals_for"] = round(goals_for / matches, 2)
                    self.data["away_away_avg_goals_against"] = round(goals_against / matches, 2)

    def _parse_ou_btts(self):
        """Parse O/U 1.5/2.5/3.5 and BTTS percentages from stats section.

        Uses the pie_chart_container divs and their os_goals_section3_info
        siblings to identify goal lines.
        """
        containers = self.soup.find_all("div", class_="os_goals_section3_pie_chart_container")
        for c in containers:
            txt = c.get_text(" ", strip=True)
            is_btts = "__bottom_chart" in (c.get("class") or [])

            if is_btts:
                # BTTS: "Yes 24 50% 50% No 24"
                m = re.match(r"Yes\s+(\d+)\s+(\d+)%\s+(\d+)%\s+No\s+(\d+)", txt)
                if m:
                    yes_pct = int(m.group(2))
                    no_pct = int(m.group(3))
                    if self.data.get("home_btts_yes_pct") is None:
                        self.data["home_btts_yes_pct"] = yes_pct
                        self.data["home_btts_no_pct"] = no_pct
                    else:
                        self.data["away_btts_yes_pct"] = yes_pct
                        self.data["away_btts_no_pct"] = no_pct
            else:
                # O/U: "Under /Over 14 34 29% 71%"
                m = re.match(
                    r"Under\s*/?\s*Over\s+(\d+)\s+(\d+)\s+(\d+)%\s+(\d+)%",
                    txt,
                )
                if not m:
                    continue
                under_cnt, over_cnt, under_pct, over_pct = (
                    int(m.group(1)), int(m.group(2)),
                    int(m.group(3)), int(m.group(4)),
                )

                # Determine goal line from sibling os_goals_section3_info
                goal_line = None
                ns = c.find_next_sibling("div", class_="os_goals_section3_info")
                if ns:
                    ns_txt = ns.get_text(" ", strip=True)
                    if "1.5" in ns_txt:
                        goal_line = 1.5
                    elif "2.5" in ns_txt:
                        goal_line = 2.5
                    elif "3.5" in ns_txt:
                        goal_line = 3.5
                if goal_line is None:
                    ps = c.find_previous_sibling("div", class_="os_goals_section3_info")
                    if ps:
                        ps_txt = ps.get_text(" ", strip=True)
                        if "1.5" in ps_txt:
                            goal_line = 1.5
                        elif "2.5" in ps_txt:
                            goal_line = 2.5
                        elif "3.5" in ps_txt:
                            goal_line = 3.5
                if goal_line is None:
                    continue

                # Determine CDH vs PUM: CDH has next sibling with goal info,
                # PUM has prev sibling with goal info
                has_next_info = c.find_next_sibling("div", class_="os_goals_section3_info") is not None

                prefix = "home" if has_next_info else "away"
                if goal_line == 1.5:
                    self.data[f"{prefix}_over15_pct"] = over_pct
                    self.data[f"{prefix}_under15_pct"] = under_pct
                elif goal_line == 2.5:
                    self.data[f"{prefix}_over25_pct"] = over_pct
                    self.data[f"{prefix}_under25_pct"] = under_pct
                elif goal_line == 3.5:
                    self.data[f"{prefix}_over35_pct"] = over_pct
                    self.data[f"{prefix}_under35_pct"] = under_pct

        # Scored a goal % (Yes/No counts with percentages)
        yn_div = self.soup.find("div", class_="os_goals_section2_container")
        if yn_div:
            txt = yn_div.get_text(" ", strip=True)
            scored_matches = re.findall(
                r"Yes\s+No\s+(\d+)\s+\((\d+)%\)\s+(\d+)\s+\((\d+)%\)",
                txt,
            )
            if len(scored_matches) >= 2:
                self.data["home_scored_pct"] = int(scored_matches[0][1])
                self.data["home_conceded_pct"] = int(scored_matches[0][3])
                self.data["away_scored_pct"] = int(scored_matches[1][1])
                self.data["away_conceded_pct"] = int(scored_matches[1][3])

    def _parse_shots(self):
        """Parse shots on target and total shots per game from stats section.

        Two HTML variants:
          - Newer pages use data-stat/data-team attributes (team="h"/"a")
          - Older pages use CDH/PUM text labels
        """
        os_parents = self.soup.find_all("div", class_="os_shots_parent")
        for parent in os_parents:
            # Try data-attribute approach first
            h_total = parent.find("span", {"data-stat": "shots_total", "data-team": "h"})
            a_total = parent.find("span", {"data-stat": "shots_total", "data-team": "a"})
            h_avg = parent.find("span", {"data-stat": "shots_total_avg", "data-team": "h"})
            a_avg = parent.find("span", {"data-stat": "shots_total_avg", "data-team": "a"})
            h_on = parent.find("span", {"data-stat": "shots_on_target", "data-team": "h"})
            a_on = parent.find("span", {"data-stat": "shots_on_target", "data-team": "a"})

            if h_avg:
                val = h_avg.get_text(strip=True)
                if val.replace('.', '').replace('-', '').replace(' ', '').isdigit():
                    self.data["home_total_shots_pg"] = float(val)
            if a_avg:
                val = a_avg.get_text(strip=True)
                if val.replace('.', '').replace('-', '').replace(' ', '').isdigit():
                    self.data["away_total_shots_pg"] = float(val)
            if h_total:
                val = h_total.get_text(strip=True)
                if val.isdigit():
                    self.data["home_total_shots"] = int(val)
            if a_total:
                val = a_total.get_text(strip=True)
                if val.isdigit():
                    self.data["away_total_shots"] = int(val)

            # ON target % from data-stat or text (may include '%' suffix)
            if h_on:
                val = h_on.get_text(strip=True).replace('%', '').strip()
                if val.replace('.', '').replace('-', '').isdigit():
                    self.data["home_shots_ontarget_pct"] = int(float(val))
            if a_on:
                val = a_on.get_text(strip=True).replace('%', '').strip()
                if val.replace('.', '').replace('-', '').isdigit():
                    self.data["away_shots_ontarget_pct"] = int(float(val))

            # Fallback to text-based CDH/PUM regex for older pages
            if self.data.get("home_total_shots_pg") is None:
                txt = parent.get_text(" ", strip=True)
                cdh_match = re.search(r"CDH.*?Total shots\s+(\d+)\s+([\d.]+)", txt)
                if cdh_match:
                    self.data["home_total_shots_pg"] = float(cdh_match.group(2))
                cdh_on = re.search(r"CDH.*?(\d+)%\s*ON\s+target", txt)
                if cdh_on:
                    self.data["home_shots_ontarget_pct"] = int(cdh_on.group(1))
                pum_match = re.search(r"PUM.*?Total shots\s+(\d+)\s+([\d.]+)", txt)
                if pum_match:
                    self.data["away_total_shots_pg"] = float(pum_match.group(2))
                pum_on = re.search(r"PUM.*?(\d+)%\s*ON\s+target", txt)
                if pum_on:
                    self.data["away_shots_ontarget_pct"] = int(pum_on.group(1))

        # Clean sheets from os_others_container
        others_div = self.soup.find("div", class_="os_others_container")
        if others_div:
            txt = others_div.get_text(" ", strip=True)
            # Played games: look for two numbers around "Played games"
            # e.g. "YAF 19 Played games 20 UNI" or "CDH 19 Played games 20 PUM"
            gp_m = re.search(r"(\d+)\s+Played games\s+(\d+)", txt)
            cs_m = re.search(r"(\d+)\s+Clean sheets\s+(\d+)", txt)
            if gp_m and cs_m:
                home_gp = int(gp_m.group(1))
                away_gp = int(gp_m.group(2))
                home_cs = int(cs_m.group(1))
                away_cs = int(cs_m.group(2))
                if home_gp > 0:
                    self.data["home_clean_sheets_pct"] = round(home_cs / home_gp * 100, 1)
                if away_gp > 0:
                    self.data["away_clean_sheets_pct"] = round(away_cs / away_gp * 100, 1)

    def _parse_probabilities(self):
        """Extract Forebet's probability percentages and prediction.

        Handles both combined divs ("Prob. % 1 X 2 33 38 29") and
        split divs ("Prob. % 1 X 2" in one div, "33 38 29" in next).
        """
        fprc_divs = self.soup.find_all("div", {"class": "fprc"})
        parts = []
        for div in fprc_divs:
            text = div.get_text(" ", strip=True)
            lines = text.split("\n")
            clean = " ".join(l.strip() for l in lines if l.strip())
            if clean:
                parts.append(clean)
        merged = " ".join(parts)

        # 1X2 probabilities: "Prob. % 1 X 2 33 44 23" or "Probabilidad % 1 X 2 22 46 32"
        m1 = re.search(r"(?:Prob\.|Probabilidad)\s*%\s*1\s*X\s*2\s*(\d{1,3})\s*(\d{1,3})\s*(\d{1,3})", merged)
        if m1:
            self.data["forebet_home_pct"] = int(m1.group(1))
            self.data["forebet_draw_pct"] = int(m1.group(2))
            self.data["forebet_away_pct"] = int(m1.group(3))
            pcts = [
                (int(m1.group(1)), "1"),
                (int(m1.group(2)), "X"),
                (int(m1.group(3)), "2"),
            ]
            max_pct = max(pcts, key=lambda x: x[0])
            self.data["forebet_pred"] = max_pct[1]

        # Over/Under 2.5: "Prob. % Under/Over 2.5 48 52" or "Probabilidad % Menos/Más 2.5"
        m2 = re.search(r"(?:Prob\.|Probabilidad)\s*%\s*(?:Under/Over|Menos/Más)\s*2\.5\s*(\d+)\s*(\d+)", merged)
        if m2:
            self.data["forebet_over25_pct"] = int(m2.group(2))

        # BTTS: "Prob. % No Yes 41 59" or "Probabilidad % No Sí 41 59"
        m3 = re.search(r"(?:Prob\.|Probabilidad)\s*%\s*No\s*(?:Yes|Sí)\s*(\d+)\s*(\d+)", merged)
        if m3:
            self.data["forebet_btts_yes_pct"] = int(m3.group(2))

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
