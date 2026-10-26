"""Regression tests for the scraper, the gate and the parallel runner.

Run:  .venv/bin/python scripts/test_regressions.py

No network access: every fixture here is a synthetic page or a stubbed fetch.
The bugs these lock down each shipped once and silently corrupted thousands of
DB rows, so they are worth pinning rather than re-discovering.
"""
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bs4 import BeautifulSoup  # noqa: E402

from forebet_scraper import ForebetScraper  # noqa: E402


# ── Fixtures ────────────────────────────────────────────────────────────────

def _market(match_id: int, code, prices, prominent: float) -> str:
    """Build one Forebet .rcnt market block.

    ``prominent`` is the .lscrsp cell: the price of the outcome Forebet
    *predicts*. It deliberately repeats one of ``prices`` on real pages, which
    is what made the old text-order parser emit 1.75/1.75/3.30.
    """
    args = f"this,{match_id}" + (f",'{code}'" if code is not None else "")
    spans = "".join(f"<span>{p}</span>" for p in prices)
    return (
        '<div class="rcnt">'
        f'  <div class="prmod">'
        f'    <span class="lscrsp" onclick="getHodd({args})">{prominent}</span>'
        f'    <div class="haodd">{spans}</div>'
        "  </div>"
        "</div>"
    )


# Forebet predicts the home win ('1'), so the prominent price duplicates home.
HOME_WIN_PAGE = (
    "<html><body>"
    '<div class="rcnt">50 25 25 1 2-1 18°</div>'
    + _market(12345, None, ["1.75", "3.30", "5.10"], 1.75)      # 1X2
    + _market(12345, "uo", ["1.60", "2.20"], 2.20)              # under, over
    + _market(12345, "gg", ["1.90", "1.85"], 1.90)              # yes, no
    + _market(12345, "lp", ["1.75", "3.30", "5.10"], 1.75)      # not a market
    + "</body></html>"
)

# Forebet predicts the away win ('2'): the prominent price is the *last* slot.
AWAY_WIN_PAGE = (
    "<html><body>"
    '<div class="rcnt">30 28 42 2 1-0 21°</div>'
    + _market(999, None, ["4.20", "3.40", "1.95"], 1.95)
    + _market(999, "uo", ["2.10", "1.70"], 1.70)
    + _market(999, "gg", ["2.00", "1.80"], 2.00)
    + "</body></html>"
)


def _parse_odds_only(html: str) -> dict:
    s = ForebetScraper.__new__(ForebetScraper)
    s.data = {k: None for k in (
        "odds_home", "odds_draw", "odds_away", "odds_under25", "odds_over25",
        "odds_btts_yes", "odds_btts_no", "forebet_home_pct", "forebet_draw_pct",
        "forebet_away_pct", "forebet_pred")}
    s.soup = BeautifulSoup(html, "html.parser")
    s._parse_odds()
    return s.data


class TestOddsParsing(unittest.TestCase):
    """The 9,393-row duplicate-price and 5,666-row favourite-mismatch bugs."""

    def test_prices_stay_in_market_order(self):
        d = _parse_odds_only(HOME_WIN_PAGE)
        self.assertEqual([d["odds_home"], d["odds_draw"], d["odds_away"]],
                         [1.75, 3.30, 5.10])
        self.assertEqual([d["odds_under25"], d["odds_over25"]], [1.60, 2.20])
        self.assertEqual([d["odds_btts_yes"], d["odds_btts_no"]], [1.90, 1.85])

    def test_no_duplicate_prices_on_home_forecast(self):
        """1.75/1.75/3.30 was the signature of the old parser."""
        d = _parse_odds_only(HOME_WIN_PAGE)
        three = [d["odds_home"], d["odds_draw"], d["odds_away"]]
        self.assertEqual(len(set(three)), 3, f"duplicate 1X2 price: {three}")

    def test_away_forecast_does_not_shift_prices(self):
        """A prominent price at slot 3 must not move it into slot 1."""
        d = _parse_odds_only(AWAY_WIN_PAGE)
        self.assertEqual([d["odds_home"], d["odds_draw"], d["odds_away"]],
                         [4.20, 3.40, 1.95])
        self.assertEqual([d["odds_under25"], d["odds_over25"]], [2.10, 1.70])

    def test_last_price_row_is_not_priced(self):
        """The 'lp' row repeats 1X2 and would overwrite it."""
        d = _parse_odds_only(HOME_WIN_PAGE)
        self.assertEqual(d["odds_home"], 1.75)

    def test_forebet_prediction_still_read(self):
        d = _parse_odds_only(HOME_WIN_PAGE)
        self.assertEqual(d["forebet_pred"], "1")
        self.assertEqual([d["forebet_home_pct"], d["forebet_draw_pct"],
                          d["forebet_away_pct"]], [50, 25, 25])

    def test_valid_page_passes_odds_guard(self):
        import odds_guard
        d = _parse_odds_only(HOME_WIN_PAGE)
        verdict = odds_guard.validate_1x2(d["odds_home"], d["odds_draw"],
                                           d["odds_away"],
                                           d["forebet_home_pct"],
                                           d["forebet_draw_pct"],
                                           d["forebet_away_pct"])
        self.assertTrue(verdict.ok, msg=verdict.reason)
        self.assertEqual(verdict.reason, "ok")

    def test_scrambled_page_is_rejected_not_guessed(self):
        """A page whose prices cannot be trusted must be rejected outright —
        the guard must never 'repair' one, since the bad slot is unknowable."""
        import odds_guard
        d = _parse_odds_only(HOME_WIN_PAGE)
        d["odds_draw"] = d["odds_home"]  # the historic duplicate bug
        verdict = odds_guard.validate_1x2(d["odds_home"], d["odds_draw"],
                                           d["odds_away"],
                                           d["forebet_home_pct"],
                                           d["forebet_draw_pct"],
                                           d["forebet_away_pct"])
        self.assertFalse(verdict.ok)
        self.assertEqual(verdict.reason, "duplicate")
        # The rejection must not silently substitute a "corrected" price.
        self.assertEqual(d["odds_draw"], d["odds_home"])


class TestPrefetchOrdering(unittest.TestCase):
    """workers>1 overlaps fetches; the consumer must still see file order."""

    def _run(self, workers, n=12):
        import predict as P
        seen = []

        def fake(url):
            # Reverse the durations so completion order != input order.
            import time
            time.sleep((n - int(url)) * 0.01)
            return {"home_team": f"H{url}", "away_team": f"A{url}"}

        urls = [str(i) for i in range(n)]
        out = []
        orig = P.scrape_url
        P.scrape_url = fake
        try:
            for u, d in P._prefetch_scrape(urls, workers):
                seen.append(u)
                out.append(d)
        finally:
            P.scrape_url = orig
        return seen, out

    def test_serial_preserves_order(self):
        seen, out = self._run(1)
        self.assertEqual(seen, [str(i) for i in range(12)])
        self.assertEqual(len(out), 12)

    def test_parallel_preserves_order(self):
        seen, _ = self._run(6)
        self.assertEqual(seen, [str(i) for i in range(12)],
                         "parallel prefetch reordered the slate")

    def test_worker_exception_does_not_kill_run(self):
        import predict as P
        orig = P.scrape_url

        def fake(url):
            if url == "3":
                raise RuntimeError("boom")
            return {"home_team": f"H{url}", "away_team": f"A{url}"}

        got = []
        P.scrape_url = fake
        try:
            for u, d in P._prefetch_scrape([str(i) for i in range(6)], 4):
                got.append((u, bool(d.get("_scrape_error"))))
        finally:
            P.scrape_url = orig
        self.assertEqual(len(got), 6, "a single bad match dropped the rest")
        self.assertEqual(dict(got)["3"], True)

    def test_all_urls_yielded_once(self):
        urls = [str(i) for i in range(37)]
        seen, _ = self._run(4, n=37)
        self.assertEqual(seen, urls)


class TestAnalysisContract(unittest.TestCase):
    """--analysis pipeline must be a drop-in for analyze_from_data."""

    CONTRACT = ("pick", "confidence", "score_lean", "_poisson_probs",
                "_ml", "_method", "market", "_kelly_stake")

    def test_pipeline_exposes_every_field_the_loop_reads(self):
        import pipeline as PL
        import inspect
        src = inspect.getsource(PL.run_pipeline)
        # Cheap structural check: the docstring's compatibility claim is the
        # one thing the pr2 merge rests on.
        self.assertIn("compatible with analyze_from_data", src)

    def test_both_backends_are_importable(self):
        import predict as P
        import pipeline as PL
        self.assertTrue(callable(P.analyze_from_data))
        self.assertTrue(callable(PL.run_pipeline))


class TestMarketAwareScoring(unittest.TestCase):
    """A pick must be scored against the scoreline its market is decided by.

    The ledger used to group by selection label alone and score everything on
    full-time goals. A first-half Draw was therefore judged against the
    full-time Draw base rate (24.1%) instead of its own (39.9%), and because
    ``Draw`` was well sampled overall the first-half buckets inherited that
    volume: 91 first-half picks published as RELIABLE on evidence that never
    described them, and ``1X2|Draw|Medium-High`` reached n=320 instead of the
    true n=90 — which carried it over the 200-sample floor it fails on its own.
    """

    def _row(self, **kw):
        base = {"our_market": "1X2", "our_prediction": "Draw",
                "hg": 1, "ag": 1, "ht_home_goals": None, "ht_away_goals": None}
        base.update(kw)
        return base

    def test_first_half_is_decided_by_half_time_goals(self):
        from scripts.audit_reliability import scoreline_for, HT, FT
        # Level at half time, but home ran away in the second half.
        r = self._row(our_market="1H", hg=4, ag=0,
                      ht_home_goals=0, ht_away_goals=0)
        self.assertEqual(period_of := scoreline_for(r, "1H"), (0, 0))
        self.assertEqual(scoreline_for(r, "1X2"), (4, 0))

    def test_half_time_pick_without_ht_score_is_unscoreable(self):
        """No HT columns must return None — never silently substitute FT goals."""
        from scripts.audit_reliability import resolve_selection, scoreline_for
        r = self._row(our_market="1H")          # hg=ag=1, no HT scores
        hg, ag = scoreline_for(r, "1H")
        self.assertIsNone(hg)
        self.assertIsNone(ag)
        self.assertIsNone(resolve_selection("Draw", hg, ag))

    def test_first_half_draw_scores_on_ht_not_ft(self):
        from scripts.audit_reliability import resolve_selection
        # Level HT, home wins FT: a first-half Draw wins, a full-time Draw loses.
        self.assertTrue(resolve_selection("Draw", 0, 0))
        self.assertFalse(resolve_selection("Draw", 3, 1))

    def test_bucket_key_carries_the_market(self):
        from scripts.audit_reliability import bucket_key
        self.assertNotEqual(bucket_key("1X2", "Draw", "Low"),
                            bucket_key("1H", "Draw", "Low"))
        self.assertEqual(bucket_key("1H", "Draw", "Low"), "1H|Draw|Low")

    def test_unknown_market_normalises(self):
        from scripts.audit_reliability import market_of, MARKET_UNKNOWN
        self.assertEqual(market_of({"our_market": None}), MARKET_UNKNOWN)
        self.assertEqual(market_of({"our_market": ""}), MARKET_UNKNOWN)
        self.assertEqual(market_of({"our_market": "1X2"}), "1X2")

    def test_report_does_not_share_a_first_half_bucket_with_full_time(self):
        import reliable_report as R
        ledger = {"gate": {
            "1H|Draw|Medium": {"n": 900, "hits": 700, "hit_rate": 0.778,
                               "expected": 0.399, "shrunk_rate": 0.7,
                               "edge_pts": 30.0, "publish": True,
                               "market": "1H", "reason": "ok"},
            "1X2|Draw|Medium": {"n": 5, "hits": 1, "hit_rate": 0.2,
                                "expected": 0.241, "shrunk_rate": 0.24,
                                "edge_pts": -0.1, "publish": False,
                                "market": "1X2", "reason": "too thin"},
        }}
        ht = R.classify_pick("Draw", "Medium", ledger, "1H")
        ft = R.classify_pick("Draw", "Medium", ledger, "1X2")
        self.assertEqual(ht["tier"], R.TIER_RELIABLE)
        self.assertEqual(ft["tier"], R.TIER_HELD)

    def test_first_half_base_rate_is_its_own_market(self):
        """1H and 1X2 Draw base rates differ by ~15pts and must not be shared."""
        import sqlite3, tempfile, os
        import scripts.audit_reliability as A
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            conn = sqlite3.connect(path)
            conn.row_factory = sqlite3.Row
            conn.execute("""CREATE TABLE matches (
                id INTEGER PRIMARY KEY, created_at TEXT, forebet_url TEXT,
                home_team TEXT, away_team TEXT, match_date TEXT,
                actual_home_goals INT, actual_away_goals INT,
                ht_home_goals INT, ht_away_goals INT)""")
            # 100 matches: level at HT in all of them, drawn at FT in 70 of
            # them. So the first-half Draw rate is 1.00 and the full-time one
            # 0.70 -- a 30pt gap the old shared base rate could not represent.
            # Unique 6-digit Forebet ids: dedupe_fixtures() keys on them, and
            # identical teams+date would otherwise collapse all 100 rows.
            rows = [(i, "2026-01-01", f"http://forebet/10000{i}", f"H{i}", "A",
                     "2026-01-01",
                     1, 0 if i < 30 else 1,
                     0, 0)
                    for i in range(100)]
            conn.executemany(
                "INSERT INTO matches VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
            conn.commit()
            br = A.compute_base_rates(conn)
            self.assertAlmostEqual(br["rates"][A.base_key("1H", "Draw")]["rate"],
                                   1.0, places=6)
            self.assertAlmostEqual(br["rates"][A.base_key("1X2", "Draw")]["rate"],
                                   0.70, places=6)
            self.assertEqual(br["self_check"], "ok")
            conn.close()
        finally:
            os.unlink(path)


class TestOddsStamping(unittest.TestCase):
    """``--no-stamp`` must still report odds integrity.

    stamp_odds() both classifies and writes.  audit_reliability.py offered
    --no-stamp to avoid touching the DB, but it skipped the classification too,
    so the ledger shipped with ``odds_integrity: {}`` and the report's
    "why most expected values are absent" note never rendered — the one
    sentence on the page explaining why 88% of prices are unusable was
    suppressed by the flag meant to protect the database.
    """

    @staticmethod
    def _connect(path):
        import sqlite3
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        conn.execute("""CREATE TABLE matches (
            id INTEGER PRIMARY KEY, odds_home REAL, odds_draw REAL,
            odds_away REAL, forebet_home_pct REAL, forebet_draw_pct REAL,
            forebet_away_pct REAL)""")
        conn.executemany(
            "INSERT INTO matches VALUES (?,?,?,?,?,?,?)", [
                # duplicate: the legacy parser's signature defect
                (1, 1.75, 1.75, 3.30, 30.0, 47.0, 23.0),
                # clean triplet, favourite agrees with Forebet's top %
                (2, 2.10, 3.30, 3.60, 45.0, 28.0, 27.0),
            ])
        conn.commit()
        return conn

    def test_scan_odds_classifies_without_writing(self):
        import os, tempfile
        import scripts.audit_reliability as A
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            conn = self._connect(path)
            counts, updates = A.scan_odds(conn)
            self.assertEqual(counts, {"duplicate": 1, "ok": 1})
            self.assertEqual(len(updates), 2)
            # Read-only: no odds_status column, no row rewritten.
            cols = {r[1] for r in conn.execute("PRAGMA table_info(matches)")}
            self.assertNotIn("odds_status", cols)
            conn.close()
        finally:
            os.unlink(path)

    def test_stamp_odds_writes_what_scan_reported(self):
        import os, tempfile
        import scripts.audit_reliability as A
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            conn = self._connect(path)
            counts = A.stamp_odds(conn)
            self.assertEqual(counts, {"duplicate": 1, "ok": 1})
            cols = {r[1] for r in conn.execute("PRAGMA table_info(matches)")}
            self.assertIn("odds_status", cols)
            stamped = {r[0] for r in
                       conn.execute("SELECT odds_status FROM matches")}
            self.assertEqual(stamped, {"duplicate", "ok"})
            conn.close()
        finally:
            os.unlink(path)


class TestDrawNoBet(unittest.TestCase):
    """A draw pushes the stake back: it is neither a hit nor a miss.

    DNB rows used to be scored by resolve_selection(), which did not know the
    labels "Home"/"Away" and returned None, so all 764 resolved draw-no-bet
    picks were silently dropped from the ledger and shown with no measured
    reliability at all.
    """

    def test_labels_are_recognised(self):
        from scripts.audit_reliability import resolve_selection
        self.assertTrue(resolve_selection("Home", 2, 1))
        self.assertFalse(resolve_selection("Home", 1, 2))
        self.assertTrue(resolve_selection("Away", 1, 2))

    def test_draw_is_a_push_not_a_miss(self):
        from scripts.audit_reliability import resolve_selection
        self.assertIsNone(resolve_selection("Home", 1, 1))
        self.assertIsNone(resolve_selection("Away", 1, 1))

    def test_pushes_leave_the_base_rate_denominator(self):
        import sqlite3, tempfile, os
        import scripts.audit_reliability as A
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            conn = sqlite3.connect(path)
            conn.row_factory = sqlite3.Row
            conn.execute("""CREATE TABLE matches (
                id INTEGER PRIMARY KEY, created_at TEXT, forebet_url TEXT,
                home_team TEXT, away_team TEXT, match_date TEXT,
                actual_home_goals INT, actual_away_goals INT,
                ht_home_goals INT, ht_away_goals INT)""")
            # 6 home wins, 2 draws, 2 away wins over 10 matches. The
            # draws are pushes, so the DNB base rates are measured on the 8
            # decisive matches: 6/8 and 2/8. Counting the draws would give
            # 6/10 and understate both.
            scores = [(2, 0), (3, 1), (1, 0), (4, 2), (2, 1), (1, 0),
                      (0, 0), (1, 1),
                      (1, 2), (1, 9)]
            conn.executemany(
                "INSERT INTO matches VALUES (?,?,?,?,?,?,?,?,?,?)",
                [(i, "2026-01-01", f"http://forebet/20000{i}", f"H{i}", "A",
                  "2026-01-01", s, a, None, None)
                 for i, (s, a) in enumerate(scores)])
            conn.commit()
            br = A.compute_base_rates(conn)
            self.assertAlmostEqual(br["rates"][A.base_key("DNB", "Home")]["rate"],
                                   6 / 8, places=6)
            self.assertAlmostEqual(br["rates"][A.base_key("DNB", "Away")]["rate"],
                                   2 / 8, places=6)
            # The full-time population still counts the draws.
            self.assertEqual(br["population_n"], 10)
            conn.close()
        finally:
            os.unlink(path)

    def test_dnb_is_not_treated_as_a_complement_pair(self):
        """Home + Away must sum to less than 1.0, by exactly the draw rate."""
        import scripts.audit_reliability as A
        self.assertNotIn("Home", A.COMPLEMENTS)
        self.assertNotIn("Away", A.COMPLEMENTS)


class TestOverFourFiveMeasured(unittest.TestCase):
    """Over 4.5 was missing from SELECTIONS: 554 resolved picks went ungated."""

    def test_line_is_resolvable(self):
        from scripts.audit_reliability import resolve_selection
        self.assertTrue(resolve_selection("Over 4.5", 3, 2))
        self.assertFalse(resolve_selection("Over 4.5", 2, 2))
        self.assertTrue(resolve_selection("Under 4.5", 2, 2))
        self.assertTrue(resolve_selection("Over 5.5", 4, 2))
        self.assertFalse(resolve_selection("Over 5.5", 3, 2))

    def test_line_is_in_the_selection_table(self):
        import scripts.audit_reliability as A
        for sel in ("Over 4.5", "Under 4.5", "Over 5.5", "Under 5.5"):
            self.assertIn(sel, A.SELECTIONS)
            partner = A.COMPLEMENTS[sel]
            self.assertNotEqual(partner, sel)
            self.assertEqual(A.COMPLEMENTS[partner], sel)  # involution


class TestReserveFilter(unittest.TestCase):
    """B-teams must never reach RELIABLE.

    The reserve regex lived only in scripts/bettor.py, so predictions/picks.html
    had no reserve filtering at all and published picks on Betis III,
    UD Almeria B, Pontevedra II and Puerto Cabello II — the fixtures where the
    model's senior-team inputs describe the wrong players.
    """

    def test_reserve_sides_are_detected(self):
        from reserve_filter import is_reserve_side
        for team in ("Betis III", "UD Almería B", "Pontevedra II",
                     "Puerto Cabello II", "Real Madrid Castilla",
                     "Atlético Madrid B", "Barcelona U19", "FC B",
                     "Real Madrid IV", "Villarreal B",
                     "Boca Juniors Res.",  # Argentine "Res." abbreviation
                     "River Plate Res.", "Colón Res."):
            self.assertTrue(is_reserve_side(team), f"{team} not flagged")

    def test_senior_clubs_are_not_false_positives(self):
        """Every one of these was flagged by an earlier, sloppier pattern."""
        from reserve_filter import is_reserve_side
        for team in ("Willem II",        # Roman numeral in the club name
                     "FCSB",             # ends in B, is not a B-team
                     "Kilb",             # scrape artefact ending in "b"
                     "NK Ivancica Iv.",  # "Iv." = Ivan, Croatian abbreviation
                     "Tekstilshchik Iv.",
                     "AC Milan", "Inter", "Deportivo Alavés", "KuPS",
                     "Algeciras", "Rayo Majadahonda"):
            self.assertFalse(is_reserve_side(team), f"{team} wrongly flagged")

    def test_eastern_european_second_teams_are_detected(self):
        """These carry no Roman numeral or "B" — the marker is a "-2" suffix.

        Rubin-2 Kazan, Rodina-2 Moscow and Baltika-2 all reached the report as
        RELIABLE picks because "2" alone was not a token. A bare \\b2\\b would
        fix them and wreck senior clubs, so the hyphen does the work.
        """
        from reserve_filter import is_reserve_side
        for team in ("Rubin-2 Kazan", "Rodina-2 Moscow", "Baltika-2",
                     "Zenit-2", "Dinamo-2 Tbilisi", "Spartak-2 Moscow",
                     "FK Minsk-2", "Urartu-2", "Shirak-2"):
            self.assertTrue(is_reserve_side(team), f"{team} not flagged")

    def test_hyphenated_founding_years_are_not_second_teams(self):
        """The flip side of the rule above: a 2-digit year is not a squad.

        Matching one *or two* digits flagged all three of these senior clubs.
        Restricting the suffix to a single digit is what separates a numbered
        second team from a club named after its founding year.
        """
        from reserve_filter import is_reserve_side
        for team in ("SC Weiche-08",    # now SV Weiche Braunschweig
                     "PK-35 Vantaa",    # Finnish Veikkausliiga side
                     "TP-47 Tornio",    # Finnish league club
                     "TSV 1860 Munchen", "1860 Munich"):
            self.assertFalse(is_reserve_side(team), f"{team} wrongly flagged")

    def test_fixture_is_reserve_if_either_side_is(self):
        from reserve_filter import is_reserve_fixture
        self.assertTrue(is_reserve_fixture("Real Betis", "Betis III"))
        self.assertTrue(is_reserve_fixture("Betis III", "Real Betis"))
        self.assertFalse(is_reserve_fixture("Algeciras", "Rayo Majadahonda"))

    def test_reserve_reason_names_the_side(self):
        from reserve_filter import reserve_reason
        self.assertEqual(reserve_reason("Real Betis", "Betis III"),
                         ("Betis III", "away"))
        self.assertEqual(reserve_reason("Betis III", "Sevilla"), ("Betis III", "home"))
        self.assertIsNone(reserve_reason("Algeciras", "Rayo Majadahonda"))

    def test_report_holds_reserve_picks_even_when_gate_publishes(self):
        import reliable_report as R
        ledger = {"policy": {"min_samples": 200}, "gate": {
            "1X2|Away win|Medium-High": {
                "n": 1500, "hits": 800, "hit_rate": 0.53, "expected": 0.33,
                "shrunk_rate": 0.51, "edge_pts": 18.0, "publish": True,
                "market": "1X2", "reason": "ok"},
        }}
        pick = {"pick": "Away win", "market": "1X2", "confidence": "Medium-High",
                "home": "Real Betis", "away": "Betis III", "_exp_goals": (1.0, 1.6)}
        row = R.build_rows([pick], ledger)[0]
        self.assertEqual(row["tier"], R.TIER_HELD)
        self.assertIn("reserve", row["verdict"]["reason"].lower())
        # The same pick on a senior fixture is untouched by the reserve rule.
        senior = dict(pick, home="Algeciras", away="Rayo Majadahonda")
        self.assertEqual(R.build_rows([senior], ledger)[0]["tier"],
                         R.TIER_RELIABLE)


class TestReportDedupe(unittest.TestCase):
    """One fixture must print once.

    Real Zaragoza v CD Teruel appeared twice as identical RELIABLE rows because
    its URL was stored both percent-encoded and raw.
    """

    def test_identical_fixture_collapses(self):
        import reliable_report as R
        base = {"pick": "1X", "market": "DC", "confidence": "Medium-High",
                "home": "Real Zaragoza", "away": "CD Teruel",
                "date": "2026-10-04 21:00", "_p_1x": 0.9}
        dup_a = dict(base, forebet_url="x-2443574")
        dup_b = dict(base, forebet_url="x-2443574%28vr%29")
        self.assertEqual(len(R.build_rows([dup_a, dup_b], None)), 1)

    def test_distinct_fixtures_survive(self):
        import reliable_report as R
        a = {"pick": "1X", "market": "DC", "confidence": "Medium",
             "home": "Real Zaragoza", "away": "CD Teruel",
             "date": "2026-10-04 21:00"}
        b = dict(a, away="CD Malagon")
        c = dict(a, date="2026-10-05 21:00")
        self.assertEqual(len(R.build_rows([a, b, c], None)), 3)


class TestGateMath(unittest.TestCase):
    """wilson_lower_bound(hits, n) — note the argument order is counts, not rate."""

    def test_wilson_lower_is_below_point_estimate(self):
        from scripts.audit_reliability import wilson_lower_bound, Z_ONE_SIDED_95
        self.assertLess(wilson_lower_bound(90, 100), 0.90)
        self.assertGreaterEqual(wilson_lower_bound(90, 100), 0.0)
        self.assertAlmostEqual(Z_ONE_SIDED_95, 1.645, places=3)

    def test_same_rate_more_samples_widens_the_floor(self):
        from scripts.audit_reliability import wilson_lower_bound
        thin = wilson_lower_bound(11, 12)    # 91.7% on n=12
        thick = wilson_lower_bound(366, 400)  # 91.5% on n=400
        self.assertGreater(thick, thin)
        self.assertGreater(thin, 0.5)
        self.assertGreater(thick, 0.88)

    def test_wilson_bounded(self):
        from scripts.audit_reliability import wilson_lower_bound
        self.assertGreaterEqual(wilson_lower_bound(50, 50), 0.0)
        self.assertLessEqual(wilson_lower_bound(0, 50), 1.0)
        self.assertEqual(wilson_lower_bound(5, 0), 0.0)

    # ── The gate asks a relative question, not a fixed one ─────────────────

    @staticmethod
    def _bucket(key, n, hit_rate, expected, market=None, selection=None,
                confidence="Medium", edge_pts=None):
        """A measured bucket, with every field build_gate() reads."""
        hits = int(round(n * hit_rate))
        shrunk = (hits + 120 * expected) / (n + 120)   # K = 120
        market = market or key.split("|")[0]
        selection = selection or key.split("|")[1]
        return {key: {"market": market, "period": "FT",
                      "selection": selection, "confidence": confidence,
                      "n": n, "hits": hits,
                      "hit_rate": round(hits / n, 4),
                      "expected": expected,
                      "shrunk_rate": round(shrunk, 4),
                      "edge_pts": round((shrunk - expected) * 100, 2),
                      "priced_n": 0}}

    def test_no_absolute_win_rate_constant(self):
        """A 40% pick that beats a 33.8% base by a clear margin must publish.

        The gate used to require every published pick to win more than half the
        time (Wilson lower bound >= 0.55), which no goal-line market can ever
        satisfy at a sane base rate. Judged against its own base rate, this is
        a real, statistically solid edge — and it is exactly the shape of
        "Over 3.5", which the fixed floor kept off the page.
        """
        from scripts.audit_reliability import build_gate, wilson_lower_bound
        b = self._bucket("O/U|Over 3.5|Medium", n=600, hit_rate=0.40,
                         expected=0.338)
        self.assertLess(b["O/U|Over 3.5|Medium"]["hit_rate"], 0.50)
        gate = build_gate({"buckets": b}, min_samples=200, min_edge_pts=2.0)
        e = gate["O/U|Over 3.5|Medium"]
        self.assertGreaterEqual(
            (wilson_lower_bound(e["hits"], e["n"]) - e["expected"]) * 100, 2.0)
        self.assertTrue(e["publish"], e["reason"])
        # ...and it still publishes with the optional absolute floor switched
        # off explicitly, because that is the default.
        gate2 = build_gate({"buckets": b}, min_samples=200, min_edge_pts=2.0,
                           min_hit=0.0)
        self.assertTrue(gate2["O/U|Over 3.5|Medium"]["publish"])

    def test_sliver_edge_over_base_is_held(self):
        """Clearing base by a sliver is not significance, whatever the hit rate.

        These are the real numbers behind O/U Under 1.5 Medium: 27.5% realised
        against a 23.1% base (n=338), shrunk edge +3.2pts, but the 95% lower
        bound stops only 0.6pt clear of base.  Walk-forward validation showed
        it published when the bar was merely "beats base at all" and then lost
        its edge out-of-sample, so the margin — not a fixed win rate — is what
        holds it.
        """
        from scripts.audit_reliability import build_gate
        b = self._bucket("O/U|Under 1.5|Medium", n=338, hit_rate=0.2752,
                         expected=0.2314)
        self.assertGreaterEqual(
            b["O/U|Under 1.5|Medium"]["edge_pts"], 2.0)   # the edge is there
        gate = build_gate({"buckets": b}, min_samples=200, min_edge_pts=2.0)
        e = gate["O/U|Under 1.5|Medium"]
        self.assertFalse(e["publish"])
        self.assertIn("not significant", e["reason"])
        self.assertIn("need 2", e["reason"])

    def test_optional_absolute_floor_still_bites(self):
        """--min-hit remains available as an opt-in absolute floor."""
        from scripts.audit_reliability import build_gate
        b = self._bucket("O/U|Over 3.5|Medium", n=600, hit_rate=0.40,
                         expected=0.338)
        gate = build_gate({"buckets": b}, min_samples=200, min_edge_pts=2.0,
                          min_hit=0.55)
        e = gate["O/U|Over 3.5|Medium"]
        self.assertFalse(e["publish"])
        self.assertIn("absolute floor", e["reason"])


class TestReportRobustness(unittest.TestCase):
    """A pick the report cannot price must degrade, not take the run down.

    A 1X pick with no ``_p_1x`` used to reach _from_poisson with a None
    predicate and raise TypeError, which _write_html swallowed — leaving the
    run reporting "Reliable picks report skipped" with no picks.html at all.
    """

    def test_double_chance_has_a_predicate(self):
        import reliable_report as R
        for pick in ("1X", "X2", "12"):
            self.assertIsNotNone(R._predicate_for(pick), f"{pick} unpriceable")

    def test_double_chance_probabilities(self):
        import reliable_report as R
        res = {"_exp_goals": (1.2, 1.0)}   # home the stronger side
        x2 = R._match_prob(res, "X2")
        self.assertIsNotNone(x2)
        # X2 is "draw or away" = "not a home win", so it clears 50% whenever
        # home is the favourite.
        self.assertGreater(x2, 0.5)
        for pick in ("1X", "X2", "12"):
            p = R._match_prob(res, pick)
            self.assertIsNotNone(p, f"{pick} unpriceable")
            self.assertTrue(0.0 < p < 1.0, f"{pick} = {p} out of range")
        # They overlap rather than sum to 1: 1X and 12 both include a home win.
        self.assertGreater(R._match_prob(res, "1X") + R._match_prob(res, "12"), 1.0)
        # Draw is not derivable from an expected-goals pair — it needs the
        # Poisson triple — so the report must say so rather than guess.
        self.assertIsNone(R._match_prob(res, "Draw"))
        triple = {"_poisson_probs": (0.5, 0.25, 0.25)}
        self.assertAlmostEqual(R._match_prob(triple, "Draw"), 0.25, places=6)

    def test_unknown_pick_returns_none(self):
        import reliable_report as R
        self.assertIsNone(R._match_prob({"_exp_goals": (1.2, 1.0)},
                                        "Corners Over 9.5"))

    def test_poisson_guard_rejects_none_predicate(self):
        import reliable_report as R
        self.assertIsNone(R._from_poisson({"_exp_goals": (1.2, 1.0)}, None))

    def test_build_rows_survives_a_mixed_slate(self):
        import reliable_report as R
        results = [
            {"pick": "Under 2.5", "confidence": "Medium", "_exp_goals": (1.2, 1.0)},
            {"pick": "X2", "confidence": "High", "_exp_goals": (0.8, 1.4)},
            {"pick": "1X", "confidence": "Low"},
            {"pick": "Something Unmapped", "confidence": "Low"},
        ]
        rows = R.build_rows(results, R.load_ledger())
        self.assertEqual(len(rows), 4)

    def test_stale_ledger_without_totals_shrunk_rate_degrades(self):
        """A ledger whose totals lack shrunk_rate must not abort the report.

        audit_reliability.py writes it today, but a ledger produced before that
        fix is still on disk in some clones. classify_pick() used to KeyError on
        the totals fallback, which an exception handler turned into "Reliable
        picks report skipped" — every pick of a 722-match run vanished from
        picks.html even though the buckets themselves were fine.
        """
        import reliable_report as R
        # The crashing ledger was market-aware (gate populated) but its totals
        # predated the shrunk_rate fix. An empty gate would flip the ledger to
        # market-unaware lookups and dodge the exact code path that blew up.
        ledger = {"policy": {"min_samples": 200},
                  "gate": {"1X2|Home win|High": {"n": 182, "hits": 115,
                                                 "hit_rate": 0.632, "expected": 0.431,
                                                 "shrunk_rate": 0.552, "edge_pts": 12.1,
                                                 "publish": False, "market": "1X2",
                                                 "reason": "only 182 samples"}},
                  "totals": {
            # Old shape: measured numbers but no shrunk_rate.
            "O/U|Over 2.5": {"market": "O/U", "selection": "Over 2.5",
                             "n": 84, "hit_rate": 0.631, "expected": 0.549,
                             "edge_pts": 3.4},
        }}
        pick = {"pick": "Over 2.5", "market": "O/U", "confidence": "Medium-High",
                "home": "Some Home", "away": "Some Away"}
        row = R.build_rows([pick], ledger)[0]
        self.assertEqual(row["tier"], R.TIER_HELD)
        self.assertIsNone(row["verdict"]["shrunk_rate"])
        self.assertEqual(row["verdict"]["n"], 84)


class TestBestBet(unittest.TestCase):
    """picks.html must be able to recommend a market that is not 1/X/2.

    A fixture whose value sits in BTTS or on the other side of a goal line used
    to be invisible: the report read only the first six entries of the synthesis
    shortlist, printed the primary pick, and had no notion of a price at all.
    A best bet is only worth showing if three things hold — it is ranked across
    every market the fixture offers, it never quotes a price ``odds_guard``
    rejected, and it still says something when nothing prices.
    """

    @staticmethod
    def _ledger():
        return {
            "policy": {"min_samples": 200, "credibility_k": 120,
                       "min_edge_pts": 2.0},
            "gate": {
                "1X2|Draw|Medium": {"n": 640, "hits": 164, "hit_rate": 0.256,
                                    "expected": 0.2426, "shrunk_rate": 0.254,
                                    "edge_pts": 1.15, "publish": False,
                                    "market": "1X2", "reason": "edge too small"},
                "BTTS|No|Medium": {"n": 900, "hits": 620, "hit_rate": 0.689,
                                   "expected": 0.4567, "shrunk_rate": 0.655,
                                   "edge_pts": 19.8, "publish": True,
                                   "market": "BTTS", "reason": "ok"},
                "O/U|Over 2.5|Low": {"n": 400, "hits": 150, "hit_rate": 0.375,
                                     "expected": 0.5479, "shrunk_rate": 0.417,
                                     "edge_pts": -13.1, "publish": False,
                                     "market": "O/U", "reason": "below base"},
            },
            "totals": {},
        }

    @staticmethod
    def _fixture(**over):
        base = {
            "home": "Estudiantes Río Cuarto", "away": "CA Aldosivi",
            "date": "18/10/2026", "market": "1X2", "pick": "Draw",
            "confidence": "Medium",
            "forebet_pct": [30.0, 47.0, 23.0],
            # home == draw: the legacy parser's signature defect.
            "odds_home": 2.87, "odds_draw": 2.87, "odds_away": 2.80,
            "odds_btts_yes": 2.25, "odds_btts_no": 1.57,
            "_p_btts_no": 0.78, "_p_btts_yes": 0.22,
            "_exp_goals": (1.0, 1.6),
            "all_picks": [
                {"market": "1X2", "pick": "Draw", "confidence": "Medium",
                 "model_prob": 0.40},
                {"market": "BTTS", "pick": "No", "confidence": "Medium",
                 "model_prob": 0.78},
            ],
        }
        base.update(over)
        return base

    def test_best_bet_ranks_across_markets_not_just_the_primary(self):
        import reliable_report as R
        row = R.build_rows([self._fixture()], self._ledger())[0]
        self.assertEqual((row["pick"], row["market"]), ("Draw", "1X2"))
        best = row["best"]
        self.assertEqual((best["market"], best["pick"]), ("BTTS", "No"))
        self.assertFalse(best["is_primary"])
        self.assertEqual(best["ranked_by"], "EV")
        self.assertAlmostEqual(best["odds"], 1.57, places=2)
        self.assertGreater(best["ev_pct"], 0.0)

    def test_rejected_price_is_never_given_an_ev(self):
        import reliable_report as R
        res = self._fixture()
        prices, verdict = R._price_group(res, "1X2", "Draw")
        self.assertIsNotNone(verdict)
        self.assertFalse(verdict.ok)
        self.assertEqual(verdict.reason, "duplicate")
        self.assertEqual(len(prices), 3)

        # With BTTS priced the EV ranking still works and simply never selects
        # a 1X2 leg; with BTTS also stripped, the fixture falls back to ranking
        # by model lift and says the prices were rejected rather than pricing
        # them.
        only_1x2 = self._fixture(odds_btts_yes=None, odds_btts_no=None,
                                 all_picks=[{"market": "1X2", "pick": "Draw",
                                             "confidence": "Medium",
                                             "model_prob": 0.40}])
        best = R.best_candidate(only_1x2, self._ledger(),
                                primary=("1X2", "Draw"))
        self.assertIsNone(best["ev_pct"])
        self.assertIsNone(best["odds"])
        self.assertEqual(best["ranked_by"], "model lift")
        # Either the candidate's own rejection note or the fixture-wide one —
        # what matters is that the price is reported as unusable, not quoted.
        self.assertTrue("validation" in best["note"]
                        or "odds_guard" in best["note"], best["note"])

    def test_missing_side_of_a_two_way_market_is_offered(self):
        import reliable_report as R
        res = self._fixture(
            all_picks=[{"market": "BTTS", "pick": "No",
                        "confidence": "Medium", "model_prob": 0.78}])
        found = {(c["market"], c["pick"]): c
                 for c in R._candidate_universe(res)}
        self.assertIn(("BTTS", "Yes"), found)
        self.assertAlmostEqual(found[("BTTS", "Yes")]["model_prob"], 0.22,
                               places=6)
        # A three-way market is never completed by subtraction — only by a
        # probability the model actually produced.
        for key in (("1X2", "Home win"), ("1X2", "Away win")):
            self.assertNotIn(key, found)

    def test_falls_back_to_model_lift_when_nothing_prices(self):
        import reliable_report as R
        res = self._fixture(odds_btts_yes=None, odds_btts_no=None,
                            odds_home=None, odds_draw=None, odds_away=None)
        best = R.best_candidate(res, self._ledger(), primary=("1X2", "Draw"))
        self.assertEqual((best["market"], best["pick"]), ("BTTS", "No"))
        self.assertEqual(best["ranked_by"], "model lift")
        self.assertIsNone(best["odds"])
        self.assertEqual(best["tier"], R.TIER_RELIABLE)

    def test_fallback_ranks_this_match_not_the_luckiest_bucket(self):
        """The bucket edge is one number per bucket — it cannot rank fixtures.

        Give Away win the biggest ledger edge in the set while the model barely
        rates it in *this* match, and give BTTS No a smaller edge it comfortably
        beats on this fixture.  Ranking by bucket edge would recommend Away win
        purely because its bucket is historically juicy; ranking by lift has to
        pick BTTS No.
        """
        import reliable_report as R
        ledger = self._ledger()
        ledger["gate"]["1X2|Away win|Medium"] = {
            "n": 1200, "hits": 617, "hit_rate": 0.514, "expected": 0.3244,
            "shrunk_rate": 0.508, "edge_pts": 26.0, "publish": True,
            "market": "1X2", "reason": "ok"}
        res = self._fixture(
            odds_home=None, odds_draw=None, odds_away=None,
            odds_btts_yes=None, odds_btts_no=None,
            all_picks=[{"market": "BTTS", "pick": "No",
                        "confidence": "Medium", "model_prob": 0.70},
                       {"market": "1X2", "pick": "Away win",
                        "confidence": "Medium", "model_prob": 0.30}])
        best = R.best_candidate(res, ledger, primary=("BTTS", "No"))
        self.assertGreater(ledger["gate"]["1X2|Away win|Medium"]["edge_pts"],
                           ledger["gate"]["BTTS|No|Medium"]["edge_pts"])
        self.assertEqual(best["ranked_by"], "model lift")
        self.assertEqual((best["market"], best["pick"]), ("BTTS", "No"))
        self.assertGreater(best["lift_pts"], 0.0)

    def test_primary_pick_prices_off_the_same_probability_the_row_shows(self):
        import reliable_report as R
        res = self._fixture(
            market="O/U", pick="Over 2.5", confidence="Low",
            _p_over25=0.30,
            all_picks=[{"market": "O/U", "pick": "Over 2.5",
                        "confidence": "Low", "model_prob": 0.44}])
        row = R.build_rows([res], self._ledger())[0]
        best = row["best"]
        self.assertEqual((best["market"], best["pick"]), ("O/U", "Over 2.5"))
        self.assertTrue(best["is_primary"])
        self.assertAlmostEqual(best["stated_prob"],
                               R._match_prob(res, "Over 2.5"), places=6)
        self.assertEqual(best["shown_prob"], row["shown_prob"])

    def test_selection_absent_from_totals_gets_a_real_sample_count(self):
        """"0 matches" for Over 2.5 would be false — thousands are known."""
        import reliable_report as R
        ledger = self._ledger()
        verdict = R.classify_pick("Over 2.5", "", ledger, "O/U")
        self.assertEqual(verdict["n"], 400)
        self.assertGreater(verdict["shrunk_rate"], 0.0)
        self.assertLess(verdict["edge_pts"], 0.0)
        self.assertEqual(verdict["tier"], R.TIER_WATCH)

    # ── tier-first ranking, passes and disclosures ─────────────────────

    def test_tier_outranks_a_fatter_ev(self):
        """EV grows where the model disagrees with the book — the tier doesn't.

        Over 2.5 prices +33% (WATCH: measured, no edge) against BTTS No at
        +22% (RELIABLE: beaten its base rate).  Selecting on EV alone named
        the WATCH leg; the tier decides and EV only ranks inside it.
        """
        import reliable_report as R
        res = self._fixture(
            odds_over25=1.90, odds_under25=1.95,
            all_picks=[
                {"market": "1X2", "pick": "Draw", "confidence": "Medium",
                 "model_prob": 0.40},
                {"market": "BTTS", "pick": "No", "confidence": "Medium",
                 "model_prob": 0.78},
                {"market": "O/U", "pick": "Over 2.5", "confidence": "Low",
                 "model_prob": 0.70},
            ])
        best = R.best_candidate(res, self._ledger(), primary=("1X2", "Draw"))
        self.assertEqual((best["market"], best["pick"]), ("BTTS", "No"))
        self.assertEqual(best["tier"], R.TIER_RELIABLE)
        # the WATCH alternative really did price higher, or this proves nothing
        self.assertLess(best["ev_pct"], (0.70 * 1.90 - 1.0) * 100)

    def test_every_price_negative_prints_a_pass_not_a_best_bet(self):
        """A negative EV under a *Best bet* heading reads as a bet."""
        import html as _html
        import reliable_report as R
        # 0.60 sits in the window where BOTH sides of the BTTS price lose
        # (>0.637 would make No positive, <0.556 would make Yes positive), and
        # the goal-line prices are set just wide enough that both Over and
        # Under (derived from the fixture's Poisson mean, 51.8%) lose too — so
        # every price on the fixture comes back at or below break-even.
        res = self._fixture(
            market="BTTS", pick="No", confidence="Medium",
            _p_btts_no=0.60, _p_btts_yes=0.40,
            odds_over25=1.90, odds_under25=1.90,
            all_picks=[
                {"market": "BTTS", "pick": "No", "confidence": "Medium",
                 "model_prob": 0.60},
                {"market": "O/U", "pick": "Over 2.5", "confidence": "Low",
                 "model_prob": 0.50},
            ])
        best = R.best_candidate(res, self._ledger(), primary=("BTTS", "No"))
        self.assertTrue(best["no_value"])
        self.assertLess(best["ev_pct"], 0.0)
        cell = R._best_cell(best, _html.escape)
        self.assertIn("no +EV selection", cell)
        self.assertIn("pass", cell)
        self.assertIn(best["tier"], cell)   # still says what was available

    def test_held_best_bet_names_the_sample_it_lacks(self):
        """15 matches is not "no history" and not a recommendation either."""
        import html as _html
        import reliable_report as R
        ledger = self._ledger()
        ledger["gate"]["O/U|Over 3.5|Medium"] = {
            "n": 15, "hits": 6, "hit_rate": 0.40, "expected": 0.336,
            "shrunk_rate": 0.392, "edge_pts": 5.6, "publish": False,
            "market": "O/U", "reason": "only 15 samples"}
        res = self._fixture(
            odds_btts_yes=None, odds_btts_no=None,
            odds_over35=3.00, odds_under35=1.40,
            all_picks=[{"market": "O/U", "pick": "Over 3.5",
                        "confidence": "Medium", "model_prob": 0.45}])
        best = R.best_candidate(res, ledger, primary=("1X2", "Draw"))
        self.assertEqual((best["market"], best["pick"]), ("O/U", "Over 3.5"))
        self.assertEqual(best["tier"], R.TIER_HELD)
        cell = R._best_cell(best, _html.escape)
        self.assertIn("no measured evidence (n=15)", cell)

    def test_model_market_gap_beside_the_ev_is_disclosed(self):
        """A wide gap is the model disagreeing with a well-calibrated price.

        78% against a de-vigged market near 59% is a 19pt disagreement; the
        cell prints it so the size of the EV cannot pass for confidence.
        """
        import html as _html
        import reliable_report as R
        best = R.best_candidate(self._fixture(), self._ledger(),
                                primary=("1X2", "Draw"))
        self.assertEqual((best["market"], best["pick"]), ("BTTS", "No"))
        self.assertIsNotNone(best["gap_warning"])
        self.assertGreater(abs(best["price_edge_pct"]),
                           R.MODEL_MARKET_GAP_PTS)
        cell = R._best_cell(best, _html.escape)
        self.assertIn("model error, not edge", cell)


class TestBettorStaking(unittest.TestCase):
    """Stakes must be earned by evidence and must never over-bet the bankroll.

    Two shipped defects: an unmeasured league was paid the same full stake as
    a proven one, and the cap was applied by scaling every stake down to fill
    exactly 100% of the bankroll — so a thin card was always all-in.
    """

    @staticmethod
    def _match(home, prob, odds_over=1.90, odds_under=1.95, league="Atlantis Test League"):
        return {
            "home": home, "away": "Rivals FC", "date": "19/10/2026",
            "league": league, "pick": "Over 2.5", "market": "O/U",
            "odds_over25": odds_over, "odds_under25": odds_under,
            "all_picks": [{"market": "O/U", "pick": "Over 2.5",
                           "confidence": "Medium", "model_prob": prob}],
        }

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import bettor
        from unittest import mock
        cls.B = bettor
        # These tests are about stakes, not the ledger: force every bucket
        # RELIABLE so a stale or absent reliability.json cannot empty the
        # ticket out from under them (the HELD gate has its own tests).
        mock.patch.object(
            bettor, "classify_pick",
            lambda pick, confidence, ledger, market=None:
                {"tier": "RELIABLE", "n": 999}).start()
        cls.addClassCleanup(mock.patch.stopall)

    def test_unknown_league_gets_the_weak_haircut_not_full_stake(self):
        B = self.B
        acc = {"O/U": {B._league_family("Prime Liga"): 0.60,
                       B._league_family("Seconda Liga"): 0.45}}
        self.assertEqual(B._rel_factor(acc, "Prime Liga", "O/U"), (1.0, "STUD"))
        # measured-but-weak is halved …
        self.assertEqual(B._rel_factor(acc, "Seconda Liga", "O/U"), (0.5, "WEAK"))
        # … and a league nothing has ever scored sits in the same bucket as
        # the weak one, not at the STUD stake.
        self.assertEqual(B._rel_factor(acc, "Atlantis Test League", "O/U"),
                         (0.5, "unknown"))

    def test_cap_cuts_the_weakest_legs_instead_of_scaling_everyone_down(self):
        import unittest.mock as mock
        B = self.B
        matches = [self._match(f"Team {i}", prob)
                   for i, prob in enumerate((0.70, 0.68, 0.66, 0.64), start=1)]
        with mock.patch.object(B, "TOTAL_BANKROLL_CAP_PCT", 9.0):
            card = B.build_card(matches)
        legs = card["value_picks"]
        # 5% x 4 legs at the 5% Kelly cap, halved for the unverified league,
        # so the sum is 10% against a 9% cap: exactly one leg has to go.
        self.assertEqual(card["total_stake_pct"], 7.5)
        self.assertEqual(card["capped_off"], 1)
        # the weakest EV went, and nobody was diluted to fit
        cut = [p for p in card["all_candidates"]
               if "card cap" in (p.get("note") or "")]
        self.assertEqual(len(cut), 1)
        self.assertEqual(cut[0]["model_prob"], 0.64)
        for p in legs:
            self.assertGreater(p["ev_pct"], 0.0)
            self.assertEqual(p["stake_pct"],
                             round(min((p["odds"] * p["model_prob"] - 1)
                                       / (p["odds"] - 1) * 0.25, 0.05)
                                   * 100 * 0.5, 3))
        self.assertEqual(card["total_stake_pct"],
                         round(sum(p["stake_pct"] for p in legs), 2))


class TestBettorGates(unittest.TestCase):
    """Reserve sides and picks-gate HELD buckets must never reach the ticket.

    A +EV longshot on a youth side, or on a (selection, confidence) bucket
    with no measured edge, is exactly the class of leg the model is worst at
    pricing (longshots: model 34.5% vs realised 12.7%) — it stays visible in
    all_candidates, annotated, and never gets a stake.
    """

    @staticmethod
    def _match(home="Rivals United", pick="Over 2.5", prob=0.60):
        return {
            "home": home, "away": "Away FC", "date": "20/10/2026",
            "league": "Atlantis Test League", "pick": pick, "market": "O/U",
            "odds_over25": 1.90, "odds_under25": 1.95,
            "all_picks": [{"market": "O/U", "pick": pick,
                           "confidence": "Medium", "model_prob": prob}],
        }

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import bettor
        cls.B = bettor

    def _build(self, tier, n=999, match=None):
        import unittest.mock as mock
        with mock.patch.object(self.B, "classify_pick",
                               lambda pick, confidence, ledger, market=None:
                                   {"tier": tier, "n": n}):
            return self.B.build_card([match or self._match()])

    def test_held_bucket_leg_is_visible_but_never_ticketed(self):
        card = self._build("HELD", n=36)
        self.assertEqual(card["value_picks"], [])
        cand = card["all_candidates"]
        self.assertEqual(len(cand), 1)
        self.assertTrue(cand[0].get("not_bettable"))
        self.assertIn("HELD by picks gate (n=36)", cand[0]["note"])
        self.assertEqual(card["blocked_held"], 1)
        self.assertEqual(card["blocked_reserve"], 0)

    def test_publishable_bucket_still_tickets(self):
        card = self._build("RELIABLE")
        self.assertEqual(len(card["value_picks"]), 1)
        p = card["value_picks"][0]
        self.assertEqual(p["tier"], "RELIABLE")
        self.assertFalse(p.get("not_bettable"))
        self.assertEqual(card["blocked_held"], 0)

    def test_reserve_fixture_is_never_ticketed(self):
        card = self._build("RELIABLE", match=self._match(home="Betis III"))
        self.assertEqual(card["value_picks"], [])
        cand = card["all_candidates"]
        self.assertTrue(cand[0].get("not_bettable"))
        self.assertIn("reserve/second team", cand[0]["note"])
        self.assertEqual(card["blocked_reserve"], 1)

    def test_watch_bucket_is_still_stakeable(self):
        card = self._build("WATCH")
        self.assertEqual(len(card["value_picks"]), 1)
        self.assertEqual(card["value_picks"][0]["tier"], "WATCH")

    def test_reliability_badge_renders_as_html_not_escaped_text(self):
        card = self._build("RELIABLE")
        card["value_picks"][0]["reliability"] = "WEAK"
        html_out = self.B.render_card_html(card)
        self.assertIn('<span style="color:#facc15;font-weight:700;">WEAK</span>',
                      html_out)
        self.assertNotIn("WEAK&lt;/span&gt;", html_out)


class TestMLLeakageAndElo(unittest.TestCase):
    """The ML feature vector must only contain information known before kickoff.

    Half-time goals are written after the match, so reading them trained the
    model on results it can never see when serving. Elo must be built only from
    earlier results so a fixture never rates itself.
    """

    @classmethod
    def setUpClass(cls):
        import ml_model  # heavy import, so done once
        cls.M = ml_model

    def test_half_time_goals_are_never_read(self):
        row = {"home_form": "WWDLW", "away_form": "LLDWL",
               "ht_home_goals": 3, "ht_away_goals": 2}
        f = self.M.extract_features_from_db_row(row)
        self.assertEqual(list(f[61:64]), [-1.0, -1.0, -1.0])

    def test_elo_pre_match_excludes_same_day_and_own_result(self):
        rows = [
            # day 1: A thrashes B. Both fixtures on day 2 must see the post-day-1
            # rating, and the day-2 fixtures must not see each other's result.
            {"id": 1, "forebet_url": "x-1", "home_team": "A", "away_team": "B",
             "match_date": "01/01/2026", "match_time": "15:00",
             "actual_home_goals": 5, "actual_away_goals": 0},
            {"id": 2, "forebet_url": "x-2", "home_team": "A", "away_team": "C",
             "match_date": "02/01/2026", "match_time": "15:00",
             "actual_home_goals": 0, "actual_away_goals": 4},
            {"id": 3, "forebet_url": "x-3", "home_team": "B", "away_team": "A",
             "match_date": "02/01/2026", "match_time": "17:00",
             "actual_home_goals": 1, "actual_away_goals": 1},
        ]
        pre = self.M.elo_pre_match_for_rows(rows)
        self.assertEqual(pre[1], (1500.0, 1500.0))            # nothing known yet
        self.assertGreater(pre[2][0], 1500.0)                 # A won 5-0 on day 1
        self.assertLess(pre[3][0], 1500.0)                    # B lost 5-0 on day 1
        self.assertEqual(pre[2][1], 1500.0)                   # C unrated before day 2
        # Both day-2 fixtures involving A read A's post-day-1 rating, not the
        # value after the other day-2 fixture.
        self.assertEqual(pre[3][1], pre[2][0])

    def test_post_kickoff_rows_detected(self):
        # Stored the day after the match: may carry the result, so excluded.
        self.assertTrue(self.M._created_after_kickoff(
            {"match_date": "01/01/2026", "created_at": "2026-01-02 10:00:00"}))
        # Stored on the match day: ex-ante.
        self.assertFalse(self.M._created_after_kickoff(
            {"match_date": "01/01/2026", "created_at": "2026-01-01 09:00:00"}))
        # Unreadable timestamp is kept, not silently dropped.
        self.assertFalse(self.M._created_after_kickoff(
            {"match_date": "01/01/2026", "created_at": None}))

    def test_over25_calibration_map(self):
        m = self.M.MLPredictor()
        self.assertEqual(m._calibrate_over(0.6), 0.6)          # no map: identity
        m.ou_platt = (-0.42, 1.78)                              # fitted shape
        lo, mid, hi = (m._calibrate_over(p) for p in (0.3, 0.5, 0.8))
        self.assertLess(lo, 0.3)                                # pulls low p down
        self.assertLess(mid, 0.5)                               # level bias removed
        self.assertLess(mid, hi)                                # still monotone

    def test_post_match_features_never_selected(self):
        import numpy as np
        rng = np.random.default_rng(0)
        X = rng.normal(size=(400, len(self.M.FEATURE_NAMES))).astype(np.float32)
        y1 = (X[:, 61] > 0).astype(int)        # HT home goals perfectly "predictive"
        y2 = (X[:, 5] > 0).astype(int)
        sel = self.M.select_features(X, y1, y2, max_features=50, min_independent=30)
        self.assertFalse(set(int(i) for i in sel) & self.M.POST_MATCH_FEATURES)


if __name__ == "__main__":
    unittest.main(verbosity=2)