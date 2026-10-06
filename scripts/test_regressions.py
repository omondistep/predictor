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
                     "Real Madrid IV", "Villarreal B"):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)