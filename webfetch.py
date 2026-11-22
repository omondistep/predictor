"""Shared HTTP session factory that passes Forebet's Cloudflare protection.

cloudscraper can no longer solve the challenge Forebet serves (403 "Just a
moment..."), so we impersonate a real Chrome TLS fingerprint via curl_cffi
(curl-impersonate) *and* reuse a cf_clearance cookie harvested once from a real
headed Chrome. Combined, this presents a fully trusted identity,
so sustained scraping no longer trips the per-IP block. Falls back to
cloudscraper if curl_cffi is unavailable.

Mitigations, in order:
  * harvest cf_clearance + friends from real Chrome once (refresh_cf_cookies),
    persist to data/cf_cookies.json and attach to every curl_cffi session;
  * rotate TLS impersonation across several recent Chrome versions on failure;
  * retry with exponential backoff + jitter to wait out a temporary block;
  * detect challenge/blocked bodies so callers can say so instead of
    reporting a silent "no data".
"""

import os
import sys
import json
import time
import random

IMPERSONATE = "chrome124"

# Cloudflare grants a *cookie* (cf_clearance) to a browser that solves its
# managed challenge. We harvest that cookie through a real headed Chrome once,
# persist it, and reuse it in the fast curl_cffi path so every subsequent
# request presents the same trusted identity. Gitignored (lives in data/).
_CF_COOKIE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "cf_cookies.json"
)
_last_browser_refresh = 0.0
BROWSER_REFRESH_MIN_INTERVAL = 300.0  # seconds between browser re-solves

# Recent desktop Chrome TLS profiles, newest first. Forebet can single one out,
# so we rotate on failure rather than trusting a fixed fingerprint.
ROTATE_AMONG = [
    "chrome150", "chrome146", "chrome145", "chrome142",
    "chrome136", "chrome133a", "chrome131", "chrome124",
]

_CHALLENGE_MARKERS = (
    "just a moment",
    "cf-chl-",
    "challenge-platform",
    "cf-please-wait",
    "attention required",
    "verifying you are human",
    "enable javascript and cookies",
)

# IP-level Cloudflare blocks hit *every* URL, so when more than one distinct
# URL is blocked we stop retrying per-URL and make subsequent calls fail fast
# until this window lapses. A single bad URL won't poison a batch.
GLOBAL_BLOCK_COOLDOWN = 300.0  # seconds
_blocked_until = 0.0
_blocked_urls = set()


def _enter_global_backoff():
    global _blocked_until
    _blocked_until = time.time() + GLOBAL_BLOCK_COOLDOWN


def _in_global_backoff() -> bool:
    return time.time() < _blocked_until


def _note_blocked_url(url):
    """Trigger global fail-fast once two *different* URLs are seen blocked.

    Returns True if the caller should report a block instead of retrying.
    """
    _blocked_urls.add(url)
    if len(_blocked_urls) >= 2:
        _enter_global_backoff()
        return True
    return False


def _looks_blocked(body: str) -> bool:
    """Heuristic for whether a body is a Cloudflare challenge / block page."""
    if not body:
        return True
    low = body.lower()
    return (
        "cf-ray" in low
        and any(m in low for m in _CHALLENGE_MARKERS)
    ) or any(m in low for m in _CHALLENGE_MARKERS)


def _load_cookies() -> dict:
    """Load the persisted Cloudflare cookies ({} if none / stale)."""
    try:
        if os.path.exists(_CF_COOKIE_PATH):
            with open(_CF_COOKIE_PATH) as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def _save_cookies(cookies: dict) -> None:
    try:
        os.makedirs(os.path.dirname(_CF_COOKIE_PATH), exist_ok=True)
        with open(_CF_COOKIE_PATH, "w") as f:
            json.dump(cookies, f, indent=2)
    except Exception as e:
        print(f"[webfetch] could not persist cookies: {e}", file=sys.stderr)


def refresh_cf_cookies(timeout: int = 60) -> bool:
    """Solve Cloudflare in a real browser once and harvest its cookies.

    Navigates headed Chrome to Forebet (which auto-solves the managed
    challenge), extracts cf_clearance + friends and the real UA, then persists
    them for curl_cffi to reuse. Cooldown-gated so we never spam the browser.
    """
    global _last_browser_refresh
    now = time.time()
    if now - _last_browser_refresh < BROWSER_REFRESH_MIN_INTERVAL:
        return bool(_load_cookies())
    _last_browser_refresh = now
    try:
        from browser_fetch import (
            fetch as browser_get, is_available, get_cookies, get_user_agent,
        )
        if not is_available():
            return False
        b = browser_get(
            "https://www.forebet.com/en/football-tips-and-predictions-for-today",
            timeout=timeout,
        )
        if not b.ok:
            return False
        cookies = get_cookies()
        if not cookies:
            return False
        cookies["__ua__"] = get_user_agent()
        _save_cookies(cookies)
        return True
    except Exception as e:
        print(f"[webfetch] cookie refresh failed: {e}", file=sys.stderr)
        return False


def _attach_cookies(session) -> None:
    """Apply persisted cf cookies + real browser UA onto a session."""
    cookies = _load_cookies()
    ua = cookies.pop("__ua__", "")
    if ua:
        try:
            session.headers.update({"User-Agent": ua})
        except Exception:
            pass
    for name, value in cookies.items():
        try:
            session.cookies.set(name, value, domain=".forebet.com")
        except Exception:
            pass


def create_session(impersonate: str = IMPERSONATE):
    """Return an HTTP session whose GETs pass Cloudflare on forebet.com.

    ``impersonate`` may be overridden to rotate the TLS profile. Persisted
    cf_clearance cookies harvested from real Chrome are attached when present.
    """
    try:
        from curl_cffi import requests as curl_requests
        session = curl_requests.Session(impersonate=impersonate)
        _attach_cookies(session)
        return session
    except ImportError:
        pass
    print("[webfetch] curl_cffi missing — falling back to cloudscraper "
          "(may get blocked: pip install curl_cffi)", file=sys.stderr)
    import cloudscraper
    session = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "linux", "mobile": False}
    )
    _attach_cookies(session)
    return session


class FetchResult:
    """Lightweight wrapper so retry logic can work with any session type."""

    __slots__ = ("status_code", "text", "ok")

    def __init__(self, status_code=0, text="", ok=False):
        self.status_code = status_code
        self.text = text
        self.ok = ok


def _get_once(session, url, timeout):
    """Single GET that normalises both curl_cffi and requests responses."""
    try:
        r = session.get(url, timeout=timeout)
    except Exception:
        return None
    return r


def _browser_fallback(url, timeout):
    """Fetch a URL through a real headed Chrome once. Returns FetchResult."""
    try:
        from browser_fetch import fetch as browser_get, is_available
        if not is_available():
            return None
        b = browser_get(url)
        if b.ok:
            global _blocked_until
            _blocked_until = 0.0
            _blocked_urls.clear()
            return FetchResult(status_code=b.status_code, text=b.text, ok=True)
        return FetchResult(status_code=b.status_code, text=b.text, ok=False)
    except Exception as e:
        print(f"[webfetch] browser fallback failed for {url}: {e}", file=sys.stderr)
        return None


def fetch_with_retry(url, timeout=25, max_attempts=5, session=None):
    """GET a URL surviving Forebet's Cloudflare protection.

    Fast path: curl_cffi TLS impersonation (rotates across Chrome profiles on
    failure, with exponential backoff + jitter).

    Fallback path: when Forebet challenges our automated client (403 or a 200
    "Just a moment..." page) we fetch through a real headed Chrome, which
    solves the managed challenge automatically. Once curl is observed failing
    on several distinct URLs we skip further curl attempts and use Chrome
    directly.

    ``session`` may supply a caller-owned session for the first (fast) attempt;
    on failure we fall back to rotating a fresh one per retry.

    Returns a FetchResult. ``ok`` is True only for a genuine pass (HTTP 200
    with non-blocked body). ``status_code``/``text`` are populated so callers
    can report the failure precisely.
    """
    # Already known that curl is blocked (2+ distinct URLs failed): use Chrome.
    if _in_global_backoff():
        b = _browser_fallback(url, timeout)
        return b if b is not None else FetchResult(ok=False)

    last = FetchResult()
    browser_tried = False
    cookie_refreshed = False
    for attempt in range(max_attempts):
        try:
            if attempt == 0 and session is not None:
                imp = IMPERSONATE
                s = session
            else:
                imp = ROTATE_AMONG[attempt % len(ROTATE_AMONG)]
                s = create_session(imp)
            r = _get_once(s, url, timeout)
        except Exception as e:  # network blip — retry
            last = FetchResult(ok=False, text=str(e))
            if attempt < max_attempts - 1:
                _sleep_backoff(attempt)
            continue

        if r is None:
            last = FetchResult(ok=False)
            if attempt < max_attempts - 1:
                _sleep_backoff(attempt)
            continue

        code = getattr(r, "status_code", 0)
        body = getattr(r, "text", "") or ""
        blocked = bool(body) and _looks_blocked(body)

        if code == 200 and not blocked:
            # A real pass clears any earlier global block state.
            global _blocked_until
            _blocked_until = 0.0
            _blocked_urls.clear()
            return FetchResult(status_code=200, text=body, ok=True)

        # 403, or a 200 challenge page -> treat as blocked.
        last = FetchResult(status_code=code, text=body, ok=False)
        if attempt == 0:
            _note_blocked_url(url)
            if not browser_tried and not _in_global_backoff():
                browser_tried = True
                # Prefer baking cf_clearance via a real browser once, then
                # reusing it in curl for this URL and everything after (much
                # cheaper than per-request browser navigation).
                if refresh_cf_cookies():
                    cookie_refreshed = True
                    if session is not None:
                        _attach_cookies(session)
                    continue  # retry immediately with the fresh cookies
                # No display/browser: fall back to direct browser fetch.
                b = _browser_fallback(url, timeout)
                if b is not None:
                    return b
        if attempt < max_attempts - 1:
            _sleep_backoff(attempt)

    return last


def _sleep_backoff(attempt):
    """Exponential backoff with jitter, base 4s, capped at ~30s."""
    base = 4.0 * (2 ** attempt)
    jitter = random.uniform(0, 1.5)
    delay = min(base + jitter, 30.0)
    time.sleep(delay)
