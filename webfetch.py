"""Shared HTTP session factory that passes Forebet's Cloudflare protection.

cloudscraper can no longer solve the challenge Forebet serves (403 "Just a
moment..."), so we impersonate a real Chrome TLS fingerprint via curl_cffi
(curl-impersonate). Falls back to cloudscraper if curl_cffi is unavailable.
"""

import sys

IMPERSONATE = "chrome124"


def create_session():
    """Return an HTTP session whose GETs pass Cloudflare on forebet.com."""
    try:
        from curl_cffi import requests as curl_requests
        return curl_requests.Session(impersonate=IMPERSONATE)
    except ImportError:
        pass
    print("[webfetch] curl_cffi missing — falling back to cloudscraper "
          "(may get blocked: pip install curl_cffi)", file=sys.stderr)
    import cloudscraper
    return cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "linux", "mobile": False}
    )
