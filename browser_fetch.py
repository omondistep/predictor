"""Real-browser fetch transport for Forebet.

curl_cffi (TLS impersonation) normally passes Cloudflare, but under sustained
scraping Forebet starts challenging every automated client on the IP — curl and
headless browsers get "Just a moment...", while a *real* headed Chrome passes
automatically (it solves the managed challenge and gets a cf_clearance cookie).

This module provides a fallback that fetches pages through real Google Chrome
via undetected-chromedriver (stealth-patched chromedriver launched headed).
A persistent profile is reused so once Chrome has solved the challenge it stays
trusted across runs.

Only available on machines with a display (DISPLAY set on Linux); otherwise it
reports unavailability and callers keep using the fast client.
"""

import os
import sys
import shutil
import time
import atexit
import glob

# Gitignored so the profile/harvested cookies never enter the repo.
_PROFILE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "browser_profile"
)

_driver = None


def is_available() -> bool:
    """True when we can realistically drive a real headed Chrome."""
    if os.environ.get("FOREBET_BROWSER_FALLBACK") == "0":
        return False
    if not _find_chrome_binary():
        return False
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        return False
    return True


def _find_chrome_binary() -> str:
    """Locate the real Google Chrome binary."""
    try:
        import undetected_chromedriver as uc
        return uc.find_chrome_executable()
    except Exception:
        pass
    for cand in (
        os.environ.get("CHROME_BINARY", ""),
        shutil.which("google-chrome-stable") or "",
        shutil.which("google-chrome") or "",
        shutil.which("chromium") or "",
        "/opt/google/chrome/chrome",
    ):
        if cand and os.path.exists(cand):
            return cand
    return ""


def _ensure_chromedriver() -> str:
    """Return a writable, patchable chromedriver path for undetected_chromedriver.

    ucd 3.5.x patches the driver executable to evade bot detection, which
    fails if chromedriver lives in a root-owned directory — so we keep a
    writable copy inside the browser profile dir.
    """
    src = shutil.which("chromedriver") or os.environ.get("CHROMEDRIVER", "")
    if not src or not os.path.exists(src):
        # ucd can download one itself as a last resort.
        return ""
    if os.access(src, os.W_OK):
        return src
    dst = os.path.join(_PROFILE_DIR, "chromedriver")
    if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
        os.makedirs(_PROFILE_DIR, exist_ok=True)
        shutil.copy2(src, dst)
        os.chmod(dst, 0o755)
    return dst


class Result:
    __slots__ = ("status_code", "text", "ok")

    def __init__(self, status_code=0, text="", ok=False):
        self.status_code = status_code
        self.text = text
        self.ok = ok


def _launch():
    """Launch (once) the persistent headed Chrome session."""
    global _driver
    if _driver is not None:
        return _driver

    import undetected_chromedriver as uc

    opts = uc.ChromeOptions()
    opts.add_argument("--no-first-run")
    opts.add_argument("--no-default-browser-check")
    opts.add_argument("--ozone-platform=x11")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--start-maximized")

    os.makedirs(_PROFILE_DIR, exist_ok=True)
    for stale in glob.glob(os.path.join(_PROFILE_DIR, "Singleton*")):
        try:
            os.remove(stale)
        except OSError:
            pass
    _driver = uc.Chrome(
        user_data_dir=_PROFILE_DIR,
        options=opts,
        driver_executable_path=_ensure_chromedriver() or None,
        headless=False,
    )
    return _driver


def get_cookies() -> dict:
    """Return session cookies from the live Chrome driver, e.g. cf_clearance.

    The real browser passes Cloudflare's managed challenge and is granted a
    cf_clearance cookie; harvesting it lets the fast curl_cffi path reuse the
    same identity instead of re-solving the challenge per request.
    """
    if _driver is None:
        return {}
    try:
        return {c["name"]: c["value"] for c in _driver.get_cookies()}
    except Exception:
        return {}


def get_user_agent() -> str:
    """Return the real Chrome UA so curl_cffi presents the same identity."""
    if _driver is None:
        return ""
    try:
        return str(_driver.execute_script("return navigator.userAgent") or "")
    except Exception:
        return ""


def shutdown():
    """Quit the persistent Chrome session (call at process exit)."""
    global _driver
    if _driver is not None:
        try:
            _driver.quit()
        except Exception:
            pass
        _driver = None


def fetch(url: str, timeout: int = 40, settle: float = 0.8) -> Result:
    """Fetch a URL through real Chrome. Returns Result or Result(ok=False).

    Cloudflare sometimes greets a fresh profile with a "Just a moment..." page
    that auto-solves in a real browser within a second or two; we poll briefly
    for the real content instead of assuming the first paint is final.

    If Chrome's session dies mid-navigation (seen sporadically on Wayland/X11
    builds), we relaunch once and retry before giving up.
    """
    if not is_available():
        return Result(ok=False)
    for attempt in range(2):  # allow one relaunch
        try:
            driver = _launch()
            driver.set_page_load_timeout(timeout)
            driver.get(url)
            deadline = time.time() + 15
            src = driver.page_source or ""
            low = src.lower()
            while (time.time() < deadline) and (
                (not src) or "just a moment" in low or "verifying you are human" in low
            ):
                time.sleep(1.0)
                src = driver.page_source or ""
                low = src.lower()
            time.sleep(settle)
            src = driver.page_source or ""
            low = src.lower()
            if "just a moment" in low or "verifying you are human" in low:
                return Result(status_code=403, text=src, ok=False)
            return Result(status_code=200, text=src, ok=bool(src))
        except Exception as e:
            print(f"[browser_fetch] attempt {attempt + 1} failed for {url}: {e}",
                  file=sys.stderr)
            shutdown()  # drop dead session so the next attempt relaunches fresh
    return Result(ok=False)


atexit.register(shutdown)