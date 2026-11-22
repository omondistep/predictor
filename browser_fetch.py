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
import re
import time
import json
import atexit
import glob
import subprocess
import urllib.request
import zipfile
import io
import platform

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
    """Locate the real Google Chrome binary, deterministically.

    ``undetected_chromedriver.find_chrome_executable()`` picks from a *set* of
    candidates, so it can return Chromium or Chrome at random depending on hash
    order — and then disagree with whichever browser ``uc.Chrome`` launches.
    We pin one binary (real Chrome first) and pass it to ``uc.Chrome`` so the
    chromedriver we fetch always matches the browser that actually starts.
    """
    for cand in (
        os.environ.get("CHROME_BINARY", ""),
        shutil.which("google-chrome-stable") or "",
        shutil.which("google-chrome") or "",
        "/opt/google/chrome/chrome",
        shutil.which("chromium") or "",
        shutil.which("chromium-browser") or "",
    ):
        if cand and os.path.exists(cand):
            return cand
    return ""


_CHROME_FOR_TESTING = (
    "https://googlechromelabs.github.io/chrome-for-testing/"
    "latest-versions-per-milestone-with-downloads.json"
)


def _major_version(text: str) -> int:
    """Pull the leading integer (major version) out of a version string."""
    m = re.search(r"(\d+)", text or "")
    return int(m.group(1)) if m else 0


def _chrome_major_version() -> int:
    """Major version of the installed Google Chrome."""
    binary = _find_chrome_binary()
    if not binary:
        return 0
    try:
        out = subprocess.check_output(
            [binary, "--version"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=15,
        )
        return _major_version(out)
    except Exception:
        return 0


def _chromedriver_major_version(path: str) -> int:
    try:
        out = subprocess.check_output(
            [path, "--version"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=15,
        )
        return _major_version(out)
    except Exception:
        return 0


def _download_chromedriver(major: int) -> str:
    """Download the chrome-for-testing chromedriver matching Chrome's major version.

    ChromeDriver's major version must match the browser's major version, so when
    the system chromedriver is for a different Chrome (e.g. driver 155 vs
    Chrome 157) we fetch a matching build from Google's official archive and
    keep a writable copy inside the browser profile dir.

    Returns the path to the extracted chromedriver binary, or "" on failure.
    """
    try:
        with urllib.request.urlopen(_CHROME_FOR_TESTING, timeout=30) as r:
            data = json.load(r)
    except Exception:
        return ""
    entry = data.get("milestones", {}).get(str(major))
    if not entry:
        return ""
    machine = platform.machine().lower()
    target = "linux-arm64" if machine in ("aarch64", "arm64") else "linux64"
    url = None
    for d in entry.get("downloads", {}).get("chromedriver", []):
        if d.get("platform") == target:
            url = d["url"]
            break
    if not url:
        return ""
    try:
        with urllib.request.urlopen(url, timeout=90) as r:
            blob = r.read()
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            for name in z.namelist():
                if os.path.basename(name) == "chromedriver":
                    os.makedirs(_PROFILE_DIR, exist_ok=True)
                    dst = os.path.join(_PROFILE_DIR, "chromedriver")
                    with open(dst, "wb") as f:
                        f.write(z.read(name))
                    os.chmod(dst, 0o755)
                    return dst
    except Exception:
        return ""
    return ""


def _ensure_chromedriver() -> str:
    """Return a writable, patchable chromedriver path for undetected_chromedriver.

    ucd 3.5.x patches the driver executable to evade bot detection, which
    fails if chromedriver lives in a root-owned directory — so we keep a
    writable copy inside the browser profile dir.

    The system chromedriver is only used when its major version matches the
    installed Chrome; a mismatch (ChromeDriver 155 vs Chrome 157) aborts every
    session, so in that case we download a matching build from the official
    chrome-for-testing archive instead (and reuse it on later launches).
    """
    chrome_major = _chrome_major_version()
    cached = os.path.join(_PROFILE_DIR, "chromedriver")

    # Reuse a previously downloaded driver when it still matches the browser.
    if chrome_major and os.path.exists(cached):
        if _chromedriver_major_version(cached) == chrome_major:
            return cached

    src = shutil.which("chromedriver") or os.environ.get("CHROMEDRIVER", "")
    if not src or not os.path.exists(src):
        # No system driver: fetch a matching one, else let ucd download it.
        if chrome_major:
            downloaded = _download_chromedriver(chrome_major)
            if downloaded:
                return downloaded
        return ""

    driver_major = _chromedriver_major_version(src)
    if chrome_major and driver_major and chrome_major != driver_major:
        downloaded = _download_chromedriver(chrome_major)
        if downloaded:
            return downloaded
        # Let undetected-chromedriver fetch its own matching driver.
        return ""

    if os.access(src, os.W_OK):
        return src
    if not os.path.exists(cached) or os.path.getmtime(cached) < os.path.getmtime(src):
        os.makedirs(_PROFILE_DIR, exist_ok=True)
        shutil.copy2(src, cached)
        os.chmod(cached, 0o755)
    return cached


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
        browser_executable_path=_find_chrome_binary() or None,
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