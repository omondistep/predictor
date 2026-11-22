"""Generated-report filename helpers.

Every report the pipeline generates gets a timestamped filename --
``<base>_<YYYYmmdd_HHMMSS>.html`` -- so that consecutive runs write *fresh*
files instead of replacing the previous output.  The timestamp differentiates
two outputs generated at different times (or even within the same run), and a
same-second collision gets a numeric suffix rather than silently overwriting.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


def timestamp_slug(now: Optional[datetime] = None) -> str:
    """Seconds-resolution run timestamp, e.g. ``20261122_093012``."""
    return (now or datetime.now()).strftime(TIMESTAMP_FORMAT)


def timestamped_name(base: str, now: Optional[datetime] = None) -> str:
    """Return ``<base>_<YYYYmmdd_HHMMSS>.html``.

    A time-differentiated name, so a later run's output can never replace an
    earlier run's output that happens to share the same base name.
    """
    return f"{base}_{timestamp_slug(now)}.html"


def stamped_path(base_path: Path, now: Optional[datetime] = None) -> Path:
    """Return a fresh, timestamped path based on ``base_path``.

    ``predictions/picks.html`` comes back as
    ``predictions/picks_20261122_093012.html`` (a same-second collision gets a
    ``_<n>`` suffix).  Writes through the returned path therefore never clobber
    an older report.
    """
    pred_dir = base_path.parent
    pred_dir.mkdir(parents=True, exist_ok=True)
    slug = timestamp_slug(now)
    candidate = pred_dir / f"{base_path.stem}_{slug}{base_path.suffix}"
    n = 1
    while candidate.exists():
        n += 1
        candidate = pred_dir / f"{base_path.stem}_{slug}_{n}{base_path.suffix}"
    return candidate


def latest_stamp(pred_dir: Path, base: str) -> Optional[Path]:
    """Latest timestamped report in ``pred_dir`` whose stem starts with ``base``.

    e.g. ``base="latest"`` -> newest ``predictions/latest_<ts>.html``.
    """
    files = sorted(
        (f for f in pred_dir.glob(f"{base}_*.html") if f.is_file()),
        key=lambda f: f.name,
    )
    return files[-1] if files else None
