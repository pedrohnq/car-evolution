"""
Project path resolution and evolution log file naming.

``PROJECT_ROOT`` is the directory that **contains** the ``car_evolution`` package (the same
folder as ``main.py``). It is derived from this file's location: ``io/`` -> parent is
``car_evolution/`` -> its parent is the repository root.

``LOGS_DIR`` is always ``PROJECT_ROOT / "logs"``. :func:`ensure_logs_dir` creates it.
Each run typically writes one timestamped CSV via :func:`evolution_log_path`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

_CAR_EVOLUTION_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _CAR_EVOLUTION_DIR.parent

LOGS_DIR = PROJECT_ROOT / "logs"


def ensure_logs_dir() -> Path:
    """
    Create the logs directory if it does not exist.

    Returns:
        Absolute path to the logs directory.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return LOGS_DIR


def evolution_log_filename(timestamp: datetime | None = None) -> str:
    """
    Build a timestamped CSV filename for one evolution run.

    Args:
        timestamp: Moment to encode; defaults to "now".

    Returns:
        Filename like ``evolution_log_20260403_134527.csv`` (no directory).
    """
    ts = timestamp or datetime.now()
    return f"evolution_log_{ts.strftime('%Y%m%d_%H%M%S')}.csv"


def evolution_log_path(timestamp: datetime | None = None) -> Path:
    """
    Full path for a new evolution log file under :data:`LOGS_DIR`.

    Args:
        timestamp: Optional fixed time for reproducible names in tests.

    Returns:
        Absolute path where the CSV should be written.
    """
    ensure_logs_dir()
    return LOGS_DIR / evolution_log_filename(timestamp)
