"""Helpers for locating ``data/daily_news`` telemetry trees (CI artifacts vs local)."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def _has_date_run_subdirs(path: Path) -> bool:
    """True if ``path`` contains ``YYYY-MM-DD`` directories (telemetry layout)."""
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if not child.is_dir() or len(child.name) != 10:
            continue
        try:
            datetime.strptime(child.name, "%Y-%m-%d")
        except ValueError:
            continue
        return True
    return False


def normalize_daily_news_root(root: Path) -> Path:
    """Collapse ``data/daily_news/data/daily_news/<date>/…`` from artifact downloads."""
    if not root.exists():
        return root
    nested = root / "data" / "daily_news"
    if (
        nested.is_dir()
        and not _has_date_run_subdirs(root)
        and _has_date_run_subdirs(nested)
    ):
        LOGGER.info(
            "Using nested telemetry root %s (artifact path normalization)",
            nested,
        )
        return nested.resolve()
    return root
