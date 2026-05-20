"""Helpers for locating ``data/daily_news`` telemetry trees (CI artifacts vs local)."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger(__name__)

PREMARKET_CONTEXT_FILENAME = "premarket_context.json"


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


def resolve_premarket_context_path(
    date_str: str,
    project_root: Path,
) -> Optional[Path]:
    """Locate ``premarket_context.json`` for *date_str* (CI artifact layouts).

    ``upload-artifact@v4`` uses the least common ancestor of matched paths. A lone
    ``data/daily_news/<date>/processed/premarket_context.json`` is often stored as
    ``premarket_context.json`` at the artifact root, so callers must check both.
    """
    root = project_root.resolve()
    daily_root = normalize_daily_news_root(root / "data" / "daily_news")
    candidates = [
        daily_root / date_str / "processed" / PREMARKET_CONTEXT_FILENAME,
        root / "data" / "daily_news" / date_str / "processed" / PREMARKET_CONTEXT_FILENAME,
        root / PREMARKET_CONTEXT_FILENAME,
        root
        / "data"
        / "daily_news"
        / "data"
        / "daily_news"
        / date_str
        / "processed"
        / PREMARKET_CONTEXT_FILENAME,
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved

    if daily_root.is_dir():
        for match in sorted(daily_root.rglob(PREMARKET_CONTEXT_FILENAME)):
            if date_str in match.as_posix():
                return match.resolve()
    return None
