"""Tests for telemetry directory normalization."""

from __future__ import annotations

from pathlib import Path

from src.utils.daily_news_paths import normalize_daily_news_root


def test_normalize_daily_news_root_unwraps_duplicate_prefix(tmp_path: Path) -> None:
    outer = tmp_path / "data" / "daily_news"
    inner = outer / "data" / "daily_news"
    day = inner / "2026-05-01" / "processed"
    day.mkdir(parents=True)
    (day / "session_summary.json").write_text("{}", encoding="utf-8")

    resolved = normalize_daily_news_root(outer)
    assert resolved == inner.resolve()
    date_dirs = [
        p for p in resolved.iterdir() if p.is_dir() and len(p.name) == 10
    ]
    assert len(date_dirs) >= 1


def test_normalize_daily_news_root_noop_when_dates_at_root(tmp_path: Path) -> None:
    root = tmp_path / "data" / "daily_news"
    day = root / "2026-05-02" / "processed"
    day.mkdir(parents=True)
    (day / "session_summary.json").write_text("{}", encoding="utf-8")

    assert normalize_daily_news_root(root) == root.resolve()
