"""Tests for locating premarket_context.json across CI artifact layouts."""

from __future__ import annotations

from pathlib import Path

from src.utils.daily_news_paths import resolve_premarket_context_path


def test_resolve_standard_layout(tmp_path: Path) -> None:
    target = (
        tmp_path
        / "data"
        / "daily_news"
        / "2026-05-20"
        / "processed"
        / "premarket_context.json"
    )
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")

    resolved = resolve_premarket_context_path("2026-05-20", tmp_path)
    assert resolved == target.resolve()


def test_resolve_flat_artifact_at_repo_root(tmp_path: Path) -> None:
    flat = tmp_path / "premarket_context.json"
    flat.write_text("{}", encoding="utf-8")

    resolved = resolve_premarket_context_path("2026-05-20", tmp_path)
    assert resolved == flat.resolve()
