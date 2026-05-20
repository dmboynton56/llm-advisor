"""Tests for flat CI telemetry relocation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.normalize_telemetry_layout import _collect_flat_sources


def test_collect_flat_sources_at_repo_root(tmp_path: Path) -> None:
    summary = tmp_path / "session_summary.json"
    summary.write_text(json.dumps({"date": "2026-05-20", "mode": "live"}), encoding="utf-8")
    found = _collect_flat_sources(tmp_path)
    assert summary in found


def test_relocate_moves_flat_files(tmp_path: Path) -> None:
    import scripts.normalize_telemetry_layout as mod

    summary = tmp_path / "session_summary.json"
    summary.write_text(
        json.dumps({"date": "2026-05-20", "mode": "live", "total_trades": 0}),
        encoding="utf-8",
    )
    log = tmp_path / "live_loop_log.jsonl"
    log.write_text(
        '{"ts":"2026-05-20T16:00:00+00:00","symbols":{},"shutdown":true}\n',
        encoding="utf-8",
    )

    argv = sys.argv
    try:
        sys.argv = [
            "normalize_telemetry_layout.py",
            "--project-root",
            str(tmp_path),
            "--date",
            "2026-05-20",
        ]
        mod.main()
    finally:
        sys.argv = argv

    dest = tmp_path / "data" / "daily_news" / "2026-05-20" / "processed"
    assert (dest / "session_summary.json").is_file()
    assert (dest / "live_loop_log.jsonl").is_file()
