#!/usr/bin/env python3
"""Relocate flat CI artifact telemetry into ``data/daily_news/<date>/processed/``."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.daily_news_paths import normalize_daily_news_root

TELEMETRY_NAMES = (
    "session_summary.json",
    "live_loop_log.jsonl",
    "order_events.jsonl",
    "backtest_results.json",
    "premarket_context.json",
)


def _date_from_payload(path: Path) -> str | None:
    if not path.exists() or path.suffix != ".json":
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    raw = data.get("date")
    if isinstance(raw, str) and len(raw) == 10:
        return raw
    return None


def _infer_date(project_root: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    for name in ("session_summary.json", "backtest_results.json"):
        for candidate in (
            project_root / name,
            project_root / "processed" / name,
        ):
            found = _date_from_payload(candidate)
            if found:
                return found
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    except Exception:
        return datetime.utcnow().date().isoformat()


def _collect_flat_sources(project_root: Path) -> list[Path]:
    found: list[Path] = []
    search_roots = (project_root, project_root / "processed")
    for root in search_roots:
        if not root.is_dir():
            continue
        for name in TELEMETRY_NAMES:
            path = root / name
            if path.is_file():
                found.append(path)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="Trading date YYYY-MM-DD (default: infer from summary JSON or ET today)")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=_PROJECT_ROOT,
        help="Repository root (default: parent of scripts/)",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    daily_root = normalize_daily_news_root(project_root / "data" / "daily_news")
    run_date = _infer_date(project_root, args.date)
    dest = daily_root / run_date / "processed"
    dest.mkdir(parents=True, exist_ok=True)

    existing = [dest / name for name in TELEMETRY_NAMES if (dest / name).is_file()]
    if existing:
        print(f"OK: telemetry already under {dest} ({len(existing)} file(s))")
        return

    sources = _collect_flat_sources(project_root)
    if not sources:
        print(f"No flat telemetry files to relocate (dest={dest})")
        return

    for src in sources:
        target = dest / src.name
        if target.exists():
            continue
        shutil.move(str(src), str(target))
        print(f"Moved {src} -> {target}")

    print(f"Normalized telemetry layout for {run_date} at {dest}")


if __name__ == "__main__":
    main()
