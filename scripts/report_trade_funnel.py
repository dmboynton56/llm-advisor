#!/usr/bin/env python3
"""Report the signal-to-exit trading funnel from local telemetry or Supabase."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen


FUNNEL_EVENTS = (
    "signal_detected",
    "validation_approved",
    "validation_rejected",
    "execution_attempt",
    "execution_failed",
    "execution_succeeded",
)


@dataclass
class DayFunnel:
    run_date: str
    counts: Counter[str] = field(default_factory=Counter)
    failure_reasons: Counter[str] = field(default_factory=Counter)
    exit_reasons: Counter[str] = field(default_factory=Counter)
    total_pnl: float = 0.0
    trade_count: int = 0
    position_notional: float = 0.0
    position_samples: int = 0
    sources: set[str] = field(default_factory=set)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_date": self.run_date,
            **{event: self.counts[event] for event in FUNNEL_EVENTS},
            "failure_reasons": dict(self.failure_reasons.most_common()),
            "exit_reasons": dict(self.exit_reasons.most_common()),
            "trade_count": self.trade_count,
            "total_pnl": round(self.total_pnl, 2),
            "average_position_size": (
                round(self.position_notional / self.position_samples, 2)
                if self.position_samples
                else None
            ),
            "sources": sorted(self.sources),
        }


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _details(event: dict[str, Any]) -> dict[str, Any]:
    details = event.get("details")
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except json.JSONDecodeError:
            details = {}
    return details if isinstance(details, dict) else {}


def _add_events(day: DayFunnel, events: Iterable[dict[str, Any]]) -> None:
    for event in events:
        event_type = str(event.get("event_type") or "").strip()
        if not event_type:
            continue
        day.counts[event_type] += 1
        details = _details(event)
        if event_type == "execution_failed":
            reason = str(details.get("reason") or details.get("error") or "unknown")
            day.failure_reasons[reason] += 1
        if event_type in {"position_closed", "trade_closed", "exit_succeeded", "execution_exit"}:
            reason = str(details.get("reason") or details.get("exit_reason") or "unknown")
            day.exit_reasons[reason] += 1


def _add_trades(day: DayFunnel, trades: Iterable[dict[str, Any]]) -> None:
    for trade in trades:
        if not isinstance(trade, dict):
            continue
        day.trade_count += 1
        try:
            day.total_pnl += float(trade.get("pnl") or 0.0)
        except (TypeError, ValueError):
            pass
        reason = str(trade.get("exit_reason") or "").strip()
        if reason:
            day.exit_reasons[reason] += 1
        try:
            qty = abs(float(trade.get("qty") or 0.0))
            price = float(trade.get("entry_price") or 0.0)
            multiplier = 100.0 if str(trade.get("asset_class") or "").lower() == "option" else 1.0
            notional = qty * price * multiplier
            if notional > 0:
                day.position_notional += notional
                day.position_samples += 1
        except (TypeError, ValueError):
            pass


def collect_local(data_root: Path, start: str | None = None, end: str | None = None) -> dict[str, DayFunnel]:
    days: dict[str, DayFunnel] = {}
    if not data_root.exists():
        return days
    for day_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        run_date = day_dir.name
        try:
            date.fromisoformat(run_date)
        except ValueError:
            continue
        if (start and run_date < start) or (end and run_date > end):
            continue
        processed = day_dir / "processed"
        event_path = processed / "order_events.jsonl"
        summary_path = processed / "session_summary.json"
        if not summary_path.exists():
            summary_path = processed / "backtest_results.json"
        events = _read_jsonl(event_path)
        summary = _read_json(summary_path)
        log_path = processed / "live_loop_log.jsonl"
        if not events and log_path.exists():
            # Compatibility for operational runs predating order_events.jsonl.
            for loop_row in _read_jsonl(log_path):
                for signal in loop_row.get("signals") or []:
                    if isinstance(signal, dict):
                        events.append({"event_type": "signal_detected", "signal": signal})
            source = "live_loop_log.jsonl (signal fallback)"
        else:
            source = "order_events.jsonl"
        if not events and not isinstance(summary, dict):
            continue
        day = days.setdefault(run_date, DayFunnel(run_date))
        if events:
            _add_events(day, events)
            day.sources.add(source)
        if isinstance(summary, dict):
            trades = summary.get("trades") or []
            _add_trades(day, trades if isinstance(trades, list) else [])
            if not trades:
                day.trade_count = max(day.trade_count, int(summary.get("total_trades") or 0))
                day.total_pnl = float(summary.get("total_pnl") or day.total_pnl)
            day.sources.add(summary_path.name)
    return days


def _supabase_rows(table: str, query: dict[str, str]) -> list[dict[str, Any]]:
    base = (os.getenv("SUPABASE_URL") or "").rstrip("/")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY") or ""
    if not base or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_KEY) are required")
    if base.endswith("/rest/v1"):
        base = base[: -len("/rest/v1")]
    request = Request(
        f"{base}/rest/v1/{table}?{urlencode(query, safe='(),.*')}",
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310 - configured API endpoint
        payload = json.load(response)
    return payload if isinstance(payload, list) else []


def collect_supabase(start: str | None, end: str | None) -> dict[str, DayFunnel]:
    filters = ["select=run_date,event_type,details"]
    if start:
        filters.append(f"run_date=gte.{start}")
    if end:
        filters.append(f"run_date=lte.{end}")
    event_query = dict(item.split("=", 1) for item in filters)
    trade_query = {"select": "run_date,asset_class,qty,entry_price,exit_reason,pnl"}
    if start:
        event_query["run_date"] = f"gte.{start}"
        trade_query["run_date"] = f"gte.{start}"
    # PostgREST needs an `and` expression when both date bounds are supplied.
    if start and end:
        date_filter = f"(run_date.gte.{start},run_date.lte.{end})"
        event_query.pop("run_date", None)
        trade_query.pop("run_date", None)
        event_query["and"] = date_filter
        trade_query["and"] = date_filter
    elif end:
        event_query["run_date"] = f"lte.{end}"
        trade_query["run_date"] = f"lte.{end}"
    events = _supabase_rows("llm_advisor_order_events", event_query)
    trades = _supabase_rows("llm_advisor_backtest_trades", trade_query)
    days: dict[str, DayFunnel] = {}
    for run_date in sorted({str(row.get("run_date")) for row in events + trades if row.get("run_date")}):
        day = days.setdefault(run_date, DayFunnel(run_date))
        _add_events(day, (row for row in events if str(row.get("run_date")) == run_date))
        _add_trades(day, (row for row in trades if str(row.get("run_date")) == run_date))
        day.sources.add("supabase")
    return days


def merge_days(target: dict[str, DayFunnel], incoming: dict[str, DayFunnel]) -> None:
    for run_date, source in incoming.items():
        if run_date in target and "order_events.jsonl" in target[run_date].sources:
            continue
        target[run_date] = source


def build_report(days: dict[str, DayFunnel]) -> dict[str, Any]:
    total = DayFunnel("TOTAL")
    for day in days.values():
        total.counts.update(day.counts)
        total.failure_reasons.update(day.failure_reasons)
        total.exit_reasons.update(day.exit_reasons)
        total.total_pnl += day.total_pnl
        total.trade_count += day.trade_count
        total.position_notional += day.position_notional
        total.position_samples += day.position_samples
        total.sources.update(day.sources)
    return {"days": [days[key].as_dict() for key in sorted(days)], "total": total.as_dict()}


def render_markdown(report: dict[str, Any]) -> str:
    rows = [*report["days"], report["total"]]
    lines = [
        "| Date | Signals | Approved | Rejected | Attempts | Failed | Filled | Trades | PnL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {run_date} | {signal_detected} | {validation_approved} | {validation_rejected} | "
            "{execution_attempt} | {execution_failed} | {execution_succeeded} | {trade_count} | ${total_pnl:,.2f} |".format(**row)
        )
    total = report["total"]
    lines.extend(["", "Failure reasons: " + (", ".join(f"`{k}` ({v})" for k, v in total["failure_reasons"].items()) or "none recorded")])
    lines.append("Exit reasons: " + (", ".join(f"`{k}` ({v})" for k, v in total["exit_reasons"].items()) or "none recorded"))
    avg = total["average_position_size"]
    lines.append(f"Average entry position size: {'$' + format(avg, ',.2f') if avg is not None else 'n/a'}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/daily_news"))
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--supabase", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("data/daily_news/funnel_report.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    days = collect_local(args.data_root, args.start_date, args.end_date)
    if args.supabase:
        merge_days(days, collect_supabase(args.start_date, args.end_date))
    report = build_report(days)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
