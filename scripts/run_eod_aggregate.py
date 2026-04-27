#!/usr/bin/env python3
"""Aggregate daily backtest artifacts into Supabase serving tables."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2.extras import execute_values

LOGGER = logging.getLogger("run_eod_aggregate")


@dataclass
class RunRow:
    run_date: str
    total_trades: int
    closed_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float | None
    average_win: float | None
    average_loss: float | None
    final_equity: float | None
    return_pct: float | None
    daily_return_pct: float | None
    win_rate: float | None
    source_file: str


@dataclass
class TradeRow:
    trade_uid: str
    run_date: str
    order_id: str | None
    symbol: str
    side: str | None
    qty: int | None
    entry_price: float | None
    stop_loss: float | None
    take_profit: float | None
    entry_time: str | None
    exit_time: str | None
    exit_price: float | None
    exit_reason: str | None
    pnl: float | None
    status: str | None
    source_file: str


@dataclass
class HeartbeatRow:
    source_date: str
    heartbeat_ts: str
    loop_count: int | None
    symbols_tracked: int | None
    backtest: bool
    source_file: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate EOD artifacts to Supabase.")
    parser.add_argument("--date", help="Single date to process (YYYY-MM-DD).")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=3,
        help="Lookback window when --date is omitted (default: 3).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override data/daily_news directory.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow successful exit when no run directories or no ingestable rows are found.",
    )
    parser.add_argument("--validate", action="store_true", help="Run post-write checks.")
    return parser.parse_args()


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_iso(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text if "T" in text else text.replace(" ", "T")
    if not (normalized.endswith("Z") or "+" in normalized[10:]):
        normalized = f"{normalized}Z"
    try:
        return datetime.fromisoformat(normalized.replace("Z", "+00:00")).astimezone(
            timezone.utc
        ).isoformat()
    except ValueError:
        return None


def resolve_data_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env_dir = os.getenv("LLM_ADVISOR_DAILY_NEWS_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return (Path(__file__).resolve().parents[1] / "data" / "daily_news").resolve()


def collect_run_dirs(root: Path, date: str | None, lookback_days: int) -> list[tuple[str, Path]]:
    if not root.exists():
        return []
    if date:
        candidate = root / date
        return [(date, candidate)] if candidate.is_dir() else []

    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=lookback_days)).isoformat()
    run_dirs: list[tuple[str, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        run_date = child.name
        if len(run_date) != 10:
            continue
        if run_date >= cutoff:
            run_dirs.append((run_date, child))
    run_dirs.sort(key=lambda item: item[0])
    return run_dirs


def parse_backtest(run_date: str, backtest_path: Path) -> tuple[RunRow | None, list[TradeRow]]:
    if not backtest_path.exists():
        return None, []
    with backtest_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    run = RunRow(
        run_date=run_date,
        total_trades=_as_int(payload.get("total_trades")) or 0,
        closed_trades=_as_int(payload.get("closed_trades")) or 0,
        winning_trades=_as_int(payload.get("winning_trades")) or 0,
        losing_trades=_as_int(payload.get("losing_trades")) or 0,
        total_pnl=_as_float(payload.get("total_pnl")),
        average_win=_as_float(payload.get("average_win")),
        average_loss=_as_float(payload.get("average_loss")),
        final_equity=_as_float(payload.get("final_equity")),
        return_pct=_as_float(payload.get("return_pct")),
        daily_return_pct=_as_float(payload.get("daily_return_pct")),
        win_rate=_as_float(payload.get("win_rate")),
        source_file=str(backtest_path),
    )

    trade_rows: list[TradeRow] = []
    for idx, trade in enumerate(payload.get("trades", []), start=1):
        symbol = str(trade.get("symbol", "")).strip()
        if not symbol:
            continue
        order_id = str(trade.get("order_id")).strip() if trade.get("order_id") else None
        trade_uid = f"{run_date}:{order_id or f'idx-{idx}'}"
        trade_rows.append(
            TradeRow(
                trade_uid=trade_uid,
                run_date=run_date,
                order_id=order_id,
                symbol=symbol,
                side=trade.get("side"),
                qty=_as_int(trade.get("qty")),
                entry_price=_as_float(trade.get("entry_price")),
                stop_loss=_as_float(trade.get("stop_loss")),
                take_profit=_as_float(trade.get("take_profit")),
                entry_time=_as_iso(trade.get("entry_time")),
                exit_time=_as_iso(trade.get("exit_time")),
                exit_price=_as_float(trade.get("exit_price")),
                exit_reason=trade.get("exit_reason"),
                pnl=_as_float(trade.get("pnl")),
                status=trade.get("status"),
                source_file=str(backtest_path),
            )
        )
    return run, trade_rows


def parse_heartbeat(run_date: str, log_path: Path) -> HeartbeatRow | None:
    if not log_path.exists():
        return None
    latest: dict[str, Any] | None = None
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                latest = json.loads(line)
            except json.JSONDecodeError:
                continue
    if not latest:
        return None

    heartbeat_ts = _as_iso(latest.get("ts"))
    if not heartbeat_ts:
        return None
    symbols = latest.get("symbols")
    symbols_tracked = len(symbols) if isinstance(symbols, dict) else None
    return HeartbeatRow(
        source_date=run_date,
        heartbeat_ts=heartbeat_ts,
        loop_count=_as_int(latest.get("loop_count")),
        symbols_tracked=symbols_tracked,
        backtest=bool(latest.get("backtest")),
        source_file=str(log_path),
    )


def connect_supabase() -> psycopg2.extensions.connection:
    host = os.getenv("SUPABASE_DB_HOST")
    db = os.getenv("SUPABASE_DB_NAME", "postgres")
    user = os.getenv("SUPABASE_DB_USER", "postgres")
    port = int(os.getenv("SUPABASE_DB_PORT", "5432"))
    password = os.getenv("SUPABASE_DB_PASSWORD") or os.getenv("supabaseDBpass")

    if not host or not password:
        raise SystemExit("Missing Supabase DB credentials (host/password).")

    return psycopg2.connect(
        host=host,
        dbname=db,
        user=user,
        password=password,
        port=port,
        sslmode="require",
    )


def upsert_runs(cur, rows: list[RunRow], now_iso: str) -> int:
    if not rows:
        return 0
    values = [
        (
            row.run_date,
            row.total_trades,
            row.closed_trades,
            row.winning_trades,
            row.losing_trades,
            row.total_pnl,
            row.average_win,
            row.average_loss,
            row.final_equity,
            row.return_pct,
            row.daily_return_pct,
            row.win_rate,
            row.source_file,
            now_iso,
        )
        for row in rows
    ]
    execute_values(
        cur,
        """
        INSERT INTO llm_advisor_backtest_runs
        (run_date,total_trades,closed_trades,winning_trades,losing_trades,total_pnl,average_win,average_loss,final_equity,return_pct,daily_return_pct,win_rate,source_file,updated_at)
        VALUES %s
        ON CONFLICT (run_date) DO UPDATE SET
          total_trades = EXCLUDED.total_trades,
          closed_trades = EXCLUDED.closed_trades,
          winning_trades = EXCLUDED.winning_trades,
          losing_trades = EXCLUDED.losing_trades,
          total_pnl = EXCLUDED.total_pnl,
          average_win = EXCLUDED.average_win,
          average_loss = EXCLUDED.average_loss,
          final_equity = EXCLUDED.final_equity,
          return_pct = EXCLUDED.return_pct,
          daily_return_pct = EXCLUDED.daily_return_pct,
          win_rate = EXCLUDED.win_rate,
          source_file = EXCLUDED.source_file,
          updated_at = EXCLUDED.updated_at
        """,
        values,
    )
    return len(rows)


def upsert_trades(cur, rows: list[TradeRow], now_iso: str) -> int:
    if not rows:
        return 0
    values = [
        (
            row.trade_uid,
            row.run_date,
            row.order_id,
            row.symbol,
            row.side,
            row.qty,
            row.entry_price,
            row.stop_loss,
            row.take_profit,
            row.entry_time,
            row.exit_time,
            row.exit_price,
            row.exit_reason,
            row.pnl,
            row.status,
            row.source_file,
            now_iso,
        )
        for row in rows
    ]
    execute_values(
        cur,
        """
        INSERT INTO llm_advisor_backtest_trades
        (trade_uid,run_date,order_id,symbol,side,qty,entry_price,stop_loss,take_profit,entry_time,exit_time,exit_price,exit_reason,pnl,status,source_file,updated_at)
        VALUES %s
        ON CONFLICT (trade_uid) DO UPDATE SET
          run_date = EXCLUDED.run_date,
          order_id = EXCLUDED.order_id,
          symbol = EXCLUDED.symbol,
          side = EXCLUDED.side,
          qty = EXCLUDED.qty,
          entry_price = EXCLUDED.entry_price,
          stop_loss = EXCLUDED.stop_loss,
          take_profit = EXCLUDED.take_profit,
          entry_time = EXCLUDED.entry_time,
          exit_time = EXCLUDED.exit_time,
          exit_price = EXCLUDED.exit_price,
          exit_reason = EXCLUDED.exit_reason,
          pnl = EXCLUDED.pnl,
          status = EXCLUDED.status,
          source_file = EXCLUDED.source_file,
          updated_at = EXCLUDED.updated_at
        """,
        values,
    )
    return len(rows)


def upsert_heartbeats(cur, rows: list[HeartbeatRow], now_iso: str) -> int:
    if not rows:
        return 0
    values = [
        (
            row.source_date,
            row.heartbeat_ts,
            row.loop_count,
            row.symbols_tracked,
            row.backtest,
            row.source_file,
            now_iso,
        )
        for row in rows
    ]
    execute_values(
        cur,
        """
        INSERT INTO llm_advisor_runtime_heartbeats
        (source_date,heartbeat_ts,loop_count,symbols_tracked,backtest,source_file,updated_at)
        VALUES %s
        ON CONFLICT (source_date, heartbeat_ts) DO UPDATE SET
          loop_count = EXCLUDED.loop_count,
          symbols_tracked = EXCLUDED.symbols_tracked,
          backtest = EXCLUDED.backtest,
          source_file = EXCLUDED.source_file,
          updated_at = EXCLUDED.updated_at
        """,
        values,
    )
    return len(rows)


def validate(cur) -> dict[str, int]:
    checks: dict[str, int] = {}
    cur.execute("SELECT COUNT(*) FROM llm_advisor_backtest_runs WHERE run_date >= CURRENT_DATE - INTERVAL '7 days'")
    checks["runs_7d"] = int(cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM llm_advisor_backtest_trades WHERE run_date >= CURRENT_DATE - INTERVAL '7 days'")
    checks["trades_7d"] = int(cur.fetchone()[0])
    cur.execute(
        "SELECT COUNT(*) FROM llm_advisor_runtime_heartbeats WHERE heartbeat_ts >= NOW() - INTERVAL '7 days'"
    )
    checks["heartbeats_7d"] = int(cur.fetchone()[0])
    return checks


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    data_dir = resolve_data_dir(args.data_dir)
    run_dirs = collect_run_dirs(data_dir, args.date, args.lookback_days)
    if not run_dirs:
        message = f"No run directories found under {data_dir}"
        if args.allow_empty:
            LOGGER.warning("%s (allow-empty enabled)", message)
            return
        raise SystemExit(f"{message}. Failing fast to avoid silent-success EOD runs.")

    runs: list[RunRow] = []
    trades: list[TradeRow] = []
    heartbeats: list[HeartbeatRow] = []
    for run_date, run_dir in run_dirs:
        processed = run_dir / "processed"
        run_row, trade_rows = parse_backtest(run_date, processed / "backtest_results.json")
        if run_row:
            runs.append(run_row)
        trades.extend(trade_rows)
        heartbeat_row = parse_heartbeat(run_date, processed / "live_loop_log.jsonl")
        if heartbeat_row:
            heartbeats.append(heartbeat_row)

    LOGGER.info(
        "Prepared aggregate rows | runs=%d trades=%d heartbeats=%d",
        len(runs),
        len(trades),
        len(heartbeats),
    )
    if not (runs or trades or heartbeats):
        message = "No ingestable rows were parsed from located run directories"
        if args.allow_empty:
            LOGGER.warning("%s (allow-empty enabled)", message)
            return
        raise SystemExit(f"{message}. Failing fast to avoid empty EOD writes.")

    now_iso = datetime.now(timezone.utc).isoformat()
    conn = connect_supabase()
    try:
        with conn, conn.cursor() as cur:
            upsert_runs(cur, runs, now_iso)
            upsert_trades(cur, trades, now_iso)
            upsert_heartbeats(cur, heartbeats, now_iso)
            if args.validate:
                checks = validate(cur)
                LOGGER.info("Validation checks: %s", json.dumps(checks, sort_keys=True))
                if checks["runs_7d"] == 0:
                    raise SystemExit("Validation failed: no run rows in the last 7 days.")
            LOGGER.info(
                "EOD ingest complete | runs=%d trades=%d heartbeats=%d",
                len(runs),
                len(trades),
                len(heartbeats),
            )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
