#!/usr/bin/env python3
"""Aggregate daily backtest artifacts into Supabase serving tables."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    psycopg2 = None  # type: ignore[assignment]
    execute_values = None  # type: ignore[assignment]

from src.utils.daily_news_paths import normalize_daily_news_root
from src.utils.env_sanitize import getenv_strip

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
    underlying_symbol: str | None = None
    asset_class: str | None = None
    setup_type: str | None = None
    option_dte: int | None = None
    option_metadata: dict[str, Any] | None = None


@dataclass
class AccountSnapshotRow:
    snapshot_date: str
    captured_at: str
    equity: float | None
    last_equity: float | None
    buying_power: float | None
    daily_pnl: float | None
    daily_pnl_pct: float | None
    source: str


@dataclass
class HeartbeatRow:
    source_date: str
    heartbeat_ts: str
    loop_count: int | None
    symbols_tracked: int | None
    backtest: bool
    source_file: str


@dataclass
class OrderEventRow:
    event_uid: str
    run_date: str
    event_ts: str
    event_type: str
    symbol: str
    loop_count: int | None
    setup_type: str | None
    side: str | None
    entry_price: float | None
    z_score: float | None
    order_id: str | None
    details: dict[str, Any]
    source_file: str


@dataclass
class BrokerReconciliationRow:
    reconciliation_date: str
    booked_realized_pnl: float
    broker_daily_pnl: float | None
    pnl_gap: float | None
    lifecycle_exit_count: int
    tolerance: float
    status: str
    details: dict[str, Any]


@dataclass
class TradeLifecycleRow:
    lifecycle_uid: str
    entry_order_id: str | None
    exit_order_id: str | None
    symbol: str
    underlying_symbol: str | None
    opened_at: str | None
    closed_at: str | None
    filled_qty: float | None
    entry_fill_price: float | None
    exit_fill_price: float | None
    protective_stop_order_id: str | None
    protective_stop_price: float | None
    exit_reason: str | None
    realized_pnl: float | None
    status: str
    details: dict[str, Any]


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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse artifacts and BigQuery rows, then exit before Supabase writes.",
    )
    parser.add_argument(
        "--use-bigquery",
        dest="use_bigquery",
        action="store_true",
        default=None,
        help="Merge daily aggregates from BigQuery (dataset trades + live_loop_logs).",
    )
    parser.add_argument(
        "--no-bigquery",
        dest="use_bigquery",
        action="store_false",
        help="Disable BigQuery merge even if GCP_PROJECT_ID is set.",
    )
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


_OCC_SUFFIX_LEN = 15  # OCC option symbols end with YYMMDD[C|P]<8-digit strike>


def _underlying_from_occ(symbol: str) -> str | None:
    """SPY260620C00500000 -> SPY. Returns None when symbol isn't OCC-shaped."""
    text = str(symbol or "").strip().upper()
    if len(text) <= _OCC_SUFFIX_LEN:
        return None
    root, suffix = text[:-_OCC_SUFFIX_LEN], text[-_OCC_SUFFIX_LEN:]
    if not root.isalpha():
        return None
    if not (suffix[:6].isdigit() and suffix[6] in ("C", "P") and suffix[7:].isdigit()):
        return None
    return root


def _parse_option_metadata(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def derive_trade_enrichment(
    symbol: str,
    asset_class: Any,
    underlying_symbol: Any,
    setup_type: Any,
    option_metadata: Any,
) -> dict[str, Any]:
    """Normalize breakdown columns, falling back to option_metadata / OCC parsing."""
    meta = _parse_option_metadata(option_metadata)
    occ_underlying = _underlying_from_occ(symbol)

    asset = str(asset_class).strip().lower() if asset_class else None
    if not asset:
        asset = "option" if (meta or occ_underlying) else "stock"

    underlying = str(underlying_symbol).strip().upper() if underlying_symbol else None
    if not underlying and meta:
        meta_underlying = meta.get("underlying_symbol") or meta.get("underlying")
        underlying = str(meta_underlying).strip().upper() if meta_underlying else None
    if not underlying:
        underlying = occ_underlying if asset == "option" else str(symbol or "").strip().upper() or None

    dte = None
    if meta:
        dte = _as_int(meta.get("dte"))

    setup = str(setup_type).strip().upper() if setup_type else None

    return {
        "underlying_symbol": underlying,
        "asset_class": asset,
        "setup_type": setup,
        "option_dte": dte,
        "option_metadata": meta,
    }


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


def parse_daily_run_payload(run_date: str, processed: Path) -> tuple[RunRow | None, list[TradeRow]]:
    """Prefer backtest_results.json; fall back to session_summary.json for live/paper days."""
    for name in ("backtest_results.json", "session_summary.json"):
        path = processed / name
        if path.exists():
            return parse_backtest(run_date, path)
    return None, []


def _run_row_rank(row: RunRow) -> tuple:
    """Higher is better when merging artifact + BigQuery run summaries."""
    artifact = 0 if str(row.source_file).startswith("bq://") else 1
    return (
        row.closed_trades or 0,
        row.total_trades or 0,
        1 if row.final_equity is not None else 0,
        artifact,
        float(row.total_pnl or 0),
    )


def dedupe_runs(rows: list[RunRow]) -> list[RunRow]:
    by_date: dict[str, RunRow] = {}
    for r in rows:
        prev = by_date.get(r.run_date)
        if prev is None or _run_row_rank(r) > _run_row_rank(prev):
            by_date[r.run_date] = r
    return sorted(by_date.values(), key=lambda x: x.run_date)


def _trade_row_rank(row: TradeRow) -> tuple:
    artifact = 0 if str(row.source_file).startswith("bq://") else 1
    return (
        1 if row.exit_price is not None else 0,
        1 if row.exit_time is not None else 0,
        1 if row.setup_type is not None else 0,
        1 if row.option_metadata is not None else 0,
        artifact,
        float(row.pnl or 0),
    )


def dedupe_trades(rows: list[TradeRow]) -> list[TradeRow]:
    by_uid: dict[str, TradeRow] = {}
    for r in rows:
        prev = by_uid.get(r.trade_uid)
        if prev is None or _trade_row_rank(r) > _trade_row_rank(prev):
            by_uid[r.trade_uid] = r
    return list(by_uid.values())


def dedupe_heartbeats(rows: list[HeartbeatRow]) -> list[HeartbeatRow]:
    by_day: dict[str, HeartbeatRow] = {}
    for r in rows:
        prev = by_day.get(r.source_date)
        if prev is None or r.heartbeat_ts > prev.heartbeat_ts:
            by_day[r.source_date] = r
    return sorted(by_day.values(), key=lambda x: x.source_date)


def dedupe_account_snapshots(rows: list[AccountSnapshotRow]) -> list[AccountSnapshotRow]:
    by_key: dict[tuple[str, str], AccountSnapshotRow] = {}
    for row in rows:
        by_key[(row.snapshot_date, row.captured_at)] = row
    return sorted(by_key.values(), key=lambda x: (x.snapshot_date, x.captured_at))


def backfill_run_equity_from_snapshots(
    runs: list[RunRow], snapshots: list[AccountSnapshotRow]
) -> list[RunRow]:
    """Use the latest same-day account snapshot for missing cohort equity."""
    latest: dict[str, AccountSnapshotRow] = {}
    for snapshot in snapshots:
        if snapshot.equity is None:
            continue
        previous = latest.get(snapshot.snapshot_date)
        if previous is None or snapshot.captured_at > previous.captured_at:
            latest[snapshot.snapshot_date] = snapshot

    return [
        replace(
            run,
            final_equity=latest[run.run_date].equity,
        )
        if run.final_equity is None and run.run_date in latest
        else run
        for run in runs
    ]


def dedupe_order_events(rows: list[OrderEventRow]) -> list[OrderEventRow]:
    by_uid: dict[str, OrderEventRow] = {}
    for row in rows:
        by_uid[row.event_uid] = row
    return sorted(by_uid.values(), key=lambda x: (x.run_date, x.event_ts, x.event_uid))


def build_broker_reconciliations(
    run_dates: list[str],
    order_events: list[OrderEventRow],
    account_snapshots: list[AccountSnapshotRow],
    tolerance: float = 50.0,
) -> list[BrokerReconciliationRow]:
    """Reconcile final flat-account daily PnL to actual lifecycle fill events."""
    fills_by_date: dict[str, list[OrderEventRow]] = {}
    requests_by_date: dict[str, list[OrderEventRow]] = {}
    for event in order_events:
        if event.event_type == "option_exit_filled":
            fills_by_date.setdefault(event.run_date, []).append(event)
        elif event.event_type == "option_exit_requested":
            requests_by_date.setdefault(event.run_date, []).append(event)

    latest_snapshot: dict[str, AccountSnapshotRow] = {}
    for snapshot in account_snapshots:
        previous = latest_snapshot.get(snapshot.snapshot_date)
        if previous is None or snapshot.captured_at > previous.captured_at:
            latest_snapshot[snapshot.snapshot_date] = snapshot

    rows: list[BrokerReconciliationRow] = []
    for run_date in sorted(set(run_dates)):
        events = fills_by_date.get(run_date, [])
        source = "option_exit_filled"
        if not events:
            # Backward-compatible estimate for days recorded before fill events existed.
            events = requests_by_date.get(run_date, [])
            source = "option_exit_requested_estimate"
        booked = 0.0
        for event in events:
            key = "realized_pnl" if source == "option_exit_filled" else "unrealized_pl"
            booked += _as_float(event.details.get(key)) or 0.0

        snapshot = latest_snapshot.get(run_date)
        final_exit_ts = max((event.event_ts for event in events), default=None)
        snapshot_is_final = bool(
            snapshot
            and (
                final_exit_ts is None
                or snapshot.captured_at >= final_exit_ts
            )
        )
        broker_pnl = snapshot.daily_pnl if snapshot_is_final and snapshot else None
        gap = (broker_pnl - booked) if broker_pnl is not None else None
        status = "pending"
        if gap is not None:
            status = "ok" if abs(gap) <= tolerance else "alert"
        rows.append(
            BrokerReconciliationRow(
                reconciliation_date=run_date,
                booked_realized_pnl=booked,
                broker_daily_pnl=broker_pnl,
                pnl_gap=gap,
                lifecycle_exit_count=len(events),
                tolerance=tolerance,
                status=status,
                details={
                    "booked_source": source,
                    "snapshot_captured_at": snapshot.captured_at if snapshot else None,
                    "final_exit_at": final_exit_ts,
                    "snapshot_after_final_exit": snapshot_is_final,
                    "flat_account_required": True,
                },
            )
        )
    return rows


def build_trade_lifecycles(
    order_events: list[OrderEventRow],
) -> list[TradeLifecycleRow]:
    """Build one broker-position lifecycle per protected, non-aggregated entry."""
    by_uid: dict[str, TradeLifecycleRow] = {}
    symbol_open_uid: dict[str, str] = {}

    for event in sorted(order_events, key=lambda row: row.event_ts):
        if event.event_type != "option_protective_stop_submitted":
            continue
        details = event.details
        entry_order_id = str(details.get("entry_order_id") or "").strip() or None
        uid = entry_order_id or f"{event.run_date}:{event.symbol}:{event.event_ts}"
        row = TradeLifecycleRow(
            lifecycle_uid=uid,
            entry_order_id=entry_order_id,
            exit_order_id=None,
            symbol=event.symbol,
            underlying_symbol=_underlying_from_occ(event.symbol),
            opened_at=event.event_ts,
            closed_at=None,
            filled_qty=_as_float(details.get("actual_filled_qty")),
            entry_fill_price=_as_float(details.get("actual_entry_price")),
            exit_fill_price=None,
            protective_stop_order_id=str(
                details.get("stop_order_id") or ""
            ).strip()
            or None,
            protective_stop_price=_as_float(details.get("stop_price")),
            exit_reason=None,
            realized_pnl=None,
            status="open",
            details={"entry_event_uid": event.event_uid},
        )
        by_uid[uid] = row
        symbol_open_uid[event.symbol] = uid

    for event in sorted(order_events, key=lambda row: row.event_ts):
        if event.event_type != "option_partial_exit_filled":
            continue
        details = event.details
        state = details.get("tiered_exit_state") if isinstance(details.get("tiered_exit_state"), dict) else {}
        entry_order_id = str(details.get("entry_order_id") or "").strip() or None
        uid = str(entry_order_id or symbol_open_uid.get(event.symbol) or state.get("lifecycle_id") or "")
        if not uid:
            uid = f"{event.run_date}:{event.symbol}:reconciled"
        row = by_uid.get(uid)
        if row is None:
            row = TradeLifecycleRow(
                lifecycle_uid=uid,
                entry_order_id=entry_order_id,
                exit_order_id=None,
                symbol=event.symbol,
                underlying_symbol=_underlying_from_occ(event.symbol),
                opened_at=None,
                closed_at=None,
                filled_qty=None,
                entry_fill_price=None,
                exit_fill_price=None,
                protective_stop_order_id=None,
                protective_stop_price=None,
                exit_reason=None,
                realized_pnl=0.0,
                status="open",
                details={},
            )
            by_uid[uid] = row
        row.details.setdefault("tiered_partial_fills", []).append(
            {
                "event_uid": event.event_uid,
                "event_ts": event.event_ts,
                "stage": details.get("stage"),
                "filled_qty": _as_float(details.get("filled_qty")),
                "filled_avg_price": _as_float(details.get("filled_avg_price")),
                "realized_pnl": _as_float(details.get("realized_pnl")),
            }
        )

    for event in sorted(order_events, key=lambda row: row.event_ts):
        if event.event_type != "option_exit_filled":
            continue
        details = event.details
        entry_order_id = str(details.get("entry_order_id") or "").strip() or None
        uid = entry_order_id or symbol_open_uid.get(event.symbol)
        if uid is None:
            uid = f"{event.run_date}:{event.symbol}:reconciled"
        row = by_uid.get(uid)
        if row is None:
            position = details.get("position") if isinstance(details.get("position"), dict) else {}
            row = TradeLifecycleRow(
                lifecycle_uid=uid,
                entry_order_id=entry_order_id,
                exit_order_id=None,
                symbol=event.symbol,
                underlying_symbol=_underlying_from_occ(event.symbol),
                opened_at=None,
                closed_at=None,
                filled_qty=_as_float(
                    details.get("actual_filled_qty") or position.get("qty")
                ),
                entry_fill_price=_as_float(position.get("entry_price")),
                exit_fill_price=None,
                protective_stop_order_id=None,
                protective_stop_price=None,
                exit_reason=None,
                realized_pnl=None,
                status="open",
                details={},
            )
            by_uid[uid] = row

        exit_order = (
            details.get("exit_order")
            if isinstance(details.get("exit_order"), dict)
            else {}
        )
        row.exit_order_id = str(exit_order.get("order_id") or "").strip() or None
        row.closed_at = str(exit_order.get("filled_at") or event.event_ts)
        row.exit_fill_price = _as_float(details.get("actual_exit_price"))
        row.filled_qty = _as_float(details.get("actual_filled_qty")) or row.filled_qty
        row.exit_reason = str(details.get("reason") or "position_closed")
        row.realized_pnl = _as_float(details.get("realized_pnl"))
        row.status = "closed"
        row.details.update(
            {
                "exit_event_uid": event.event_uid,
                "exit_order_status": exit_order.get("status"),
                "protective_stop_fill": bool(exit_order.get("is_protective_stop")),
                "initial_qty": _as_float(
                    (details.get("tiered_exit_state") or {}).get("initial_qty")
                )
                if isinstance(details.get("tiered_exit_state"), dict)
                else row.details.get("initial_qty"),
                "final_exit_qty": max(
                    0.0,
                    (_as_float(details.get("actual_filled_qty")) or 0.0)
                    - sum(
                        (_as_float(fill.get("filled_qty")) or 0.0)
                        for fill in row.details.get("tiered_partial_fills", [])
                        if isinstance(fill, dict)
                    ),
                ),
                "final_exit_price": _as_float(details.get("actual_exit_price")),
                "tiered_exit_state": details.get("tiered_exit_state"),
            }
        )

    return sorted(by_uid.values(), key=lambda row: (row.opened_at or "", row.lifecycle_uid))


def _ts_to_iso(val: Any) -> str | None:
    if val is None:
        return None
    if hasattr(val, "isoformat"):
        try:
            return val.isoformat()  # type: ignore[no-any-return]
        except TypeError:
            pass
    return _as_iso(val)


def fetch_bq_ingest_for_dates(
    project_id: str, dataset_id: str, run_dates: list[str]
) -> tuple[list[RunRow], list[TradeRow], list[HeartbeatRow]]:
    """Load RunRow / TradeRow / HeartbeatRow from BigQuery `trades` and `live_loop_logs`."""
    from google.cloud import bigquery

    if not run_dates:
        return [], [], []

    client = bigquery.Client(project=project_id)
    fq = f"`{project_id}.{dataset_id}.trades`"
    fq_logs = f"`{project_id}.{dataset_id}.live_loop_logs`"

    # Enrichment columns may predate the live-loop deploy that stamps them.
    try:
        client.query(f"ALTER TABLE {fq} ADD COLUMN IF NOT EXISTS setup_type STRING").result()
    except Exception as exc:
        LOGGER.warning("Could not ensure trades.setup_type column exists: %s", exc)

    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("dates", "STRING", run_dates)]
    )

    agg_sql = f"""
    SELECT
      FORMAT_DATE('%Y-%m-%d', DATE(entry_time, 'America/New_York')) AS run_date,
      COUNT(*) AS total_trades,
      COUNTIF(exit_time IS NOT NULL) AS closed_trades,
      COUNTIF(exit_time IS NOT NULL AND pnl > 0) AS winning_trades,
      COUNTIF(exit_time IS NOT NULL AND pnl <= 0 AND pnl IS NOT NULL) AS losing_trades,
      SUM(IFNULL(pnl, 0)) AS total_pnl,
      AVG(IF((pnl > 0), pnl, NULL)) AS average_win,
      AVG(IF((pnl <= 0), pnl, NULL)) AS average_loss,
      SAFE_DIVIDE(
        COUNTIF(exit_time IS NOT NULL AND pnl > 0),
        NULLIF(COUNTIF(exit_time IS NOT NULL), 0)
      ) AS win_rate
    FROM {fq}
    WHERE entry_time IS NOT NULL
      AND FORMAT_DATE('%Y-%m-%d', DATE(entry_time, 'America/New_York')) IN UNNEST(@dates)
    GROUP BY run_date
    """

    fq_signals = f"`{project_id}.{dataset_id}.trade_signals`"
    # setup_type is stamped at execution time going forward; backfill older rows
    # from the nearest preceding trade_signal on the same underlying (<=15 min).
    trades_sql = f"""
    WITH filtered AS (
      SELECT *
      FROM {fq}
      WHERE entry_time IS NOT NULL
        AND FORMAT_DATE('%Y-%m-%d', DATE(entry_time, 'America/New_York')) IN UNNEST(@dates)
    ),
    joined AS (
      SELECT
        t.*,
        s.setup_type AS signal_setup_type,
        ROW_NUMBER() OVER (
          PARTITION BY t.id
          ORDER BY s.timestamp DESC
        ) AS signal_rank
      FROM filtered t
      LEFT JOIN {fq_signals} s
        ON s.symbol = COALESCE(t.underlying_symbol, t.symbol)
       AND s.timestamp <= t.entry_time
       AND s.timestamp >= TIMESTAMP_SUB(t.entry_time, INTERVAL 15 MINUTE)
    )
    SELECT
      id,
      trade_id,
      symbol,
      side,
      entry_price,
      stop_loss,
      take_profit,
      qty,
      status,
      entry_time,
      exit_time,
      exit_price,
      pnl,
      exit_reason,
      asset_class,
      underlying_symbol,
      option_metadata,
      COALESCE(SAFE_CAST(setup_type AS STRING), signal_setup_type) AS setup_type,
      FORMAT_DATE('%Y-%m-%d', DATE(entry_time, 'America/New_York')) AS run_date
    FROM joined
    WHERE signal_rank = 1
    """

    hb_sql = f"""
    SELECT
      FORMAT_DATE('%Y-%m-%d', DATE(timestamp, 'America/New_York')) AS source_date,
      MAX(timestamp) AS heartbeat_ts,
      COUNT(DISTINCT symbol) AS symbols_tracked
    FROM {fq_logs}
    WHERE timestamp IS NOT NULL
      AND FORMAT_DATE('%Y-%m-%d', DATE(timestamp, 'America/New_York')) IN UNNEST(@dates)
    GROUP BY source_date
    """

    runs: list[RunRow] = []
    for row in client.query(agg_sql, job_config=job_config).result():
        rd = row.run_date.isoformat() if hasattr(row.run_date, "isoformat") else str(row.run_date)
        runs.append(
            RunRow(
                run_date=rd,
                total_trades=int(row.total_trades or 0),
                closed_trades=int(row.closed_trades or 0),
                winning_trades=int(row.winning_trades or 0),
                losing_trades=int(row.losing_trades or 0),
                total_pnl=float(row.total_pnl) if row.total_pnl is not None else None,
                average_win=float(row.average_win) if row.average_win is not None else None,
                average_loss=float(row.average_loss) if row.average_loss is not None else None,
                final_equity=None,
                return_pct=None,
                daily_return_pct=None,
                win_rate=float(row.win_rate) if row.win_rate is not None else None,
                source_file=f"bq://{project_id}.{dataset_id}.trades",
            )
        )

    trades_out: list[TradeRow] = []
    for row in client.query(trades_sql, job_config=job_config).result():
        rd = row.run_date.isoformat() if hasattr(row.run_date, "isoformat") else str(row.run_date)
        tid = str(row.trade_id).strip() if row.trade_id else str(row.id)
        trade_uid = f"{rd}:{tid}"
        symbol = str(row.symbol or "").strip()
        enrichment = derive_trade_enrichment(
            symbol=symbol,
            asset_class=row.asset_class,
            underlying_symbol=row.underlying_symbol,
            setup_type=row.setup_type,
            option_metadata=row.option_metadata,
        )
        trades_out.append(
            TradeRow(
                trade_uid=trade_uid,
                run_date=rd,
                order_id=str(row.trade_id).strip() if row.trade_id else None,
                symbol=symbol,
                side=str(row.side) if row.side else None,
                qty=_as_int(row.qty),
                entry_price=float(row.entry_price) if row.entry_price is not None else None,
                stop_loss=float(row.stop_loss) if row.stop_loss is not None else None,
                take_profit=float(row.take_profit) if row.take_profit is not None else None,
                entry_time=_ts_to_iso(row.entry_time),
                exit_time=_ts_to_iso(row.exit_time),
                exit_price=float(row.exit_price) if row.exit_price is not None else None,
                exit_reason=str(row.exit_reason) if row.exit_reason else None,
                pnl=float(row.pnl) if row.pnl is not None else None,
                status=str(row.status) if row.status else None,
                source_file=f"bq://{project_id}.{dataset_id}.trades",
                **enrichment,
            )
        )

    heartbeats: list[HeartbeatRow] = []
    for row in client.query(hb_sql, job_config=job_config).result():
        hb_ts = _ts_to_iso(row.heartbeat_ts)
        if not hb_ts:
            continue
        heartbeats.append(
            HeartbeatRow(
                source_date=row.source_date.isoformat()
                if hasattr(row.source_date, "isoformat")
                else str(row.source_date),
                heartbeat_ts=hb_ts,
                loop_count=None,
                symbols_tracked=int(row.symbols_tracked) if row.symbols_tracked is not None else None,
                backtest=False,
                source_file=f"bq://{project_id}.{dataset_id}.live_loop_logs",
            )
        )

    return runs, trades_out, heartbeats


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
        enrichment = derive_trade_enrichment(
            symbol=symbol,
            asset_class=trade.get("asset_class"),
            underlying_symbol=trade.get("underlying_symbol"),
            setup_type=trade.get("setup_type"),
            option_metadata=trade.get("option_metadata"),
        )
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
                **enrichment,
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


def parse_account_snapshots(run_date: str, snapshot_path: Path) -> list[AccountSnapshotRow]:
    """Read processed/account_snapshot.json (written by the live loop) into rows."""
    if not snapshot_path.exists():
        return []
    try:
        with snapshot_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    raw_rows = payload.get("snapshots") if isinstance(payload, dict) else None
    if not isinstance(raw_rows, list):
        return []

    rows: list[AccountSnapshotRow] = []
    for raw in raw_rows:
        if not isinstance(raw, dict):
            continue
        captured_at = _as_iso(raw.get("captured_at"))
        if not captured_at:
            continue
        rows.append(
            AccountSnapshotRow(
                snapshot_date=run_date,
                captured_at=captured_at,
                equity=_as_float(raw.get("equity")),
                last_equity=_as_float(raw.get("last_equity")),
                buying_power=_as_float(raw.get("buying_power")),
                daily_pnl=_as_float(raw.get("daily_pnl")),
                daily_pnl_pct=_as_float(raw.get("daily_pnl_pct")),
                source=str(raw.get("source") or "alpaca_paper"),
            )
        )
    return rows


def parse_order_events(run_date: str, events_path: Path) -> list[OrderEventRow]:
    if not events_path.exists():
        return []

    rows: list[OrderEventRow] = []
    with events_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue

            event_ts = _as_iso(payload.get("ts"))
            event_type = str(payload.get("event_type") or "").strip()
            symbol = str(payload.get("symbol") or "").strip()
            if not event_ts or not event_type or not symbol:
                continue

            signal = payload.get("signal")
            if not isinstance(signal, dict):
                signal = {}
            details = payload.get("details")
            if not isinstance(details, dict):
                details = {}
            else:
                details = dict(details)
            for context_key in ("signal_context", "trade_plan"):
                context_value = payload.get(context_key)
                if isinstance(context_value, dict):
                    details.setdefault(context_key, context_value)
            order = details.get("order")
            order_id = None
            if isinstance(order, dict):
                order_id = str(order.get("order_id") or "").strip() or None
            if not order_id:
                order_id = str(
                    details.get("entry_order_id")
                    or details.get("exit_order_id")
                    or ""
                ).strip() or None

            event_uid = f"{run_date}:{event_ts}:{event_type}:{symbol}:{idx}"
            rows.append(
                OrderEventRow(
                    event_uid=event_uid,
                    run_date=run_date,
                    event_ts=event_ts,
                    event_type=event_type,
                    symbol=symbol,
                    loop_count=_as_int(payload.get("loop_count")),
                    setup_type=str(signal.get("setup_type") or "").strip() or None,
                    side=str(signal.get("side") or "").strip() or None,
                    entry_price=_as_float(signal.get("entry_price")),
                    z_score=_as_float(signal.get("z_score")),
                    order_id=order_id,
                    details=details,
                    source_file=str(events_path),
                )
            )
    return rows


def run_row_from_heartbeat(run_date: str, heartbeat: HeartbeatRow) -> RunRow:
    return RunRow(
        run_date=run_date,
        total_trades=0,
        closed_trades=0,
        winning_trades=0,
        losing_trades=0,
        total_pnl=0.0,
        average_win=0.0,
        average_loss=0.0,
        final_equity=None,
        return_pct=None,
        daily_return_pct=None,
        win_rate=None,
        source_file=heartbeat.source_file,
    )


def run_row_from_order_events(run_date: str, events: list[OrderEventRow]) -> RunRow:
    return RunRow(
        run_date=run_date,
        total_trades=0,
        closed_trades=0,
        winning_trades=0,
        losing_trades=0,
        total_pnl=0.0,
        average_win=0.0,
        average_loss=0.0,
        final_equity=None,
        return_pct=None,
        daily_return_pct=None,
        win_rate=None,
        source_file=events[0].source_file if events else "",
    )


def connect_supabase() -> Any:
    if psycopg2 is None:
        raise SystemExit("Missing psycopg2. Install requirements before running EOD Supabase sync.")

    host = getenv_strip("SUPABASE_DB_HOST")
    db = getenv_strip("SUPABASE_DB_NAME") or "postgres"
    user = getenv_strip("SUPABASE_DB_USER") or "postgres"
    port_raw = getenv_strip("SUPABASE_DB_PORT") or "5432"
    try:
        port = int(port_raw)
    except ValueError as exc:
        raise SystemExit(f"Invalid SUPABASE_DB_PORT: {port_raw!r}") from exc
    password = getenv_strip("SUPABASE_DB_PASSWORD") or getenv_strip("supabaseDBpass")

    if not host or not password:
        raise SystemExit(
            "Missing Supabase DB credentials. Set GitHub secrets: "
            "SUPABASE_DB_HOST, SUPABASE_DB_PASSWORD (and optionally "
            "SUPABASE_DB_NAME, SUPABASE_DB_USER, SUPABASE_DB_PORT)."
        )

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
    if execute_values is None:
        raise SystemExit("Missing psycopg2. Install requirements before running EOD Supabase sync.")
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
          final_equity = COALESCE(EXCLUDED.final_equity, llm_advisor_backtest_runs.final_equity),
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
    if execute_values is None:
        raise SystemExit("Missing psycopg2. Install requirements before running EOD Supabase sync.")
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
            row.underlying_symbol,
            row.asset_class,
            row.setup_type,
            row.option_dte,
            json.dumps(row.option_metadata, sort_keys=True) if row.option_metadata is not None else None,
            row.source_file,
            now_iso,
        )
        for row in rows
    ]
    execute_values(
        cur,
        """
        INSERT INTO llm_advisor_backtest_trades
        (trade_uid,run_date,order_id,symbol,side,qty,entry_price,stop_loss,take_profit,entry_time,exit_time,exit_price,exit_reason,pnl,status,underlying_symbol,asset_class,setup_type,option_dte,option_metadata,source_file,updated_at)
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
          underlying_symbol = COALESCE(EXCLUDED.underlying_symbol, llm_advisor_backtest_trades.underlying_symbol),
          asset_class = COALESCE(EXCLUDED.asset_class, llm_advisor_backtest_trades.asset_class),
          setup_type = COALESCE(EXCLUDED.setup_type, llm_advisor_backtest_trades.setup_type),
          option_dte = COALESCE(EXCLUDED.option_dte, llm_advisor_backtest_trades.option_dte),
          option_metadata = COALESCE(EXCLUDED.option_metadata, llm_advisor_backtest_trades.option_metadata),
          source_file = EXCLUDED.source_file,
          updated_at = EXCLUDED.updated_at
        """,
        values,
    )
    return len(rows)


def upsert_heartbeats(cur, rows: list[HeartbeatRow], now_iso: str) -> int:
    if not rows:
        return 0
    if execute_values is None:
        raise SystemExit("Missing psycopg2. Install requirements before running EOD Supabase sync.")
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


def upsert_account_snapshots(cur, rows: list[AccountSnapshotRow], now_iso: str) -> int:
    if not rows:
        return 0
    if execute_values is None:
        raise SystemExit("Missing psycopg2. Install requirements before running EOD Supabase sync.")
    values = [
        (
            row.snapshot_date,
            row.captured_at,
            row.equity,
            row.last_equity,
            row.buying_power,
            row.daily_pnl,
            row.daily_pnl_pct,
            row.source,
            now_iso,
        )
        for row in rows
    ]
    execute_values(
        cur,
        """
        INSERT INTO llm_advisor_account_snapshots
        (snapshot_date,captured_at,equity,last_equity,buying_power,daily_pnl,daily_pnl_pct,source,updated_at)
        VALUES %s
        ON CONFLICT (snapshot_date, captured_at) DO UPDATE SET
          equity = EXCLUDED.equity,
          last_equity = EXCLUDED.last_equity,
          buying_power = EXCLUDED.buying_power,
          daily_pnl = EXCLUDED.daily_pnl,
          daily_pnl_pct = EXCLUDED.daily_pnl_pct,
          source = EXCLUDED.source,
          updated_at = EXCLUDED.updated_at
        """,
        values,
    )
    return len(rows)


def upsert_order_events(cur, rows: list[OrderEventRow], now_iso: str) -> int:
    if not rows:
        return 0
    if execute_values is None:
        raise SystemExit("Missing psycopg2. Install requirements before running EOD Supabase sync.")
    values = [
        (
            row.event_uid,
            row.run_date,
            row.event_ts,
            row.event_type,
            row.symbol,
            row.loop_count,
            row.setup_type,
            row.side,
            row.entry_price,
            row.z_score,
            row.order_id,
            json.dumps(row.details, sort_keys=True),
            row.source_file,
            now_iso,
        )
        for row in rows
    ]
    execute_values(
        cur,
        """
        INSERT INTO llm_advisor_order_events
        (event_uid,run_date,event_ts,event_type,symbol,loop_count,setup_type,side,entry_price,z_score,order_id,details,source_file,updated_at)
        VALUES %s
        ON CONFLICT (event_uid) DO UPDATE SET
          run_date = EXCLUDED.run_date,
          event_ts = EXCLUDED.event_ts,
          event_type = EXCLUDED.event_type,
          symbol = EXCLUDED.symbol,
          loop_count = EXCLUDED.loop_count,
          setup_type = EXCLUDED.setup_type,
          side = EXCLUDED.side,
          entry_price = EXCLUDED.entry_price,
          z_score = EXCLUDED.z_score,
          order_id = EXCLUDED.order_id,
          details = EXCLUDED.details,
          source_file = EXCLUDED.source_file,
          updated_at = EXCLUDED.updated_at
        """,
        values,
    )
    return len(rows)


def upsert_broker_reconciliations(
    cur,
    rows: list[BrokerReconciliationRow],
    now_iso: str,
) -> int:
    if not rows:
        return 0
    if execute_values is None:
        raise SystemExit("Missing psycopg2. Install requirements before running EOD Supabase sync.")
    values = [
        (
            row.reconciliation_date,
            row.booked_realized_pnl,
            row.broker_daily_pnl,
            row.pnl_gap,
            row.lifecycle_exit_count,
            row.tolerance,
            row.status,
            json.dumps(row.details, sort_keys=True),
            now_iso,
        )
        for row in rows
    ]
    execute_values(
        cur,
        """
        INSERT INTO llm_advisor_broker_reconciliation_daily
        (reconciliation_date,booked_realized_pnl,broker_daily_pnl,pnl_gap,
         lifecycle_exit_count,tolerance,status,details,updated_at)
        VALUES %s
        ON CONFLICT (reconciliation_date) DO UPDATE SET
          booked_realized_pnl = EXCLUDED.booked_realized_pnl,
          broker_daily_pnl = EXCLUDED.broker_daily_pnl,
          pnl_gap = EXCLUDED.pnl_gap,
          lifecycle_exit_count = EXCLUDED.lifecycle_exit_count,
          tolerance = EXCLUDED.tolerance,
          status = EXCLUDED.status,
          details = EXCLUDED.details,
          updated_at = EXCLUDED.updated_at
        """,
        values,
    )
    return len(rows)


def upsert_trade_lifecycles(
    cur,
    rows: list[TradeLifecycleRow],
    now_iso: str,
) -> int:
    if not rows:
        return 0
    if execute_values is None:
        raise SystemExit("Missing psycopg2. Install requirements before running EOD Supabase sync.")
    values = [
        (
            row.lifecycle_uid,
            row.entry_order_id,
            row.exit_order_id,
            row.symbol,
            row.underlying_symbol,
            row.opened_at,
            row.closed_at,
            row.filled_qty,
            row.entry_fill_price,
            row.exit_fill_price,
            row.protective_stop_order_id,
            row.protective_stop_price,
            row.exit_reason,
            row.realized_pnl,
            row.status,
            json.dumps(row.details, sort_keys=True),
            now_iso,
        )
        for row in rows
    ]
    execute_values(
        cur,
        """
        INSERT INTO llm_advisor_trade_lifecycles
        (lifecycle_uid,entry_order_id,exit_order_id,symbol,underlying_symbol,
         opened_at,closed_at,filled_qty,entry_fill_price,exit_fill_price,
         protective_stop_order_id,protective_stop_price,exit_reason,realized_pnl,
         status,details,updated_at)
        VALUES %s
        ON CONFLICT (lifecycle_uid) DO UPDATE SET
          entry_order_id = COALESCE(EXCLUDED.entry_order_id, llm_advisor_trade_lifecycles.entry_order_id),
          exit_order_id = COALESCE(EXCLUDED.exit_order_id, llm_advisor_trade_lifecycles.exit_order_id),
          symbol = EXCLUDED.symbol,
          underlying_symbol = COALESCE(EXCLUDED.underlying_symbol, llm_advisor_trade_lifecycles.underlying_symbol),
          opened_at = COALESCE(EXCLUDED.opened_at, llm_advisor_trade_lifecycles.opened_at),
          closed_at = COALESCE(EXCLUDED.closed_at, llm_advisor_trade_lifecycles.closed_at),
          filled_qty = COALESCE(EXCLUDED.filled_qty, llm_advisor_trade_lifecycles.filled_qty),
          entry_fill_price = COALESCE(EXCLUDED.entry_fill_price, llm_advisor_trade_lifecycles.entry_fill_price),
          exit_fill_price = COALESCE(EXCLUDED.exit_fill_price, llm_advisor_trade_lifecycles.exit_fill_price),
          protective_stop_order_id = COALESCE(EXCLUDED.protective_stop_order_id, llm_advisor_trade_lifecycles.protective_stop_order_id),
          protective_stop_price = COALESCE(EXCLUDED.protective_stop_price, llm_advisor_trade_lifecycles.protective_stop_price),
          exit_reason = COALESCE(EXCLUDED.exit_reason, llm_advisor_trade_lifecycles.exit_reason),
          realized_pnl = COALESCE(EXCLUDED.realized_pnl, llm_advisor_trade_lifecycles.realized_pnl),
          status = EXCLUDED.status,
          details = llm_advisor_trade_lifecycles.details || EXCLUDED.details,
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
    cur.execute("SELECT COUNT(*) FROM llm_advisor_order_events WHERE run_date >= CURRENT_DATE - INTERVAL '7 days'")
    checks["order_events_7d"] = int(cur.fetchone()[0])
    return checks


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    data_dir = normalize_daily_news_root(resolve_data_dir(args.data_dir))
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
    order_events: list[OrderEventRow] = []
    account_snapshots: list[AccountSnapshotRow] = []
    for run_date, run_dir in run_dirs:
        processed = run_dir / "processed"
        run_row, trade_rows = parse_daily_run_payload(run_date, processed)
        heartbeat_row = parse_heartbeat(run_date, processed / "live_loop_log.jsonl")
        event_rows = parse_order_events(run_date, processed / "order_events.jsonl")
        snapshot_rows = parse_account_snapshots(run_date, processed / "account_snapshot.json")
        if run_row:
            runs.append(run_row)
        elif heartbeat_row:
            runs.append(run_row_from_heartbeat(run_date, heartbeat_row))
        elif event_rows:
            runs.append(run_row_from_order_events(run_date, event_rows))
        trades.extend(trade_rows)
        if heartbeat_row:
            heartbeats.append(heartbeat_row)
        order_events.extend(event_rows)
        account_snapshots.extend(snapshot_rows)

    use_bq = args.use_bigquery
    if use_bq is None:
        use_bq = bool(os.getenv("GCP_PROJECT_ID", "").strip())

    run_dates_list = [d for d, _ in run_dirs]
    if use_bq:
        project_id = os.getenv("GCP_PROJECT_ID", "").strip()
        dataset_id = getenv_strip("GCP_DATASET_ID") or "trading_signals"
        if project_id:
            try:
                bq_runs, bq_trades, bq_hb = fetch_bq_ingest_for_dates(
                    project_id, dataset_id, run_dates_list
                )
                LOGGER.info(
                    "BigQuery merge | bq_runs=%d bq_trades=%d bq_heartbeats=%d",
                    len(bq_runs),
                    len(bq_trades),
                    len(bq_hb),
                )
                runs.extend(bq_runs)
                trades.extend(bq_trades)
                heartbeats.extend(bq_hb)
            except Exception as exc:
                LOGGER.warning("BigQuery ingest failed (continuing with artifacts only): %s", exc)
        else:
            LOGGER.info("BigQuery merge requested but GCP_PROJECT_ID unset; skipping.")

    runs = dedupe_runs(runs)
    trades = dedupe_trades(trades)
    heartbeats = dedupe_heartbeats(heartbeats)
    order_events = dedupe_order_events(order_events)
    account_snapshots = dedupe_account_snapshots(account_snapshots)
    runs = backfill_run_equity_from_snapshots(runs, account_snapshots)
    reconciliation_tolerance = float(
        os.getenv("BROKER_RECONCILIATION_TOLERANCE", "50")
    )
    reconciliations = build_broker_reconciliations(
        run_dates_list,
        order_events,
        account_snapshots,
        tolerance=reconciliation_tolerance,
    )
    lifecycles = build_trade_lifecycles(order_events)

    LOGGER.info(
        "Prepared aggregate rows | runs=%d trades=%d heartbeats=%d order_events=%d "
        "account_snapshots=%d reconciliations=%d lifecycles=%d",
        len(runs),
        len(trades),
        len(heartbeats),
        len(order_events),
        len(account_snapshots),
        len(reconciliations),
        len(lifecycles),
    )
    if not (runs or trades or heartbeats or order_events or account_snapshots):
        message = "No ingestable rows were parsed from located run directories"
        if args.allow_empty:
            LOGGER.warning("%s (allow-empty enabled)", message)
            return
        raise SystemExit(f"{message}. Failing fast to avoid empty EOD writes.")

    if args.dry_run:
        LOGGER.info("Dry run enabled; skipping Supabase writes.")
        return

    now_iso = datetime.now(timezone.utc).isoformat()
    conn = connect_supabase()
    try:
        with conn, conn.cursor() as cur:
            upsert_runs(cur, runs, now_iso)
            upsert_trades(cur, trades, now_iso)
            upsert_heartbeats(cur, heartbeats, now_iso)
            upsert_order_events(cur, order_events, now_iso)
            upsert_account_snapshots(cur, account_snapshots, now_iso)
            upsert_broker_reconciliations(cur, reconciliations, now_iso)
            upsert_trade_lifecycles(cur, lifecycles, now_iso)
            alerts = [row for row in reconciliations if row.status == "alert"]
            for row in alerts:
                LOGGER.error(
                    "Broker reconciliation alert %s: booked=%.2f broker=%s gap=%s",
                    row.reconciliation_date,
                    row.booked_realized_pnl,
                    row.broker_daily_pnl,
                    row.pnl_gap,
                )
            if args.validate:
                checks = validate(cur)
                LOGGER.info("Validation checks: %s", json.dumps(checks, sort_keys=True))
                if checks["runs_7d"] == 0:
                    raise SystemExit("Validation failed: no run rows in the last 7 days.")
                strict = os.getenv("EOD_STRICT_TELEMETRY", "").lower() in ("1", "true", "yes")
                if checks["heartbeats_7d"] == 0:
                    msg = "Validation failed: no runtime heartbeats in the last 7 days."
                    if strict:
                        raise SystemExit(msg)
                    LOGGER.warning("%s (set EOD_STRICT_TELEMETRY=1 to fail on this)", msg)
            LOGGER.info(
                "EOD ingest complete | runs=%d trades=%d heartbeats=%d order_events=%d account_snapshots=%d",
                len(runs),
                len(trades),
                len(heartbeats),
                len(order_events),
                len(account_snapshots),
            )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
