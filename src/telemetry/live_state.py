"""Publish a single upserted live-state row to Supabase during the live loop."""
from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from src.execution.trade_tracker import option_dte
from src.utils.env_sanitize import getenv_strip

logger = logging.getLogger(__name__)

_consecutive_failures = 0
_disabled = False
_DISABLE_AFTER = 10


def reset_publish_state_for_tests() -> None:
    """Test helper: clear failure/disable counters."""
    global _consecutive_failures, _disabled
    _consecutive_failures = 0
    _disabled = False


def connect():
    """psycopg2 connection from SUPABASE_DB_* (same contract as EOD aggregate)."""
    import psycopg2

    host = getenv_strip("SUPABASE_DB_HOST")
    db = getenv_strip("SUPABASE_DB_NAME") or "postgres"
    user = getenv_strip("SUPABASE_DB_USER") or "postgres"
    port_raw = getenv_strip("SUPABASE_DB_PORT") or "5432"
    password = getenv_strip("SUPABASE_DB_PASSWORD") or getenv_strip("supabaseDBpass")
    if not host or not password:
        raise RuntimeError("Missing SUPABASE_DB_HOST / SUPABASE_DB_PASSWORD")
    try:
        port = int(port_raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid SUPABASE_DB_PORT: {port_raw!r}") from exc
    return psycopg2.connect(
        host=host,
        dbname=db,
        user=user,
        password=password,
        port=port,
        sslmode="require",
        connect_timeout=5,
    )


def _account_fields(order_manager: Any) -> tuple[Optional[float], Optional[float], Optional[float]]:
    equity: Optional[float] = None
    last_equity: Optional[float] = None
    client = getattr(order_manager, "trading_client", None)
    if client is not None:
        try:
            account = client.get_account()
            equity = float(account.equity)
            last_equity = float(account.last_equity)
        except Exception as exc:
            logger.debug("live_state account fetch via trading_client failed: %s", exc)
    if equity is None and hasattr(order_manager, "get_account_equity"):
        try:
            equity = float(order_manager.get_account_equity())
        except Exception as exc:
            logger.debug("live_state get_account_equity failed: %s", exc)
    daily_pnl = (
        equity - last_equity if equity is not None and last_equity is not None else None
    )
    return equity, last_equity, daily_pnl


def _setup_type_from_meta(meta: Dict[str, Any]) -> Optional[str]:
    if meta.get("setup_type"):
        return str(meta["setup_type"])
    plan = meta.get("option_plan")
    if isinstance(plan, dict) and plan.get("setup_type"):
        return str(plan["setup_type"])
    return None


def _opened_at_iso(meta: Dict[str, Any]) -> Optional[str]:
    opened = meta.get("opened_at")
    if isinstance(opened, datetime):
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=timezone.utc)
        return opened.astimezone(timezone.utc).isoformat()
    if isinstance(opened, str) and opened:
        return opened
    return None


def _tiered_state_payload(meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    state = meta.get("tiered_exit_state") if isinstance(meta, dict) else None
    if state is None and isinstance(meta, dict):
        state = meta.get("exit_state")
    if hasattr(state, "to_dict"):
        try:
            return state.to_dict()
        except Exception:
            return None
    if isinstance(state, dict):
        return dict(state)
    if isinstance(state, str):
        try:
            parsed = json.loads(state)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def build_live_state_row(
    trade_tracker: Any,
    order_manager: Any,
    settings: Any,
    *,
    session_date: date,
    loop_count: int,
    session_end_reason: Optional[str] = None,
    source: str = "paper",
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Build the upsert payload (pure aside from broker account fetch)."""
    now = now or datetime.now(timezone.utc)
    equity, last_equity, daily_pnl = _account_fields(order_manager)

    positions_map = (
        trade_tracker.get_all_positions()
        if trade_tracker and hasattr(trade_tracker, "get_all_positions")
        else {}
    )
    order_meta = getattr(trade_tracker, "_order_meta", {}) or {}
    open_positions: List[Dict[str, Any]] = []
    unrealized_total = 0.0
    for symbol, pos in positions_map.items():
        meta = order_meta.get(symbol, {})
        upl = float(pos.get("unrealized_pl") or 0.0)
        unrealized_total += upl
        asset_class = str(pos.get("asset_class") or "")
        option_symbol = pos.get("option_symbol") or (
            symbol if "option" in asset_class.lower() else None
        )
        dte = option_dte(str(option_symbol or symbol), session_date) if option_symbol else None
        open_positions.append(
            {
                "symbol": symbol,
                "option_symbol": option_symbol,
                "underlying_symbol": meta.get("underlying_symbol")
                or (pos.get("underlying_symbol")),
                "asset_class": asset_class or None,
                "qty": float(pos.get("qty") or 0),
                "side": str(pos.get("side") or ""),
                "entry_price": pos.get("entry_price")
                if pos.get("entry_price") is not None
                else pos.get("avg_entry_price"),
                "current_price": pos.get("current_price"),
                "unrealized_pl": upl,
                "unrealized_plpc": float(pos.get("unrealized_plpc") or 0.0),
                "opened_at": _opened_at_iso(meta),
                "setup_type": _setup_type_from_meta(meta),
                "dte": dte,
                "tiered_exit_state": _tiered_state_payload(meta),
            }
        )

    closed = (
        trade_tracker.get_session_closed()
        if trade_tracker and hasattr(trade_tracker, "get_session_closed")
        else []
    )
    realized = sum(float(c.get("pnl") or 0.0) for c in closed)
    wins = sum(1 for c in closed if float(c.get("pnl") or 0.0) > 0)
    losses = sum(1 for c in closed if float(c.get("pnl") or 0.0) < 0)
    session_stats: Dict[str, Any] = {
        "fills": len(closed),
        "realized_pnl": realized,
        "wins": wins,
        "losses": losses,
        "closed": closed,
    }
    if session_end_reason:
        session_stats["session_end_reason"] = session_end_reason

    options = getattr(settings, "options", None)
    exit_policy = {
        "stop_loss_pct": float(getattr(options, "stop_loss_pct", 0.35) or 0.35),
        "profit_target_pct": float(getattr(options, "profit_target_pct", 0.25) or 0.25),
        "max_hold_minutes": int(getattr(options, "max_hold_minutes", 2880) or 2880),
        "allow_overnight": bool(getattr(options, "allow_overnight", True)),
        "tiered_exit_enabled": bool(getattr(options, "tiered_exit_enabled", False)),
        "tiered_policy_version": "tiered_v1",
        "tiered_exit_underlyings": list(getattr(options, "tiered_exit_underlyings", []) or []),
        "tiered_min_contracts": int(getattr(options, "tiered_min_contracts", 4) or 4),
        "tiered_tp1_return_pct": float(getattr(options, "tiered_tp1_return_pct", 0.25) or 0.25),
        "tiered_tp2_return_pct": float(getattr(options, "tiered_tp2_return_pct", 0.50) or 0.50),
        "tiered_post_tp1_stop_return_pct": float(getattr(options, "tiered_post_tp1_stop_return_pct", -0.05) or -0.05),
        "tiered_runner_floor_return_pct": float(getattr(options, "tiered_runner_floor_return_pct", 0.25) or 0.25),
        "tiered_runner_giveback_pct": float(getattr(options, "tiered_runner_giveback_pct", 0.25) or 0.25),
        "tiered_exit_fill_timeout_seconds": int(getattr(options, "tiered_exit_fill_timeout_seconds", 120) or 120),
    }

    return {
        "source": source,
        "session_date": session_date.isoformat(),
        "heartbeat_ts": now.astimezone(timezone.utc).isoformat(),
        "loop_count": int(loop_count),
        "equity": equity,
        "last_equity": last_equity,
        "daily_pnl": daily_pnl,
        "unrealized_pnl": unrealized_total,
        "open_position_count": len(open_positions),
        "open_positions": open_positions,
        "session_stats": session_stats,
        "exit_policy": exit_policy,
    }


def publish_live_state(row: Dict[str, Any]) -> bool:
    """
    Upsert one live-state row. Never raises to the caller.
    Self-disables after repeated consecutive failures.
    """
    global _consecutive_failures, _disabled
    if _disabled:
        return False
    try:
        conn = connect()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO llm_advisor_live_state (
                          source, session_date, heartbeat_ts, loop_count,
                          equity, last_equity, daily_pnl, unrealized_pnl,
                          open_position_count, open_positions, session_stats,
                          exit_policy, updated_at
                        ) VALUES (
                          %(source)s, %(session_date)s, %(heartbeat_ts)s, %(loop_count)s,
                          %(equity)s, %(last_equity)s, %(daily_pnl)s, %(unrealized_pnl)s,
                          %(open_position_count)s, %(open_positions)s::jsonb,
                          %(session_stats)s::jsonb, %(exit_policy)s::jsonb, now()
                        )
                        ON CONFLICT (source) DO UPDATE SET
                          session_date = EXCLUDED.session_date,
                          heartbeat_ts = EXCLUDED.heartbeat_ts,
                          loop_count = EXCLUDED.loop_count,
                          equity = EXCLUDED.equity,
                          last_equity = EXCLUDED.last_equity,
                          daily_pnl = EXCLUDED.daily_pnl,
                          unrealized_pnl = EXCLUDED.unrealized_pnl,
                          open_position_count = EXCLUDED.open_position_count,
                          open_positions = EXCLUDED.open_positions,
                          session_stats = EXCLUDED.session_stats,
                          exit_policy = EXCLUDED.exit_policy,
                          updated_at = now()
                        """,
                        {
                            **row,
                            "open_positions": json.dumps(row.get("open_positions") or []),
                            "session_stats": json.dumps(row.get("session_stats") or {}),
                            "exit_policy": json.dumps(row.get("exit_policy") or {}),
                        },
                    )
        finally:
            conn.close()
        _consecutive_failures = 0
        return True
    except Exception as exc:
        _consecutive_failures += 1
        if _consecutive_failures >= _DISABLE_AFTER:
            _disabled = True
            logger.warning(
                "live_state publish disabled after %s consecutive failures (last: %s)",
                _DISABLE_AFTER,
                exc,
            )
        else:
            logger.warning(
                "live_state publish failed (%s/%s): %s",
                _consecutive_failures,
                _DISABLE_AFTER,
                exc,
            )
        return False


def publish_interval_ticks() -> int:
    """LIVE_STATE_PUBLISH_TICKS env (default 1)."""
    raw = os.getenv("LIVE_STATE_PUBLISH_TICKS", "1").strip() or "1"
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def should_publish_this_tick(loop_count: int) -> bool:
    interval = publish_interval_ticks()
    return loop_count % interval == 0
