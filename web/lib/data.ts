import { supabaseSelect } from "@/lib/supabase";
import type {
  AccountSnapshot,
  Heartbeat,
  LiveStateRow,
  OpsMetricsDaily,
  RunRow,
  TradeRow,
  ValidationEvent,
} from "@/lib/types";

function daysAgoIso(days: number): string {
  const d = new Date();
  d.setUTCDate(d.getUTCDate() - days);
  return d.toISOString().slice(0, 10);
}

export async function getAccountSnapshots(days = 90): Promise<AccountSnapshot[]> {
  const rows = await supabaseSelect<AccountSnapshot>(
    "llm_advisor_account_snapshots",
    `select=snapshot_date,captured_at,equity,last_equity,buying_power,daily_pnl,daily_pnl_pct,source&snapshot_date=gte.${daysAgoIso(days)}&order=captured_at.asc`,
  );
  return rows ?? [];
}

export async function getRuns(days = 30): Promise<RunRow[]> {
  const rows = await supabaseSelect<RunRow>(
    "llm_advisor_backtest_runs",
    `select=run_date,total_trades,closed_trades,winning_trades,losing_trades,total_pnl,win_rate,final_equity&run_date=gte.${daysAgoIso(days)}&order=run_date.asc`,
  );
  return rows ?? [];
}

export async function getLatestHeartbeat(): Promise<Heartbeat | null> {
  const rows = await supabaseSelect<Heartbeat>(
    "llm_advisor_runtime_heartbeats",
    "select=source_date,heartbeat_ts,loop_count,symbols_tracked,backtest&order=heartbeat_ts.desc&limit=1",
  );
  return rows?.[0] ?? null;
}

export async function getTrades(days = 90): Promise<TradeRow[]> {
  const rows = await supabaseSelect<TradeRow>(
    "llm_advisor_backtest_trades",
    `select=trade_uid,run_date,order_id,symbol,underlying_symbol,asset_class,side,setup_type,option_dte,qty,entry_price,exit_price,entry_time,exit_time,exit_reason,pnl,status&run_date=gte.${daysAgoIso(days)}&order=run_date.desc,entry_time.desc&limit=1000`,
  );
  return rows ?? [];
}

export async function getLatestOpsMetrics(): Promise<OpsMetricsDaily | null> {
  const rows = await supabaseSelect<OpsMetricsDaily>(
    "llm_advisor_ops_metrics_daily",
    "select=metric_date,payload&order=metric_date.desc&limit=1",
  );
  return rows?.[0] ?? null;
}

export async function getValidationEvents(days = 30): Promise<ValidationEvent[]> {
  const rows = await supabaseSelect<ValidationEvent>(
    "llm_advisor_order_events",
    `select=run_date,event_type&event_type=in.(validation_approved,validation_rejected)&run_date=gte.${daysAgoIso(days)}&order=run_date.asc&limit=5000`,
  );
  return rows ?? [];
}

export async function getLiveState(
  source = "paper",
): Promise<LiveStateRow | null> {
  const rows = await supabaseSelect<LiveStateRow>(
    "llm_advisor_live_state",
    `select=source,session_date,heartbeat_ts,loop_count,equity,last_equity,daily_pnl,unrealized_pnl,open_position_count,open_positions,session_stats,exit_policy,updated_at&source=eq.${encodeURIComponent(source)}&limit=1`,
  );
  return rows?.[0] ?? null;
}
