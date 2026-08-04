import { supabaseSelect } from "@/lib/supabase";
import type {
  AccountSnapshot,
  Heartbeat,
  JsonRecord,
  LiveStateRow,
  OpsMetricsDaily,
  RunRow,
  TradeRow,
  TradeLifecycleRow,
  ValidationEvent,
} from "@/lib/types";
import {
  deriveTradeDirection,
  tradeDirectionDetails,
} from "@/lib/tradeDirection";

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

export async function getTradeLifecycles(
  days = 30,
): Promise<TradeLifecycleRow[]> {
  type ExecutionEvent = {
    event_type: string;
    event_ts: string | null;
    order_id: string | null;
    symbol: string;
    side: string | null;
    details: JsonRecord | null;
  };

  const [rows, executionEvents] = await Promise.all([
    supabaseSelect<TradeLifecycleRow>(
      "llm_advisor_trade_lifecycles",
      `select=lifecycle_uid,entry_order_id,exit_order_id,symbol,underlying_symbol,opened_at,closed_at,filled_qty,entry_fill_price,exit_fill_price,exit_reason,realized_pnl,status,details&closed_at=gte.${daysAgoIso(days)}T00:00:00Z&order=closed_at.desc&limit=1000`,
    ),
    // Execution events already contain the option plan used at entry. Joining
    // them here gives historical lifecycles explicit direction metadata even
    // though the lifecycle table predates those columns.
    supabaseSelect<ExecutionEvent>(
      "llm_advisor_order_events",
      `select=event_type,event_ts,order_id,symbol,side,details&event_type=eq.execution_succeeded&run_date=gte.${daysAgoIso(days)}&order=event_ts.desc&limit=2000`,
    ),
  ]);

  const eventsByOrderId = new Map<string, ExecutionEvent>();
  const eventsBySymbol = new Map<string, ExecutionEvent[]>();

  function optionSymbolFromEvent(event: ExecutionEvent): string | null {
    const details = event.details ?? {};
    const order = details.order;
    if (!order || typeof order !== "object" || Array.isArray(order)) return null;
    const optionPlan = (order as JsonRecord).option_plan;
    if (!optionPlan || typeof optionPlan !== "object" || Array.isArray(optionPlan)) {
      return null;
    }
    const optionSymbol = (optionPlan as JsonRecord).option_symbol;
    return typeof optionSymbol === "string" ? optionSymbol : null;
  }

  for (const event of executionEvents ?? []) {
    if (event.order_id) eventsByOrderId.set(event.order_id, event);
    const symbols = [event.symbol, optionSymbolFromEvent(event)]
      .filter((symbol): symbol is string => Boolean(symbol))
      .map((symbol) => symbol.toUpperCase());
    for (const symbol of symbols) {
      eventsBySymbol.set(symbol, [...(eventsBySymbol.get(symbol) ?? []), event]);
    }
  }

  function matchingExecutionEvent(row: TradeLifecycleRow): ExecutionEvent | null {
    if (row.entry_order_id) return eventsByOrderId.get(row.entry_order_id) ?? null;
    const candidates = eventsBySymbol.get(row.symbol.toUpperCase()) ?? [];
    if (!candidates.length) return null;
    const target = Date.parse(row.opened_at ?? row.closed_at ?? "");
    return candidates.reduce((best, candidate) => {
      if (!best) return candidate;
      if (!Number.isFinite(target)) return best;
      const bestDistance = Math.abs(Date.parse(best.event_ts ?? "") - target);
      const candidateDistance = Math.abs(Date.parse(candidate.event_ts ?? "") - target);
      return candidateDistance < bestDistance ? candidate : best;
    }, null as ExecutionEvent | null);
  }

  return (rows ?? []).map((row) => {
    const event = matchingExecutionEvent(row);
    if (!event) return row;

    const eventDetails = event.details ?? {};
    const order = eventDetails.order;
    const optionPlan =
      order && typeof order === "object" && !Array.isArray(order)
        ? (order as JsonRecord).option_plan
        : null;
    const direction = deriveTradeDirection({
      symbol: row.symbol,
      side:
        order && typeof order === "object" && !Array.isArray(order)
          ? (order as JsonRecord).side
          : null,
      details: {
        trade_direction: {
          position_side:
            order && typeof order === "object" && !Array.isArray(order)
              ? (order as JsonRecord).side
              : null,
          contract_type:
            optionPlan && typeof optionPlan === "object" && !Array.isArray(optionPlan)
              ? (optionPlan as JsonRecord).contract_type
              : null,
          signal_bias:
            optionPlan && typeof optionPlan === "object" && !Array.isArray(optionPlan)
              ? (optionPlan as JsonRecord).signal_side
              : event.side,
          entry_action:
            optionPlan && typeof optionPlan === "object" && !Array.isArray(optionPlan)
              ? (optionPlan as JsonRecord).position_intent
              : null,
        },
      },
    });

    return {
      ...row,
      details: {
        ...(row.details ?? {}),
        trade_direction: tradeDirectionDetails(direction),
      },
    };
  });
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
