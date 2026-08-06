import { supabaseSelect, supabaseSelectPaged } from "@/lib/supabase";
import type {
  AccountSnapshot,
  Decision,
  DecisionEvent,
  DailyBiasSummary,
  DecisionLog,
  Heartbeat,
  JsonRecord,
  LiveStateRow,
  OpsMetricsDaily,
  RunRow,
  TradeRow,
  TradeLifecycleRow,
  TradeValidationSummary,
  ValidationEvent,
} from "@/lib/types";
import { dateEtIso } from "@/lib/format";
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
  const rows = await supabaseSelectPaged<AccountSnapshot>(
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

type TradeEnrichmentRow = {
  trade_uid: string;
  run_date: string;
  order_id: string | null;
  symbol: string;
  entry_time: string | null;
  setup_type: string | null;
  option_dte: number | null;
  option_metadata: JsonRecord | null;
};

type DailyBiasRow = DailyBiasSummary;

function asJsonRecord(value: unknown): JsonRecord | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as JsonRecord)
    : null;
}

function firstString(...values: unknown[]): string | null {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return null;
}

function firstNumber(...values: unknown[]): number | null {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value)) return value;
    if (typeof value === "string" && value.trim()) {
      const parsed = Number(value);
      if (Number.isFinite(parsed)) return parsed;
    }
  }
  return null;
}

function asBias(value: unknown): DailyBiasSummary["ml_bias"] {
  const normalized = String(value ?? "").trim().toLowerCase();
  return normalized === "bullish" || normalized === "bearish" || normalized === "choppy"
    ? normalized
    : "unavailable";
}

function asAgreement(value: unknown): DailyBiasSummary["agreement"] {
  const normalized = String(value ?? "").trim().toLowerCase();
  return normalized === "agree" || normalized === "partial" || normalized === "disagree"
    ? normalized
    : "unknown";
}

function dailyBiasFromDetails(details: JsonRecord | null): DailyBiasSummary | null {
  const order = asJsonRecord(details?.order);
  const optionPlan = asJsonRecord(order?.option_plan);
  const raw =
    asJsonRecord(details?.daily_bias) ??
    asJsonRecord(details?.bias_snapshot) ??
    asJsonRecord(optionPlan?.daily_bias) ??
    asJsonRecord(optionPlan?.bias_snapshot);
  if (!raw) return null;
  return {
    bias_date: firstString(raw.bias_date, raw.date, raw.run_date) ?? "",
    symbol: firstString(raw.symbol) ?? "",
    ml_bias: asBias(raw.ml_bias ?? raw.daily_bias ?? raw.bias),
    ml_confidence: firstNumber(raw.ml_confidence, raw.confidence),
    llm_bias: raw.llm_bias == null ? null : asBias(raw.llm_bias),
    llm_confidence: firstNumber(raw.llm_confidence),
    agreement: asAgreement(raw.agreement),
    bias_available: raw.bias_available !== false,
    bias_error: firstString(raw.bias_error, raw.error),
    llm_reasoning: firstString(raw.llm_reasoning, raw.reasoning),
    context_version: firstString(raw.context_version),
    generated_at: firstString(raw.generated_at, raw.timestamp),
  };
}

function validationFromDetails(
  details: JsonRecord | null,
  eventType?: string | null,
): TradeValidationSummary | null {
  const raw = asJsonRecord(details?.validation) ?? details;
  if (!raw) return null;
  const hasValidation =
    raw.should_execute != null ||
    raw.verdict != null ||
    raw.confidence != null ||
    raw.reasoning != null ||
    raw.veto_flags != null ||
    raw.gate_results != null;
  if (!hasValidation) return null;
  const inferredVerdict =
    eventType === "execution_succeeded"
      ? "approved"
      : eventType === "validation_rejected"
        ? "rejected"
        : eventType === "validation_error"
          ? "error"
          : null;
  const verdictValue = String(
    raw.verdict ??
      inferredVerdict ??
      (raw.should_execute === true
        ? "approved"
        : raw.should_execute === false
          ? "rejected"
          : "unknown"),
  );
  const verdict: TradeValidationSummary["verdict"] =
    verdictValue === "approved" ? "approved" : verdictValue === "rejected" ? "rejected" : verdictValue === "error" ? "error" : "unknown";
  const gates = Array.isArray(raw.gate_results) ? raw.gate_results : [];
  return {
    signal_uid: firstString(raw.signal_uid),
    verdict,
    confidence: firstNumber(raw.confidence),
    reasoning: firstString(raw.reasoning, raw.reason),
    risk_assessment: firstString(raw.risk_assessment),
    veto_flags: Array.isArray(raw.veto_flags)
      ? raw.veto_flags.filter((value): value is string => typeof value === "string")
      : [],
    gate_results: gates.filter((value): value is Record<string, unknown> => Boolean(value && typeof value === "object")) as TradeValidationSummary["gate_results"],
    model: firstString(raw.model, raw.llm_model),
  };
}

function riskPlanFromDetails(details: JsonRecord | null): JsonRecord | null {
  const order = asJsonRecord(details?.order);
  const optionPlan = asJsonRecord(order?.option_plan);
  return (
    asJsonRecord(details?.risk_plan) ??
    asJsonRecord(details?.execution_risk_plan) ??
    asJsonRecord(optionPlan?.risk_plan)
  );
}

function rrFromPlan(value: unknown): number | null {
  const plan = asJsonRecord(value);
  if (!plan) return null;
  const entry = firstNumber(plan.underlying_entry, plan.entry_price);
  const stop = firstNumber(plan.underlying_stop, plan.stop_loss);
  const target = firstNumber(plan.underlying_target, plan.take_profit);
  if (entry == null || stop == null || target == null) return null;
  const risk = Math.abs(entry - stop);
  return risk > 0 ? Math.abs(target - entry) / risk : null;
}

function plannedRrFromDetails(
  details: JsonRecord | null,
  optionMetadata: JsonRecord | null,
): number | null {
  const order = asJsonRecord(details?.order);
  const optionPlan = asJsonRecord(order?.option_plan);
  const riskPlan = riskPlanFromDetails(details);
  return firstNumber(
    details?.planned_underlying_rr,
    riskPlan?.planned_underlying_rr,
    rrFromPlan(riskPlan),
    rrFromPlan(details?.trade_plan),
    rrFromPlan(optionPlan?.underlying_trade_plan),
    rrFromPlan(optionMetadata?.underlying_trade_plan),
  );
}

function optionDteFromSymbol(symbol: string, entryAt: string | null): number | null {
  const match = /^([A-Z0-9.]{1,10})(\d{6})[CP]\d{8}$/i.exec(
    symbol.replace(/\s+/g, "").toUpperCase(),
  );
  const entryDate = entryAt?.slice(0, 10);
  if (!match || !entryDate) return null;

  const expiry = match[2];
  const expiryMs = Date.parse(
    `20${expiry.slice(0, 2)}-${expiry.slice(2, 4)}-${expiry.slice(4, 6)}T00:00:00Z`,
  );
  const entryMs = Date.parse(`${entryDate}T00:00:00Z`);
  if (!Number.isFinite(expiryMs) || !Number.isFinite(entryMs)) return null;
  return Math.round((expiryMs - entryMs) / 86_400_000);
}

export async function getTradeLifecycles(
  days = 30,
): Promise<TradeLifecycleRow[]> {
  type ExecutionEvent = {
    event_type: string;
    event_ts: string | null;
    order_id: string | null;
    symbol: string;
    setup_type: string | null;
    side: string | null;
    details: JsonRecord | null;
  };

  const [rows, executionEvents, tradeRows, dailyBiasRows] = await Promise.all([
    supabaseSelect<TradeLifecycleRow>(
      "llm_advisor_trade_lifecycles",
      `select=lifecycle_uid,entry_order_id,exit_order_id,symbol,underlying_symbol,opened_at,closed_at,filled_qty,entry_fill_price,exit_fill_price,exit_reason,realized_pnl,status,details&closed_at=gte.${daysAgoIso(days)}T00:00:00Z&order=closed_at.desc&limit=1000`,
    ),
    // Execution events already contain the option plan used at entry. Joining
    // them here gives historical lifecycles explicit direction metadata even
    // though the lifecycle table predates those columns.
    supabaseSelect<ExecutionEvent>(
      "llm_advisor_order_events",
      `select=event_type,event_ts,order_id,symbol,setup_type,side,details&event_type=eq.execution_succeeded&run_date=gte.${daysAgoIso(days)}&order=event_ts.desc&limit=2000`,
    ),
    // Some legacy lifecycle rows have no entry order id. The enriched trade
    // row is the second historical source for setup/DTE in those cases.
    supabaseSelect<TradeEnrichmentRow>(
      "llm_advisor_backtest_trades",
      `select=trade_uid,run_date,order_id,symbol,entry_time,setup_type,option_dte,option_metadata&run_date=gte.${daysAgoIso(days)}&order=run_date.desc,entry_time.desc&limit=1000`,
    ),
    supabaseSelect<DailyBiasRow>(
      "llm_advisor_daily_bias",
      `select=bias_date,symbol,ml_bias,ml_confidence,llm_bias,llm_confidence,agreement,bias_available,bias_error,llm_reasoning,context_version,generated_at&bias_date=gte.${daysAgoIso(days)}&order=bias_date.desc&limit=1000`,
    ),
  ]);

  const eventsByOrderId = new Map<string, ExecutionEvent>();
  const eventsBySymbol = new Map<string, ExecutionEvent[]>();
  const tradesByOrderId = new Map<string, TradeEnrichmentRow>();
  const tradesBySymbol = new Map<string, TradeEnrichmentRow[]>();
  const dailyBiasByKey = new Map<string, DailyBiasSummary>();

  for (const bias of dailyBiasRows ?? []) {
    dailyBiasByKey.set(`${bias.bias_date}|${bias.symbol.toUpperCase()}`, bias);
  }

  for (const trade of tradeRows ?? []) {
    if (trade.order_id) tradesByOrderId.set(trade.order_id, trade);
    const symbol = trade.symbol.toUpperCase();
    tradesBySymbol.set(symbol, [...(tradesBySymbol.get(symbol) ?? []), trade]);
  }

  function optionPlanFromEvent(event: ExecutionEvent | null): JsonRecord | null {
    if (!event) return null;
    const details = event.details ?? {};
    const order = asJsonRecord(details.order);
    return asJsonRecord(order?.option_plan) ?? asJsonRecord(details.option_plan);
  }

  function optionSymbolFromEvent(event: ExecutionEvent): string | null {
    return firstString(optionPlanFromEvent(event)?.option_symbol);
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

  function matchingTradeEnrichment(row: TradeLifecycleRow): TradeEnrichmentRow | null {
    if (row.entry_order_id) {
      const exact = tradesByOrderId.get(row.entry_order_id);
      if (exact) return exact;
    }

    const candidates = tradesBySymbol.get(row.symbol.toUpperCase()) ?? [];
    if (!candidates.length) return null;
    const target = Date.parse(row.opened_at ?? row.closed_at ?? "");
    return candidates.reduce((best, candidate) => {
      if (!best) return candidate;
      if (!Number.isFinite(target)) return best;
      const bestDistance = Math.abs(Date.parse(best.entry_time ?? "") - target);
      const candidateDistance = Math.abs(
        Date.parse(candidate.entry_time ?? "") - target,
      );
      return candidateDistance < bestDistance ? candidate : best;
    }, null as TradeEnrichmentRow | null);
  }

  return (rows ?? []).map((row) => {
    const event = matchingExecutionEvent(row);
    const trade = matchingTradeEnrichment(row);
    if (!event && !trade) return row;

    const eventDetails = event?.details ?? {};
    const order = asJsonRecord(eventDetails.order);
    const optionPlan = optionPlanFromEvent(event);
    const metadata = trade?.option_metadata;
    const tradePlan = asJsonRecord(eventDetails.trade_plan);
    const optionSymbol =
      firstString(optionPlan?.option_symbol, metadata?.option_symbol, row.symbol) ??
      row.symbol;
    const underlyingSymbol =
      firstString(
        row.underlying_symbol,
        optionPlan?.underlying_symbol,
        metadata?.underlying_symbol,
      ) ?? row.symbol;
    const direction = deriveTradeDirection({
      symbol: row.symbol,
      side: firstString(order?.side, optionPlan?.side, metadata?.side, event?.side),
      details: {
        trade_direction: {
          position_side: firstString(order?.side, optionPlan?.side, metadata?.side),
          contract_type: firstString(
            optionPlan?.contract_type,
            metadata?.contract_type,
          ),
          signal_bias: firstString(
            optionPlan?.signal_side,
            metadata?.signal_side,
            event?.side,
          ),
          entry_action: firstString(
            optionPlan?.position_intent,
            metadata?.position_intent,
          ),
        },
      },
    });

    const setupType = firstString(
      row.setup_type,
      trade?.setup_type,
      optionPlan?.setup_type,
      event?.setup_type,
      metadata?.setup_type,
      tradePlan?.setup,
    );
    const optionDte =
      firstNumber(
        row.option_dte,
        trade?.option_dte,
        optionPlan?.dte,
        metadata?.dte,
      ) ?? optionDteFromSymbol(optionSymbol, row.opened_at);

    const biasSnapshot =
      dailyBiasFromDetails(row.details) ??
      dailyBiasFromDetails(eventDetails) ??
      dailyBiasByKey.get(`${dateEtIso(row.opened_at ?? row.closed_at)}|${underlyingSymbol.toUpperCase()}`) ??
      null;
    const riskPlan = riskPlanFromDetails(eventDetails) ?? riskPlanFromDetails(row.details);
    const validationSummary =
      validationFromDetails(eventDetails, event?.event_type) ??
      validationFromDetails(row.details);
    const realizedR = firstNumber(
      row.details?.realized_r,
      eventDetails.realized_r,
      riskPlan && typeof row.realized_pnl === "number" && Number(riskPlan.planned_option_risk_dollars) > 0
        ? row.realized_pnl / Number(riskPlan.planned_option_risk_dollars)
        : null,
    );

    return {
      ...row,
      underlying_symbol: underlyingSymbol === row.symbol ? row.underlying_symbol : underlyingSymbol,
      setup_type: setupType,
      option_dte: optionDte,
      daily_bias: biasSnapshot,
      planned_underlying_rr: firstNumber(
        row.details?.planned_underlying_rr,
        eventDetails.planned_underlying_rr,
        riskPlan?.planned_underlying_rr,
        plannedRrFromDetails(row.details, metadata ?? null),
        plannedRrFromDetails(eventDetails, metadata ?? null),
      ),
      realized_r: realizedR,
      validation_summary: validationSummary,
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

const DECISION_EVENT_TYPES = [
  "signal_detected",
  "validation_approved",
  "validation_rejected",
  "validation_error",
  "execution_succeeded",
] as const;

function decisionText(details: JsonRecord | null, keys: string[]): string | null {
  if (!details) return null;
  for (const key of keys) {
    const value = details[key];
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return null;
}

/**
 * The most recent session's signals and the verdict the model gave each one.
 * Scoped to a single run_date so the ledger always describes one session
 * rather than blending days.
 */
export async function getDecisionLog(limit = 12): Promise<DecisionLog> {
  const rows = await supabaseSelect<DecisionEvent>(
    "llm_advisor_order_events",
    `select=run_date,event_ts,event_type,symbol,setup_type,side,details&event_type=in.(${DECISION_EVENT_TYPES.join(
      ",",
    )})&run_date=gte.${daysAgoIso(7)}&order=event_ts.desc&limit=500`,
  );

  const empty: DecisionLog = {
    runDate: null,
    decisions: [],
    signals: 0,
    approved: 0,
    filled: 0,
  };
  if (!rows?.length) return empty;

  const runDate = rows.reduce(
    (latest, row) => (row.run_date > latest ? row.run_date : latest),
    rows[0].run_date,
  );
  const sessionRows = rows.filter((row) => row.run_date === runDate);

  const decisions: Decision[] = sessionRows
    .filter(
      (row) =>
        row.event_type === "validation_approved" ||
        row.event_type === "validation_rejected" ||
        row.event_type === "validation_error",
    )
    .slice(0, limit)
    .map((row, index) => {
      const confidence = row.details?.confidence;
      return {
        key: `${row.event_ts ?? row.run_date}-${row.symbol}-${index}`,
        eventTs: row.event_ts,
        symbol: row.symbol,
        setupType: row.setup_type,
        verdict:
          row.event_type === "validation_approved" ? "approved" : "vetoed",
        // Approved and rejected events both carry the model's `reasoning`;
        // validation_error carries `reason`/`error` instead.
        reason: decisionText(row.details, ["reasoning", "reason", "error"]),
        confidence: typeof confidence === "number" ? confidence : null,
      };
    });

  const countOf = (eventType: string) =>
    sessionRows.filter((row) => row.event_type === eventType).length;

  return {
    runDate,
    decisions,
    signals: countOf("signal_detected"),
    approved: countOf("validation_approved"),
    filled: countOf("execution_succeeded"),
  };
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
