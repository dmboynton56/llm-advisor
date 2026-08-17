import {
  getAccountSnapshots,
  getDecisionLog,
  getLatestHeartbeat,
  getLatestOpsMetrics,
  getLiveState,
  getTradeLifecycles,
  getTrades,
} from "@/lib/data";
import { firstJsonString, jsonNumber, jsonRecords } from "@/lib/json";
import type {
  AccountSnapshot,
  CellStats,
  Decision,
  LiveOpenPosition,
  LiveStateRow,
  OpsMetricsPayload,
  TradeLifecycleRow,
  TradeRow,
  ValidationGateResult,
} from "@/lib/types";
import { getTodayOverviewPositions } from "@/lib/positions";

export type MobileAccount = {
  equity: number | null;
  dailyPnl: number | null;
  dailyPnlPercent: number | null;
  buyingPower: number | null;
};

export type MobileHealth = {
  isHealthy: boolean;
  heartbeat: string | null;
  loopCount: number | null;
  symbolsTracked: number | null;
  message: string;
};

export type MobilePosition = {
  id: string;
  symbol: string;
  optionSymbol: string | null;
  side: string | null;
  quantity: number;
  entryPrice: number | null;
  currentPrice: number | null;
  unrealizedPnl: number;
  unrealizedPnlPercent: number;
  setup: string | null;
  dte: number | null;
  openedAt: string | null;
  stopMark: number | null;
  targetMark: number | null;
};

export type MobileDecision = {
  id: string;
  symbol: string;
  setup: string | null;
  verdict: string;
  confidence: number | null;
  reasoning: string | null;
  createdAt: string | null;
};

export type MobileTradeFill = {
  id: string;
  kind: string;
  timestamp: string | null;
  quantity: number | null;
  price: number | null;
  pnl: number | null;
  reason: string | null;
};

export type MobileTrade = {
  id: string;
  symbol: string;
  underlying: string | null;
  optionSymbol: string | null;
  side: string | null;
  setup: string | null;
  status: string;
  dte: number | null;
  quantity: number | null;
  entryPrice: number | null;
  exitPrice: number | null;
  pnl: number | null;
  returnPercent: number | null;
  entryAt: string | null;
  exitAt: string | null;
  exitReason: string | null;
  bias: string | null;
  validationVerdict: string | null;
  confidence: number | null;
  reasoning: string | null;
  riskAssessment: string | null;
  vetoFlags: string[];
  gateResults: string[];
  fills: MobileTradeFill[];
};

export type MobilePerformance = {
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number | null;
  totalPnl: number | null;
  maxDrawdown: number | null;
  averageWin: number | null;
  averageLoss: number | null;
};

export type MobileBreakdown = {
  id: string;
  trades: number;
  winRate: number | null;
  pnl: number | null;
  averageRiskReward: number | null;
};

export type MobileFunnel = {
  signals: number;
  approved: number;
  executed: number;
  approvalRate: number | null;
  rejectionReasons: Record<string, number>;
};

export type MobileSnapshot = {
  schemaVersion: 1;
  generatedAt: string;
  staleAfterSeconds: number;
  account: MobileAccount;
  health: MobileHealth;
  equityHistory: Array<{ capturedAt: string; equity: number; dailyPnl: number | null }>;
  positions: MobilePosition[];
  latestDecision: MobileDecision | null;
  performance: MobilePerformance;
  breakdowns: MobileBreakdown[];
  funnel: MobileFunnel;
  trades: MobileTrade[];
};

function latestSnapshot(rows: AccountSnapshot[]): AccountSnapshot | null {
  return rows.length ? rows[rows.length - 1] : null;
}

function healthFrom(live: LiveStateRow | null, heartbeat: Awaited<ReturnType<typeof getLatestHeartbeat>>): MobileHealth {
  const timestamp = live?.updated_at ?? live?.heartbeat_ts ?? heartbeat?.heartbeat_ts ?? null;
  const age = timestamp ? Date.now() - Date.parse(timestamp) : Number.POSITIVE_INFINITY;
  const isHealthy = Number.isFinite(age) && age < 3 * 60 * 1000;
  return {
    isHealthy,
    heartbeat: timestamp,
    loopCount: live?.loop_count ?? heartbeat?.loop_count ?? null,
    symbolsTracked: heartbeat?.symbols_tracked ?? null,
    message: !timestamp ? "No heartbeat" : isHealthy ? "Heartbeat received" : "Heartbeat is stale",
  };
}

function accountFrom(snapshot: AccountSnapshot | null, live: LiveStateRow | null): MobileAccount {
  return {
    equity: snapshot?.equity ?? live?.equity ?? null,
    dailyPnl: snapshot?.daily_pnl ?? live?.daily_pnl ?? null,
    dailyPnlPercent: snapshot?.daily_pnl_pct ?? null,
    buyingPower: snapshot?.buying_power ?? null,
  };
}

function positionFromLive(position: LiveOpenPosition): MobilePosition {
  return {
    id: position.position_id ?? position.entry_order_id ?? `${position.symbol}-${position.opened_at ?? "open"}`,
    symbol: position.underlying_symbol ?? position.symbol,
    optionSymbol: position.option_symbol ?? position.symbol,
    side: position.side ?? null,
    quantity: position.remaining_qty ?? position.qty,
    entryPrice: position.entry_price,
    currentPrice: position.current_price,
    unrealizedPnl: position.unrealized_pl ?? 0,
    unrealizedPnlPercent: position.unrealized_plpc ?? 0,
    setup: position.setup_type ?? null,
    dte: position.dte ?? null,
    openedAt: position.opened_at ?? null,
    stopMark: position.stop_mark ?? null,
    targetMark: position.tp_mark ?? null,
  };
}

function positionFromOverview(position: ReturnType<typeof getTodayOverviewPositions>[number]): MobilePosition {
  return {
    id: position.id,
    symbol: position.underlying_symbol ?? position.symbol,
    optionSymbol: position.option_symbol ?? position.symbol,
    side: position.side ?? null,
    quantity: position.remaining_qty ?? position.initial_qty ?? 0,
    entryPrice: position.entry_price,
    currentPrice: position.current_price,
    unrealizedPnl: position.unrealized_pnl ?? 0,
    unrealizedPnlPercent: position.return_pct ?? 0,
    setup: position.setup_type ?? null,
    dte: position.dte ?? null,
    openedAt: position.opened_at,
    stopMark: null,
    targetMark: null,
  };
}

function gateText(gate: ValidationGateResult): string {
  return `${gate.code}: ${gate.status}`;
}

function fillsFromLifecycle(row: TradeLifecycleRow): MobileTradeFill[] {
  return jsonRecords(row.details?.fills).map((fill, index) => ({
    id: `${row.lifecycle_uid}-fill-${index}`,
    kind: firstJsonString(fill.kind) ?? "fill",
    timestamp: firstJsonString(fill.timestamp),
    quantity: jsonNumber(fill.qty ?? fill.quantity),
    price: jsonNumber(fill.price),
    pnl: jsonNumber(fill.pnl),
    reason: firstJsonString(fill.reason),
  }));
}

type MobileEquityPoint = {
  capturedAt: string;
  equity: number;
  dailyPnl: number | null;
};

function equityPoint(row: AccountSnapshot): MobileEquityPoint | null {
  if (row.equity === null || !row.captured_at) return null;
  return {
    capturedAt: row.captured_at,
    equity: row.equity,
    dailyPnl: row.daily_pnl,
  };
}

function tradeFromLifecycle(row: TradeLifecycleRow): MobileTrade {
  const validation = row.validation_summary;
  return {
    id: row.lifecycle_uid,
    symbol: row.underlying_symbol ?? row.symbol,
    underlying: row.underlying_symbol,
    optionSymbol: row.symbol,
    side: null,
    setup: row.setup_type ?? null,
    status: row.status,
    dte: row.option_dte ?? null,
    quantity: row.filled_qty,
    entryPrice: row.entry_fill_price,
    exitPrice: row.exit_fill_price,
    pnl: row.realized_pnl,
    returnPercent: null,
    entryAt: row.opened_at,
    exitAt: row.closed_at,
    exitReason: row.exit_reason,
    bias: row.daily_bias?.llm_bias ?? row.daily_bias?.ml_bias ?? null,
    validationVerdict: validation?.verdict ?? null,
    confidence: validation?.confidence ?? null,
    reasoning: validation?.reasoning ?? null,
    riskAssessment: validation?.risk_assessment ?? null,
    vetoFlags: validation?.veto_flags ?? [],
    gateResults: validation?.gate_results?.map(gateText) ?? [],
    fills: fillsFromLifecycle(row),
  };
}

function tradeFromRow(row: TradeRow): MobileTrade {
  const validation = row.validation_summary;
  return {
    id: row.trade_uid,
    symbol: row.underlying_symbol ?? row.symbol,
    underlying: row.underlying_symbol,
    optionSymbol: row.symbol,
    side: row.side,
    setup: row.setup_type,
    status: row.status ?? "unknown",
    dte: row.option_dte,
    quantity: row.qty,
    entryPrice: row.entry_price,
    exitPrice: row.exit_price,
    pnl: row.pnl,
    returnPercent: null,
    entryAt: row.entry_time,
    exitAt: row.exit_time,
    exitReason: row.exit_reason,
    bias: row.daily_bias?.llm_bias ?? row.daily_bias?.ml_bias ?? null,
    validationVerdict: validation?.verdict ?? null,
    confidence: validation?.confidence ?? null,
    reasoning: validation?.reasoning ?? null,
    riskAssessment: validation?.risk_assessment ?? null,
    vetoFlags: validation?.veto_flags ?? [],
    gateResults: validation?.gate_results?.map(gateText) ?? [],
    fills: [],
  };
}

function cellStats(cell: CellStats | undefined): MobilePerformance {
  return {
    totalTrades: cell?.trades ?? 0,
    winningTrades: cell?.winning_trades ?? 0,
    losingTrades: cell?.losing_trades ?? 0,
    winRate: cell?.win_rate ?? null,
    totalPnl: cell?.total_pnl ?? null,
    maxDrawdown: null,
    averageWin: cell?.average_win ?? null,
    averageLoss: cell?.average_loss ?? null,
  };
}

function performanceFrom(ops: OpsMetricsPayload | null, lifecycles: TradeLifecycleRow[]): MobilePerformance {
  const overall = ops?.overall;
  if (overall) {
    return {
      ...cellStats(overall),
      maxDrawdown: overall.max_drawdown ?? null,
    };
  }
  const closed = lifecycles.filter((row) => row.status !== "open");
  const winners = closed.filter((row) => (row.realized_pnl ?? 0) > 0);
  const losers = closed.filter((row) => (row.realized_pnl ?? 0) < 0);
  return {
    totalTrades: closed.length,
    winningTrades: winners.length,
    losingTrades: losers.length,
    winRate: closed.length ? winners.length / closed.length : null,
    totalPnl: closed.reduce((sum, row) => sum + (row.realized_pnl ?? 0), 0),
    maxDrawdown: null,
    averageWin: winners.length ? winners.reduce((sum, row) => sum + (row.realized_pnl ?? 0), 0) / winners.length : null,
    averageLoss: losers.length ? losers.reduce((sum, row) => sum + (row.realized_pnl ?? 0), 0) / losers.length : null,
  };
}

export async function getMobileSnapshot(): Promise<MobileSnapshot> {
  const [snapshots, heartbeat, live, lifecycles, decisionLog, opsRow, trades] = await Promise.all([
    getAccountSnapshots(90),
    getLatestHeartbeat(),
    getLiveState("paper"),
    getTradeLifecycles(90),
    getDecisionLog(12),
    getLatestOpsMetrics(),
    getTrades(90),
  ]);

  const latest = latestSnapshot(snapshots);
  const overview = getTodayOverviewPositions(live, lifecycles);
  const positions = live?.open_positions?.length
    ? live.open_positions.map(positionFromLive)
    : overview.filter((position) => position.status === "open").map(positionFromOverview);
  const ops = opsRow?.payload ?? null;
  const latestDecision: Decision | undefined = decisionLog.decisions[0];
  const tradesById = new Map<string, MobileTrade>();
  for (const row of trades) tradesById.set(row.trade_uid, tradeFromRow(row));
  for (const row of lifecycles) tradesById.set(row.lifecycle_uid, tradeFromLifecycle(row));

  const breakdowns = Object.entries(ops?.breakdowns?.by_underlying ?? {}).map(([id, cell]) => ({
    id,
    trades: cell.trades,
    winRate: cell.win_rate,
    pnl: cell.total_pnl,
    averageRiskReward: cell.avg_realized_rr ?? cell.avg_planned_rr,
  }));
  const funnel = ops?.funnel;
  const generatedAt = new Date().toISOString();

  return {
    schemaVersion: 1,
    generatedAt,
    staleAfterSeconds: 180,
    account: accountFrom(latest, live),
    health: healthFrom(live, heartbeat),
    equityHistory: snapshots
      .map(equityPoint)
      .filter((point): point is MobileEquityPoint => Boolean(point)),
    positions,
    latestDecision: latestDecision
      ? {
          id: latestDecision.key,
          symbol: latestDecision.symbol,
          setup: latestDecision.setupType,
          verdict: latestDecision.verdict,
          confidence: latestDecision.confidence,
          reasoning: latestDecision.reason,
          createdAt: latestDecision.eventTs,
        }
      : null,
    performance: performanceFrom(ops, lifecycles),
    breakdowns,
    funnel: {
      signals: funnel?.stages?.signals ?? decisionLog.signals,
      approved: funnel?.stages?.approved ?? decisionLog.approved,
      executed: funnel?.stages?.executed ?? decisionLog.filled,
      approvalRate: funnel?.llm_approval_rate ?? (decisionLog.signals ? decisionLog.approved / decisionLog.signals : null),
      rejectionReasons: funnel?.rejection_reasons ?? {},
    },
    trades: [...tradesById.values()].sort((a, b) => Date.parse(b.entryAt ?? "") - Date.parse(a.entryAt ?? "")),
  };
}

export function mobileLivePayload(snapshot: MobileSnapshot) {
  return {
    schemaVersion: snapshot.schemaVersion,
    generatedAt: snapshot.generatedAt,
    staleAfterSeconds: snapshot.staleAfterSeconds,
    account: snapshot.account,
    health: snapshot.health,
    positions: snapshot.positions,
    latestDecision: snapshot.latestDecision,
  };
}
