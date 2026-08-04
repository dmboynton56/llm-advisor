export type AccountSnapshot = {
  snapshot_date: string;
  captured_at: string;
  equity: number | null;
  last_equity: number | null;
  buying_power: number | null;
  daily_pnl: number | null;
  daily_pnl_pct: number | null;
  source: string;
};

export type RunRow = {
  run_date: string;
  total_trades: number;
  closed_trades: number;
  winning_trades: number;
  losing_trades: number;
  total_pnl: number | null;
  win_rate: number | null;
  final_equity: number | null;
};

export type JsonRecord = Record<string, unknown>;

export type TradePosition = "long" | "short";
export type OptionContractType = "call" | "put";
export type TradeBias = "bullish" | "bearish";
export type TradeEntryAction = "buy_to_open" | "sell_to_open";

export type TradeDirection = {
  position_side: TradePosition | null;
  contract_type: OptionContractType | null;
  signal_bias: TradeBias | null;
  entry_action: TradeEntryAction | null;
};

export type Heartbeat = {
  source_date: string;
  heartbeat_ts: string;
  loop_count: number | null;
  symbols_tracked: number | null;
  backtest: boolean;
};

export type TradeRow = TradeDirection & {
  trade_uid: string;
  run_date: string;
  order_id: string | null;
  symbol: string;
  underlying_symbol: string | null;
  asset_class: string | null;
  side: string | null;
  setup_type: string | null;
  option_dte: number | null;
  qty: number | null;
  entry_price: number | null;
  exit_price: number | null;
  entry_time: string | null;
  exit_time: string | null;
  exit_reason: string | null;
  pnl: number | null;
  status: string | null;
};

export type ValidationEvent = {
  run_date: string;
  event_type: string;
};

/** Raw order-event row backing the decision ledger. */
export type DecisionEvent = {
  run_date: string;
  event_ts: string | null;
  event_type: string;
  symbol: string;
  setup_type: string | null;
  side: string | null;
  details: JsonRecord | null;
};

/** One signal and the verdict the model gave it. */
export type Decision = {
  key: string;
  eventTs: string | null;
  symbol: string;
  setupType: string | null;
  verdict: "approved" | "vetoed";
  reason: string | null;
  confidence: number | null;
};

export type DecisionLog = {
  runDate: string | null;
  decisions: Decision[];
  signals: number;
  approved: number;
  filled: number;
};

export type TradeLifecycleRow = {
  lifecycle_uid: string;
  entry_order_id: string | null;
  exit_order_id: string | null;
  symbol: string;
  underlying_symbol: string | null;
  opened_at: string | null;
  closed_at: string | null;
  filled_qty: number | null;
  entry_fill_price: number | null;
  exit_fill_price: number | null;
  exit_reason: string | null;
  realized_pnl: number | null;
  status: string;
  details: JsonRecord | null;
  /** Recovered from entry telemetry for legacy lifecycle rows. */
  setup_type?: string | null;
  option_dte?: number | null;
};

export type PositionFill = {
  kind: "entry" | "partial_exit" | "final_exit";
  stage?: string | null;
  timestamp: string | null;
  qty: number | null;
  price: number | null;
  pnl: number | null;
  reason?: string | null;
};

/** Shared position shape for the overview list and detail dialog. */
export type OverviewPosition = {
  id: string;
  /** Stable broker-entry identity used when merging live and durable rows. */
  entry_order_id?: string | null;
  status: "open" | "closed";
  symbol: string;
  option_symbol: string;
  underlying_symbol: string | null;
  side: string | null;
  opened_at: string | null;
  closed_at: string | null;
  initial_qty: number | null;
  remaining_qty: number | null;
  entry_price: number | null;
  current_price: number | null;
  exit_price: number | null;
  realized_pnl: number | null;
  unrealized_pnl: number | null;
  total_pnl: number | null;
  return_pct: number | null;
  exit_reason: string | null;
  setup_type: string | null;
  dte: number | null;
  fills: PositionFill[];
};

export type CellStats = {
  trades: number;
  closed_trades: number;
  winning_trades: number;
  losing_trades: number;
  total_pnl: number | null;
  win_rate: number | null;
  average_win: number | null;
  average_loss: number | null;
  avg_realized_rr: number | null;
  avg_planned_rr: number | null;
  profit_factor: number | null;
};

export type BiggestLoser = {
  trade_uid: string | null;
  run_date: string | null;
  symbol: string | null;
  underlying_symbol: string | null;
  side: string | null;
  setup_type: string | null;
  option_dte: number | null;
  pnl: number;
  exit_reason: string | null;
  validation_reasoning: string | null;
};

export type OpsMetricsPayload = {
  range?: { start: string | null; end: string | null };
  overall?: CellStats & {
    max_drawdown: number | null;
    trades_per_day: number | null;
  };
  breakdowns?: {
    by_underlying?: Record<string, CellStats>;
    by_side?: Record<string, CellStats>;
    by_setup_type?: Record<string, CellStats>;
    by_dte_bucket?: Record<string, CellStats>;
  };
  biggest_losers?: BiggestLoser[];
  funnel?: {
    stages: Record<string, number>;
    rejection_reasons: Record<string, number>;
    llm_approval_rate: number | null;
  };
  generated_at?: string;
};

export type OpsMetricsDaily = {
  metric_date: string;
  payload: OpsMetricsPayload;
};

export type LiveExitPolicy = {
  stop_loss_pct: number;
  profit_target_pct: number;
  max_hold_minutes?: number;
  allow_overnight?: boolean;
};

export type LiveOpenPosition = {
  symbol: string;
  position_id?: string | null;
  entry_order_id?: string | null;
  option_symbol?: string | null;
  underlying_symbol?: string | null;
  asset_class?: string | null;
  qty: number;
  initial_qty?: number | null;
  remaining_qty?: number | null;
  side: string;
  entry_price: number | null;
  current_price: number | null;
  unrealized_pl: number;
  unrealized_plpc: number;
  opened_at?: string | null;
  realized_pnl?: number | null;
  fills?: PositionFill[];
  setup_type?: string | null;
  dte?: number | null;
  stop_mark?: number | null;
  tp_mark?: number | null;
  pct_to_stop?: number | null;
  pct_to_tp?: number | null;
};

export type LiveOrder = {
  id?: string;
  symbol: string;
  side: string;
  type: string;
  qty: number | null;
  filled_qty?: number | null;
  limit_price: number | null;
  stop_price?: number | null;
  filled_avg_price?: number | null;
  status: string;
  submitted_at: string | null;
  filled_at?: string | null;
};

export type LiveAccount = {
  equity: number | null;
  last_equity: number | null;
  buying_power: number | null;
  daily_pnl: number | null;
  daily_pnl_pct: number | null;
};

export type LiveStateRow = {
  source: string;
  session_date: string;
  heartbeat_ts: string;
  loop_count: number | null;
  equity: number | null;
  last_equity: number | null;
  daily_pnl: number | null;
  unrealized_pnl: number | null;
  open_position_count: number;
  open_positions: LiveOpenPosition[];
  session_stats: {
    fills?: number;
    realized_pnl?: number;
    wins?: number;
    losses?: number;
    closed?: Array<{
      position_id?: string;
      entry_order_id?: string | null;
      symbol: string;
      option_symbol?: string | null;
      underlying_symbol?: string | null;
      side?: string | null;
      opened_at?: string | null;
      initial_qty?: number | null;
      remaining_qty?: number | null;
      entry_price?: number | null;
      exit_price?: number | null;
      pnl: number;
      exit_reason: string;
      closed_at: string;
      fills?: PositionFill[];
    }>;
    session_end_reason?: string;
  };
  exit_policy: LiveExitPolicy;
  updated_at?: string;
};

export type LiveBlotterPayload = {
  account: LiveAccount;
  positions: LiveOpenPosition[];
  openOrders: LiveOrder[];
  todaysOrders: LiveOrder[];
  fetchedAt: string;
  exitPolicy: LiveExitPolicy;
  liveState: LiveStateRow | null;
  error?: string;
};
