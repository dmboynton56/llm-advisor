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

export type Heartbeat = {
  source_date: string;
  heartbeat_ts: string;
  loop_count: number | null;
  symbols_tracked: number | null;
  backtest: boolean;
};

export type TradeRow = {
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
