-- 005: Intraday live-state row published by src/telemetry/live_state.py every
-- LIVE_STATE_PUBLISH_TICKS loop ticks. One row per engine source; upsert-only,
-- so the table never grows. Additive: no existing table is modified.

create table if not exists llm_advisor_live_state (
  source text primary key default 'paper',
  session_date date not null,
  heartbeat_ts timestamptz not null,
  loop_count integer,
  equity numeric,
  last_equity numeric,
  daily_pnl numeric,
  unrealized_pnl numeric,
  open_position_count integer not null default 0,
  open_positions jsonb not null default '[]'::jsonb,
  -- [{symbol, option_symbol, underlying_symbol, asset_class, qty, side,
  --   entry_price, current_price, unrealized_pl, unrealized_plpc,
  --   opened_at, setup_type, dte}]
  session_stats jsonb not null default '{}'::jsonb,
  -- {fills, realized_pnl, wins, losses, closed: [{symbol, pnl, exit_reason, closed_at}],
  --  session_end_reason?}
  exit_policy jsonb not null default '{}'::jsonb,
  -- {stop_loss_pct, profit_target_pct, max_hold_minutes, allow_overnight}
  updated_at timestamptz not null default now()
);

alter table llm_advisor_live_state enable row level security;

drop policy if exists "public read llm advisor live state" on llm_advisor_live_state;

create policy "public read llm advisor live state"
  on llm_advisor_live_state for select using (true);

grant select on table public.llm_advisor_live_state to anon, authenticated;
grant select, insert, update, delete on table public.llm_advisor_live_state to service_role;
