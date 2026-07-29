-- Daily close-to-close reconciliation between broker equity PnL and booked lifecycles.

create table if not exists llm_advisor_broker_reconciliation_daily (
  reconciliation_date date primary key,
  booked_realized_pnl numeric not null default 0,
  broker_daily_pnl numeric,
  pnl_gap numeric,
  lifecycle_exit_count integer not null default 0,
  tolerance numeric not null default 50,
  status text not null default 'pending',
  details jsonb not null default '{}'::jsonb,
  inserted_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_llm_advisor_broker_reconciliation_status
  on llm_advisor_broker_reconciliation_daily(status, reconciliation_date desc);

alter table llm_advisor_broker_reconciliation_daily enable row level security;

drop policy if exists "public read llm advisor broker reconciliation"
  on llm_advisor_broker_reconciliation_daily;

create policy "public read llm advisor broker reconciliation"
  on llm_advisor_broker_reconciliation_daily for select using (true);

grant select on table public.llm_advisor_broker_reconciliation_daily
  to anon, authenticated;
revoke insert, update, delete, truncate, references, trigger
  on table public.llm_advisor_broker_reconciliation_daily
  from anon, authenticated;
grant select, insert, update, delete
  on table public.llm_advisor_broker_reconciliation_daily to service_role;

create table if not exists llm_advisor_trade_lifecycles (
  lifecycle_uid text primary key,
  entry_order_id text,
  exit_order_id text,
  symbol text not null,
  underlying_symbol text,
  opened_at timestamptz,
  closed_at timestamptz,
  filled_qty numeric,
  entry_fill_price numeric,
  exit_fill_price numeric,
  protective_stop_order_id text,
  protective_stop_price numeric,
  exit_reason text,
  realized_pnl numeric,
  status text not null default 'open',
  details jsonb not null default '{}'::jsonb,
  inserted_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create unique index if not exists idx_llm_advisor_lifecycle_entry_order
  on llm_advisor_trade_lifecycles(entry_order_id)
  where entry_order_id is not null and entry_order_id <> '';
create index if not exists idx_llm_advisor_lifecycle_closed
  on llm_advisor_trade_lifecycles(closed_at desc);

alter table llm_advisor_trade_lifecycles enable row level security;

drop policy if exists "public read llm advisor trade lifecycles"
  on llm_advisor_trade_lifecycles;
create policy "public read llm advisor trade lifecycles"
  on llm_advisor_trade_lifecycles for select using (true);

grant select on table public.llm_advisor_trade_lifecycles to anon, authenticated;
revoke insert, update, delete, truncate, references, trigger
  on table public.llm_advisor_trade_lifecycles from anon, authenticated;
grant select, insert, update, delete
  on table public.llm_advisor_trade_lifecycles to service_role;
