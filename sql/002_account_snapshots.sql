-- 002: Alpaca paper-account equity/PnL time series for the ops dashboard.
-- Written by scripts/run_eod_aggregate.py from processed/account_snapshot.json artifacts.

create table if not exists llm_advisor_account_snapshots (
  id bigserial primary key,
  snapshot_date date not null,
  captured_at timestamptz not null,
  equity numeric,
  last_equity numeric,
  buying_power numeric,
  daily_pnl numeric,
  daily_pnl_pct numeric,
  source text not null default 'alpaca_paper',
  inserted_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (snapshot_date, captured_at)
);

create index if not exists idx_llm_advisor_account_snapshots_date
  on llm_advisor_account_snapshots(snapshot_date desc);

create index if not exists idx_llm_advisor_account_snapshots_captured
  on llm_advisor_account_snapshots(captured_at desc);

alter table llm_advisor_account_snapshots enable row level security;

drop policy if exists "public read llm advisor account snapshots" on llm_advisor_account_snapshots;

create policy "public read llm advisor account snapshots"
  on llm_advisor_account_snapshots for select using (true);

grant select on table public.llm_advisor_account_snapshots to anon, authenticated;
grant select, insert, update, delete on table public.llm_advisor_account_snapshots to service_role;
