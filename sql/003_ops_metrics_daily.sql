-- 003: Daily ops-metrics rollup so the dashboard reads one row instead of
-- recomputing aggregations. Payload shape is produced by src/analytics/ops_metrics.py.

create table if not exists llm_advisor_ops_metrics_daily (
  metric_date date primary key,
  payload jsonb not null default '{}'::jsonb,
  inserted_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_llm_advisor_ops_metrics_daily_date
  on llm_advisor_ops_metrics_daily(metric_date desc);

alter table llm_advisor_ops_metrics_daily enable row level security;

drop policy if exists "public read llm advisor ops metrics" on llm_advisor_ops_metrics_daily;

create policy "public read llm advisor ops metrics"
  on llm_advisor_ops_metrics_daily for select using (true);

grant select on table public.llm_advisor_ops_metrics_daily to anon, authenticated;
grant select, insert, update, delete on table public.llm_advisor_ops_metrics_daily to service_role;
