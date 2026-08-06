-- 010: Persist the auditable daily-bias reading used by the paper loop.
-- ML remains the primary value; LLM output is stored as an opinion/diagnostic.

create table if not exists public.llm_advisor_daily_bias (
  bias_date date not null,
  symbol text not null,
  ml_bias text not null default 'unavailable',
  ml_confidence numeric,
  llm_bias text,
  llm_confidence numeric,
  agreement text not null default 'unknown',
  bias_available boolean not null default false,
  bias_error text,
  llm_reasoning text,
  context_version text,
  generated_at timestamptz,
  source_artifact text,
  updated_at timestamptz not null default now(),
  primary key (bias_date, symbol),
  check (ml_bias in ('bullish', 'bearish', 'choppy', 'unavailable')),
  check (llm_bias is null or llm_bias in ('bullish', 'bearish', 'choppy', 'unavailable')),
  check (agreement in ('agree', 'partial', 'disagree', 'unknown'))
);

create index if not exists idx_llm_advisor_daily_bias_symbol_date
  on public.llm_advisor_daily_bias(symbol, bias_date desc);

alter table public.llm_advisor_daily_bias enable row level security;

drop policy if exists "public read llm advisor daily bias" on public.llm_advisor_daily_bias;

create policy "public read llm advisor daily bias"
  on public.llm_advisor_daily_bias for select using (true);

grant select on table public.llm_advisor_daily_bias to anon, authenticated;
grant select, insert, update, delete on table public.llm_advisor_daily_bias to service_role;
