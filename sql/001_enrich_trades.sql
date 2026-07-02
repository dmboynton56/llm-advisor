-- 001: Enrich llm_advisor_backtest_trades for ops-dashboard breakdowns.
-- Additive-only: the portfolio metrics route keeps reading the original columns.
-- Base DDL lives in personal-portfolio/supabase/migrations/005_llm_advisor_telemetry.sql;
-- new llm-advisor DDL lives in this repo going forward.

alter table llm_advisor_backtest_trades
  add column if not exists underlying_symbol text,
  add column if not exists asset_class text,
  add column if not exists side text,
  add column if not exists setup_type text,
  add column if not exists option_dte integer,
  add column if not exists option_metadata jsonb,
  add column if not exists entry_time timestamptz,
  add column if not exists exit_time timestamptz,
  add column if not exists exit_reason text;

create index if not exists idx_llm_advisor_trades_underlying
  on llm_advisor_backtest_trades(underlying_symbol);

create index if not exists idx_llm_advisor_trades_setup_type
  on llm_advisor_backtest_trades(setup_type);
