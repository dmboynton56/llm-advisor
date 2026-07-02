-- 004: Grant anon/authenticated SELECT on core llm_advisor telemetry tables.
-- Migration 005 created RLS policies but omitted table-level grants; the ops
-- dashboard reads via PostgREST with the publishable (anon) key on Vercel.

grant select on table public.llm_advisor_backtest_runs to anon, authenticated;
grant select on table public.llm_advisor_backtest_trades to anon, authenticated;
grant select on table public.llm_advisor_runtime_heartbeats to anon, authenticated;
