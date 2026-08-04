-- 009: Backfill dashboard fields from the telemetry that already records them.
-- Only NULL/blank values are changed; authoritative values are preserved.

with execution_enrichment as (
  select distinct on (order_id)
    order_id,
    nullif(btrim(setup_type), '') as event_setup_type,
    nullif(btrim(details #>> '{order,option_plan,setup_type}'), '') as plan_setup_type,
    nullif(btrim(details -> 'trade_plan' ->> 'setup'), '') as trade_plan_setup,
    case
      when (details #>> '{order,option_plan,dte}') ~ '^-?[0-9]+$'
        then (details #>> '{order,option_plan,dte}')::integer
      else null
    end as event_option_dte
  from public.llm_advisor_order_events
  where event_type = 'execution_succeeded'
    and order_id is not null
  order by order_id, event_ts desc
)
update public.llm_advisor_backtest_trades as trades
set
  setup_type = coalesce(
    nullif(btrim(trades.setup_type), ''),
    nullif(btrim(trades.option_metadata ->> 'setup_type'), ''),
    enrichment.event_setup_type,
    enrichment.plan_setup_type,
    enrichment.trade_plan_setup
  ),
  option_dte = coalesce(
    trades.option_dte,
    case
      when (trades.option_metadata ->> 'dte') ~ '^-?[0-9]+$'
        then (trades.option_metadata ->> 'dte')::integer
      else null
    end,
    enrichment.event_option_dte
  ),
  updated_at = now()
from execution_enrichment as enrichment
where trades.order_id = enrichment.order_id
  and (
    trades.setup_type is null
    or btrim(trades.setup_type) = ''
    or trades.option_dte is null
  );

with latest_snapshot as (
  select distinct on (snapshot_date)
    snapshot_date,
    equity
  from public.llm_advisor_account_snapshots
  where equity is not null
  order by snapshot_date, captured_at desc
)
update public.llm_advisor_backtest_runs as runs
set
  final_equity = snapshot.equity,
  updated_at = now()
from latest_snapshot as snapshot
where runs.run_date = snapshot.snapshot_date
  and runs.final_equity is null;
