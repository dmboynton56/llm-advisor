-- Backfill legacy broker-position exits at the clean lifecycle grain.
-- Historical rows remain explicitly marked as pre-fill-telemetry estimates.

insert into llm_advisor_trade_lifecycles
(lifecycle_uid,symbol,underlying_symbol,opened_at,closed_at,filled_qty,
 entry_fill_price,exit_fill_price,exit_reason,realized_pnl,status,details)
select
  'legacy:' || event_uid,
  symbol,
  case when length(symbol) > 15 then left(symbol, length(symbol)-15) else null end,
  case
    when nullif(details->>'hold_minutes','') is not null
      then event_ts - ((details->>'hold_minutes')::double precision * interval '1 minute')
    else null
  end,
  event_ts,
  nullif(details#>>'{position,qty}','')::numeric,
  nullif(details#>>'{position,entry_price}','')::numeric,
  nullif(details#>>'{position,current_price}','')::numeric,
  details->>'reason',
  nullif(details#>>'{position,unrealized_pl}','')::numeric,
  'closed',
  jsonb_build_object(
    'source_event_uid', event_uid,
    'legacy_mark_based_exit', true,
    'actual_exit_fill_unavailable', true
  )
from llm_advisor_order_events
where event_type='option_exit_requested'
on conflict (lifecycle_uid) do nothing;

with booked as (
  select
    (closed_at at time zone 'America/New_York')::date as reconciliation_date,
    coalesce(sum(realized_pnl),0) as booked_realized_pnl,
    count(*)::integer as lifecycle_exit_count,
    max(closed_at) as final_exit_at
  from llm_advisor_trade_lifecycles
  where status='closed' and closed_at is not null
  group by 1
),
snapshots as (
  select distinct on (snapshot_date)
    snapshot_date, daily_pnl, captured_at
  from llm_advisor_account_snapshots
  order by snapshot_date, captured_at desc
)
insert into llm_advisor_broker_reconciliation_daily
(reconciliation_date,booked_realized_pnl,broker_daily_pnl,pnl_gap,
 lifecycle_exit_count,tolerance,status,details)
select
  b.reconciliation_date,
  b.booked_realized_pnl,
  case when s.captured_at >= b.final_exit_at then s.daily_pnl else null end,
  case
    when s.captured_at >= b.final_exit_at
      then s.daily_pnl-b.booked_realized_pnl
    else null
  end,
  b.lifecycle_exit_count,
  50,
  case
    when s.daily_pnl is null or s.captured_at < b.final_exit_at then 'pending'
    when abs(s.daily_pnl-b.booked_realized_pnl) <= 50 then 'ok'
    else 'alert'
  end,
  jsonb_build_object(
    'booked_source','legacy_option_exit_requested',
    'snapshot_captured_at',s.captured_at,
    'final_exit_at',b.final_exit_at,
    'snapshot_after_final_exit',coalesce(s.captured_at >= b.final_exit_at,false)
  )
from booked b
left join snapshots s on s.snapshot_date=b.reconciliation_date
on conflict (reconciliation_date) do update set
  booked_realized_pnl=excluded.booked_realized_pnl,
  broker_daily_pnl=excluded.broker_daily_pnl,
  pnl_gap=excluded.pnl_gap,
  lifecycle_exit_count=excluded.lifecycle_exit_count,
  tolerance=excluded.tolerance,
  status=excluded.status,
  details=excluded.details,
  updated_at=now();
