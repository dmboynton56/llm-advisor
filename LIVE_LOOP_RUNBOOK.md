# Live Loop Runbook

This runbook defines a successful LLM Advisor paper-trading day.

## Daily Flow

1. `Premarket` completes and uploads `premarket-context`.
2. `Live Trading Loop` downloads the same-day premarket artifact, reconciles
   warehouse positions against Alpaca, runs during the configured entry window,
   monitors any open positions until flat or the hard cutoff, and uploads
   `llm-advisor-daily-news-*`.
3. The telemetry artifact should include:
   - `processed/premarket_context.json`
   - `processed/live_loop_log.jsonl`
   - `processed/order_events.jsonl` when signals or order attempts occur
   - `processed/session_summary.json`
4. `EOD Aggregate` runs after the live workflow completes unless the live
   workflow was cancelled. It ingests the artifact, optionally merges BigQuery,
   and upserts Supabase telemetry rows.
5. The portfolio reads Supabase through `/api/llm-advisor/metrics`.

Google Cloud Scheduler owns the daily timing and dispatches the GitHub
`workflow_dispatch` events. The workflows still check the NYSE calendar and
fail closed on weekends or holidays.

## Success Criteria

A day with no orders can still be successful when:

- the live loop reaches the end of the configured session,
- a shutdown heartbeat is present in `live_loop_log.jsonl`,
- `session_summary.json` exists with `total_trades=0`, and
- EOD writes a run row and heartbeat row to Supabase.

A day with signals but no orders is successful only if the rejection path is
visible in either BigQuery or the artifact:

- signal detected,
- LLM validation approved/rejected/error,
- risk/reward, max-position, or execution failure reason when execution is
attempted.

A day with paper orders is successful only if the lifecycle can be reconstructed:

- signal and validation,
- execution attempt,
- Alpaca order id and submitted status,
- open trade row,
- position update or exit,
- final summary and EOD Supabase sync.

## Verification Commands

Inspect a live-loop run:

```bash
gh run view <run-id> --repo dmboynton56/llm-advisor --json status,conclusion,jobs,url
gh run view <run-id> --repo dmboynton56/llm-advisor --log
gh run download <run-id> --repo dmboynton56/llm-advisor --dir /tmp/llm-advisor-run-<run-id>
```

Validate a downloaded telemetry artifact without writing to Supabase:

```bash
python3 scripts/run_eod_aggregate.py \
  --data-dir /tmp/llm-advisor-run-<run-id>/<artifact-name>/data/daily_news \
  --date YYYY-MM-DD \
  --no-bigquery \
  --dry-run
```

The dry run should report at least one run row and one heartbeat row. Trade rows
are expected only on days where a paper order is actually submitted.

## Current Reliability Rules

- Alpaca is the source of truth for open live positions at startup. Warehouse
  `positions` rows that Alpaca does not confirm are closed/deleted before
  strategy recovery runs, and the action is written to `order_events.jsonl`.
- `TRADING_WINDOW_END` is the last time the loop may open a new trade, not the
  process shutdown time. After that boundary, the loop keeps running only to
  monitor real open positions until they hit bracket exits or the
  `END_OF_DAY_CLOSE_TIME` hard cutoff.
- If Alpaca is flat after the entry window closes, the live loop exits with an
  `entry_window_closed_flat` summary reason. If positions remain open, the loop
  keeps calling `TradeTracker.update_positions()` so warehouse rows are deleted
  when Alpaca reports the position closed.
- At the hard cutoff, the loop requests market closes for any remaining Alpaca
  positions and polls `TradeTracker` before writing the final summary.
- BigQuery trade retrieval must not compare `TIMESTAMP` columns to `DATE`
  parameters directly.
- LLM validation parse errors are not approvals.
- `session_summary.json` must be written even when warehouse summary retrieval
  fails; degraded summaries are acceptable and should still let EOD proceed.
- EOD should ingest artifacts from completed live workflows even if the live
  workflow concluded as failed after uploading telemetry.
- EOD should no-op, not fail, when a failed live workflow uploaded only an
  artifact anchor and no telemetry payload files.
- Signal/order lifecycle evidence lives in `order_events.jsonl` and Supabase
  `llm_advisor_order_events`.
