# Trade Funnel Baseline

Generated: 2026-07-15
Command: `.venv/bin/python scripts/report_trade_funnel.py`

| Date | Signals | Approved | Rejected | Attempts | Failed | Filled | Trades | PnL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025-04-30 | 151 | 0 | 0 | 0 | 0 | 0 | 0 | $0.00 |
| 2025-05-01 | 142 | 0 | 0 | 0 | 0 | 0 | 0 | $0.00 |
| 2025-05-02 | 143 | 0 | 0 | 0 | 0 | 0 | 0 | $0.00 |
| **2026-05-21 (operational live loop)** | **32** | **0** | **0** | **0** | **0** | **0** | **0** | **$0.00** |
| TOTAL | 468 | 0 | 0 | 0 | 0 | 0 | 0 | $0.00 |

Failure reasons: none recorded.
Exit reasons: none recorded.
Average entry position size: n/a.

## Interpretation

The operational 2026-05-21 run produced 32 signals and zero trades. Its local artifact set predates `order_events.jsonl`, so approval, attempt, and structured failure-reason counts are unavailable rather than observed zeros. The older dates are included for completeness but are pre-operational backtest-era logs.

Supabase enrichment could not run because this checkout has no `SUPABASE_URL` or Supabase key configured. Re-run with `--supabase` in the workflow environment to fill dates absent from disk and recover structured failure reasons.

The dominant failure mechanism remains the independently verified code-path diagnosis in `BACKTEST_GAPS.md`: the legacy $200 candidate cap rejects SPY-like contracts before quantity sizing (`premium_exceeds_risk_budget`). T2 corrects that mechanism; this statement is a diagnosis, not a measured failure-reason count from the incomplete baseline telemetry.
