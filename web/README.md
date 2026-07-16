# LLM Advisor — Ops Dashboard

Next.js (App Router) dashboard for the LLM Advisor paper-trading system,
modeled on the sports-edge ops dashboard. Intended deploy: Vercel at
`llm-advisor.drewboynton.com`.

## Pages

| Route | Contents |
|---|---|
| `/` | Equity curve, daily PnL bars, current equity/buying power, trades today, live-loop heartbeat freshness, recent sessions |
| `/trades` | Filterable trade table (date / underlying / side / setup / DTE) + biggest losers with LLM validation reasoning |
| `/breakdowns` | Win rate / PnL / RR grids per underlying, long vs short, MR vs TC, DTE buckets (cells with n < 10 greyed out) |
| `/funnel` | Signal → validation → execution funnel, rejection-reason histogram, LLM approval rate over time |
| `/command-center` | Env-gated private watchlist, mock opportunity feed, and Robinhood MCP status |

## Data access

Read-only. All queries go through Supabase PostgREST with the anon key —
the `llm_advisor_*` tables have public `SELECT` RLS policies (see
`../sql/` and the original DDL in the portfolio repo's migrations 005/007).
No secrets ship to the client; requests are made from server components with
5-minute revalidation.

Tables read:

- `llm_advisor_account_snapshots` — equity curve, balances (fed by the live loop via `account_snapshot.json` artifacts)
- `llm_advisor_backtest_runs` — daily PnL / session rollups
- `llm_advisor_backtest_trades` — enriched trade rows (underlying, setup, DTE, option metadata)
- `llm_advisor_runtime_heartbeats` — loop liveness
- `llm_advisor_order_events` — validation decisions for the approval-rate chart
- `llm_advisor_ops_metrics_daily` — precomputed rollup payload from `scripts/compute_ops_metrics.py`

## Local dev

```bash
cp .env.example .env.local   # fill in the shared Supabase project URL + anon key
npm install
npm run dev
```

Pages render graceful empty states when Supabase env vars are missing or
tables are empty (weekends, fresh deploys).

## Private command center

Set both server-only values and the public navigation flag:

```bash
COMMAND_CENTER_ENABLED=true
COMMAND_CENTER_PASSWORD=replace-with-a-long-random-password
NEXT_PUBLIC_COMMAND_CENTER_ENABLED=true
```

Manual gate check:

1. Leave the flags unset: `/command-center` returns 404 and no nav item appears.
2. Enable the flags: unauthenticated visits redirect to `/command-center/login`.
3. A wrong password remains blocked; the configured password sets an 8-hour HttpOnly cookie and renders mock data.

This shared-password gate is only a private scaffold. Replace it with identity-backed auth before enabling Robinhood connectivity. Watchlist edits currently persist only in browser `localStorage`.

## Deploy (Vercel)

1. Import the repo in Vercel with root directory `web/`.
2. Set `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_ANON_KEY`.
3. Add the `llm-advisor.drewboynton.com` domain to the project and create the
   CNAME on the drewboynton.com DNS zone.
