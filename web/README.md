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
| `/command-center` | Env-gated private surface: **live Alpaca paper blotter** (positions, open orders, software stop/TP, loop-health banner), watchlist mocks, Robinhood MCP status |

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
- `llm_advisor_live_state` — single upserted intraday row from the live loop (`sql/005_live_state.sql`); powers the overview "Live session" card and the blotter loop-health banner

## Local dev

```bash
cp .env.example .env.local   # fill in the shared Supabase project URL + anon key
npm install
npm run dev
```

Pages render graceful empty states when Supabase env vars are missing or
tables are empty (weekends, fresh deploys).

## Private command center + live blotter

Set both server-only values and the public navigation flag:

```bash
COMMAND_CENTER_ENABLED=true
COMMAND_CENTER_PASSWORD=replace-with-a-long-random-password
NEXT_PUBLIC_COMMAND_CENTER_ENABLED=true
```

For the live blotter (`GET /command-center/api/live`), also set **server-only** paper Alpaca credentials on Vercel:

```bash
ALPACA_API_KEY=PK...
ALPACA_SECRET_KEY=...
ALPACA_PAPER_TRADING=true
```

The route self-checks: 404 when command center is disabled, 401 without the gate cookie, 500 refusal (no Alpaca call) when `ALPACA_PAPER_TRADING` is not `true`. Keys never appear in the client bundle or JSON responses.

Manual gate check:

1. Leave the flags unset: `/command-center` returns 404 and no nav item appears.
2. Enable the flags: unauthenticated visits redirect to `/command-center/login`.
3. A wrong password remains blocked; the configured password sets an 8-hour HttpOnly cookie and renders the blotter + mock panels.

This shared-password gate is only a private scaffold. Replace it with identity-backed auth before enabling Robinhood connectivity. Watchlist edits currently persist only in browser `localStorage`.

### Live-state loop publish (GitHub Actions)

The live loop upserts `llm_advisor_live_state` each tick when `SUPABASE_DB_*` secrets are present (wired in `.github/workflows/live_loop.yml`). Apply `sql/005_live_state.sql` to the shared Supabase project once. Publish failures are swallowed and self-disable after 10 consecutive errors so trading is never blocked.

Verification sketch (paper session):

1. Deploy web with Alpaca + command-center env → open `/command-center` → marks match Alpaca `/v2/positions`.
2. Dispatch `live_loop.yml` → within ~2 minutes `select heartbeat_ts, loop_count from llm_advisor_live_state` advances.
3. Cancel the workflow mid-session with an open position → blotter shows the red "NO stop enforcement" banner within ~3 minutes.

## Deploy (Vercel)

1. Import the repo in Vercel with root directory `web/`.
2. Set `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_ANON_KEY`.
3. For the blotter: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_PAPER_TRADING=true`, plus command-center password flags above.
4. Add the `llm-advisor.drewboynton.com` domain to the project and create the
   CNAME on the drewboynton.com DNS zone.
