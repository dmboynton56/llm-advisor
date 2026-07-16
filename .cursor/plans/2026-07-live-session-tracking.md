# Plan: Live session tracking (open PnL, positions, working orders during market hours)

Repo: `llm-advisor`. Round: 2026-07. Self-contained — cross-repo facts are inlined; do not read sibling repos.

## Verdict

**Hybrid: A (dashboard polls Alpaca) for broker truth + a minimal B (loop publishes one live-state row to Supabase) for loop liveness and exit context. A ships first; B is small and lands in the same round.**

Why not A alone or B alone — this fell out of the recon, not preference:

- Option stops/TP are **software-enforced by the loop**, not broker bracket legs. `TradeTracker._option_exit_reason` (`src/execution/trade_tracker.py:255-283`) decides exits from `unrealized_plpc` against `OPTION_PROFIT_TARGET_PCT=0.25` / `OPTION_STOP_LOSS_PCT=0.35` (`src/core/config.py:120-121`, `.github/workflows/live_loop.yml`). **If the GH Actions loop dies, an open position has no stop protection.** The operator therefore has two distinct in-session questions: (1) "is my put alive and at what mark?" — only Alpaca answers this reliably when the loop is dead, and (2) "is the loop alive to enforce my software stop?" — only loop-published telemetry answers this. Neither architecture alone covers both.
- The codebase **never lists open orders** — `get_orders`/`GetOrdersRequest` appear nowhere in `src/`. A working-orders view is new capability, and an Alpaca proxy route is by far the cheapest way to get it.
- B alone shows stale data exactly when it matters most (loop crash with an open position). A alone can't show the loop's exit thresholds, setup context, or that the stop engine is running.
- The "B is better long-term for joined context" argument holds, but v1 doesn't need a full intraday `order_events` flush or a positions time series. A single upserted `live_state` row per engine gives loop-heartbeat freshness, exit policy, and session stats at trivial cost. Richer history (intraday equity ticks) is a bolt-on follow-up.

So: accept the prior informal lean ("A as bootstrap, B as durable path") in spirit, but B-minimal is not deferred — it's required for the loop-dead-with-open-position alert, which is the single most valuable thing this feature can show.

## Current state (recon summary, with file refs)

**The gap:** everything the dashboard shows is EOD-batch. During market hours there is no surface for open marks, distance to stop/TP, or working orders. Concrete instance (2026-07-16): long 2× `QQQ260724P00711000` @ $8.75 at 09:31 ET, software TP exit @ $11.58 at 09:48 ET — none of it visible anywhere until EOD ingest.

- **The loop already has all the live data, it just doesn't publish it.** Every ~60s tick (`--fast 60` in `live_loop.yml`) the loop calls `update_positions()` (`src/live/loop.py:2129-2135` in-session, `:1442-1449` post-entry-window), which hits Alpaca `/v2/positions` via `OptionsOrderManager.get_open_positions()` (`src/execution/options_order_manager.py:65-71`). The position dicts (`_position_to_dict`, `options_order_manager.py:162-185`) carry `symbol`, `option_symbol`, `qty`, `side`, `entry_price` (from cost_basis/qty/100), `current_price` (from market_value), `unrealized_pl`, `unrealized_plpc`, `asset_class`.
- **Tracker context exists in-memory:** `TradeTracker._order_meta` holds `opened_at`, `entry_price`, `option_plan`, `asset_class`, `underlying_symbol` per symbol (`trade_tracker.py:53-66`, registration at `loop.py:2022-2035`). `option_dte()` (`trade_tracker.py:31-33`) derives DTE from the OCC symbol. Exit thresholds live on `settings.options`.
- **Telemetry is EOD-only.** `capture_account_snapshot()` runs at loop start and session finalize only (`loop.py:1333`, `:1352-1364`), appending to `processed/account_snapshot.json`; heartbeats are parsed from `live_loop_log.jsonl` and order events from `order_events.jsonl` — all ingested by `scripts/run_eod_aggregate.py` after the session (`eod_aggregate.yml` triggers on `workflow_run` of the live loop). Supabase gets nothing intraday.
- **Live loop workflow has no Supabase credentials.** `live_loop.yml` env has Alpaca/GCP/LLM keys only; the `SUPABASE_DB_*` secrets are wired only into `eod_aggregate.yml` (lines 30-35). They exist in the same repo secret store, so adding them to the live loop is a one-line-per-secret change.
- **EOD Supabase write path is direct Postgres**, not PostgREST: psycopg2 with `SUPABASE_DB_HOST/NAME/USER/PORT/PASSWORD` (`scripts/run_eod_aggregate.py:771-785`), upserting `llm_advisor_backtest_runs`, `llm_advisor_backtest_trades`, `llm_advisor_runtime_heartbeats` (`source_date, heartbeat_ts, loop_count, symbols_tracked, backtest, source_file, updated_at`), `llm_advisor_account_snapshots`, `llm_advisor_order_events` (`event_uid, run_date, event_ts, event_type, symbol, loop_count, setup_type, side, entry_price, z_score, order_id, details, source_file, updated_at`).
- **Dashboard (`web/`)** is a Next.js App Router app (Vercel), server components with `dynamic = "force-dynamic"`, reading Supabase via raw PostgREST fetches (`web/lib/supabase.ts` — note the publishable-key header quirk: `sb_publishable_` keys must not send `Authorization: Bearer`). Data layer: `web/lib/data.ts`. Overview (`web/app/page.tsx`) shows EOD equity curve + heartbeat freshness with a 30-hour "Healthy" window — useless intraday. There is **no `web/app/api/` directory** — no route handlers exist yet.
- **Command center** is gated by `web/proxy.ts` (cookie `llm_advisor_command_center`, sha256 of `COMMAND_CENTER_PASSWORD`, matcher `/command-center/:path*`) and is still mock data (`web/lib/commandCenter.ts` returns hardcoded rows). ⚠️ The proxy returns `NextResponse.next()` when `COMMAND_CENTER_ENABLED !== "true"` — the page 404s via `notFound()`, but a **route handler under that path would be exposed**; any new API route must self-check env + cookie.
- **Cross-repo contract (inlined):** the portfolio site reads `llm_advisor_*` tables via its `/api/llm-advisor/metrics` route. Base DDL for runs/trades/heartbeats/order_events lives in the portfolio repo's migrations 005/007; **new DDL is owned by this repo under `llm-advisor/sql/`** (established by `sql/001`–`004`). All changes must be additive; existing tables get no breaking alters (this plan adds one new table and touches nothing else).

## Phase 1 — Broker-truth live blotter (architecture A)

Ships value in one PR with zero loop/schema changes. Everything lives under the command-center gate because it proxies live account data on a public site.

1. **`web/app/command-center/api/live/route.ts`** — server-only route handler (`runtime: "nodejs"`, `dynamic: "force-dynamic"`):
   - Guards, in order: return 404 unless `COMMAND_CENTER_ENABLED === "true"`; return 401 unless the request cookie equals `commandCenterToken(COMMAND_CENTER_PASSWORD)` (import from `web/lib/commandCenterAuth.ts` — do not rely on `proxy.ts` alone, per the gap above); return 500 refusal unless `ALPACA_PAPER_TRADING === "true"` (hard paper-only guard: never construct the live base URL).
   - Fetches from `https://paper-api.alpaca.markets` with `APCA-API-KEY-ID`/`APCA-API-SECRET-KEY` headers (same env names as Python: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`):
     - `/v2/account` → equity, last_equity, buying_power → daily_pnl = equity − last_equity.
     - `/v2/positions` → per position: symbol, qty, side, avg_entry_price, current_price, unrealized_pl, unrealized_plpc, asset_class. Derive option entry/mark the same way `_position_to_dict` does (÷100 multiplier) so numbers match the loop's view.
     - `/v2/orders?status=open&nested=true` → working orders (symbol, side, type, qty, limit_price, status, submitted_at).
     - `/v2/orders?status=closed&after=<today 09:00 ET as ISO>&limit=200` → fills today (count filled orders; show symbol/side/qty/filled_avg_price/filled_at).
   - Response JSON: `{ account, positions[], openOrders[], todaysOrders[], fetchedAt }`. Cache: none (`cache: "no-store"`).
   - Derived fields computed server-side per option position: `stop_mark = entry × (1 − stopLossPct)`, `tp_mark = entry × (1 + profitTargetPct)` (long premium; both software levels), `pct_to_stop` / `pct_to_tp` from `unrealized_plpc`. Thresholds: read the Phase-2 `live_state.exit_policy` when present, else env `OPTION_STOP_LOSS_PCT`/`OPTION_PROFIT_TARGET_PCT`, else code defaults 0.35/0.25. Label them "software stop (loop-enforced)" in the UI.
2. **`web/components/command-center/LiveBlotter.tsx`** — client component, `setInterval` poll of the route every 20s (pause when `document.hidden`):
   - Account strip: equity, daily PnL ($/%), buying power, open uPnL total, realized-so-far approximation (`daily_pnl − Σ unrealized_pl`).
   - Positions table: symbol (OCC + parsed underlying/expiry/strike/type for readability), qty, entry, mark, uPnL $, uPnL %, distance to software stop/TP, DTE.
   - Working-orders table; fills-today list; last-refresh timestamp + error state when the route fails.
3. **Wire into `web/app/command-center/page.tsx`** above the existing mock panels (leave the mocks; they're a separate cleanup).
4. **`web/lib/types.ts`** additions for the route payload; small OCC-symbol parser in `web/lib/format.ts` (mirror the regex in `trade_tracker.py:17`).
5. **`web/.env.example`**: document `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_PAPER_TRADING=true` as server-only (never `NEXT_PUBLIC_*`), command-center-gated.

## Phase 2 — Loop publishes live state (architecture B, minimal)

1. **`sql/005_live_state.sql`** — see §Schema. One row per engine source, upserted every tick; no growth, no retention problem.
2. **`src/telemetry/live_state.py`** (new module):
   - `connect()` — psycopg2 connection from the same `SUPABASE_DB_*` env contract as `run_eod_aggregate.py:771-785` (~20 lines, duplicated deliberately so the EOD script is untouched; consolidating both into a shared helper is a follow-up).
   - `build_live_state_row(trade_tracker, order_manager, settings, *, session_date, loop_count, session_closed)` — pure function returning the row dict: equity/last_equity via `order_manager.get_account_equity()` + account fetch, positions from `trade_tracker.get_all_positions()` enriched with `_order_meta` (opened_at, setup_type from option_plan, underlying) and `option_dte()`, `exit_policy` from `settings.options` (stop_loss_pct, profit_target_pct, max_hold_minutes, allow_overnight), `session_stats` from `session_closed` (fills, realized pnl, wins/losses).
   - `publish_live_state(row)` — connect-per-write with 5s timeout, upsert `ON CONFLICT (source) DO UPDATE`. **Never raises to the caller**; logs and increments a consecutive-failure counter, disabling itself after 10 straight failures (one warning log, not per-tick spam).
3. **`src/execution/trade_tracker.py`** — add `self.session_closed: List[Dict]` and append `{symbol, pnl, exit_reason, closed_at}` in both close paths (the option-exit path `_persist_closed_position` caller at `:246` and the disappeared-position path at `:102-152`); expose `get_session_closed()`. Additive; no behavior change.
4. **`src/live/loop.py`** — after the in-session `update_positions(...)` call (`:2129-2135`), publish every `LIVE_STATE_PUBLISH_TICKS` ticks (env, default 1 → 60s; live-mode only, never backtest). Also publish once from `finalize_live_session` (`:1352`) with `session_end_reason` folded into `session_stats` so the dashboard can render "session complete" instead of "stale".
5. **`.github/workflows/live_loop.yml`** — add `SUPABASE_DB_HOST/NAME/USER/PORT/PASSWORD` to the job env from existing repo secrets; add `LIVE_STATE_PUBLISH_TICKS: 1`.
6. **Unit tests** (`tests/unit/`): `build_live_state_row` against fixture tracker/positions (option + stock, empty session, closed-trades stats); tracker `session_closed` accrual in both close paths. No network in tests.

## Phase 3 — Dashboard reads live state

1. **`web/lib/data.ts`**: `getLiveState()` selecting the `source='paper'` row; type in `web/lib/types.ts`.
2. **Overview (`web/app/page.tsx`)**: "Live session" section rendered when `heartbeat_ts` is fresher than 3 minutes: loop-alive badge, equity, daily PnL, open uPnL, compact open-positions list (symbol, uPnL%), session stats. When stale during market hours (weekday 09:30–16:00 ET), show a neutral "loop offline" note. Table is public-read like the rest of `llm_advisor_*` — the row contains only what `llm_advisor_account_snapshots` already exposes publicly plus position marks; acceptable for paper, revisit before any real-money flow.
3. **Command center `LiveBlotter`**: merge the live_state row (fetched server-side alongside the Alpaca route data, or via a second lightweight fetch) to render the one alert that motivates this whole plan: **red banner when Alpaca shows open positions but `live_state.heartbeat_ts` is stale >3 min during the session — "positions open with NO stop enforcement (loop down)"**. Also use `exit_policy` from the row as the authoritative threshold source for distance-to-stop/TP.

## Phase 4 — Follow-ups (explicitly not v1)

- `llm_advisor_live_ticks` append table (every ~5 min: equity, uPnL, open count) for an intraday equity sparkline and post-session replay.
- Mid-day `order_events` flush so validation/exit reasoning is joinable intraday.
- Consolidate the psycopg2 connect helper shared by `run_eod_aggregate.py` and `live_state.py`.
- Replace command-center mock watchlist/opportunities with real data.

## Schema (new DDL, owned by this repo in `llm-advisor/sql/`)

```sql
-- 005: Intraday live-state row published by src/telemetry/live_state.py every
-- LIVE_STATE_PUBLISH_TICKS loop ticks. One row per engine source; upsert-only,
-- so the table never grows. Additive: no existing table is modified.

create table if not exists llm_advisor_live_state (
  source text primary key default 'paper',
  session_date date not null,
  heartbeat_ts timestamptz not null,
  loop_count integer,
  equity numeric,
  last_equity numeric,
  daily_pnl numeric,
  unrealized_pnl numeric,
  open_position_count integer not null default 0,
  open_positions jsonb not null default '[]'::jsonb,
  -- [{symbol, option_symbol, underlying_symbol, asset_class, qty, side,
  --   entry_price, current_price, unrealized_pl, unrealized_plpc,
  --   opened_at, setup_type, dte}]
  session_stats jsonb not null default '{}'::jsonb,
  -- {fills, realized_pnl, wins, losses, closed: [{symbol, pnl, exit_reason, closed_at}],
  --  session_end_reason?}
  exit_policy jsonb not null default '{}'::jsonb,
  -- {stop_loss_pct, profit_target_pct, max_hold_minutes, allow_overnight}
  updated_at timestamptz not null default now()
);

alter table llm_advisor_live_state enable row level security;

drop policy if exists "public read llm advisor live state" on llm_advisor_live_state;

create policy "public read llm advisor live state"
  on llm_advisor_live_state for select using (true);

grant select on table public.llm_advisor_live_state to anon, authenticated;
grant select, insert, update, delete on table public.llm_advisor_live_state to service_role;
```

Note: the loop writes via direct Postgres (`SUPABASE_DB_USER`, typically `postgres`), same as EOD — RLS/grants above only govern dashboard reads.

## Secrets / deploy notes

- **Vercel (web):** add `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_PAPER_TRADING=true` as server-only env vars (Production + Preview). Never `NEXT_PUBLIC_*`; never committed. These are paper keys — the route hard-refuses if `ALPACA_PAPER_TRADING` isn't `"true"`. `COMMAND_CENTER_ENABLED=true` + `COMMAND_CENTER_PASSWORD` must be set for the blotter to be reachable at all.
- **GitHub Actions (`live_loop.yml`):** add the five existing `SUPABASE_DB_*` repo secrets to the job env (already consumed by `eod_aggregate.yml`; no new secret values needed).
- **Failure isolation:** telemetry publish failures must never affect trading — swallow-and-log with self-disable, mirroring the `capture_account_snapshot` pattern (`loop.py:348-369`).
- Paper-only throughout; no real-money paths are added or enabled.

## Verification (during a live paper session)

1. **Phase 1, same-day:** deploy web with Alpaca env; during market hours open `/command-center` → positions/orders/equity match `curl -H "APCA-API-KEY-ID: ..." https://paper-api.alpaca.markets/v2/positions`. With a live position on (e.g. the next QQQ put), confirm mark/uPnL move on ~20s refresh and distance-to-TP hits ~0 as `unrealized_plpc` approaches +25%. Confirm the route returns 401 without the gate cookie and 404 with `COMMAND_CENTER_ENABLED` unset (test on a preview deploy).
2. **Phase 2:** dispatch `live_loop.yml`; within 2 minutes of session start, `select heartbeat_ts, loop_count, open_position_count from llm_advisor_live_state` shows the row advancing each ~60s. After session finalize, `session_stats.session_end_reason` is set and matches `session_summary.json`.
3. **Phase 3 / loop-dead alert:** cancel the workflow run mid-session while flat → overview flips to "loop offline" within 3 min. (If a position is open when testing the cancel, verify the red no-stop-enforcement banner, then manually close via Alpaca UI.)
4. **Regressions:** `pytest tests/unit`, `npm run build` in `web/`; run one full daily chain and confirm EOD aggregate + portfolio metrics endpoint (`/api/llm-advisor/metrics?source=supabase&force=true`) still work — nothing in this plan alters existing tables or ingest.
5. **Reconciliation spot-check:** blotter uPnL vs `live_state.unrealized_pnl` agree within one tick's drift; both match the 09:31→09:48-style trade lifecycle end to end.

## Out of scope

OPRA/greeks streaming or websockets (Alpaca position marks suffice; account may lack OPRA agreement). Order placement/cancel from the UI (read-only surface). A second order engine or any strategy/threshold change. Real-money auth hardening beyond the existing command-center gate. Intraday charts beyond the Phase-4 follow-up sparkline. Replacing the command-center mock watchlist/opportunities.

## Ordered task list (implementer handoff)

Each task is independently landable in order; acceptance criteria are testable without sibling repos.

1. **T1 — Alpaca live route.** Add `web/app/command-center/api/live/route.ts` per Phase 1.1. *Accept:* with valid gate cookie + paper env, returns JSON with `account`, `positions`, `openOrders`, `todaysOrders`, `fetchedAt`; returns 404 when `COMMAND_CENTER_ENABLED` unset, 401 without/with-wrong cookie, and a refusal (no Alpaca call) when `ALPACA_PAPER_TRADING !== "true"`. No Alpaca key material appears in any response or client bundle.
2. **T2 — LiveBlotter UI.** `web/components/command-center/LiveBlotter.tsx` + wiring into `command-center/page.tsx`, OCC parser in `web/lib/format.ts`, types in `web/lib/types.ts`. *Accept:* `npm run build` passes; blotter polls every 20s, pauses when tab hidden, renders positions with entry/mark/uPnL/uPnL%/distance-to-stop-TP/DTE, working orders, fills today, account strip, and a visible error state when the route 500s. Empty state renders cleanly when flat.
3. **T3 — live_state DDL.** Add `sql/005_live_state.sql` exactly as §Schema; apply to the shared Supabase project. *Accept:* file matches the schema section; `select * from llm_advisor_live_state` succeeds with anon key (empty), insert as service works.
4. **T4 — Tracker session-closed accrual.** `TradeTracker.session_closed` + `get_session_closed()` covering both close paths. *Accept:* new unit tests prove a tracked option exit (TP path) and a disappeared-position close each append `{symbol, pnl, exit_reason, closed_at}`; existing `tests/unit` suite still green.
5. **T5 — Publisher module.** `src/telemetry/live_state.py` with `build_live_state_row` + `publish_live_state` per Phase 2.2. *Accept:* unit tests for row-building on fixtures (option position with meta → correct dte/setup_type/exit_policy; empty portfolio → zeroed row; session_closed → correct wins/losses/realized); publish failure raises nothing and disables after 10 consecutive errors (test with a stubbed connection).
6. **T6 — Loop wiring + workflow env.** Publish calls in `loop.py` (in-session tick + finalize) gated to live mode and `LIVE_STATE_PUBLISH_TICKS`; `SUPABASE_DB_*` + `LIVE_STATE_PUBLISH_TICKS` in `live_loop.yml`. *Accept:* backtest mode makes zero publish calls (unit-testable via monkeypatch); a live paper session shows the row's `heartbeat_ts`/`loop_count` advancing per verification §2; a run with missing DB env logs one warning and trades normally.
7. **T7 — Dashboard live-state readers.** `getLiveState()` + overview "Live session" section + blotter loop-health banner per Phase 3. *Accept:* `npm run build` passes; with a fresh row the overview shows the live section, with a stale row during market hours it shows "loop offline"; blotter shows the red no-stop-enforcement banner iff (Alpaca positions > 0) ∧ (live_state stale > 3 min) ∧ (weekday 09:30–16:00 ET); `exit_policy` from the row overrides env-default thresholds in distance math.
8. **T8 — Docs + env examples.** Update `web/.env.example` (done partially in T1 if convenient) and README/ops notes: new Vercel vars, new workflow env, verification runbook from this plan. *Accept:* a fresh operator can configure both deploys from the docs alone; no secret values anywhere in the repo.
