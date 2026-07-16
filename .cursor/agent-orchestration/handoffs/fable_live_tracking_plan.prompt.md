# Fable planning task — live session tracking

You are **Fable**, planning lead.
Working directory: `/home/dmboynton/projects/llm-advisor`

## Job
Explore the repo with your tools (read-only recon is fine; you may write the plan file). **Do not change product code.** Write one self-contained plan file that an implementer can execute without sibling repos.

## Write this file (required)
`.cursor/plans/2026-07-live-session-tracking.md`

Follow the style/structure of the existing plan at `.cursor/plans/2026-07-ops-dashboard.md` (goals, current-state recon with file refs, phased design, schema if needed, verification, out of scope). Inline any cross-repo Supabase contract details the plan needs.

## Goal (from operator)
Biggest gap in the daily live loop / trading system right now: **track current prices and live orders during the session** — open PnL, live trade statistics, working orders, etc. Today the ops dashboard is mostly post-EOD; during market hours you have to script Alpaca manually.

Concrete example from 2026-07-16:
- Entered `QQQ260724P00711000` (QQQ Jul 24 711 Put) long 2 @ $8.75 at 09:31 ET
- Hit ~+25% open PnL / software TP, exited 09:48 ET @ $11.58
- No in-session UI surface for open mark, distance to stop/TP, or open orders

## Recommendations already discussed (evaluate; pick what makes most sense)
You are **not** required to rubber-stamp these. Challenge them against the codebase and choose the architecture you think is best. Document tradeoffs.

1. **Source of truth for marks:** Alpaca paper `/positions` already exposes `avg_entry_price`, `current_price`, `unrealized_pl`, `unrealized_plpc`. Live loop already uses this via `TradeTracker.update_positions` / `OptionsOrderManager.get_open_positions`. Option stops/TP are **software** (`OPTION_STOP_LOSS_PCT=0.35`, `OPTION_PROFIT_TARGET_PCT=0.25` in `src/core/config.py` / `trade_tracker._option_exit_reason`) — no broker bracket legs for options.

2. **Two architectures on the table:**
   - **A. Dashboard polls Alpaca** via Next.js server route (paper keys server-side only). Fastest for “is my put alive?” Works even if GH Actions loop dies. Dual source of truth vs telemetry.
   - **B. Loop publishes live state to Supabase** each N ticks (positions + heartbeat equity/uPnL; optionally mid-day order_events flush). Dashboard stays on existing Supabase contract. Better long-term for joined stop/setup/funnel context.
   - Prior informal lean: **A as bootstrap, B as durable path** — accept or reject with rationale.

3. **v1 scope suggestion (challenge freely):**
   - Live blotter: open positions (symbol, qty, entry, mark, uPnL, uPnL%), open/working orders, account equity + daily PnL
   - Distance to option stop/TP from entry × config (and/or stamped metadata)
   - Session stats: fills today, realized so far, win/loss count if cheap
   - Poll every 15–60s is fine for paper; no OPRA websocket required (account may lack OPRA agreement — marks from Alpaca positions are enough)
   - UI: extend ops `web/` (overview and/or command-center `/live`) — don’t invent a parallel app

4. **Avoid for v1:** full OPRA greek stream, second order engine, fancy charts, real-money auth hardening beyond existing command-center gate patterns.

## Current-state anchors to verify
- Live loop: `src/live/loop.py` (position updates, heartbeats, `capture_account_snapshot`, order_events append)
- Tracker/exits: `src/execution/trade_tracker.py`
- Account snapshot: `src/api_clients/account_snapshot.py` (today mostly start/end → EOD ingest)
- EOD → Supabase: `scripts/run_eod_aggregate.py`, tables `llm_advisor_*` (see `sql/001_*`–`004_*`, CONTRACTS facts in `.cursor/plans/2026-07-ops-dashboard.md`)
- Dashboard: `web/` (overview is EOD-oriented; command-center still mocky in places)
- Workflows: `.github/workflows/live_loop.yml`, `eod_aggregate.yml`

## Plan requirements
In the plan file, include:
1. **Verdict** — chosen architecture (A / B / hybrid) and why
2. **Current-state recon** with file:line or path evidence for the gap
3. **Phased implementation** (schema/API/UI/loop changes) with concrete file targets
4. **Supabase DDL** if any new tables/columns (additive; note ownership stays in `llm-advisor/sql/`)
5. **Secrets / deploy notes** (GH Actions vs Vercel; never commit keys; paper-only)
6. **Verification** — how to prove it during a live paper session
7. **Out of scope** + follow-ups
8. **Ordered task list** suitable for a later Codex/implementer handoff (testable acceptance criteria)

## Done when
- Plan file exists at the path above and is self-contained
- End your response with a short summary of the verdict + path to the plan
