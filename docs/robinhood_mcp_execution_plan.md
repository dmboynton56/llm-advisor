# Robinhood Agentic Trading (MCP) Execution Plan

Status: Phase 1 scaffold implemented — no Robinhood connection or capital
Date: 2026-07-14
Owner: Drew Boynton

## 1. Why

Robinhood now ships an official Trading MCP for AI agents:

```
https://agent.robinhood.com/mcp/trading
```

This removes any need for browser automation or unofficial Robinhood equity
libraries, both of which are explicitly disallowed by Robinhood's
[third-party connection policy](https://robinhood.com/us/en/support/articles/third-party-connections/).
The supported path is:

- **Agentic Trading MCP** for equities and options: the agent can *read*
  portfolio data across all Robinhood accounts, but can only *trade* inside a
  dedicated Agentic brokerage account created during MCP authentication.
  Setup docs: [Agentic Trading overview](https://robinhood.com/us/en/support/articles/agentic-trading-overview/),
  [MCP tool list](https://robinhood.com/us/en/support/articles/trading-with-your-agent/).
- **Robinhood Crypto API** (separate, signed-credential REST API) for crypto —
  out of scope for this plan. [Docs](https://docs.robinhood.com/).

The MCP currently supports: portfolio/buying power/realized P&L/trade history;
quotes, fundamentals, OHLCV, technical indicators, earnings; watchlists and
scanners; long-equity and options positions; **preflight simulation via
`review_equity_order`**; and placing/canceling equity and options orders.

Critical constraint driving the whole rollout: **the Agentic account is a
real, funded brokerage account — there is no paper sandbox.** Alpaca paper
trading therefore remains the strategy-validation environment; Robinhood is a
carefully gated execution venue only.

## 2. Current repo state (what this builds on)

The repo already has the right separation of concerns:

| Layer | Where it lives today | Disposition |
|---|---|---|
| Deterministic STDEV engine (z-scores, MR/TC setups) | `src/features/`, `src/live/feature_computer.py`, `src/live/threshold_evaluator.py` | Keep unchanged |
| Risk sizing & R:R validation | `src/execution/risk_calculator.py` | Keep; reuse for notional caps |
| LLM layer (Gemini threshold multipliers, trade validation) | `src/analysis/` | Keep as analysis/validation only — never the sole signal source |
| Execution adapters (pluggable) | `src/execution/order_manager.py` (Alpaca stocks), `options_order_manager.py`, `mock_order_manager.py`, selected in `src/live/loop.py` (~line 1244) | Keep Alpaca for paper validation; **add** a Robinhood MCP path alongside, not replacing |
| Trade-plan structure | `SymbolState.trade` serialized as `trade_plan` in `append_order_event` (`src/live/loop.py` ~line 329): setup, side, entry_price, sl_price, tp_price, triggered_at, execution_attempts | Formalize into an exported `trade_plan.json` contract |
| Audit trail | `order_events.jsonl`, BigQuery `trading_signals`, Supabase EOD telemetry | Extend with Robinhood execution events |
| Operator MCP pattern | `docs/alpaca_mcp_workflow.md` (read-only vs trading profiles, prompt library, safety rules) | Mirror the same structure for Robinhood |

Existing safety posture to preserve: dry-run by default, options-paper-only
default (`OPTIONS_PAPER_ONLY=true`), market-calendar checks,
`MAX_CONCURRENT_TRADES`, Discord alerting, and state recovery from BigQuery.

## 3. Target architecture

```
premarket pipeline ──► live loop (STDEV engine + Gemini overlay)
                            │
                            ├── Alpaca paper (unchanged): options-first paper
                            │   execution for strategy validation
                            │
                            └── trade_plan.json export (new, deterministic)
                                        │
                                        ▼
                        Codex + Robinhood Trading MCP
                        (execution controller prompt, human approval gate)
                                        │
                                        ▼
                        execution_result.json (new, auditable)
                                        │
                                        ▼
                        telemetry ingest → order_events.jsonl → BQ/Supabase
```

Design decisions:

1. **The strategy engine never talks to Robinhood directly in early phases.**
   It emits a signed-off, structured `trade_plan.json`. Codex (connected to
   the Robinhood MCP via Settings → MCP servers → Streamable HTTP) is the
   only component that touches Robinhood, and only under the execution
   controller prompt below.
2. **The LLM cannot invent trades.** The prompt restricts the agent to plans
   present in `trade_plan.json`; every plan carries a unique `trade_plan_id`
   that must appear in the human approval message.
3. **Human approval is a hard gate** (`APPROVE <trade_plan_id>`) until Phase 4,
   and Phase 4 only unlocks after sufficient out-of-sample results.
4. **Robinhood's own disclosures are treated as requirements**: agents can
   misunderstand instructions, use stale data, and trade without confirmation
   if told to — so the prompt mandates fresh quotes, preflight
   `review_equity_order`, duplicate rejection, and no-retry-without-approval.
   [Disclosures](https://robinhood.com/us/en/agentic-trading/).

## 4. The `trade_plan.json` contract (new)

Exported by the live loop (or a standalone exporter) whenever the STDEV engine
produces an executable signal. Schema `llm_advisor_trade_plan_v1`:

```json
{
  "schema": "llm_advisor_trade_plan_v1",
  "trade_plan_id": "2026-07-14-SPY-mr-long-001",
  "generated_at": "2026-07-14T14:32:05Z",
  "source": "stdev_live_loop",
  "symbol": "SPY",
  "setup": "mean_reversion",
  "side": "buy",
  "instrument": "equity_long",
  "order_type": "limit",
  "limit_price": 512.34,
  "stop_loss": 509.10,
  "take_profit": 517.20,
  "max_notional_pct_of_equity": 0.02,
  "signal": {
    "z_score": -2.1,
    "thresholds_used": {},
    "gemini_multiplier": 1.0,
    "premarket_bias": "neutral"
  },
  "constraints": {
    "expires_at": "2026-07-14T15:02:05Z",
    "market_session": "regular",
    "one_new_position_per_day": true
  },
  "status": "pending_approval"
}
```

Notes:

- `trade_plan_id` is deterministic and unique per day/symbol/setup so the
  approval message unambiguously references one plan.
- `expires_at` (e.g. signal time + 30 min) lets the agent reject stale plans
  mechanically instead of judging staleness itself.
- `stop_loss` / `take_profit` are advisory in Phase 2–3: Robinhood MCP places
  single limit orders, not Alpaca-style bracket orders. Exit handling is an
  explicit open question (§9).

An `execution_result.json` (schema `llm_advisor_execution_result_v1`) is the
required counterpart: trade_plan_id, Robinhood order ID, status, fill price,
timestamps, `review_equity_order` warnings shown, and the approval message
text. This is what gets ingested into `order_events.jsonl` → BigQuery →
Supabase so the dashboard reflects Robinhood activity with the same fidelity
as Alpaca.

## 5. Execution controller prompt (Phase 2+)

Stored at `prompts/robinhood_execution_controller.md` (new), versioned in git:

```
You are the execution controller for LLM Advisor.

Only consider trades supplied in a structured trade_plan generated by the
deterministic strategy engine. Never invent a trade.

Allowed:
- Symbols: SPY, QQQ, IWM
- Long equities only
- Limit orders only
- One new position per day
- Maximum order notional: 2% of Agentic account equity
- Maximum total exposure: 10%
- No margin, options, shorts, crypto, or averaging down

Before proposing an order:
1. Read the Agentic portfolio, positions, buying power, and open orders.
2. Obtain a current quote and confirm the symbol is tradable.
3. Reject duplicate or conflicting orders.
4. Reject the trade if its quote is stale, the market is closed, available
   buying power is insufficient, or the daily-loss circuit breaker has fired.
5. Call review_equity_order and show its warnings.

Do not call place_equity_order until I respond:
APPROVE <trade_plan_id>

After placement, retrieve the order, report its Robinhood order ID and status,
and write an auditable execution result. Never retry a rejected order without
new approval.
```

The daily-loss circuit breaker referenced in step 4 is computed by the repo
(realized P&L from Robinhood trade history vs. a configured limit, default
−1% of Agentic equity per day) and written into `trade_plan.json` as a
boolean the agent must check — again, mechanical rather than judged.

## 6. Phased rollout

### Phase 0 — Read-only MCP (no code changes)
- Create the Agentic account; connect Codex to the Robinhood MCP.
- Use read tools only: portfolio, quotes, OHLCV, indicators, trade history.
- Write `docs/robinhood_mcp_workflow.md` mirroring `alpaca_mcp_workflow.md`
  (profiles, premarket/intraday/EOD prompt library, safety rules).
- Exit criteria: read tools verified against known account state; tool names
  and shapes documented from live inspection, not from marketing pages.

### Phase 1 — Trade-plan export + audit plumbing (code, no capital)
- `src/execution/trade_plan_exporter.py`: build and validate
  `llm_advisor_trade_plan_v1` from `SignalEvent` + `SymbolState.trade`;
  write to `data/daily_news/<date>/processed/trade_plans/`.
- `scripts/export_trade_plan.py`: standalone CLI for backfills and manual runs.
- `scripts/ingest_execution_result.py`: validate an `execution_result.json`,
  append to `order_events.jsonl`, push to BQ/Supabase, alert Discord.
- Unit tests: schema round-trip, ID uniqueness, expiry logic, circuit-breaker
  flag computation.
- Exit criteria: a full dry-run day produces trade plans and a hand-written
  fake execution result flows end-to-end into the dashboard.

### Phase 2 — Reviewed orders, human approval, tiny balance
- Fund the Agentic account minimally (e.g. $200–500 — enough that a 2%
  notional cap yields a marketable 1-share-ish order on SPY/QQQ/IWM; note
  fractional support is another §9 verification item).
- Operate the controller prompt manually in Codex: paste/point it at the
  day's `trade_plan.json`, review `review_equity_order` warnings, approve or
  reject explicitly.
- Every session ends with `execution_result.json` ingested via Phase 1
  tooling. Alpaca paper keeps running in parallel as the control group.
- Exit criteria: ≥20 approved-or-rejected plans with zero contract violations
  (no invented trades, no unapproved placements, no stale executions).

### Phase 3 — Native adapter (optional, decision point)
- If the manual Codex loop proves out and tighter integration is wanted:
  `src/execution/robinhood_mcp_order_manager.py`, a Python Streamable-HTTP
  MCP client implementing the same duck-typed interface the live loop already
  selects against (`get_account_equity`, `get_buying_power`,
  `get_open_positions`, `close_position`, `cancel_order`,
  `execute_stock_trade`) so it slots into the existing adapter switch in
  `src/live/loop.py`.
- Env: `TRADING_INSTRUMENT=stocks_robinhood`, plus explicit
  `ROBINHOOD_MCP_ENABLED=true` and `ROBINHOOD_MAX_NOTIONAL_PCT`,
  `ROBINHOOD_MAX_TOTAL_EXPOSURE_PCT`, `ROBINHOOD_DAILY_LOSS_LIMIT_PCT`.
  Refuse to construct the adapter unless all are set — same pattern as
  `OPTIONS_PAPER_ONLY`.
- Approval stays out-of-band (Discord approve/deny or CLI prompt); the
  adapter enforces it, not the LLM.

### Phase 4 — Limited automation (earliest possible, not scheduled)
- Only after sufficient out-of-sample results from Phases 2–3 (target: ≥3
  months, positive expectancy net of costs, drawdown within backtest
  envelope) does any auto-approval get considered, and then only within the
  Phase 2 allowlist (SPY/QQQ/IWM, long equity, limit, 2%/10% caps, one
  position/day) with the daily-loss breaker unchanged and Discord
  notification on every action.

## 7. Guardrails summary (enforced mechanically wherever possible)

| Guardrail | Enforced by |
|---|---|
| Symbols SPY/QQQ/IWM only | exporter (won't emit others) + prompt |
| Long equity, limit only | exporter + prompt + (Phase 3) adapter |
| 2% notional / 10% exposure caps | exporter via `risk_calculator` + prompt + `review_equity_order` |
| One new position per day | exporter (plan-ID scheme) + prompt |
| Plan expiry (stale-quote defense) | `expires_at` in plan |
| Daily-loss circuit breaker | computed by repo, flag in plan |
| Human approval | `APPROVE <trade_plan_id>` gate; Phase 3 out-of-band approval |
| No retry after rejection | prompt + `execution_attempts` counter carried into plan |
| Audit trail | mandatory `execution_result.json` → order_events → BQ/Supabase |
| Trading only in Agentic account | enforced by Robinhood platform itself |

## 8. What explicitly stays the same

- Alpaca paper trading (options-first) remains the default live-loop path and
  the only place strategy changes are validated. Backtests, notebooks, and
  the Supabase/portfolio telemetry contract are untouched except for the new
  Robinhood event types.
- No browser automation, no unofficial Robinhood libraries, ever — the
  supported surfaces are the Agentic MCP and (if crypto ever enters scope)
  the signed Crypto API.
- X-bookmark trading content (QQQ/IWM/SPY levels, mean-reversion setups,
  TradingView indicators) is treated as hypothesis input to the premarket
  research layer only — never as deployable signals; most of it is marketing
  or survivorship-flavored retrospectives.

## 9. Open questions / verification items (resolve in Phase 0–1)

1. **Exit management without brackets.** Alpaca bracket orders bundle
   TP/SL; the Robinhood MCP appears to place single orders. Options:
   agent-monitored exits (weak), repo-side EOD/intraday close instructions
   (reuse `eod_position_manager.py` patterns), or standing sell-limit placed
   immediately after fill. Decide before Phase 2 funds anything.
2. **Fractional shares / minimum notional** on the Agentic account — affects
   the tiny-balance sizing math.
3. **Exact MCP tool names and response shapes** — verify against the live
   server in Phase 0; do not code Phase 3 against support-article prose.
4. **Rate limits, session/auth lifetime, and market-hours behavior** of the
   MCP endpoint.
5. **Codex file access** — how the controller prompt reads
   `trade_plan.json` (workspace file vs. pasted payload) and where it writes
   `execution_result.json`.
6. **Tax/accounting**: real fills in the Agentic account create taxable
   events even at tiny size; confirm that's acceptable before Phase 2.
