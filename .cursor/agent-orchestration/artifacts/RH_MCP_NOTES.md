# Robinhood Trading MCP Notes

Verified against Robinhood's official support pages on 2026-07-15. No Robinhood connection or order was made.

## Supported surface

The official Streamable HTTP endpoint is `https://agent.robinhood.com/mcp/trading`. Authentication creates/connects a dedicated Agentic investing account. The agent receives read access across Robinhood accounts (account details, balances, positions, transactions/order history, watchlists, and scans), but Robinhood restricts placement to the Agentic account. See [Agentic Trading overview](https://robinhood.com/us/en/support/articles/agentic-trading-overview/).

The current official [Trading with your agent](https://robinhood.com/us/en/support/articles/trading-with-your-agent/) tool inventory includes:

- Account/portfolio: `get_accounts`, `get_portfolio`, `get_realized_pnl`, `get_pnl_trade_history`, and symbol `search`.
- Market research: equity OHLCV, fundamentals, technical indicators, earnings results/calendar, index data and quotes.
- Watchlists/scanners: read and mutate equity/option watchlists; create, configure, and run scans.
- Equities: positions, quotes, order history, tradability/fractional checks, `review_equity_order` preflight warnings, place, and cancel.
- Options: chains, instruments, quotes, positions, order history, `review_option_order` preflight warnings, place, and cancel.

Robinhood currently documents long-equity and options placement. Tool names and response shapes may change, so a live read-only discovery pass remains mandatory before building a native adapter.

## Constraint and architecture decision

Robinhood's official materials describe a dedicated self-directed brokerage account and real order-placement tools; they do not document a paper/sandbox Agentic account. Treat every placement call as real-money execution. Alpaca paper therefore remains the validation venue, while Robinhood stays disabled and human-gated.

Phase-1 plumbing in this repository is intentionally disconnected from MCP:

1. `trade_plan_exporter.py` emits an allowlisted, long-equity, limit-only `llm_advisor_trade_plan_v1` with deterministic ID, 30-minute expiry, 2% notional/10% exposure caps, required human approval, and a fail-closed daily-loss decision.
2. A future command-center/controller reads the plan, refreshes portfolio/quote/tradability, calls `review_equity_order`, and displays all warnings.
3. Only an exact `APPROVE <trade_plan_id>` may precede placement.
4. `execution_result_ingestor.py` validates the result and appends idempotent Robinhood lifecycle telemetry to `order_events.jsonl` for the existing BigQuery/Supabase pipeline.

The command center added this cycle uses mock data only. Future extension points are a private Supabase `watchlist_flags` table and read-only Robinhood watchlist/scanner/quote tools. Real order controls should not be added until identity-backed auth, exact live tool-shape verification, exit management, and a funded-account approval policy are resolved.

## Safety follow-ups

- Replace the shared-password scaffold with identity-backed authorization before any connection.
- Confirm fractional-share behavior, MCP auth/session lifetime, rate limits, and market-hours behavior live.
- Design monitored exits; Robinhood does not support bracket orders according to its [order-types documentation](https://robinhood.com/us/en/support/articles/360001213963/).
- Keep fresh quote, duplicate/open-order check, buying-power check, daily-loss breaker, preflight review, and no automatic retry as mechanical gates.
