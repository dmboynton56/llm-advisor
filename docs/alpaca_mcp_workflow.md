# Alpaca MCP Workflow

Use Alpaca MCP as an operator and research layer around the deterministic
options engine in this repository. The trading loop should own strategy
mapping, risk checks, order payloads, and telemetry.

## Profiles

- `alpaca-readonly`: account, assets, stock data, options data, news, and
  corporate actions. Use this for normal research.
- `alpaca-paper-trading`: readonly tools plus trading. Use only with paper
  keys and `ALPACA_PAPER_TRADE=true`.

Copy `.vscode/mcp.example.json` to `.vscode/mcp.json` if your MCP client uses
VS Code project config. Do not commit real credentials.

## Premarket Prompts

- What is my Alpaca paper account equity, options buying power, and open
  positions?
- Is the market open today, and what are today's regular session open and close
  times?
- Show SPY, QQQ, and IWM latest stock snapshots and recent news.
- Show option chains for SPY, QQQ, and IWM with 7-14 DTE and strikes within
  10 percent of spot.
- Which candidate contracts have tight bid/ask spreads, reasonable open
  interest, and deltas between 0.35 and 0.55?

## Intraday Prompts

- Show all open Alpaca paper orders and current option positions.
- Get the latest quote and Greeks for the option contracts currently held.
- Compare current bid/ask against the submitted limit prices from today's order
  events.

## EOD Prompts

- Show filled and canceled orders from today.
- Show account activities from today for fills and option events.
- Show current open option positions and flag contracts expiring within three
  trading days.
- Compare Alpaca paper positions against `data/daily_news/<date>/processed/order_events.jsonl`.

## Safety Rules

- MCP trading tools are for operator-confirmed paper actions only.
- The live loop defaults to `TRADING_INSTRUMENT=options` and
  `OPTIONS_PAPER_ONLY=true`.
- The repository uses `ALPACA_PAPER_TRADING`; Alpaca MCP uses
  `ALPACA_PAPER_TRADE`.
- Never configure a remote MCP server without HTTPS/TLS and authentication.
