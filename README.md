# LLM-Advisor

An AI-powered day trading system that combines technical indicators with Large Language Models (LLMs) to execute a standard deviation based trading strategy.

## System Overview

This project implements a hybrid trading approach focused on Standard Deviation (STDEV) tactics:

1. Premarket Analysis: Gathers daily bias and news context using ML models and news scrapers to establish market sentiment.
2. Technical Execution: Uses a live loop to compute real-time technical features, including z-scores, mu, and sigma, to identify Mean Reversion (MR) and Trend Continuation (TC) opportunities.
3. LLM-Powered Intelligence: Integrates Google Gemini 3 Flash to perform periodic market analysis every 15 minutes, adjusting technical thresholds via multipliers based on qualitative market context.
4. Automated Risk Management: Converts stock-derived STDEV signals into
   paper-only options trades by default, with stock bracket orders available
   only when explicitly enabled.

## Infrastructure

The system is built for reliable, automated daily execution:

- Database: Google BigQuery for persistent storage of trades, logs, and market analysis.
- Serving telemetry: Supabase tables populated after EOD for portfolio metrics.
- Orchestration: GitHub Actions for daily automated runs (Premarket and Live Loop).
- Scheduler: Google Cloud Scheduler dispatches the Premarket and Live Loop
  workflows via `workflow_dispatch`.
- AI Engine: Google Gemini 3 Flash (via Google Generative AI API).
- Broker: Alpaca Markets (options-first paper trading by default; stock
  execution remains opt-in).
- Notifications: Discord Webhook integration for real-time trade alerts and heartbeat monitoring.

## Current public state

LLM Advisor is still an in-progress flagship project. The live architecture is
real, but the public portfolio intentionally presents it as a consolidation
story: premarket ML bias, Gemini threshold overlays, execution guardrails, and
EOD telemetry are being brought into one recruiter-facing dashboard.

Current serving contract:

- BigQuery dataset `trading_signals` stores cold-path live-loop rows such as
  trades, trade signals, and loop logs when `STORAGE_ENV=bq`.
- EOD aggregation upserts into Supabase tables:
  `llm_advisor_backtest_runs`, `llm_advisor_backtest_trades`,
  `llm_advisor_runtime_heartbeats`, and `llm_advisor_order_events`.
- The portfolio reads those Supabase tables through
  `/api/llm-advisor/metrics` and uses docs under
  `personal-portfolio/docs/project-knowledge/llm-advisor/` for scoped chat.
- Last terminal verification: 2026-05-23. Supabase contained 1 recent
  `llm_advisor_backtest_runs` row, 0 recent `llm_advisor_backtest_trades`
  rows, 3 recent `llm_advisor_runtime_heartbeats` rows, and the
  `llm_advisor_order_events` table was live with RLS and API grants.

## Getting Started

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone [repository-url]
cd llm-advisor
pip install -r requirements.txt
```

### Daily bias models (premarket)

The premarket pipeline loads per-symbol classifiers from [`models/`](models/README.md). For CI defaults (`SPY`, `QQQ`, `IWM`), commit the `.pkl` files or generate dev placeholders:

```bash
python scripts/export_minimal_bias_models.py SPY QQQ IWM
```

Replace placeholders with production-trained models before relying on ML bias for execution decisions.

### 2. Configuration

Set up your environment variables in a .env file:

```text
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_PAPER_TRADING=true
TRADING_INSTRUMENT=options
OPTIONS_PAPER_ONLY=true
ALLOW_STOCK_FALLBACK=false
OPTIONS_STRATEGY_TYPE=single_long
OPTION_DTE_MIN=7
OPTION_DTE_MAX=14
OPTION_DELTA_MIN=0.35
OPTION_DELTA_MAX=0.55
MAX_OPTION_PREMIUM_PER_TRADE=200
MAX_OPTION_BID_ASK_SPREAD_PCT=0.15
MIN_OPTION_OPEN_INTEREST=100
OPTION_STRIKE_WINDOW_PCT=0.10
OPTION_PROFIT_TARGET_PCT=0.25
OPTION_STOP_LOSS_PCT=0.35
OPTION_MAX_HOLD_MINUTES=30
OPTION_CLOSE_AT_ENTRY_WINDOW_END=true
GOOGLE_API_KEY=your_gemini_key
GCP_PROJECT_ID=your_gcp_project
GCP_DATASET_ID=trading_signals
GCP_SA_KEY=your_service_account_json_content
DISCORD_WEBHOOK_URL=your_webhook_url
STORAGE_ENV=bq
```

**GitHub Actions (EOD → Supabase)** also require repo secrets: `SUPABASE_DB_HOST`, `SUPABASE_DB_PORT`, `SUPABASE_DB_NAME`, `SUPABASE_DB_USER`, `SUPABASE_DB_PASSWORD`. Optional: `PORTFOLIO_METRICS_URL`, `EOD_STRICT_TELEMETRY=1`.

Alpaca keys are read with leading/trailing whitespace stripped so CRLF line endings from Windows-saved `.env` files do not break HTTP headers.

### 3. Local Usage

Run the premarket pipeline to generate daily context:
```bash
python scripts/run_premarket.py --symbols SPY QQQ IWM --use-db
```

Run the live trading loop:
```bash
python scripts/run_live_loop.py --symbols SPY QQQ IWM --use-db --fast 60
```

Run a backtest on historical data:
```bash
python scripts/run_backtest.py --date YYYY-MM-DD --symbols SPY
```

Batch weekdays (writes `data/daily_news/<date>/processed/backtest_results.json` per day):
```bash
python scripts/run_backtest_batch.py --start YYYY-MM-DD --end YYYY-MM-DD --symbols SPY QQQ IWM
```

Roll up completed backtests into one JSON for narratives / portfolio metrics:
```bash
python scripts/aggregate_backtest_results.py --output notebooks/_exported/backtest_roll_summary.json
```

To refresh the portfolio artifact (`personal-portfolio/public/data/llm_advisor_backtest_snapshot.json`), re-run your dates, run `aggregate_backtest_results.py`, then merge the rollup object plus `experiment` / `linkedin_snippets` fields following schema `llm_advisor_portfolio_snapshot_v1`.

### Execution Modes

- **Dry Run**: Default behavior when credentials are missing or not explicitly set to trade. Signals are logged but no orders are placed.
- **Options Paper Trading**: Default live-loop execution path. Set
  `TRADING_INSTRUMENT=options`, `OPTIONS_PAPER_ONLY=true`, and
  `ALPACA_PAPER_TRADING=true`. STDEV stock signals are mapped to 7-14 DTE long
  calls or long puts after delta, bid/ask spread, open interest, and premium
  checks. The default profile simulates a small account with one position at a
  time, a $200 max premium, 25% profit target, 35% stop, 30 minute time stop,
  and a forced option close when the entry window ends.
- **Stock Paper Trading**: Set `TRADING_INSTRUMENT=stocks` and
  `ALLOW_STOCK_FALLBACK=true` to use the legacy stock bracket order manager.
- **Live Trading**: The options engine refuses live mode while
  `OPTIONS_PAPER_ONLY=true`. Use with extreme caution and only after adding
  explicit live-options approval controls.

Successful Alpaca submits are persisted when `--use-db` is set:
`trade_signals`, optional `llm_validations`, `trades`, `positions`, and
`live_loop_log.jsonl` / `session_summary.json` in the telemetry artifact for
EOD. Option-specific order metadata is written to `order_events.jsonl`.

For Alpaca MCP setup and daily operator prompts, see
[`docs/alpaca_mcp_workflow.md`](docs/alpaca_mcp_workflow.md). For the
options-first paper runbook, see
[`docs/options_paper_runbook.md`](docs/options_paper_runbook.md).

For next-day operational checks and artifact expectations, see
[`LIVE_LOOP_RUNBOOK.md`](LIVE_LOOP_RUNBOOK.md).

### Optional

- `MAX_CONCURRENT_TRADES` (default `1`) caps simultaneous open positions before new entries are sent.

## Testing

The system includes unit tests using pytest:

```bash
pytest tests/unit/
```

Individual modules can be tested separately:
```bash
pytest tests/unit/test_risk_calculator.py
```

## Repository Structure

- src/premarket/: Daily bias and snapshot generation logic.
- src/live/: Core trading loop and feature computation.
- src/analysis/: LLM client and market analysis modules.
- src/data/: Database adapters and broker clients.
- src/execution/: Order management and position tracking.
- .github/workflows/: Automated CI/CD pipelines for daily trading.
- scripts/: Utility scripts for testing and manual execution.

## Analysis Notebooks

Week-2 analysis notebooks live in `notebooks/` and are used to back portfolio metrics and narrative claims.

- `notebooks/trade_journal.ipynb`
- `notebooks/pnl_attribution.ipynb`
- `notebooks/threshold_sensitivity.ipynb`
- `notebooks/prompt_ablation.ipynb`
- `notebooks/premarket_bias_evaluation.ipynb`
- `notebooks/backtest_roll_summary.ipynb` — multi-day backtest rollup via `aggregate_backtest_results.py`

Use the template at `notebooks/_templates/llm_advisor_eval.ipynb` for additional analyses.

## Safety and Production Shields

- Market Calendar Check: Scripts automatically verify if NYSE is open before running tasks.
- State Persistence: The system syncs with BigQuery on startup to recover existing positions and prevent duplicate trades.
- Error Alerting: Critical failures and trade executions are sent immediately to Discord.
- Timeout Protection: GitHub Actions are configured with strict timeouts to ensure process termination after market close.

## Portfolio chat

The portfolio page `/projects/llm-advisor` includes scoped chat (`scope=llm-advisor`) that answers from `personal-portfolio/docs/project-knowledge/llm-advisor/` and Supabase telemetry via `get_model_metrics` when EOD/sync has populated tables.

Keep portfolio-facing claims aligned with:

- `personal-portfolio/docs/project-knowledge/llm-advisor/`
- `personal-portfolio/supabase/migrations/005_llm_advisor_telemetry.sql`
- `.github/workflows/premarket.yml`
- `.github/workflows/live_loop.yml`
- `.github/workflows/eod_aggregate.yml`

## GitHub Actions and portfolio verification

- **Premarket** (`premarket.yml`): triggered via `workflow_dispatch` (GCP Cloud Scheduler at `09:20 America/New_York`, Mon–Fri). Skips NYSE holidays/weekends via `check_market_open.py`. Infra lives in `infra/gcp/` (`./infra/gcp/deploy.sh`).
- **Live Trading Loop** (`live_loop.yml`): triggered via `workflow_dispatch` (GCP Cloud Scheduler at `09:27 America/New_York`, Mon–Fri). Waits up to 12 minutes for a **same-ET-day** `premarket-context` artifact, fails if today’s premarket artifact is missing, and fails telemetry upload if the loop produced no processed files.
- **EOD Aggregate** (`eod_aggregate.yml`) auto-runs via `workflow_run` after a successful **Live Trading Loop** and downloads the triggering run’s exact `llm-advisor-daily-news-*` artifact at the **repo root** so paths stay `data/daily_news/<YYYY-MM-DD>/processed/…`. Triggering Live runs with no telemetry artifact noop EOD; manual `workflow_dispatch` still supports **`live_loop_run_id`** (Actions → Live run URL → numeric id) and fails loudly if the pinned artifact is missing.
- **Portfolio checks:** optional secret **`PORTFOLIO_METRICS_URL`** = deployed `https://…/api/llm-advisor/metrics` base (no path suffix). When set, EOD curls metrics with `?source=supabase&force=true`. Unset the secret temporarily if the endpoint is still empty while debugging ingest.

Optional repo secret: **`EOD_STRICT_TELEMETRY=1`** — fail EOD if Supabase has no `llm_advisor_*` rows in the 7-day rollout check (default is warn-only until telemetry is consistently populated).
