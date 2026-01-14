# LLM-Advisor

An AI-powered day trading system that combines technical indicators with Large Language Models (LLMs) to execute a standard deviation based trading strategy.

## System Overview

This project implements a hybrid trading approach focused on Standard Deviation (STDEV) tactics:

1. Premarket Analysis: Gathers daily bias and news context using ML models and news scrapers to establish market sentiment.
2. Technical Execution: Uses a live loop to compute real-time technical features, including z-scores, mu, and sigma, to identify Mean Reversion (MR) and Trend Continuation (TC) opportunities.
3. LLM-Powered Intelligence: Integrates Google Gemini 3 Flash to perform periodic market analysis every 15 minutes, adjusting technical thresholds via multipliers based on qualitative market context.
4. Automated Risk Management: Executes bracket orders via Alpaca API with autonomously calculated risk parameters, stop losses, and take profit levels.

## Infrastructure

The system is built for reliable, automated daily execution:

- Database: Google BigQuery for persistent storage of trades, logs, and market analysis.
- Orchestration: GitHub Actions for daily automated runs (Premarket and Live Loop).
- AI Engine: Google Gemini 3 Flash (via Google Generative AI API).
- Broker: Alpaca Markets (supports paper and live trading).
- Notifications: Discord Webhook integration for real-time trade alerts and heartbeat monitoring.

## Getting Started

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone [repository-url]
cd llm-advisor
pip install -r requirements.txt
```

### 2. Configuration

Set up your environment variables in a .env file:

```text
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_PAPER_TRADING=true
GOOGLE_API_KEY=your_gemini_key
GCP_PROJECT_ID=your_gcp_project
GCP_SA_KEY=your_service_account_json_content
DISCORD_WEBHOOK_URL=your_webhook_url
STORAGE_ENV=bq
```

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
python scripts/run_backtest.py --date YYYY-MM-DD --symbols SPY --test
```

### Execution Modes

- **Dry Run**: Default behavior when credentials are missing or not explicitly set to trade. Signals are logged but no orders are placed.
- **Paper Trading**: Set `ALPACA_PAPER_TRADING=true` in your environment.
- **Live Trading**: Set `ALPACA_PAPER_TRADING=false`. Use with extreme caution.

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

## Safety and Production Shields

- Market Calendar Check: Scripts automatically verify if NYSE is open before running tasks.
- State Persistence: The system syncs with BigQuery on startup to recover existing positions and prevent duplicate trades.
- Error Alerting: Critical failures and trade executions are sent immediately to Discord.
- Timeout Protection: GitHub Actions are configured with strict timeouts to ensure process termination after market close.
