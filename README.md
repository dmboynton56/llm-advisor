# llm-advisor

# Liquidity Flow Agent

**An AI-powered day trading bot that combines traditional machine learning with Large Language Models (LLMs) to execute a sophisticated ICT-based trading strategy.**

This project leverages a hybrid approach:
1.  **Daily Bias Prediction:** Pre-trained ML models (Random Forest/XGBoost) analyze historical data to determine the most probable market direction for the day (Bullish/Bearish/Choppy) for a watchlist of symbols.
2.  **LLM-Powered Real-Time Analysis:** An AI agent, powered by models like GPT-4o, continuously analyzes real-time market data, looking for specific ICT entry patterns that align with the pre-determined daily bias.
3.  **Automated Execution:** When the AI agent identifies a high-confidence trade setup, it autonomously calculates risk parameters and executes the trade via the Alpaca API.

---

## 🚀 Quick Start

1.  **Clone the repository:**
    ```bash
    git clone [repository-url]
    cd liquidity_flow_agent
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure your settings:**
    * Create a `.env` file and add your `ALPACA_` and `OPENAI_` API keys.
    * Edit `config/settings.py` to define your symbol watchlist and risk parameters.

4.  **Train Daily Bias Models (if not already present):**
    ```bash
    python training/train_daily_bias_models.py
    ```

5.  **Run the bot:**

    **STDEV Trading System (New):**
    ```bash
    # Run premarket pipeline
    python scripts/run_premarket.py --date YYYY-MM-DD --symbols SPY QQQ
    
    # Run live trading loop
    python scripts/run_live_loop.py --date YYYY-MM-DD --symbols SPY QQQ
    
    # Or backtest on historical data
    python scripts/run_backtest.py --date YYYY-MM-DD --symbols SPY QQQ
    ```
    
    **Legacy ICT Strategy:**
    ```bash
    python main.py  # Old orchestrator (deprecated)
    ```

---

## 📚 Project Documentation

For a complete understanding of the system, please refer to the documentation:

* **[Project Plan](./docs/PROJECT_PLAN.md):** The overall architecture, data flow, and development roadmap.
* **[Trading Strategy](./docs/STRATEGY.md):** A detailed explanation of the 5-step ICT methodology used by the AI agent.
* **[Execution & Risk](./docs/EXECUTION.md):** The rules governing trade execution, position sizing, and risk management.