# Project Plan: Liquidity Flow Agent

This document outlines the architecture, data flow, and development plan for the AI-powered trading bot.

## 1. System Architecture

The system is designed as a modular, event-driven application that operates in distinct phases throughout the trading day.

**Key Components:**
* **Data Processing:** Gathers, cleans, and formats all necessary data (market prices, news).
* **Strategy & Analysis:** Contains the core logic for both the ML-based bias prediction and the LLM-based real-time analysis.
* **API Clients:** Manages all external API communications (Alpaca for trading, LLM providers for analysis).
* **Execution:** Handles the placement and management of trades.

## 2. Data & Logic Flow

The bot's operational flow is a top-down process, moving from a high-level daily view to a micro-level 1-minute analysis.

**Phase 1: Pre-Market (approx. 9:00 AM EST)**
1.  **News Scraping:** The `news_scraper.py` module gathers key financial headlines and articles for the monitored symbols.
2.  **Daily Bias Calculation:** The `bias_calculator.py` module loads the pre-trained `.pkl` models. It uses the latest market data to calculate the daily bias (Bullish/Bearish/Choppy) and a confidence score for each symbol.
3.  **Context Compilation:** The daily biases and summarized news are compiled as the foundational context for the day.

**Phase 2: Market Open & Monitoring (9:30 AM EST onwards)**
1.  **Real-Time Data Feed:** The `price_aggregator.py` begins polling the Alpaca API every 30-60 seconds for the latest price data across the watchlist.
2.  **Feature Calculation:** The `feature_engine.py` calculates key technical metrics from the real-time data.
3.  **LLM Analysis Loop:**
    * The `trade_analyzer.py` formats the daily context (bias, news) and the latest real-time data into a structured prompt.
    * This prompt is sent to one or more LLM APIs via the `llm_client.py`.
    * The bot asks the LLM to analyze the data according to the ICT strategy and return a verdict, including a confidence score.
4.  **Decision Gate:** The main loop in `main.py` aggregates the confidence scores from the LLMs.

**Phase 3: Trade Execution**
1.  **Trigger:** If the combined confidence score for a specific symbol's setup crosses a pre-defined threshold (e.g., 85%), a trade signal is generated.
2.  **Parameter Calculation:** The LLM is asked for precise stop loss and take profit levels.
3.  **Order Placement:** The `order_manager.py` receives the final trade parameters and places the bracket order via the `alpaca_client.py`.

## 3. Development Plan

The project will be built in phases to ensure each component is robust before integration.

* **Phase 1: Foundation & Data:** Build the Alpaca client, data aggregators, and feature engine. Ensure a reliable stream of real-time data.
* **Phase 2: Daily Bias Integration:** Integrate the pre-trained models from `ICTML` to ensure the bot can generate daily biases at the start of each session.
* **Phase 3: LLM Agent Core:** Develop the `llm_client` and `trade_analyzer`. Focus on prompt engineering and parsing structured JSON responses.
* **Phase 4: Execution & Paper Trading:** Build the `order_manager` and deploy the full system in Alpaca's paper trading environment for at least 1-2 months.
* **Phase 5: Live Deployment:** After consistent profitability in paper trading, deploy with a small amount of real capital and scale gradually.