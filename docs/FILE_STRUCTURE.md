liquidity_flow_agent/
├── client/
│   ├── #web client interface?
├── config/
│   ├── __init__.py
│   └── settings.py                 # Main configuration file (API keys, symbols, thresholds)
│
├── data/
│   └── daily_news/                 # Stores scraped news articles for the day
│   │   └── YYYY-MM-DD
│
├── docs/
│   ├── PROJECT_PLAN.md             # Detailed architecture and phased plan
│   ├── STRATEGY.md                 # In-depth explanation of the ICT methodology
│   └── EXECUTION.md                # Risk management and execution rules
│
├── models/                         # Stores trained .pkl models for daily bias
│   ├── NQ1_daily_bias.pkl
│   ├── SPY_daily_bias.pkl
│   └── ...
│
├── prompts/
│   └── prompts.json                # Central repository for all LLM prompts
│
├── src/
│   ├── __init__.py
│   ├── api_clients/
│   │   ├── __init__.py
│   │   ├── alpaca_client.py        # Handles all Alpaca API interactions
│   │   └── llm_client.py           # Manages interactions with one or more LLM APIs
│   │
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── feature_engine.py       # Calculates technical features (from ICTML)
│   │   ├── news_scraper.py         # Scrapes and processes financial news
│   │   └── price_aggregator.py     # Fetches and aggregates market data
│   │
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── bias_calculator.py      # Loads .pkl models to calculate daily bias
│   │   └── trade_analyzer.py       # Main logic: formats prompts, queries LLMs, evaluates responses
│   │
│   ├── execution/
│   │   ├── __init__.py
│   │   └── order_manager.py        # Places, monitors, and manages trades via Alpaca
│   │
│   └── utils/
│       ├── __init__.py
│       └── logger.py               # Centralized logging setup
│
├── training/
│   ├── train_daily_bias_models.py  # Script to train/retrain the .pkl models
│   └── notebooks/
│       └── feature_engineering.ipynb # Jupyter notebook for exploring new features
│
├── .env                            # Stores secret API keys
├── .gitignore
├── main.py                         # Main entry point to run the bot
└── requirements.txt                # Project dependencies