-- Initial database schema for LLM-Advisor trading system
-- Supports PostgreSQL (production) and SQLite (development)

-- Table: daily_bias
-- Stores daily bias predictions per symbol
CREATE TABLE IF NOT EXISTS daily_bias (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    bias VARCHAR(20) NOT NULL,  -- 'bullish', 'bearish', 'choppy'
    confidence INTEGER NOT NULL,  -- 0-100
    model_output JSONB,
    news_summary TEXT,
    premarket_price DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, symbol)
);

CREATE INDEX IF NOT EXISTS idx_daily_bias_date ON daily_bias(date);
CREATE INDEX IF NOT EXISTS idx_daily_bias_symbol ON daily_bias(symbol);

-- Table: premarket_snapshots
-- Stores STDEV premarket snapshots (HTF stats, 5m bands)
CREATE TABLE IF NOT EXISTS premarket_snapshots (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    htf_stats JSONB,  -- {ema_slope_daily, ema_slope_hourly, hh_ll_tag, atr_percentile_daily}
    bands_5m JSONB,   -- {mu, sigma, atr_5m, k}
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, symbol)
);

CREATE INDEX IF NOT EXISTS idx_premarket_snapshots_date ON premarket_snapshots(date);
CREATE INDEX IF NOT EXISTS idx_premarket_snapshots_symbol ON premarket_snapshots(symbol);

-- Table: market_analysis
-- Stores periodic LLM market analysis (every 15 minutes)
CREATE TABLE IF NOT EXISTS market_analysis (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    analysis_text TEXT,
    threshold_multipliers JSONB,  -- {mr_arm_multiplier, tc_arm_multiplier, ...}
    confidence INTEGER,
    llm_model VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_market_analysis_timestamp ON market_analysis(timestamp);

-- Table: live_loop_logs
-- Stores minute-by-minute symbol state snapshots
CREATE TABLE IF NOT EXISTS live_loop_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    z_score DECIMAL(10, 4),
    mu DECIMAL(10, 4),
    sigma DECIMAL(10, 4),
    status VARCHAR(20),  -- 'idle', 'mr_armed', 'tc_triggered', etc.
    side VARCHAR(10),    -- 'long', 'short', NULL
    current_price DECIMAL(10, 2),
    atr_percentile DECIMAL(5, 2),
    ema_slope_hourly DECIMAL(10, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_live_loop_timestamp ON live_loop_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_live_loop_symbol_date ON live_loop_logs(symbol, timestamp);

-- Table: trade_signals
-- Stores detected trading signals
CREATE TABLE IF NOT EXISTS trade_signals (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    setup_type VARCHAR(10) NOT NULL,  -- 'MR', 'TC'
    side VARCHAR(10) NOT NULL,
    entry_price DECIMAL(10, 2),
    z_score DECIMAL(10, 4),
    threshold_multipliers JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trade_signals_timestamp ON trade_signals(timestamp);
CREATE INDEX IF NOT EXISTS idx_trade_signals_symbol ON trade_signals(symbol);

-- Table: llm_validations
-- Stores LLM trade validation results
CREATE TABLE IF NOT EXISTS llm_validations (
    id SERIAL PRIMARY KEY,
    signal_id INTEGER REFERENCES trade_signals(id),
    timestamp TIMESTAMP NOT NULL,
    should_execute BOOLEAN,
    confidence INTEGER,
    reasoning TEXT,
    risk_assessment VARCHAR(20),
    llm_model VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_llm_validations_signal_id ON llm_validations(signal_id);

-- Table: trades
-- Stores executed trades
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    trade_id VARCHAR(50) UNIQUE,  -- Alpaca order ID
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(10) NOT NULL,
    entry_price DECIMAL(10, 2),
    stop_loss DECIMAL(10, 2),
    take_profit DECIMAL(10, 2),
    qty INTEGER,
    status VARCHAR(20),  -- 'pending', 'filled', 'closed', 'cancelled'
    entry_time TIMESTAMP,
    exit_time TIMESTAMP,
    exit_price DECIMAL(10, 2),
    pnl DECIMAL(10, 2),
    exit_reason VARCHAR(50),  -- 'stop_loss', 'take_profit', 'eod', 'manual'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);

-- Table: positions
-- Stores current open positions
CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    trade_id INTEGER REFERENCES trades(id),
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(10) NOT NULL,
    entry_price DECIMAL(10, 2),
    current_price DECIMAL(10, 2),
    stop_loss DECIMAL(10, 2),
    take_profit DECIMAL(10, 2),
    qty INTEGER,
    unrealized_pnl DECIMAL(10, 2),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(trade_id)
);

CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_trade_id ON positions(trade_id);

