# LLM-Advisor STDEV Refactor & AWS Integration Plan

## Overview

Refactor the trading system to focus on STDEV trading tactics with a clean architecture that supports:

1. Premarket data gathering and bias/news context
2. Live loop with technical feature computation and threshold-based triggers
3. **Periodic LLM market analysis (every 15 minutes)** that adjusts thresholds via multipliers
4. Trade execution when thresholds are met (with optional LLM validation)
5. AWS infrastructure in phases: Premarket Lambda → Database → Bedrock → RAG chatbot
6. Portfolio website integration via API Gateway

## Architecture Decision: Trading System First, Then API Layer

**Recommendation:** Build the trading system first, then add API Gateway layer.

**Rationale:**

- Need to understand data structures before designing API contracts
- Faster iteration on core logic without API constraints
- Can use API Gateway + Lambda to expose existing functionality without major refactor
- Natural separation: trading system generates data → API layer reads it
- Portfolio website can consume REST APIs regardless of internal implementation

**Implementation Path:**

1. Build complete trading system with local storage
2. Define data access patterns during development
3. Create Lambda functions that wrap data access
4. Expose via API Gateway as REST endpoints
5. Portfolio website consumes REST APIs

## Current State Analysis

**Existing Components:**

- `src/features/stdev_features.py` - RollingStats and z-score computation
- `src/strategy/live_loop_stdev.py` - STDEV live loop with state machine (120-window RollingStats)
- `src/data_processing/premarket_stdev.py` - Premarket snapshot computation (HTF stats, 5m bands)
- `src/data_processing/news_scraper.py` - News gathering
- `src/data_processing/daily_bias_computing.py` - ML model inference for daily bias
- `src/execution/order_manager.py` - Trade execution (currently options-focused)
- `main.py` - Premarket pipeline orchestrator

**Gaps to Address:**

- Premarket → Live loop integration missing
- LLM integration: periodic market analysis (15 min) + threshold multipliers
- No database layer for storing trades/results
- No AWS infrastructure
- Order manager needs STDEV stock-focused refactor
- Threshold adjustment system missing (LLM multiplier)

---

## Phase 1: Core System Refactor & LLM Integration

### Goal

Create a clean, modular trading system with STDEV logic, periodic LLM market analysis, and threshold adjustment system.

### 1.1 Project Structure Reorganization

**Create new directory structure:**

```
llm-advisor/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py          # Centralized config (Pydantic models)
│   │   └── logging.py          # Unified logging setup
│   ├── premarket/
│   │   ├── __init__.py
│   │   ├── bias_gatherer.py   # Daily bias + news gathering (combines existing scripts)
│   │   └── snapshot_builder.py # STDEV premarket snapshot (refactor premarket_stdev.py)
│   ├── live/
│   │   ├── __init__.py
│   │   ├── loop.py            # Main live loop orchestrator
│   │   ├── feature_computer.py # Technical feature computation (z-scores, ATR, etc.)
│   │   ├── threshold_evaluator.py # Threshold checking with multiplier support
│   │   └── state_manager.py   # Symbol state management
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── llm_client.py      # LLM client abstraction (OpenAI, Anthropic, future Bedrock)
│   │   ├── market_analyzer.py # Periodic market analysis (15 min intervals)
│   │   └── trade_validator.py # Optional LLM validation for trades
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── risk_calculator.py # Position sizing logic
│   │   ├── order_manager.py   # Refactored for stocks + bracket orders
│   │   └── trade_tracker.py  # Track open positions
│   ├── data/
│   │   ├── __init__.py
│   │   ├── alpaca_client.py   # Alpaca API wrapper (data + trading)
│   │   └── storage.py         # Database abstraction layer (SQLite dev, RDS prod)
│   └── features/
│       ├── stdev_features.py   # Keep existing RollingStats
│       └── technical_indicators.py # Additional STDEV indicators
├── config/
│   ├── settings.py            # Refactor to use Pydantic models
│   └── thresholds.py          # STDEV threshold configuration
├── scripts/
│   ├── run_premarket.py      # Premarket workflow orchestrator
│   └── run_live_loop.py      # Live trading workflow orchestrator
├── aws/
│   ├── lambda/
│   │   ├── premarket_handler.py
│   │   ├── live_loop_handler.py
│   │   ├── trade_executor_handler.py
│   │   └── position_monitor_handler.py
│   └── infrastructure/
│       └── cloudformation/    # Infrastructure as code
└── tests/
    └── ...                    # Unit tests for key modules
```

**Implementation Steps:**

1. Create all new directories with `__init__.py` files
2. Move existing files to appropriate locations
3. Update imports across codebase
4. Create stub files for new modules

### 1.2 Refactor Premarket Pipeline

**File:** `src/premarket/bias_gatherer.py`

**Purpose:** Integrate news scraping and daily bias computation into unified module.

**Implementation:**

- Import and wrap existing `news_scraper.py` logic
- Import and wrap existing `daily_bias_computing.py` logic
- Combine outputs into unified structure:
  ```python
  @dataclass
  class PremarketBias:
      symbol: str
      daily_bias: str  # "bullish", "bearish", "choppy"
      confidence: int   # 0-100
      model_output: Dict[str, Any]  # Raw ML model predictions
      news_summary: str  # Summarized news context
      premarket_price: float
      premarket_context: str  # LLM-generated context (optional)
      
  @dataclass
  class PremarketContext:
      date: str  # YYYY-MM-DD
      symbols: Dict[str, PremarketBias]
      market_context: Dict[str, Any]  # SPY bias, VIX, macro news
  ```

- Save to local JSON initially (migrate to DB in Phase 2)

**File:** `src/premarket/snapshot_builder.py`

**Purpose:** Refactor `premarket_stdev.py` to match new structure.

**Implementation:**

- Move `premarket_stdev.py` logic into this module
- Keep existing HTF stats computation (EMA slopes, HH/LL tags, ATR percentile)
- Keep existing 5m bands computation (mu, sigma, ATR)
- Output compatible with `live_loop_stdev.py` seed_state function
- Return `SymbolSnapshot` dataclass instances

**File:** `scripts/run_premarket.py`

**Purpose:** Replace `main.py` with cleaner orchestrator.

**Implementation:**

1. Run `bias_gatherer.py` → get premarket context
2. Run `snapshot_builder.py` → get STDEV snapshots
3. Combine outputs into single JSON file
4. Save to `data/daily_news/{date}/processed/premarket_context.json`
5. Log execution time and errors

**Success Criteria:**

- Single command runs entire premarket pipeline
- Output JSON contains all necessary data for live loop
- Compatible with existing live loop seed_state logic

### 1.3 Refactor Live Loop Architecture

**File:** `src/live/feature_computer.py`

**Purpose:** Extract and modularize technical feature computation.

**Implementation:**

- Move RollingStats updates from `live_loop_stdev.py`
- Compute z-scores, mu, sigma for each symbol
- Optionally compute additional indicators (RSI, MACD, etc.)
- Return structured feature dict:
  ```python
  @dataclass
  class SymbolFeatures:
      symbol: str
      z_score: float
      mu: float
      sigma: float
      atr_5m: float
      ema_slope_hourly: float
      atr_percentile: float
      timestamp: datetime
  ```


**File:** `src/live/threshold_evaluator.py`

**Purpose:** Extract threshold checking logic with multiplier support.

**Implementation:**

- Move threshold logic from `live_loop_stdev.py` evaluate_signals function
- Accept `ThresholdMultiplier` from market analyzer:
  ```python
  @dataclass
  class ThresholdMultiplier:
      mr_arm_multiplier: float = 1.0    # Multiply k1 by this
      mr_trigger_multiplier: float = 1.0 # Multiply k3 by this
      tc_arm_multiplier: float = 1.0     # Multiply k2 by this
      tc_trigger_multiplier: float = 1.0 # Multiply k3 by this
      confidence: float = 0.0            # LLM confidence in opportunity
      reasoning: str = ""                # LLM reasoning
  ```

- Apply multipliers to base thresholds when checking conditions
- Return signal events when thresholds crossed:
  ```python
  @dataclass
  class SignalEvent:
      symbol: str
      setup_type: str  # "MR" or "TC"
      side: str        # "long" or "short"
      entry_price: float
      z_score: float
      thresholds_used: Thresholds  # With multipliers applied
      timestamp: datetime
  ```


**File:** `src/live/state_manager.py`

**Purpose:** Manage symbol state and status transitions.

**Implementation:**

- Refactor `SymbolState` dataclass from `live_loop_stdev.py`
- Add `threshold_multiplier` field to SymbolState
- Track status transitions: idle → mr_armed → mr_triggered → trade_executed
- Store trade plans when signals trigger
- Update state based on price updates and threshold evaluations

**File:** `src/live/loop.py`

**Purpose:** Main orchestrator for live trading loop.

**Implementation Flow:**

```python
def main():
    # 1. Load premarket context
    premarket_context = load_premarket_context(date)
    premarket_snapshots = load_premarket_snapshots(date)
    
    # 2. Initialize symbol states
    states = seed_states_from_snapshots(premarket_snapshots, seed_bars)
    
    # 3. Initialize market analyzer (for periodic LLM calls)
    market_analyzer = MarketAnalyzer(interval_minutes=15)
    last_market_analysis = None
    
    # 4. Main loop
    while market_is_open():
        # 4a. Fetch latest price data (every 1 minute)
        current_bars = fetch_latest_bars(symbols)
        
        # 4b. Update features for each symbol
        for symbol in symbols:
            features = compute_features(symbol, current_bars, states[symbol])
            states[symbol].update_features(features)
        
        # 4c. Check if 15 minutes passed since last market analysis
        if should_run_market_analysis(market_analyzer, last_market_analysis):
            # Get LLM market consensus
            multiplier = market_analyzer.analyze_market(
                states=states,
                premarket_context=premarket_context,
                recent_price_action=current_bars
            )
            
            # Apply multiplier to all symbol states
            for symbol in symbols:
                states[symbol].threshold_multiplier = multiplier
            
            last_market_analysis = datetime.now()
        
        # 4d. Evaluate thresholds (with multipliers applied)
        for symbol in symbols:
            signal = evaluate_thresholds(
                state=states[symbol],
                current_price=current_bars[symbol].close,
                multiplier=states[symbol].threshold_multiplier
            )
            
            if signal:
                # 4e. Optional: LLM trade validation
                if should_validate_trade(signal):
                    validation = validate_trade_with_llm(signal, states[symbol])
                    if not validation.should_execute:
                        continue
                
                # 4f. Execute trade
                execute_trade(signal, states[symbol])
        
        # 4g. Monitor open positions
        update_positions()
        
        # 4h. Sleep until next iteration
        time.sleep(60)  # 1 minute
```

**Key Integration Points:**

- Uses `feature_computer.py` for technical calculations
- Uses `threshold_evaluator.py` for signal detection
- Uses `state_manager.py` for state updates
- Calls `market_analyzer.py` every 15 minutes
- Calls `trade_validator.py` optionally before execution

### 1.4 LLM Integration: Periodic Market Analysis

**File:** `src/analysis/llm_client.py`

**Purpose:** Abstract LLM provider calls.

**Implementation:**

- Support OpenAI, Anthropic APIs initially
- Structured JSON response parsing
- Rate limiting and retry logic
- Future: AWS Bedrock integration (Phase 6)
- Return structured responses:
  ```python
  @dataclass
  class LLMResponse:
      content: Dict[str, Any]
      model: str
      tokens_used: int
      latency_ms: float
  ```


**File:** `src/analysis/market_analyzer.py`

**Purpose:** Periodic market analysis every 15 minutes.

**Implementation:**

```python
class MarketAnalyzer:
    def __init__(self, llm_client: LLMClient, interval_minutes: int = 15):
        self.llm_client = llm_client
        self.interval_minutes = interval_minutes
    
    def analyze_market(
        self,
        states: Dict[str, SymbolState],
        premarket_context: PremarketContext,
        recent_price_action: Dict[str, Any]
    ) -> ThresholdMultiplier:
        """
        Prompt LLM with:
        - Current technical features (z-scores, ATR, etc.) for all symbols
        - Premarket context (bias, news)
        - Recent price action trends
        
        Ask LLM:
        - "What's your assessment of current market conditions?"
        - "Do you see any good trading opportunities coming up?"
        - "How confident are you in these opportunities?"
        
        Parse response to get multiplier adjustments.
        """
        prompt = build_market_analysis_prompt(
            states=states,
            premarket_context=premarket_context,
            recent_price_action=recent_price_action
        )
        
        response = self.llm_client.call_structured(
            prompt=prompt,
            schema=market_analysis_schema
        )
        
        # Parse response to ThresholdMultiplier
        return parse_multiplier_response(response)
```

**Prompt Structure:**

```
You are analyzing the current market state for a STDEV trading system.

Current Technical State:
- NVDA: z-score=1.5, ATR percentile=65%, HTF bias=bullish
- SPY: z-score=0.8, ATR percentile=70%, HTF bias=bullish
... (all symbols)

Premarket Context:
- Daily biases: NVDA=bullish (75% confidence), SPY=bullish (80% confidence)
- News: "Fed signals dovish stance..."
- VIX: 15.2

Recent Price Action (last 15 minutes):
- NVDA: +0.5%, showing strength
- SPY: +0.2%, steady uptrend

Questions:
1. What's your assessment of current market conditions?
2. Do you see any good trading opportunities coming up in the next 15-30 minutes?
3. How confident are you in these opportunities? (0-100)
4. Should we adjust our trading thresholds? If yes, suggest multipliers:
   - If highly confident: lower thresholds (multiplier < 1.0, e.g., 0.8)
   - If less confident: raise thresholds (multiplier > 1.0, e.g., 1.2)
   - If neutral: keep thresholds as-is (multiplier = 1.0)

Return JSON:
{
  "market_assessment": "...",
  "opportunities": ["...", "..."],
  "confidence": 85,
  "threshold_multipliers": {
    "mr_arm_multiplier": 0.9,
    "mr_trigger_multiplier": 1.0,
    "tc_arm_multiplier": 0.85,
    "tc_trigger_multiplier": 1.0
  },
  "reasoning": "..."
}
```

**File:** `src/analysis/trade_validator.py`

**Purpose:** Optional LLM validation before trade execution.

**Implementation:**

- Called when threshold triggers but before execution
- Can be disabled via config (default: enabled)
- Build prompt with specific trade setup:
  ```python
  def validate_trade(signal: SignalEvent, state: SymbolState) -> TradeValidation:
      prompt = f"""
      A trade signal has been triggered:
      Symbol: {signal.symbol}
      Setup: {signal.setup_type} ({signal.side})
      Entry: {signal.entry_price}
      Stop Loss: {calculated_sl}
      Take Profit: {calculated_tp}
      
      Technical Context:
      - z-score: {state.last_z}
      - ATR percentile: {state.atr_percentile}
      - HTF bias: {state.htf_bias}
      
      Premarket Context:
      {premarket_context[symbol]}
      
      Should we execute this trade? Analyze risk/reward.
      """
      
      response = llm_client.call_structured(prompt, schema=trade_validation_schema)
      return parse_validation_response(response)
  ```


**Success Criteria:**

- Market analyzer runs every 15 minutes
- Multipliers applied to threshold checks
- LLM responses parsed correctly
- System continues operating if LLM call fails (graceful degradation)

### 1.5 Execution Refactor

**File:** `src/execution/risk_calculator.py`

**Purpose:** Calculate position sizing based on risk parameters.

**Implementation:**

```python
def calculate_position_size(
    account_equity: float,
    entry_price: float,
    stop_loss_price: float,
    max_risk_percent: float,
    atr_5m: float
) -> int:
    """
    Calculate position size in shares.
    
    Logic:
    1. Calculate risk per share (entry - stop_loss)
    2. Calculate total risk amount (equity * max_risk_percent / 100)
    3. Calculate shares (total_risk / risk_per_share)
    4. Apply ATR-based position sizing (optional)
    """
    risk_per_share = abs(entry_price - stop_loss_price)
    if risk_per_share == 0:
        return 0
    
    total_risk = account_equity * (max_risk_percent / 100)
    shares = int(total_risk / risk_per_share)
    
    # Optionally adjust based on ATR
    # Larger ATR = smaller position size
    return shares
```

**File:** `src/execution/order_manager.py`

**Purpose:** Refactor for stock trading with bracket orders.

**Implementation:**

- Remove options-specific logic
- Use Alpaca bracket orders for stocks:
  ```python
  def execute_stock_trade(
      symbol: str,
      side: str,  # "buy" or "sell"
      entry_price: float,
      stop_loss: float,
      take_profit: float,
      qty: int
  ) -> Order:
      """
      Place bracket order:
      - Market order for entry
      - Stop loss order
      - Take profit limit order
      """
      bracket_order = BracketOrderRequest(
          symbol=symbol,
          qty=qty,
          side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
          time_in_force=TimeInForce.DAY,
          order_class=OrderClass.BRACKET,
          stop_loss=StopLossRequest(stop_price=stop_loss),
          take_profit=TakeProfitRequest(limit_price=take_profit)
      )
      
      return trading_client.submit_order(order_data=bracket_order)
  ```

- Handle order lifecycle
- Store order ID in database

**File:** `src/execution/trade_tracker.py`

**Purpose:** Track open positions and monitor for exits.

**Implementation:**

- Query Alpaca for open positions
- Update position status in database
- Calculate unrealized P/L
- Check for stop loss/take profit fills
- Handle manual position management

**Success Criteria:**

- Stock trades execute via bracket orders
- Position sizing calculated correctly
- Positions tracked and monitored
- P/L calculated accurately

### 1.6 Configuration Refactor

**File:** `config/thresholds.py`

**Purpose:** STDEV threshold configuration with multiplier support.

**Implementation:**

```python
from dataclasses import dataclass
from pydantic import BaseModel

@dataclass
class STDEVThresholds:
    # Mean reversion
    mr_arm_z: float = 1.2      # Base threshold: arm when |z| >= this
    mr_trigger_z: float = 0.6   # Base threshold: trigger when z returns to this
    
    # Trend continuation
    tc_arm_z: float = 1.8       # Base threshold: arm when |z| >= this with trend
    tc_trigger_z: float = 0.6   # Base threshold: trigger when z continues
    
    # Risk management
    atr_multiplier_sl: float = 1.4  # Stop loss = entry ± ATR * this
    min_rr_ratio: float = 1.5        # Minimum risk:reward
    max_risk_per_trade: float = 1.0 # % of account
    
    # Filtering
    atr_percentile_cap: float = 85.0 # Only trade if ATR percentile <= this
    
    def apply_multiplier(self, multiplier: ThresholdMultiplier) -> "STDEVThresholds":
        """Return new thresholds with multiplier applied."""
        return STDEVThresholds(
            mr_arm_z=self.mr_arm_z * multiplier.mr_arm_multiplier,
            mr_trigger_z=self.mr_trigger_z * multiplier.mr_trigger_multiplier,
            tc_arm_z=self.tc_arm_z * multiplier.tc_arm_multiplier,
            tc_trigger_z=self.tc_trigger_z * multiplier.tc_trigger_multiplier,
            atr_multiplier_sl=self.atr_multiplier_sl,
            min_rr_ratio=self.min_rr_ratio,
            max_risk_per_trade=self.max_risk_per_trade,
            atr_percentile_cap=self.atr_percentile_cap
        )
```

**File:** `config/settings.py`

**Purpose:** Centralized configuration with Pydantic validation.

**Implementation:**

```python
from pydantic import BaseModel, Field
from typing import List

class TradingSettings(BaseModel):
    watchlist: List[str] = Field(default=["SPY", "QQQ", "IWM", "NVDA", "TSLA"])
    trading_window_start: str = "09:30"
    trading_window_end: str = "12:00"
    end_of_day_close_time: str = "15:50"
    
class RiskSettings(BaseModel):
    max_risk_per_trade_percent: float = Field(default=1.0, ge=0.1, le=5.0)
    min_risk_reward_ratio: float = Field(default=1.5, ge=1.0)
    
class LLMSettings(BaseModel):
    provider: str = "openai"  # "openai", "anthropic", "bedrock"
    model: str = "gpt-4o-mini"
    market_analysis_interval_minutes: int = Field(default=15, ge=5, le=60)
    enable_trade_validation: bool = True
    
class Settings(BaseModel):
    trading: TradingSettings = TradingSettings()
    risk: RiskSettings = RiskSettings()
    llm: LLMSettings = LLMSettings()
    
    @classmethod
    def load(cls) -> "Settings":
        """Load from environment variables or config file."""
        # Implementation
```

**Success Criteria:**

- All configuration validated with Pydantic
- Environment-specific configs (dev, prod)
- Thresholds can be adjusted via multipliers
- Settings loaded from .env file

---

## Phase 2: Database Layer & Storage Abstraction

### Goal

Design and implement database schema for storing all trading data, with abstraction layer supporting local SQLite (dev) and AWS RDS PostgreSQL (prod).

### 2.1 Database Schema Design

**Decision Point:** After Phase 1 implementation, we'll understand:

- What data we generate daily
- What queries we need to run
- Data volume and access patterns

**Proposed Schema (PostgreSQL):**

**Table: `daily_bias`**

```sql
CREATE TABLE daily_bias (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    bias VARCHAR(20) NOT NULL,  -- 'bullish', 'bearish', 'choppy'
    confidence INTEGER NOT NULL,  -- 0-100
    model_output JSONB,
    news_summary TEXT,
    premarket_price DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(date, symbol)
);
CREATE INDEX idx_daily_bias_date ON daily_bias(date);
CREATE INDEX idx_daily_bias_symbol ON daily_bias(symbol);
```

**Table: `premarket_snapshots`**

```sql
CREATE TABLE premarket_snapshots (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    htf_stats JSONB,  -- {ema_slope_daily, ema_slope_hourly, hh_ll_tag, atr_percentile}
    bands_5m JSONB,   -- {mu, sigma, atr_5m}
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(date, symbol)
);
```

**Table: `market_analysis`**

```sql
CREATE TABLE market_analysis (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    analysis_text TEXT,
    threshold_multipliers JSONB,  -- {mr_arm_multiplier, tc_arm_multiplier, ...}
    confidence INTEGER,
    llm_model VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_market_analysis_timestamp ON market_analysis(timestamp);
```

**Table: `live_loop_logs`**

```sql
CREATE TABLE live_loop_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    z_score DECIMAL(10, 4),
    mu DECIMAL(10, 4),
    sigma DECIMAL(10, 4),
    status VARCHAR(20),  -- 'idle', 'mr_armed', 'tc_triggered', etc.
    side VARCHAR(10),    -- 'long', 'short', NULL
    current_price DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_live_loop_timestamp ON live_loop_logs(timestamp);
CREATE INDEX idx_live_loop_symbol_date ON live_loop_logs(symbol, timestamp);
```

**Table: `trade_signals`**

```sql
CREATE TABLE trade_signals (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    setup_type VARCHAR(10) NOT NULL,  -- 'MR', 'TC'
    side VARCHAR(10) NOT NULL,
    entry_price DECIMAL(10, 2),
    z_score DECIMAL(10, 4),
    threshold_multipliers JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

**Table: `llm_validations`**

```sql
CREATE TABLE llm_validations (
    id SERIAL PRIMARY KEY,
    signal_id INTEGER REFERENCES trade_signals(id),
    timestamp TIMESTAMP NOT NULL,
    should_execute BOOLEAN,
    confidence INTEGER,
    reasoning TEXT,
    llm_model VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);
```

**Table: `trades`**

```sql
CREATE TABLE trades (
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
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_entry_time ON trades(entry_time);
```

**Table: `positions`**

```sql
CREATE TABLE positions (
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
    last_updated TIMESTAMP DEFAULT NOW(),
    UNIQUE(trade_id)
);
CREATE INDEX idx_positions_symbol ON positions(symbol);
```

### 2.2 Storage Abstraction Layer

**File:** `src/data/storage.py`

**Purpose:** Abstract database operations, support SQLite (dev) and PostgreSQL (prod).

**Implementation:**

```python
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime, date
import sqlite3
from contextlib import contextmanager

class StorageAdapter(ABC):
    @abstractmethod
    def save_daily_bias(self, date: date, symbol: str, bias_data: Dict) -> None:
        pass
    
    @abstractmethod
    def get_daily_bias(self, date: date, symbol: str) -> Optional[Dict]:
        pass
    
    @abstractmethod
    def save_market_analysis(self, analysis: Dict) -> None:
        pass
    
    @abstractmethod
    def save_live_loop_log(self, log_entry: Dict) -> None:
        pass
    
    @abstractmethod
    def save_trade_signal(self, signal: Dict) -> int:
        pass
    
    @abstractmethod
    def save_trade(self, trade: Dict) -> None:
        pass
    
    @abstractmethod
    def update_position(self, position: Dict) -> None:
        pass

class SQLiteStorage(StorageAdapter):
    """Local development storage."""
    def __init__(self, db_path: str = "data/trading.db"):
        self.db_path = db_path
        self._init_schema()
    
    def _init_schema(self):
        # Create tables if not exist
        pass

class PostgreSQLStorage(StorageAdapter):
    """Production storage."""
    def __init__(self, connection_string: str):
        self.conn_string = connection_string
    
    @contextmanager
    def get_connection(self):
        conn = psycopg2.connect(self.conn_string)
        try:
            yield conn
            conn.commit()
        except:
            conn.rollback()
            raise
        finally:
            conn.close()

class Storage:
    """Factory for storage adapters."""
    @staticmethod
    def create(env: str = "dev") -> StorageAdapter:
        if env == "dev":
            return SQLiteStorage()
        else:
            conn_string = os.getenv("DATABASE_URL")
            return PostgreSQLStorage(conn_string)
```

**Integration Points:**

- `premarket/bias_gatherer.py` → saves daily bias
- `premarket/snapshot_builder.py` → saves premarket snapshots
- `analysis/market_analyzer.py` → saves market analysis
- `live/loop.py` → saves live loop logs
- `live/threshold_evaluator.py` → saves trade signals
- `execution/order_manager.py` → saves trades
- `execution/trade_tracker.py` → updates positions

**Success Criteria:**

- SQLite works locally for development
- PostgreSQL schema ready for AWS RDS
- All trading operations save to database
- Queries supported for portfolio website

---

## Phase 3: AWS Infrastructure - Premarket Lambda (First Priority)

### Goal

Migrate premarket pipeline to AWS Lambda with EventBridge scheduling.

### 3.1 Premarket Lambda Function

**File:** `aws/lambda/premarket_handler.py`

**Purpose:** Lambda handler for premarket pipeline.

**Implementation:**

```python
import json
from src.premarket.bias_gatherer import gather_premarket_bias
from src.premarket.snapshot_builder import build_premarket_snapshots
from src.data.storage import Storage

def lambda_handler(event, context):
    """
    EventBridge scheduled event triggers this daily at 8:30 AM EST.
    """
    try:
        # Get today's date
        date = get_trading_date()
        
        # Run premarket pipeline
        bias_context = gather_premarket_bias(date)
        snapshots = build_premarket_snapshots(date)
        
        # Save to database
        storage = Storage.create(env="prod")
        storage.save_daily_bias(date, bias_context)
        storage.save_premarket_snapshots(date, snapshots)
        
        # Also save to S3 for backup/archival
        save_to_s3(date, bias_context, snapshots)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'date': date,
                'symbols_processed': len(bias_context.symbols)
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

**Lambda Configuration:**

- Runtime: Python 3.11
- Timeout: 15 minutes
- Memory: 512 MB
- Environment variables: API keys, database URL
- Layers: Include dependencies (pandas, numpy, etc.)

### 3.2 EventBridge Rule

**Configuration:**

- Rule name: `premarket-pipeline-daily`
- Schedule: `cron(30 8 ? * MON-FRI *)` (8:30 AM EST = 13:30 UTC, Monday-Friday)
- Target: Premarket Lambda function
- Enable rule: Yes

### 3.3 S3 Integration

**Purpose:** Backup and archival of premarket data.

**Implementation:**

- S3 bucket: `llm-advisor-premarket-data`
- Structure: `s3://bucket/{year}/{month}/{date}/premarket_context.json`
- Also store snapshots as JSON files
- Enable versioning for audit trail

**Success Criteria:**

- Lambda runs daily at scheduled time
- Premarket data saved to RDS and S3
- Errors logged to CloudWatch
- Can monitor execution in AWS Console

---

## Phase 4: Database Decision & AWS RDS Setup

### Goal

After understanding data requirements from Phase 1-2, choose and set up production database.

### 4.1 Database Decision Matrix

**Evaluate after Phase 1-2 completion:**

**DynamoDB Pros:**

- Fast writes (high-frequency live loop logs)
- Serverless, auto-scaling
- Good for time-series data

**DynamoDB Cons:**

- Less flexible queries
- Harder to do complex analytics
- More expensive at scale

**RDS PostgreSQL Pros:**

- Flexible SQL queries
- Better for portfolio website queries
- Easier to migrate existing data
- Support for complex joins

**RDS PostgreSQL Cons:**

- Need to manage scaling
- Connection pooling required
- More setup complexity

**Recommendation:** Start with RDS PostgreSQL for flexibility. Migrate to DynamoDB later if needed for high-frequency writes.

### 4.2 RDS PostgreSQL Setup

**Configuration:**

- Engine: PostgreSQL 15
- Instance: db.t3.micro (for development), db.t3.small (for production)
- Storage: 20 GB GP3 (auto-scaling enabled)
- Multi-AZ: Disabled initially (enable for production)
- Backup: 7-day retention
- Security: VPC with private subnet
- Parameter group: Optimized for connection pooling

**Connection Management:**

- Use connection pooling (PgBouncer or RDS Proxy)
- Environment variable: `DATABASE_URL`
- SSL required for connections

**Migration Scripts:**

- `aws/infrastructure/database/migrations/001_initial_schema.sql`
- Use Alembic or similar for version control
- Run migrations on Lambda startup or via separate process

**Success Criteria:**

- RDS instance accessible from Lambda
- All tables created via migrations
- Connection pooling configured
- Backup strategy in place

---

## Phase 5: Live Loop Lambda & EventBridge

### Goal

Migrate live loop to AWS Lambda with EventBridge scheduling during market hours.

### 5.1 Live Loop Lambda Function

**File:** `aws/lambda/live_loop_handler.py`

**Purpose:** Lambda handler for live trading loop.

**Implementation:**

```python
def lambda_handler(event, context):
    """
    EventBridge triggers this every 1 minute during market hours.
    """
    try:
        # Load premarket context
        date = get_trading_date()
        premarket_context = load_premarket_context(date)
        
        # Initialize or load state from DynamoDB/RDS
        # (State needs to persist between Lambda invocations)
        states = load_symbol_states(date)
        
        # Check if 15 minutes passed for market analysis
        if should_run_market_analysis():
            multiplier = run_market_analysis(states, premarket_context)
            save_market_analysis(multiplier)
            apply_multiplier_to_states(states, multiplier)
        
        # Fetch latest price data
        current_bars = fetch_latest_bars(symbols)
        
        # Compute features and evaluate thresholds
        signals = []
        for symbol in symbols:
            features = compute_features(symbol, current_bars, states[symbol])
            states[symbol].update_features(features)
            
            signal = evaluate_thresholds(
                state=states[symbol],
                current_price=current_bars[symbol].close,
                multiplier=states[symbol].threshold_multiplier
            )
            
            if signal:
                signals.append(signal)
        
        # Save state back to database
        save_symbol_states(states)
        save_live_loop_log(states)
        
        # Send signals to SQS for trade execution
        if signals:
            send_signals_to_sqs(signals)
        
        return {'statusCode': 200}
    except Exception as e:
        log_error(e)
        return {'statusCode': 500}
```

**State Persistence:**

- Store symbol states in DynamoDB or RDS
- Key: `{date}#{symbol}`
- Update on each Lambda invocation
- Load at start of each invocation

**Lambda Configuration:**

- Runtime: Python 3.11
- Timeout: 3 minutes
- Memory: 512 MB
- Reserved concurrency: 1 (prevent overlapping executions)

### 5.2 EventBridge Rules

**Live Loop Rule:**

- Rule name: `live-loop-market-hours`
- Schedule: `cron(*/1 9-16 ? * MON-FRI *)` (Every minute, 9:30 AM - 4:00 PM EST)
- Target: Live Loop Lambda
- Enable: Yes

**Market Analysis Rule (Alternative):**

- If live loop runs every minute, use internal timer
- Or separate rule every 15 minutes calling market analyzer Lambda

### 5.3 Trade Execution Queue

**SQS Queue:**

- Queue name: `trade-signals`
- Type: Standard queue (FIFO if order matters)
- Dead-letter queue: `trade-signals-dlq`
- Visibility timeout: 30 seconds

**Trade Executor Lambda:**

- Triggered by SQS messages
- Executes trades via Alpaca
- Updates database with trade records
- Handles errors gracefully

**Success Criteria:**

- Lambda runs every minute during market hours
- State persists between invocations
- Signals sent to SQS
- Trades execute correctly

---

## Phase 6: AWS Bedrock Integration

### Goal

Migrate LLM calls from OpenAI/Anthropic to AWS Bedrock for cost savings and better integration.

### 6.1 Bedrock Setup

**Services:**

- AWS Bedrock (model access)
- Request Claude 3 Sonnet or Llama 3 models
- Set up IAM roles for Lambda access

**File:** `src/analysis/llm_client.py` (update)

**Implementation:**

```python
class BedrockLLMClient(LLMClient):
    def __init__(self, model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"):
        self.bedrock = boto3.client('bedrock-runtime')
        self.model_id = model_id
    
    def call_structured(self, prompt: str, schema: Dict) -> Dict:
        # Build Bedrock request
        # Parse structured response
        # Return JSON
```

**Update `market_analyzer.py` and `trade_validator.py`:**

- Use Bedrock client when `LLM_PROVIDER=bedrock`
- Fallback to OpenAI/Anthropic if Bedrock unavailable

**Success Criteria:**

- Bedrock models accessible from Lambda
- Market analysis uses Bedrock
- Trade validation uses Bedrock
- Cost savings verified

---

## Phase 7: Portfolio Website Integration

### Goal

Expose trading data via API Gateway for portfolio website consumption.

### 7.1 API Gateway Design

**Endpoints:**

1. **GET /api/daily-bias**

            - Query params: `date` (optional, defaults to today), `symbol` (optional)
            - Returns: Daily bias predictions for date/symbol
            - Lambda: `api-get-daily-bias`

2. **GET /api/trades**

            - Query params: `start_date`, `end_date`, `symbol` (optional)
            - Returns: Trade history with filters
            - Lambda: `api-get-trades`

3. **GET /api/pnl**

            - Query params: `start_date`, `end_date`
            - Returns: P/L summary over period
            - Lambda: `api-get-pnl`

4. **GET /api/positions**

            - Returns: Current open positions
            - Lambda: `api-get-positions`

5. **GET /api/market-analysis**

            - Query params: `timestamp` (optional, latest if not provided)
            - Returns: Recent market analysis from LLM
            - Lambda: `api-get-market-analysis`

**Lambda Functions:**

**File:** `aws/lambda/api-get-daily-bias.py`

```python
def lambda_handler(event, context):
    date = event.get('queryStringParameters', {}).get('date')
    symbol = event.get('queryStringParameters', {}).get('symbol')
    
    storage = Storage.create(env="prod")
    bias_data = storage.get_daily_bias(date, symbol)
    
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps(bias_data)
    }
```

**API Gateway Configuration:**

- REST API
- CORS enabled for portfolio website domain
- Authentication: API Key (optional, for public access)
- Rate limiting: 100 requests/minute
- Integration: Lambda proxy integration

**Success Criteria:**

- All endpoints return correct data
- CORS configured properly
- Portfolio website can consume APIs
- Rate limiting prevents abuse

### 7.2 Data Export for RAG (Future)

**File:** `src/api/export_for_rag.py`

**Purpose:** Structure trading data for vector embeddings.

**Implementation:**

- Export daily biases, trades, market analysis as structured JSON
- Include metadata (dates, symbols, context)
- Format for embedding generation
- Store in S3 for RAG system consumption

**Future Integration:**

- AWS Bedrock Knowledge Bases
- Amazon OpenSearch
- Vector embeddings via Bedrock Embeddings API

---

## Phase 8: RAG Chatbot (Long-term)

### Goal

Implement chatbot for portfolio website using RAG on trading data.

### 8.1 Architecture

**Components:**

- AWS Bedrock Knowledge Bases
- Amazon OpenSearch (vector store)
- Lambda function for query processing
- API Gateway endpoint for chatbot

**Data Sources:**

- Daily bias predictions
- Trade history
- Market analysis logs
- P/L summaries

**Query Examples:**

- "What were the daily biases for NVDA on October 27th?"
- "What is the P/L of the paper trading account over the last 5 days?"
- "Show me all trades executed in the last week"

**Implementation:**

- Vectorize trading data documents
- Store in OpenSearch
- Query via Bedrock Knowledge Bases
- Return natural language responses

**Success Criteria:**

- Chatbot answers questions accurately
- Integrates with portfolio website
- Handles complex queries
- Provides helpful context

---

## Implementation Timeline

**Phase 1: Core System Refactor (Weeks 1-3)**

- Week 1: Project reorganization, premarket refactor
- Week 2: Live loop refactor, LLM integration (market analysis)
- Week 3: Execution refactor, configuration, testing

**Phase 2: Database Layer (Week 4)**

- Design schema
- Implement storage abstraction
- Migrate to database
- Test queries

**Phase 3: AWS Premarket Lambda (Week 5)**

- Set up Lambda function
- Configure EventBridge
- Set up S3 integration
- Test end-to-end

**Phase 4: Database Setup (Week 6)**

- Set up RDS PostgreSQL
- Run migrations
- Configure connection pooling
- Test connectivity

**Phase 5: Live Loop Lambda (Week 7)**

- Set up Lambda function
- Configure EventBridge
- Set up SQS queue
- Test trade execution

**Phase 6: Bedrock Integration (Week 8)**

- Request model access
- Update LLM client
- Test market analysis
- Verify cost savings

**Phase 7: API Gateway (Week 9)**

- Create Lambda functions
- Set up API Gateway
- Configure CORS
- Test endpoints

**Phase 8: RAG Chatbot (Future)**

- Set up vector store
- Create embeddings
- Build chatbot
- Integrate with website

---

## Success Metrics

**Phase 1:**

- ✅ Premarket pipeline runs successfully
- ✅ Live loop computes features correctly
- ✅ LLM market analysis runs every 15 minutes
- ✅ Threshold multipliers applied correctly
- ✅ Trades execute via bracket orders

**Phase 2:**

- ✅ All data saved to database
- ✅ Queries return correct results
- ✅ Storage abstraction works for dev/prod

**Phase 3:**

- ✅ Premarket Lambda runs daily
- ✅ Data saved to RDS and S3
- ✅ Errors logged to CloudWatch

**Phase 4:**

- ✅ RDS accessible from Lambda
- ✅ Connection pooling configured
- ✅ Migrations run successfully

**Phase 5:**

- ✅ Live loop Lambda runs every minute
- ✅ State persists between invocations
- ✅ Trades execute via SQS

**Phase 6:**

- ✅ Bedrock models accessible
- ✅ Market analysis uses Bedrock
- ✅ Cost savings verified

**Phase 7:**

- ✅ API endpoints return correct data
- ✅ Portfolio website consumes APIs
- ✅ CORS configured correctly

**Phase 8:**

- ✅ Chatbot answers questions accurately
- ✅ Integrates with portfolio website

---

## Key Decisions & Rationale

1. **Trading System First, Then API Layer:** Faster iteration, API Gateway can wrap existing functionality
2. **RDS PostgreSQL Initially:** More flexible for queries, easier migration, can move to DynamoDB later if needed
3. **Periodic LLM Analysis (15 min):** Reduces API costs, provides market context without overwhelming system
4. **Threshold Multipliers:** Allows LLM to influence trading decisions without directly executing trades
5. **Bracket Orders:** Simpler than manual stop loss/take profit management
6. **SQS for Trade Execution:** Decouples signal detection from execution, allows retry logic
7. **State Persistence:** Required for Lambda stateless architecture

---

## Migration Strategy for Existing Data

**Data to Migrate:**

- Daily bias predictions (if useful for context)
- Premarket snapshots (if recent enough)
- Will be determined during Phase 1-2 implementation

**Migration Process:**

- Export existing JSON files
- Transform to database schema
- Import via migration script
- Verify data integrity

---

## Risk Mitigation

1. **Lambda Timeouts:** Increase timeout, optimize code, use Step Functions for long-running tasks
2. **Database Connection Limits:** Use connection pooling, RDS Proxy
3. **LLM API Failures:** Graceful degradation, fallback to base thresholds
4. **Trade Execution Failures:** SQS retry logic, dead-letter queue, manual review
5. **State Loss:** Persist state to database, load on each Lambda invocation

---

## Future Enhancements

- SageMaker integration for daily bias models
- Advanced risk management (portfolio-level)
- Backtesting framework integration
- Real-time WebSocket updates for portfolio website
- Multi-strategy support
- Performance analytics dashboard