"""Centralized configuration using Pydantic models."""
from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()


class TradingSettings(BaseModel):
    watchlist: List[str] = Field(default=["SPY", "QQQ", "IWM", "NVDA", "TSLA"])
    trading_window_start: str = "09:30"
    trading_window_end: str = "12:00"
    end_of_day_close_time: str = "15:50"
    max_concurrent_trades: int = Field(default=1, ge=1, le=20)
    instrument: str = Field(default="options")
    allow_stock_fallback: bool = False


class RiskSettings(BaseModel):
    max_risk_per_trade_percent: float = Field(default=1.0, ge=0.1, le=5.0)
    min_risk_reward_ratio: float = Field(default=1.5, ge=1.0)
    max_position_notional_pct: float = Field(default=0.95, ge=0.1, le=1.0)


class OptionsSettings(BaseModel):
    paper_only: bool = True
    strategy_type: str = "single_long"
    min_dte: int = Field(default=7, ge=0)
    max_dte: int = Field(default=14, ge=1)
    min_delta: float = Field(default=0.35, ge=0.0, le=1.0)
    max_delta: float = Field(default=0.55, ge=0.0, le=1.0)
    max_premium_per_trade: float = Field(default=200.0, ge=1.0)
    max_bid_ask_spread_pct: float = Field(default=0.15, ge=0.0, le=1.0)
    min_open_interest: int = Field(default=100, ge=0)
    strike_window_pct: float = Field(default=0.10, ge=0.0, le=1.0)
    contract_limit: int = Field(default=100, ge=1, le=500)
    fallback_enabled: bool = False
    fallback_min_dte: int = Field(default=3, ge=0)
    fallback_max_dte: int = Field(default=21, ge=1)
    fallback_min_delta: float = Field(default=0.25, ge=0.0, le=1.0)
    fallback_max_delta: float = Field(default=0.65, ge=0.0, le=1.0)
    fallback_max_premium_per_trade: float = Field(default=300.0, ge=1.0)
    fallback_max_bid_ask_spread_pct: float = Field(default=0.25, ge=0.0, le=1.0)
    fallback_min_open_interest: int = Field(default=25, ge=0)
    fallback_strike_window_pct: float = Field(default=0.15, ge=0.0, le=1.0)
    fallback_contract_limit: int = Field(default=250, ge=1, le=500)
    max_spread_width: float = Field(default=5.0, ge=0.01)
    order_price_buffer_pct: float = Field(default=0.02, ge=0.0, le=0.25)
    profit_target_pct: float = Field(default=0.25, ge=0.01, le=5.0)
    stop_loss_pct: float = Field(default=0.35, ge=0.01, le=1.0)
    max_hold_minutes: int = Field(default=30, ge=1, le=390)
    close_at_entry_window_end: bool = True
    data_feed: str = "indicative"


class LLMSettings(BaseModel):
    provider: str = "google"  # "google", "openai", "anthropic", "grok"
    model: str = "gemini-3-flash-preview"
    market_analysis_interval_minutes: int = Field(default=15, ge=5, le=60)
    enable_trade_validation: bool = True


class Settings(BaseModel):
    trading: TradingSettings = TradingSettings()
    risk: RiskSettings = RiskSettings()
    options: OptionsSettings = OptionsSettings()
    llm: LLMSettings = LLMSettings()
    
    @classmethod
    def load(cls) -> "Settings":
        """Load from environment variables or config file."""
        # Load from .env - can be extended to load from config file
        return cls(
            trading=TradingSettings(
                watchlist=os.getenv("WATCHLIST", "SPY,QQQ,IWM,NVDA,TSLA").split(","),
                trading_window_start=os.getenv("TRADING_WINDOW_START", "09:30"),
                trading_window_end=os.getenv("TRADING_WINDOW_END", "12:00"),
                end_of_day_close_time=os.getenv("END_OF_DAY_CLOSE_TIME", "15:50"),
                max_concurrent_trades=int(os.getenv("MAX_CONCURRENT_TRADES", "1")),
                instrument=os.getenv("TRADING_INSTRUMENT", "options").lower(),
                allow_stock_fallback=os.getenv("ALLOW_STOCK_FALLBACK", "false").lower() == "true",
            ),
            risk=RiskSettings(
                max_risk_per_trade_percent=float(os.getenv("MAX_RISK_PER_TRADE_PERCENT", "1.0")),
                min_risk_reward_ratio=float(os.getenv("MIN_RISK_REWARD_RATIO", "1.5")),
                max_position_notional_pct=float(os.getenv("MAX_POSITION_NOTIONAL_PCT", "0.95")),
            ),
            options=OptionsSettings(
                paper_only=os.getenv("OPTIONS_PAPER_ONLY", "true").lower() == "true",
                strategy_type=os.getenv("OPTIONS_STRATEGY_TYPE", "single_long"),
                min_dte=int(os.getenv("OPTION_DTE_MIN", "7")),
                max_dte=int(os.getenv("OPTION_DTE_MAX", "14")),
                min_delta=float(os.getenv("OPTION_DELTA_MIN", "0.35")),
                max_delta=float(os.getenv("OPTION_DELTA_MAX", "0.55")),
                max_premium_per_trade=float(os.getenv("MAX_OPTION_PREMIUM_PER_TRADE", "200")),
                max_bid_ask_spread_pct=float(os.getenv("MAX_OPTION_BID_ASK_SPREAD_PCT", "0.15")),
                min_open_interest=int(os.getenv("MIN_OPTION_OPEN_INTEREST", "100")),
                strike_window_pct=float(os.getenv("OPTION_STRIKE_WINDOW_PCT", "0.10")),
                contract_limit=int(os.getenv("OPTION_CONTRACT_LIMIT", "100")),
                fallback_enabled=os.getenv("OPTION_FALLBACK_ENABLED", "false").lower() == "true",
                fallback_min_dte=int(os.getenv("OPTION_FALLBACK_DTE_MIN", "3")),
                fallback_max_dte=int(os.getenv("OPTION_FALLBACK_DTE_MAX", "21")),
                fallback_min_delta=float(os.getenv("OPTION_FALLBACK_DELTA_MIN", "0.25")),
                fallback_max_delta=float(os.getenv("OPTION_FALLBACK_DELTA_MAX", "0.65")),
                fallback_max_premium_per_trade=float(
                    os.getenv("OPTION_FALLBACK_MAX_PREMIUM_PER_TRADE", "300")
                ),
                fallback_max_bid_ask_spread_pct=float(
                    os.getenv("OPTION_FALLBACK_MAX_BID_ASK_SPREAD_PCT", "0.25")
                ),
                fallback_min_open_interest=int(os.getenv("OPTION_FALLBACK_MIN_OPEN_INTEREST", "25")),
                fallback_strike_window_pct=float(os.getenv("OPTION_FALLBACK_STRIKE_WINDOW_PCT", "0.15")),
                fallback_contract_limit=int(os.getenv("OPTION_FALLBACK_CONTRACT_LIMIT", "250")),
                max_spread_width=float(os.getenv("MAX_OPTION_SPREAD_WIDTH", "5.0")),
                order_price_buffer_pct=float(os.getenv("OPTION_ORDER_PRICE_BUFFER_PCT", "0.02")),
                profit_target_pct=float(os.getenv("OPTION_PROFIT_TARGET_PCT", "0.25")),
                stop_loss_pct=float(os.getenv("OPTION_STOP_LOSS_PCT", "0.35")),
                max_hold_minutes=int(os.getenv("OPTION_MAX_HOLD_MINUTES", "30")),
                close_at_entry_window_end=os.getenv("OPTION_CLOSE_AT_ENTRY_WINDOW_END", "true").lower() == "true",
                data_feed=os.getenv("OPTION_DATA_FEED", "indicative").lower(),
            ),
            llm=LLMSettings(
                provider=os.getenv("LLM_PROVIDER", "google"),
                model=os.getenv("LLM_MODEL", "gemini-3-flash-preview"),
                market_analysis_interval_minutes=int(os.getenv("MARKET_ANALYSIS_INTERVAL_MINUTES", "15")),
                enable_trade_validation=os.getenv("ENABLE_TRADE_VALIDATION", "true").lower() == "true",
            ),
        )
