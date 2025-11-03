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
        # Load from .env - can be extended to load from config file
        return cls(
            trading=TradingSettings(
                watchlist=os.getenv("WATCHLIST", "SPY,QQQ,IWM,NVDA,TSLA").split(","),
                trading_window_start=os.getenv("TRADING_WINDOW_START", "09:30"),
                trading_window_end=os.getenv("TRADING_WINDOW_END", "12:00"),
                end_of_day_close_time=os.getenv("END_OF_DAY_CLOSE_TIME", "15:50"),
            ),
            risk=RiskSettings(
                max_risk_per_trade_percent=float(os.getenv("MAX_RISK_PER_TRADE_PERCENT", "1.0")),
                min_risk_reward_ratio=float(os.getenv("MIN_RISK_REWARD_RATIO", "1.5")),
            ),
            llm=LLMSettings(
                provider=os.getenv("LLM_PROVIDER", "openai"),
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                market_analysis_interval_minutes=int(os.getenv("MARKET_ANALYSIS_INTERVAL_MINUTES", "15")),
                enable_trade_validation=os.getenv("ENABLE_TRADE_VALIDATION", "true").lower() == "true",
            ),
        )

