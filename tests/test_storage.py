#!/usr/bin/env python3
"""Test script for database storage layer.

Tests SQLite storage adapter with sample data.
"""
import sys
from pathlib import Path
from datetime import date, datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.storage import Storage


def test_storage():
    """Test storage operations."""
    print("Testing SQLite storage...")
    
    # Create storage adapter
    storage = Storage.create(env="dev", db_path="data/test_trading.db")
    
    # Test daily bias
    test_date = date(2025, 1, 7)
    test_symbol = "SPY"
    
    print(f"\n1. Testing daily bias save/get...")
    bias_data = {
        "bias": "bullish",
        "confidence": 75,
        "model_output": {"prediction": "bullish", "score": 0.75},
        "news_summary": "Market shows strength",
        "premarket_price": 450.50
    }
    storage.save_daily_bias(test_date, test_symbol, bias_data)
    print(f"   [OK] Saved daily bias for {test_symbol}")
    
    retrieved = storage.get_daily_bias(test_date, test_symbol)
    assert retrieved is not None
    assert retrieved["bias"] == "bullish"
    assert retrieved["confidence"] == 75
    print(f"   [OK] Retrieved daily bias: {retrieved['bias']} ({retrieved['confidence']}%)")
    
    # Test premarket snapshot
    print(f"\n2. Testing premarket snapshot save/get...")
    snapshot_data = {
        "htf": {
            "ema_slope_daily": 0.5,
            "ema_slope_hourly": 0.3,
            "hh_ll_tag": "HH",
            "atr_percentile_daily": 65.0
        },
        "bands_5m": {
            "mu": 450.0,
            "sigma": 2.5,
            "atr_5m": 1.8,
            "k": {"k1": 1.2, "k2": 1.8, "k3": 0.6}
        }
    }
    storage.save_premarket_snapshot(test_date, test_symbol, snapshot_data)
    print(f"   [OK] Saved premarket snapshot for {test_symbol}")
    
    retrieved_snapshot = storage.get_premarket_snapshot(test_date, test_symbol)
    assert retrieved_snapshot is not None
    assert retrieved_snapshot["htf"]["ema_slope_daily"] == 0.5
    print(f"   [OK] Retrieved premarket snapshot")
    
    # Test market analysis
    print(f"\n3. Testing market analysis save...")
    analysis_data = {
        "timestamp": datetime.now(),
        "analysis_text": "Market conditions are favorable",
        "threshold_multipliers": {
            "mr_arm_multiplier": 0.9,
            "mr_trigger_multiplier": 1.0,
            "tc_arm_multiplier": 0.85,
            "tc_trigger_multiplier": 1.0
        },
        "confidence": 80,
        "llm_model": "gpt-4o-mini"
    }
    analysis_id = storage.save_market_analysis(analysis_data)
    print(f"   [OK] Saved market analysis (ID: {analysis_id})")
    
    # Test trade signal
    print(f"\n4. Testing trade signal save...")
    signal_data = {
        "timestamp": datetime.now(),
        "symbol": test_symbol,
        "setup_type": "MR",
        "side": "long",
        "entry_price": 450.25,
        "z_score": -0.8,
        "threshold_multipliers": {
            "mr_arm_z": 1.2,
            "mr_trigger_z": 0.6
        }
    }
    signal_id = storage.save_trade_signal(signal_data)
    print(f"   [OK] Saved trade signal (ID: {signal_id})")
    
    # Test LLM validation
    print(f"\n5. Testing LLM validation save...")
    validation_data = {
        "signal_id": signal_id,
        "timestamp": datetime.now(),
        "should_execute": True,
        "confidence": 75,
        "reasoning": "Good risk/reward ratio",
        "risk_assessment": "low",
        "llm_model": "gpt-4o-mini"
    }
    validation_id = storage.save_llm_validation(validation_data)
    print(f"   [OK] Saved LLM validation (ID: {validation_id})")
    
    # Test trade
    print(f"\n6. Testing trade save...")
    trade_data = {
        "trade_id": "alpaca_order_123",
        "symbol": test_symbol,
        "side": "long",
        "entry_price": 450.25,
        "stop_loss": 448.50,
        "take_profit": 453.50,
        "qty": 10,
        "status": "filled",
        "entry_time": datetime.now(),
        "exit_time": None,
        "exit_price": None,
        "pnl": None,
        "exit_reason": None
    }
    trade_id = storage.save_trade(trade_data)
    print(f"   [OK] Saved trade (ID: {trade_id})")
    
    # Test position
    print(f"\n7. Testing position update...")
    position_data = {
        "trade_id": trade_id,
        "symbol": test_symbol,
        "side": "long",
        "entry_price": 450.25,
        "current_price": 451.00,
        "stop_loss": 448.50,
        "take_profit": 453.50,
        "qty": 10,
        "unrealized_pnl": 7.50
    }
    storage.update_position(position_data)
    print(f"   [OK] Updated position")
    
    positions = storage.get_open_positions()
    assert len(positions) > 0
    print(f"   [OK] Retrieved {len(positions)} open position(s)")
    
    # Test trades query
    print(f"\n8. Testing trades query...")
    trades = storage.get_trades(symbol=test_symbol)
    assert len(trades) > 0
    print(f"   [OK] Retrieved {len(trades)} trade(s) for {test_symbol}")
    
    print("\n" + "="*60)
    print("All storage tests passed!")
    print("="*60)
    print(f"\nDatabase file: data/test_trading.db")
    print("You can inspect it with: sqlite3 data/test_trading.db")


if __name__ == "__main__":
    try:
        test_storage()
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

