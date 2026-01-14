#!/usr/bin/env python3
"""Test script for BigQuery storage adapter.

Tests BigQuery connection, table creation, and basic CRUD operations.
"""
import sys
from pathlib import Path
from datetime import date, datetime
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.storage import Storage


def test_bigquery():
    """Test BigQuery storage operations."""
    print("=" * 60)
    print("BigQuery Storage Adapter Test")
    print("=" * 60)
    
    # Configuration
    project_id = "gen-lang-client-0189185649"
    dataset_id = "trading_signals"
    credentials_path = "gen-lang-client-0189185649-c545abd919a4.json"
    
    # Check if credentials file exists
    if not os.path.exists(credentials_path):
        print(f"\n[ERROR] Credentials file not found: {credentials_path}")
        print("Please ensure the service account JSON file is in the project root.")
        return False
    
    try:
        print(f"\n1. Initializing BigQuery storage...")
        print(f"   Project ID: {project_id}")
        print(f"   Dataset ID: {dataset_id}")
        print(f"   Credentials: {credentials_path}")
        
        # Create storage adapter (this will create dataset and tables)
        storage = Storage.create(
            env="bq",
            project_id=project_id,
            dataset_id=dataset_id,
            credentials_path=credentials_path
        )
        print(f"   [OK] BigQuery storage initialized successfully")
        print(f"   [OK] Dataset and tables created/verified")
        
        # Test daily bias
        test_date = date.today()
        test_symbol = "SPY"
        
        print(f"\n2. Testing daily bias save/get...")
        bias_data = {
            "bias": "bullish",
            "confidence": 75,
            "model_output": {"prediction": "bullish", "score": 0.75},
            "news_summary": "Market shows strength",
            "premarket_price": 450.50
        }
        storage.save_daily_bias(test_date, test_symbol, bias_data)
        print(f"   [OK] Saved daily bias for {test_symbol} on {test_date}")
        
        retrieved = storage.get_daily_bias(test_date, test_symbol)
        assert retrieved is not None
        assert retrieved["bias"] == "bullish"
        assert retrieved["confidence"] == 75
        print(f"   [OK] Retrieved daily bias: {retrieved['bias']} ({retrieved['confidence']}%)")
        
        # Test premarket snapshot
        print(f"\n3. Testing premarket snapshot save/get...")
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
        print(f"\n4. Testing market analysis save...")
        analysis_data = {
            "timestamp": datetime.now(),
            "analysis_text": "Market conditions are favorable",
            "threshold_multipliers": {"mr_arm": 1.2, "tc_arm": 1.5},
            "confidence": 80,
            "llm_model": "gpt-4"
        }
        analysis_id = storage.save_market_analysis(analysis_data)
        print(f"   [OK] Saved market analysis (ID: {analysis_id})")
        
        # Test live loop log
        print(f"\n5. Testing live loop log save...")
        log_entry = {
            "timestamp": datetime.now(),
            "symbol": test_symbol,
            "z_score": 2.5,
            "mu": 450.0,
            "sigma": 2.0,
            "status": "mr_armed",
            "side": "long",
            "current_price": 452.5,
            "atr_percentile": 70.0,
            "ema_slope_hourly": 0.4
        }
        storage.save_live_loop_log(log_entry)
        print(f"   [OK] Saved live loop log entry")
        
        # Test trade signal
        print(f"\n6. Testing trade signal save...")
        signal_data = {
            "timestamp": datetime.now(),
            "symbol": test_symbol,
            "setup_type": "MR",
            "side": "long",
            "entry_price": 452.0,
            "z_score": 2.5,
            "threshold_multipliers": {"mr_arm": 1.2}
        }
        signal_id = storage.save_trade_signal(signal_data)
        print(f"   [OK] Saved trade signal (ID: {signal_id})")
        
        # Test LLM validation
        print(f"\n7. Testing LLM validation save...")
        validation_data = {
            "signal_id": signal_id,
            "timestamp": datetime.now(),
            "should_execute": True,
            "confidence": 85,
            "reasoning": "Strong setup with good risk/reward",
            "risk_assessment": "low",
            "llm_model": "gpt-4"
        }
        validation_id = storage.save_llm_validation(validation_data)
        print(f"   [OK] Saved LLM validation (ID: {validation_id})")
        
        # Test trade
        print(f"\n8. Testing trade save...")
        trade_data = {
            "trade_id": "test-trade-001",
            "symbol": test_symbol,
            "side": "long",
            "entry_price": 452.0,
            "stop_loss": 450.0,
            "take_profit": 456.0,
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
        print(f"\n9. Testing position update...")
        position_data = {
            "trade_id": trade_id,
            "symbol": test_symbol,
            "side": "long",
            "entry_price": 452.0,
            "current_price": 453.5,
            "stop_loss": 450.0,
            "take_profit": 456.0,
            "qty": 10,
            "unrealized_pnl": 15.0
        }
        storage.update_position(position_data)
        print(f"   [OK] Updated position")
        
        # Test get open positions
        print(f"\n10. Testing get open positions...")
        positions = storage.get_open_positions()
        print(f"   [OK] Retrieved {len(positions)} open position(s)")
        
        # Test get trades
        print(f"\n11. Testing get trades...")
        trades = storage.get_trades(symbol=test_symbol)
        print(f"   [OK] Retrieved {len(trades)} trade(s) for {test_symbol}")
        
        print("\n" + "=" * 60)
        print("✅ All BigQuery tests passed!")
        print("=" * 60)
        print(f"\nYou can now view your data in BigQuery Console:")
        print(f"https://console.cloud.google.com/bigquery?project={project_id}")
        print(f"\nDataset: {dataset_id}")
        print("Tables: daily_bias, premarket_snapshots, market_analysis, etc.")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_bigquery()
    sys.exit(0 if success else 1)
