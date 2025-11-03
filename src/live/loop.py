"""Main live trading loop orchestrator.

Integrates all components:
- Premarket context loading
- Symbol state initialization
- Feature computation
- Threshold evaluation
- Periodic LLM market analysis
- Optional LLM trade validation
- Trade execution (stub)
"""
import sys
import os
import time
import argparse
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, List, Any
import pytz

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.core.config import Settings
from src.core.logging import setup_logging
from src.data.alpaca_client import AlpacaDataClient
from src.premarket.bias_gatherer import load_premarket_context
from src.premarket.snapshot_builder import SymbolSnapshot
from src.features.stdev_features import RollingStats
from src.live.state_manager import SymbolState
from src.live.feature_computer import compute_features
from src.live.threshold_evaluator import evaluate_thresholds, SignalEvent
from src.analysis.llm_client import create_llm_client
from src.analysis.market_analyzer import MarketAnalyzer
from src.analysis.trade_validator import validate_trade_with_llm
from config.thresholds import STDEVThresholds

ET = pytz.timezone("US/Eastern")
logger = setup_logging()


def parse_hhmm(s: str) -> tuple[int, int]:
    """Parse HH:MM string."""
    h, m = s.split(":")
    return int(h), int(m)


def et_dt(day: datetime.date, hhmm: str) -> datetime:
    """Create ET datetime from date and HH:MM."""
    h, m = parse_hhmm(hhmm)
    return ET.localize(datetime(day.year, day.month, day.day, h, m))


def to_utc(dt_et: datetime) -> datetime:
    """Convert ET datetime to UTC."""
    return dt_et.astimezone(timezone.utc)


def seed_states_from_snapshots(
    snapshots: Dict[str, Any],
    bars_1m: Dict[str, List[Dict[str, Any]]],
    thresholds: STDEVThresholds,
    window: int = 120
) -> Dict[str, SymbolState]:
    """
    Seed symbol states from premarket snapshots and historical bars.
    
    Args:
        snapshots: Premarket snapshots dict (from snapshot_builder)
        bars_1m: Historical 1-minute bars per symbol
        thresholds: Base thresholds
        window: Rolling window size
        
    Returns:
        Dict of symbol -> SymbolState
    """
    states = {}
    
    snapshot_list = snapshots.get("symbols", [])
    snapshot_by_symbol = {s["symbol"]: s for s in snapshot_list}
    
    for symbol, bars in bars_1m.items():
        if symbol not in snapshot_by_symbol:
            logger.warning(f"No premarket snapshot for {symbol}, skipping")
            continue
        
        snapshot = snapshot_by_symbol[symbol]
        htf = snapshot["htf"]
        bands_5m = snapshot["bands_5m"]
        
        # Extract closes for rolling stats
        closes = [bar["c"] for bar in bars[-window:]]
        if len(closes) < window:
            logger.warning(f"Insufficient seed data for {symbol} ({len(closes)}/{window} bars)")
            continue
        
        # Initialize rolling stats
        rolling = RollingStats.from_seed(closes, window=window)
        mu = rolling.mean
        sigma = rolling.std
        z = 0.0 if sigma == 0 else (closes[-1] - mu) / sigma
        
        # Create state
        state = SymbolState(
            symbol=symbol,
            rolling=rolling,
            htf_bias=htf["hh_ll_tag"],
            ema_slope_hourly=htf["ema_slope_hourly"],
            atr_percentile=htf["atr_percentile_daily"],
            atr_5m=bands_5m["atr_5m"],
            thresholds=thresholds,
            last_mu=mu,
            last_sigma=sigma,
            last_z=z,
        )
        
        states[symbol] = state
    
    return states


def should_run_market_analysis(
    analyzer: MarketAnalyzer,
    last_analysis: Optional[datetime],
    current_time: datetime
) -> bool:
    """Check if market analysis should run."""
    if last_analysis is None:
        return True
    
    interval = timedelta(minutes=analyzer.interval_minutes)
    return (current_time - last_analysis) >= interval


def market_is_open(current_et: datetime, start_time: str, end_time: str) -> bool:
    """Check if market is open."""
    start_dt = et_dt(current_et.date(), start_time)
    end_dt = et_dt(current_et.date(), end_time)
    return start_dt <= current_et <= end_dt


def log_tick(
    log_path: Path,
    states: Dict[str, SymbolState],
    signals: List[SignalEvent],
    extra: Dict[str, Any]
) -> None:
    """Log tick data."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "symbols": {
            sym: {
                "status": st.status,
                "side": st.side,
                "z": st.last_z,
                "mu": st.last_mu,
                "sigma": st.last_sigma,
                "trade": {
                    "setup": st.trade.setup,
                    "side": st.trade.side,
                    "entry": st.trade.entry_price,
                    "sl": st.trade.sl_price,
                    "tp": st.trade.tp_price,
                    "triggered_at": st.trade.triggered_at.isoformat(),
                } if st.trade else None,
            }
            for sym, st in states.items()
        },
        "signals": [
            {
                "symbol": sig.symbol,
                "setup_type": sig.setup_type,
                "side": sig.side,
                "entry_price": sig.entry_price,
                "z_score": sig.z_score,
                "timestamp": sig.timestamp.isoformat(),
            }
            for sig in signals
        ],
        **extra,
    }
    
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def execute_trade(signal: SignalEvent, state: SymbolState, order_manager: Optional[Any] = None) -> None:
    """
    Execute trade via order manager.
    
    Args:
        signal: Signal event
        state: Symbol state with trade plan
        order_manager: Optional StockOrderManager instance
    """
    logger.info(f"TRADE SIGNAL: {signal.symbol} {signal.setup_type} {signal.side} @ {signal.entry_price}")
    logger.info(f"  Stop Loss: {state.trade.sl_price if state.trade else 'N/A'}")
    logger.info(f"  Take Profit: {state.trade.tp_price if state.trade else 'N/A'}")
    
    if order_manager:
        try:
            from src.execution.order_manager import execute_trade_from_signal
            result = execute_trade_from_signal(signal, state, order_manager)
            if result:
                logger.info(f"Trade executed: Order ID {result.get('order_id')}")
            else:
                logger.warning(f"Trade execution failed for {signal.symbol}")
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
    else:
        logger.info("Order manager not initialized - trade not executed (dry run)")


def update_positions(trade_tracker: Optional[Any] = None) -> None:
    """Update open positions via trade tracker."""
    if trade_tracker:
        try:
            positions = trade_tracker.update_positions()
            if positions:
                total_pl = trade_tracker.calculate_total_unrealized_pl()
                logger.debug(f"Open positions: {len(positions)}, Total unrealized P/L: ${total_pl:.2f}")
        except Exception as e:
            logger.error(f"Position update error: {e}")


def main():
    """Main live loop."""
    parser = argparse.ArgumentParser(description="STDEV Live Trading Loop")
    parser.add_argument("--date", default=None, help="Trading date (YYYY-MM-DD), defaults to today")
    parser.add_argument("--symbols", nargs="*", default=None, help="Symbols to trade")
    parser.add_argument("--test", action="store_true", help="Test mode (use historical data)")
    parser.add_argument("--fast", type=int, default=60, help="Loop interval in seconds")
    parser.add_argument("--output", default=None, help="Output directory for logs")
    args = parser.parse_args()
    
    # Load settings
    settings = Settings.load()
    symbols = args.symbols or settings.trading.watchlist
    
    # Determine trading date
    if args.date:
        trading_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        trading_date = datetime.now(ET).date()
    
    date_str = trading_date.strftime("%Y-%m-%d")
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = PROJECT_ROOT / "data" / "daily_news" / date_str / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = output_dir / "live_loop_log.jsonl"
    
    logger.info(f"Starting live loop for {date_str}")
    logger.info(f"Symbols: {', '.join(symbols)}")
    
    # Load premarket context
    try:
        premarket_context = load_premarket_context(date_str, output_dir.parent)
        logger.info(f"Loaded premarket context for {len(premarket_context.symbols)} symbols")
    except FileNotFoundError:
        logger.error(f"Premarket context not found. Run premarket pipeline first.")
        return
    
    # Load premarket snapshots (from combined output)
    snapshot_path = output_dir / "premarket_context.json"
    if not snapshot_path.exists():
        logger.error(f"Premarket snapshots not found at {snapshot_path}")
        return
    
    with open(snapshot_path, 'r') as f:
        premarket_data = json.load(f)
    snapshots = premarket_data.get("snapshots", {})
    
    # Initialize Alpaca client
    try:
        alpaca_client = AlpacaDataClient()
    except Exception as e:
        logger.error(f"Failed to initialize Alpaca client: {e}")
        return
    
    # Initialize thresholds
    thresholds = STDEVThresholds(
        mr_arm_z=1.2,
        mr_trigger_z=0.6,
        tc_arm_z=1.8,
        tc_trigger_z=0.6,
        atr_multiplier_sl=1.4,
        min_rr_ratio=settings.risk.min_risk_reward_ratio,
        max_risk_per_trade=settings.risk.max_risk_per_trade_percent,
        atr_percentile_cap=85.0,
    )
    
    # Seed states with historical bars
    run_start_et = et_dt(trading_date, settings.trading.trading_window_start)
    run_start_utc = to_utc(run_start_et)
    seed_start_utc = run_start_utc - timedelta(minutes=240)  # 4 hours before
    
    logger.info(f"Fetching seed bars from {seed_start_utc} to {run_start_utc}")
    seed_bars = alpaca_client.fetch_window_bars(symbols, seed_start_utc, run_start_utc)
    
    states = seed_states_from_snapshots(snapshots, seed_bars, thresholds, window=120)
    logger.info(f"Initialized {len(states)} symbol states")
    
    # Initialize market analyzer
    llm_client = create_llm_client(settings.llm.provider, settings.llm.model)
    market_analyzer = MarketAnalyzer(llm_client, interval_minutes=settings.llm.market_analysis_interval_minutes)
    last_market_analysis: Optional[datetime] = None
    
    # Initialize order manager and trade tracker (optional - for paper/live trading)
    order_manager = None
    trade_tracker = None
    try:
        from src.execution.order_manager import StockOrderManager
        from src.execution.trade_tracker import TradeTracker
        
        paper_trading = os.getenv("ALPACA_PAPER_TRADING", "true").lower() == "true"
        order_manager = StockOrderManager(paper=paper_trading)
        trade_tracker = TradeTracker(order_manager)
        logger.info(f"Order manager initialized (paper trading: {paper_trading})")
    except Exception as e:
        logger.warning(f"Order manager initialization failed: {e}. Running in dry-run mode.")
    
    # Trading window
    run_end_et = et_dt(trading_date, settings.trading.trading_window_end)
    run_end_utc = to_utc(run_end_et)
    
    logger.info(f"Trading window: {run_start_et.strftime('%H:%M')} - {run_end_et.strftime('%H:%M')} ET")
    
    # Main loop
    loop_count = 0
    while True:
        current_et = datetime.now(ET)
        current_utc = current_et.astimezone(timezone.utc)
        
        # Check if before session start
        if current_et < run_start_et:
            wait_s = max(1, int((run_start_et - current_et).total_seconds()))
            logger.info(f"Before session start. Waiting {wait_s}s...")
            time.sleep(min(wait_s, args.fast))
            continue
        
        # Check if after session end
        if current_et > run_end_et:
            logger.info("Trading session complete")
            break
        
        # Check if market is open
        if not market_is_open(current_et, settings.trading.trading_window_start, settings.trading.trading_window_end):
            time.sleep(args.fast)
            continue
        
        loop_count += 1
        logger.debug(f"Loop iteration {loop_count} at {current_et.strftime('%H:%M:%S')} ET")
        
        try:
            # Fetch latest bars (last 30 minutes)
            fetch_start_utc = current_utc - timedelta(minutes=30)
            bars = alpaca_client.fetch_window_bars(symbols, fetch_start_utc, current_utc)
            
            # Update features for each symbol
            for symbol in symbols:
                if symbol not in states:
                    continue
                
                symbol_bars = bars.get(symbol, {})
                bars_1m = symbol_bars.get("bars_1m", [])
                
                if not bars_1m:
                    continue
                
                latest_price = bars_1m[-1]["c"]
                state = states[symbol]
                
                # Compute features
                features = compute_features(symbol, latest_price, state, bars_1m)
                
                # Update state with features
                state.update_features(features.mu, features.sigma, features.z_score, current_utc)
            
            # Check if market analysis should run
            if should_run_market_analysis(market_analyzer, last_market_analysis, current_utc):
                logger.info("Running periodic market analysis...")
                try:
                    # Build recent price action summary
                    recent_price_action = {
                        sym: {
                            "price_change_pct": ((bars.get(sym, {}).get("bars_1m", [{}])[-1].get("c", 0) / 
                                                  bars.get(sym, {}).get("bars_1m", [{}])[0].get("c", 1)) - 1) * 100
                            if len(bars.get(sym, {}).get("bars_1m", [])) > 1 else 0.0
                        }
                        for sym in symbols
                    }
                    
                    multiplier = market_analyzer.analyze_market(
                        states=states,
                        premarket_context=premarket_context,
                        recent_price_action=recent_price_action
                    )
                    
                    # Apply multiplier to all symbol states
                    for symbol in symbols:
                        if symbol in states:
                            states[symbol].threshold_multiplier = multiplier
                    
                    logger.info(f"Applied threshold multipliers: MR={multiplier.mr_arm_multiplier:.2f}, "
                              f"TC={multiplier.tc_arm_multiplier:.2f}, confidence={multiplier.confidence}")
                    last_market_analysis = current_utc
                except Exception as e:
                    logger.error(f"Market analysis failed: {e}")
            
            # Evaluate thresholds and collect signals
            signals: List[SignalEvent] = []
            for symbol in symbols:
                if symbol not in states:
                    continue
                
                symbol_bars = bars.get(symbol, {})
                bars_1m = symbol_bars.get("bars_1m", [])
                
                if not bars_1m:
                    continue
                
                latest_price = bars_1m[-1]["c"]
                state = states[symbol]
                
                # Evaluate thresholds
                signal = evaluate_thresholds(
                    state=state,
                    current_price=latest_price,
                    multiplier=state.threshold_multiplier
                )
                
                if signal:
                    # Optional LLM trade validation
                    if settings.llm.enable_trade_validation:
                        try:
                            validation = validate_trade_with_llm(
                                signal=signal,
                                state=state,
                                premarket_context=premarket_context,
                                llm_client=llm_client
                            )
                            
                            if not validation.should_execute:
                                logger.info(f"LLM validation rejected trade for {symbol}: {validation.reasoning}")
                                continue
                        except Exception as e:
                            logger.error(f"Trade validation failed: {e}, proceeding with trade")
                    
                    signals.append(signal)
                    execute_trade(signal, state, order_manager)
            
            # Monitor positions
            update_positions(trade_tracker)
            
            # Log tick
            log_tick(
                log_path=log_path,
                states=states,
                signals=signals,
                extra={"label": current_et.strftime("%H:%M:%S ET"), "loop_count": loop_count}
            )
            
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
        
        # Sleep until next iteration
        time.sleep(args.fast)


if __name__ == "__main__":
    main()

