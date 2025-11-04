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
from src.premarket.bias_gatherer import load_premarket_context, PremarketContext
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


def _create_minimal_snapshot_from_bars(
    symbol: str,
    bars_1m: List[Dict[str, Any]],
    bars_5m: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Create minimal snapshot from bars for backtesting when premarket data isn't available.
    Uses defaults for HTF stats that can't be computed from limited historical bars.
    """
    if not bars_1m:
        raise ValueError(f"No bars provided for {symbol}")
    
    # Compute 5m bands from available bars
    if bars_5m:
        import pandas as pd
        df_5m = pd.DataFrame(bars_5m)
        if not df_5m.empty and "c" in df_5m.columns:
            closes_5m = df_5m["c"].dropna()
            if len(closes_5m) >= 20:
                mu_5m = closes_5m.tail(120).mean() if len(closes_5m) >= 120 else closes_5m.mean()
                sigma_5m = closes_5m.tail(120).std() if len(closes_5m) >= 120 else closes_5m.std()
                
                # Simple ATR from 5m bars
                if len(df_5m) > 1:
                    atr_5m = ((df_5m["h"] - df_5m["l"]).abs().mean())
                else:
                    atr_5m = abs(df_5m.iloc[-1]["h"] - df_5m.iloc[-1]["l"])
            else:
                mu_5m = bars_1m[-1]["c"]
                sigma_5m = 0.01 * mu_5m  # Default 1% volatility
                atr_5m = abs(bars_1m[-1]["h"] - bars_1m[-1]["l"]) if len(bars_1m) > 0 else 0.01 * mu_5m
        else:
            mu_5m = bars_1m[-1]["c"]
            sigma_5m = 0.01 * mu_5m
            atr_5m = abs(bars_1m[-1]["h"] - bars_1m[-1]["l"]) if len(bars_1m) > 0 else 0.01 * mu_5m
    else:
        mu_5m = bars_1m[-1]["c"]
        sigma_5m = 0.01 * mu_5m
        atr_5m = abs(bars_1m[-1]["h"] - bars_1m[-1]["l"]) if len(bars_1m) > 0 else 0.01 * mu_5m
    
    # Default HTF stats (can't compute from limited bars)
    return {
        "symbol": symbol,
        "htf": {
            "ema_slope_daily": 0.0,
            "ema_slope_hourly": 0.0,
            "hh_ll_tag": "neutral",  # Default to neutral
            "atr_percentile_daily": 50.0,  # Default to median
        },
        "bands_5m": {
            "mu": float(mu_5m),
            "sigma": float(sigma_5m),
            "atr_5m": float(atr_5m),
            "k": {"k1": 1.2, "k2": 1.8, "k3": 0.6},
        },
    }


def seed_states_from_snapshots(
    snapshots: Optional[Dict[str, Any]],
    bars_1m: Dict[str, List[Dict[str, Any]]],
    thresholds: STDEVThresholds,
    window: int = 120,
    bars_5m: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    create_minimal_snapshots: bool = False
) -> Dict[str, SymbolState]:
    """
    Seed symbol states from premarket snapshots and historical bars.
    
    Args:
        snapshots: Premarket snapshots dict (from snapshot_builder), or None
        bars_1m: Historical 1-minute bars per symbol
        thresholds: Base thresholds
        window: Rolling window size
        bars_5m: Optional 5-minute bars (for backtesting without premarket)
        create_minimal_snapshots: If True and snapshots is None, create minimal snapshots from bars
        
    Returns:
        Dict of symbol -> SymbolState
    """
    states = {}
    
    # If no snapshots and create_minimal_snapshots is True, create them from bars
    if snapshots is None or not snapshots.get("symbols"):
        if create_minimal_snapshots:
            logger.info("No premarket snapshots found, creating minimal snapshots from bars for backtesting")
            snapshot_list = []
            for symbol in bars_1m.keys():
                try:
                    snapshot = _create_minimal_snapshot_from_bars(
                        symbol,
                        bars_1m[symbol],
                        bars_5m.get(symbol) if bars_5m else None
                    )
                    snapshot_list.append(snapshot)
                except Exception as e:
                    logger.warning(f"Failed to create minimal snapshot for {symbol}: {e}")
                    continue
            snapshots = {"symbols": snapshot_list}
        else:
            logger.warning("No snapshots provided and create_minimal_snapshots=False")
            return states
    
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
        
        # Require minimum bars for seed (relaxed for backtests)
        min_bars_required = min(window, max(30, len(bars)))  # At least 30 bars, or whatever we have
        
        if len(closes) < min_bars_required:
            logger.warning(f"Insufficient seed data for {symbol} ({len(closes)}/{min_bars_required} bars, need at least 30)")
            continue
        
        if len(closes) < window:
            logger.info(f"Using {len(closes)} bars for {symbol} seed (target: {window}), RollingStats will adjust")
        
        # Initialize rolling stats (will use available bars, window sets max)
        rolling = RollingStats.from_seed(closes, window=window)
        mu = rolling.mean
        sigma = rolling.std
        z = 0.0 if sigma == 0 else (closes[-1] - mu) / sigma
        
        # Create state
        state = SymbolState(
            symbol=symbol,
            rolling=rolling,
            htf_bias=htf.get("hh_ll_tag", "neutral"),
            ema_slope_hourly=htf.get("ema_slope_hourly", 0.0),
            atr_percentile=htf.get("atr_percentile_daily", 50.0),
            atr_5m=bands_5m.get("atr_5m", abs(bars[-1]["h"] - bars[-1]["l"])),
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
    
    # Detect backtest mode
    is_backtest = args.test or (args.date and trading_date < datetime.now(ET).date())
    
    # Load premarket context (optional for backtests)
    premarket_context: Optional[PremarketContext] = None
    try:
        premarket_context = load_premarket_context(date_str, output_dir.parent)
        logger.info(f"Loaded premarket context for {len(premarket_context.symbols)} symbols")
    except FileNotFoundError:
        if is_backtest:
            logger.info("Premarket context not found - running pure technical backtest (no bias/news)")
        else:
            logger.error(f"Premarket context not found. Run premarket pipeline first.")
            return
    
    # Load premarket snapshots (optional for backtests)
    snapshots: Optional[Dict[str, Any]] = None
    snapshot_path = output_dir / "premarket_context.json"
    if snapshot_path.exists():
        with open(snapshot_path, 'r') as f:
            premarket_data = json.load(f)
        snapshots = premarket_data.get("snapshots", {})
        if snapshots:
            logger.info(f"Loaded premarket snapshots for {len(snapshots.get('symbols', []))} symbols")
    elif not is_backtest:
        logger.error(f"Premarket snapshots not found at {snapshot_path}")
        return
    else:
        logger.info("Premarket snapshots not found - will create minimal snapshots from bars")
    
    # Initialize data client (real or mock)
    if is_backtest:
        logger.info("Running in BACKTEST mode - using historical data")
        
        # Calculate seed window to ensure we have enough data
        run_start_et = et_dt(trading_date, settings.trading.trading_window_start)
        run_start_utc = to_utc(run_start_et)
        
        # For backtest seed data, we need hours of 1m bars before trading start
        # Try to fetch from previous day's market open, or at least 4 hours before today's open
        # Previous day calculation
        prev_day = trading_date
        days_back = 1
        while days_back < 5:  # Try up to 5 days back
            prev_day_et = prev_day - timedelta(days=days_back)
            # Skip weekends (Saturday=5, Sunday=6)
            if prev_day_et.weekday() < 5:  # Monday=0, Friday=4
                break
            days_back += 1
        
        # Fetch from previous trading day's market open (09:30 ET)
        seed_start_et = et_dt(prev_day_et, "09:30")
        seed_start_utc = to_utc(seed_start_et)
        
        # If seed start is too far back (more than 2 days), use 4 hours before trading start instead
        if (run_start_utc - seed_start_utc).total_seconds() > 2 * 24 * 3600:
            seed_start_utc = run_start_utc - timedelta(minutes=240)  # 4 hours before
        
        # Prefetch all bars from seed start to end of day
        eod_et = et_dt(trading_date, "16:00")
        eod_utc = to_utc(eod_et)
        
        # Fetch historical data using real client (from seed start to EOD)
        try:
            real_client = AlpacaDataClient()
            logger.info(f"Fetching historical bars from {seed_start_utc} to {eod_utc} (seed window from prev day if needed)")
            all_bars = real_client.fetch_window_bars(symbols, seed_start_utc, eod_utc)
            
            # Create mock client
            from src.data.mock_data_client import MockDataClient
            alpaca_client = MockDataClient(all_bars)
            logger.info("Mock data client initialized")
        except Exception as e:
            logger.error(f"Failed to fetch historical data for backtest: {e}")
            return
    else:
        # Use real client for live trading
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
    # Note: run_start_et and seed_start_utc already calculated above for backtest mode
    if not is_backtest:
        run_start_et = et_dt(trading_date, settings.trading.trading_window_start)
        run_start_utc = to_utc(run_start_et)
        seed_start_utc = run_start_utc - timedelta(minutes=240)  # 4 hours before
    
    logger.info(f"Fetching seed bars from {seed_start_utc} to {run_start_utc}")
    seed_bars_1m = alpaca_client.fetch_window_bars(symbols, seed_start_utc, run_start_utc)
    # Extract 1m bars
    bars_1m_dict = {sym: seed_bars_1m.get(sym, {}).get("bars_1m", []) for sym in symbols}
    # Extract 5m bars (for minimal snapshot creation if needed)
    bars_5m_dict = {sym: seed_bars_1m.get(sym, {}).get("bars_5m", []) for sym in symbols}
    
    # Set simulated time for mock client
    if is_backtest and hasattr(alpaca_client, 'set_current_time'):
        alpaca_client.set_current_time(run_start_utc)
    
    states = seed_states_from_snapshots(
        snapshots=snapshots,
        bars_1m=bars_1m_dict,
        thresholds=thresholds,
        window=120,
        bars_5m=bars_5m_dict if is_backtest else None,
        create_minimal_snapshots=is_backtest  # Create minimal snapshots if backtesting and no premarket data
    )
    logger.info(f"Initialized {len(states)} symbol states")
    
    # Initialize market analyzer
    llm_client = create_llm_client(settings.llm.provider, settings.llm.model)
    market_analyzer = MarketAnalyzer(llm_client, interval_minutes=settings.llm.market_analysis_interval_minutes)
    last_market_analysis: Optional[datetime] = None
    
    # Initialize order manager and trade tracker
    order_manager = None
    trade_tracker = None
    
    if is_backtest:
        # Use mock order manager for backtesting
        from src.execution.mock_order_manager import MockOrderManager
        order_manager = MockOrderManager(
            initial_equity=100000.0,
            eod_close_time=settings.trading.end_of_day_close_time
        )
        logger.info("Mock order manager initialized for backtesting")
    else:
        # Use real order manager for live trading
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
    eod_close_et = et_dt(trading_date, settings.trading.end_of_day_close_time)
    eod_close_utc = to_utc(eod_close_et)
    
    logger.info(f"Trading window: {run_start_et.strftime('%H:%M')} - {run_end_et.strftime('%H:%M')} ET")
    
    # Simulated time for backtesting
    sim_time_et = run_start_et if is_backtest else None
    
    # Main loop
    loop_count = 0
    while True:
        if is_backtest:
            # Simulate time advancement in backtest mode
            if sim_time_et is None or sim_time_et > eod_close_et:
                # End backtest - close all positions
                if order_manager and hasattr(order_manager, 'close_all_positions_at_eod'):
                    # Fetch final prices
                    final_bars = alpaca_client.fetch_window_bars(symbols, eod_close_utc - timedelta(minutes=5), eod_close_utc)
                    current_prices = {
                        sym: final_bars.get(sym, {}).get("bars_1m", [{}])[-1].get("c", final_bars.get(sym, {}).get("bars_1m", [{}])[0].get("c", 0))
                        for sym in symbols
                        if final_bars.get(sym, {}).get("bars_1m")
                    }
                    order_manager.close_all_positions_at_eod(eod_close_utc, current_prices)
                    
                    # Generate backtest summary
                    if hasattr(order_manager, 'get_trade_summary'):
                        summary = order_manager.get_trade_summary()
                        summary["date"] = date_str
                        
                        # Save backtest results
                        backtest_path = output_dir / "backtest_results.json"
                        with open(backtest_path, 'w') as f:
                            json.dump(summary, f, indent=2, default=str)
                        logger.info(f"Backtest complete. Results saved to {backtest_path}")
                        logger.info(f"Total trades: {summary['total_trades']}, P/L: ${summary['total_pnl']:.2f}")
                break
            
            current_et = sim_time_et
            current_utc = to_utc(current_et)
            
            # Update mock client time
            if hasattr(alpaca_client, 'set_current_time'):
                alpaca_client.set_current_time(current_utc)
        else:
            # Real-time mode
            current_et = datetime.now(ET)
            current_utc = current_et.astimezone(timezone.utc)
            
            # Check EOD close for live trading
            if order_manager and trade_tracker:
                from src.execution.eod_position_manager import check_and_close_eod
                if check_and_close_eod(
                    order_manager,
                    trade_tracker,
                    current_et,
                    settings.trading.end_of_day_close_time
                ):
                    logger.info("End-of-day positions closed. Exiting.")
                    break
        
        # Check if before session start (skip in backtest)
        if not is_backtest and current_et < run_start_et:
            wait_s = max(1, int((run_start_et - current_et).total_seconds()))
            logger.info(f"Before session start. Waiting {wait_s}s...")
            time.sleep(min(wait_s, args.fast))
            continue
        
        # Check if after session end (skip in backtest, we handle it above)
        if not is_backtest and current_et > run_end_et:
            logger.info("Trading session complete")
            break
        
        # Check if market is open (skip time check in backtest)
        if not is_backtest and not market_is_open(current_et, settings.trading.trading_window_start, settings.trading.trading_window_end):
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
            
            # Check if market analysis should run (skip if no premarket context for backtests)
            if premarket_context and should_run_market_analysis(market_analyzer, last_market_analysis, current_utc):
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
            elif not premarket_context and is_backtest:
                # In backtest mode without premarket, use neutral multipliers
                logger.debug("Skipping market analysis (no premarket context in backtest mode)")
            
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
                    # Optional LLM trade validation (skip if no premarket context in backtest)
                    if settings.llm.enable_trade_validation and premarket_context:
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
                    elif settings.llm.enable_trade_validation and not premarket_context:
                        logger.debug(f"LLM trade validation skipped (no premarket context for {symbol})")
                    
                    signals.append(signal)
                    # Use appropriate execute function based on mode
                    if is_backtest and hasattr(order_manager, 'execute_stock_trade'):
                        # Use mock manager's execute function directly
                        from src.execution.mock_order_manager import execute_trade_from_signal
                        execute_trade_from_signal(signal, state, order_manager)
                    else:
                        execute_trade(signal, state, order_manager)
            
            # Check for position exits (stop loss/take profit) - for mock manager
            if is_backtest and order_manager and hasattr(order_manager, 'check_exits'):
                current_prices = {
                    sym: bars.get(sym, {}).get("bars_1m", [{}])[-1].get("c", bars.get(sym, {}).get("bars_1m", [{}])[0].get("c", 0))
                    for sym in symbols
                    if bars.get(sym, {}).get("bars_1m")
                }
                order_manager.check_exits(current_prices, current_utc)
            
            # Monitor positions
            if not is_backtest:
                update_positions(trade_tracker)
            
            # Log tick
            label_time = sim_time_et if is_backtest else current_et
            log_tick(
                log_path=log_path,
                states=states,
                signals=signals,
                extra={"label": label_time.strftime("%H:%M:%S ET"), "loop_count": loop_count, "backtest": is_backtest}
            )
            
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
        
        # Advance simulated time or sleep
        if is_backtest:
            # Advance by --fast seconds (converted to simulated minutes)
            sim_time_et += timedelta(seconds=args.fast)
            # Don't sleep in backtest mode - run as fast as possible
        else:
            # Sleep until next iteration (real-time mode)
            time.sleep(args.fast)


if __name__ == "__main__":
    main()

