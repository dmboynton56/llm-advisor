#!/usr/bin/env python3
"""
Handles the execution of trades based on signals from the live analyzer.
This module translates a signal on an underlying stock into a specific
options trade and places a bracket order via the Alpaca API.
"""

import os
import sys
from datetime import datetime, timedelta
import pytz

# Add project root to path for local imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config.settings import MAX_RISK_PER_TRADE_PERCENT, MINIMUM_RISK_REWARD_RATIO, OPTIONS_EXPIRATION_DAYS, STRIKE_PRICE_LOGIC
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, BracketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.datav2.options import OptionDataClient

ET_TZ = pytz.timezone("US/Eastern")

def _find_options_contract(symbol, clients, strike_logic="ATM", exp_days=0):
    """Finds a specific, tradable options contract based on the desired logic."""
    data_client = OptionDataClient(api_key=os.getenv("ALPACA_API_KEY"), secret_key=os.getenv("ALPACA_SECRET_KEY"))
    
    # 1. Get the current price of the underlying
    try:
        latest_quote = data_client.get_latest_quote(symbol)
        current_price = (latest_quote.ask_price + latest_quote.bid_price) / 2
    except Exception as e:
        print(f"  ! ERROR: Could not get latest quote for {symbol}: {e}")
        return None

    # 2. Determine expiration date
    exp_date = (datetime.now(ET_TZ) + timedelta(days=exp_days)).strftime('%Y-%m-%d')

    # 3. Find the At-the-Money (ATM) strike price
    if strike_logic == "ATM":
        # Round the current price to the nearest valid strike
        target_strike = round(current_price)
    else:
        # Future logic could allow for ITM/OTM strikes
        target_strike = round(current_price)

    # 4. Fetch the options chain to find the specific contract
    try:
        chain = data_client.get_snapshots(symbol, feed='opra')[symbol]
        
        # Filter for the right expiration and strike
        best_contract = None
        min_strike_diff = float('inf')

        for contract_symbol, snapshot in chain.items():
            if snapshot.expiration_date.strftime('%Y-%m-%d') == exp_date and snapshot.strike_price == target_strike:
                 # Found an exact match
                 best_contract = snapshot
                 break
        
        if best_contract:
            print(f"  > Found contract: {best_contract.symbol} (Strike: {best_contract.strike_price}, Exp: {best_contract.expiration_date.date()})")
            return best_contract.symbol
        else:
            print(f"  ! ERROR: No contract found for {symbol} with strike {target_strike} and expiration {exp_date}")
            return None

    except Exception as e:
        print(f"  ! ERROR: Could not fetch options chain for {symbol}: {e}")
        return None

def execute_trade_signal(signal_packet, clients):
    """
    Parses a signal from the analyzer, calculates risk, and places a bracket order.
    """
    trading_client = clients.get('trading')
    if not trading_client:
        print("  ! ERROR: Trading client not initialized.")
        return

    symbol = signal_packet['symbol']
    side = OrderSide.BUY if signal_packet['trade_signal'] == 'long' else OrderSide.SELL
    params = signal_packet['trade_parameters']
    
    stop_loss_price = params['stop_loss']
    take_profit_price = params['take_profit']
    entry_price = params['entry_price_target']

    # 1. Find the specific options contract to trade
    option_symbol = _find_options_contract(symbol, clients, STRIKE_PRICE_LOGIC, OPTIONS_EXPIRATION_DAYS)
    if not option_symbol:
        print(f"  ! Halting trade execution for {symbol}, could not find a valid contract.")
        return

    # 2. Calculate position size based on risk
    try:
        account = trading_client.get_account()
        equity = float(account.equity)
        risk_per_share_on_underlying = abs(entry_price - stop_loss_price)
        
        if risk_per_share_on_underlying == 0:
            print("  ! ERROR: Risk per share is zero, cannot calculate position size.")
            return

        # Note: Options contracts are for 100 shares of the underlying
        risk_per_contract = risk_per_share_on_underlying * 100
        total_risk_amount = equity * (MAX_RISK_PER_TRADE_PERCENT / 100)
        
        qty = int(total_risk_amount // risk_per_contract)
        if qty == 0:
            print(f"  ! Position size is zero for {symbol}. Increase risk or trade a smaller underlying. Halting trade.")
            return

    except Exception as e:
        print(f"  ! ERROR during risk calculation: {e}")
        return

    # 3. Construct the Bracket Order
    # This order buys the option and simultaneously sets SL/TP based on the UNDERLYING's price
    order_data = BracketOrderRequest(
        symbol=option_symbol,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(
            limit_price=take_profit_price  # This should be the price of the option, need to calculate this. Placeholder for now.
            # For now, we will manage this manually or in a separate loop
        ),
        stop_loss=StopLossRequest(
            stop_price=stop_loss_price, # The price of the underlying to trigger the stop
            # limit_price can be added for a Stop Limit order
        )
    )

    # 4. Submit the Order
    try:
        print(f"  > Submitting {side.value} order for {qty} contract(s) of {option_symbol}...")
        bracket_order = trading_client.submit_order(order_data=order_data)
        print("  --- TRADE EXECUTED SUCCESSFULLY ---")
        print(f"    Order ID: {bracket_order.id}")
        print(f"    Symbol: {bracket_order.symbol}")
        print(f"    Qty: {bracket_order.qty}")
        print(f"    Status: {bracket_order.status}")
        return bracket_order

    except Exception as e:
        print(f"  --- TRADE FAILED TO EXECUTE ---")
        print(f"  ! ERROR: {e}")
        return None