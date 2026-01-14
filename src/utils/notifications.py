"""Notification utility for trading alerts."""
import os
import requests
import json
from datetime import datetime
from typing import Optional

def send_discord_alert(message: str, webhook_url: Optional[str] = None):
    """Send alert message to Discord via webhook."""
    url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        return
    
    data = {
        "content": f"🔔 **LLM-Advisor Alert** | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{message}"
    }
    
    try:
        response = requests.post(url, json=data, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to send Discord alert: {e}")

def send_trade_alert(symbol: str, side: str, price: float, setup: str, webhook_url: Optional[str] = None):
    """Send trade execution alert to Discord."""
    url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        return
    
    color = 0x00ff00 if side.lower() == "long" else 0xff0000
    
    data = {
        "embeds": [{
            "title": f"🚀 Trade Executed: {symbol}",
            "color": color,
            "fields": [
                {"name": "Symbol", "value": symbol, "inline": True},
                {"name": "Side", "value": side.upper(), "inline": True},
                {"name": "Price", "value": f"${price:.2f}", "inline": True},
                {"name": "Setup", "value": setup, "inline": True},
            ],
            "timestamp": datetime.utcnow().isoformat()
        }]
    }
    
    try:
        response = requests.post(url, json=data, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to send trade alert: {e}")
