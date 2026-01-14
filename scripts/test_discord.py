#!/usr/bin/env python3
"""Quick test for Discord notifications."""
import sys
from pathlib import Path
import os
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.utils.notifications import send_discord_alert, send_trade_alert

def test_notifications():
    load_dotenv()
    
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    
    if not webhook_url:
        print("❌ Error: DISCORD_WEBHOOK_URL not found in .env file.")
        print("Please add it to your .env file or export it in your shell.")
        return
    
    print("Sending test alerts to Discord...")
    
    # Test 1: Simple Alert
    send_discord_alert("🧪 **Test Alert**: Your bot is talking to Discord correctly!")
    print("✅ Sent simple alert.")
    
    # Test 2: Trade Embed
    send_trade_alert(
        symbol="TEST",
        side="long",
        price=123.45,
        setup="Verification"
    )
    print("✅ Sent trade embed test.")
    
    print("\nCheck your Discord channel! If you see a text message and a green box, you are ready.")

if __name__ == "__main__":
    test_notifications()
