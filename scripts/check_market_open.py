#!/usr/bin/env python3
"""Market calendar check script.

Exits with code 0 if market is open today, code 1 if closed (holiday/weekend).
Used by GitHub Actions to skip runs on non-trading days.
"""
import sys
import pandas_market_calendars as mcal
from datetime import datetime
import pytz

def is_market_open():
    """Check if NYSE is open today."""
    nyse = mcal.get_calendar('NYSE')
    
    # Get current time in ET
    tz = pytz.timezone('America/New_York')
    now = datetime.now(tz)
    
    # Check if it's a weekend
    if now.weekday() >= 5:
        print(f"Today is {'Saturday' if now.weekday() == 5 else 'Sunday'}. Market is closed.")
        return False
        
    # Check NYSE calendar for today
    schedule = nyse.schedule(start_date=now.date(), end_date=now.date())
    
    if schedule.empty:
        print(f"Market is closed today ({now.date()}) per NYSE calendar (Holiday).")
        return False
        
    print(f"Market is open today ({now.date()}). Proceeding...")
    return True

if __name__ == "__main__":
    # If market is open, exit with 0, otherwise exit with 1
    if is_market_open():
        sys.exit(0)
    else:
        sys.exit(1)
