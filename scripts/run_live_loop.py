#!/usr/bin/env python3
"""Live trading loop orchestrator script.

Wrapper script to run the live loop with proper configuration.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import and run main from live loop
from src.live.loop import main

if __name__ == "__main__":
    main()

