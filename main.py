"""
DEPRECATED: This is the old premarket pipeline orchestrator.

For the new STDEV trading system, use:
    python scripts/run_premarket.py --date YYYY-MM-DD --symbols SPY QQQ

This file is kept for backward compatibility with the legacy ICT strategy workflow.
"""
import sys
import os
import subprocess
import time  # Import the time module
from datetime import datetime
import pytz

def run_script(script_path):
    """Executes a python script using the same interpreter and raises an error if it fails."""
    interpreter_path = sys.executable
    print(f"  > Running command: {os.path.basename(interpreter_path)} {os.path.relpath(script_path)}")
    
    result = subprocess.run([interpreter_path, script_path], check=True, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("--- Stderr ---")
        print(result.stderr)


def main():
    """
    Main orchestrator to run the pre-market data pipeline scripts in sequence.
    """
    # --- Start Timer ---
    start_time = time.perf_counter()

    project_root = os.path.dirname(os.path.abspath(__file__))
    
    scripts_to_run = [
        "src/data_processing/news_scraper.py",
        "src/data_processing/daily_bias_computing.py",
        "src/data_processing/identify_key_levels.py",
        "src/data_processing/synthesize_briefing.py",
        "src/data_processing/premarket_focus.py"
    ]

    print("="*60)
    print("Starting Pre-Market Data Pipeline Execution...")
    print(f"Timestamp: {datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S')} EST")
    print("="*60)

    try:
        for i, script_rel_path in enumerate(scripts_to_run):
            script_abs_path = os.path.join(project_root, script_rel_path)
            print(f"\n[{i+1}/{len(scripts_to_run)}] Executing {script_rel_path}...")
            
            if not os.path.exists(script_abs_path):
                print(f"  ! ERROR: Script not found at {script_abs_path}")
                break

            run_script(script_abs_path)
            print(f"--- Successfully completed {script_rel_path} ---")

        print("\n Pre-Market Data Pipeline finished successfully.")

    except FileNotFoundError:
        print(f"\n ERROR: Could not find the Python interpreter at '{sys.executable}'")
    except subprocess.CalledProcessError as e:
        print(f"\n FATAL ERROR: The script '{os.path.basename(e.cmd[-1])}' failed with a non-zero exit code.")
        print("\n--- STDOUT from failed script ---")
        print(e.stdout)
        print("\n--- STDERR from failed script ---")
        print(e.stderr)
        print("--- Halting pipeline execution. ---")
    
    print("="*60)
    
    # --- End Timer and Report Duration ---
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"Total execution time: {duration:.2f} seconds")


if __name__ == "__main__":
    main()