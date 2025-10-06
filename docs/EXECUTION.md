# Trading Strategy: ICT Liquidity Flow Model

This document defines the 5-step, top-down trading methodology that the AI agent (`LiquidityFlowGPT`) must follow. The strategy is based on Inner Circle Trader (ICT) concepts, focusing on how institutional order flow creates predictable patterns of liquidity.

## The 5-Step Methodology

### Step 1: Determine Daily Bias
**Objective:** Identify the most likely direction for the day's primary price move.
* **Method:** Analyze High-Timeframe (HTF) charts (1-hour, 4-hour, Daily) to identify key **Draws on Liquidity**. These are price levels where a significant number of buy-stops or sell-stops are likely resting.
* **Key Levels to Mark:**
    * Previous Day High/Low
    * Previous Session High/Low (Asia, London)
    * Old Highs/Lows on the 1h/4h chart
    * Highs/Lows created by major news or volume events.
* **Thesis:** The market will likely move toward the most obvious pool of liquidity. A bias is **Bullish** if the draw is to the upside; **Bearish** if the draw is to the downside.

### Step 2: Await Reversal at Key Levels
**Objective:** Wait for a "stop hunt" or liquidity grab at a key HTF level.
* **Method:** Once the daily bias is established, patiently wait for the price to sweep a key level *against* the expected direction (e.g., a sweep of the previous day's low in a Bullish bias). This is often a manipulation move to fill institutional orders.
* **Action:** Once the sweep occurs, zoom into the 5-minute chart to watch for the first sign of a reversal.

### Step 3: Confirm Low-Timeframe (LTF) Reversal
**Objective:** Confirm that the manipulation is over and the true trend is resuming.
* **Method:** Look for a **confluence** of at least two confirmation signals on the 5-minute chart.
* **Confirmation Signals:**
    * **Break of Structure (BOS):** A decisive close above a recent swing high (for bullish) or below a swing low (for bearish). This is the most important signal.
    * **Inverse Fair Value Gap (iFVG):** A small FVG that forms *against* the new direction and is immediately disrespected and traded through.
    * **79% Fibonacci Retracement:** A bounce from the 79% ("Optimal Trade Entry") level of the initial reversal move.
    * **SMT Divergence:** A divergence between correlated assets (e.g., ES/NQ or SPY/QQQ) where one fails to make a new high/low while the other succeeds.

### Step 4: Identify LTF Entry
**Objective:** Find a precise, low-risk entry point to join the newly confirmed trend.
* **Method:** After the 5-minute reversal is confirmed, wait for a pullback to a high-probability continuation setup. Then, refine the entry on the 1-minute chart.
* **5-Minute Continuation Setups:**
    * **Fair Value Gap (FVG):** A 3-candle imbalance left during the reversal move. This is the highest probability entry.
    * **Order Block:** The last down-candle before a strong up-move (or vice-versa).
    * **Breaker Block:** A failed order block that price has traded through.
* **1-Minute Refinement:** Look for a 1-minute BOS or iFVG *within* the 5-minute FVG to time the entry with maximum precision.

### Step 5: Define Trade Parameters
**Objective:** Set clear exit points based on market structure.
* **Entry:** Taken at the 1-minute confirmation within the 5-minute continuation setup.
* **Stop Loss:** Placed just below the swing low of the reversal structure (for bullish) or above the swing high (for bearish).
* **Take Profit:** Set at the next major HTF Draw on Liquidity, as identified in Step 1.