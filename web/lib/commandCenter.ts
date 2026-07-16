export interface WatchlistFlag {
  symbol: string;
  flagged: boolean;
  note: string;
}

export interface Opportunity {
  id: string;
  symbol: string;
  thesis: string;
  source: string;
  direction: "bullish" | "bearish" | "neutral";
}

export async function getWatchlistFlags(): Promise<WatchlistFlag[]> {
  // EXTENSION POINT: replace mock rows with a private Supabase watchlist_flags table.
  return [
    { symbol: "SPY", flagged: true, note: "Watch for a 2σ mean-reversion setup." },
    { symbol: "QQQ", flagged: false, note: "Confirm breadth before acting." },
    { symbol: "IWM", flagged: true, note: "Relative-strength divergence vs SPY." },
  ];
}

export async function getOpportunities(): Promise<Opportunity[]> {
  // EXTENSION POINT: combine Supabase signal events with Robinhood MCP scanners and quotes.
  return [
    {
      id: "mock-spy-reversion",
      symbol: "SPY",
      thesis: "Mock opportunity: downside z-score stretched while the higher-timeframe bias remains neutral.",
      source: "STDEV signal preview",
      direction: "bullish",
    },
    {
      id: "mock-iwm-relative-strength",
      symbol: "IWM",
      thesis: "Mock opportunity: small-cap relative strength is improving into the afternoon window.",
      source: "Command-center mock feed",
      direction: "neutral",
    },
  ];
}
