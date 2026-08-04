-- Persisted option lifecycle state for the paper tiered-exit trial.
ALTER TABLE positions
  ADD COLUMN IF NOT EXISTS exit_state JSONB;

ALTER TABLE premarket_snapshots
  ADD COLUMN IF NOT EXISTS levels JSONB;
