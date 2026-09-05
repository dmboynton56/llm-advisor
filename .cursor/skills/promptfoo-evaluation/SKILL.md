---
name: promptfoo-evaluation
description: PR eval gate for prompt and trading-policy changes via Promptfoo. Use when changing LLM prompts, validation schemas, threshold-review text, or comparing models before merge. Triggers on promptfoo, eval, LLM evaluation, prompt testing, or model comparison.
---

# Promptfoo — PR eval gate

This repo skill is a **pointer**. The full playbook is the community skill [daymade/claude-code-skills `promptfoo-evaluation`](https://github.com/daymade/claude-code-skills/tree/main/promptfoo-evaluation) (global install: `~/.cursor/skills/promptfoo-evaluation`). CLI and config: [promptfoo.dev](https://www.promptfoo.dev/) / schema `https://promptfoo.dev/config-schema.json`. Do not invent assertion types, provider IDs, or CLI flags.

**When:** before merging diffs that touch prompts or trading-policy text. AGENTS.md: Promptfoo is the PR gate; Langfuse is live traces — do not swap them.

**Not when:** live-loop production traces (use `langfuse`); EOD/recon/dashboard-only diffs with no prompt change.

## This repo's prompts

Inline in Python, not a `prompts/` tree yet:

- Premarket posterior — `src/premarket/bias_validator.py`
- Live threshold review — `src/analysis/market_analyzer.py`
- Trade validation — `src/analysis/trade_validator.py`
- Client / default model `gpt-5.4-nano` — `src/analysis/llm_client.py`

If no `promptfooconfig.yaml` exists, create one from the official schema and the daymade skill (do not guess YAML keys). Prefer `file://` prompts extracted from those modules over rewriting policy in the config.

## Known-good CLI (from daymade / Promptfoo docs)

```bash
npx promptfoo@latest init
npx promptfoo@latest eval
npx promptfoo@latest eval --config path/to/promptfooconfig.yaml --output results/eval-results.json
npx promptfoo@latest view
```

- `maxConcurrency` belongs under `commandLineOptions`, not top-level (silently ignored otherwise).
- `file://` paths resolve relative to the `promptfooconfig.yaml` directory.
- Preview without API spend: provider `echo`.
- Python missing: `export PROMPTFOO_PYTHON=python3`.
- `llm-rubric` graders do **not** inherit a parent provider's `apiBaseUrl` — set per assertion if using a relay.

Depth: load the global daymade `SKILL.md` / `references/promptfoo_api.md` if installed; otherwise fetch those files from GitHub. Do not improvise HTTP APIs.
