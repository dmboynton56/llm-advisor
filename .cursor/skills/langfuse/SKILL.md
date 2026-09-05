---
name: langfuse
description: Live-loop LLM traces and Langfuse observability. Use when instrumenting or querying production traces, prompts, datasets, or scores for the advisor loop — not for PR prompt eval (that is Promptfoo).
---

# Langfuse — live loop traces

This repo skill is a **pointer**. The full skill is [langfuse/skills](https://github.com/langfuse/skills) (`skills/langfuse`). Install globally:

```bash
npx skills add langfuse/skills --skill langfuse
# or Cursor: /add-plugin langfuse
```

Prefer `~/.cursor/skills/langfuse` (or the plugin) when present. **Documentation first** — Langfuse changes often. Do not implement from memory or invent SDK/API shapes.

**When:** instrument or debug live-loop / premarket LLM calls (`src/analysis/`, `src/premarket/`, `src/live/`). AGENTS.md: Langfuse = live traces; Promptfoo = PR eval gate.

**Not when:** adding a Promptfoo fixture, merging a prompt-only PR, or changing EOD recon display.

## Access docs (official)

1. Index: `https://langfuse.com/llms.txt`
2. Page as markdown: append `.md` or `Accept: text/markdown` (e.g. `https://langfuse.com/docs/observability/overview.md`)
3. Search: `https://langfuse.com/api/search-docs?query=...`
4. Agents overview: `https://langfuse.com/agents.md`

Prefer native WebFetch/WebSearch over curl when available.

## Data plane (official CLI — do not hand-roll REST)

```bash
npx langfuse-cli api __schema
npx langfuse-cli api <resource> --help
npx langfuse-cli api <resource> <action> --help
```

Env (never paste keys into chat; never commit them):

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_BASE_URL=https://cloud.langfuse.com   # or us.cloud.langfuse.com / self-host
export LANGFUSE_HOST="$LANGFUSE_BASE_URL"             # if the CLI expects HOST
```

If unset, ask the operator to set them locally or in the live-loop secret store — not in git.

## Use-case references

Load from the **global** skill's `references/` (instrumentation, prompt-engineering, setting-up-evals, cli, ci-cd, sdk-upgrade). Those files are the source of truth. This repo does not vendor them.

CI experiment gates use `langfuse/experiment-action` only when docs say so — that is not a substitute for the Promptfoo PR gate in AGENTS.md.
