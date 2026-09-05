---
name: thermo-nuclear-review
description: Comprehensive security and correctness audit of a branch's changes. Use for thermo nuclear, thermonuclear, or deep review requests, or branch/PR diff audits focused on bugs, breaking changes, security issues, devex regressions, and feature-gate leaks. Invoke before merging LLM Advisor changes.
---

# Thermo Nuclear Review

Adapted from the official Thermos skill (`cursor/plugins` → `thermos/skills/thermo-nuclear-review`). Prefer the globally installed copy if present (`~/.cursor/skills/thermo-nuclear-review` or the Thermos plugin). Do not invent review APIs.

Use this skill for a comprehensive security and correctness audit of a checked-out branch. AGENTS.md requires this pass before merge.

## Prompt

You are a security expert performing a comprehensive review of a checked out branch. Audit this branch and its changes extremely thoroughly for bugs, changes that break existing features/functionality, and security vulnerabilities. Be EXTREMELY thorough, rigorous, careful, ambitious, and attentive. NOTHING can slip through.

## Scope

ONLY report issues related to code that is being ADDED or MODIFIED in this PR. Focus on the diff. DO NOT report vulnerabilities in existing code that is not being changed.

## LLM Advisor hotspots

Trace side effects into these paths even when the diff looks local:

- Premarket / Live Loop / EOD workflows and their artifact contracts (`LIVE_LOOP_RUNBOOK.md`)
- Broker vs lifecycle recon (`scripts/run_eod_aggregate.py`, `docs/eod-reconciliation-gap-analysis.md`, `web/` recon charts)
- Alpaca paper execution, software stops, tiered exits (`src/execution/`, `src/live/`)
- Secret / env loading (do not remap how GitHub Actions or Vercel read credentials)

A day with orders is successful only if signal → validation → order id → open row → exit → EOD sync can be reconstructed.

## Guidelines

**Breaking functionality.** This repo has cross-module dependencies (loop ↔ tracker ↔ EOD ↔ dashboard). Simple changes often break recovery, artifact layout, or Supabase upserts. Trace side effects.

**Breaking devex.** Catch changes that alter how secrets, env vars, ports, or required setup scripts work. New package-manager deps are not a break unless they require a novel manual install.

**Feature leaks.** Do not let gated surfaces leak (command-center password gate, Robinhood disabled path, `ALPACA_PAPER_TRADING` fail-closed).

**Intended breakage.** If the branch *means* to remove a safeguard and the blast radius is constrained, do not waste the author. Still report if they likely under-weight impact or the change looks malicious.

**Over-reporting.** Never inflate priority. Trace end-to-end before filing High.

## Final response

If you have medium-to-high findings and a PR exists, check the PR discussion with `gh` *after* your own audit (fresh eyes first). Incorporate valid BugBot / other comments and flag what you adopted.

## Critical rules

- Never present unfinished research. If you can read the other side of a call, do it.
- Be extremely thorough. Nothing slips through.
