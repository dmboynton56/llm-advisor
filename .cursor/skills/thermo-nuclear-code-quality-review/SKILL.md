---
name: thermo-nuclear-code-quality-review
description: Run an extremely strict maintainability review for abstraction quality, giant files, and spaghetti-condition growth. Use for a thermo-nuclear code quality review, thermonuclear review, deep code quality audit, or especially harsh maintainability review.
disable-model-invocation: true
---

# Thermo-Nuclear Code Quality Review

Adapted from the official Thermos skill (`cursor/plugins` → `thermos/skills/thermo-nuclear-code-quality-review`). Prefer the globally installed copy if present. Invoke explicitly via `/thermo-nuclear-code-quality-review` (this skill does not auto-attach).

Unusually strict review of **implementation quality** on the current branch. Preserve behavior. Hunt for "code judo": reframe so whole branches, helpers, or layers disappear.

## Core prompt

> Perform a deep code quality audit of the current branch's changes.
> Rethink how to structure / implement the changes to meaningfully improve code quality without impacting behavior.
> Work to improve abstractions, modularity, reduce spaghetti, improve succinctness and legibility.
> Be ambitious. If a clear path to a simpler implementation exists, take it.
> Be extremely thorough. Measure twice, cut once.

## This repo

Do not "simplify" away recon honesty (lifecycle vs broker) or fail-closed trading-job contracts. Prefer boring, direct code in `src/live/` and `src/execution/` over clever indirection.

## Load next

Read [references/rubric.md](references/rubric.md) for the official standards, flag list, remedies, tone, and approval bar. Do not invent extra rubrics.
