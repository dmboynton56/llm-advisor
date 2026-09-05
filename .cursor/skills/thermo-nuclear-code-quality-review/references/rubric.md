# Official code-quality rubric

Source: `cursor/plugins` Thermos skill `thermo-nuclear-code-quality-review`. Apply as written.

## Non-negotiable standards

0. **Be ambitious about structural simplification.** Look for reframes that delete branches, helpers, modes, or layers. Prefer the solution that feels inevitable. If you can delete complexity rather than rearrange it, push that path.

1. **Do not let a PR push a file from under 1k lines to over 1k lines** without a very strong reason. Prefer extract over sprawl. Waive only with a compelling structural reason and a still-organized file.

2. **Do not allow random spaghetti growth.** New ad-hoc conditionals in unrelated flows are a design problem. Push logic into a dedicated abstraction, helper, state machine, or module.

3. **Bias toward cleaning the design**, not accepting "it works." Prefer simplifications that remove moving pieces over refactors that spread the same complexity.

4. **Prefer direct, boring, maintainable code** over hacky or magical code. Flag thin wrappers and generic mechanisms that hide simple data-shape assumptions.

5. **Type and boundary cleanliness.** Question unnecessary optionality, `unknown`/`any`, and cast-heavy code. Prefer explicit contracts over silent fallbacks.

6. **Keep logic in the canonical layer.** Reuse existing helpers. Do not leak feature logic into shared paths.

7. **Unnecessary sequential orchestration / non-atomic updates** are smells when a cleaner structure is obvious. Do not micro-optimize; do flag brittle half-applied state.

## Primary questions

- Is there a code-judo move that would make this dramatically simpler?
- Can this be reframed with fewer concepts, branches, or helper layers?
- Did the diff add branching where a better abstraction should exist?
- Did a cohesive module become more coupled, stateful, or harder to scan?
- Is this logic in the right file and layer?
- Did a file cross a healthy size boundary?
- Repeated conditionals → missing model or helper?
- Is the abstraction earning its keep, or just a wrapper?
- Casts / optionality / ad-hoc shapes obscuring the invariant?
- Orchestration more sequential or less atomic than it needs to be?

## Flag aggressively

Complicated implementations a reframe could delete; refactors that move complexity without reducing it; files crossing 1000 lines; conditionals bolted onto unrelated paths; one-off booleans/flags; feature logic in general-purpose modules; magic generics; thin identity wrappers; unnecessary casts/`any`; copy-paste instead of helpers; edge-case handling mid busy function; "temporary" branches; bespoke helpers that duplicate a canonical utility; logic in the wrong layer; sequential async for independent work; partial updates that leave state half-applied.

## Preferred remedies

Delete a layer of indirection. Reframe the state model so conditionals disappear. Change the ownership boundary. Turn special cases into a simpler default. Extract a pure helper. Split a large file. Replace condition chains with a typed model or dispatcher. Separate orchestration from business logic. Collapse duplicate branches. Reuse the canonical helper. Make type boundaries explicit. Parallelize independent work when that also simplifies. Make related updates atomic.

Do not settle for "maybe rename this" when the issue is structural. Do not settle for a cleaner version of the same messy idea if a simpler idea exists.

## Tone

Direct, serious, demanding. Not rude. Do not soften major maintainability issues.

Useful phrases: file past 1k — decompose first; special-case in a busy flow — own abstraction; works but more spaghetti — keep behavior, restructure; feature logic leaking into a shared path; unnecessary abstraction; why the cast/optional; reuse the canonical helper; code-judo to make branches disappear; refactor moved complexity but did not delete it.

## Output order

1. Structural regressions
2. Missed dramatic simplification
3. Spaghetti / branching growth
4. Boundary / type-contract problems
5. File-size / decomposition
6. Modularity
7. Legibility

Fewer high-conviction comments over cosmetic nits.

## Approval bar

Do not approve merely because behavior seems correct. Blockers unless clearly justified:

- incidental complexity when a code-judo delete is visible
- file crosses 1000 lines
- ad-hoc branching that tangles an existing flow
- feature checks scattered across shared code
- unnecessary wrapper / cast-heavy contract
- duplicated helper or logic in the wrong layer
