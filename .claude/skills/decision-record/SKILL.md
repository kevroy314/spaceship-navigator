---
name: decision-record
description: Use when a design decision, rejected approach, or measurement that changed the plan needs recording in docs/decisions — and when revisiting or superseding an earlier decision. Enforces the project's standing mandate that reasoning outlives the session.
---

# Writing a decision record

`docs/decisions/NNNN-short-slug.md`, numbered in sequence, append-only.

## When to write one

- A design choice with a real alternative that was rejected.
- A measurement that changed the plan — **especially one that killed your own
  idea**. These are the most valuable records and the easiest to skip.
- A literature finding that redirected the work.
- A retraction. Never silently drop a claim; record it.

Not for: routine implementation, bug fixes without a design consequence,
anything already obvious from the code.

## Template

```markdown
# NNNN — one line, in the imperative or as a claim

**Date** YYYY-MM-DD · **Status** proposed | accepted | superseded by NNNN

## Context
What forced the decision. The constraint, the measurement, the thing that broke.

## Decision
What we are doing. Short.

## Consequences
What this buys, what it costs, and what it rules out. Include the known
weaknesses — a record that only lists upsides is marketing.

## Evidence
Script paths with sample sizes for anything measured here; arXiv/DOI links for
literature. If there is no evidence, write "preference, no evidence" and mean it.
```

## Rules

1. **Numbers, not adjectives.** "eta rose from 0.0026 to 0.0172, sixfold and
   still negligible" beats "the knob helped a little".
2. **Record the mistakes.** If a previous record is now wrong, write a new one
   that supersedes it and add `Superseded by NNNN` to the old. Do not edit the
   old conclusion away.
3. **Say when something is unverified.** `Status: proposed` for designs not yet
   built. Do not write accepted records for things nobody has run.
4. **Update the table** in `docs/decisions/README.md`.
5. Keep each record under about 60 lines. If it is longer, it is probably two
   decisions.

## After writing

Commit the record with the change it describes, not separately — the record and
the code should land together so `git log` and `docs/decisions/` agree.
