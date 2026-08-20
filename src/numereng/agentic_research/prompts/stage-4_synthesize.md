<!--
stage-4_synthesize.md — closeout phase 4 prompt (tracked, generic; no machine-specific paths).
The runner substitutes {{CONTEXT_JSON}} with the bounded SYNTHESIZE context: the classification's
selected branch topics, their master-ledger views, and CURRENT.md. The LLM returns exactly one delta
for each selected topic plus a full CURRENT.md replacement. Python merges deterministically.
-->

# Closeout Phase 4 — SYNTHESIZE

You are folding one completed experiment's selected research-memory topics into the master research
memory and compressed frontier file `CURRENT.md`. Work only from the supplied context.

## Inputs

`relevant_topics` is the exact classification-selected topic list. `branch_files` contains the
README plus only those topic files. Each selected ledger in `ledgers` is shown as its two mutable top sections — `## Current
Overview` and `## Current Best Understanding` — plus its newest entries. `current_md` is the current
frontier file. `experiment_id` is this experiment's id. Use ONLY these — never invent metrics.

```json
{{CONTEXT_JSON}}
```

## Standing doctrine (apply throughout)

- Ledgers are **append-only** per experiment: you add one new entry, you never rewrite prior entries.
- Each mutable top section has its own optional replacement. Supply `overview_replacement_markdown`
  only when this experiment changes the standing overview for that topic, and
  `best_understanding_replacement_markdown` only when it changes the standing best understanding;
  otherwise return `null` for that field and the harness preserves that section verbatim. Each
  replacement is a **section body only** — it must NOT contain any `## ` heading (the harness re-adds
  the section heading); replacing one section never touches the other.
- `CURRENT.md` is a **compression**, not an accumulation: rewrite it so it stays a tight frontier,
  folding this experiment in and dropping anything it supersedes.
- Preserve evidence discipline: carry the branch's evidence levels and design-space roles into the
  claims you promote; do not upgrade a `hypothesis / next-step` into a settled result.

## Comparison class (label before integrating)

Label the integrated experiment so later synthesis cannot mix unlike evidence, and carry the label
into the text you write:

- `broad screening surface`: many targets, feature sets, or model variants; moves priors and search
  direction.
- `narrow candidate-quality packet`: a focused test of one family or a small candidate set; supports
  candidate confidence, not broad replacement claims.
- `champion / production evidence`: validated handoff evidence with sufficient scoring, ensemble,
  exposure, and operating-gate coverage.

A narrow experiment must not overwrite broad-screening priors unless it is directly comparable or
repeated. When it is not, append the scoped learning and leave the standing section alone (`null`).

## Frontier update rules

Before changing any `CURRENT.md` content — and before supplying either replacement body — state in
the markdown you produce:

- scope boundary: exactly what this experiment tested;
- comparison class: broad screen, narrow packet, or champion evidence;
- active candidate set: which families or configs remain worth testing;
- frontier belief: what changed in the research direction;
- champion state: usually `none`; never promote a candidate without explicit production-ready evidence;
- blocking gates: missing ensemble, full scoring, exposure, live, or direct-comparison evidence;
- confounds changed together: features, targets, depth or model class, horizons, seed count, or
  neutralization.

Do not compress `medium standard-large Ender worked` into `medium is better`; preserve the actual
tested surface.

## Required output

Return exactly one JSON object. `deltas` must contain each topic in `relevant_topics` exactly once
and no other topic. Do not manufacture no-op deltas for unselected topics.

```json
{"deltas": [
  {"topic": "<selected-topic>", "new_entry_markdown": "...", "overview_replacement_markdown": null, "best_understanding_replacement_markdown": null}
 ],
 "current_md": "<the full rewritten CURRENT.md>",
 "notes": "<one-line summary of the frontier change>"}
```

For each topic, `new_entry_markdown` MUST:
- have as its sole markdown heading `### <experiment_id>` (the exact id from the context, and no other
  heading of any level);
- link the branch source for that topic as `../experiments/<experiment_id>/<topic>.md` (for example
  `../experiments/<experiment_id>/features.md`);
- summarize this experiment's contribution to that topic, carrying its evidence level and comparison
  class.

`overview_replacement_markdown` and `best_understanding_replacement_markdown` are each either the full
replacement body for that section (no `## ` heading — body only), or `null` to preserve that section
verbatim. The two fields are independent: replacing one never affects the other.

`current_md` MUST include the level-2 sections `## Compressed Frontier`, `## Comparison Anchors`, and
`## Current Constraints`; name this `experiment_id` explicitly; carry a line beginning `Full record:`
that points to `experiments/<experiment_id>/README.md`; and be a substantial compressed rewrite (well
over 2,000 characters). Cover the frontier update rules above within those sections. Output only the
JSON object, nothing else.
