---
name: numerai-docs
description: Search and summarize the local Numerai docs mirror and forum archive for tournament, scoring, data, model upload, staking, and community-reference questions.
user-invocable: true
---

# Numerai Docs

## Role / Purpose

Use this skill when the user wants answers grounded in Numerai documentation or archived community/forum material. Prefer targeted search and exact source paths over broad reading.

## Source Hierarchy

1. Official local docs mirror:
   `docs/numerai/`
2. Table of contents:
   `docs/numerai/SUMMARY.md`
3. Official tournament docs:
   `docs/numerai/numerai-tournament/`
4. Official Signals and Crypto docs:
   `docs/numerai/numerai-signals/`
   `docs/numerai/numerai-crypto/`
5. Local forum/community archive:
   `docs/numerai/forum/`
   `docs/numerai/community/content/`
   `docs/numerai/community/numerai-council-of-elders/`

Treat official docs as authoritative. Use forum/community material as supporting context only, and label it as community evidence. If the user asks for the latest/current Numerai rules, verify against current official sources instead of relying only on the local mirror.

## Search Workflow

1. Classify the question:
   - Tournament model/data/scoring/submission/staking question: start in `numerai-tournament/`.
   - Signals or Crypto question: start in the matching product folder.
   - API, model upload, hosted model, or automation question: search `submissions/`, `models`, `connect/`, and `mcp.md`.
   - Community strategy or historical discussion: search `forum/INDEX.md`, then `forum/posts/` and `community/content/`.
2. If routing is unclear, read `SUMMARY.md` first and choose the narrowest docs folder.
3. Search before reading whole files. Use `rg` with a small set of synonyms:
   ```bash
   rg -n -i "bmc|benchmark|meta model|mmc|corr|fnc|tc" docs/numerai
   ```
4. Read only the highest-signal matched files or sections.
5. Answer with source type and path:
   - `official docs`: exact local path
   - `forum/community`: exact local path and caveat that it is not official policy

## Path Map

- Tournament overview: `docs/numerai/README.md`
- Tournament data: `docs/numerai/numerai-tournament/data.md`
- Tournament models: `docs/numerai/numerai-tournament/models.md`
- Tournament submissions: `docs/numerai/numerai-tournament/submissions/`
- Model uploads: `docs/numerai/numerai-tournament/submissions/model-uploads.md`
- Scoring overview: `docs/numerai/numerai-tournament/scoring/README.md`
- Scoring definitions: `docs/numerai/numerai-tournament/scoring/definitions.md`
- CORR: `docs/numerai/numerai-tournament/scoring/correlation-corr.md`
- MMC: `docs/numerai/numerai-tournament/scoring/meta-model-contribution-mmc.md`
- FNC: `docs/numerai/numerai-tournament/scoring/feature-neutral-correlation.md`
- TC: `docs/numerai/numerai-tournament/scoring/true-contribution-tc.md`
- Grandmasters/seasons: `docs/numerai/numerai-tournament/scoring/grandmasters-and-seasons.md`
- Staking: `docs/numerai/numerai-tournament/staking.md`
- MCP/API surface: `docs/numerai/numerai-tournament/mcp.md`
- General FAQ: `docs/numerai/tournament/numerai-general-faq.md`
- Forum index: `docs/numerai/forum/INDEX.md`

## Common Search Patterns

- Metrics and scoring:
  `rg -n -i "corr|mmc|fnc|tc|cwmm|payout|multiplier|drawdown|benchmark" docs/numerai/numerai-tournament docs/numerai/help`
- Model uploads and live submission:
  `rg -n -i "upload|pickle|submission|prediction|diagnostics|cron|compute" docs/numerai/numerai-tournament docs/numerai/connect docs/numerai/forum`
- Data, eras, targets, features:
  `rg -n -i "era|target|feature|validation|live|training|data version|v5" docs/numerai/numerai-tournament`
- Grandmaster, leaderboard, staking:
  `rg -n -i "grandmaster|season|leaderboard|stake|staking|payout|rank" docs/numerai/numerai-tournament docs/numerai/help`

## Output Rules

- Keep answers scoped to what the docs actually support.
- Quote sparingly; paraphrase and cite paths.
- Distinguish official documentation from forum/community practice.
- Call out local-mirror staleness when the user asks about current rules, current APIs, or anything likely to change.
