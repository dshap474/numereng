---
name: numerai-research
description: Numerai-focused research combining academic papers and tournament community insights
user-invocable: true
argument-hint: <research query> (e.g., "tabular ML ensembling for Numerai Classic")
---

# Numerai Research

External-first research workflow for Numerai Classic tournament topics. This skill returns research only (no implementation work in the codebase).

Run from:
- `<repo>/packages/numereng`

## Hard requirements (sub-agents)

- Allowed sub-agents: `numerai-papers`, `numerai-community`
- Forbidden sub-agents: any other `subagent_type` (including `web-search`, `code-explorer`, `quant-papers`, `quant-community`)
- Sub-agent lifecycle must be explicit: dispatch -> wait -> capture outputs -> close completed sub-agent threads immediately after completion.
- Coordination must be main-agent-mediated: sub-agents report to the main agent only.
- Do not coordinate sub-agents directly with each other.
- Sub-agents must not spawn additional sub-agents.
- Do NOT edit code, do NOT read local repo files. Only return external research with links.

## Arguments

- `$ARGUMENTS` = Research query (natural language)
- Example: `/numerai-research tabular ML ensembling for Numerai Classic`
- Example: `/numerai-research feature neutralization strategies and MMC optimization`
- Example: `/numerai-research era-aware cross-validation methods`

---

## Step 1: Parallel research (required)

Your next assistant message MUST contain exactly 2 parallel `Task(...)` tool calls (in a single message), with `subagent_type` exactly:

- `numerai-papers`
- `numerai-community`

Do not substitute any other sub-agent.

Execution contract:
- Dispatch both `Task(...)` calls in the same assistant turn.
- Do not add prose before, between, or after those two calls in that turn.
- Wait for both results to return before starting Step 2 synthesis.
- Close completed sub-agent threads immediately after results are returned; do not carry completed threads into Step 2.
- If one task fails, retry only the failed task once; keep the successful task result.
- Do not message sub-agents to coordinate with each other. Main agent relays and synthesizes.

### Task: numerai-papers

```
Task(subagent_type="numerai-papers", model="sonnet", prompt="""
Research query: {$ARGUMENTS}

You are an EXTRACTION agent. Return all unique findings as a structured numbered list. Do NOT synthesize, summarize across sources, or draw conclusions.

Coordination/lifecycle rule: complete this work inside this assigned sub-agent thread only. Do NOT spawn additional sub-agents.

Scope — focus on: theory, empirical evidence, statistical methodology, failure modes, research gaps. Prioritize tabular ML, cross-sectional prediction, and competition ML. Do NOT cover implementation details or parameter tuning.

SOURCING RULES: Every finding MUST have a specific author, year, and URL. Do NOT write "Multiple studies" or "Academic consensus" as a source — find the actual paper. Be exhaustive — find everything relevant and let the synthesizer decide what matters.

MANDATORY MULTI-ROUND WEB SEARCH (required):
1. Round 1 (discovery): run 3-5 broad WebSearch queries to map foundational and recent literature.
2. Round 2 (focus): run 3-5 additional WebSearch queries based on Round 1 gaps/contradictions.
3. Round 3 (adaptive): run 0-5 additional WebSearch queries for unresolved gaps, contradictory evidence, or Numerai-specific edge cases.
4. Round 3 is a decision checkpoint: if you run 0 additional queries, explicitly explain why coverage is already sufficient.
5. Keep a "SEARCH LOG" section before findings with: round number, exact query, and why it was run.

Search strategy by round:
1. Start with seminal/foundational papers for the topic (even if older)
2. Then search for recent work (2020+) that extends, contradicts, or refines the foundations
3. Specifically search for failure modes and known limitations in cross-sectional prediction
4. Search for Numerai-specific terms: "obfuscated features", "feature neutralization", "era-aware", "meta-model"
5. Limit WebFetch calls to 2-3 at a time to avoid cascade failures from parallel 403 errors

Per finding, include:
- Source title (exact paper title)
- Author(s) (specific names, not "various")
- Year
- URL (direct link to paper or abstract)
- Key claim (one sentence with a specific number or result when possible)
- Evidence strength: sample size, time period, asset classes tested
- Limitations noted by the authors
- Numerai Applicability: one sentence on how this applies to the Numerai tournament (obfuscated features, era structure, ~5000-stock universe)

Dedup rules:
- Remove duplicate findings across sources but keep the best-cited version.
- Flag contradictions between sources but do NOT resolve them — this is critical, the synthesizer needs to see disagreements.
- If unsure whether two findings are duplicates, keep both.

At the end, add a "CONTRADICTIONS" section listing any pairs of findings that disagree, with a one-sentence description of the tension.

Output format: numbered list of findings, not prose.
""")
```

### Task: numerai-community

```
Task(subagent_type="numerai-community", model="sonnet", prompt="""
Research query: {$ARGUMENTS}

You are an EXTRACTION agent. Return all unique findings as a structured numbered list. Do NOT synthesize, summarize across sources, or draw conclusions.

Coordination/lifecycle rule: complete this work inside this assigned sub-agent thread only. Do NOT spawn additional sub-agents.

Scope — focus on: implementation patterns, model choices, parameter ranges, tournament results, staking strategy, neutralization, era handling, pitfalls, code examples. Do NOT cover theoretical foundations.

THOROUGHNESS MATTERS: Search broadly first, then drill into forum.numer.ai (highest value source). Aim for 15-25 findings covering:
- At least 2-3 sources with actual code (Python)
- At least 2-3 sources with live tournament results or realistic validation results
- At least 2-3 sources discussing failure modes or what went wrong
- Specific parameter ranges with numbers (not just "tune your learning rate")

MANDATORY MULTI-ROUND WEB SEARCH (required):
1. Round 1 (discovery): run 3-5 broad WebSearch queries to map the community landscape.
2. Round 2 (focus): run 3-5 additional WebSearch queries based on Round 1 findings, prioritizing high-signal forum and code/result sources.
3. Round 3 (adaptive): run 0-5 additional WebSearch queries for unresolved implementation questions or missing metrics/code examples.
4. Round 3 is a decision checkpoint: if you run 0 additional queries, explicitly explain why coverage is already sufficient.
5. Keep a "SEARCH LOG" section before findings with: round number, exact query, and why it was run.

Search strategy by round:
1. Broad quality search first: "numerai" "{topic}"
2. Forum-specific: site:forum.numer.ai {topic}
3. GitHub/NumerBlox for code examples
4. Limit WebFetch calls to 2-3 at a time to avoid cascade failures from parallel 403 errors — if a batch fails, retry individually
5. Prioritize sources that include live staking results or validation metrics (corr, MMC, sharpe)

Per finding, include:
- Source title
- Author/platform
- URL
- Key claim (one sentence)
- Parameter ranges mentioned (specific numbers)
- Tournament metrics (if any): corr, MMC, sharpe, max drawdown, feature exposure
- Code available: Yes/No (language)
- Caveats noted by the author

Dedup rules:
- Remove duplicate findings across sources but keep the best-cited version.
- Flag contradictions between sources but do NOT resolve them.
- If unsure whether two findings are duplicates, keep both.

Flag any source that:
- Reports sustained corr > 0.05 without discussing variance
- Ignores neutralization entirely
- Uses default "target" instead of payout target
- Doesn't mention purge/embargo in validation
- Claims "easy" or "guaranteed" performance

Output format: numbered list of findings, not prose.
""")
```

## Step 2: Synthesize (research-only output)

Only run this step after both Step 1 sub-agent results are available.

Combine both agents' findings into a single report. Your job is to ADD VALUE beyond what the agents returned — resolve contradictions, identify consensus vs. disagreement, and produce an actionable spec tailored to the Numerai tournament.

```
## Research Brief: {topic}

### Executive Summary
2-3 sentences: what the evidence suggests, the most robust approach, and the biggest risk for Numerai Classic.

### Academic Foundation
Key findings from the papers agent. For each major claim, include the specific paper citation and URL. Call out where the literature disagrees and which side has stronger evidence. Focus on what translates to obfuscated cross-sectional prediction.

### Tournament Community Perspective
Common implementation patterns with specific parameter ranges and numbers. Include URLs. Note where community experience contradicts academic findings. Highlight approaches validated by live staking results.

### Numerai-Specific Gotchas
Numbered list of specific technical traps for Numerai Classic. Each gotcha should be actionable. Aim for 8-12 gotchas covering:
- Era leakage (how to detect and prevent)
- Feature correlation handling (groups, selection)
- MMC traps (when optimizing for MMC hurts overall performance)
- Neutralization pitfalls (over/under neutralization, feature vs meta-model)
- Payout formula implications (current scoring, target selection)
- Temporal effects (era-to-era variation, regime changes)
- Ensemble pitfalls (correlated models, diversity measurement)

### Key Sources
| # | Title | Type | Source | URL |
|---|-------|------|--------|-----|
| 1 | ... | Paper | arXiv | ... |
| 2 | ... | Forum | forum.numer.ai | ... |

### Implementation Handoff Spec
Bullet list of concrete requirements for implementation in the numereng package:
- **Target selection**: Which target(s) to use, prediction horizon
- **Feature strategy**: Feature set, feature groups, selection method
- **Model architecture**: Algorithm(s), key hyperparameters, ensemble plan
- **Era handling**: Era weighting scheme, CV method, purge gap size
- **Neutralization strategy**: Proportion, neutralize to features or meta-model
- **Validation plan**: Metrics to track (corr_mean, corr_sharpe, max_dd, feature_exposure, uniqueness/MMC)
- **Ensemble plan**: Diversity sources, combination method (rank avg, weighted blend, stacking)
- **Expected performance range**: Validation estimates + live degradation estimate (validation corr vs expected live corr)
```

## Step 3: Save Response

Save the output research report in this location: `.numereng/notes/research/research-briefs`

Filename MUST be:
`YYYY-MM-DD_{Brief-Name}.md`

Naming rules:
- `YYYY-MM-DD` = current local date at save time.
- `{Brief-Name}` = report title/topic slug.
- Replace spaces with `_`.
- Remove filesystem-unsafe characters: `/ \ : * ? " < > |`.
- Keep filename ASCII-safe.
- If a file with the same name already exists, append a numeric suffix:
  - `YYYY-MM-DD_{Brief-Name}_01.md`
  - `YYYY-MM-DD_{Brief-Name}_02.md`

Examples:
- `.numereng/notes/research/research-briefs/2026-02-25_Memory-Efficient_GBDT_Training_for_Numerai.md`
- `.numereng/notes/research/research-briefs/2026-02-25_Memory-Efficient_GBDT_Training_for_Numerai_01.md`

Ensure the directory exists before saving:
`mkdir -p .numereng/notes/research/research-briefs`

- Include source URLs for every claim.
- Separate consensus from contradictions.
- Include concrete parameter ranges and validation caveats when available.
- Keep this workflow research-only (no repository implementation changes unless explicitly requested in a separate step).
