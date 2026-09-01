---
name: "utility-cleanup-report"
description: "Survey heavy numereng artifacts on the local machine and the remote PC (run dirs, package pickles, experiment dirs, datasets) and produce a read-only cleanup report: what is safe to delete, what is protected, and why. Triggers: cleanup report, disk usage, what can we delete, prune heavy artifacts, free up space. Report only — never deletes; hands actionable recommendations to utility-store-ops."
---

# Cleanup Report

Read-only skill. It inventories heavy artifacts on the local workspace and (when reachable)
the remote PC, classifies them against saved-results evidence, and delivers a report.
It never deletes, moves, or mutates anything. Actual deletion belongs to `utility-store-ops`.

Run from the repo root.

## Hard Rules

- Never delete, rename, or rewrite any artifact. No `--execute` anything. Report only.
- Never classify something as safe without checking the rules in
  `references/safety-classification.md`.
- Sizes in the report come from the survey JSON, not estimates.
- If the remote is unreachable, produce the local-only report and say so explicitly.

## Reference Loading Guide

| Request Type | Load |
|---|---|
| Classifying any artifact as safe/protected | `references/safety-classification.md` |

If the task is only "how big is X", skip the reference and just run the survey.

## Asset Usage Guide

| Task | Use |
|---|---|
| Inventory one machine's store | `scripts/survey_artifacts.py` |

## Core Workflow

### 1) Local survey

```bash
uv run python .agents/skills/utility-cleanup-report/scripts/survey_artifacts.py > .numereng/tmp/cleanup_survey_local.json
```

### 2) Remote survey (PC)

Resolve the remote repo root from `src/numereng/platform/remotes/profiles/*.yaml`
(do not hardcode). Then:

```bash
scp .agents/skills/utility-cleanup-report/scripts/survey_artifacts.py <host>:<repo_root>/.numereng/tmp/
ssh <host> powershell -Command "cd <repo_root>; uv run python .numereng/tmp/survey_artifacts.py" > .numereng/tmp/cleanup_survey_remote.json
```

Note: the PC's default ssh shell is cmd.exe — avoid `|` pipes inside the remote command
unless the whole `powershell -Command "..."` string is single-quoted locally.

### 3) Cross-reference saved results

Load `references/safety-classification.md`, then gather:

- distilled experiments: `ls .numereng/notes/__RESEARCH_MEMORY__/experiments/`
- hosted packages + their protected component run IDs (from the survey JSON's
  `packages` entries with `hosted: true`)
- run IDs duplicated across both surveys

### 4) Classify and write the report

Classify every heavy group (runs by experiment, package pickles, big experiment dirs,
datasets) into: **Safe now / Safe with prerequisite / Protected / Needs human judgment**.

Save the report to `.numereng/tmp/cleanup_report_<YYYY-MM-DD>.md` and present a
summary in chat. Report structure:

```markdown
# Cleanup Report <date>
## Totals (per machine)
## Safe to delete now            (what, where, size, why safe)
## Safe after prerequisite       (what, prerequisite, size)
## Protected — do not delete     (what, why)
## Needs human judgment          (what, size, open question)
## Recommended next steps        (utility-store-ops commands, dry-run first; metadata archiving)
```

Every "safe" row must name its evidence (research-memory branch, hosted upload ID,
duplicate location). Every recommendation must route deletion through `utility-store-ops`
dry-run → execute, followed by `store doctor` — never raw `rm` for indexed runs.

## Error Handling

- Remote ssh/scp failure: retry once, then produce local-only report and flag the gap.
- Survey script traceback: fix nothing on the remote silently; report the error verbatim.
- Missing `__RESEARCH_MEMORY__`: classify all experiment runs as "needs human judgment".

## Done Criteria

- Survey JSON exists for each surveyed machine under `.numereng/tmp/`.
- Report file written and summarized in chat with per-machine totals and a
  clearly separated safe-now vs prerequisite vs protected breakdown.
- Zero mutations performed anywhere.
