---
name: experiment-contracts
description: "Factory-native mirror of the numereng experiment contract workflow."
---

# Experiment Contracts

This skill mirrors the existing repo-local experiment workflow knowledge.

## Canonical sources

- Primary long-form skill: `.agents/skills/numereng-experiment-ops/SKILL.md`
- Training schema source: `src/numereng/config/training/CLAUDE.md`
- Deep execution map: `docs/llms.txt`

## Use this skill for

- experiment layout questions
- config placement and naming
- training config schema routing
- experiment CLI/API contract questions
- expected run artifacts for experiments

## Rule

- Keep `.factory/skills/*` and `.agents/skills/*` aligned; do not let the mirrored Factory route drift from the canonical repo workflow.
