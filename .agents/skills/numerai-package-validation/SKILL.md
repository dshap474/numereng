---
name: numerai-package-validation
description: Post-closeout deploy gate for an agentic-research candidate recipe. Use when a research run has finished and closed out and the human wants package-level evidence for a believed_best (or runner-up) recipe before a live-slot decision — package build, validation scoring on the PC, live-calibration placement, and the unstaked-deploy handoff.
---

# Package Validation (Deploy Gate)

Run from the numereng repo root. This skill turns a within-lane research champion into
package-level deploy evidence. It is human-triggered and ends at a human decision — it never
deploys on its own. Within-lane BMC200 rank is a candidate ranker, **not** a deploy signal.

Related skills: `numerai-live-calibration-sync` for the calibration refresh/interpretation,
`numerai-api-ops` for hosted-model/account operations, `experiment-finalize` for closeout itself.

## Inputs

- Experiment id of the finished agentic run.
- Candidate recipe(s): default to the closeout proposal / `believed_best`; optionally 1–2 diverse
  runners-up.

## Phase 1: Identify Candidates

1. Read `.numereng/experiments/<id>/agentic_research/state.json` → `believed_best`
   (`config` + `run_ids`) and `champion`; cross-check `closeout/next/PROPOSAL.md`.
2. For runners-up, reconstruct recipe-trio groups from `journal.jsonl` via `aggregate_recipes()`
   in `src/numereng/agentic_research/engine/aggregate.py`.
3. Confirm the candidate's seed-trio run ids exist and are `FINISHED`. Note which machine holds
   the run artifacts (usually the PC) — packages must be built where the runs live.

## Phase 2: Build And Score The Package (PC only)

ALL scoring compute runs on the PC (`ssh pc`) — never on the Mac, even when it would fit.

1. Sync any local code/config changes first: `uv run numereng remote repo sync ...`.
2. On the PC, build and score:

```bash
uv run numereng serve package create --experiment-id <id> --package-id <pkg> --components <json>
uv run numereng serve package score --experiment-id <id> --package-id <pkg> --runtime local
```

PC gotchas: use `--runtime local` (pickle-runtime smoke hits WinError 206 on Windows); for
long-running scores, detach via WMI `Win32_Process.Create` — Windows OpenSSH kills child
processes when the SSH session closes.

3. Pull results back for Mac-side analysis with
   `uv run numereng remote experiment pull --mode scoring` (use `--mode full` only if prediction
   parquets are needed later for submit/ensemble work).

## Phase 3: Read The Package Metrics

From the package `summaries.json`, read `bmc_last_200_eras_mean` and `fnc_mean`.

Scale-convention rules (violating these produced a bad read once — respect them):

- Package FHR (train+validation) scores are **in-sample** and capacity-inflated: a deeper/larger
  recipe inflates more. Compare package-scale numbers only between same-capacity recipes; use a
  known live model's package score as an anchor, never as a cross-capacity ranker.
- Never mix package-level BMC200 with agentic per-era research metrics (OOF purged-WF scale).
  The honest within-lane ordering evidence is the OOF scale from the research run itself.

## Phase 4: Live-Calibration Placement

1. Refresh if stale: `uv run numereng submissions calibration update --format json`
   (or run the `numerai-live-calibration-sync` skill for a full interpretation pass).
2. Place the candidate's local metrics on the regression in
   `.numereng/analysis/live_calibration/report.json`.
3. Do not extrapolate: if the candidate's local metric falls outside the observed regression
   domain, say so explicitly — no live mapping exists for it.
4. Deploy recommendation requires BOTH: the candidate is a credible point on the calibration
   regression AND it adds coverage the live portfolio lacks (new lane, target family, or feature
   scope). Never recommend deploy on within-lane rank alone — within-lane local→live rank
   correlation is ≈ 0.

## Phase 5: Human Decision And Deploy Handoff

Present the evidence and stop. The slot decision is the human's.

If (and only if) the human explicitly authorizes deploy:

- Deploy via hosted pickle, **unstaked**, to gather live data
  (`serve live build` → upload path; hosted upload is stricter than local live builds — local
  success does not imply upload success).
- As rounds resolve, re-run `uv run numereng submissions calibration update --format json` and
  fold the new observations into the calibration stance.

## Safety

- No Numerai writes (slot creation, pickle upload, submissions, staking) without the human's
  explicit authorization in the current conversation.
- No heavy compute on the Mac. Route every multi-minute score to the PC.
- Do not delete or rewrite `.numereng/` state; packages and runs are append-only evidence.
- Do not rank cross-capacity recipes on the in-sample package convention.
- This gate reuses existing entrypoints only — it never requires harness code changes.

## Done Criteria

Report:

- candidate recipe(s), their seed-trio run ids, and the source (believed_best / runner-up);
- package id(s) and where they were scored;
- `bmc_last_200_eras_mean` + `fnc_mean` per package, with the scale-convention caveat stated;
- calibration placement: predicted live range, or an explicit out-of-domain statement;
- the coverage argument (what the candidate adds that live slots lack);
- a deploy recommendation framed as evidence for the human decision, not an action taken.
