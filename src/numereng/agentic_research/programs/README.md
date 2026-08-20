"""Tracked marker for local agentic research programs."""

# Programs

`PROGRAM.md` is the tracked canonical/default program; its CORE sections are the byte-verbatim
contract every custom program must copy.

Put local, experiment-specific `*.md` programs in this directory (or, preferred, in the
experiment's own folder) and select one with `metadata.agentic_research_program` in an experiment
manifest. Finished-experiment programs belong in `archive/` (exempt from the CORE-drift lint).

Custom programs and `archive/` are gitignored; only `PROGRAM.md` and this README are tracked. See
`src/numereng/agentic_research/README.md` for authoring rules.
