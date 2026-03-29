# Agentic Research Programs

This folder is the default local program catalog for `numereng research`.

Tracked by default:
- `numerai-experiment-loop.md`

Custom programs:
- Add a new `<program-id>.md` file in this folder.
- Use the same YAML frontmatter plus markdown prompt-body format as the default program.
- Initialize a session with `uv run numereng research init --experiment-id <id> --program <program-id>`.
- Extra program files in this folder are ignored by git by default, so you can keep local experiments here without polluting the repo.

If you want to point the loader somewhere else temporarily, set `NUMERENG_AGENTIC_RESEARCH_PROGRAMS_DIR`.
