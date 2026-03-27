#!/usr/bin/env sh
set -eu

repo_dir="${FACTORY_PROJECT_DIR:-$(pwd)}"

printf 'Load repo context from:\\n'
printf -- '- %s/AGENTS.md\\n' "$repo_dir"
printf -- '- %s/.factory/memories.md\\n' "$repo_dir"
printf -- '- %s/.factory/rules/architecture.md\\n' "$repo_dir"
printf -- '- %s/docs/llms.txt\\n' "$repo_dir"
