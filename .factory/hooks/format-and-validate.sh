#!/usr/bin/env sh
set -eu

repo_dir="${FACTORY_PROJECT_DIR:-$(pwd)}"

cd "$repo_dir"

make test
