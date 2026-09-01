#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS_DIR="${ROOT_DIR}/src/numereng/assets"
DOCS_SHARED_SRC="${ROOT_DIR}/docs/assets"
DOCS_SHARED_DST="${ASSETS_DIR}/docs/assets"
SKILLS_SRC_ROOT="${ROOT_DIR}/.agents/skills"
SKILLS_DST_ROOT="${ASSETS_DIR}/shipped_skills"
MANIFEST_PATH="${SKILLS_SRC_ROOT}/SHIPPED.txt"

mkdir -p "${DOCS_SHARED_DST}" "${SKILLS_DST_ROOT}"

rm -rf "${ASSETS_DIR}/docs/numereng"
rsync -a --delete "${DOCS_SHARED_SRC}/" "${DOCS_SHARED_DST}/"

rm -rf "${ASSETS_DIR}/docs/numerai"

find "${SKILLS_DST_ROOT}" -mindepth 1 -maxdepth 1 ! -name 'SHIPPED.txt' -exec rm -rf {} +
cp "${MANIFEST_PATH}" "${SKILLS_DST_ROOT}/SHIPPED.txt"

grep -E '^[A-Za-z0-9_-]+$' "${MANIFEST_PATH}" \
  | awk '!seen[$0]++' \
  | while read -r skill_id; do
      rsync -a --delete "${SKILLS_SRC_ROOT}/${skill_id}/" "${SKILLS_DST_ROOT}/${skill_id}/"
    done
