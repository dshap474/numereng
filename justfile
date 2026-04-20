set shell := ["sh", "-cu"]

bootstrap:
    uv sync --extra dev

fmt:
    uv run ruff check --fix-only .
    uv run ruff format .

lint:
    uv run ruff format --check .
    uv run ruff check .

type:
    uv run ty check

test:
    uv run ruff format --check .
    uv run ruff check .
    uv run ty check
    uv run pytest -q -m "not slow" --cov=numereng --cov=numereng_viz --cov-report=term-missing

test-all:
    uv run ruff format --check .
    uv run ruff check .
    uv run ty check
    uv run pytest -q --cov=numereng --cov=numereng_viz --cov-report=term-missing

hpo-smoke:
    ./scripts/hpo_v2_smoke.sh

deps-lint:
    uv run deptry .

arch-lint:
    uv run lint-imports

readiness:
    just deps-lint
    just arch-lint
    test -f AGENTS.md
    test -f .python-version
    test -f justfile
    test -f docs/project/public-repo-boundary.md
    test -f docs/numerai/SYNC_POLICY.md
    test -f src/numereng/platform/remotes/profiles/.gitignore
    test -f src/numereng/platform/remotes/profiles/README.md
    find docs/project/runbooks -maxdepth 1 -type f | grep -q .

oss-preflight:
    echo "Running OSS preflight checks..."
    forbidden=$(git ls-files --others --exclude-standard | rg '^(docs/numerai/forum/|viz/.*\.pid$|.*__pycache__/|.*\.py[co]$)' || true); \
    if [ -n "$forbidden" ]; then \
        echo "FAIL: non-ignored generated/local files detected:"; \
        echo "$forbidden"; \
        exit 1; \
    fi
    tracked_generated=$(git ls-files | rg '(^|/)(__pycache__/|\\.mypy_cache/|\\.pytest_cache/|\\.ruff_cache/|.*\\.py[co]$|docs/numerai/forum/|viz/.*\\.pid$|docs/numerai/\\.sync-meta\\.json$)' || true); \
    if [ -n "$tracked_generated" ]; then \
        echo "FAIL: generated/cache files are tracked:"; \
        echo "$tracked_generated"; \
        exit 1; \
    fi
    tracked_remote_profiles=$(git ls-files 'src/numereng/platform/remotes/profiles/*.yaml' 'src/numereng/platform/remotes/profiles/*.yml' || true); \
    if [ -n "$tracked_remote_profiles" ]; then \
        echo "FAIL: real remote profile YAML files are tracked:"; \
        echo "$tracked_remote_profiles"; \
        exit 1; \
    fi
    tracked_env_files=$(git ls-files | rg '(^|/)\\.env(\\..+)?$' | rg -v '(^|/)\\.env\\.example$' || true); \
    if [ -n "$tracked_env_files" ]; then \
        echo "FAIL: .env-style files are tracked and must remain local-only:"; \
        echo "$tracked_env_files"; \
        exit 1; \
    fi
    absolute_path_hits=$(git grep -nI -E \
        -e '/Users/[A-Za-z0-9._-]+/' \
        -e '[A-Z]:\\\\Users\\\\[A-Za-z0-9._-]+\\\\' \
        -e "\\b[A-Z][a-z]+'s PC\\b" \
        -- . || true); \
    if [ -n "$absolute_path_hits" ]; then \
        echo "FAIL: machine-specific absolute paths found in tracked text files:"; \
        echo "$absolute_path_hits"; \
        exit 1; \
    fi
    secret_hits=$(git grep -nI -E \
        -e 'AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}' \
        -e 'gh[pousr]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{60,}' \
        -e 'sk-[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z\-_]{35}' \
        -e '-----BEGIN (RSA|EC|DSA|OPENSSH|PGP|PRIVATE) KEY-----' \
        -- . || true); \
    if [ -n "$secret_hits" ]; then \
        echo "FAIL: high-confidence secret patterns found in tracked files:"; \
        echo "$secret_hits"; \
        exit 1; \
    fi
    echo "PASS: OSS preflight checks passed."

security:
    just oss-preflight
    uv run pip-audit --ignore-vuln CVE-2026-4539

build:
    uv build --package numereng --wheel --no-build-logs

kill-viz:
    ./scripts/viz-stop.sh

viz:
    ./scripts/viz-start.sh

ci:
    just security
    just readiness
    just test
