.PHONY: ci fmt test test-all hpo-smoke viz viz-start kill-viz oss-preflight security readiness deps-lint arch-lint

API_PORT ?= 8502
VITE_PORT ?= 5173
VIZ_DIR := $(CURDIR)/viz
VIZ_WEB := $(VIZ_DIR)/web
API_PID_FILE := $(VIZ_DIR)/api.pid
VITE_PID_FILE := $(VIZ_DIR)/vite.pid

ci: security readiness test

fmt:
	uv run ruff check --fix-only .
	uv run ruff format .

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

readiness: deps-lint arch-lint
	@test -f AGENTS.md
	@test -f .python-version
	@test -f justfile
	@test -f .devcontainer/devcontainer.json
	@find runbooks -maxdepth 1 -type f | grep -q .

security: oss-preflight
	uv run pip-audit --ignore-vuln CVE-2026-4539

oss-preflight:
	@echo "Running OSS preflight checks..."
	@forbidden=$$(git ls-files --others --exclude-standard | rg '^(docs/numerai/forum/|viz/.*\.pid$$|.*__pycache__/|.*\.py[co]$$)' || true); \
	if [ -n "$$forbidden" ]; then \
		echo "FAIL: non-ignored generated/local files detected:"; \
		echo "$$forbidden"; \
		exit 1; \
	fi
	@secret_hits=$$(git grep -nE \
		-e 'AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}' \
		-e 'gh[pousr]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{60,}' \
		-e 'sk-[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z\-_]{35}' \
		-e '-----BEGIN (RSA|EC|DSA|OPENSSH|PGP|PRIVATE) KEY-----' \
		-- . || true); \
	if [ -n "$$secret_hits" ]; then \
		echo "FAIL: high-confidence secret patterns found in tracked files:"; \
		echo "$$secret_hits"; \
		exit 1; \
	fi
	@if git ls-files --error-unmatch .env >/dev/null 2>&1; then \
		echo "FAIL: .env is tracked and must remain local-only."; \
		exit 1; \
	fi
	@echo "PASS: OSS preflight checks passed."

kill-viz:
	@./scripts/viz-stop.sh

viz:
	@echo "DEPRECATED: use 'just viz' instead of 'make viz'."
	@just viz

viz-start:
	@./scripts/viz-start.sh
