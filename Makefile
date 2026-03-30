.PHONY: ci fmt test test-all viz kill-viz oss-preflight security readiness deps-lint arch-lint

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
	@for pid_file in "$(API_PID_FILE)" "$(VITE_PID_FILE)"; do \
		if [ -f "$$pid_file" ]; then \
			pid=$$(cat "$$pid_file" 2>/dev/null || true); \
			if [ -n "$$pid" ] && kill -0 "$$pid" 2>/dev/null; then \
				kill "$$pid" 2>/dev/null || true; \
			fi; \
			rm -f "$$pid_file"; \
		fi; \
	done
	@lsof -ti:$(API_PORT) 2>/dev/null | xargs kill 2>/dev/null || true
	@lsof -ti:$(VITE_PORT) 2>/dev/null | xargs kill 2>/dev/null || true
	@for port in $(API_PORT) $(VITE_PORT); do \
		attempt=0; \
		while lsof -ti:$$port >/dev/null 2>&1; do \
			attempt=$$((attempt + 1)); \
			if [ $$attempt -gt 20 ]; then \
				lsof -ti:$$port 2>/dev/null | xargs kill -9 2>/dev/null || true; \
				break; \
			fi; \
			sleep 0.2; \
		done; \
	done
	@echo "Viz servers stopped"

viz: kill-viz
	@mkdir -p $(VIZ_DIR)
	@if [ ! -d $(VIZ_WEB)/node_modules ]; then \
		echo "Installing npm dependencies..."; \
		cd $(VIZ_WEB) && npm install --include=dev; \
	fi
	@rm -f "$(VIZ_DIR)/bootstrap.log" "$(VIZ_DIR)/api.log" "$(VIZ_DIR)/vite.log"
	@cd $(CURDIR); \
	uv run numereng remote bootstrap-viz --store-root "$(CURDIR)/.numereng" > "$(VIZ_DIR)/bootstrap.log" 2>&1; \
	status=$$?; \
	cat "$(VIZ_DIR)/bootstrap.log"; \
	exit $$status
	@cd $(CURDIR); nohup uv run python -m uvicorn viz.api:app --host 127.0.0.1 --port $(API_PORT) \
		> "$(VIZ_DIR)/api.log" 2>&1 & echo $$! > "$(API_PID_FILE)"
	@cd $(VIZ_WEB); nohup npm run dev -- --host 127.0.0.1 --port $(VITE_PORT) \
		> "$(VIZ_DIR)/vite.log" 2>&1 & echo $$! > "$(VITE_PID_FILE)"
	@attempt=0; \
	until curl -fsS "http://127.0.0.1:$(API_PORT)/healthz" >/dev/null 2>&1; do \
		attempt=$$((attempt + 1)); \
		if [ $$attempt -gt 50 ]; then \
			echo "API failed to start on port $(API_PORT)"; \
			tail -n 80 "$(VIZ_DIR)/api.log" || true; \
			exit 1; \
		fi; \
		sleep 0.2; \
	done
	@attempt=0; \
	until curl -fsS "http://127.0.0.1:$(VITE_PORT)" >/dev/null 2>&1; do \
		attempt=$$((attempt + 1)); \
		if [ $$attempt -gt 80 ]; then \
			echo "Vite failed to start on port $(VITE_PORT)"; \
			tail -n 80 "$(VIZ_DIR)/vite.log" || true; \
			exit 1; \
		fi; \
		sleep 0.2; \
	done
	@echo "API started (pid $$(cat "$(API_PID_FILE)")) on http://127.0.0.1:$(API_PORT)"
	@echo "Vite started (pid $$(cat "$(VITE_PID_FILE)")) on http://127.0.0.1:$(VITE_PORT)"
	@echo "Viz running - logs: viz/bootstrap.log, viz/api.log, viz/vite.log"
