#!/usr/bin/env sh
# Start the local dev stack (read-only viz API + Vite web app), idempotently.
#
# USAGE:
#   ./scripts/dev-start.sh
#   just dev
#
# Always stops any previously started dev processes first, then binds the first
# free port at or above the defaults (API 8502, Vite 5173). The chosen ports are
# recorded in viz/ports.env so `just kill` can shut the same processes down.

set -eu

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

ROOT_DIR=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
VIZ_DIR="${VIZ_DIR:-$ROOT_DIR/viz}"
VIZ_WEB="${VIZ_WEB:-$VIZ_DIR/web}"
API_PORT="${API_PORT:-8502}"
VITE_PORT="${VITE_PORT:-5173}"
API_PID_FILE="${API_PID_FILE:-$VIZ_DIR/api.pid}"
VITE_PID_FILE="${VITE_PID_FILE:-$VIZ_DIR/vite.pid}"
PORTS_FILE="${PORTS_FILE:-$VIZ_DIR/ports.env}"
PORT_SCAN_LIMIT="${PORT_SCAN_LIMIT:-50}"

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

# Echo the first free port at or above $1, scanning at most PORT_SCAN_LIMIT ports.
find_free_port() {
	port=$1
	tried=0
	while lsof -ti:"$port" >/dev/null 2>&1; do
		tried=$((tried + 1))
		if [ "$tried" -ge "$PORT_SCAN_LIMIT" ]; then
			echo "No free port found in range $1-$port" >&2
			exit 1
		fi
		port=$((port + 1))
	done
	echo "$port"
}

# --------------------------------------------------------------------------- #
# Restart
# --------------------------------------------------------------------------- #

"$ROOT_DIR/scripts/dev-stop.sh"

mkdir -p "$VIZ_DIR"

API_PORT=$(find_free_port "$API_PORT")
VITE_PORT=$(find_free_port "$VITE_PORT")

printf 'DEV_API_PORT=%s\nDEV_VITE_PORT=%s\n' "$API_PORT" "$VITE_PORT" > "$PORTS_FILE"

if [ ! -d "$VIZ_WEB/node_modules" ]; then
	echo "Installing npm dependencies..."
	(
		cd "$VIZ_WEB"
		npm install --include=dev
	)
fi

rm -f "$VIZ_DIR/bootstrap.log" "$VIZ_DIR/api.log" "$VIZ_DIR/vite.log"

(
	cd "$ROOT_DIR"
	uv run numereng remote bootstrap-viz --workspace "$ROOT_DIR" > "$VIZ_DIR/bootstrap.log" 2>&1
)
cat "$VIZ_DIR/bootstrap.log"

(
	cd "$ROOT_DIR"
	nohup uv run python -m uvicorn viz.api:app --host 127.0.0.1 --port "$API_PORT" > "$VIZ_DIR/api.log" 2>&1 &
	echo $! > "$API_PID_FILE"
)

(
	cd "$VIZ_WEB"
	export VIZ_API_PORT="$API_PORT"
	nohup npm run dev -- --host 127.0.0.1 --port "$VITE_PORT" --strictPort > "$VIZ_DIR/vite.log" 2>&1 &
	echo $! > "$VITE_PID_FILE"
)

# --------------------------------------------------------------------------- #
# Readiness
# --------------------------------------------------------------------------- #

attempt=0
until curl -fsS "http://127.0.0.1:$API_PORT/healthz" >/dev/null 2>&1; do
	attempt=$((attempt + 1))
	if [ "$attempt" -gt 50 ]; then
		echo "API failed to start on port $API_PORT"
		tail -n 80 "$VIZ_DIR/api.log" || true
		exit 1
	fi
	sleep 0.2
done

attempt=0
until curl -fsS "http://127.0.0.1:$VITE_PORT" >/dev/null 2>&1; do
	attempt=$((attempt + 1))
	if [ "$attempt" -gt 80 ]; then
		echo "Vite failed to start on port $VITE_PORT"
		tail -n 80 "$VIZ_DIR/vite.log" || true
		exit 1
	fi
	sleep 0.2
done

echo "API started (pid $(cat "$API_PID_FILE")) on http://127.0.0.1:$API_PORT"
echo "Vite started (pid $(cat "$VITE_PID_FILE")) on http://127.0.0.1:$VITE_PORT"
echo "Dev stack running - logs: viz/bootstrap.log, viz/api.log, viz/vite.log"
