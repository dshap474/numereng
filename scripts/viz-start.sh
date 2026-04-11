#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
VIZ_DIR="${VIZ_DIR:-$ROOT_DIR/viz}"
VIZ_WEB="${VIZ_WEB:-$VIZ_DIR/web}"
API_PORT="${API_PORT:-8502}"
VITE_PORT="${VITE_PORT:-5173}"
API_PID_FILE="${API_PID_FILE:-$VIZ_DIR/api.pid}"
VITE_PID_FILE="${VITE_PID_FILE:-$VIZ_DIR/vite.pid}"

"$ROOT_DIR/scripts/viz-stop.sh"

mkdir -p "$VIZ_DIR"

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
	nohup npm run dev -- --host 127.0.0.1 --port "$VITE_PORT" > "$VIZ_DIR/vite.log" 2>&1 &
	echo $! > "$VITE_PID_FILE"
)

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
echo "Viz running - logs: viz/bootstrap.log, viz/api.log, viz/vite.log"
