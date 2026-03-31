#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
VIZ_DIR="${VIZ_DIR:-$ROOT_DIR/viz}"
API_PORT="${API_PORT:-8502}"
VITE_PORT="${VITE_PORT:-5173}"
API_PID_FILE="${API_PID_FILE:-$VIZ_DIR/api.pid}"
VITE_PID_FILE="${VITE_PID_FILE:-$VIZ_DIR/vite.pid}"

for pid_file in "$API_PID_FILE" "$VITE_PID_FILE"; do
	if [ -f "$pid_file" ]; then
		pid=$(cat "$pid_file" 2>/dev/null || true)
		if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
			kill "$pid" 2>/dev/null || true
		fi
		rm -f "$pid_file"
	fi
done

lsof -ti:"$API_PORT" 2>/dev/null | xargs kill 2>/dev/null || true
lsof -ti:"$VITE_PORT" 2>/dev/null | xargs kill 2>/dev/null || true

for port in "$API_PORT" "$VITE_PORT"; do
	attempt=0
	while lsof -ti:"$port" >/dev/null 2>&1; do
		attempt=$((attempt + 1))
		if [ "$attempt" -gt 20 ]; then
			lsof -ti:"$port" 2>/dev/null | xargs kill -9 2>/dev/null || true
			break
		fi
		sleep 0.2
	done
done

echo "Viz servers stopped"
