#!/usr/bin/env sh
# Stop every local dev process started by scripts/dev-start.sh (API + Vite).
#
# USAGE:
#   ./scripts/dev-stop.sh
#   just kill
#
# Kills the recorded PIDs, then anything still holding the recorded or default
# ports. Safe to run when nothing is running.

set -eu

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

ROOT_DIR=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
VIZ_DIR="${VIZ_DIR:-$ROOT_DIR/viz}"
API_PORT="${API_PORT:-8502}"
VITE_PORT="${VITE_PORT:-5173}"
API_PID_FILE="${API_PID_FILE:-$VIZ_DIR/api.pid}"
VITE_PID_FILE="${VITE_PID_FILE:-$VIZ_DIR/vite.pid}"
PORTS_FILE="${PORTS_FILE:-$VIZ_DIR/ports.env}"

# --------------------------------------------------------------------------- #
# Shutdown
# --------------------------------------------------------------------------- #

# Only ports we recorded ourselves are killed, so an unrelated process squatting
# on a default port is left alone (dev-start falls forward to the next free one).
PORTS=""
if [ -f "$PORTS_FILE" ]; then
	# shellcheck disable=SC1090
	. "$PORTS_FILE"
	PORTS="${DEV_API_PORT:-} ${DEV_VITE_PORT:-}"
elif [ -f "$API_PID_FILE" ] || [ -f "$VITE_PID_FILE" ]; then
	# Pre-ports.env run: fall back to the defaults it would have used.
	PORTS="$API_PORT $VITE_PORT"
fi

for pid_file in "$API_PID_FILE" "$VITE_PID_FILE"; do
	if [ -f "$pid_file" ]; then
		pid=$(cat "$pid_file" 2>/dev/null || true)
		if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
			kill "$pid" 2>/dev/null || true
		fi
		rm -f "$pid_file"
	fi
done

for port in $PORTS; do
	lsof -ti:"$port" 2>/dev/null | xargs kill 2>/dev/null || true
done

for port in $PORTS; do
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

rm -f "$PORTS_FILE"

echo "Dev servers stopped"
