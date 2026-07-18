#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs/fast_slow_aaai}"
DEVICE="${DEVICE:-cuda}"
PROFILE="${PROFILE:-paper}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8766}"

cd "${PROJECT_DIR}"
mkdir -p "${RUN_ROOT}/logs"

python -u scripts/serve_dashboard.py --host "${HOST}" --port "${PORT}" \
  >"${RUN_ROOT}/logs/dashboard.log" 2>&1 &
DASHBOARD_PID=$!
trap 'kill "${DASHBOARD_PID}" 2>/dev/null || true' EXIT INT TERM

echo "Dashboard PID: ${DASHBOARD_PID}"
echo "Open through an SSH tunnel: http://127.0.0.1:${PORT}/web/fast_slow.html"
echo "Training profile=${PROFILE}, device=${DEVICE}, run_root=${RUN_ROOT}"

python -u scripts/run_fast_slow_server_pipeline.py \
  --run-root "${RUN_ROOT}" \
  --profile "${PROFILE}" \
  --device "${DEVICE}" \
  2>&1 | tee -a "${RUN_ROOT}/logs/pipeline_console.log"
