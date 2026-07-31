#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs/fast_slow_aaai}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8766}"
TENSORBOARD_HOST="${TENSORBOARD_HOST:-127.0.0.1}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"

cd "${PROJECT_DIR}"
mkdir -p "${RUN_ROOT}/logs"

if ! command -v tensorboard >/dev/null 2>&1; then
  echo "TensorBoard is not installed. Run: python -m pip install -e '.[monitoring]'" >&2
  exit 1
fi

python -u scripts/serve_dashboard.py --host "${HOST}" --port "${PORT}" \
  >"${RUN_ROOT}/logs/dashboard.log" 2>&1 &
DASHBOARD_PID=$!
tensorboard --logdir "${RUN_ROOT}/training" --host "${TENSORBOARD_HOST}" \
  --port "${TENSORBOARD_PORT}" \
  >"${RUN_ROOT}/logs/tensorboard.log" 2>&1 &
TENSORBOARD_PID=$!
trap 'kill "${DASHBOARD_PID}" "${TENSORBOARD_PID}" 2>/dev/null || true' EXIT INT TERM

echo "HTML dashboard: http://127.0.0.1:${PORT}/web/fast_slow.html"
echo "TensorBoard:     http://127.0.0.1:${TENSORBOARD_PORT}"
echo "Run root:        ${RUN_ROOT}"
wait
