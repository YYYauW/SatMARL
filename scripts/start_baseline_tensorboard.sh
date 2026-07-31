#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOTS="${RUN_ROOTS:-${PROJECT_DIR}/runs/rl64_seed701 ${PROJECT_DIR}/runs/rl64_seed702 ${PROJECT_DIR}/runs/rl64_seed703}"
EVENT_ROOT="${EVENT_ROOT:-${PROJECT_DIR}/runs/baseline_tensorboard}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-6007}"
POLL_SECONDS="${POLL_SECONDS:-5}"

cd "${PROJECT_DIR}"
mkdir -p "${EVENT_ROOT}" "${EVENT_ROOT}/logs"

if ! command -v tensorboard >/dev/null 2>&1; then
  echo "TensorBoard is not installed. Run: python -m pip install -e '.[monitoring]'" >&2
  exit 1
fi

read -r -a ROOT_ARRAY <<< "${RUN_ROOTS}"
BRIDGE_ARGS=()
for root in "${ROOT_ARRAY[@]}"; do
  BRIDGE_ARGS+=(--run-root "${root}")
done

python -u scripts/metrics_to_tensorboard.py \
  "${BRIDGE_ARGS[@]}" \
  --output-dir "${EVENT_ROOT}" \
  --poll-seconds "${POLL_SECONDS}" \
  >"${EVENT_ROOT}/logs/bridge.log" 2>&1 &
BRIDGE_PID=$!

tensorboard --logdir "${EVENT_ROOT}" --host "${HOST}" --port "${PORT}" \
  >"${EVENT_ROOT}/logs/tensorboard.log" 2>&1 &
TENSORBOARD_PID=$!

trap 'kill "${BRIDGE_PID}" "${TENSORBOARD_PID}" 2>/dev/null || true' EXIT INT TERM

echo "Metrics bridge PID: ${BRIDGE_PID}"
echo "Baseline TensorBoard PID: ${TENSORBOARD_PID}"
echo "Baseline TensorBoard: http://127.0.0.1:${PORT}"
echo "Input roots: ${RUN_ROOTS}"
echo "Event root: ${EVENT_ROOT}"
wait
