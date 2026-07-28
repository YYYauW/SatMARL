#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs/point_aaai_final}"
PHASES="${PHASES:-core,ablations}"
SEEDS="${SEEDS:-701,702,703}"
GPU_IDS="${GPU_IDS:-0,1}"
EPISODES="${EPISODES:-300}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6010}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8766}"
RESUME="${RESUME:-1}"
FORCE_REBUILD_SCENARIO="${FORCE_REBUILD_SCENARIO:-0}"
TRAIN_CATALOG="${TRAIN_CATALOG:-${PROJECT_DIR}/data/targets/train_requests.csv}"
EVAL_CATALOG="${EVAL_CATALOG:-${PROJECT_DIR}/data/targets/test_requests.csv}"

cd "${PROJECT_DIR}"
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/tensorboard"

for required in "${TRAIN_CATALOG}" "${EVAL_CATALOG}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Required point-target catalog not found: ${required}" >&2
    exit 2
  fi
done
if ! command -v tensorboard >/dev/null 2>&1; then
  echo "TensorBoard is unavailable. Install with: python -m pip install tensorboard" >&2
  exit 2
fi
for port in "${TENSORBOARD_PORT}" "${DASHBOARD_PORT}"; do
  if ss -lnt 2>/dev/null | grep -q ":${port} "; then
    echo "Port ${port} is already in use. Set a different TENSORBOARD_PORT or DASHBOARD_PORT." >&2
    exit 3
  fi
done

python -u scripts/metrics_to_tensorboard.py \
  --run-root "${RUN_ROOT}" \
  --output-dir "${RUN_ROOT}/tensorboard" \
  --poll-seconds 5 \
  >"${RUN_ROOT}/logs/tensorboard_bridge.log" 2>&1 &
BRIDGE_PID=$!

tensorboard --logdir "${RUN_ROOT}/tensorboard" --host 127.0.0.1 \
  --port "${TENSORBOARD_PORT}" \
  >"${RUN_ROOT}/logs/tensorboard.log" 2>&1 &
TENSORBOARD_PID=$!

python -u scripts/serve_dashboard.py --host 127.0.0.1 --port "${DASHBOARD_PORT}" \
  >"${RUN_ROOT}/logs/dashboard.log" 2>&1 &
DASHBOARD_PID=$!

cleanup() {
  kill "${BRIDGE_PID}" "${TENSORBOARD_PID}" "${DASHBOARD_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

ARGS=(
  --run-root "${RUN_ROOT}"
  --phases "${PHASES}"
  --seeds "${SEEDS}"
  --gpu-ids "${GPU_IDS}"
  --episodes "${EPISODES}"
  --eval-episodes "${EVAL_EPISODES}"
  --train-catalog "${TRAIN_CATALOG}"
  --eval-catalog "${EVAL_CATALOG}"
  --tensorboard-port "${TENSORBOARD_PORT}"
)
if [[ "${RESUME}" == "1" ]]; then
  ARGS+=(--resume)
fi
if [[ "${FORCE_REBUILD_SCENARIO}" == "1" ]]; then
  ARGS+=(--force-rebuild-scenario)
fi

echo "Point-target AAAI suite: ${RUN_ROOT}"
echo "Methods/phases:          ${PHASES}"
echo "Training seeds:          ${SEEDS}"
echo "GPU workers:             ${GPU_IDS}"
echo "HTML dashboard:          http://127.0.0.1:${DASHBOARD_PORT}/web/fast_slow.html?run=/runs/$(basename "${RUN_ROOT}")"
echo "TensorBoard:             http://127.0.0.1:${TENSORBOARD_PORT}"

python -u scripts/run_point_aaai_suite.py "${ARGS[@]}" \
  2>&1 | tee -a "${RUN_ROOT}/logs/suite.log"
