#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs/thousand_paper_suite}"
SCENARIO_ROOT="${SCENARIO_ROOT:-${PROJECT_DIR}/runs/thousand_coalition_shared/scenario}"
DATA_ROOT="${DATA_ROOT:-${RUN_ROOT}/catalogs}"
CONFIG="${CONFIG:-${PROJECT_DIR}/configs/thousand_paper_suite.json}"
PROFILE="${PROFILE:-paper}"
STAGE="${STAGE:-all}"
GPU_IDS="${GPU_IDS:-0,1}"
METHODS="${METHODS:-}"
SEEDS="${SEEDS:-}"
RESUME="${RESUME:-1}"
START_MONITORING="${START_MONITORING:-1}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6012}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8770}"

cd "${PROJECT_DIR}"
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/tensorboard"

MONITOR_PIDS=()
cleanup() {
  if [[ ${#MONITOR_PIDS[@]} -gt 0 ]]; then
    kill "${MONITOR_PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "${START_MONITORING}" == "1" ]]; then
  if ! command -v tensorboard >/dev/null 2>&1; then
    echo "TensorBoard is unavailable. Install with python -m pip install -e '.[monitoring]' --no-build-isolation" >&2
    exit 2
  fi
  for port in "${TENSORBOARD_PORT}" "${DASHBOARD_PORT}"; do
    if ss -lnt 2>/dev/null | grep -q ":${port} "; then
      echo "Monitoring port ${port} is already in use." >&2
      exit 3
    fi
  done

  python -u scripts/metrics_to_tensorboard.py \
    --run-root "${RUN_ROOT}" \
    --output-dir "${RUN_ROOT}/tensorboard" \
    --poll-seconds 5 \
    >"${RUN_ROOT}/logs/tensorboard_bridge.log" 2>&1 &
  MONITOR_PIDS+=("$!")

  tensorboard --logdir "${RUN_ROOT}/tensorboard" --host 127.0.0.1 \
    --port "${TENSORBOARD_PORT}" \
    >"${RUN_ROOT}/logs/tensorboard.log" 2>&1 &
  MONITOR_PIDS+=("$!")

  python -u scripts/serve_dashboard.py --host 127.0.0.1 --port "${DASHBOARD_PORT}" \
    >"${RUN_ROOT}/logs/dashboard.log" 2>&1 &
  MONITOR_PIDS+=("$!")

  RUN_RELATIVE="${RUN_ROOT#"${PROJECT_DIR}"/}"
  echo "Dashboard:   http://127.0.0.1:${DASHBOARD_PORT}/web/fast_slow.html?run=/${RUN_RELATIVE}"
  echo "TensorBoard: http://127.0.0.1:${TENSORBOARD_PORT}"
  echo "SSH tunnel:  ssh -L ${DASHBOARD_PORT}:127.0.0.1:${DASHBOARD_PORT} -L ${TENSORBOARD_PORT}:127.0.0.1:${TENSORBOARD_PORT} USER@SERVER"
fi

ARGS=(
  --config "${CONFIG}"
  --run-root "${RUN_ROOT}"
  --scenario-root "${SCENARIO_ROOT}"
  --data-root "${DATA_ROOT}"
  --profile "${PROFILE}"
  --stage "${STAGE}"
  --gpus "${GPU_IDS}"
)
if [[ -n "${METHODS}" ]]; then
  ARGS+=(--methods "${METHODS}")
fi
if [[ -n "${SEEDS}" ]]; then
  ARGS+=(--seeds "${SEEDS}")
fi
if [[ "${RESUME}" == "1" ]]; then
  ARGS+=(--resume)
fi

python -u scripts/run_thousand_paper_suite.py "${ARGS[@]}" \
  2>&1 | tee -a "${RUN_ROOT}/logs/suite.log"
