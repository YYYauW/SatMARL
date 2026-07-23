#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TLE_FILE="${TLE_FILE:?Set TLE_FILE to an archived three-line TLE catalog.}"
TARGET_CSV="${TARGET_CSV:?Set TARGET_CSV to the task catalog CSV.}"
START_UTC="${START_UTC:?Set START_UTC to an ISO-8601 UTC timestamp near the TLE epoch.}"

RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs/real_orbit_aaai}"
SATELLITES="${SATELLITES:-64}"
TASKS="${TASKS:-3072}"
PLANES="${PLANES:-8}"
MAX_STEPS="${MAX_STEPS:-240}"
LOOKAHEAD_STEPS="${LOOKAHEAD_STEPS:-30}"
STEP_SECONDS="${STEP_SECONDS:-30}"
EPISODES="${EPISODES:-300}"
SEED="${SEED:-701}"
GPU_ID="${GPU_ID:-0}"
ARCHITECTURE="${ARCHITECTURE:-fast_slow_graph}"
SLOW_INTERVAL="${SLOW_INTERVAL:-8}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"
START_MONITORING="${START_MONITORING:-1}"
RESUME="${RESUME:-0}"

CACHE="${RUN_ROOT}/scenario/ephemeris.npz"
REPORT="${RUN_ROOT}/scenario/scenario_report.json"
TRAIN_DIR="${RUN_ROOT}/training/${ARCHITECTURE}/seed_${SEED}"

cd "${PROJECT_DIR}"
mkdir -p "${RUN_ROOT}/scenario" "${RUN_ROOT}/logs" "${TRAIN_DIR}"

if [[ ! -f "${CACHE}" ]]; then
  python -u scripts/build_tle_ephemeris.py \
    --tle "${TLE_FILE}" \
    --output "${CACHE}" \
    --start-utc "${START_UTC}" \
    --satellites "${SATELLITES}" \
    --max-steps "${MAX_STEPS}" \
    --lookahead-steps "${LOOKAHEAD_STEPS}" \
    --step-duration-seconds "${STEP_SECONDS}" \
    2>&1 | tee "${RUN_ROOT}/logs/build_ephemeris.log"
fi

python -u scripts/validate_real_scenario.py \
  --ephemeris-cache "${CACHE}" \
  --task-catalog "${TARGET_CSV}" \
  --satellites "${SATELLITES}" \
  --tasks "${TASKS}" \
  --max-steps "${MAX_STEPS}" \
  --lookahead-steps "${LOOKAHEAD_STEPS}" \
  --step-duration-seconds "${STEP_SECONDS}" \
  --min-elevation-deg 3 \
  --max-off-nadir-deg 45 \
  --output "${REPORT}" \
  2>&1 | tee "${RUN_ROOT}/logs/validate_scenario.log"

MONITOR_PIDS=()
if [[ "${START_MONITORING}" == "1" ]]; then
  python -u scripts/serve_dashboard.py --host 127.0.0.1 --port 8766 \
    >"${RUN_ROOT}/logs/dashboard.log" 2>&1 &
  MONITOR_PIDS+=("$!")
  tensorboard --logdir "${RUN_ROOT}/training" --host 127.0.0.1 \
    --port "${TENSORBOARD_PORT}" \
    >"${RUN_ROOT}/logs/tensorboard.log" 2>&1 &
  MONITOR_PIDS+=("$!")
  trap 'kill "${MONITOR_PIDS[@]}" 2>/dev/null || true' EXIT INT TERM
  echo "HTML file server: http://127.0.0.1:8766"
  echo "TensorBoard:      http://127.0.0.1:${TENSORBOARD_PORT}"
fi

TRAIN_ARGS=(
  --architecture "${ARCHITECTURE}"
  --slow-interval "${SLOW_INTERVAL}"
  --satellites "${SATELLITES}"
  --tasks "${TASKS}"
  --planes "${PLANES}"
  --max-steps "${MAX_STEPS}"
  --planning-lookahead-steps "${LOOKAHEAD_STEPS}"
  --candidate-k 24
  --neighbor-k 6
  --episodes "${EPISODES}"
  --seed "${SEED}"
  --task-layout catalog
  --ephemeris-cache "${CACHE}"
  --task-catalog "${TARGET_CSV}"
  --step-duration-seconds "${STEP_SECONDS}"
  --point-observation-seconds 5
  --fov-deg 45
  --max-off-nadir-deg 45
  --semantic-opportunity-balancing
  --hidden-dim 128
  --min-decision-samples 1024
  --max-buffered-episodes 8
  --checkpoint-every 10
  --run-dir "${TRAIN_DIR}"
  --tensorboard
  --device cuda
)
if [[ "${RESUME}" == "1" ]]; then
  TRAIN_ARGS+=(--resume)
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  python -u scripts/train_oasis.py "${TRAIN_ARGS[@]}" \
  2>&1 | tee -a "${RUN_ROOT}/logs/train_${ARCHITECTURE}_seed_${SEED}.log"
