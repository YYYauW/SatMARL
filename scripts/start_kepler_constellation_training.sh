#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_CSV="${TARGET_CSV:?Set TARGET_CSV to the task catalog CSV.}"
EVAL_TARGET_CSV="${EVAL_TARGET_CSV:-${TARGET_CSV}}"

if [[ ! -f "${TARGET_CSV}" ]]; then
  echo "Training task catalog not found: ${TARGET_CSV}" >&2
  echo "Generate the standard catalogs with:" >&2
  echo "  python scripts/generate_task_catalogs.py --output-dir data/targets \\" >&2
  echo "    --area-fraction 0.75 --file-prefix area_ --overwrite" >&2
  exit 2
fi
if [[ ! -f "${EVAL_TARGET_CSV}" ]]; then
  echo "Evaluation task catalog not found: ${EVAL_TARGET_CSV}" >&2
  exit 2
fi

RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs/kepler_aaai}"
START_UTC="${START_UTC:-2026-07-20T00:00:00Z}"
SATELLITES="${SATELLITES:-64}"
TASKS="${TASKS:-3072}"
EVAL_TASKS="${EVAL_TASKS:-${TASKS}}"
PLANES="${PLANES:-8}"
MAX_STEPS="${MAX_STEPS:-240}"
LOOKAHEAD_STEPS="${LOOKAHEAD_STEPS:-30}"
STEP_SECONDS="${STEP_SECONDS:-30}"
ALTITUDE_KM="${ALTITUDE_KM:-550}"
ECCENTRICITY="${ECCENTRICITY:-0.001}"
INCLINATION_DEG="${INCLINATION_DEG:-97.6}"
WALKER_PHASING="${WALKER_PHASING:-1}"
RAAN_OFFSET_DEG="${RAAN_OFFSET_DEG:-0}"
ARGUMENT_OF_PERIGEE_DEG="${ARGUMENT_OF_PERIGEE_DEG:-0}"
MEAN_ANOMALY_OFFSET_DEG="${MEAN_ANOMALY_OFFSET_DEG:-0}"
ELEMENTS_CSV="${ELEMENTS_CSV:-}"
EPISODES="${EPISODES:-300}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
EVAL_SEED="${EVAL_SEED:-1701}"
SEED="${SEED:-701}"
GPU_ID="${GPU_ID:-0}"
ARCHITECTURE="${ARCHITECTURE:-fast_slow_graph}"
SLOW_INTERVAL="${SLOW_INTERVAL:-8}"
AREA_TASK_FRACTION="${AREA_TASK_FRACTION:-0.75}"
AREA_OBSERVATION_SECONDS="${AREA_OBSERVATION_SECONDS:-20}"
AREA_COVERAGE_THRESHOLD="${AREA_COVERAGE_THRESHOLD:-0.95}"
AREA_COVERAGE_SAMPLES="${AREA_COVERAGE_SAMPLES:-512}"
AREA_MIN_MARGINAL_COVERAGE="${AREA_MIN_MARGINAL_COVERAGE:-0.01}"
AREA_MIN_COOPERATIVE_SATELLITES="${AREA_MIN_COOPERATIVE_SATELLITES:-2}"
AREA_MAX_REQUIRED_STRIPS="${AREA_MAX_REQUIRED_STRIPS:-8}"
AREA_OUTSIDE_PENALTY_WEIGHT="${AREA_OUTSIDE_PENALTY_WEIGHT:-1.0}"
AREA_REDUNDANCY_PENALTY_WEIGHT="${AREA_REDUNDANCY_PENALTY_WEIGHT:-0.6}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"
START_MONITORING="${START_MONITORING:-1}"
REBUILD_SCENARIO="${REBUILD_SCENARIO:-0}"
RESUME="${RESUME:-0}"

CACHE="${RUN_ROOT}/scenario/kepler_ephemeris.npz"
ELEMENTS_OUTPUT="${RUN_ROOT}/scenario/orbital_elements.csv"
REPORT="${RUN_ROOT}/scenario/scenario_report.json"
TRAIN_DIR="${RUN_ROOT}/training/${ARCHITECTURE}/seed_${SEED}"

cd "${PROJECT_DIR}"
mkdir -p "${RUN_ROOT}/scenario" "${RUN_ROOT}/logs" "${TRAIN_DIR}"

if [[ "${RUN_ROOT}" == "${PROJECT_DIR}"/* ]]; then
  METRICS_URL="/${TRAIN_DIR#${PROJECT_DIR}/}/metrics.json"
  DASHBOARD_RUN="/${RUN_ROOT#${PROJECT_DIR}/}"
else
  METRICS_URL="${TRAIN_DIR}/metrics.json"
  DASHBOARD_RUN="${RUN_ROOT}"
fi
python scripts/update_pipeline_status.py \
  --root "${RUN_ROOT}" --status running --stage preparing \
  --message "Building/validating the six-element constellation and area catalogs." \
  --method "${ARCHITECTURE}" --seed "${SEED}" \
  --metrics-url "${METRICS_URL}" \
  --tensorboard-url "http://127.0.0.1:${TENSORBOARD_PORT}"

if [[ "${REBUILD_SCENARIO}" == "1" ]]; then
  rm -f "${CACHE}" "${ELEMENTS_OUTPUT}"
fi

if [[ ! -f "${CACHE}" ]]; then
  BUILD_ARGS=(
    --output "${CACHE}"
    --elements-output "${ELEMENTS_OUTPUT}"
    --start-utc "${START_UTC}"
    --satellites "${SATELLITES}"
    --planes "${PLANES}"
    --max-steps "${MAX_STEPS}"
    --lookahead-steps "${LOOKAHEAD_STEPS}"
    --step-duration-seconds "${STEP_SECONDS}"
  )
  if [[ -n "${ELEMENTS_CSV}" ]]; then
    BUILD_ARGS+=(--elements-csv "${ELEMENTS_CSV}")
  else
    BUILD_ARGS+=(
      --altitude-km "${ALTITUDE_KM}"
      --eccentricity "${ECCENTRICITY}"
      --inclination-deg "${INCLINATION_DEG}"
      --walker-phasing "${WALKER_PHASING}"
      --raan-offset-deg "${RAAN_OFFSET_DEG}"
      --argument-of-perigee-deg "${ARGUMENT_OF_PERIGEE_DEG}"
      --mean-anomaly-offset-deg "${MEAN_ANOMALY_OFFSET_DEG}"
    )
  fi
  python -u scripts/build_kepler_ephemeris.py "${BUILD_ARGS[@]}" \
    2>&1 | tee "${RUN_ROOT}/logs/build_kepler_ephemeris.log"
fi

python -u scripts/validate_orbit_scenario.py \
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

if [[ "${EVAL_TARGET_CSV}" != "${TARGET_CSV}" ]]; then
  python -u scripts/validate_orbit_scenario.py \
    --ephemeris-cache "${CACHE}" \
    --task-catalog "${EVAL_TARGET_CSV}" \
    --satellites "${SATELLITES}" \
    --tasks "${EVAL_TASKS}" \
    --max-steps "${MAX_STEPS}" \
    --lookahead-steps "${LOOKAHEAD_STEPS}" \
    --step-duration-seconds "${STEP_SECONDS}" \
    --min-elevation-deg 3 \
    --max-off-nadir-deg 45 \
    --output "${RUN_ROOT}/scenario/eval_scenario_report.json" \
    2>&1 | tee "${RUN_ROOT}/logs/validate_eval_scenario.log"
fi

MONITOR_PIDS=()
if [[ "${START_MONITORING}" == "1" ]]; then
  if ss -lnt 2>/dev/null | grep -q ":${TENSORBOARD_PORT} "; then
    echo "TensorBoard port ${TENSORBOARD_PORT} is already in use." >&2
    echo "Stop the old TensorBoard or set TENSORBOARD_PORT to another port." >&2
    exit 3
  fi
  python -u scripts/serve_dashboard.py --host 127.0.0.1 --port 8766 \
    >"${RUN_ROOT}/logs/dashboard.log" 2>&1 &
  MONITOR_PIDS+=("$!")
  tensorboard --logdir "${RUN_ROOT}/training" --host 127.0.0.1 \
    --port "${TENSORBOARD_PORT}" \
    >"${RUN_ROOT}/logs/tensorboard.log" 2>&1 &
  MONITOR_PIDS+=("$!")
  trap 'kill "${MONITOR_PIDS[@]}" 2>/dev/null || true' EXIT INT TERM
  echo "HTML file server: http://127.0.0.1:8766"
  echo "Live experiment:  http://127.0.0.1:8766/web/fast_slow.html?run=${DASHBOARD_RUN}"
  echo "TensorBoard:      http://127.0.0.1:${TENSORBOARD_PORT}"
fi

python scripts/update_pipeline_status.py \
  --root "${RUN_ROOT}" --status running --stage training \
  --message "Training ${ARCHITECTURE}; metrics and area coverage refresh live." \
  --method "${ARCHITECTURE}" --seed "${SEED}" \
  --metrics-url "${METRICS_URL}" \
  --tensorboard-url "http://127.0.0.1:${TENSORBOARD_PORT}"

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
  --area-task-fraction "${AREA_TASK_FRACTION}"
  --area-observation-seconds "${AREA_OBSERVATION_SECONDS}"
  --area-coverage-threshold "${AREA_COVERAGE_THRESHOLD}"
  --area-coverage-samples "${AREA_COVERAGE_SAMPLES}"
  --area-min-marginal-coverage "${AREA_MIN_MARGINAL_COVERAGE}"
  --area-min-cooperative-satellites "${AREA_MIN_COOPERATIVE_SATELLITES}"
  --area-max-required-strips "${AREA_MAX_REQUIRED_STRIPS}"
  --area-outside-penalty-weight "${AREA_OUTSIDE_PENALTY_WEIGHT}"
  --area-redundancy-penalty-weight "${AREA_REDUNDANCY_PENALTY_WEIGHT}"
  --fov-deg 45
  --max-off-nadir-deg 45
  --semantic-opportunity-balancing
  --no-adaptive-curriculum
  --hidden-dim 128
  --min-decision-samples 1024
  --max-buffered-episodes 8
  --checkpoint-every 10
  --progress-every-steps 10
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

python scripts/update_pipeline_status.py \
  --root "${RUN_ROOT}" --status running --stage evaluation \
  --message "Training complete; running independent held-out evaluation." \
  --method "${ARCHITECTURE}" --seed "${SEED}" \
  --metrics-url "${METRICS_URL}" \
  --tensorboard-url "http://127.0.0.1:${TENSORBOARD_PORT}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  python -u scripts/evaluate_marl.py \
  --checkpoint "${TRAIN_DIR}/checkpoints/latest.pt" \
  --output "${TRAIN_DIR}/evaluation.json" \
  --satellites "${SATELLITES}" \
  --tasks "${EVAL_TASKS}" \
  --planes "${PLANES}" \
  --max-steps "${MAX_STEPS}" \
  --candidate-k 24 \
  --neighbor-k 6 \
  --ephemeris-cache "${CACHE}" \
  --task-catalog "${EVAL_TARGET_CSV}" \
  --eval-task-layout catalog \
  --seed "${EVAL_SEED}" \
  --eval-episodes "${EVAL_EPISODES}" \
  --no-capture-frames \
  --device cuda \
  2>&1 | tee "${RUN_ROOT}/logs/eval_${ARCHITECTURE}_seed_${SEED}.log"

python scripts/update_pipeline_status.py \
  --root "${RUN_ROOT}" --status complete --stage complete \
  --message "Training and independent evaluation are complete." \
  --method "${ARCHITECTURE}" --seed "${SEED}" \
  --metrics-url "${METRICS_URL}" \
  --tensorboard-url "http://127.0.0.1:${TENSORBOARD_PORT}"
