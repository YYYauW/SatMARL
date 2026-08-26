#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs/thousand_coalition}"
SCENARIO_ROOT="${SCENARIO_ROOT:-${RUN_ROOT}/scenario}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data/targets}"
TRAIN_CSV="${TRAIN_CSV:-${DATA_DIR}/thousand_train_requests.csv}"
EVAL_CSV="${EVAL_CSV:-${DATA_DIR}/thousand_test_requests.csv}"
TRAIN_SATELLITES="${TRAIN_SATELLITES:-512}"
TRAIN_PLANES="${TRAIN_PLANES:-16}"
TRAIN_TASKS="${TRAIN_TASKS:-24576}"
EVAL_SATELLITES="${EVAL_SATELLITES:-1024}"
EVAL_PLANES="${EVAL_PLANES:-32}"
EVAL_TASKS="${EVAL_TASKS:-49152}"
MAX_STEPS="${MAX_STEPS:-240}"
LOOKAHEAD_STEPS="${LOOKAHEAD_STEPS:-30}"
EPISODES="${EPISODES:-300}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
SEED="${SEED:-701}"
EVAL_SEED="${EVAL_SEED:-1701}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6012}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8770}"
START_MONITORING="${START_MONITORING:-1}"
REBUILD_SCENARIO="${REBUILD_SCENARIO:-0}"
GENERATE_CATALOGS="${GENERATE_CATALOGS:-1}"
RESUME="${RESUME:-0}"
VARIANT="${VARIANT:-full}"

TRAIN_CACHE="${SCENARIO_ROOT}/kepler_n${TRAIN_SATELLITES}.npz"
EVAL_CACHE="${SCENARIO_ROOT}/kepler_n${EVAL_SATELLITES}.npz"
TRAIN_DIR="${RUN_ROOT}/training/${VARIANT}/seed_${SEED}"
if [[ "${RUN_ROOT}" == "${PROJECT_DIR}"/* ]]; then
  DASHBOARD_RUN="/${RUN_ROOT#"${PROJECT_DIR}/"}"
else
  DASHBOARD_RUN=""
fi
METRICS_URL="${DASHBOARD_RUN}/training/${VARIANT}/seed_${SEED}/metrics.json"

ARCHITECTURE="hierarchical_coalition_graph"
VARIANT_ARGS=(--coalition-reservations --weak-mean-field --factorized-critic)
case "${VARIANT}" in
  full)
    ;;
  no_reservation)
    VARIANT_ARGS=(--no-coalition-reservations --weak-mean-field --factorized-critic)
    ;;
  no_weak_mean_field)
    VARIANT_ARGS=(--coalition-reservations --no-weak-mean-field --factorized-critic)
    ;;
  no_factorized_critic)
    VARIANT_ARGS=(--coalition-reservations --weak-mean-field --no-factorized-critic)
    ;;
  oasis_graph)
    ARCHITECTURE="opportunity_graph"
    VARIANT_ARGS=(--no-coalition-reservations)
    ;;
  *)
    echo "Unknown VARIANT=${VARIANT}" >&2
    exit 2
    ;;
esac

cd "${PROJECT_DIR}"
mkdir -p "${SCENARIO_ROOT}" "${RUN_ROOT}/logs" "${TRAIN_DIR}" "${DATA_DIR}"

update_pipeline() {
  local status="$1"
  local stage="$2"
  local message="$3"
  python scripts/update_pipeline_status.py \
    --root "${RUN_ROOT}" \
    --status "${status}" \
    --stage "${stage}" \
    --message "${message}" \
    --method "${VARIANT}" \
    --seed "${SEED}" \
    --metrics-url "${METRICS_URL}" \
    --tensorboard-url "http://127.0.0.1:${TENSORBOARD_PORT}"
}

mark_failed() {
  update_pipeline failed failed \
    "${VARIANT} stopped; inspect logs and resume from the latest checkpoint." \
    || true
}
trap mark_failed ERR
update_pipeline running preparing "Preparing catalogs and Kepler caches."

if [[ "${GENERATE_CATALOGS}" == "1" ]] && \
   { [[ ! -f "${TRAIN_CSV}" ]] || [[ ! -f "${EVAL_CSV}" ]]; }; then
  python -u scripts/generate_task_catalogs.py \
    --output-dir "${DATA_DIR}" \
    --file-prefix thousand_ \
    --train-count "${TRAIN_TASKS}" \
    --test-count "${EVAL_TASKS}" \
    --train-seed 4701 \
    --test-seed 4702 \
    --max-steps "${MAX_STEPS}" \
    --min-window-steps 8 \
    --max-window-steps 32 \
    --minimum-separation-deg 0 \
    --skip-separation-audit \
    --cooperative-observers-min 2 \
    --cooperative-observers-max 5 \
    --overwrite \
    >"${RUN_ROOT}/logs/generate_catalogs.log"
fi

for catalog in "${TRAIN_CSV}" "${EVAL_CSV}"; do
  if [[ ! -f "${catalog}" ]]; then
    echo "Task catalog not found: ${catalog}" >&2
    exit 2
  fi
done

build_cache() {
  local output="$1"
  local satellites="$2"
  local planes="$3"
  if [[ "${REBUILD_SCENARIO}" == "1" ]]; then
    rm -f "${output}"
  fi
  if [[ ! -f "${output}" ]]; then
    python -u scripts/build_kepler_ephemeris.py \
      --output "${output}" \
      --elements-output "${output%.npz}_elements.csv" \
      --start-utc 2026-08-01T00:00:00Z \
      --satellites "${satellites}" \
      --planes "${planes}" \
      --max-steps "${MAX_STEPS}" \
      --lookahead-steps "${LOOKAHEAD_STEPS}" \
      --step-duration-seconds 30 \
      --altitude-km 550 \
      --eccentricity 0.001 \
      --inclination-deg 97.6 \
      --walker-phasing 1
  fi
}

build_cache "${TRAIN_CACHE}" "${TRAIN_SATELLITES}" "${TRAIN_PLANES}"
build_cache "${EVAL_CACHE}" "${EVAL_SATELLITES}" "${EVAL_PLANES}"

MONITOR_PIDS=()
if [[ "${START_MONITORING}" == "1" ]]; then
  for port in "${TENSORBOARD_PORT}" "${DASHBOARD_PORT}"; do
    if ss -lnt 2>/dev/null | grep -q ":${port} "; then
      echo "Monitoring port ${port} is already in use." >&2
      exit 3
    fi
  done
  python -u scripts/serve_dashboard.py --host 127.0.0.1 --port "${DASHBOARD_PORT}" \
    >"${RUN_ROOT}/logs/dashboard.log" 2>&1 &
  MONITOR_PIDS+=("$!")
  tensorboard --logdir "${RUN_ROOT}/training" --host 127.0.0.1 \
    --port "${TENSORBOARD_PORT}" \
    >"${RUN_ROOT}/logs/tensorboard.log" 2>&1 &
  MONITOR_PIDS+=("$!")
  trap 'kill "${MONITOR_PIDS[@]}" 2>/dev/null || true' EXIT INT TERM
  if [[ -n "${DASHBOARD_RUN}" ]]; then
    echo "Dashboard:   http://127.0.0.1:${DASHBOARD_PORT}/web/fast_slow.html?run=${DASHBOARD_RUN}"
  else
    echo "Dashboard root is outside the project tree; use TensorBoard and metrics.json."
  fi
  echo "TensorBoard: http://127.0.0.1:${TENSORBOARD_PORT}"
fi

TRAIN_ARGS=(
  --architecture "${ARCHITECTURE}"
  "${VARIANT_ARGS[@]}"
  --reservation-ttl-steps 2
  --cooperative-observers-min 2
  --cooperative-observers-max 5
  --satellites "${TRAIN_SATELLITES}"
  --tasks "${TRAIN_TASKS}"
  --planes "${TRAIN_PLANES}"
  --max-steps "${MAX_STEPS}"
  --candidate-k 24
  --neighbor-k 6
  --planning-lookahead-steps "${LOOKAHEAD_STEPS}"
  --step-duration-seconds 30
  --point-observation-seconds 5
  --fov-deg 45
  --task-layout catalog
  --ephemeris-cache "${TRAIN_CACHE}"
  --task-catalog "${TRAIN_CSV}"
  --episodes "${EPISODES}"
  --seed "${SEED}"
  --hidden-dim 128
  --semantic-opportunity-balancing
  --no-adaptive-curriculum
  --min-decision-samples 4096
  --max-buffered-episodes 2
  --minibatch-size 2048
  --checkpoint-every 10
  --progress-every-steps 10
  --run-dir "${TRAIN_DIR}"
  --tensorboard
  --device "${DEVICE}"
)
if [[ "${RESUME}" == "1" ]]; then
  TRAIN_ARGS+=(--resume)
fi

update_pipeline running training \
  "Training ${VARIANT} on ${TRAIN_SATELLITES} satellites."
CUDA_VISIBLE_DEVICES="${GPU_ID}" python -u scripts/train_oasis.py \
  "${TRAIN_ARGS[@]}" 2>&1 | tee -a "${RUN_ROOT}/logs/train_${VARIANT}_seed_${SEED}.log"

update_pipeline running evaluating \
  "Evaluating held-out tasks at ${TRAIN_SATELLITES} and ${EVAL_SATELLITES} satellites."
CUDA_VISIBLE_DEVICES="${GPU_ID}" python -u scripts/evaluate_marl.py \
  --checkpoint "${TRAIN_DIR}/checkpoints/latest.pt" \
  --output "${TRAIN_DIR}/evaluation_n${TRAIN_SATELLITES}.json" \
  --satellites "${TRAIN_SATELLITES}" \
  --tasks "${TRAIN_TASKS}" \
  --planes "${TRAIN_PLANES}" \
  --max-steps "${MAX_STEPS}" \
  --candidate-k 24 \
  --neighbor-k 6 \
  --ephemeris-cache "${TRAIN_CACHE}" \
  --task-catalog "${EVAL_CSV}" \
  --eval-task-layout catalog \
  --seed "${EVAL_SEED}" \
  --eval-episodes "${EVAL_EPISODES}" \
  --no-capture-frames \
  --device "${DEVICE}" \
  2>&1 | tee "${RUN_ROOT}/logs/eval_${VARIANT}_n${TRAIN_SATELLITES}_seed_${SEED}.log"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python -u scripts/evaluate_marl.py \
  --checkpoint "${TRAIN_DIR}/checkpoints/latest.pt" \
  --output "${TRAIN_DIR}/evaluation_n${EVAL_SATELLITES}.json" \
  --satellites "${EVAL_SATELLITES}" \
  --tasks "${EVAL_TASKS}" \
  --planes "${EVAL_PLANES}" \
  --max-steps "${MAX_STEPS}" \
  --candidate-k 24 \
  --neighbor-k 6 \
  --ephemeris-cache "${EVAL_CACHE}" \
  --task-catalog "${EVAL_CSV}" \
  --eval-task-layout catalog \
  --seed "$((EVAL_SEED + 1000))" \
  --eval-episodes "${EVAL_EPISODES}" \
  --no-capture-frames \
  --device "${DEVICE}" \
  2>&1 | tee "${RUN_ROOT}/logs/eval_${VARIANT}_n${EVAL_SATELLITES}_seed_${SEED}.log"

update_pipeline complete complete \
  "${VARIANT} training and both held-out evaluations are complete."
echo "${VARIANT} training and 512/1024-satellite evaluations are complete."
