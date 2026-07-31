# SatMARL / Fast–Slow OASIS-Graph

OASIS-Graph is a constraint-rich multi-agent reinforcement learning project for
large-scale Earth-observation satellite scheduling. It models sparse,
satellite-specific decision opportunities as a semi-Markov process and uses a
size-invariant satellite--task opportunity graph for learned coordination.

The current method adds a bi-timescale hierarchy: a slow strategic encoder
refreshes a persistent constellation-local intent every K physical steps, and
the fast opportunity policy uses that intent for asynchronous observation,
downlink, cooperation, and contention decisions.

The current research protocol trains on 64 satellites and 3,072 tasks, then
evaluates the same graph policy without architectural changes on 128 and 256
satellites. Point-target acquisition lasts five seconds, each episode contains
240 planning steps, and payload field of view is fixed at 45 degrees.

## Method

The implementation combines:

- agent-specific asynchronous decision opportunities;
- duration-corrected opportunity GAE;
- semantic balancing of downlink, single-task, and multi-task choices;
- a sparse bipartite satellite--task opportunity graph;
- a shared, permutation-equivariant candidate scorer;
- policy logits reused as learned resource-contention bids;
- persistent slow strategic intent plus fast opportunity-level execution;
- area- and payload-derived single/multi-satellite strip collaboration;
- directed continuous strip footprints with marginal-coverage bidding;
- centralized training with decentralized execution;
- exact action masks plus execution-time constraint revalidation.

The actor parameter count is independent of constellation size for fixed
candidate and neighbor budgets. Controlled ablations remove graph factor
messages or learned bids, and a same-budget MLP policy tests whether gains come
from the graph architecture rather than the opportunity-learning pipeline.

## Environment

The parallel multi-agent environment enforces:

- two-body Kepler propagation and Earth rotation;
- Earth occultation, elevation, off-nadir, field-of-view, and sunlight limits;
- heterogeneous optical, SAR, and infrared payload constraints;
- resolution, swath, roll, pitch, yaw, slew, stabilization, and payload timing;
- energy, onboard storage, compression, packets, and segmented downlink;
- ground-station visibility, channel capacity, and contention;
- single-satellite, simultaneous cooperative, and sequential cooperative tasks;
- oriented area requests covered by complementary ground-track strips;
- explicit inside coverage, outside imaging, and redundant-overlap accounting;
- hard feasibility masks and execution-time validation.

The orbit model is designed for MARL throughput rather than mission-grade
ephemeris accuracy. It does not yet include J2, drag, cloud cover, weather, or
TLE uncertainty.

## Installation

Python 3.10 or newer is required. Install a PyTorch build compatible with the
server's CUDA driver first, then install the project:

```bash
python -m pip install --upgrade pip
# Install PyTorch using the command recommended for your CUDA version.
python -m pip install -e .
```

Check the GPU and run all tests:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m unittest discover -s tests -v
```

The test suite covers physical constraints, cooperative-task semantics,
opportunity returns and weighting, graph size invariance, graph factor
messages, and QMIX monotonicity.

## Quick start

Run a feasible random or greedy rollout:

```bash
python examples/random_rollout.py --policy greedy --satellites 64 --tasks 3072
```

Train the graph policy:

```bash
python scripts/train_oasis.py \
  --architecture opportunity_graph \
  --satellites 64 --tasks 3072 --planes 8 \
  --max-steps 240 --episodes 250 \
  --point-observation-seconds 5 --fov-deg 45 \
  --semantic-opportunity-balancing \
  --run-dir runs/oasis_graph_aaai/graph --device cuda
```

Train the fast--slow policy directly:

```bash
python scripts/train_oasis.py \
  --architecture fast_slow_graph --slow-interval 8 --slow-intent-dim 64 \
  --satellites 64 --tasks 3072 --planes 8 \
  --max-steps 240 --episodes 300 \
  --point-observation-seconds 5 --fov-deg 45 \
  --semantic-opportunity-balancing \
  --tensorboard \
  --run-dir runs/fast_slow_aaai/training/fast_slow_full/seed_701 \
  --device cuda
```

### Thousand-satellite coalition extension

The `hierarchical_coalition_graph` policy adds versioned reserve/commit leases
for asynchronous coalition formation, exact messages for strong shared-task
relations, plane/region mean fields for weak interactions, and a task-factor
pooled critic. With `candidate_k=24`, `neighbor_k=6`, and hidden width 128, its
actor input width (790), critic input width (802), and 321,924 trainable
parameters remain unchanged from 64 through 1,024 satellites.

Launch the 512-satellite training and held-out 512/1,024-satellite evaluation:

```bash
bash scripts/start_thousand_coalition_training.sh
```

See `docs/thousand_coalition_server_experiment.md` for two-A6000 multi-seed
commands, TensorBoard ports, held-out catalogs, the previous OASIS-Graph
control, and the three mechanism ablations.

Evaluate a checkpoint and export a standard schedule:

```bash
python scripts/evaluate_marl.py \
  --checkpoint runs/oasis_graph_aaai/graph/checkpoints/latest.pt \
  --output runs/oasis_graph_aaai/evaluation/graph/n64/rollout.json \
  --satellites 64 --tasks 3072 --planes 8 \
  --eval-task-layout global_random --eval-episodes 20 --device cuda
```

The evaluation directory contains `summary.json`, `rollout.json`, and
`schedule.csv`. The schedule records satellite, task, decision and acquisition
times, quality, reward, energy, compressed data, observers, stations, downlink,
winner, and rejection reason.

## Reproducible paper pipeline

For a server-ready, multi-seed fast--slow paper pipeline:

```bash
python scripts/run_fast_slow_server_pipeline.py --profile paper --device cuda
```

See `docs/server_fast_slow_training.md` for conda, tmux, resume, GPU, SSH
tunnel, fair RL baseline, and artifact instructions.

The primary physics-based experiment now uses a simulated Walker constellation
defined by six Keplerian orbital elements; see
`docs/kepler_constellation_server_experiment.md`. Archived TLE/SGP4 input is
retained only as an optional future extension in
`docs/real_orbit_server_experiment.md`. Both paths apply the same 5-second
acquisition, 45-degree visibility, resource, and MARL constraints.

Generate the fixed, algorithm-agnostic 3,072/3,072 train/test task catalogs
with `python scripts/generate_task_catalogs.py --output-dir data/targets`.
The checked-in manifest records the seeds, class quotas, spatial balance,
split separation, and file hashes.

For the mixed point/area benchmark (25% point and 75% area), use the checked-in
`area_train_requests.csv` and `area_test_requests.csv`, or regenerate them:

```bash
python scripts/generate_task_catalogs.py --output-dir data/targets \
  --train-count 3072 --test-count 3072 \
  --train-seed 3701 --test-seed 3702 \
  --area-fraction 0.75 --file-prefix area_ --overwrite
```

Area collaboration is not a random label. At environment reset, target size,
observation duration, compatible payload resolution, swath, and orbital ground
speed determine whether one strip is sufficient or multiple satellites must
contribute. See `docs/area_strip_server_experiment.md` for the server command,
metrics, and ablations.

Install monitoring support and launch the JSON dashboard plus TensorBoard:

```bash
python -m pip install -e ".[monitoring]"
bash scripts/start_monitoring.sh
```

The complete development pipeline runs tests, waits for the five-baseline
reference suite, trains the 64-satellite graph method and ablations, evaluates
64/128/256-satellite transfer, and generates CSV, HTML, and LaTeX artifacts:

```bash
python scripts/run_graph_paper_pipeline.py
```

Primary RL comparisons are parameter-shared DQN, IPPO, MAPPO, QMIX, and the
non-graph OASIS policy. Existing transfer baselines trained at a different
constellation size are labeled explicitly and must not be interpreted as a
same-scale training comparison.

Training checkpoints and experiment outputs are intentionally excluded from
Git. Copy `runs/` separately when resuming on another machine.

## Dashboard

Start the local experiment server:

```bash
python scripts/serve_dashboard.py --port 8766
```

Open:

- OASIS-Graph paper pipeline: `http://127.0.0.1:8766/web/oasis_graph.html`
- Fast--Slow training curves: `http://127.0.0.1:8766/web/fast_slow.html`
- TensorBoard scalar explorer: `http://127.0.0.1:6006`
- live baseline TensorBoard bridge: `http://127.0.0.1:6007`
- OASIS training board: `http://127.0.0.1:8766/web/oasis.html`
- baseline comparison: `http://127.0.0.1:8766/web/compare.html`
- representative schedule/orbit replay: `http://127.0.0.1:8766/web/eval.html`

For a remote server, forward the port over SSH:

```bash
ssh -L 8766:127.0.0.1:8766 -L 6006:127.0.0.1:6006 \
  -L 6007:127.0.0.1:6007 user@server
```

Baseline trainers keep their resumable history in `metrics.json`.  Mirror an
already-running 64-satellite baseline suite into TensorBoard without restarting
training:

```bash
RUN_ROOTS="$PWD/runs/rl64_seed701 $PWD/runs/rl64_seed702 $PWD/runs/rl64_seed703" \
  bash scripts/start_baseline_tensorboard.sh
```

Open `http://127.0.0.1:6007`.  The bridge backfills existing episodes and then
polls each algorithm's `metrics.json` every five seconds.  Its event files are
written below `runs/baseline_tensorboard/`; the source experiment artifacts are
never modified.

## Paper

The AAAI-format draft and reproducibility notes are under `docs/aaai27/`.
Large-scale tables are generated from independent evaluation summaries rather
than entered manually.

## Repository layout

```text
src/sat_marl_env/       environment, entities, and orbital geometry
scripts/                training, evaluation, pipelines, and dashboards
tests/                  environment and MARL regression tests
configs/                controlled baseline definitions
examples/               rollout examples
web/                    local HTML experiment boards
docs/aaai27/            paper draft and Overleaf package
```
