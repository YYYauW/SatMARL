# SatMARL / OASIS-Graph

OASIS-Graph is a constraint-rich multi-agent reinforcement learning project for
large-scale Earth-observation satellite scheduling. It models sparse,
satellite-specific decision opportunities as a semi-Markov process and uses a
size-invariant satellite--task opportunity graph for learned coordination.

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
- OASIS training board: `http://127.0.0.1:8766/web/oasis.html`
- baseline comparison: `http://127.0.0.1:8766/web/compare.html`
- representative schedule/orbit replay: `http://127.0.0.1:8766/web/eval.html`

For a remote server, forward the port over SSH:

```bash
ssh -L 8766:127.0.0.1:8766 user@server
```

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
