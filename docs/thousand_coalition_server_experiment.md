# Thousand-satellite coalition experiment

This experiment adds a new `hierarchical_coalition_graph` architecture while
preserving `opportunity_graph`, `fast_slow_graph`, IPPO, MAPPO, QMIX, and
PS-DQN as unchanged baselines.

The new method combines three mechanisms in one task-factor formulation:

1. versioned two-phase reserve/commit leases for asynchronous simultaneous
   observations;
2. a constellation-size-invariant critic that pools task factors, plane/region
   mean fields, and the normalized global summary;
3. exact messages for strong shared-task relations and fixed-width mean-field
   summaries for weak plane/region interactions.

## Server preparation

```bash
cd ~/SatMARL
git fetch origin agent/thousand-agent-cooperation
git switch agent/thousand-agent-cooperation
git merge --ff-only origin/agent/thousand-agent-cooperation
conda activate satmarl
python -m pip install -e ".[monitoring]" --no-build-isolation
python -m unittest discover -s tests -v
```

The launcher generates balanced disjoint train/test catalogs when they do not
exist, builds separate 512- and 1024-satellite Kepler caches, trains on 512
satellites, then evaluates on held-out tasks at 512 and 1024 satellites.

## Seed 701 on GPU 0

```bash
cd ~/SatMARL
conda activate satmarl
tmux new -s coalition701

RUN_ROOT="$PWD/runs/thousand_coalition_seed701" \
SCENARIO_ROOT="$PWD/runs/thousand_coalition_shared/scenario" \
SEED=701 EVAL_SEED=1701 GPU_ID=0 \
TENSORBOARD_PORT=6012 DASHBOARD_PORT=8770 \
bash scripts/start_thousand_coalition_training.sh
```

Detach with `Ctrl-b d`.

## Seed 702 on GPU 1

Reuse the generated catalogs and orbital caches but use an independent run
root. Start this after seed 701 has generated the two catalog files and cache
files; training itself can then run concurrently on the second A6000.

```bash
cd ~/SatMARL
conda activate satmarl
tmux new -s coalition702

RUN_ROOT="$PWD/runs/thousand_coalition_seed702" \
SCENARIO_ROOT="$PWD/runs/thousand_coalition_shared/scenario" \
SEED=702 EVAL_SEED=2701 GPU_ID=1 \
GENERATE_CATALOGS=0 \
TENSORBOARD_PORT=6013 DASHBOARD_PORT=8771 \
bash scripts/start_thousand_coalition_training.sh
```

## Seed 703

After either GPU becomes free, run the third full-method seed with the same
catalogs and caches:

```bash
RUN_ROOT="$PWD/runs/thousand_coalition_seed703" \
SCENARIO_ROOT="$PWD/runs/thousand_coalition_shared/scenario" \
SEED=703 EVAL_SEED=3701 GPU_ID=0 \
GENERATE_CATALOGS=0 \
TENSORBOARD_PORT=6014 DASHBOARD_PORT=8772 \
bash scripts/start_thousand_coalition_training.sh
```

## Same-budget controls and ablations

The launcher accepts a `VARIANT` value. Use seed 701 for screening; retain all
three seeds for the final full-vs-OASIS-Graph comparison if compute permits.

| `VARIANT` | Purpose |
| --- | --- |
| `full` | reserve/commit + weak mean field + factorized critic |
| `oasis_graph` | previous opportunity-graph policy at the same scale |
| `no_reservation` | ablates asynchronous coalition consistency |
| `no_weak_mean_field` | ablates weak plane/region interactions |
| `no_factorized_critic` | replaces pooled factor critic with flat critic |

Example after the shared catalogs and caches exist:

```bash
RUN_ROOT="$PWD/runs/thousand_coalition_ablation701" \
SCENARIO_ROOT="$PWD/runs/thousand_coalition_shared/scenario" \
VARIANT=no_reservation SEED=701 EVAL_SEED=1701 GPU_ID=0 \
GENERATE_CATALOGS=0 START_MONITORING=0 \
bash scripts/start_thousand_coalition_training.sh
```

## Monitoring

From the client machine, tunnel both TensorBoard services:

```powershell
ssh -L 6012:127.0.0.1:6012 -L 6013:127.0.0.1:6013 yw@SERVER_IP
```

Open:

- `http://127.0.0.1:6012` for seed 701;
- `http://127.0.0.1:6013` for seed 702.

Progress and coalition diagnostics are also written to:

```text
runs/thousand_coalition_seed*/pipeline.json
runs/thousand_coalition_seed*/training/full/seed_*/metrics.json
runs/thousand_coalition_seed*/logs/train_full_seed_*.log
```

The HTML dashboard reads `pipeline.json` every five seconds and reports the
preparing, training, evaluating, complete, or failed stage. TensorBoard reads
the event files and shows the full curves.

TensorBoard contains the additional groups:

```text
coalitions/active_reservations
coalitions/reserved_satellites
coalitions/created
coalitions/committed
coalitions/expired
coalitions/member_waste
coalitions/commit_rate
coalitions/mean_fill
```

## Resume

Use exactly the same run root, seed, and GPU, and add `RESUME=1`:

```bash
RUN_ROOT="$PWD/runs/thousand_coalition_seed701" \
SCENARIO_ROOT="$PWD/runs/thousand_coalition_shared/scenario" \
SEED=701 EVAL_SEED=1701 GPU_ID=0 RESUME=1 \
GENERATE_CATALOGS=0 TENSORBOARD_PORT=6012 DASHBOARD_PORT=8770 \
bash scripts/start_thousand_coalition_training.sh
```

Do not describe a 1024-satellite result as complete until
`evaluation_n1024.json` exists and reports all requested held-out episodes.
