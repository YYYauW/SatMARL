# Point-target AAAI experiment suite

This suite is the paper-facing protocol for the submitted point-target abstract.
It deliberately excludes area/strip targets and uses the `opportunity_graph`
method, where the two time scales are (1) 30-second physical propagation and
(2) asynchronous semi-Markov decision opportunities. The extra
`fast_slow_graph` hierarchy is not mixed into the abstract results.

## What is compared

All learning methods train on the same 64-satellite, 3,072-task Walker-Delta
scenario for 300 episodes and seeds 701/702/703:

- OASIS-Graph (`opportunity_graph`)
- parameter-comparable MLP opportunity policy
- IPPO
- MAPPO
- QMIX
- parameter-shared DQN

The matched-budget ablations are:

- no duration-corrected opportunity GAE
- no task-factor messages
- no learned contention bids
- no opportunity balancing

Every core checkpoint is independently evaluated on the same paired held-out
episodes at 64, 128, and 256 satellites. Task load remains 48 requests per
satellite. Ablations are evaluated at 64 satellites. Results report variation
across independent training seeds, not variation across test episodes alone.

## Server setup

```bash
cd ~/SatMARL
git switch agent/initial-satmarl-import
git pull --ff-only

conda activate satmarl
python -m pip install -e .
python -m pip install tensorboard

python -m unittest discover -s tests -v
```

Verify both A6000 devices and ports:

```bash
nvidia-smi
ss -lntp | grep -E ':6010|:8766' || true
```

## Run the paper-critical suite

Use `tmux` so the process survives a remote desktop or SSH disconnect:

```bash
tmux new -s point-aaai
cd ~/SatMARL
conda activate satmarl

RUN_ROOT="$PWD/runs/point_aaai_final" \
PHASES="core,ablations" \
SEEDS="701,702,703" \
GPU_IDS="0,1" \
EPISODES=300 \
EVAL_EPISODES=20 \
TENSORBOARD_PORT=6010 \
DASHBOARD_PORT=8766 \
RESUME=1 \
bash scripts/start_point_aaai_server.sh
```

Detach with `Ctrl-b d`; reconnect with:

```bash
tmux attach -t point-aaai
```

The launcher uses one independent process per GPU. It does not use DDP because
the environment simulator is CPU-heavy and the paper needs independent seeds.
Before training, it also writes geometry/visibility audit reports for the
64-satellite training case and every evaluation scale under `scenario/`.

## Add the sparsity sweep

After the core suite completes:

```bash
tmux new -s point-sparsity
cd ~/SatMARL
conda activate satmarl

RUN_ROOT="$PWD/runs/point_aaai_sparsity" \
PHASES="sparsity" \
SEEDS="701,702,703" \
GPU_IDS="0,1" \
EPISODES=300 \
EVAL_EPISODES=20 \
TENSORBOARD_PORT=6011 \
DASHBOARD_PORT=8767 \
RESUME=1 \
bash scripts/start_point_aaai_server.sh
```

The sweep trains OASIS-Graph and the MLP opportunity policy at 768 and 1,536
tasks; the 3,072-task point comes from the core suite. Plot performance against
the measured task-decision opportunity rate, not merely the requested task
count.

## Monitoring from a local computer

Create SSH tunnels from the local computer:

```bash
ssh -N \
  -L 8766:127.0.0.1:8766 \
  -L 6010:127.0.0.1:6010 \
  USER@SERVER
```

Then open:

- HTML: `http://127.0.0.1:8766/web/fast_slow.html?run=/runs/point_aaai_final`
- TensorBoard: `http://127.0.0.1:6010`

Useful terminal checks:

```bash
cd ~/SatMARL
python -m json.tool runs/point_aaai_final/pipeline.json | less
tail -f runs/point_aaai_final/logs/suite.log
find runs/point_aaai_final/training -name metrics.json -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort
nvidia-smi pmon -c 1
```

Final artifacts:

- `runs/point_aaai_final/pipeline.json`
- `runs/point_aaai_final/results.json`
- `runs/point_aaai_final/results.csv`
- per-seed checkpoints under `training/METHOD/seed_SEED/checkpoints/`
- per-seed independent evaluation and schedules under `evaluation/`
- commands and stdout/stderr under `logs/`

## Recovery

Re-run the identical launch command with `RESUME=1`. Completed jobs are skipped,
training jobs with `latest.pt` resume, and completed evaluations are preserved.
Do not delete the run root when recovering.
