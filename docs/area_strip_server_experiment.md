# Area-strip cooperative observation experiment

This experiment extends the point-target benchmark with physically directed
imaging strips. It is intended for the AAAI fast--slow MARL study.

## Environment semantics

- An area request is a continuous oriented rectangle in a local tangent plane.
- A satellite action is one directed rectangular imaging strip. Its long axis
  follows the Earth-fixed satellite ground-track heading.
- Strip length is ground speed multiplied by acquisition duration. Strip width
  is constrained by the payload swath and 45-degree field of view.
- The environment clips each strip polygon against the target polygon to
  compute exact inside and outside imaged area.
- A deterministic low-discrepancy sample set estimates only the cumulative
  union of multiple strips. It is not an action grid.
- Reward is proportional to newly covered in-target area and observation
  quality. Outside-target imaging and repeated in-target overlap are penalized.
- Small requests become single-satellite one-shot tasks when a compatible
  payload can meet the 95% threshold. Larger requests derive a required strip
  count and at least two contributing satellites when physically feasible.
- Multiple complementary strips may win in the same planning step; materially
  overlapping claims are rejected as strip conflicts.

Point targets retain the original five-second observation model.

## One-time server setup

```bash
cd ~/SatMARL
conda activate satmarl
python -m pip install -e ".[monitoring]"
python -m unittest discover -s tests -v
```

The mixed benchmark files are checked into Git. If they are absent, regenerate
them without using satellite trajectories:

```bash
python scripts/generate_task_catalogs.py \
  --output-dir data/targets \
  --train-count 3072 --test-count 3072 \
  --train-seed 3701 --test-seed 3702 \
  --max-steps 240 --minimum-separation-deg 0.2 \
  --area-fraction 0.75 --file-prefix area_ --overwrite
```

## Training and independent evaluation

Run inside `tmux`. The command builds a Walker-Delta 64/8/1 constellation from
six Keplerian elements, validates geometry, trains, writes TensorBoard events,
and evaluates on the disjoint test catalog.

```bash
tmux new -s area701
cd ~/SatMARL
conda activate satmarl

TARGET_CSV="$PWD/data/targets/area_train_requests.csv" \
EVAL_TARGET_CSV="$PWD/data/targets/area_test_requests.csv" \
RUN_ROOT="$PWD/runs/area_strip_aaai" \
START_UTC="2026-07-20T00:00:00Z" \
SATELLITES=64 PLANES=8 TASKS=3072 EVAL_TASKS=3072 \
ALTITUDE_KM=550 ECCENTRICITY=0.001 INCLINATION_DEG=97.6 \
WALKER_PHASING=1 \
ARCHITECTURE=fast_slow_graph SLOW_INTERVAL=8 \
AREA_TASK_FRACTION=0.75 AREA_OBSERVATION_SECONDS=20 \
AREA_COVERAGE_THRESHOLD=0.95 AREA_COVERAGE_SAMPLES=512 \
AREA_MIN_COOPERATIVE_SATELLITES=2 AREA_MAX_REQUIRED_STRIPS=8 \
AREA_OUTSIDE_PENALTY_WEIGHT=1.0 \
AREA_REDUNDANCY_PENALTY_WEIGHT=0.6 \
SEED=701 EVAL_SEED=1701 \
GPU_ID=0 EPISODES=300 EVAL_EPISODES=20 \
REBUILD_SCENARIO=1 \
bash scripts/start_kepler_constellation_training.sh
```

Resume without rebuilding the scenario:

```bash
RESUME=1 REBUILD_SCENARIO=0 ... \
bash scripts/start_kepler_constellation_training.sh
```

## Monitoring

On the client machine:

```powershell
ssh -L 6006:127.0.0.1:6006 -L 8766:127.0.0.1:8766 yw@SERVER_IP
```

Open TensorBoard at `http://127.0.0.1:6006`. In addition to reward, loss,
opportunity rate, conflicts, and invalid actions, inspect:

- `area/mean_coverage`
- `area/priority_weighted_coverage`
- `area/completed`
- `area/cooperative_completed`
- `area/outside_ratio`
- `area/redundancy_ratio`
- `area/strip_count`

The live versions are under `live/*`. Final `evaluation.json`, `summary.json`,
and `schedule.csv` include strip heading, length, width, marginal coverage,
inside area, outside area, redundancy, and contributing satellites.

## Required paper comparisons

Use identical catalogs, orbital elements, seeds, candidate budgets, reward
weights, and evaluation episodes for all methods.

1. Fast--slow graph (full method).
2. Same graph with `SLOW_INTERVAL=1` (no slow timescale).
3. Same graph without factor messages.
4. Same graph without learned bids.
5. Same-budget MLP, IPPO, MAPPO, QMIX, and PS-DQN baselines.
6. Area reward ablations: no outside penalty and no redundancy penalty.
7. Scale transfer at 64, 128, and 256 satellites.
8. Small/medium/large area stratification and single-shot versus cooperative
   completion rates.

Report at least three training seeds and independent evaluation seeds with
mean, standard deviation, and paired significance or bootstrap confidence
intervals.
