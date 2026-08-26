# Thousand-satellite EO paper experiment suite

This suite turns the thousand-agent extension into one reproducible paper
protocol. It does **not** fabricate results: the scripts prepare scenarios,
train models, evaluate checkpoints, and generate paper tables only from
completed JSON artifacts.

## Experimental claim and fixed controls

- Train: 512 Walker-Delta satellites, 16 planes, 24,576 point requests.
- Scale evaluation: 128, 256, 512, and 1,024 satellites at 48 requests per
  satellite; no retraining between scales.
- Physics: 240 steps, 30 s propagation, 5 s point acquisition, 45 degree FOV,
  550 km altitude, 97.6 degree inclination.
- Statistics: training seeds 701/702/703; 20 paired held-out episodes per scale.
- Cooperation: fixed catalog quotas with single, sequential, and simultaneous
  requests and required coalition sizes from two through five satellites.
- Safety: the same hard visibility, attitude, energy, storage, payload,
  cooldown, downlink, and contention constraints are used by every method.

The task CSVs are declared synthetic EO demand, while orbital visibility is
computed from reproducible Keplerian Walker elements. Geographic stress tests
are synthetic demand shifts, not claims of operational traffic or imagery.

## Methods

The same training episode count, scenario, task catalog, network width, action
mask, and paired evaluation seeds are used for:

- `full`: hierarchical coalition graph with reserve/commit, exact strong
  task-factor interactions, weak mean field, and factorized critic.
- `oasis_graph`: the previous sparse opportunity graph control.
- `mlp_opportunity`: same opportunity-learning loop with an MLP policy.
- `ippo`, `mappo`, `qmix`, `ps_dqn`: reinforcement-learning baselines.
- `no_reservation`, `no_weak_mean_field`, `no_factorized_critic`,
  `no_duration_correction`, `no_opportunity_balance`, `no_factor_messages`,
  `no_learned_bids`: causal ablations. The last two independently remove the
  task-factor coordination messages and the learned contention bid while
  retaining the hierarchical coalition architecture.

The stress suite tests three axes at 1,024 satellites: global/clustered/event
geography, 0/10/30/50 percent cooperative demand, and 12,288/24,576/49,152
requests. Stress evaluations use 10 held-out episodes per training seed.

## Server setup

```bash
cd ~/SatMARL
git fetch origin
git switch agent/thousand-agent-cooperation
git pull --ff-only

conda activate satmarl
python -m pip install -e ".[monitoring]" --no-build-isolation
python -m compileall -q src scripts tests
python -m unittest \
  tests.test_task_catalog_generator \
  tests.test_thousand_paper_suite -v
```

First verify the complete command graph without running jobs:

```bash
python scripts/run_thousand_paper_suite.py \
  --profile smoke --stage all --gpus 0,1 --dry-run \
  --run-root runs/thousand_paper_plan
```

Then run the actual smoke test. It uses 8/16 satellites and one episode:

```bash
PROFILE=smoke \
RUN_ROOT="$PWD/runs/thousand_paper_smoke" \
SCENARIO_ROOT="$PWD/runs/thousand_paper_smoke/scenario" \
GPU_IDS=0,1 TENSORBOARD_PORT=6012 DASHBOARD_PORT=8770 \
bash scripts/start_thousand_paper_suite.sh
```

Do not start the formal suite until the smoke suite reports `status=complete`.

## Recommended two-A6000 schedule

Use one job per GPU by default. The launcher assigns queued jobs to GPU 0 and
GPU 1 and records every exact command in `command_plan_*.json`. Values above
one for `resources.jobs_per_gpu` use distinct logical worker slots on each
physical GPU; use them only after checking host-memory and simulator throughput.

### 1. Prepare shared scenarios and catalogs

```bash
START_MONITORING=0 STAGE=prepare GPU_IDS=0,1 \
RUN_ROOT="$PWD/runs/thousand_paper_suite" \
SCENARIO_ROOT="$PWD/runs/thousand_paper_shared/scenario" \
bash scripts/start_thousand_paper_suite.sh
```

### 2. Train the core comparison first

```bash
STAGE=train GPU_IDS=0,1 \
METHODS=full,oasis_graph,mlp_opportunity,ippo,mappo,qmix,ps_dqn \
RUN_ROOT="$PWD/runs/thousand_paper_suite" \
SCENARIO_ROOT="$PWD/runs/thousand_paper_shared/scenario" \
TENSORBOARD_PORT=6012 DASHBOARD_PORT=8770 \
bash scripts/start_thousand_paper_suite.sh
```

### 3. Train causal ablations

Reuse the same `RUN_ROOT`; completed jobs are skipped and incomplete jobs resume
from `checkpoints/latest.pt`.

```bash
START_MONITORING=0 STAGE=train GPU_IDS=0,1 \
METHODS=no_reservation,no_weak_mean_field,no_factorized_critic,no_duration_correction,no_opportunity_balance,no_factor_messages,no_learned_bids \
RUN_ROOT="$PWD/runs/thousand_paper_suite" \
SCENARIO_ROOT="$PWD/runs/thousand_paper_shared/scenario" \
bash scripts/start_thousand_paper_suite.sh
```

### 4. Paired scale evaluation

```bash
START_MONITORING=0 STAGE=evaluate GPU_IDS=0,1 \
RUN_ROOT="$PWD/runs/thousand_paper_suite" \
SCENARIO_ROOT="$PWD/runs/thousand_paper_shared/scenario" \
bash scripts/start_thousand_paper_suite.sh
```

### 5. Cooperation and EO-demand stress

```bash
START_MONITORING=0 STAGE=stress GPU_IDS=0,1 \
METHODS=full,oasis_graph,no_reservation \
RUN_ROOT="$PWD/runs/thousand_paper_suite" \
SCENARIO_ROOT="$PWD/runs/thousand_paper_shared/scenario" \
bash scripts/start_thousand_paper_suite.sh
```

### 6. Generate statistics and LaTeX

```bash
START_MONITORING=0 STAGE=summarize GPU_IDS=0,1 \
RUN_ROOT="$PWD/runs/thousand_paper_suite" \
SCENARIO_ROOT="$PWD/runs/thousand_paper_shared/scenario" \
bash scripts/start_thousand_paper_suite.sh
```

Running `STAGE=all` performs all six stages. The staged commands are preferred
because they expose failures earlier and prioritize the main paper table.

## Monitoring

From the client machine, replace `USER@SERVER` and create both tunnels:

```bash
ssh -L 8770:127.0.0.1:8770 -L 6012:127.0.0.1:6012 USER@SERVER
```

Open:

- Dashboard: `http://127.0.0.1:8770/web/fast_slow.html?run=/runs/thousand_paper_suite`
- TensorBoard: `http://127.0.0.1:6012`

Quick terminal status:

```bash
watch -n 10 'python - <<"PY"
import json
from collections import Counter
from pathlib import Path
p = json.loads(Path("runs/thousand_paper_suite/suite.json").read_text())
print("status:", p["status"], "stage:", p["stage"], "message:", p["message"])
print(Counter(x.get("status") for x in p.get("jobs", {}).values()))
for name, row in p.get("jobs", {}).items():
    if row.get("status") == "running":
        print(name, "gpu", row.get("gpu"), "pid", row.get("pid"), row.get("log"))
PY'
```

## Resume and failure recovery

After a disconnect or reboot, rerun the same stage with the same `RUN_ROOT`.
The launcher passes `--resume`, skips complete outputs, and resumes an existing
checkpoint. Inspect `runs/thousand_paper_suite/logs/*.log` before deleting or
moving any artifact. Set `START_MONITORING=0` when ports are already occupied.

## Paper-ready outputs

`runs/thousand_paper_suite/paper_results/` contains:

- `results_per_seed.csv`: one row per independent training seed and condition.
- `results_summary.csv`: mean and standard deviation across training seeds.
- `paired_comparisons.csv`: full-minus-baseline paired differences and 95%
  Student-t intervals.
- `results.json`: machine-readable evidence bundle.
- `generated_thousand_results.tex`: base scale table for the paper.

Each evaluation JSON also records parameter count, constraint violations,
conflicts, cooperative completions, opportunity rates, and rollout throughput.
Only completed artifacts may be quoted. Smoke results validate execution but
must not be reported as formal 512/1,024-satellite evidence.
