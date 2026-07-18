from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def web_path(path: Path) -> str:
    try:
        return "/" + path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def complete(path: Path) -> bool:
    return load_json(path).get("status") == "complete"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the bi-timescale opportunity-graph MARL method."
    )
    parser.add_argument(
        "--run-root", type=Path, default=ROOT / "runs" / "fast_slow_aaai"
    )
    parser.add_argument("--profile", choices=["smoke", "paper"], default="paper")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=[701, 702, 703])
    parser.add_argument(
        "--include-ablations", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    pipeline_path = run_root / "pipeline.json"
    results_path = run_root / "results.json"
    if args.profile == "smoke":
        satellites, tasks_per_satellite, planes = 8, 8, 2
        max_steps, candidate_k, neighbor_k = 8, 6, 2
        episodes, eval_plan = 2, ((8, 1),)
        hidden_dim, min_samples = 32, 16
        seeds = args.seeds[:1]
        train_layout, eval_layout = "mixed", "curriculum_visible"
    else:
        satellites, tasks_per_satellite, planes = 64, 48, 8
        max_steps, candidate_k, neighbor_k = 240, 24, 6
        episodes, eval_plan = 300, ((64, 20), (128, 20), (256, 10))
        hidden_dim, min_samples = 128, 1024
        seeds = args.seeds
        train_layout = eval_layout = "global_random"

    methods: list[dict[str, Any]] = [
        {
            "name": "fast_slow_full",
            "architecture": "fast_slow_graph",
            "flags": ["--slow-interval", "8", "--slow-intent-dim", "64"],
            "role": "proposed",
        },
        {
            "name": "graph",
            "architecture": "opportunity_graph",
            "flags": [],
            "role": "same_scale_graph_baseline",
        },
    ]
    if args.include_ablations:
        methods.extend(
            [
                {
                    "name": "fast_slow_k1",
                    "architecture": "fast_slow_graph",
                    "flags": ["--slow-interval", "1", "--slow-intent-dim", "64"],
                    "role": "no_temporal_abstraction",
                },
                {
                    "name": "fast_slow_no_factor",
                    "architecture": "fast_slow_graph",
                    "flags": ["--slow-interval", "8", "--no-graph-factor-messages"],
                    "role": "no_competition_factor",
                },
                {
                    "name": "fast_slow_no_bid",
                    "architecture": "fast_slow_graph",
                    "flags": ["--slow-interval", "8", "--no-learned-resource-bids"],
                    "role": "no_learned_conflict_bid",
                },
            ]
        )

    state: dict[str, Any] = {
        "status": "running",
        "stage": "initializing",
        "message": "Preparing fast-slow AAAI training and evaluation",
        "started_at": time.time(),
        "updated_at": time.time(),
        "pid": os.getpid(),
        "profile": args.profile,
        "protocol": {
            "train_satellites": satellites,
            "tasks_per_satellite": tasks_per_satellite,
            "train_episodes": episodes,
            "seeds": seeds,
            "eval_plan": [list(item) for item in eval_plan],
            "horizon_steps": max_steps,
            "point_observation_seconds": 5,
            "fov_deg": 45,
            "slow_interval_steps": 8,
            "train_layout": train_layout,
            "test_layout": eval_layout,
        },
        "methods": methods,
        "training_runs": [],
        "stages": [],
        "dashboard": "/web/fast_slow.html",
        "results": web_path(results_path),
    }

    def update(stage: str, message: str, **extra: Any) -> None:
        state.update(
            {"stage": stage, "message": message, "updated_at": time.time(), **extra}
        )
        atomic_json(pipeline_path, state)

    def run(stage: str, command: list[str], artifact: Path | None = None) -> None:
        if artifact is not None and complete(artifact):
            state["stages"].append(
                {"name": stage, "status": "reused", "artifact": web_path(artifact)}
            )
            update(stage, f"Reusing complete artifact: {artifact}")
            return
        log_path = run_root / "logs" / f"{stage}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "name": stage,
            "status": "running",
            "started_at": time.time(),
            "command": subprocess.list2cmdline(command),
            "log": web_path(log_path),
        }
        state["stages"].append(record)
        update(stage, record["command"])
        with log_path.open("a", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        record.update(
            {
                "status": "complete" if completed.returncode == 0 else "failed",
                "finished_at": time.time(),
                "returncode": completed.returncode,
            }
        )
        update(stage, f"return code {completed.returncode}")
        if completed.returncode:
            raise RuntimeError(f"{stage} failed; inspect {log_path}")
        if artifact is not None and not complete(artifact):
            raise RuntimeError(f"{stage} did not create a complete {artifact}")

    try:
        atomic_json(pipeline_path, state)
        if not args.skip_tests:
            run(
                "preflight_tests",
                [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            )

        for method in methods:
            for seed in seeds:
                train_dir = run_root / "training" / method["name"] / f"seed_{seed}"
                metrics_path = train_dir / "metrics.json"
                checkpoint = train_dir / "checkpoints" / "latest.pt"
                state["training_runs"].append(
                    {
                        "method": method["name"],
                        "role": method["role"],
                        "seed": seed,
                        "metrics": web_path(metrics_path),
                        "checkpoint": str(checkpoint),
                    }
                )
                update(
                    f"train_{method['name']}_seed_{seed}",
                    "Training run registered",
                )
                command = [
                    sys.executable,
                    str(ROOT / "scripts" / "train_oasis.py"),
                    "--architecture", method["architecture"],
                    "--satellites", str(satellites),
                    "--tasks", str(satellites * tasks_per_satellite),
                    "--planes", str(planes),
                    "--max-steps", str(max_steps),
                    "--candidate-k", str(candidate_k),
                    "--neighbor-k", str(neighbor_k),
                    "--episodes", str(episodes),
                    "--seed", str(seed),
                    "--task-layout", train_layout,
                    "--curriculum-visible-fraction", "0.75",
                    "--curriculum-end-fraction", "0.05",
                    "--target-opportunity-rate", "0.12",
                    "--semantic-opportunity-balancing",
                    "--hidden-dim", str(hidden_dim),
                    "--min-decision-samples", str(min_samples),
                    "--max-buffered-episodes", "8",
                    "--checkpoint-every", "10",
                    "--point-observation-seconds", "5",
                    "--fov-deg", "45",
                    "--max-off-nadir-deg", "45",
                    "--run-dir", str(train_dir),
                    "--device", args.device,
                    *method["flags"],
                ]
                if metrics_path.exists() and checkpoint.exists() and not complete(metrics_path):
                    command.append("--resume")
                run(
                    f"train_{method['name']}_seed_{seed}", command, metrics_path
                )

                for eval_satellites, eval_episodes in eval_plan:
                    eval_tasks = eval_satellites * tasks_per_satellite
                    eval_planes = max(2, eval_satellites // 8)
                    output_dir = (
                        run_root
                        / "evaluation"
                        / method["name"]
                        / f"seed_{seed}"
                        / f"n{eval_satellites}"
                    )
                    summary = output_dir / "summary.json"
                    eval_command = [
                        sys.executable,
                        str(ROOT / "scripts" / "evaluate_marl.py"),
                        "--checkpoint", str(checkpoint),
                        "--output", str(output_dir / "rollout.json"),
                        "--satellites", str(eval_satellites),
                        "--tasks", str(eval_tasks),
                        "--planes", str(eval_planes),
                        "--max-steps", str(max_steps),
                        "--candidate-k", str(candidate_k),
                        "--neighbor-k", str(neighbor_k),
                        "--eval-task-layout", eval_layout,
                        "--seed", "12001",
                        "--eval-episodes", str(eval_episodes),
                        "--no-capture-frames",
                        "--device", args.device,
                    ]
                    run(
                        f"eval_{method['name']}_seed_{seed}_n{eval_satellites}",
                        eval_command,
                        summary,
                    )

        update("aggregate", "Aggregating held-out evaluations across seeds")
        metrics = (
            "mean_episode_reward",
            "completed_tasks",
            "cooperative_completed_tasks",
            "total_priority_completed",
            "mean_observation_quality",
            "task_decision_opportunity_rate",
            "avoidable_idle_actions",
            "total_conflicts",
            "total_ground_conflicts",
            "total_invalid_actions",
            "total_window_misses",
        )
        seed_rows: list[dict[str, Any]] = []
        for method in methods:
            for seed in seeds:
                for eval_satellites, _ in eval_plan:
                    summary_path = (
                        run_root
                        / "evaluation"
                        / method["name"]
                        / f"seed_{seed}"
                        / f"n{eval_satellites}"
                        / "summary.json"
                    )
                    summary = load_json(summary_path)
                    aggregate = summary.get("benchmark", {}).get("aggregate", {})
                    row = {
                        "method": method["name"],
                        "role": method["role"],
                        "seed": seed,
                        "satellites": eval_satellites,
                        "tasks": eval_satellites * tasks_per_satellite,
                        "eval_episodes": summary.get("eval_episodes"),
                        "summary": web_path(summary_path),
                    }
                    for metric in metrics:
                        row[metric] = aggregate.get(metric, {}).get("mean")
                    row["completion_rate"] = float(row["completed_tasks"] or 0.0) / max(
                        1, row["tasks"]
                    )
                    agent_steps = max_steps * eval_satellites
                    row["conflicts_per_1000_agent_steps"] = 1000.0 * float(
                        row["total_conflicts"] or 0.0
                    ) / max(1, agent_steps)
                    row["ground_conflicts_per_1000_agent_steps"] = 1000.0 * float(
                        row["total_ground_conflicts"] or 0.0
                    ) / max(1, agent_steps)
                    seed_rows.append(row)

        rows: list[dict[str, Any]] = []
        for method in methods:
            for eval_satellites, _ in eval_plan:
                group = [
                    row
                    for row in seed_rows
                    if row["method"] == method["name"]
                    and row["satellites"] == eval_satellites
                ]
                row = {
                    "method": method["name"],
                    "role": method["role"],
                    "satellites": eval_satellites,
                    "tasks": eval_satellites * tasks_per_satellite,
                    "seeds": len(group),
                }
                for metric in (*metrics, "completion_rate", "conflicts_per_1000_agent_steps", "ground_conflicts_per_1000_agent_steps"):
                    values = [float(item[metric]) for item in group if item.get(metric) is not None]
                    row[metric] = statistics.mean(values) if values else None
                    row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
                rows.append(row)

        payload = {
            "status": "complete",
            "created_at": time.time(),
            "protocol": state["protocol"],
            "rows": rows,
            "seed_rows": seed_rows,
            "interpretation": (
                "Primary evidence is multi-seed, same-scale training. Cross-scale rows are "
                "zero-shot generalization tests of the same checkpoint architecture."
            ),
        }
        atomic_json(results_path, payload)
        with (run_root / "results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        state["status"] = "complete"
        update("complete", "Training, ablations, scale tests, and aggregation completed")
    except Exception as exc:
        state["status"] = "failed"
        update("failed", repr(exc), error=repr(exc))
        raise


if __name__ == "__main__":
    main()
