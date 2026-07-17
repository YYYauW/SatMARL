from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from marl_common import atomic_write_json


ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = ("oasis", "ippo", "mappo", "qmix", "ps_dqn")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def common_args(args: argparse.Namespace) -> list[str]:
    return [
        "--satellites",
        str(args.satellites),
        "--tasks",
        str(args.tasks),
        "--planes",
        str(args.planes),
        "--max-steps",
        str(args.max_steps),
        "--candidate-k",
        str(args.candidate_k),
        "--neighbor-k",
        str(args.neighbor_k),
        "--step-duration-seconds",
        str(args.step_duration_seconds),
        "--point-observation-seconds",
        str(args.point_observation_seconds),
        "--fov-deg",
        str(args.fov_deg),
        "--task-layout",
        args.task_layout,
        "--curriculum-visible-fraction",
        str(args.curriculum_visible_fraction),
        "--curriculum-ground-track-jitter-deg",
        str(args.curriculum_ground_track_jitter_deg),
        "--curriculum-time-jitter-steps",
        str(args.curriculum_time_jitter_steps),
        "--curriculum-payload-match-probability",
        str(args.curriculum_payload_match_probability),
        "--min-observation-elevation-deg",
        str(args.min_observation_elevation_deg),
        "--max-off-nadir-deg",
        str(args.max_off_nadir_deg),
        "--optical-min-sun-elevation-deg",
        str(args.optical_min_sun_elevation_deg),
        "--min-task-window",
        str(args.min_task_window),
        "--max-task-window",
        str(args.max_task_window),
        "--planning-lookahead-steps",
        str(args.planning_lookahead_steps),
        "--scenario-seed-cycle",
        str(args.scenario_seed_cycle),
        "--episodes",
        str(args.episodes),
        "--seed",
        str(args.seed),
        "--hidden-dim",
        str(args.hidden_dim),
        "--hidden-layers",
        str(args.hidden_layers),
        "--activation",
        "relu",
        "--layer-norm",
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--gamma",
        str(args.gamma),
        "--grad-clip",
        str(args.grad_clip),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--metrics-every",
        "1",
        "--device",
        args.device,
    ]


def training_command(
    algorithm: str, args: argparse.Namespace, run_dir: Path
) -> list[str]:
    common = common_args(args)
    if algorithm == "oasis":
        command = [
            sys.executable,
            str(ROOT / "scripts" / "train_oasis.py"),
            *common,
            "--no-adaptive-curriculum",
            "--opportunity-gae",
            "--opportunity-balancing",
            "--min-decision-samples",
            "512",
            "--max-buffered-episodes",
            "4",
            "--gae-lambda",
            "0.95",
            "--clip-coef",
            "0.2",
            "--value-clip-coef",
            "0.2",
            "--entropy-coef",
            "0.02",
            "--value-coef",
            "0.5",
            "--update-epochs",
            "4",
            "--minibatch-size",
            "256",
            "--normalize-advantages",
            "--reward-scale",
            "1.0",
            "--reward-clip",
            "10.0",
            "--run-dir",
            str(run_dir),
        ]
    elif algorithm == "ps_dqn":
        command = [
            sys.executable,
            str(ROOT / "scripts" / "train_dqn.py"),
            *common,
            "--buffer-size",
            "60000",
            "--batch-size",
            "256",
            "--learning-starts",
            "1024",
            "--store-agents",
            str(args.satellites),
            "--optimizer",
            "adamw",
            "--loss-type",
            "huber",
            "--huber-delta",
            "1.0",
            "--double-dqn",
            "--local-observation-only",
            "--reward-scale",
            "1.0",
            "--reward-clip",
            "10.0",
            "--epsilon-start",
            "0.8",
            "--epsilon-end",
            "0.05",
            "--epsilon-decay-steps",
            "60000",
            "--target-update-steps",
            "800",
            "--target-tau",
            "1.0",
            "--train-frequency",
            "1",
            "--gradient-steps",
            "1",
            "--run-dir",
            str(run_dir),
        ]
    elif algorithm in {"ippo", "mappo"}:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "train_ppo.py"),
            "--algorithm",
            algorithm,
            *common,
            "--gae-lambda",
            "0.95",
            "--clip-coef",
            "0.2",
            "--value-clip-coef",
            "0.2",
            "--entropy-coef",
            "0.02",
            "--value-coef",
            "0.5",
            "--update-epochs",
            "4",
            "--minibatch-size",
            "1024",
            "--normalize-advantages",
            "--reward-scale",
            "1.0",
            "--reward-clip",
            "10.0",
            "--run-dir",
            str(run_dir),
        ]
    else:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "train_qmix.py"),
            *common,
            "--mixer-embed-dim",
            "64",
            "--buffer-size",
            "2400",
            "--batch-size",
            "32",
            "--learning-starts",
            "256",
            "--huber-delta",
            "1.0",
            "--double-q",
            "--reward-scale",
            "1.0",
            "--reward-clip",
            "10.0",
            "--epsilon-start",
            "0.8",
            "--epsilon-end",
            "0.05",
            "--epsilon-decay-steps",
            "60000",
            "--target-update-steps",
            "800",
            "--target-tau",
            "1.0",
            "--gradient-steps",
            "1",
            "--run-dir",
            str(run_dir),
        ]
    if args.resume:
        command.append("--resume")
    return command


def evaluate_command(
    run_dir: Path, seed: int, eval_episodes: int, device: str, eval_task_layout: str | None
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_marl.py"),
        "--checkpoint",
        str(run_dir / "checkpoints" / "latest.pt"),
        "--output",
        str(run_dir / "eval" / "rollout.json"),
        "--seed",
        str(seed),
        "--eval-episodes",
        str(eval_episodes),
        "--device",
        device,
    ]
    if eval_task_layout:
        command.extend(["--eval-task-layout", eval_task_layout])
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=ROOT / "runs" / "comparison")
    parser.add_argument("--satellites", type=int, default=16)
    parser.add_argument("--tasks", type=int, default=768)
    parser.add_argument("--planes", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--candidate-k", type=int, default=24)
    parser.add_argument("--neighbor-k", type=int, default=6)
    parser.add_argument(
        "--task-layout",
        choices=["global_random", "curriculum_visible", "mixed", "mixed_curriculum"],
        default="global_random",
    )
    parser.add_argument("--curriculum-visible-fraction", type=float, default=0.0)
    parser.add_argument("--curriculum-ground-track-jitter-deg", type=float, default=5.0)
    parser.add_argument("--curriculum-time-jitter-steps", type=int, default=3)
    parser.add_argument("--curriculum-payload-match-probability", type=float, default=0.8)
    parser.add_argument("--min-observation-elevation-deg", type=float, default=3.0)
    parser.add_argument("--step-duration-seconds", type=float, default=30.0)
    parser.add_argument("--point-observation-seconds", type=float, default=5.0)
    parser.add_argument("--fov-deg", type=float, default=45.0)
    parser.add_argument("--max-off-nadir-deg", type=float, default=45.0)
    parser.add_argument("--optical-min-sun-elevation-deg", type=float, default=8.0)
    parser.add_argument("--min-task-window", type=int, default=40)
    parser.add_argument("--max-task-window", type=int, default=160)
    parser.add_argument("--planning-lookahead-steps", type=int, default=30)
    parser.add_argument("--scenario-seed-cycle", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--eval-seed", type=int, default=9001)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument(
        "--eval-task-layout",
        choices=["global_random", "curriculum_visible", "mixed", "mixed_curriculum"],
        default=None,
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    args.base_dir = args.base_dir.resolve()
    args.base_dir.mkdir(parents=True, exist_ok=True)
    suite_path = args.base_dir / "suite.json"
    suite_stop = args.base_dir / "STOP"
    if suite_stop.exists():
        suite_stop.unlink()

    suite: dict = {
        "status": "running",
        "started_at": time.time(),
        "updated_at": time.time(),
        "current_algorithm": None,
        "algorithms": list(ALGORITHMS),
        "config": {
            "satellites": args.satellites,
            "tasks": args.tasks,
            "planes": args.planes,
            "max_steps": args.max_steps,
            "candidate_k": args.candidate_k,
            "neighbor_k": args.neighbor_k,
            "step_duration_seconds": args.step_duration_seconds,
            "point_observation_seconds": args.point_observation_seconds,
            "fov_deg": args.fov_deg,
            "episodes": args.episodes,
            "seed": args.seed,
            "eval_seed": args.eval_seed,
            "eval_episodes": args.eval_episodes,
            "eval_task_layout": args.eval_task_layout,
            "device": args.device,
            "task_layout": args.task_layout,
            "curriculum_visible_fraction": args.curriculum_visible_fraction,
            "curriculum_ground_track_jitter_deg": args.curriculum_ground_track_jitter_deg,
            "curriculum_time_jitter_steps": args.curriculum_time_jitter_steps,
            "curriculum_payload_match_probability": args.curriculum_payload_match_probability,
            "min_observation_elevation_deg": args.min_observation_elevation_deg,
            "max_off_nadir_deg": args.max_off_nadir_deg,
            "optical_min_sun_elevation_deg": args.optical_min_sun_elevation_deg,
            "min_task_window": args.min_task_window,
            "max_task_window": args.max_task_window,
            "planning_lookahead_steps": args.planning_lookahead_steps,
            "scenario_seed_cycle": args.scenario_seed_cycle,
        },
        "results": {},
    }
    atomic_write_json(suite_path, suite)

    for algorithm in ALGORITHMS:
        if suite_stop.exists():
            suite["status"] = "stopped"
            break
        run_dir = args.base_dir / algorithm
        run_dir.mkdir(parents=True, exist_ok=True)
        child_stop = run_dir / "STOP"
        if child_stop.exists():
            child_stop.unlink()
        existing = read_json(run_dir / "metrics.json")
        suite.update(
            {
                "current_algorithm": algorithm,
                "stage": "training",
                "updated_at": time.time(),
            }
        )
        atomic_write_json(suite_path, suite)

        if not (args.resume and existing.get("status") == "complete"):
            exit_code = subprocess.call(
                training_command(algorithm, args, run_dir), cwd=ROOT
            )
            metrics = read_json(run_dir / "metrics.json")
            if exit_code != 0 or metrics.get("status") == "failed":
                suite.update(
                    {
                        "status": "failed",
                        "stage": "training",
                        "updated_at": time.time(),
                        "error": f"{algorithm} training exited with {exit_code}",
                    }
                )
                atomic_write_json(suite_path, suite)
                return
            if metrics.get("status") == "stopped":
                suite["status"] = "stopped"
                break

        existing_evaluation = read_json(run_dir / "eval" / "summary.json")
        if args.resume and existing_evaluation.get("status") == "complete":
            suite["results"][algorithm] = existing_evaluation
            suite["updated_at"] = time.time()
            atomic_write_json(suite_path, suite)
            continue

        suite.update({"stage": "evaluation", "updated_at": time.time()})
        atomic_write_json(suite_path, suite)
        exit_code = subprocess.call(
            evaluate_command(
                run_dir,
                args.eval_seed,
                args.eval_episodes,
                args.device,
                args.eval_task_layout,
            ),
            cwd=ROOT,
        )
        if exit_code != 0:
            suite.update(
                {
                    "status": "failed",
                    "updated_at": time.time(),
                    "error": f"{algorithm} evaluation exited with {exit_code}",
                }
            )
            atomic_write_json(suite_path, suite)
            return
        suite["results"][algorithm] = read_json(run_dir / "eval" / "summary.json")
        suite["updated_at"] = time.time()
        atomic_write_json(suite_path, suite)

    if suite.get("status") == "running":
        suite.update(
            {
                "status": "complete",
                "stage": "complete",
                "current_algorithm": None,
                "updated_at": time.time(),
            }
        )
    else:
        suite["updated_at"] = time.time()
    atomic_write_json(suite_path, suite)


if __name__ == "__main__":
    main()
