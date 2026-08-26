from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_provenance() -> dict[str, Any]:
    def capture(*arguments: str) -> str | None:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = capture("status", "--porcelain")
    return {
        "commit": capture("rev-parse", "HEAD"),
        "branch": capture("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
    }


def parse_int_list(value: str | None, default: list[int]) -> list[int]:
    if not value:
        return list(default)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def complete_json(path: Path, expected_episodes: int | None = None) -> bool:
    if not path.exists():
        return False
    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return False
    if payload.get("status") != "complete":
        return False
    if expected_episodes is None:
        return True
    benchmark = payload.get("benchmark") or {}
    return int(benchmark.get("episodes", 0)) >= expected_episodes


@dataclass
class Job:
    job_id: str
    stage: str
    command: list[str]
    log_path: Path
    markers: list[Path] = field(default_factory=list)
    expected_eval_episodes: int | None = None
    gpu: str | None = None

    def command_text(self) -> str:
        return shlex.join(self.command)

    def is_complete(self) -> bool:
        if self.expected_eval_episodes is not None:
            return bool(self.markers) and complete_json(
                self.markers[0], self.expected_eval_episodes
            )
        if self.stage == "train":
            if len(self.markers) < 2 or not self.markers[1].exists():
                return False
            return complete_json(self.markers[0])
        return all(marker.exists() for marker in self.markers)


def apply_smoke_profile(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result["training"].update(
        {
            "satellites": 8,
            "planes": 2,
            "tasks": 64,
            "episodes": 1,
            "candidate_k": 8,
            "neighbor_k": 2,
            "seeds": [701],
        }
    )
    result["evaluation"]["episodes"] = 1
    result["stress"]["episodes"] = 1
    result["evaluation"]["scales"] = [
        {"satellites": 8, "planes": 2, "tasks": 64},
        {"satellites": 16, "planes": 4, "tasks": 128},
    ]
    result["catalog"]["window_steps"] = [2, 6]
    result["stress"]["scale"] = {"satellites": 16, "planes": 4}
    result["stress"]["task_density"] = [
        {"name": "density_025", "tasks": 32},
        {"name": "density_050", "tasks": 64},
        {"name": "density_100", "tasks": 128},
    ]
    result["physics"]["max_steps"] = 16
    result["physics"]["planning_lookahead_steps"] = 4
    return result


class Suite:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config_path = args.config.expanduser().resolve()
        self.config = read_json(self.config_path)
        if args.profile == "smoke":
            self.config = apply_smoke_profile(self.config)
        self.run_root = args.run_root.expanduser().resolve()
        self.scenario_root = (
            args.scenario_root.expanduser().resolve()
            if args.scenario_root
            else self.run_root / "scenario"
        )
        self.data_root = (
            args.data_root.expanduser().resolve()
            if args.data_root
            else self.run_root / "catalogs"
        )
        self.logs_root = self.run_root / "logs"
        self.state_path = self.run_root / "suite.json"
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.logs_root.mkdir(parents=True, exist_ok=True)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.scenario_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.run_root / "resolved_config.json", self.config)
        self.provenance = {
            "git": git_provenance(),
            "config_path": str(self.config_path),
            "config_sha256": sha256_path(self.config_path),
            "python": sys.version,
            "executable": sys.executable,
        }
        atomic_write_json(self.run_root / "provenance.json", self.provenance)
        self.python = sys.executable
        configured_methods = list(self.config["methods"])
        self.methods = parse_str_list(args.methods, configured_methods)
        unknown = sorted(set(self.methods) - set(configured_methods))
        if unknown:
            raise ValueError(f"Unknown methods: {', '.join(unknown)}")
        self.seeds = parse_int_list(
            args.seeds, [int(value) for value in self.config["training"]["seeds"]]
        )
        configured_gpus = [str(value) for value in self.config["resources"]["gpus"]]
        self.gpus = parse_str_list(args.gpus, configured_gpus)
        if not self.gpus:
            raise ValueError("At least one GPU id is required.")
        self.state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            try:
                previous = read_json(self.state_path)
            except (OSError, ValueError):
                previous = {}
        else:
            previous = {}
        return {
            "version": 1,
            "experiment": self.config["experiment"],
            "profile": self.args.profile,
            "status": previous.get("status", "pending"),
            "stage": previous.get("stage", "pending"),
            "created_at": previous.get("created_at", time.time()),
            "updated_at": time.time(),
            "config": str(self.config_path),
            "run_root": str(self.run_root),
            "methods": self.methods,
            "seeds": self.seeds,
            "gpus": self.gpus,
            "provenance": self.provenance,
            "jobs": dict(previous.get("jobs") or {}),
            "message": previous.get("message", ""),
        }

    def save_state(self, *, status: str, stage: str, message: str) -> None:
        self.state.update(
            {
                "status": status,
                "stage": stage,
                "message": message,
                "updated_at": time.time(),
            }
        )
        atomic_write_json(self.state_path, self.state)
        atomic_write_json(self.run_root / "pipeline.json", self.state)

    @property
    def train_catalog(self) -> Path:
        return self.data_root / "thousand_train_requests.csv"

    @property
    def test_catalog(self) -> Path:
        return self.data_root / "thousand_test_requests.csv"

    def cache_path(self, satellites: int) -> Path:
        return self.scenario_root / f"kepler_n{satellites}.npz"

    def train_dir(self, method: str, seed: int) -> Path:
        return self.run_root / "training" / method / f"seed_{seed}"

    def base_catalog_command(self) -> list[str]:
        training = self.config["training"]
        evaluation = self.config["evaluation"]
        catalog = self.config["catalog"]
        largest_tasks = max(int(scale["tasks"]) for scale in evaluation["scales"])
        window_min, window_max = catalog["window_steps"]
        observers_min, observers_max = catalog["cooperative_observers"]
        fractions = catalog["cooperation_fractions"]
        return [
            self.python,
            "scripts/generate_task_catalogs.py",
            "--output-dir",
            str(self.data_root),
            "--file-prefix",
            "thousand_",
            "--train-count",
            str(training["tasks"]),
            "--test-count",
            str(largest_tasks),
            "--train-seed",
            str(catalog["train_seed"]),
            "--test-seed",
            str(catalog["test_seed"]),
            "--max-steps",
            str(self.config["physics"]["max_steps"]),
            "--min-window-steps",
            str(window_min),
            "--max-window-steps",
            str(window_max),
            "--minimum-separation-deg",
            str(catalog["minimum_separation_deg"]),
            "--skip-separation-audit",
            "--cooperative-observers-min",
            str(observers_min),
            "--cooperative-observers-max",
            str(observers_max),
            "--single-fraction",
            str(fractions["single"]),
            "--sequential-fraction",
            str(fractions["sequential"]),
            "--simultaneous-fraction",
            str(fractions["simultaneous"]),
            "--overwrite",
        ]

    def cache_command(self, scale: dict[str, Any]) -> list[str]:
        physics = self.config["physics"]
        cache = self.cache_path(int(scale["satellites"]))
        return [
            self.python,
            "scripts/build_kepler_ephemeris.py",
            "--output",
            str(cache),
            "--elements-output",
            str(cache.with_name(cache.stem + "_elements.csv")),
            "--start-utc",
            str(physics["start_utc"]),
            "--satellites",
            str(scale["satellites"]),
            "--planes",
            str(scale["planes"]),
            "--max-steps",
            str(physics["max_steps"]),
            "--lookahead-steps",
            str(physics["planning_lookahead_steps"]),
            "--step-duration-seconds",
            str(physics["step_duration_seconds"]),
            "--altitude-km",
            str(physics["altitude_km"]),
            "--eccentricity",
            str(physics["eccentricity"]),
            "--inclination-deg",
            str(physics["inclination_deg"]),
            "--walker-phasing",
            str(physics["walker_phasing"]),
        ]

    def prepare_jobs(self) -> list[Job]:
        jobs = [
            Job(
                "prepare:base_catalogs",
                "prepare",
                self.base_catalog_command(),
                self.logs_root / "prepare_base_catalogs.log",
                [self.train_catalog, self.test_catalog],
            )
        ]
        scales: dict[int, dict[str, Any]] = {
            int(scale["satellites"]): scale
            for scale in self.config["evaluation"]["scales"]
        }
        training = self.config["training"]
        scales[int(training["satellites"])] = {
            "satellites": training["satellites"],
            "planes": training["planes"],
        }
        for satellites, scale in sorted(scales.items()):
            cache = self.cache_path(satellites)
            jobs.append(
                Job(
                    f"prepare:cache_n{satellites}",
                    "prepare",
                    self.cache_command(scale),
                    self.logs_root / f"prepare_cache_n{satellites}.log",
                    [cache],
                )
            )
        jobs.extend(self.stress_catalog_jobs())
        return jobs

    def common_train_args(self, seed: int, run_dir: Path) -> list[str]:
        training = self.config["training"]
        physics = self.config["physics"]
        observers_min, observers_max = self.config["catalog"][
            "cooperative_observers"
        ]
        window_min, window_max = self.config["catalog"]["window_steps"]
        return [
            "--satellites",
            str(training["satellites"]),
            "--tasks",
            str(training["tasks"]),
            "--planes",
            str(training["planes"]),
            "--max-steps",
            str(physics["max_steps"]),
            "--candidate-k",
            str(training["candidate_k"]),
            "--neighbor-k",
            str(training["neighbor_k"]),
            "--planning-lookahead-steps",
            str(physics["planning_lookahead_steps"]),
            "--step-duration-seconds",
            str(physics["step_duration_seconds"]),
            "--point-observation-seconds",
            str(physics["point_observation_seconds"]),
            "--fov-deg",
            str(physics["fov_deg"]),
            "--max-off-nadir-deg",
            str(physics["fov_deg"]),
            "--min-task-window",
            str(window_min),
            "--max-task-window",
            str(window_max),
            "--cooperative-observers-min",
            str(observers_min),
            "--cooperative-observers-max",
            str(observers_max),
            "--task-layout",
            "catalog",
            "--ephemeris-cache",
            str(self.cache_path(int(training["satellites"]))),
            "--task-catalog",
            str(self.train_catalog),
            "--episodes",
            str(training["episodes"]),
            "--seed",
            str(seed),
            "--scenario-seed-cycle",
            "0",
            "--hidden-dim",
            str(training["hidden_dim"]),
            "--checkpoint-every",
            "10",
            "--run-dir",
            str(run_dir),
            "--device",
            "cuda",
        ]

    @staticmethod
    def oasis_variant(method: str) -> tuple[str, list[str]]:
        architecture = "hierarchical_coalition_graph"
        flags = ["--coalition-reservations", "--weak-mean-field", "--factorized-critic"]
        if method == "oasis_graph":
            return "opportunity_graph", ["--no-coalition-reservations"]
        if method == "mlp_opportunity":
            return "mlp", ["--no-coalition-reservations"]
        if method == "no_reservation":
            flags[0] = "--no-coalition-reservations"
        elif method == "no_weak_mean_field":
            flags[1] = "--no-weak-mean-field"
        elif method == "no_factorized_critic":
            flags[2] = "--no-factorized-critic"
        elif method == "no_duration_correction":
            flags.append("--no-opportunity-gae")
        elif method == "no_opportunity_balance":
            flags.extend(
                ["--no-opportunity-balancing", "--no-semantic-opportunity-balancing"]
            )
        elif method == "no_factor_messages":
            flags.append("--no-graph-factor-messages")
        elif method == "no_learned_bids":
            flags.append("--no-learned-resource-bids")
        elif method != "full":
            raise ValueError(f"Unsupported OASIS method: {method}")
        return architecture, flags

    def train_command(self, method: str, seed: int) -> list[str]:
        method_config = self.config["methods"][method]
        trainer = method_config["trainer"]
        run_dir = self.train_dir(method, seed)
        common = self.common_train_args(seed, run_dir)
        resume = self.args.resume and (run_dir / "checkpoints" / "latest.pt").exists()
        if trainer == "oasis":
            architecture, flags = self.oasis_variant(method)
            balancing_flags = (
                []
                if method == "no_opportunity_balance"
                else ["--semantic-opportunity-balancing"]
            )
            command = [
                self.python,
                "-u",
                "scripts/train_oasis.py",
                "--architecture",
                architecture,
                *flags,
                "--reservation-ttl-steps",
                "2",
                *common,
                *balancing_flags,
                "--no-adaptive-curriculum",
                "--min-decision-samples",
                "32" if self.args.profile == "smoke" else "4096",
                "--max-buffered-episodes",
                "2",
                "--minibatch-size",
                "32" if self.args.profile == "smoke" else "2048",
                "--progress-every-steps",
                "4" if self.args.profile == "smoke" else "10",
                "--tensorboard",
            ]
        elif trainer == "ppo":
            command = [
                self.python,
                "-u",
                "scripts/train_ppo.py",
                "--algorithm",
                method,
                *common,
                "--minibatch-size",
                "32" if self.args.profile == "smoke" else "2048",
            ]
        elif trainer == "qmix":
            command = [
                self.python,
                "-u",
                "scripts/train_qmix.py",
                *common,
                "--buffer-size",
                "64" if self.args.profile == "smoke" else "1200",
                "--batch-size",
                "4" if self.args.profile == "smoke" else "16",
                "--learning-starts",
                "4" if self.args.profile == "smoke" else "128",
            ]
        elif trainer == "dqn":
            command = [
                self.python,
                "-u",
                "scripts/train_dqn.py",
                *common,
                "--buffer-size",
                "64" if self.args.profile == "smoke" else "60000",
                "--batch-size",
                "4" if self.args.profile == "smoke" else "256",
                "--learning-starts",
                "4" if self.args.profile == "smoke" else "2048",
                "--store-agents",
                "8" if self.args.profile == "smoke" else "96",
            ]
        else:
            raise ValueError(f"Unsupported trainer: {trainer}")
        if resume:
            command.append("--resume")
        return command

    def train_jobs(self) -> list[Job]:
        jobs: list[Job] = []
        for method in self.methods:
            for seed in self.seeds:
                run_dir = self.train_dir(method, seed)
                jobs.append(
                    Job(
                        f"train:{method}:seed{seed}",
                        "train",
                        self.train_command(method, seed),
                        self.logs_root / f"train_{method}_seed{seed}.log",
                        [
                            run_dir / "metrics.json",
                            run_dir / "checkpoints" / "latest.pt",
                        ],
                    )
                )
        return jobs

    def evaluation_command(
        self,
        method: str,
        seed: int,
        scale: dict[str, Any],
        *,
        catalog: Path,
        output: Path,
        eval_seed: int,
        eval_episodes: int | None = None,
    ) -> list[str]:
        training = self.config["training"]
        physics = self.config["physics"]
        return [
            self.python,
            "-u",
            "scripts/evaluate_marl.py",
            "--checkpoint",
            str(self.train_dir(method, seed) / "checkpoints" / "latest.pt"),
            "--output",
            str(output),
            "--satellites",
            str(scale["satellites"]),
            "--tasks",
            str(scale["tasks"]),
            "--planes",
            str(scale["planes"]),
            "--max-steps",
            str(physics["max_steps"]),
            "--candidate-k",
            str(training["candidate_k"]),
            "--neighbor-k",
            str(training["neighbor_k"]),
            "--ephemeris-cache",
            str(self.cache_path(int(scale["satellites"]))),
            "--task-catalog",
            str(catalog),
            "--eval-task-layout",
            "catalog",
            "--seed",
            str(eval_seed),
            "--eval-episodes",
            str(eval_episodes or self.config["evaluation"]["episodes"]),
            "--no-capture-frames",
            "--device",
            "cuda",
        ]

    def evaluation_jobs(self) -> list[Job]:
        jobs: list[Job] = []
        episodes = int(self.config["evaluation"]["episodes"])
        seed_map = self.config["evaluation"]["seed_by_training_seed"]
        for method in self.methods:
            for seed in self.seeds:
                base_seed = int(seed_map.get(str(seed), seed + 1000))
                for scale_index, scale in enumerate(
                    self.config["evaluation"]["scales"]
                ):
                    satellites = int(scale["satellites"])
                    output = (
                        self.train_dir(method, seed)
                        / "evaluation"
                        / "base"
                        / f"n{satellites}.json"
                    )
                    jobs.append(
                        Job(
                            f"evaluate:{method}:seed{seed}:n{satellites}",
                            "evaluate",
                            self.evaluation_command(
                                method,
                                seed,
                                scale,
                                catalog=self.test_catalog,
                                output=output,
                                eval_seed=base_seed + 1000 * scale_index,
                            ),
                            self.logs_root
                            / f"eval_{method}_seed{seed}_n{satellites}.log",
                            [output],
                            expected_eval_episodes=episodes,
                        )
                    )
        return jobs

    def stress_specs(self) -> list[dict[str, Any]]:
        stress = self.config["stress"]
        default_tasks = max(
            int(scale["tasks"]) for scale in self.config["evaluation"]["scales"]
        )
        specifications: list[dict[str, Any]] = []
        for index, item in enumerate(stress["geography"]):
            specifications.append(
                {
                    "group": "geography",
                    "name": item["name"],
                    "tasks": default_tasks,
                    "profile": item["profile"],
                    "single": 0.70,
                    "sequential": 0.20,
                    "simultaneous": 0.10,
                    "seed": 6000 + index * 2,
                }
            )
        for index, item in enumerate(stress["cooperation"]):
            specifications.append(
                {
                    "group": "cooperation",
                    "name": item["name"],
                    "tasks": default_tasks,
                    "profile": "global_uniform",
                    "single": item["single"],
                    "sequential": item["sequential"],
                    "simultaneous": item["simultaneous"],
                    "seed": 6200 + index * 2,
                }
            )
        for index, item in enumerate(stress["task_density"]):
            specifications.append(
                {
                    "group": "task_density",
                    "name": item["name"],
                    "tasks": int(item["tasks"]),
                    "profile": "global_uniform",
                    "single": 0.70,
                    "sequential": 0.20,
                    "simultaneous": 0.10,
                    "seed": 6400 + index * 2,
                }
            )
        return specifications

    def stress_catalog(self, specification: dict[str, Any]) -> Path:
        return self.data_root / f"stress_{specification['name']}_test_requests.csv"

    def stress_catalog_jobs(self) -> list[Job]:
        physics = self.config["physics"]
        observers_min, observers_max = self.config["catalog"][
            "cooperative_observers"
        ]
        jobs: list[Job] = []
        for specification in self.stress_specs():
            prefix = f"stress_{specification['name']}_"
            output = self.stress_catalog(specification)
            command = [
                self.python,
                "scripts/generate_task_catalogs.py",
                "--output-dir",
                str(self.data_root),
                "--file-prefix",
                prefix,
                "--train-count",
                # The stress training split is a provenance sentinel only; all
                # policies are trained on the shared base catalog.
                str(min(64, int(specification["tasks"]))),
                "--test-count",
                str(specification["tasks"]),
                "--train-seed",
                str(specification["seed"]),
                "--test-seed",
                str(specification["seed"] + 1),
                "--max-steps",
                str(physics["max_steps"]),
                "--min-window-steps",
                str(self.config["catalog"]["window_steps"][0]),
                "--max-window-steps",
                str(self.config["catalog"]["window_steps"][1]),
                "--minimum-separation-deg",
                "0",
                "--skip-separation-audit",
                "--cooperative-observers-min",
                str(observers_min),
                "--cooperative-observers-max",
                str(observers_max),
                "--single-fraction",
                str(specification["single"]),
                "--sequential-fraction",
                str(specification["sequential"]),
                "--simultaneous-fraction",
                str(specification["simultaneous"]),
                "--spatial-profile",
                str(specification["profile"]),
                "--overwrite",
            ]
            jobs.append(
                Job(
                    f"prepare:stress_catalog:{specification['name']}",
                    "prepare",
                    command,
                    self.logs_root
                    / f"prepare_stress_catalog_{specification['name']}.log",
                    [output],
                )
            )
        return jobs

    def stress_jobs(self) -> list[Job]:
        stress = self.config["stress"]
        episodes = int(stress.get("episodes", self.config["evaluation"]["episodes"]))
        allowed_methods = set(stress["methods"])
        methods = [method for method in self.methods if method in allowed_methods]
        scale = dict(stress["scale"])
        jobs: list[Job] = []
        seed_map = self.config["evaluation"]["seed_by_training_seed"]
        for method in methods:
            for seed in self.seeds:
                eval_seed = int(seed_map.get(str(seed), seed + 1000)) + 5000
                for specification in self.stress_specs():
                    scale_with_tasks = {
                        **scale,
                        "tasks": int(specification["tasks"]),
                    }
                    output = (
                        self.train_dir(method, seed)
                        / "evaluation"
                        / specification["group"]
                        / specification["name"]
                        / f"n{scale['satellites']}.json"
                    )
                    jobs.append(
                        Job(
                            f"stress:{specification['group']}:{specification['name']}:"
                            f"{method}:seed{seed}",
                            "stress",
                            self.evaluation_command(
                                method,
                                seed,
                                scale_with_tasks,
                                catalog=self.stress_catalog(specification),
                                output=output,
                                eval_seed=eval_seed,
                                eval_episodes=episodes,
                            ),
                            self.logs_root
                            / (
                                f"stress_{specification['group']}_"
                                f"{specification['name']}_{method}_seed{seed}.log"
                            ),
                            [output],
                            expected_eval_episodes=episodes,
                        )
                    )
        return jobs

    def record_plan(self, jobs: list[Job], stage: str) -> None:
        plan_path = self.run_root / f"command_plan_{stage}.json"
        atomic_write_json(
            plan_path,
            {
                "stage": stage,
                "generated_at": time.time(),
                "jobs": [
                    {
                        "id": job.job_id,
                        "command": job.command_text(),
                        "log": str(job.log_path),
                        "markers": [str(marker) for marker in job.markers],
                    }
                    for job in jobs
                ],
            },
        )

    def worker_slots(self) -> list[tuple[str, str]]:
        jobs_per_gpu = int(self.config["resources"].get("jobs_per_gpu", 1))
        if jobs_per_gpu < 1:
            raise ValueError("resources.jobs_per_gpu must be at least one.")
        return [
            (f"{gpu}:{worker_index}", gpu)
            for gpu in self.gpus
            for worker_index in range(jobs_per_gpu)
        ]

    def run_jobs(self, jobs: list[Job], stage: str, *, parallel: bool) -> None:
        self.record_plan(jobs, stage)
        pending = [job for job in jobs if not job.is_complete()]
        for job in jobs:
            if job.is_complete():
                self.state["jobs"][job.job_id] = {
                    "status": "complete",
                    "skipped": True,
                    "command": job.command_text(),
                    "log": str(job.log_path),
                }
        if self.args.dry_run:
            for job in pending:
                print(f"[{stage}] {job.job_id}\n  {job.command_text()}")
            self.save_state(
                status="planned",
                stage=stage,
                message=f"Dry run planned {len(jobs)} jobs; {len(pending)} pending.",
            )
            return
        self.save_state(
            status="running",
            stage=stage,
            message=f"Running {len(pending)} pending jobs ({len(jobs)} total).",
        )
        if not parallel:
            for job in pending:
                self._run_serial_job(job)
            return

        slots = self.worker_slots()
        active: dict[str, tuple[Job, subprocess.Popen[Any], Any]] = {}
        queue = list(pending)
        failures: list[str] = []
        while queue or active:
            free_slots = [slot for slot in slots if slot[0] not in active]
            while queue and free_slots:
                slot_id, gpu = free_slots.pop(0)
                job = queue.pop(0)
                handle = job.log_path.open("a", encoding="utf-8")
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = gpu
                process = subprocess.Popen(
                    job.command,
                    cwd=PROJECT_ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
                job.gpu = gpu
                active[slot_id] = (job, process, handle)
                self.state["jobs"][job.job_id] = {
                    "status": "running",
                    "gpu": gpu,
                    "worker_slot": slot_id,
                    "pid": process.pid,
                    "started_at": time.time(),
                    "command": job.command_text(),
                    "log": str(job.log_path),
                }
                self.save_state(
                    status="running",
                    stage=stage,
                    message=f"{len(active)} running, {len(queue)} queued.",
                )
            finished: list[str] = []
            for slot_id, (job, process, handle) in active.items():
                return_code = process.poll()
                if return_code is None:
                    continue
                handle.close()
                complete = return_code == 0 and job.is_complete()
                status = "complete" if complete else "failed"
                self.state["jobs"][job.job_id].update(
                    {
                        "status": status,
                        "return_code": return_code,
                        "finished_at": time.time(),
                    }
                )
                if not complete:
                    failures.append(job.job_id)
                finished.append(slot_id)
            for slot_id in finished:
                active.pop(slot_id)
            if failures and self.args.fail_fast:
                for _, process, handle in active.values():
                    process.terminate()
                    handle.close()
                self.save_state(
                    status="failed",
                    stage=stage,
                    message=f"Failed jobs: {', '.join(failures)}",
                )
                raise RuntimeError(f"Failed jobs: {', '.join(failures)}")
            if queue or active:
                time.sleep(self.args.poll_seconds)
        if failures:
            self.save_state(
                status="failed",
                stage=stage,
                message=f"Failed jobs: {', '.join(failures)}",
            )
            raise RuntimeError(f"Failed jobs: {', '.join(failures)}")
        self.save_state(
            status="running",
            stage=stage,
            message=f"All {len(jobs)} {stage} jobs complete.",
        )

    def _run_serial_job(self, job: Job) -> None:
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.state["jobs"][job.job_id] = {
            "status": "running",
            "started_at": time.time(),
            "command": job.command_text(),
            "log": str(job.log_path),
        }
        self.save_state(
            status="running", stage=job.stage, message=f"Running {job.job_id}."
        )
        with job.log_path.open("a", encoding="utf-8") as handle:
            result = subprocess.run(
                job.command,
                cwd=PROJECT_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        complete = result.returncode == 0 and job.is_complete()
        self.state["jobs"][job.job_id].update(
            {
                "status": "complete" if complete else "failed",
                "return_code": result.returncode,
                "finished_at": time.time(),
            }
        )
        if not complete:
            self.save_state(
                status="failed",
                stage=job.stage,
                message=f"{job.job_id} failed; inspect {job.log_path}.",
            )
            raise RuntimeError(f"Job failed: {job.job_id}")

    def summarize(self) -> None:
        command = [
            self.python,
            "scripts/summarize_thousand_paper_suite.py",
            "--run-root",
            str(self.run_root),
            "--output-dir",
            str(self.run_root / "paper_results"),
        ]
        if self.args.dry_run:
            print(f"[summarize]\n  {shlex.join(command)}")
            return
        result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if result.returncode:
            raise RuntimeError("Result summarization failed.")

    def run(self) -> None:
        stages = (
            ["prepare", "train", "evaluate", "stress", "summarize"]
            if self.args.stage == "all"
            else [self.args.stage]
        )
        for stage in stages:
            if stage == "prepare":
                self.run_jobs(self.prepare_jobs(), stage, parallel=False)
            elif stage == "train":
                self.run_jobs(self.train_jobs(), stage, parallel=True)
            elif stage == "evaluate":
                self.run_jobs(self.evaluation_jobs(), stage, parallel=True)
            elif stage == "stress":
                self.run_jobs(self.stress_jobs(), stage, parallel=True)
            elif stage == "summarize":
                self.summarize()
            else:
                raise ValueError(f"Unknown stage: {stage}")
        if not self.args.dry_run:
            self.save_state(
                status="complete",
                stage="complete",
                message="All requested training, evaluation, stress, and summary stages complete.",
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the reproducible two-GPU thousand-satellite paper suite."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "thousand_paper_suite.json",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=PROJECT_ROOT / "runs" / "thousand_paper_suite",
    )
    parser.add_argument("--scenario-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--profile", choices=["paper", "smoke"], default="paper")
    parser.add_argument(
        "--stage",
        choices=["all", "prepare", "train", "evaluate", "stress", "summarize"],
        default="all",
    )
    parser.add_argument("--methods", default=None, help="Comma-separated method ids.")
    parser.add_argument("--seeds", default=None, help="Comma-separated training seeds.")
    parser.add_argument("--gpus", default=None, help="Comma-separated CUDA device ids.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be positive")
    suite = Suite(args)
    try:
        suite.run()
    except Exception as error:
        suite.save_state(status="failed", stage=suite.state["stage"], message=str(error))
        raise


if __name__ == "__main__":
    main()
