from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (701, 702, 703)
DEFAULT_SCALES = (64, 128, 256)
CORE_METHODS = (
    "oasis_graph",
    "mlp_opportunity",
    "ippo",
    "mappo",
    "qmix",
    "ps_dqn",
)
ABLATION_METHODS = (
    "no_duration_correction",
    "no_factor_messages",
    "no_learned_bids",
    "no_opportunity_balance",
)
METRICS = (
    "mean_episode_reward",
    "completed_tasks",
    "cooperative_completed_tasks",
    "total_priority_completed",
    "mean_observation_quality",
    "decision_opportunity_rate",
    "task_decision_opportunity_rate",
    "avoidable_idle_actions",
    "forced_idle_actions",
    "total_conflicts",
    "total_ground_conflicts",
    "total_invalid_actions",
    "total_window_misses",
)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    trainer: str
    architecture: str | None = None
    flags: tuple[str, ...] = ()
    phase: str = "core"
    tasks: int = 3072


@dataclass(frozen=True)
class TrainJob:
    method: MethodSpec
    seed: int
    run_dir: Path

    @property
    def job_id(self) -> str:
        return f"train:{self.method.name}:seed{self.seed}"


@dataclass(frozen=True)
class EvalJob:
    method: MethodSpec
    seed: int
    scale: int
    tasks: int
    planes: int
    run_dir: Path
    output_dir: Path

    @property
    def job_id(self) -> str:
        return f"eval:{self.method.name}:seed{self.seed}:n{self.scale}"


def parse_int_csv(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def parse_gpu_csv(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError(
            "expected comma-separated nonnegative GPU indices"
        )
    return values


def parse_text_csv(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def method_specs(phases: Iterable[str], sparsity_tasks: Iterable[int]) -> list[MethodSpec]:
    selected = set(phases)
    specs: list[MethodSpec] = []
    if "core" in selected:
        specs.extend(
            [
                MethodSpec("oasis_graph", "oasis", "opportunity_graph"),
                MethodSpec("mlp_opportunity", "oasis", "mlp"),
                MethodSpec("ippo", "ppo"),
                MethodSpec("mappo", "ppo"),
                MethodSpec("qmix", "qmix"),
                MethodSpec("ps_dqn", "dqn"),
            ]
        )
    if "ablations" in selected:
        specs.extend(
            [
                MethodSpec(
                    "no_duration_correction",
                    "oasis",
                    "opportunity_graph",
                    ("--no-opportunity-gae",),
                    "ablations",
                ),
                MethodSpec(
                    "no_factor_messages",
                    "oasis",
                    "opportunity_graph",
                    ("--no-graph-factor-messages",),
                    "ablations",
                ),
                MethodSpec(
                    "no_learned_bids",
                    "oasis",
                    "opportunity_graph",
                    ("--no-learned-resource-bids",),
                    "ablations",
                ),
                MethodSpec(
                    "no_opportunity_balance",
                    "oasis",
                    "opportunity_graph",
                    (
                        "--no-opportunity-balancing",
                        "--no-semantic-opportunity-balancing",
                    ),
                    "ablations",
                ),
            ]
        )
    if "sparsity" in selected:
        for tasks in sparsity_tasks:
            if tasks == 3072:
                continue
            specs.extend(
                [
                    MethodSpec(
                        f"oasis_graph_t{tasks}",
                        "oasis",
                        "opportunity_graph",
                        (),
                        "sparsity",
                        tasks,
                    ),
                    MethodSpec(
                        f"mlp_opportunity_t{tasks}",
                        "oasis",
                        "mlp",
                        (),
                        "sparsity",
                        tasks,
                    ),
                ]
            )
    unknown = selected.difference({"core", "ablations", "sparsity"})
    if unknown:
        raise ValueError(f"unknown phases: {sorted(unknown)}")
    return specs


def common_training_args(
    args: argparse.Namespace,
    spec: MethodSpec,
    seed: int,
    run_dir: Path,
    ephemeris: Path,
    train_catalog: Path,
) -> list[str]:
    return [
        "--satellites",
        "64",
        "--tasks",
        str(spec.tasks),
        "--planes",
        "8",
        "--max-steps",
        str(args.max_steps),
        "--candidate-k",
        str(args.candidate_k),
        "--neighbor-k",
        str(args.neighbor_k),
        "--step-duration-seconds",
        str(args.step_seconds),
        "--point-observation-seconds",
        "5",
        "--fov-deg",
        "45",
        "--task-layout",
        "catalog",
        "--ephemeris-cache",
        str(ephemeris),
        "--task-catalog",
        str(train_catalog),
        "--min-observation-elevation-deg",
        "3",
        "--max-off-nadir-deg",
        "45",
        "--optical-min-sun-elevation-deg",
        "8",
        "--min-task-window",
        "40",
        "--max-task-window",
        "160",
        "--planning-lookahead-steps",
        str(args.lookahead_steps),
        "--scenario-seed-cycle",
        "0",
        "--episodes",
        str(args.episodes),
        "--seed",
        str(seed),
        "--hidden-dim",
        "128",
        "--hidden-layers",
        "2",
        "--activation",
        "relu",
        "--layer-norm",
        "--lr",
        "0.0003",
        "--weight-decay",
        "0.01",
        "--gamma",
        "0.97",
        "--grad-clip",
        "10",
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--metrics-every",
        "1",
        "--run-dir",
        str(run_dir),
        "--device",
        "cuda",
    ]


def training_command(
    args: argparse.Namespace,
    job: TrainJob,
    ephemeris: Path,
    train_catalog: Path,
) -> list[str]:
    common = common_training_args(
        args, job.method, job.seed, job.run_dir, ephemeris, train_catalog
    )
    spec = job.method
    if spec.trainer == "oasis":
        command = [
            sys.executable,
            str(ROOT / "scripts" / "train_oasis.py"),
            "--architecture",
            str(spec.architecture),
            *common,
            "--no-adaptive-curriculum",
            "--opportunity-gae",
            "--opportunity-balancing",
            "--semantic-opportunity-balancing",
            "--min-decision-samples",
            "1024",
            "--max-buffered-episodes",
            "8",
            "--gae-lambda",
            "0.95",
            "--clip-coef",
            "0.2",
            "--value-clip-coef",
            "0.2",
            "--entropy-coef",
            "0.01",
            "--value-coef",
            "0.5",
            "--update-epochs",
            "4",
            "--minibatch-size",
            "1024",
            "--normalize-advantages",
            "--reward-scale",
            "1",
            "--reward-clip",
            "10",
            "--progress-every-steps",
            "10",
            *spec.flags,
        ]
    elif spec.trainer == "ppo":
        command = [
            sys.executable,
            str(ROOT / "scripts" / "train_ppo.py"),
            "--algorithm",
            spec.name,
            *common,
            "--gae-lambda",
            "0.95",
            "--clip-coef",
            "0.2",
            "--value-clip-coef",
            "0.2",
            "--entropy-coef",
            "0.01",
            "--value-coef",
            "0.5",
            "--update-epochs",
            "4",
            "--minibatch-size",
            "1024",
            "--normalize-advantages",
            "--reward-scale",
            "1",
            "--reward-clip",
            "10",
        ]
    elif spec.trainer == "qmix":
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
            "1",
            "--double-q",
            "--reward-scale",
            "1",
            "--reward-clip",
            "10",
            "--epsilon-start",
            "0.8",
            "--epsilon-end",
            "0.05",
            "--epsilon-decay-steps",
            "60000",
            "--target-update-steps",
            "800",
            "--target-tau",
            "1",
            "--gradient-steps",
            "1",
        ]
    else:
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
            "64",
            "--optimizer",
            "adamw",
            "--loss-type",
            "huber",
            "--huber-delta",
            "1",
            "--double-dqn",
            "--local-observation-only",
            "--reward-scale",
            "1",
            "--reward-clip",
            "10",
            "--epsilon-start",
            "0.8",
            "--epsilon-end",
            "0.05",
            "--epsilon-decay-steps",
            "60000",
            "--target-update-steps",
            "800",
            "--target-tau",
            "1",
            "--train-frequency",
            "1",
            "--gradient-steps",
            "1",
        ]
    if args.resume and (job.run_dir / "checkpoints" / "latest.pt").is_file():
        command.append("--resume")
    return command


def evaluation_command(
    args: argparse.Namespace,
    job: EvalJob,
    ephemeris: Path,
    eval_catalog: Path,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_marl.py"),
        "--checkpoint",
        str(job.run_dir / "checkpoints" / "latest.pt"),
        "--output",
        str(job.output_dir / "rollout.json"),
        "--satellites",
        str(job.scale),
        "--tasks",
        str(job.tasks),
        "--planes",
        str(job.planes),
        "--max-steps",
        str(args.max_steps),
        "--candidate-k",
        str(args.candidate_k),
        "--neighbor-k",
        str(args.neighbor_k),
        "--ephemeris-cache",
        str(ephemeris),
        "--task-catalog",
        str(eval_catalog),
        "--eval-task-layout",
        "catalog",
        "--seed",
        str(args.eval_seed),
        "--eval-episodes",
        str(args.eval_episodes),
        "--no-capture-frames",
        "--device",
        "cuda",
    ]


def subprocess_call(command: list[str], log_path: Path, gpu_id: int) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write("\n$ " + subprocess.list2cmdline(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        return process.wait()


def run_checked(command: list[str], log_path: Path | None = None) -> None:
    if log_path is None:
        subprocess.run(command, cwd=ROOT, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )


def prepare_protocol(
    args: argparse.Namespace,
) -> tuple[dict[int, Path], Path, dict[int, Path]]:
    scenario = args.run_root / "scenario"
    scenario.mkdir(parents=True, exist_ok=True)
    train_catalog = args.train_catalog.resolve()
    eval_catalog_64 = args.eval_catalog.resolve()
    for path in (train_catalog, eval_catalog_64):
        if not path.is_file():
            raise FileNotFoundError(f"task catalog not found: {path}")

    max_tasks = max(scale * args.tasks_per_satellite for scale in args.scales)
    scale_catalog_dir = scenario / "catalogs"
    scale_eval_catalog = scale_catalog_dir / "scale_test_requests.csv"
    scale_manifest = scale_catalog_dir / "scale_catalog_manifest.json"
    if (
        args.force_rebuild_scenario
        or not scale_eval_catalog.is_file()
        or not scale_manifest.is_file()
    ):
        scale_catalog_dir.mkdir(parents=True, exist_ok=True)
        run_checked(
            [
                sys.executable,
                str(ROOT / "scripts" / "generate_task_catalogs.py"),
                "--output-dir",
                str(scale_catalog_dir),
                "--train-count",
                "3072",
                "--test-count",
                str(max_tasks),
                "--train-seed",
                "2701",
                "--test-seed",
                "2702",
                "--max-steps",
                str(args.max_steps),
                "--min-window-steps",
                "40",
                "--max-window-steps",
                "160",
                "--area-fraction",
                "0",
                "--file-prefix",
                "scale_",
                "--overwrite",
            ],
            scenario / "generate_scale_catalogs.log",
        )

    ephemerides: dict[int, Path] = {}
    eval_catalogs: dict[int, Path] = {}
    for scale in args.scales:
        planes = max(1, args.base_planes * scale // 64)
        cache = scenario / f"kepler_n{scale}.npz"
        elements = scenario / f"orbital_elements_n{scale}.csv"
        if args.force_rebuild_scenario:
            cache.unlink(missing_ok=True)
            elements.unlink(missing_ok=True)
        if not cache.is_file():
            run_checked(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_kepler_ephemeris.py"),
                    "--output",
                    str(cache),
                    "--elements-output",
                    str(elements),
                    "--start-utc",
                    args.start_utc,
                    "--satellites",
                    str(scale),
                    "--planes",
                    str(planes),
                    "--max-steps",
                    str(args.max_steps),
                    "--lookahead-steps",
                    str(args.lookahead_steps),
                    "--step-duration-seconds",
                    str(args.step_seconds),
                    "--altitude-km",
                    "550",
                    "--eccentricity",
                    "0.001",
                    "--inclination-deg",
                    "97.6",
                    "--walker-phasing",
                    "1",
                ],
                scenario / f"build_kepler_n{scale}.log",
            )
        ephemerides[scale] = cache
        eval_catalogs[scale] = (
            eval_catalog_64 if scale == 64 else scale_eval_catalog
        )

    validation_jobs = [
        ("train_n64", 64, 3072, ephemerides[64], train_catalog),
        *[
            (
                f"eval_n{scale}",
                scale,
                scale * args.tasks_per_satellite,
                ephemerides[scale],
                eval_catalogs[scale],
            )
            for scale in args.scales
        ],
    ]
    for name, scale, tasks, cache, catalog in validation_jobs:
        report = scenario / f"{name}_scenario_report.json"
        if report.is_file() and not args.force_rebuild_scenario:
            continue
        run_checked(
            [
                sys.executable,
                str(ROOT / "scripts" / "validate_orbit_scenario.py"),
                "--ephemeris-cache",
                str(cache),
                "--task-catalog",
                str(catalog),
                "--satellites",
                str(scale),
                "--tasks",
                str(tasks),
                "--max-steps",
                str(args.max_steps),
                "--lookahead-steps",
                str(args.lookahead_steps),
                "--step-duration-seconds",
                str(args.step_seconds),
                "--min-elevation-deg",
                "3",
                "--max-off-nadir-deg",
                "45",
                "--output",
                str(report),
            ],
            scenario / f"validate_{name}.log",
        )
    return ephemerides, train_catalog, eval_catalogs


def build_jobs(
    args: argparse.Namespace, specs: Iterable[MethodSpec]
) -> tuple[list[TrainJob], list[EvalJob]]:
    train_jobs: list[TrainJob] = []
    eval_jobs: list[EvalJob] = []
    for spec in specs:
        for seed in args.seeds:
            run_dir = args.run_root / "training" / spec.name / f"seed_{seed}"
            train_job = TrainJob(spec, seed, run_dir)
            train_jobs.append(train_job)
            scales = args.scales if spec.phase == "core" else (64,)
            for scale in scales:
                tasks = (
                    scale * args.tasks_per_satellite
                    if spec.phase == "core"
                    else spec.tasks
                )
                planes = max(1, args.base_planes * scale // 64)
                eval_jobs.append(
                    EvalJob(
                        spec,
                        seed,
                        scale,
                        tasks,
                        planes,
                        run_dir,
                        args.run_root
                        / "evaluation"
                        / spec.name
                        / f"seed_{seed}"
                        / f"n{scale}",
                    )
                )
    return train_jobs, eval_jobs


class Pipeline:
    def __init__(
        self,
        args: argparse.Namespace,
        train_jobs: list[TrainJob],
        eval_jobs: list[EvalJob],
    ) -> None:
        self.args = args
        self.path = args.run_root / "pipeline.json"
        self.lock = threading.Lock()
        previous = read_json(self.path) if args.resume else {}
        previous_jobs = {
            item.get("id"): item
            for item in previous.get("jobs", [])
            if isinstance(item, dict)
        }
        jobs: list[dict[str, Any]] = []
        for job, kind in [
            *((item, "training") for item in train_jobs),
            *((item, "evaluation") for item in eval_jobs),
        ]:
            old = previous_jobs.get(job.job_id, {})
            jobs.append(
                {
                    "id": job.job_id,
                    "kind": kind,
                    "method": job.method.name,
                    "seed": job.seed,
                    "scale": getattr(job, "scale", 64),
                    "tasks": getattr(job, "tasks", job.method.tasks),
                    "status": old.get("status", "pending"),
                    "gpu": old.get("gpu"),
                    "started_at": old.get("started_at"),
                    "finished_at": old.get("finished_at"),
                    "exit_code": old.get("exit_code"),
                }
            )
        self.payload: dict[str, Any] = {
            "status": "running",
            "stage": "preparing",
            "message": "Preparing point-target AAAI experiment protocol.",
            "started_at": previous.get("started_at", time.time()),
            "updated_at": time.time(),
            "tensorboard_url": f"http://127.0.0.1:{args.tensorboard_port}",
            "protocol": {
                "point_targets_only": True,
                "train_scale": {"satellites": 64, "tasks": 3072, "planes": 8},
                "seeds": list(args.seeds),
                "eval_seed_start": args.eval_seed,
                "eval_episodes_per_training_seed": args.eval_episodes,
                "scales": list(args.scales),
                "point_observation_seconds": 5,
                "step_duration_seconds": args.step_seconds,
                "fov_deg": 45,
                "max_steps": args.max_steps,
                "fairness": (
                    "same 64-satellite environment interactions, task catalog, "
                    "seeds, architecture width, and paired held-out scenarios"
                ),
            },
            "training_runs": [
                {
                    "method": job.method.name,
                    "seed": job.seed,
                    "metrics": self.web_path(job.run_dir / "metrics.json"),
                }
                for job in train_jobs
            ],
            "stages": previous.get("stages", []),
            "jobs": jobs,
        }
        self.write()

    @staticmethod
    def web_path(path: Path) -> str:
        try:
            return "/" + path.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            return str(path.resolve())

    def write(self) -> None:
        self.payload["updated_at"] = time.time()
        atomic_json(self.path, self.payload)

    def set_stage(self, stage: str, message: str) -> None:
        with self.lock:
            self.payload["stage"] = stage
            self.payload["message"] = message
            stages = self.payload.setdefault("stages", [])
            if not stages or stages[-1].get("name") != stage:
                stages.append(
                    {"name": stage, "status": "running", "updated_at": time.time()}
                )
            self.write()

    def update_job(
        self, job_id: str, status: str, gpu: int, exit_code: int | None = None
    ) -> None:
        with self.lock:
            item = next(item for item in self.payload["jobs"] if item["id"] == job_id)
            item["status"] = status
            item["gpu"] = gpu
            if status == "running":
                item["started_at"] = time.time()
            if status in {"complete", "failed", "skipped"}:
                item["finished_at"] = time.time()
                item["exit_code"] = exit_code
            complete = sum(
                entry["status"] in {"complete", "skipped"}
                for entry in self.payload["jobs"]
            )
            total = len(self.payload["jobs"])
            self.payload["message"] = f"{complete}/{total} jobs complete; {job_id}: {status}"
            self.write()

    def finish(self, status: str, message: str) -> None:
        with self.lock:
            self.payload["status"] = status
            self.payload["stage"] = status
            self.payload["message"] = message
            self.payload["finished_at"] = time.time()
            stages = self.payload.setdefault("stages", [])
            stages.append(
                {"name": status, "status": status, "updated_at": time.time()}
            )
            self.write()


def execute_pool(
    items: list[TrainJob] | list[EvalJob],
    gpu_ids: tuple[int, ...],
    pipeline: Pipeline,
    command_for: Any,
    log_for: Any,
    complete_for: Any,
) -> bool:
    work: queue.Queue[Any] = queue.Queue()
    for item in items:
        work.put(item)
    failure = threading.Event()

    def worker(gpu_id: int) -> None:
        while not failure.is_set():
            try:
                item = work.get_nowait()
            except queue.Empty:
                return
            try:
                if complete_for(item):
                    pipeline.update_job(item.job_id, "skipped", gpu_id, 0)
                    continue
                pipeline.update_job(item.job_id, "running", gpu_id)
                exit_code = subprocess_call(command_for(item), log_for(item), gpu_id)
                if exit_code:
                    pipeline.update_job(item.job_id, "failed", gpu_id, exit_code)
                    failure.set()
                else:
                    pipeline.update_job(item.job_id, "complete", gpu_id, 0)
            finally:
                work.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu,), daemon=False)
        for gpu in gpu_ids
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return not failure.is_set()


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def summarize(args: argparse.Namespace, eval_jobs: Iterable[EvalJob]) -> None:
    groups: dict[tuple[str, int, int], list[EvalJob]] = {}
    for job in eval_jobs:
        if (job.output_dir / "summary.json").is_file():
            groups.setdefault((job.method.name, job.scale, job.tasks), []).append(job)
    rows: list[dict[str, Any]] = []
    for (method, scale, tasks), jobs in sorted(groups.items()):
        row: dict[str, Any] = {
            "method": method,
            "satellites": scale,
            "tasks": tasks,
            "seeds": len(jobs),
            "eval_episodes_per_seed": args.eval_episodes,
        }
        for metric in METRICS:
            seed_means: list[float] = []
            for job in jobs:
                summary = read_json(job.output_dir / "summary.json")
                value = (
                    summary.get("benchmark", {})
                    .get("aggregate", {})
                    .get(metric, {})
                    .get("mean")
                )
                if value is not None:
                    seed_means.append(float(value))
            if seed_means:
                row[metric] = sum(seed_means) / len(seed_means)
                row[f"{metric}_std"] = sample_std(seed_means)
        row["completion_rate"] = row.get("completed_tasks", 0.0) / max(1, tasks)
        row["conflicts_per_1000_agent_steps"] = (
            1000.0
            * row.get("total_conflicts", 0.0)
            / max(1, scale * args.max_steps)
        )
        train_seconds: list[float] = []
        parameter_counts: list[int] = []
        for job in jobs:
            metrics = read_json(job.run_dir / "metrics.json")
            history = metrics.get("history", [])
            if isinstance(history, list):
                train_seconds.append(
                    sum(float(item.get("episode_seconds", 0.0)) for item in history)
                )
            count = (
                metrics.get("algorithm", {})
                .get("network", {})
                .get("parameter_count")
            )
            if count is not None:
                parameter_counts.append(int(count))
        if train_seconds:
            row["train_seconds"] = sum(train_seconds) / len(train_seconds)
            row["train_seconds_std"] = sample_std(train_seconds)
        if parameter_counts:
            row["parameter_count"] = parameter_counts[0]
        rows.append(row)

    payload = {
        "status": "complete" if rows else "pending",
        "created_at": time.time(),
        "statistical_unit": (
            "mean and sample standard deviation across independent training seeds; "
            "each seed is evaluated on the same paired held-out episodes"
        ),
        "rows": rows,
    }
    atomic_json(args.run_root / "results.json", payload)
    csv_path = args.run_root / "results.csv"
    if rows:
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Matched-budget point-target AAAI suite for OASIS-Graph, RL baselines, "
            "mechanism ablations, sparsity, and zero-shot scale transfer."
        )
    )
    parser.add_argument(
        "--run-root", type=Path, default=ROOT / "runs" / "point_aaai_final"
    )
    parser.add_argument("--phases", type=parse_text_csv, default=("core", "ablations"))
    parser.add_argument("--seeds", type=parse_int_csv, default=DEFAULT_SEEDS)
    parser.add_argument("--gpu-ids", type=parse_gpu_csv, default=(0, 1))
    parser.add_argument("--scales", type=parse_int_csv, default=DEFAULT_SCALES)
    parser.add_argument(
        "--sparsity-tasks", type=parse_int_csv, default=(768, 1536, 3072)
    )
    parser.add_argument("--tasks-per-satellite", type=int, default=48)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=12001)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--lookahead-steps", type=int, default=30)
    parser.add_argument("--step-seconds", type=float, default=30.0)
    parser.add_argument("--candidate-k", type=int, default=24)
    parser.add_argument("--neighbor-k", type=int, default=6)
    parser.add_argument("--base-planes", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--start-utc", default="2026-07-20T00:00:00Z")
    parser.add_argument(
        "--train-catalog",
        type=Path,
        default=ROOT / "data" / "targets" / "train_requests.csv",
    )
    parser.add_argument(
        "--eval-catalog",
        type=Path,
        default=ROOT / "data" / "targets" / "test_requests.csv",
    )
    parser.add_argument("--tensorboard-port", type=int, default=6010)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-rebuild-scenario", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.run_root = args.run_root.expanduser().resolve()
    args.run_root.mkdir(parents=True, exist_ok=True)
    specs = method_specs(args.phases, args.sparsity_tasks)
    train_jobs, eval_jobs = build_jobs(args, specs)
    pipeline = Pipeline(args, train_jobs, eval_jobs)

    if args.dry_run:
        protocol = {
            "train_jobs": [asdict(job.method) | {"seed": job.seed} for job in train_jobs],
            "eval_jobs": [
                {
                    "method": job.method.name,
                    "seed": job.seed,
                    "scale": job.scale,
                    "tasks": job.tasks,
                }
                for job in eval_jobs
            ],
        }
        atomic_json(args.run_root / "dry_run_jobs.json", protocol)
        pipeline.finish("complete", "Dry run completed; no training was started.")
        return

    try:
        pipeline.set_stage("preparing", "Building reusable Kepler caches and catalogs.")
        ephemerides, train_catalog, eval_catalogs = prepare_protocol(args)
        if args.prepare_only:
            pipeline.finish("complete", "Scenario preparation completed.")
            return

        pipeline.set_stage("training", "Running matched 64-satellite training jobs.")
        train_ok = execute_pool(
            train_jobs,
            args.gpu_ids,
            pipeline,
            lambda job: training_command(args, job, ephemerides[64], train_catalog),
            lambda job: args.run_root / "logs" / f"{job.job_id.replace(':', '_')}.log",
            lambda job: args.resume
            and read_json(job.run_dir / "metrics.json").get("status") == "complete",
        )
        if not train_ok:
            summarize(args, eval_jobs)
            pipeline.finish("failed", "A training job failed; inspect pipeline.json and logs.")
            raise SystemExit(1)

        pipeline.set_stage(
            "evaluation", "Running paired held-out evaluations at 64/128/256 satellites."
        )
        eval_ok = execute_pool(
            eval_jobs,
            args.gpu_ids,
            pipeline,
            lambda job: evaluation_command(
                args, job, ephemerides[job.scale], eval_catalogs[job.scale]
            ),
            lambda job: args.run_root / "logs" / f"{job.job_id.replace(':', '_')}.log",
            lambda job: args.resume
            and read_json(job.output_dir / "summary.json").get("status") == "complete",
        )
        summarize(args, eval_jobs)
        if not eval_ok:
            pipeline.finish(
                "failed", "An evaluation job failed; completed results were preserved."
            )
            raise SystemExit(1)
        pipeline.finish(
            "complete",
            "All requested point-target training and paired evaluations are complete.",
        )
    except Exception as exc:
        if pipeline.payload.get("status") == "running":
            pipeline.finish("failed", f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
