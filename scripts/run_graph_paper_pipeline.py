from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "oasis_graph_aaai"
PIPELINE = RUN_ROOT / "pipeline.json"
RESULTS = RUN_ROOT / "results.json"
SOURCE_SUITE = ROOT / "runs" / "aaai_v3_seed_501" / "suite.json"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


state: dict[str, Any] = {
    "status": "running",
    "stage": "initializing",
    "message": "Preparing the reproducible AAAI experiment pipeline",
    "started_at": time.time(),
    "updated_at": time.time(),
    "pid": os.getpid(),
    "protocol": {
        "train_scale": {"satellites": 64, "tasks": 3072, "planes": 8},
        "zero_shot_scales": [128, 256],
        "horizon_steps": 240,
        "decision_seconds": 30,
        "observation_seconds": 5,
        "fov_deg": 45,
        "candidate_k": 24,
        "neighbor_k": 6,
        "test_layout": "global_random",
        "test_seed_start": 12001,
    },
    "stages": [],
}


def update(stage: str, message: str, **extra: Any) -> None:
    state.update({"stage": stage, "message": message, "updated_at": time.time(), **extra})
    atomic_json(PIPELINE, state)


def artifact_complete(path: Path) -> bool:
    return load_json(path).get("status") == "complete"


def run_command(stage: str, command: list[str], artifact: Path | None = None) -> None:
    if artifact is not None and artifact_complete(artifact):
        state["stages"].append(
            {"name": stage, "status": "skipped_complete", "artifact": str(artifact)}
        )
        update(stage, f"Reusing complete artifact: {artifact}")
        return
    log_path = RUN_ROOT / "logs" / f"{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_text = subprocess.list2cmdline(command)
    record = {
        "name": stage,
        "status": "running",
        "started_at": time.time(),
        "command": command_text,
        "log": str(log_path),
        "artifact": str(artifact) if artifact else None,
    }
    state["stages"].append(record)
    update(stage, command_text)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {command_text}\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    record.update(
        {
            "status": "complete" if completed.returncode == 0 else "failed",
            "finished_at": time.time(),
            "returncode": completed.returncode,
        }
    )
    update(stage, f"return code {completed.returncode}")
    if completed.returncode != 0:
        raise RuntimeError(f"Stage {stage} failed; see {log_path}")
    if artifact is not None and not artifact_complete(artifact):
        raise RuntimeError(f"Stage {stage} returned successfully but artifact is incomplete: {artifact}")


def wait_for_reference_suite() -> None:
    while True:
        suite = load_json(SOURCE_SUITE)
        status = suite.get("status", "missing")
        algorithm = suite.get("current_algorithm", "unknown")
        if status == "complete":
            update("reference_suite", "The 16-satellite five-baseline suite is complete")
            return
        if status in {"failed", "stopped"}:
            raise RuntimeError(f"Reference suite is {status}: {suite.get('error', '')}")
        update(
            "waiting_reference_suite",
            f"Waiting without interference: status={status}, algorithm={algorithm}",
            reference_suite={"status": status, "current_algorithm": algorithm},
        )
        time.sleep(60)


def train_command(name: str, episodes: int, architecture: str, flags: list[str]) -> None:
    run_dir = RUN_ROOT / name
    metrics = run_dir / "metrics.json"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_oasis.py"),
        "--architecture",
        architecture,
        "--satellites", "64",
        "--tasks", "3072",
        "--planes", "8",
        "--max-steps", "240",
        "--candidate-k", "24",
        "--neighbor-k", "6",
        "--episodes", str(episodes),
        "--seed", "701",
        "--task-layout", "global_random",
        "--curriculum-visible-fraction", "0.75",
        "--curriculum-end-fraction", "0.05",
        "--target-opportunity-rate", "0.12",
        "--semantic-opportunity-balancing",
        "--hidden-dim", "128",
        "--min-decision-samples", "1024",
        "--max-buffered-episodes", "8",
        "--checkpoint-every", "10",
        "--point-observation-seconds", "5",
        "--fov-deg", "45",
        "--max-off-nadir-deg", "45",
        "--run-dir", str(run_dir),
        "--device", "cuda",
        *flags,
    ]
    if metrics.exists() and not artifact_complete(metrics) and (run_dir / "checkpoints" / "latest.pt").exists():
        command.append("--resume")
    run_command(f"train_{name}", command, metrics)


def evaluate(name: str, checkpoint: Path, satellites: int, episodes: int) -> None:
    tasks = satellites * 48
    planes = max(4, satellites // 8)
    output_dir = RUN_ROOT / "evaluation" / name / f"n{satellites}"
    summary = output_dir / "summary.json"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_marl.py"),
        "--checkpoint", str(checkpoint),
        "--output", str(output_dir / "rollout.json"),
        "--satellites", str(satellites),
        "--tasks", str(tasks),
        "--planes", str(planes),
        "--max-steps", "240",
        "--candidate-k", "24",
        "--neighbor-k", "6",
        "--eval-task-layout", "global_random",
        "--seed", "12001",
        "--eval-episodes", str(episodes),
        "--no-capture-frames" if satellites > 64 or name != "graph" else "--capture-frames",
        "--device", "cuda",
    ]
    run_command(f"eval_{name}_n{satellites}", command, summary)


def aggregate_results() -> None:
    rows: list[dict[str, Any]] = []
    evaluation_root = RUN_ROOT / "evaluation"
    for summary_path in sorted(evaluation_root.glob("*/n*/summary.json")):
        summary = load_json(summary_path)
        aggregate = summary.get("benchmark", {}).get("aggregate", {})
        method = summary_path.parents[1].name
        satellites = int(summary_path.parent.name.removeprefix("n"))
        row: dict[str, Any] = {
            "method": method,
            "satellites": satellites,
            "tasks": satellites * 48,
            "eval_episodes": summary.get("eval_episodes"),
            "checkpoint_episode": summary.get("checkpoint_episode"),
            "summary": str(summary_path),
            "schedule": summary.get("schedule"),
        }
        for metric in (
            "mean_episode_reward",
            "completed_tasks",
            "cooperative_completed_tasks",
            "total_priority_completed",
            "mean_observation_quality",
            "decision_opportunity_rate",
            "task_decision_opportunity_rate",
            "avoidable_idle_actions",
            "total_conflicts",
            "total_invalid_actions",
            "total_window_misses",
        ):
            values = aggregate.get(metric, {})
            row[metric] = values.get("mean")
            row[f"{metric}_std"] = values.get("std")
        rows.append(row)
    payload = {
        "status": "complete",
        "created_at": time.time(),
        "protocol": state["protocol"],
        "important_caveat": (
            "graph, graph ablations, and mlp64 are trained at N=64; IPPO/MAPPO/QMIX/"
            "PS-DQN transfer rows reuse N=16 checkpoints and are labeled as zero-shot transfer."
        ),
        "rows": rows,
    }
    atomic_json(RESULTS, payload)
    csv_path = RUN_ROOT / "results.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    latex_rows = [row for row in rows if row["satellites"] in {64, 128, 256}]
    latex = [
        r"\subsection{Large-Scale and Ablation Results}",
        r"Table~\ref{tab:graph-scale} is generated directly from independent greedy evaluations.  The transfer baselines use 16-satellite checkpoints and are interpreted only as zero-shot transfer controls; same-scale method evidence comes from Graph, MLP64, and the two graph ablations.",
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\caption{Large-scale evaluation.  Mean $\pm$ standard deviation is over held-out scenarios.}",
        r"\label{tab:graph-scale}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule Method & $N$ & Return & Tasks & Priority & Coop. & Invalid \\",
        r"\midrule",
    ]
    for row in latex_rows:
        method = str(row["method"]).replace("_", r"\_")
        latex.append(
            f"{method} & {row['satellites']} & "
            f"${row['mean_episode_reward']:.4f}\\pm{row['mean_episode_reward_std']:.4f}$ & "
            f"${row['completed_tasks']:.2f}$ & ${row['total_priority_completed']:.2f}$ & "
            f"${row['cooperative_completed_tasks']:.2f}$ & ${row['total_invalid_actions']:.2f}$ \\\\" 
        )
    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    latex_text = "\n".join(latex) + "\n"
    paper_dir = ROOT / "docs" / "aaai27"
    generated = paper_dir / "generated_graph_results.tex"
    generated.write_text(latex_text, encoding="utf-8")
    overleaf = paper_dir / "overleaf_package"
    (overleaf / "generated_graph_results.tex").write_text(latex_text, encoding="utf-8")
    zip_path = paper_dir / "OASIS_Graph_AAAI27_Overleaf.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(overleaf.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(overleaf))


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    update("preflight", "Running all environment, constraint, and MARL unit tests")
    run_command(
        "preflight_tests",
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
    )
    wait_for_reference_suite()

    # Same-scale controlled comparison followed by mechanism ablations.
    train_command("graph", 250, "opportunity_graph", [])
    evaluate("graph", RUN_ROOT / "graph" / "checkpoints" / "latest.pt", 64, 20)
    evaluate("graph", RUN_ROOT / "graph" / "checkpoints" / "latest.pt", 128, 8)
    evaluate("graph", RUN_ROOT / "graph" / "checkpoints" / "latest.pt", 256, 4)

    train_command("mlp64", 250, "mlp", [])
    evaluate("mlp64", RUN_ROOT / "mlp64" / "checkpoints" / "latest.pt", 64, 20)
    train_command("no_factor", 120, "opportunity_graph", ["--no-graph-factor-messages"])
    evaluate("no_factor", RUN_ROOT / "no_factor" / "checkpoints" / "latest.pt", 64, 20)
    train_command("no_bid", 120, "opportunity_graph", ["--no-learned-resource-bids"])
    evaluate("no_bid", RUN_ROOT / "no_bid" / "checkpoints" / "latest.pt", 64, 20)

    # Existing five-baseline checkpoints provide a transparent zero-shot transfer test.
    for method in ("oasis", "ippo", "mappo", "qmix", "ps_dqn"):
        checkpoint = SOURCE_SUITE.parent / method / "checkpoints" / "latest.pt"
        for satellites, episodes in ((64, 20), (128, 8), (256, 4)):
            evaluate(f"transfer_{method}", checkpoint, satellites, episodes)

    aggregate_results()
    state["status"] = "complete"
    update("complete", "All training, ablations, scale transfer tests, schedules, and tables are complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        state["status"] = "failed"
        update("failed", repr(exc), error=repr(exc))
        raise
