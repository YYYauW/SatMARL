from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Callable


SCALARS = {
    "mean_reward": "train/mean_reward",
    "reward_per_opportunity": "train/reward_per_opportunity",
    "completed_tasks": "tasks/completed",
    "cooperative_completed_tasks": "tasks/cooperative_completed",
    "expired_tasks": "tasks/expired",
    "total_priority_completed": "tasks/priority_completed",
    "decision_opportunities": "opportunities/count",
    "decision_opportunity_rate": "opportunities/decision_rate",
    "task_decision_opportunity_rate": "opportunities/task_rate",
    "decision_samples": "opportunities/decision_samples",
    "forced_idle_actions": "idle/forced",
    "avoidable_idle_actions": "idle/avoidable",
    "mean_loss": "loss/mean",
    "loss": "loss/raw",
    "policy_loss": "loss/policy",
    "value_loss": "loss/value",
    "entropy": "policy/entropy",
    "approx_kl": "policy/approx_kl",
    "clip_fraction": "policy/clip_fraction",
    "epsilon": "exploration/epsilon",
    "conflicts": "constraints/conflicts",
    "ground_conflicts": "constraints/ground_conflicts",
    "invalid_actions": "constraints/invalid_actions",
    "window_misses": "constraints/window_misses",
    "mean_observation_quality": "quality/observation",
    "downlinked_data_mb": "resources/downlinked_data_mb",
    "pending_data_mb": "resources/pending_data_mb",
    "episode_seconds": "throughput/episode_seconds",
    "env_steps_per_second": "throughput/env_steps_per_second",
    "agent_steps_per_second": "throughput/agent_steps_per_second",
    "replay_size": "replay/size",
}


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def numeric(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def discover_metrics(run_roots: list[Path]) -> list[tuple[Path, Path]]:
    discovered: list[tuple[Path, Path]] = []
    for run_root in run_roots:
        if not run_root.exists():
            continue
        for metrics_path in sorted(run_root.rglob("metrics.json")):
            if "tensorboard" in metrics_path.parts:
                continue
            discovered.append((run_root, metrics_path))
    return discovered


def sync_metrics(
    run_root: Path,
    metrics_path: Path,
    output_dir: Path,
    writers: dict[str, Any],
    last_steps: dict[str, int],
    writer_factory: Callable[[Path], Any],
) -> int:
    payload = load_json(metrics_path)
    history = payload.get("history", [])
    if not isinstance(history, list):
        return 0

    source_key = str(metrics_path.resolve())
    relative_parent = metrics_path.parent.relative_to(run_root)
    run_name = Path(run_root.name, relative_parent).as_posix()
    writer = writers.get(source_key)
    if writer is None:
        log_dir = output_dir / run_root.name / relative_parent
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = writer_factory(log_dir)
        writer.add_text("run/source_metrics", str(metrics_path.resolve()), 0)
        writers[source_key] = writer

    last_step = last_steps.get(source_key, -1)
    written = 0
    for record in history:
        if not isinstance(record, dict):
            continue
        try:
            step = int(record.get("episode", -1))
        except (TypeError, ValueError):
            continue
        if step <= last_step:
            continue
        for source, tag in SCALARS.items():
            value = numeric(record.get(source))
            if value is not None:
                writer.add_scalar(tag, value, step)
        last_step = max(last_step, step)
        written += 1

    if written:
        writer.flush()
        last_steps[source_key] = last_step
        print(
            f"[{time.strftime('%H:%M:%S')}] {run_name}: "
            f"wrote {written} episode(s), latest={last_step}",
            flush=True,
        )
    return written


def summary_writer_factory(log_dir: Path) -> Any:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise SystemExit(
            "TensorBoard support is unavailable. Install the monitoring extra with "
            "`python -m pip install -e '.[monitoring]'`."
        ) from exc
    return SummaryWriter(log_dir=str(log_dir), flush_secs=5)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Continuously mirror metrics.json histories from independent MARL "
            "runs into TensorBoard event files without restarting training."
        )
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        action="append",
        required=True,
        help="Experiment root to scan; repeat for multiple seeds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="TensorBoard event root.",
    )
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    run_roots = [path.expanduser().resolve() for path in args.run_root]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[str, Any] = {}
    last_steps: dict[str, int] = {}

    print("Metrics roots:", *(str(path) for path in run_roots), sep="\n  ", flush=True)
    print(f"TensorBoard events: {output_dir}", flush=True)
    try:
        while True:
            for run_root, metrics_path in discover_metrics(run_roots):
                sync_metrics(
                    run_root,
                    metrics_path,
                    output_dir,
                    writers,
                    last_steps,
                    summary_writer_factory,
                )
            if args.once:
                break
            time.sleep(max(0.5, args.poll_seconds))
    except KeyboardInterrupt:
        pass
    finally:
        for writer in writers.values():
            writer.close()


if __name__ == "__main__":
    main()
