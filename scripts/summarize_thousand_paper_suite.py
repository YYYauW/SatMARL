from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


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
    "reservation_commit_rate",
    "reservation_member_waste",
    "mean_reservation_fill",
)

PAIRED_METRICS = (
    "mean_episode_reward",
    "completed_tasks",
    "cooperative_completed_tasks",
    "total_priority_completed",
    "total_conflicts",
    "total_invalid_actions",
    "total_window_misses",
    "reservation_commit_rate",
)

T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    15: 2.131,
    20: 2.086,
    30: 2.042,
}


def t_critical_975(degrees_of_freedom: int) -> float:
    if degrees_of_freedom in T_CRITICAL_975:
        return T_CRITICAL_975[degrees_of_freedom]
    larger = sorted(value for value in T_CRITICAL_975 if value >= degrees_of_freedom)
    return T_CRITICAL_975[larger[0]] if larger else 1.96


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parameter_count(payload: dict[str, Any]) -> int | None:
    network = (payload.get("algorithm") or {}).get("network") or {}
    value = network.get("parameter_count")
    return int(value) if value is not None else None


def evaluation_identity(path: Path, run_root: Path) -> dict[str, Any]:
    relative = path.relative_to(run_root)
    parts = relative.parts
    if len(parts) < 6 or parts[0] != "training" or "evaluation" not in parts:
        raise ValueError(f"Unexpected evaluation path: {relative}")
    method = parts[1]
    seed = int(parts[2].split("_", 1)[1])
    evaluation_index = parts.index("evaluation")
    tail = parts[evaluation_index + 1 :]
    if len(tail) == 2:
        scenario_group, filename = tail
        scenario = scenario_group
    elif len(tail) == 3:
        scenario_group, scenario, filename = tail
    else:
        raise ValueError(f"Unexpected evaluation layout: {relative}")
    satellites = int(Path(filename).stem.removeprefix("n"))
    return {
        "method": method,
        "training_seed": seed,
        "scenario_group": scenario_group,
        "scenario": scenario,
        "satellites": satellites,
    }


def collect_rows(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_root / "training").glob("*/seed_*/evaluation/**/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if payload.get("status") != "complete":
            continue
        identity = evaluation_identity(path, run_root)
        aggregate = (payload.get("benchmark") or {}).get("aggregate") or {}
        env_config = payload.get("env_config") or {}
        row: dict[str, Any] = {
            **identity,
            "tasks": int(env_config.get("num_tasks", 0)),
            "eval_episodes": int((payload.get("benchmark") or {}).get("episodes", 0)),
            "checkpoint_episode": payload.get("checkpoint_episode"),
            "parameter_count": parameter_count(payload),
            "evaluation_file": str(path),
        }
        runtime = (payload.get("benchmark") or {}).get("runtime") or {}
        for name in (
            "evaluation_wall_seconds",
            "env_steps_per_second",
            "agent_steps_per_second",
        ):
            row[name] = runtime.get(name)
        for metric in METRICS:
            values = aggregate.get(metric) or {}
            row[metric] = values.get("mean")
            row[f"{metric}_episode_std"] = values.get("std")
        rows.append(row)
    return rows


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_across_training_seeds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["method"],
            row["scenario_group"],
            row["scenario"],
            row["satellites"],
            row["tasks"],
        )
        groups[key].append(row)
    summaries: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        method, scenario_group, scenario, satellites, tasks = key
        summary: dict[str, Any] = {
            "method": method,
            "scenario_group": scenario_group,
            "scenario": scenario,
            "satellites": satellites,
            "tasks": tasks,
            "training_seeds": len(group),
            "eval_episodes_per_seed": min(int(row["eval_episodes"]) for row in group),
            "parameter_count": group[0].get("parameter_count"),
        }
        for runtime_metric in (
            "evaluation_wall_seconds",
            "env_steps_per_second",
            "agent_steps_per_second",
        ):
            values = [
                float(row[runtime_metric])
                for row in group
                if row.get(runtime_metric) is not None
            ]
            mean, standard_deviation = mean_std(values)
            summary[runtime_metric] = mean
            summary[f"{runtime_metric}_seed_std"] = standard_deviation
        for metric in METRICS:
            values = [
                float(row[metric]) for row in group if row.get(metric) is not None
            ]
            mean, standard_deviation = mean_std(values)
            summary[metric] = mean
            summary[f"{metric}_seed_std"] = standard_deviation
        summaries.append(summary)
    return summaries


def paired_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (
            row["method"],
            row["training_seed"],
            row["scenario_group"],
            row["scenario"],
            row["satellites"],
            row["tasks"],
        ): row
        for row in rows
    }
    conditions = sorted(
        {
            (
                row["scenario_group"],
                row["scenario"],
                row["satellites"],
                row["tasks"],
            )
            for row in rows
            if row["method"] == "full"
        }
    )
    methods = sorted({row["method"] for row in rows if row["method"] != "full"})
    output: list[dict[str, Any]] = []
    for scenario_group, scenario, satellites, tasks in conditions:
        for method in methods:
            paired_seeds = sorted(
                {
                    row["training_seed"]
                    for row in rows
                    if row["method"] == "full"
                    and row["scenario_group"] == scenario_group
                    and row["scenario"] == scenario
                    and row["satellites"] == satellites
                    and row["tasks"] == tasks
                    and (
                        method,
                        row["training_seed"],
                        scenario_group,
                        scenario,
                        satellites,
                        tasks,
                    )
                    in index
                }
            )
            if not paired_seeds:
                continue
            for metric in PAIRED_METRICS:
                differences = []
                for seed in paired_seeds:
                    full = index[
                        ("full", seed, scenario_group, scenario, satellites, tasks)
                    ].get(metric)
                    baseline = index[
                        (method, seed, scenario_group, scenario, satellites, tasks)
                    ].get(metric)
                    if full is not None and baseline is not None:
                        differences.append(float(full) - float(baseline))
                if not differences:
                    continue
                mean, standard_deviation = mean_std(differences)
                if len(differences) > 1:
                    half_width = (
                        t_critical_975(len(differences) - 1)
                        * standard_deviation
                        / math.sqrt(len(differences))
                    )
                else:
                    half_width = math.nan
                output.append(
                    {
                        "comparison": f"full-minus-{method}",
                        "metric": metric,
                        "scenario_group": scenario_group,
                        "scenario": scenario,
                        "satellites": satellites,
                        "tasks": tasks,
                        "paired_training_seeds": len(differences),
                        "mean_difference": mean,
                        "difference_seed_std": standard_deviation,
                        "ci95_low": mean - half_width if math.isfinite(half_width) else None,
                        "ci95_high": mean + half_width if math.isfinite(half_width) else None,
                    }
                )
    return output


def latex_base_table(summaries: list[dict[str, Any]]) -> str:
    base = [row for row in summaries if row["scenario_group"] == "base"]
    lines = [
        "% Auto-generated by summarize_thousand_paper_suite.py",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & $N$ & Return & Completed & Cooperative & Priority & Conflicts \\\\",
        "\\midrule",
    ]
    for row in sorted(base, key=lambda value: (value["satellites"], value["method"])):
        lines.append(
            "{method} & {satellites} & {reward:.4f} & {completed:.1f} & "
            "{cooperative:.1f} & {priority:.1f} & {conflicts:.2f} \\\\".format(
                method=str(row["method"]).replace("_", "\\_"),
                satellites=row["satellites"],
                reward=row["mean_episode_reward"],
                completed=row["completed_tasks"],
                cooperative=row["cooperative_completed_tasks"],
                priority=row["total_priority_completed"],
                conflicts=row["total_conflicts"],
            )
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    rows = collect_rows(run_root)
    summaries = aggregate_across_training_seeds(rows)
    comparisons = paired_comparisons(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "results_per_seed.csv", rows)
    write_csv(output_dir / "results_summary.csv", summaries)
    write_csv(output_dir / "paired_comparisons.csv", comparisons)
    write_json(
        output_dir / "results.json",
        {
            "status": "complete" if rows else "incomplete",
            "statistical_unit": (
                "independent training seed; held-out episodes are paired across methods"
            ),
            "per_seed": rows,
            "summary": summaries,
            "paired_comparisons": comparisons,
        },
    )
    (output_dir / "generated_thousand_results.tex").write_text(
        latex_base_table(summaries), encoding="utf-8"
    )
    print(
        f"Wrote {len(rows)} completed evaluation rows and "
        f"{len(summaries)} grouped rows to {output_dir}"
    )


if __name__ == "__main__":
    main()
