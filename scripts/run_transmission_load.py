"""Fixed-constellation / variable-demand evaluation, with no training or weight edits.

Examples and parameter provenance: docs/transmission_load_v2_zh.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from dataclasses import asdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from sat_marl_env.transmission import MODEL_VERSION, TransmissionConfig, TransmissionEnv
from generate_task_catalogs import build_records, geographic_points


METRICS = (
    "observation_success_tasks", "observation_and_downlink_success_tasks",
    "observation_success_rate", "observation_and_downlink_success_rate",
    "observed_but_undelivered_tasks", "cooperative_completed_tasks",
    "total_priority_completed", "delivered_priority", "downlinked_data_mb",
    "pending_data_mb", "total_generated_data_mb", "committed_data_mb",
    "total_ground_conflicts", "total_conflicts", "total_invalid_actions",
    "max_rolling_imaging_seconds", "contact_antenna_seconds", "episode_wall_seconds",
    "mean_ground_station_utilization",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    tmp.replace(path)


def integer_list(text, default):
    values = list(default) if text is None else [int(v) for v in text.split(",")]
    if not values or any(v < 1 for v in values) or len(set(values)) != len(values):
        raise ValueError("Lists must contain distinct positive integers")
    return values


def priority_actions(env, observations):
    """Diagnostic rule: initiate downlink whenever possible, otherwise best plan."""
    actions = {}
    for agent, obs in observations.items():
        mask = obs["action_mask"]
        if mask[1]:
            actions[agent] = 1
            continue
        choices = []
        sat = env.satellites[env._agent_index(agent)]
        for action in np.flatnonzero(mask[2:])+2:
            task = env.tasks[int(obs["candidate_ids"][action-2])]
            _, _, plan = env._plan_task(sat, task)
            delay = max(0, float(plan["observation_start_step"])-env.step_count)
            choices.append((task.priority*float(plan["quality"])/(1+delay), int(action)))
        actions[agent] = max(choices)[1] if choices else 0
    return actions


def prepare_catalog(config, path, episode):
    if path.exists():
        return
    rng = np.random.default_rng(config["scenario_seed"]+episode)
    specification = config["catalog"]
    points, _ = geographic_points(
        max(config["task_counts"]), rng,
        profile=specification["spatial_profile"], cluster_standard_deviation_deg=6,
        event_center_latitude_deg=30, event_center_longitude_deg=110,
        event_burst_fraction=.8, minimum_separation_deg=0)
    records = build_records(
        "test", points, rng, max_steps=config["env"]["max_steps"],
        min_window_steps=specification["window_steps"][0],
        max_window_steps=specification["window_steps"][1],
        cooperative_observers_min=specification["cooperative_observers"][0],
        cooperative_observers_max=specification["cooperative_observers"][1],
        cooperation_weights=specification["fractions"])
    # Per-target sizes are attached to the master catalog once, before slicing M.
    # A separate RNG makes them independent of method, N, and checkpoint seed.
    size_rng = np.random.default_rng(config["scenario_seed"]+episode+1_000_000)
    for row in records:
        raw = round(float(size_rng.uniform(config["env"]["original_data_min_mb"],
                                         config["env"]["original_data_max_mb"])), 4)
        row.update(required_mode="optical", original_data_mb=raw,
                   compression_ratio=.25, memory_required_mb=raw/4,
                   required_resolution_m=2.0, required_swath_km=10.0,
                   min_sun_elevation_deg=8.0,
                   observation_duration_seconds=config["env"]["point_observation_seconds"])
    write_csv(path, records)


def prepare_orbit(config, path, n):
    if path.exists():
        return
    env = config["env"]
    command = [sys.executable, str(ROOT/"scripts/build_kepler_ephemeris.py"),
               "--output", str(path), "--start-utc", config["start_utc"],
               "--satellites", str(n), "--planes", str(config["planes"][str(n)]),
               "--max-steps", str(env["max_steps"]),
               "--lookahead-steps", str(max(2, env["planning_lookahead_steps"])),
               "--step-duration-seconds", str(env["step_duration_seconds"])]
    for key, value in config["orbit"].items():
        command += ["--"+key.replace("_", "-"), str(value)]
    subprocess.run(command, cwd=ROOT, check=True)


def aggregate(rows):
    return {key: {"mean": float(np.mean([r[key] for r in rows])),
                  "std": float(np.std([r[key] for r in rows], ddof=1)) if len(rows)>1 else 0.0}
            for key in METRICS}


def make_plan(config, policy, model_root):
    return [dict(method=method, training_seed=seed, satellites=n, tasks=m,
                 checkpoint=str(model_root/method/f"seed_{seed}"/"checkpoints/latest.pt")
                 if policy == "checkpoint" else None)
            for method in config["methods"] for seed in config["training_seeds"]
            for n in config["satellite_counts"] for m in config["task_counts"]]


def merge_training_seeds(rows):
    groups = {}
    for row in rows:
        key = row["method"], row["satellites"], row["tasks"]
        groups.setdefault(key, []).append(row)
    output = []
    for (method, n, m), entries in groups.items():
        item = dict(method=method, satellites=n, tasks=m, training_seed_count=len(entries))
        for metric in METRICS:
            values = [row[metric+"_mean"] for row in entries]
            item[metric+"_mean"] = float(np.mean(values))
            item[metric+"_std_between_training_seeds"] = float(np.std(values, ddof=1)) if len(values)>1 else 0.0
        output.append(item)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/transmission_load_v2.json")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=ROOT/"runs/thousand_paper_suite/training")
    parser.add_argument("--methods")
    parser.add_argument("--training-seeds")
    parser.add_argument("--satellites")
    parser.add_argument("--task-counts")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--rate-mbps", type=float)
    parser.add_argument("--imaging-minutes", type=float)
    parser.add_argument("--policy", choices=["checkpoint", "priority"], default="checkpoint")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--save-all-ledgers", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("protocol") != MODEL_VERSION:
        raise ValueError("Wrong protocol version")
    if args.smoke:
        config.update(satellite_counts=[8], task_counts=[128, 256], episodes=1,
                      training_seeds=[701], methods=["priority"], planes={"8": 2})
        config["env"].update(max_steps=240, candidate_k=8, neighbor_k=3,
                             planning_lookahead_steps=10)
    config["satellite_counts"] = integer_list(args.satellites, config["satellite_counts"])
    config["task_counts"] = integer_list(args.task_counts, config["task_counts"])
    config["training_seeds"] = integer_list(args.training_seeds, config["training_seeds"])
    if args.methods:
        config["methods"] = args.methods.split(",")
    if args.episodes is not None:
        config["episodes"] = args.episodes
    if config["episodes"] < 1:
        raise ValueError("episodes must be positive")
    if args.rate_mbps is not None:
        config["env"]["downlink_rate_mbps"] = args.rate_mbps
    if args.imaging_minutes is not None:
        config["env"]["imaging_budget_seconds"] = 60*args.imaging_minutes
    if args.policy == "priority":
        config["methods"] = ["priority"]
        config["training_seeds"] = [701]  # Diagnostic, not a trained-seed comparison.
    TransmissionConfig(**config["env"])
    for n in config["satellite_counts"]:
        if str(n) not in config["planes"]:
            raise ValueError(f"Explicit plane count needed in config for N={n}")
    plan = make_plan(config, args.policy, args.model_root.resolve())
    if args.plan_only:
        print(json.dumps(dict(config=config, jobs=plan), ensure_ascii=False, indent=2))
        return
    torch = None
    device = None
    if args.policy == "checkpoint" and not args.prepare_only:
        # Fail before generating large catalogs when checkpoint inputs are missing.
        for job in plan:
            if not Path(job["checkpoint"]).is_file():
                raise FileNotFoundError(f"Checkpoint not found: {job['checkpoint']}")
        import torch
        from marl_common import resolve_device
        device = resolve_device(args.device)

    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    lock_path = run_root/"runner.lock"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(lock_fd, str(os.getpid()).encode())
        manifest_path = run_root/"manifest.json"
        source_files = [Path(__file__), ROOT/"src/sat_marl_env/transmission.py",
                        ROOT/"src/sat_marl_env/env.py", ROOT/"src/sat_marl_env/config.py",
                        ROOT/"src/sat_marl_env/entities.py", ROOT/"src/sat_marl_env/orbital.py",
                        ROOT/"src/sat_marl_env/real_scenario.py",
                        ROOT/"scripts/build_kepler_ephemeris.py", ROOT/"scripts/generate_task_catalogs.py"]
        if args.policy == "checkpoint":
            source_files += [ROOT/"scripts/evaluate_marl.py", ROOT/"scripts/marl_common.py"]
        identity = dict(config=config, policy=args.policy, jobs=plan,
                        device=args.device, save_all_ledgers=args.save_all_ledgers,
                        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in source_files},
                        checkpoint_sha256={p: sha(p) for p in {j["checkpoint"] for j in plan}
                                           if p is not None and Path(p).exists()})
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not args.resume or old["identity"] != identity:
                raise RuntimeError("Run root already used; use a new root or resume identical inputs/code")
        else:
            if any(run_root.iterdir()) and set(run_root.iterdir()) != {lock_path}:
                raise RuntimeError("Use an empty, dedicated v2 run root")
            atomic_json(manifest_path, dict(protocol=MODEL_VERSION, identity=identity,
                                           note="checkpoint evaluation is out-of-training-distribution; no retraining"))
        scenario = run_root/"scenario"
        scenario.mkdir(exist_ok=True)
        assets_path = run_root/"scenario_sha256.json"
        expected_assets = json.loads(assets_path.read_text(encoding="utf-8")) if assets_path.exists() else {}
        assets = {}
        for episode in range(config["episodes"]):
            catalog = scenario/f"targets_episode{episode:03d}.csv"
            prepare_catalog(config, catalog, episode)
            assets[str(catalog.relative_to(run_root))] = sha(catalog)
        for n in config["satellite_counts"]:
            orbit = scenario/f"orbit_n{n}.npz"
            prepare_orbit(config, orbit, n)
            assets[str(orbit.relative_to(run_root))] = sha(orbit)
        if expected_assets and expected_assets != assets:
            raise RuntimeError("Scenario file hashes changed; refusing to mix results")
        atomic_json(assets_path, assets)
        if args.prepare_only:
            print(f"Scenario prepared: {scenario}")
            return

        episode_rows, aggregate_rows = [], []
        for job in plan:
            method, seed, n, m = (job[k] for k in ("method", "training_seed", "satellites", "tasks"))
            key = f"{method}_seed{seed}_n{n}_m{m}"
            output = run_root/"jobs"/(key+".json")
            if args.resume and output.exists():
                result = json.loads(output.read_text(encoding="utf-8"))
                if result.get("status") != "complete" or len(result["episodes"]) != config["episodes"]:
                    raise RuntimeError(f"Incomplete or malformed job: {output}")
            else:
                checkpoint, metrics, choose = None, {}, None
                if args.policy == "checkpoint":
                    from evaluate_marl import build_policy
                    checkpoint = torch.load(job["checkpoint"], map_location=device, weights_only=False)
                    metrics = dict(checkpoint.get("metrics") or {})
                rows, representative = [], None
                for episode in range(config["episodes"]):
                    options = dict(config["env"])
                    # Candidate/neighbor widths are part of checkpoint architecture.
                    if checkpoint is not None:
                        trained_env = metrics.get("env_config") or {}
                        for attr in ("candidate_k", "neighbor_k", "coalition_reservations",
                                     "reservation_ttl_steps", "reservation_failure_penalty"):
                            if attr in trained_env:
                                options[attr] = trained_env[attr]
                    options.update(num_satellites=n, num_tasks=m, num_planes=config["planes"][str(n)],
                                   ephemeris_cache_path=str(scenario/f"orbit_n{n}.npz"),
                                   task_catalog_path=str(scenario/f"targets_episode{episode:03d}.csv"),
                                   task_layout="catalog", random_seed=config["scenario_seed"]+episode)
                    started = time.perf_counter()
                    env = TransmissionEnv(TransmissionConfig(**options))
                    obs, _ = env.reset(seed=config["scenario_seed"]+episode)
                    if checkpoint is not None and choose is None:
                        _, choose = build_policy(checkpoint, metrics, obs, device)
                    reset = getattr(choose, "reset_strategy", None)
                    if callable(reset):
                        reset()
                    for _ in range(env.config.max_steps):
                        actions = choose(obs) if choose is not None else priority_actions(env, obs)
                        obs, _, terminated, truncated, _ = env.step(actions)
                        if all(terminated.values()) or all(truncated.values()):
                            break
                    summary = env.summary()
                    summary["episode_wall_seconds"] = time.perf_counter()-started
                    row = dict(method=method, training_seed=seed, satellites=n, tasks=m,
                               episode=episode, scenario_seed=config["scenario_seed"]+episode,
                               **{k: summary[k] for k in METRICS})
                    # This equality includes reserved images; finished data are a subset.
                    residual = summary["committed_data_mb"]-summary["downlinked_data_mb"]-summary["pending_data_mb"]
                    if abs(residual) > 1e-5*max(1.0, summary["committed_data_mb"]):
                        raise RuntimeError(f"Data conservation failure: {residual}")
                    if summary["max_rolling_imaging_seconds"] > env.config.imaging_budget_seconds+1e-6:
                        raise RuntimeError("Rolling imaging budget violated")
                    rows.append(row)
                    if episode == 0:
                        representative = dict(summary=summary, acquisitions=env.acquisition_ledger(),
                                              task_outcomes=env.task_outcomes(),
                                              stations=[asdict(g) for g in env.ground_stations])
                    if args.save_all_ledgers:
                        atomic_json(run_root/"ledgers"/key/f"episode{episode:03d}.json",
                                    dict(acquisitions=env.acquisition_ledger(), task_outcomes=env.task_outcomes()))
                    print(f"{key} episode {episode+1}/{config['episodes']}: observed={row['observation_success_tasks']} delivered={row['observation_and_downlink_success_tasks']}", flush=True)
                result = dict(status="complete", protocol=MODEL_VERSION, job=job, episodes=rows,
                              aggregate=aggregate(rows), representative=representative)
                atomic_json(output, result)
            episode_rows.extend(result["episodes"])
            aggregate_rows.append(dict(method=method, training_seed=seed, satellites=n, tasks=m,
                **{f"{metric}_{stat}": value for metric, stats in result["aggregate"].items()
                   for stat, value in stats.items()}))
            write_csv(run_root/"episodes.csv", episode_rows)
            write_csv(run_root/"grouped.csv", aggregate_rows)
            write_csv(run_root/"grouped_across_training_seeds.csv", merge_training_seeds(aggregate_rows))
            atomic_json(run_root/"progress.json", dict(status="running", completed_jobs=len(aggregate_rows), total_jobs=len(plan)))
        atomic_json(run_root/"progress.json", dict(status="complete", completed_jobs=len(plan), total_jobs=len(plan)))
        print(f"Saved summaries: {run_root}")
    finally:
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
