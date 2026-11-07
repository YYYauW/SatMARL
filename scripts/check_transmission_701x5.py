"""Read-only completion audit: full method, seed 701, five 24-hour scenarios.

This checks the original metric definitions, including sample standard deviation
across scenarios and the zero between-training-seed standard deviation for one
training model. It does not modify a run or infer missing experiment results.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np

import balanced_task_catalog as balanced
import run_transmission_load as serial
from transmission_accelerated_common import (
    EXECUTION_FILES, PARALLEL_VERSION, identity_digest, read_saved_episode, sha,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAME = "configs/transmission_day_v3_seed701_5scenes.json"
SATELLITES = (128, 256, 512, 1024)
TASKS = tuple(range(1000, 10001, 1000))
SEED, EPISODES = 701, 5
METRICS = serial.METRICS
RATE_FIELDS = ("observation_success_rate", "observation_and_downlink_success_rate")
COUNT_FIELDS = (
    "observation_success_tasks", "observation_and_downlink_success_tasks",
    "observed_but_undelivered_tasks", "cooperative_completed_tasks",
    "total_ground_conflicts", "total_conflicts", "total_invalid_actions",
    "event_transitions", "imaging_startup_count", "downlink_startup_count",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def inside(root, relative):
    path = (root / str(relative).replace("\\", "/")).resolve()
    require(path.is_relative_to(root), f"Path escapes run directory: {relative}")
    return path


def number(row, name):
    value = float(row[name])
    require(math.isfinite(value), f"Non-finite metric: {name}")
    return value


def same_number(actual, expected, label, tolerance=1e-10):
    require(math.isclose(float(actual), float(expected), rel_tol=1e-12,
                         abs_tol=tolerance), f"Numeric mismatch: {label}")


def key(row, episode=False):
    require(row["method"] == "full", "Unexpected method")
    result = tuple(int(row[name]) for name in ("training_seed", "satellites", "tasks"))
    return (*result, int(row["episode"])) if episode else result


def check_row(row, config):
    seed, satellites, total, episode = key(row, True)
    require(seed == SEED and satellites in SATELLITES and total in TASKS,
            "Episode is outside the agreed seed/constellation/load plan")
    require(0 <= episode < EPISODES, "Expected fixed scenario indices 0..4")
    require(int(row["scenario_seed"]) == config["scenario_seed"] + episode,
            "Wrong scenario seed")
    for name in METRICS:
        require(number(row, name) >= 0, f"Negative metric: {name}")
    for name in COUNT_FIELDS:
        require(number(row, name).is_integer(), f"Noninteger count: {name}")
    observed, delivered = (number(row, field) for field in (
        "observation_success_tasks", "observation_and_downlink_success_tasks"))
    require(0 <= delivered <= observed <= total, "Invalid success-count inclusion")
    same_number(number(row, "observation_success_rate"), observed / total,
                "observation success-rate denominator")
    same_number(number(row, "observation_and_downlink_success_rate"), delivered / total,
                "observation plus downlink success-rate denominator")
    same_number(number(row, "observed_but_undelivered_tasks"), observed - delivered,
                "observed-but-undelivered count")
    require(number(row, "cooperative_completed_tasks") <= observed,
            "Cooperative completed count exceeds all observed tasks")
    require(number(row, "delivered_priority") <= number(row, "total_priority_completed") + 1e-8,
            "Delivered priority exceeds observed completed priority")
    same_number(number(row, "simulated_seconds"), 86400, "24-hour horizon", 1e-6)
    require(number(row, "max_rolling_imaging_seconds") <= 1800 + 1e-6,
            "Rolling 90-minute imaging budget exceeded")
    committed, downlinked, pending, generated = (number(row, name) for name in (
        "committed_data_mb", "downlinked_data_mb", "pending_data_mb", "total_generated_data_mb"))
    tolerance = 1e-5 * max(1., committed)
    require(abs(committed - downlinked - pending) <= tolerance, "Data conservation violated")
    require(generated <= committed + tolerance, "Generated data exceeds committed data")
    require(downlinked <= generated + tolerance, "Downlinked data exceeds generated data")
    require(number(row, "mean_ground_station_utilization") <= 1 + 1e-8,
            "Ground-station utilization exceeds one")


def check_hashes(manifest, project):
    identity = manifest["identity"]
    require(identity["policy"] == "checkpoint" and identity["device"] == "cuda",
            "Expected original checkpoint evaluation on CUDA")
    expected_sources = {
        "scripts/run_transmission_load.py", "src/sat_marl_env/transmission.py",
        "src/sat_marl_env/env.py", "src/sat_marl_env/config.py", "src/sat_marl_env/entities.py",
        "src/sat_marl_env/orbital.py", "src/sat_marl_env/real_scenario.py",
        "scripts/build_kepler_ephemeris.py", "scripts/generate_task_catalogs.py",
        "scripts/evaluate_marl.py", "scripts/marl_common.py",
        "src/sat_marl_env/event_transmission.py", "scripts/balanced_task_catalog.py",
    }
    recorded = {name.replace("\\", "/"): value for name, value in identity["source_sha256"].items()}
    require(set(recorded) == expected_sources, "Missing/unexpected scientific source hashes")
    execution = manifest["execution_source_sha256"]
    require(set(execution) == set(EXECUTION_FILES), "Missing/unexpected accelerated execution hashes")
    for name, digest in {**recorded, **execution}.items():
        require(sha(inside(project, name)) == digest, f"Source or execution file changed: {name}")
    if "subset_preparation_source_sha256" in manifest:
        require(sha(project / "scripts/prepare_transmission_701x5.py") ==
                manifest["subset_preparation_source_sha256"], "Subset preparation source changed")
    checkpoints = identity["checkpoint_sha256"]
    expected = {job["checkpoint"] for job in identity["jobs"]}
    require(set(checkpoints) == expected and len(expected) == 1,
            "Expected exactly one original seed-701 checkpoint")
    for name, digest in checkpoints.items():
        require(str(name).replace("\\", "/").endswith("/full/seed_701/checkpoints/latest.pt"),
                "Checkpoint path must identify the original full/seed_701 model")
        path = Path(name)
        if not path.is_absolute():
            path = project / path
        require(sha(path) == digest, f"Checkpoint changed or unavailable: {name}")


def check_orbit(path, satellites, config):
    """Never enable pickle or load external filenames embedded in metadata."""
    sampling = config["env"]["ephemeris_sampling_seconds"]
    steps = math.ceil(86400 / sampling)
    lookahead = math.ceil(config["env"]["planning_lookahead_steps"] / sampling) + 2
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        require(metadata["format"] == "satmarl-ephemeris-v1", "Wrong orbit-cache format")
        require(metadata["start_utc"] == config["start_utc"], "Wrong orbit epoch")
        require(metadata["satellites"] == satellites and
                metadata["planes"] == config["planes"][str(satellites)], "Wrong orbit constellation")
        same_number(metadata["step_duration_seconds"], sampling, "orbit sample interval")
        require(metadata["max_steps"] == steps and metadata["lookahead_steps"] >= lookahead,
                "Orbit cache does not cover the day plus planning lookahead")
        total = metadata["total_steps"]
        require(total == metadata["max_steps"] + metadata["lookahead_steps"],
                "Orbit cache step counts disagree")
        require((total - 1) * sampling >= 86400 + config["env"]["planning_lookahead_steps"],
                "Orbit samples end before planning lookahead")
        for name, expected in config["orbit"].items():
            same_number(metadata["constellation"][name], expected, f"orbit {name}")
        for name in ("satellite_names", "plane_ids"):
            require(archive[name].shape == (satellites,), f"Wrong orbit array shape: {name}")
        plane_ids = archive["plane_ids"]
        require(np.all((plane_ids >= 0) & (plane_ids < config["planes"][str(satellites)])),
                "Invalid orbital-plane indices")
        for name in ("position_eci_km", "velocity_eci_km_s", "position_ecef_km", "velocity_ecef_km_s"):
            array = archive[name]
            require(array.shape == (total, satellites, 3), f"Wrong orbit array shape: {name}")
            require(np.issubdtype(array.dtype, np.number) and np.isfinite(array).all(),
                    f"Invalid numeric orbit array: {name}")


def check_migration(root, manifest):
    if "migration_sha256" not in manifest:
        return {}
    path = root / "migration.json"
    require(sha(path) == manifest["migration_sha256"], "Migration inventory changed")
    migration = read_json(path)
    inventory = migration.get("episode_files", {})
    sources = {}
    subset = migration.get("kind") == "seed701-five-scenario-subset-v1"
    if subset:
        from prepare_transmission_701x5 import compatible, row_digest
        require(migration["imported_completed_episodes"] == len(inventory),
                "Subset migration episode count disagrees with inventory")
        for source in migration["sources"]:
            require(source["root"] not in sources, "Duplicate migration source root")
            require(identity_digest(source["identity"]) == source["identity_sha256"],
                    "Migration source scientific identity changed")
            compatible(source["identity"], manifest["identity"])
            sources[source["root"]] = source
            original_manifest = Path(source["root"]) / "manifest.json"
            if original_manifest.is_file():
                require(sha(original_manifest) == source["manifest_sha256"],
                        "Original source manifest changed after subset migration")
        require(not inventory or sources, "Subset inventory has no source identity records")
    for relative, record in inventory.items():
        target = inside(root, relative)
        require(sha(target) == record["sha256"],
                f"Imported episode changed: {relative}")
        if subset:
            saved = read_json(target)
            provenance = saved["subset_provenance"]
            names = ("source_root", "source_relative", "source_file_sha256", "source_identity_sha256",
                     "source_row_sha256", "source_execution_source_sha256")
            require(all(provenance[name] == record[name] for name in names),
                    "Subset provenance differs from migration inventory")
            require(provenance["source_root"] in sources, "Unknown subset provenance source")
            source = sources[provenance["source_root"]]
            require(provenance["source_identity_sha256"] == source["identity_sha256"],
                    "Subset provenance refers to another scientific identity")
            require(row_digest(saved["row"]) == provenance["source_row_sha256"],
                    "Imported scientific metric row changed")
            require(saved["execution_source_sha256"] == provenance["source_execution_source_sha256"]
                    == record["execution_source_sha256"], "Imported execution provenance changed")
            original = inside(Path(source["root"]).resolve(), provenance["source_relative"])
            if original.is_file():
                require(sha(original) == provenance["source_file_sha256"],
                        "Original result file changed after subset migration")
                original_value = read_json(original)
                old_row = (original_value["row"] if "row" in original_value else
                           original_value["episodes"][int(saved["row"]["episode"])])
                require(saved["row"] == old_row, "Imported row differs from the original saved result")
    return inventory


def validate(run_root, project_root=ROOT):
    root, project = Path(run_root).expanduser().resolve(), Path(project_root).resolve()
    manifest = read_json(root / "manifest.json")
    require(manifest.get("runner") == PARALLEL_VERSION, "Expected accelerated-v2 run")
    config = manifest["identity"]["config"]
    require(config == read_json(project / CONFIG_NAME), "Run config differs from agreed seed701/five-scene config")
    require(config["training_seeds"] == [SEED] and config["episodes"] == EPISODES,
            "Expected one training model, seed701, and five scenarios")
    require(config["methods"] == ["full"] and config["satellite_counts"] == list(SATELLITES)
            and config["task_counts"] == list(TASKS), "Wrong method/scale/load plan")
    require(config["env"]["max_steps"] * config["env"]["step_duration_seconds"] == 86400,
            "Expected a 24-hour physical horizon")
    progress = read_json(root / "progress.json")
    require(progress.get("status") == "complete", "Run is not complete")
    require(progress.get("completed_jobs") == progress.get("total_jobs") == 40,
            "Expected 40 complete jobs")
    require(progress.get("completed_episodes") == progress.get("total_episodes") == 200,
            "Expected 200 complete episodes")
    require(progress.get("failed_jobs", 0) == 0 and progress.get("running_jobs", 0) == 0
            and progress.get("queued_jobs", 0) == 0, "Run still has failed/running/queued work")
    expected_jobs = set(itertools.product((SEED,), SATELLITES, TASKS))
    plan = manifest["identity"]["jobs"]
    require(len(plan) == 40 and {key(job) for job in plan} == expected_jobs,
            "Manifest must contain exactly 40 unique agreed jobs")
    check_hashes(manifest, project)
    imports = check_migration(root, manifest)
    fingerprint = identity_digest(manifest["identity"])
    reference, aggregates = {}, {}
    expected_filenames = {f"full_seed{seed}_n{n}_m{m}.json" for seed, n, m in expected_jobs}
    require({path.name for path in (root / "jobs").glob("*.json")} == expected_filenames,
            "Missing or unexpected job-result files")
    for job in plan:
        name = f"full_seed{SEED}_n{job['satellites']}_m{job['tasks']}"
        result = read_json(root / "jobs" / (name + ".json"))
        require(result.get("status") == "complete" and result.get("protocol") == config["protocol"],
                f"Incomplete or wrong-protocol job: {name}")
        require(result["job"] == job and len(result["episodes"]) == EPISODES,
                f"Wrong job identity/episode count: {name}")
        require([int(row["episode"]) for row in result["episodes"]] == list(range(EPISODES)),
                f"Expected ordered scenario indices 0..4: {name}")
        for row in result["episodes"]:
            check_row(row, config)
            require(key(row) == key(job), "Episode belongs to another job")
            reference[key(row, True)] = row
            saved_path = root / "episode_results" / name / f"episode{int(row['episode']):03d}.json"
            if saved_path.exists():
                saved = read_saved_episode(root, saved_path, fingerprint,
                                           manifest["execution_source_sha256"], imports)
                require(saved["row"] == row, "Saved episode and job JSON differ")
        expected = serial.aggregate(result["episodes"])
        require(set(result["aggregate"]) == set(METRICS), "Job aggregate metric fields changed")
        for metric in METRICS:
            for stat in ("mean", "std"):
                same_number(number(result["aggregate"][metric], stat), expected[metric][stat],
                            f"job aggregate {name}/{metric}/{stat}")
        aggregates[key(job)] = expected
    episodes = read_csv(root / "episodes.csv")
    require(len(episodes) == 200 and {key(row, True) for row in episodes} == set(reference),
            "episodes.csv must contain 200 unique agreed episodes")
    for row in episodes:
        check_row(row, config)
        original = reference[key(row, True)]
        for metric in (*METRICS, "scenario_seed"):
            same_number(number(row, metric), number(original, metric), f"CSV/JSON {metric}")
    grouped = read_csv(root / "grouped.csv")
    require(len(grouped) == 40 and {key(row) for row in grouped} == expected_jobs,
            "Expected 40 unique per-model aggregates")
    for row in grouped:
        expected = aggregates[key(row)]
        for metric in METRICS:
            for stat in ("mean", "std"):
                same_number(number(row, f"{metric}_{stat}"), expected[metric][stat],
                            f"grouped {metric}/{stat}")
    merged = read_csv(root / "grouped_across_training_seeds.csv")
    numeric_groups = [dict(method="full", training_seed=SEED, satellites=identifier[1], tasks=identifier[2],
                           **{f"{metric}_{stat}": value for metric, stats in values.items()
                              for stat, value in stats.items()})
                      for identifier, values in aggregates.items()]
    expected_merged = {(row["satellites"], row["tasks"]): row
                       for row in serial.merge_training_seeds(numeric_groups)}
    require(len(merged) == 40, "Expected 40 merged results")
    seen = set()
    for row in merged:
        identifier = int(row["satellites"]), int(row["tasks"])
        require(identifier in expected_merged and identifier not in seen, "Missing/duplicated merged result")
        seen.add(identifier)
        require(row["method"] == "full" and int(row["training_seed_count"]) == 1,
                "Merged result must represent exactly one training model")
        for metric in METRICS:
            for suffix in ("_mean", "_std_between_training_seeds"):
                same_number(number(row, metric + suffix), expected_merged[identifier][metric + suffix],
                            f"merged {metric}{suffix}")
    assets = read_json(root / "scenario_sha256.json")
    expected_assets = {f"scenario/targets_episode{ep:03d}.csv" for ep in range(EPISODES)}
    expected_assets.update(f"scenario/orbit_n{n}.npz" for n in SATELLITES)
    normalized = {name.replace("\\", "/"): value for name, value in assets.items()}
    require(set(normalized) == expected_assets and len(assets) == 9,
            "Expected exactly five catalogues and four orbit caches")
    for name, digest in normalized.items():
        require(sha(inside(root, name)) == digest, f"Scenario asset changed: {name}")
    paths = [root / "scenario" / f"targets_episode{ep:03d}.csv" for ep in range(EPISODES)]
    recomputed = balanced.balance_report(config, paths)
    require(read_json(root / "scenario_balance.json") == recomputed,
            "Balance report does not match actual catalogue contents")
    require(recomputed["all_prefixes_balanced"] and recomputed["spatial_counts_identical_across_episodes"],
            "Actual task catalogue balance failed")
    for path in paths:
        for row in read_csv(path):
            require(row["required_mode"] == "optical" and row["target_type"] == "point",
                    "Task observation type changed")
            same_number(number(row, "compression_ratio"), .25, "1:4 compression")
            require(2000 <= number(row, "original_data_mb") <= 8000, "Raw data size outside agreed range")
            same_number(number(row, "memory_required_mb"), number(row, "original_data_mb") / 4,
                        "Compressed per-target storage", 1e-7)
            same_number(number(row, "observation_duration_seconds"), 5., "Point observation duration")
    for satellites in SATELLITES:
        check_orbit(root / "scenario" / f"orbit_n{satellites}.npz", satellites, config)
    return dict(completed_jobs=40, episodes=200, merged_results=40, training_seeds=[SEED],
                scenario_indices=list(range(EPISODES)), balanced_scenario_scale_checks=50,
                verified_scenario_assets=9, metric_count=len(METRICS), success_rates=list(RATE_FIELDS),
                scenario_standard_deviation="sample standard deviation, ddof=1 across five scenarios",
                between_training_seed_standard_deviation="0.0, original one-model merge definition")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        result = validate(args.run_root, args.project_root)
    except (KeyError, ValueError, OSError, TypeError, OverflowError) as error:
        parser.exit(1, f"NOT COMPLETE / CHECK FAILED: {error}\n")
    print("PASS: seed701, five full-day scenarios, 40 jobs / 200 episodes; all original metrics verified.")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
