"""Prepare a new five-scenario run, retaining verified completed old episodes.

The old directories and frozen serial/v1/v2 programs are never edited. Only the
selection identity changes; imported metric rows are preserved verbatim and all
five-scenario aggregates are recomputed with the original statistical functions.
Default operation is a read-only eligibility report; use --apply to prepare.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import shutil
import time
import uuid

import run_transmission_load as serial
import run_transmission_accelerated as accelerated
import transmission_accelerated_common as common

ROOT = common.ROOT
CONFIG = ROOT / "configs/transmission_day_v3_seed701_5scenes.json"
SCRIPTS = {"run_transmission_load.py", "run_transmission_parallel.py",
           "run_transmission_accelerated.py", "transmission_parallel_worker.py",
           "transmission_accelerated_worker.py"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inside(root, relative):
    result = (Path(root) / str(relative).replace("\\", "/")).resolve()
    require(result.is_relative_to(Path(root).resolve()), f"Path escapes run: {relative}")
    return result


def active(root):
    if common.source_is_active(root):
        return True
    # Also catch an unrecorded orphan; a stale progress/lock is not evidence of exit.
    proc = Path("/proc")
    return proc.is_dir() and any(common.process_matches_run(int(p.name), root, SCRIPTS)
                                for p in proc.iterdir() if p.name.isdecimal())


def row_digest(row):
    import hashlib
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def compatible(source_identity, identity):
    """Only episode count, training-seed selection and equivalent paths may differ."""
    old, new = common.canonical_identity(source_identity), common.canonical_identity(identity)
    old_config, new_config = copy.deepcopy(old["config"]), copy.deepcopy(new["config"])
    require(old_config["episodes"] >= new_config["episodes"], "Source has fewer configured scenarios")
    require(set(new_config["training_seeds"]).issubset(old_config["training_seeds"]),
            "Source does not contain the required trained seed")
    for name in ("episodes", "training_seeds"):
        old_config.pop(name)
        new_config.pop(name)
    require(old_config == new_config, "Source changes physics, catalogue, horizon or evaluation plan")
    for name in ("policy", "device", "save_all_ledgers", "source_sha256"):
        require(old[name] == new[name], f"Source changes scientific input: {name}")
    old_jobs = {common.job_key(job): job for job in old["jobs"]}
    require(len(old_jobs) == len(old["jobs"]), "Duplicate source jobs")
    expected = {(m, seed, n, t) for m in old["config"]["methods"]
                for seed in old["config"]["training_seeds"]
                for n in old["config"]["satellite_counts"] for t in old["config"]["task_counts"]}
    require({tuple(j[k] for k in ("method", "training_seed", "satellites", "tasks"))
             for j in old_jobs.values()} == expected, "Source manifest plan is incomplete")
    for job in new["jobs"]:
        source_job = old_jobs.get(common.job_key(job))
        require(source_job is not None, "Missing required source job")
        require({k: v for k, v in source_job.items() if k != "checkpoint"} ==
                {k: v for k, v in job.items() if k != "checkpoint"}, "Changed source job")
        if job["checkpoint"] is not None:
            old_sha = old["checkpoint_sha256"].get(source_job["checkpoint"])
            new_sha = new["checkpoint_sha256"].get(job["checkpoint"])
            require(isinstance(old_sha, str) and re.fullmatch(r"[0-9a-f]{64}", old_sha)
                    and old_sha == new_sha, "Different or missing trained checkpoint hash")
    return old_jobs


def inspect_source(root, identity):
    root = Path(root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = common.read_json(manifest_path)
    jobs = compatible(manifest["identity"], identity)
    runner = manifest.get("runner", "serial")
    execution = manifest.get("execution_source_sha256")
    if runner == common.PARALLEL_VERSION:
        expected_execution = common.execution_hashes()
    elif runner == "transmission-parallel-v1":
        import transmission_parallel_common as legacy
        expected_execution = legacy.execution_hashes()
    elif runner == "serial":
        expected_execution = None
    else:
        raise ValueError(f"Unknown source runner: {runner}")
    require(execution == expected_execution, "Source execution differs from frozen release")
    assets = {name.replace("\\", "/"): digest for name, digest in
              common.read_json(root / "scenario_sha256.json").items()}
    common.verify_assets(root, assets)
    require(all(name in assets for name in required_assets(identity["config"])),
            "Source is missing required catalogue or orbit assets")
    return dict(root=root, manifest=manifest, manifest_sha256=common.sha(manifest_path),
                identity_sha256=common.identity_digest(manifest["identity"]), jobs=jobs,
                runner=runner, execution=execution, assets=assets, used_files={})


def required_assets(config):
    return ([f"scenario/targets_episode{i:03d}.csv" for i in range(config["episodes"])] +
            [f"scenario/orbit_n{n}.npz" for n in config["satellite_counts"]])


def read_candidate(source, job, episode, config):
    root, key = source["root"], common.job_key(job)
    old_job = source["jobs"][key]
    job_path = root / "jobs" / (key + ".json")
    complete = None
    if job_path.exists():
        complete = common.read_json(job_path)
        common.validate_job(complete, old_job, source["manifest"]["identity"]["config"])
        require(complete["aggregate"] == serial.aggregate(complete["episodes"]),
                f"Old aggregate does not match its rows: {job_path}")
    ep_path = root / "episode_results" / key / f"episode{episode:03d}.json"
    if ep_path.exists():
        saved = common.read_saved_episode(root, ep_path, source["identity_sha256"], source["execution"])
        row, representative, execution = saved["row"], saved.get("representative"), saved.get("execution_source_sha256")
        if complete is not None:
            require(row == complete["episodes"][episode], f"Job and episode checkpoint disagree: {ep_path}")
        original = ep_path
    elif complete is not None:
        row, representative = complete["episodes"][episode], complete.get("representative") if episode == 0 else None
        execution, original = source["execution"], job_path
    else:
        return None
    common.validate_row(row, job, episode, config)
    require(all(name in row for name in serial.METRICS), "Old row is missing original metric fields")
    if episode == 0:
        if representative is None and complete is not None:
            representative = complete.get("representative")
        require(isinstance(representative, dict), "Scenario 0 is missing the representative ledger")
    # Ledger mode is normally false, but never silently claim missing saved ledgers.
    ledger = root / "ledgers" / key / f"episode{episode:03d}.json"
    if config is not None and source["manifest"]["identity"]["save_all_ledgers"]:
        require(ledger.is_file(), f"Required full ledger is missing: {ledger}")
        source["used_files"][str(ledger.relative_to(root))] = common.sha(ledger)
    for path in (original, job_path if complete is not None else original):
        source["used_files"][path.relative_to(root).as_posix()] = common.sha(path)
    provenance = dict(source_root=str(root), source_relative=original.relative_to(root).as_posix(),
                      source_file_sha256=common.sha(original), source_identity_sha256=source["identity_sha256"],
                      source_row_sha256=row_digest(row), source_execution_source_sha256=execution)
    return dict(row=row, representative=representative, execution_source_sha256=execution,
                subset_provenance=provenance, ledger=ledger if ledger.exists() else None)


def select(sources, identity):
    selected, duplicates = {}, 0
    config = identity["config"]
    for job in identity["jobs"]:
        key = common.job_key(job)
        for episode in range(config["episodes"]):
            choice = None
            for source in sources:
                candidate = read_candidate(source, job, episode, config)
                if candidate is None:
                    continue
                if choice is None:
                    choice = candidate
                else:
                    # Runtime depends on the machine, all scientific output must agree.
                    without_wall = lambda r: {k: v for k, v in r.items() if k != "episode_wall_seconds"}
                    require(without_wall(choice["row"]) == without_wall(candidate["row"]),
                            f"Conflicting completed episode: {key} scenario {episode}")
                    duplicates += 1
            if choice is not None:
                selected[(key, episode)] = choice
    # Frozen supervisors count consecutive saved episodes. Reject corrupted or
    # manually sparse sources instead of mislabelling old rows as new throughput.
    for job in identity["jobs"]:
        key = common.job_key(job)
        episodes = sorted(ep for (name, ep) in selected if name == key)
        require(episodes == list(range(len(episodes))),
                f"Non-consecutive old checkpoints for {key}; preserve the source and repair its missing prefix first")
    return selected, duplicates


def verify_unchanged(sources):
    for source in sources:
        require(not active(source["root"]), f"Old run is still active: {source['root']}")
        require(common.sha(source["root"] / "manifest.json") == source["manifest_sha256"],
                "Old manifest changed during preparation")
        common.verify_assets(source["root"], source["assets"])
        for name, digest in source["used_files"].items():
            require(common.sha(inside(source["root"], name)) == digest, "Old result changed during preparation")


def prepare(root, identity, source_roots=(), *, apply=False):
    root = Path(root).expanduser().resolve()
    require(not root.exists(), "Use a new dedicated run path; existing runs are never overwritten")
    sources = [inspect_source(path, identity) for path in source_roots]
    require(len({source["root"] for source in sources}) == len(sources), "Duplicate reuse-run paths")
    selected, duplicates = select(sources, identity)
    total = len(identity["jobs"]) * identity["config"]["episodes"]
    complete_count = sum(all((common.job_key(job), ep) in selected for ep in range(identity["config"]["episodes"]))
                         for job in identity["jobs"])
    report = dict(run_root=str(root), total_jobs=len(identity["jobs"]), total_episodes=total,
                  reusable_episodes=len(selected), reusable_complete_jobs=complete_count,
                  remaining_episodes=total-len(selected), duplicate_agreements=duplicates,
                  sources=[dict(root=str(s["root"]), active=bool(active(s["root"]))) for s in sources])
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not apply:
        return report
    verify_unchanged(sources)
    root.parent.mkdir(parents=True, exist_ok=True)
    stage = root.parent / ("." + root.name + ".preparing-" + uuid.uuid4().hex)
    stage.mkdir()
    config, fingerprint = identity["config"], common.identity_digest(identity)
    try:
        # Generate the exact agreed five catalogs, then compare their bytes to
        # every old source. This independently checks deterministic prefix reuse.
        scenario = stage / "scenario"
        scenario.mkdir()
        for ep in range(config["episodes"]):
            path = scenario / f"targets_episode{ep:03d}.csv"
            serial.prepare_catalog(config, path, ep)
            relative = path.relative_to(stage).as_posix()
            for source in sources:
                require(common.sha(path) == source["assets"][relative], "Old catalogue differs from the fixed scenario")
        for n in config["satellite_counts"]:
            name = f"scenario/orbit_n{n}.npz"
            target = inside(stage, name)
            if sources:
                require(len({s["assets"][name] for s in sources}) == 1, "Source orbit caches disagree")
                shutil.copy2(inside(sources[0]["root"], name), target)
                for suffix in (".json", ".elements.csv"):
                    extra = target.with_suffix(suffix)
                    origin = inside(sources[0]["root"], extra.relative_to(stage).as_posix())
                    if origin.is_file():
                        shutil.copy2(origin, extra)
            else:
                serial.prepare_orbit(config, target, n)
        # Same scenario preparation and balance checks as the frozen runner;
        # existing files are not regenerated and its nine-asset manifest is new.
        accelerated.prepare_scenarios(stage, config)
        if config == common.read_json(CONFIG):
            from check_transmission_701x5 import check_orbit
            for n in config["satellite_counts"]:
                check_orbit(scenario / f"orbit_n{n}.npz", n, config)
        inventory, partials, completed = {}, {}, {}
        for job in identity["jobs"]:
            key, rows = common.job_key(job), []
            for ep in range(config["episodes"]):
                candidate = selected.get((key, ep))
                if candidate is None:
                    continue
                relative = f"episode_results/{key}/episode{ep:03d}.json"
                saved = dict(identity_sha256=fingerprint, execution_source_sha256=candidate["execution_source_sha256"],
                             row=candidate["row"], representative=candidate["representative"],
                             subset_provenance=candidate["subset_provenance"])
                common.atomic_json(stage / relative, saved)
                inventory[relative] = dict(sha256=common.sha(stage / relative),
                                          execution_source_sha256=candidate["execution_source_sha256"],
                                          **candidate["subset_provenance"])
                if identity["save_all_ledgers"]:
                    output = stage / "ledgers" / key / f"episode{ep:03d}.json"
                    output.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(candidate["ledger"], output)
                rows.append(candidate["row"])
            if len(rows) == config["episodes"]:
                result = dict(status="complete", protocol=config["protocol"], job=job, episodes=rows,
                              aggregate=serial.aggregate(rows), representative=selected[(key, 0)]["representative"],
                              execution=dict(mode="verified-subset-import", imported_episode_count=len(rows)))
                common.validate_job(result, job, config)
                common.atomic_json(stage / "jobs" / (key + ".json"), result)
                completed[key] = result
            else:
                partials[key] = rows
        migration = dict(kind="seed701-five-scenario-subset-v1", imported_at=time.time(),
                         sources=[dict(root=str(s["root"]), manifest_sha256=s["manifest_sha256"],
                                       identity=s["manifest"]["identity"], identity_sha256=s["identity_sha256"],
                                       execution_source_sha256=s["execution"]) for s in sources],
                         imported_complete_jobs=len(completed), imported_completed_episodes=len(inventory),
                         episode_files=inventory,
                         runtime_note="Original rows including wall times retained; old performance logs are not copied",
                         selection_note="Fixed episode indices 0..4, not selected by observed performance")
        common.atomic_json(stage / "migration.json", migration)
        accelerated.save_summaries(stage, identity["jobs"], completed, partials)
        common.atomic_json(stage / "progress.json", dict(status="prepared", completed_jobs=len(completed),
                           total_jobs=len(identity["jobs"]), completed_episodes=len(selected), total_episodes=total,
                           running_jobs=0, queued_jobs=len(identity["jobs"])-len(completed), failed_jobs=0,
                           workers=[], updated_at=time.time()))
        verify_unchanged(sources)
        # Commit the compatible manifest last; a failed preparation cannot be
        # mistaken for a resumable formal run. Keep staging data on any failure.
        common.atomic_json(stage / "manifest.json", dict(protocol=config["protocol"], runner=common.PARALLEL_VERSION,
                           identity=identity, execution_source_sha256=common.execution_hashes(),
                           migration_sha256=common.sha(stage / "migration.json"), prepared_at=time.time(),
                           subset_preparation_source_sha256=common.sha(Path(__file__))))
        require(not root.exists(), "Target appeared during preparation; refusing replacement")
        stage.rename(root)
    except BaseException:
        print(f"Preparation not committed; diagnostic staging directory retained: {stage}", flush=True)
        raise
    print(f"PREPARED {root}: reused {len(selected)}/{total} episodes; missing {total-len(selected)}.", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--reuse-run", type=Path, action="append", default=[])
    parser.add_argument("--model-root", type=Path, default=ROOT / "runs/thousand_paper_suite/training")
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    config = common.read_json(args.config)
    require(config == common.read_json(CONFIG), "Only the agreed 24-hour/701/five-scenario configuration is supported")
    plan = serial.make_plan(config, "checkpoint", args.model_root.resolve())
    identity = common.make_identity(config, plan, "checkpoint", args.device, False)
    prepare(args.run_root, identity, args.reuse_run, apply=args.apply)


if __name__ == "__main__":
    main()
