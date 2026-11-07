"""Shared identity, validation and process-control helpers for parallel evaluation."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import time

ROOT = Path(__file__).resolve().parents[1]
PARALLEL_VERSION = "transmission-accelerated-v2"
EXECUTION_FILES = ("scripts/run_transmission_accelerated.py", "scripts/transmission_accelerated_worker.py",
                   "scripts/transmission_accelerated_common.py", "src/sat_marl_env/accelerated_transmission.py",
                   "scripts/accelerated_features.py")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(40):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            # Windows readers/antivirus can briefly deny atomic replacement.
            # Linux keeps the original single-rename path; never truncate target.
            if os.name != "nt" or attempt == 39:
                raise
            time.sleep(.025)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_identity(identity):
    identity = json.loads(json.dumps(identity))
    identity["source_sha256"] = {name.replace("\\", "/"): digest
                                  for name, digest in identity["source_sha256"].items()}
    return identity


def identity_digest(identity):
    return hashlib.sha256(json.dumps(canonical_identity(identity), sort_keys=True,
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def execution_hashes():
    return {name: sha(ROOT / name) for name in EXECUTION_FILES}


def make_identity(config, plan, policy, device, save_all_ledgers=False):
    names = ["scripts/run_transmission_load.py", "src/sat_marl_env/transmission.py",
             "src/sat_marl_env/env.py", "src/sat_marl_env/config.py", "src/sat_marl_env/entities.py",
             "src/sat_marl_env/orbital.py", "src/sat_marl_env/real_scenario.py",
             "scripts/build_kepler_ephemeris.py", "scripts/generate_task_catalogs.py"]
    if policy == "checkpoint":
        names.extend(("scripts/evaluate_marl.py", "scripts/marl_common.py"))
    if config["protocol"] == "optical-ground-delivery-event-v3":
        names.append("src/sat_marl_env/event_transmission.py")
    if config["catalog"].get("sampling_protocol"):
        names.append("scripts/balanced_task_catalog.py")
    checkpoints = sorted({job["checkpoint"] for job in plan if job["checkpoint"] is not None})
    for checkpoint in checkpoints:
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return dict(config=config, policy=policy, jobs=plan, device=device, save_all_ledgers=save_all_ledgers,
                source_sha256={name: sha(ROOT / name) for name in names},
                checkpoint_sha256={name: sha(name) for name in checkpoints})


def job_key(job):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", job["method"]):
        raise ValueError("Unsafe method name")
    return f"{job['method']}_seed{job['training_seed']}_n{job['satellites']}_m{job['tasks']}"


def validate_row(row, job, episode, config):
    for name in ("method", "training_seed", "satellites", "tasks"):
        if row[name] != job[name]:
            raise ValueError(f"Wrong episode identity: {name}")
    if row["episode"] != episode or row["scenario_seed"] != config["scenario_seed"] + episode:
        raise ValueError("Wrong episode/scenario seed")
    for name, value in row.items():
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise ValueError(f"Non-finite episode metric: {name}")
    observed, delivered, total = (row[name] for name in (
        "observation_success_tasks", "observation_and_downlink_success_tasks", "tasks"))
    if not (0 <= delivered <= observed <= total):
        raise ValueError("Invalid observation/ground-delivery inclusion")
    for metric, count in (("observation_success_rate", observed),
                          ("observation_and_downlink_success_rate", delivered)):
        if not math.isclose(row[metric], count / total, abs_tol=1e-10, rel_tol=0):
            raise ValueError(f"Wrong success-rate denominator: {metric}")
    horizon = config["env"]["max_steps"] * config["env"]["step_duration_seconds"]
    if not math.isclose(row["simulated_seconds"], horizon, abs_tol=1e-6, rel_tol=0):
        raise ValueError("Episode did not finish the configured physical horizon")
    if row["max_rolling_imaging_seconds"] > config["env"]["imaging_budget_seconds"] + 1e-6:
        raise ValueError("Rolling imaging budget violated")
    residual = row["committed_data_mb"] - row["downlinked_data_mb"] - row["pending_data_mb"]
    if abs(residual) > 1e-5 * max(1., row["committed_data_mb"]):
        raise ValueError("Data conservation violated")


def validate_job(result, job, config):
    if result.get("status") != "complete" or result.get("protocol") != config["protocol"]:
        raise ValueError("Incomplete/wrong-protocol job")
    if result["job"] != job or len(result["episodes"]) != config["episodes"]:
        raise ValueError("Wrong job identity/episode count")
    for episode, row in enumerate(result["episodes"]):
        validate_row(row, job, episode, config)


def process_matches_run(pid, run_root, script_names, process_root=Path("/proc")):
    """Linux identity check; no PID-only signalling or stale-lock guessing."""
    if pid <= 1 or not process_root.is_dir():
        return False
    process = process_root / str(pid)
    try:
        if hasattr(os, "getuid") and process.stat().st_uid != os.getuid():
            return False
        arguments = (process / "cmdline").read_bytes().decode("utf-8", errors="replace").split("\0")
    except OSError:
        return False
    if not any(Path(argument).name in script_names for argument in arguments if argument):
        return False
    values = []
    for index, argument in enumerate(arguments):
        if argument == "--run-root" and index + 1 < len(arguments):
            values.append(arguments[index + 1])
        elif argument.startswith("--run-root="):
            values.append(argument.split("=", 1)[1])
    try:
        working_directory = (process / "cwd").resolve()
    except OSError:
        return False
    return any((Path(value).expanduser() if Path(value).is_absolute() else working_directory / value).resolve()
               == Path(run_root).resolve() for value in values)


def active_worker_pids(root):
    root = Path(root)
    candidates = set()
    for path in (root / "workers").glob("*.json"):
        state = read_json(path)
        if isinstance(state.get("pid"), int):
            candidates.add(state["pid"])
    progress = root / "progress.json"
    if progress.exists():
        for state in read_json(progress).get("workers", []):
            if isinstance(state.get("pid"), int):
                candidates.add(state["pid"])
    return sorted(pid for pid in candidates
                  if process_matches_run(pid, root, {"transmission_accelerated_worker.py", "transmission_parallel_worker.py"}))


def source_is_active(root):
    if active_worker_pids(root):
        return True
    host_file = Path(root) / "parallel_host.json"
    if host_file.exists():
        host = read_json(host_file)
        if (host.get("hostname") == socket.gethostname() and isinstance(host.get("pid"), int)
                and process_matches_run(host["pid"], root, {"run_transmission_accelerated.py", "run_transmission_parallel.py"})):
            return True
    lock = Path(root) / "runner.lock"
    if not lock.exists():
        return False
    try:
        pid = int(lock.read_text().strip())
    except (ValueError, OSError):
        raise ValueError(f"Cannot interpret source lock: {lock}")
    return process_matches_run(pid, root, {"run_transmission_load.py", "run_transmission_accelerated.py", "run_transmission_parallel.py"})


def acquire_lock(root, clear_stale=False):
    lock = Path(root) / "runner.lock"
    if source_is_active(root):
        raise RuntimeError("This run still has a live owner/worker; refusing to start a second supervisor")
    if lock.exists() and clear_stale:
        pid = int(lock.read_text().strip())
        host_file = Path(root) / "parallel_host.json"
        if not host_file.exists() or read_json(host_file)["hostname"] != socket.gethostname():
            raise RuntimeError("Cannot clear an unverified or different-host lock")
        if active_worker_pids(root):
            raise RuntimeError("Old worker processes are still alive; refusing to clear the owner lock")
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            lock.unlink()
        else:
            raise RuntimeError(f"PID {pid} still exists; refusing to remove its lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    atomic_json(Path(root) / "parallel_host.json", dict(hostname=socket.gethostname(), pid=os.getpid(), started=time.time()))
    return lock


def verify_assets(root, expected):
    root = Path(root).resolve()
    for name, digest in expected.items():
        path = (root / name.replace("\\", "/")).resolve()
        if not path.is_relative_to(root) or sha(path) != digest:
            raise ValueError(f"Changed or invalid scenario asset: {name}")


def configured_options(config, job, episode, metrics, scenario):
    """The same checkpoint architecture/TTL overrides as the frozen serial entry."""
    options = dict(config["env"])
    trained_env = metrics.get("env_config") or {}
    for name in ("candidate_k", "neighbor_k", "coalition_reservations",
                 "reservation_ttl_steps", "reservation_failure_penalty"):
        if name in trained_env:
            options[name] = (trained_env[name] * trained_env.get("step_duration_seconds", 30.)
                             if name == "reservation_ttl_steps" and config["protocol"] == "optical-ground-delivery-event-v3"
                             else trained_env[name])
    options.update(num_satellites=job["satellites"], num_tasks=job["tasks"],
                   num_planes=config["planes"][str(job["satellites"])],
                   ephemeris_cache_path=str(Path(scenario) / f"orbit_n{job['satellites']}.npz"),
                   task_catalog_path=str(Path(scenario) / f"targets_episode{episode:03d}.csv"),
                   task_layout="catalog", random_seed=config["scenario_seed"] + episode)
    return options


def read_saved_episode(root, path, fingerprint, execution, imports=None):
    """Keep imported v1 files byte-identical, with explicit verified provenance."""
    root, path = Path(root), Path(path)
    saved = read_json(path)
    if saved.get("identity_sha256") != fingerprint:
        raise ValueError("Saved episode has different scientific inputs")
    if saved.get("execution_source_sha256") != execution:
        if imports is None:
            manifest = read_json(root / "manifest.json")
            migration_path = root / "migration.json"
            if not migration_path.exists() or sha(migration_path) != manifest.get("migration_sha256"):
                raise ValueError("Unverified legacy episode migration")
            imports = read_json(migration_path).get("episode_files", {})
        relative = path.relative_to(root).as_posix()
        record = imports.get(relative)
        if (not record or sha(path) != record["sha256"]
                or saved.get("execution_source_sha256") != record["execution_source_sha256"]):
            raise ValueError("Legacy episode differs from the migration inventory")
    return saved


def read_control(root, owner_pid=None):
    path = Path(root) / "execution_control.json"
    if not path.exists():
        return {}
    state = read_json(path)
    if state.get("run_root") != str(Path(root).resolve()):
        raise ValueError("Control message belongs to another run")
    if owner_pid is not None and state.get("owner_pid") != owner_pid:
        return {}  # A previous supervisor's drain request does not affect resume.
    if state.get("action") not in {"run", "drain"}:
        raise ValueError("Unknown execution-control action")
    for name in ("workers_per_gpu", "cpu_workers", "max_large_per_gpu"):
        value = state.get(name)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
            raise ValueError("Control concurrency limits must be positive integers")
    return state


def slot_should_drain(control, slot):
    if control.get("action") == "drain":
        return True
    if ":slot" not in slot:
        raise ValueError("Invalid worker slot")
    index = int(slot.rsplit(":slot", 1)[1])
    key = "cpu_workers" if slot.startswith("cpu:") else "workers_per_gpu"
    return key in control and index >= control[key]


def validate_neural_report(report, identity):
    expected = set(identity["checkpoint_sha256"].values())
    if (report.get("status") != "passed" or report.get("identity_sha256") != identity_digest(identity)
            or report.get("execution_source_sha256") != execution_hashes()
            or report.get("checker_sha256") != sha(ROOT / "scripts/check_accelerated_equivalence.py")
            or report.get("device") != identity["device"]
            or not expected.issubset(set(report.get("checked_checkpoint_sha256", [])))):
        raise ValueError("Validation report does not cover current code, inputs, device and checkpoints")
    jobs = {job_key(job): job for job in identity["jobs"]}
    sizes = identity["config"]["satellite_counts"]
    required_sizes = {min(sizes), max(sizes)}
    required = {(identity["checkpoint_sha256"][job["checkpoint"]], job["satellites"])
                for job in jobs.values() if job["satellites"] in required_sizes}
    checked = set()
    horizon = identity["config"]["env"]["max_steps"] * identity["config"]["env"]["step_duration_seconds"]
    for case in report.get("cases", []):
        job = jobs.get(case.get("job"))
        if job is None:
            raise ValueError("Validation case is outside the experiment plan")
        digest = identity["checkpoint_sha256"][job["checkpoint"]]
        if (case.get("checkpoint_sha256") != digest or
                not (case.get("checked_transitions", 0) >= 20 or case.get("simulated_seconds", 0) >= horizon)):
            raise ValueError("Validation case has insufficient transitions or a different checkpoint")
        checked.add((digest, job["satellites"]))
    if not required.issubset(checked):
        raise ValueError("Validate both smallest and largest constellations for every original model")


