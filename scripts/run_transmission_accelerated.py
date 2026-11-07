"""Resource-bounded multi-GPU evaluation with unchanged serial experiment semantics."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from transmission_accelerated_common import (
    ROOT, PARALLEL_VERSION, acquire_lock, atomic_json, canonical_identity, execution_hashes,
    identity_digest, job_key, make_identity, read_json, sha, source_is_active,
    validate_job, validate_row, verify_assets, read_saved_episode, read_control, validate_neural_report,
)

STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True


def available_memory_gb():
    try:
        values = {line.split(":")[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()}
        available = values["MemAvailable"] * 1024
        maximum = Path("/sys/fs/cgroup/memory.max")
        current = Path("/sys/fs/cgroup/memory.current")
        if maximum.exists() and current.exists() and maximum.read_text().strip() != "max":
            available = min(available, max(0, int(maximum.read_text()) - int(current.read_text())))
        return available / 1024 ** 3
    except (OSError, KeyError, ValueError):
        return None


def available_cpus():
    count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            count = min(count, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return count


def make_slots(gpus, per_gpu, cpu_workers, threads, cpus, memory_gb, ram_per_worker_gb,
               gpu_info=None, gpu_per_worker_gb=4.):
    limit = max(1, cpus // threads)
    if memory_gb is not None:
        limit = min(limit, max(0, int((memory_gb - 8) // ram_per_worker_gb)))
    if limit < 1:
        raise RuntimeError("Insufficient available RAM for one worker plus the 8 GiB reserve")
    requested = [(gpu, index) for index in range(per_gpu) for gpu in gpus] if gpus else [(-1, index) for index in range(cpu_workers)]
    slots = []
    for gpu, index in requested:
        if gpu_info is not None:
            budget = max(0, int((gpu_info[gpu]["free_gb"] - 2) // gpu_per_worker_gb))
            if sum(slot["gpu"] == gpu for slot in slots) >= budget:
                continue
        slots.append(dict(id=f"gpu{gpu}:slot{index}" if gpu >= 0 else f"cpu:slot{index}", gpu=gpu))
        if len(slots) >= limit:
            break
    if not slots:
        raise RuntimeError("No worker slots fit the current CPU/RAM/VRAM planning budgets")
    return slots


def probe_gpus(indices):
    code = (
        "import json,torch; assert torch.cuda.is_available(), 'CUDA unavailable'; "
        "print(json.dumps([dict(index=i,name=torch.cuda.get_device_name(i),"
        "free_gb=torch.cuda.mem_get_info(i)[0]/1024**3,total_gb=torch.cuda.mem_get_info(i)[1]/1024**3) "
        "for i in range(torch.cuda.device_count())]))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    devices = {row["index"]: row for row in json.loads(result.stdout)}
    if any(index not in devices for index in indices):
        raise ValueError(f"Requested GPUs {indices}; PyTorch-visible indices are {list(devices)}. Respect CUDA_VISIBLE_DEVICES.")
    return devices


def prepare_scenarios(root, config, source=None):
    import run_transmission_load as serial
    scenario = root / "scenario"
    scenario.mkdir(exist_ok=True)
    assets_path = root / "scenario_sha256.json"
    expected = read_json(assets_path) if assets_path.exists() else None
    if source is not None:
        old_assets = read_json(source / "scenario_sha256.json")
        verify_assets(source, old_assets)
        for name in old_assets:
            relative = Path(name.replace("\\", "/"))
            target = (root / relative).resolve()
            if not target.is_relative_to(root):
                raise ValueError("Scenario target is outside the new run directory")
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / relative, target)
        for original in (source / "scenario").iterdir():
            if original.is_file() and not original.is_symlink() and original.suffix in {".csv", ".json"}:
                target = scenario / original.name
                if not target.exists():
                    shutil.copy2(original, target)
    assets = {}
    paths = []
    for episode in range(config["episodes"]):
        path = scenario / f"targets_episode{episode:03d}.csv"
        serial.prepare_catalog(config, path, episode)
        paths.append(path)
        assets[path.relative_to(root).as_posix()] = sha(path)
    if config["catalog"].get("sampling_protocol"):
        from balanced_task_catalog import balance_report
        report = balance_report(config, paths)
        atomic_json(root / "scenario_balance.json", report)
        print(f"Balanced catalogue checks passed: {report['episode_count']} scenarios, {len(config['task_counts'])} task counts each.", flush=True)
    for satellites in config["satellite_counts"]:
        path = scenario / f"orbit_n{satellites}.npz"
        serial.prepare_orbit(config, path, satellites)
        assets[path.relative_to(root).as_posix()] = sha(path)
    if expected is not None and {name.replace('\\', '/'): value for name, value in expected.items()} != assets:
        raise ValueError("Scenario hashes changed; refusing to mix results")
    if source is not None and {name.replace('\\', '/'): value for name, value in old_assets.items()} != assets:
        raise ValueError("Reused scenarios do not exactly match this evaluation plan")
    atomic_json(assets_path, assets)
    return assets


def load_completed(root, plan, config):
    results = {}
    for job in plan:
        path = root / "jobs" / (job_key(job) + ".json")
        if path.exists():
            result = read_json(path)
            validate_job(result, job, config)
            results[job_key(job)] = result
    return results


def import_completed(source, root, plan, config):
    source_manifest = read_json(source / "manifest.json")
    source_execution = source_manifest.get("execution_source_sha256")
    if source_manifest.get("runner") == "transmission-parallel-v1":
        import transmission_parallel_common as legacy
        if source_execution != legacy.execution_hashes():
            raise ValueError("Legacy execution files do not match their frozen source manifest")
    elif source_manifest.get("runner") == PARALLEL_VERSION:
        if source_execution != execution_hashes():
            raise ValueError("Different acceleration version; migration is not implicitly compatible")
    elif source_manifest.get("runner") is not None:
        raise ValueError("Unsupported source runner")
    results = load_completed(source, plan, config)
    inventory = {}
    for key, result in results.items():
        target = root / "jobs" / (key + ".json")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / "jobs" / target.name, target)
    fingerprint = identity_digest(source_manifest["identity"])
    for job in plan:
        key = job_key(job)
        for episode in range(config["episodes"]):
            path = source / "episode_results" / key / f"episode{episode:03d}.json"
            if not path.exists():
                break
            saved = read_saved_episode(source, path, fingerprint, source_execution)
            validate_row(saved["row"], job, episode, config)
            relative = path.relative_to(source).as_posix()
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and sha(target) != sha(path):
                raise ValueError("Migration target already contains a different episode")
            if not target.exists():
                shutil.copy2(path, target)
            inventory[relative] = dict(sha256=sha(path), execution_source_sha256=saved["execution_source_sha256"])
    migration = dict(source=str(source), source_runner=source_manifest.get("runner", "serial"),
                     source_manifest_sha256=sha(source / "manifest.json"), imported_complete_jobs=len(results),
                     imported_completed_episodes=len(inventory), episode_files=inventory, imported_at=time.time(),
                     runtime_note="Imported wall times are not accelerated throughput measurements; bytes are unchanged")
    atomic_json(root / "migration.json", migration)
    current = read_json(root / "manifest.json")
    current["migration_sha256"] = sha(root / "migration.json")
    atomic_json(root / "manifest.json", current)
    return len(results)


def load_partials(root, plan, config, completed, fingerprint, execution):
    partials = {}
    for job in plan:
        key = job_key(job)
        if key in completed:
            continue
        rows = []
        for episode in range(config["episodes"]):
            path = root / "episode_results" / key / f"episode{episode:03d}.json"
            if not path.exists():
                break
            saved = read_saved_episode(root, path, fingerprint, execution)
            validate_row(saved["row"], job, episode, config)
            rows.append(saved["row"])
        partials[key] = rows
    return partials


class EpisodeIndex:
    """Immutable episode checkpoints are parsed/validated once, not every tick."""
    def __init__(self, root, plan, config, completed, fingerprint, execution):
        self.root, self.plan, self.config = root, plan, config
        self.fingerprint, self.execution, self.rows = fingerprint, execution, {}
        self.imports = None
        manifest = read_json(root / "manifest.json")
        if "migration_sha256" in manifest:
            if sha(root / "migration.json") != manifest["migration_sha256"]:
                raise ValueError("Migration inventory changed")
            self.imports = read_json(root / "migration.json").get("episode_files", {})
        self.scan(completed)

    def scan(self, completed):
        for job in self.plan:
            key = job_key(job)
            if key in completed:
                self.rows.pop(key, None)
                continue
            rows = self.rows.setdefault(key, [])
            while len(rows) < self.config["episodes"]:
                episode = len(rows)
                path = self.root / "episode_results" / key / f"episode{episode:03d}.json"
                if not path.exists():
                    break
                saved = read_saved_episode(self.root, path, self.fingerprint, self.execution, self.imports)
                validate_row(saved["row"], job, episode, self.config)
                rows.append(saved["row"])
        return self.rows


def save_summaries(root, plan, completed, partials):
    import run_transmission_load as serial
    episodes, groups = [], []
    for job in plan:
        key = job_key(job)
        if key not in completed:
            episodes.extend(partials.get(key, []))
            continue
        result = completed[key]
        episodes.extend(result["episodes"])
        groups.append(dict(method=job["method"], training_seed=job["training_seed"], satellites=job["satellites"], tasks=job["tasks"],
                           **{f"{metric}_{stat}": value for metric, stats in result["aggregate"].items() for stat, value in stats.items()}))
    serial.write_csv(root / "episodes.csv", episodes)
    serial.write_csv(root / "grouped.csv", groups)
    serial.write_csv(root / "grouped_across_training_seeds.csv", serial.merge_training_seeds(groups))
    return len(episodes)


def render_progress(progress):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = progress["total_episodes"]
    done = progress["completed_episodes"]
    eta = progress.get("estimated_remaining_hours")
    print(f"[{stamp}] {progress['status']} | jobs {progress['completed_jobs']}/{progress['total_jobs']} "
          f"| episodes {done}/{total} ({100 * done / max(1, total):.1f}%) "
          f"| running {progress['running_jobs']} queued {progress['queued_jobs']} failed {progress['failed_jobs']} "
          f"| ETA {'warming up' if eta is None else f'{eta:.1f}h (rough)'}", flush=True)
    for state in progress.get("workers", []):
        horizon = state.get("simulated_horizon_seconds", 86400.)
        simulated = state.get("simulated_seconds", 0.)
        observed = state.get("last_observation_success_rate")
        delivered = state.get("last_observation_and_downlink_success_rate")
        rates = "--" if observed is None else f"obs {observed:.2%} / obs+downlink {delivered:.2%}"
        print(f"  {state['slot']} pid={state['pid']} {state['job']} "
              f"ep {state.get('episode', 1)}/{state.get('total_episodes', 20)} "
              f"sim {simulated / 3600:.2f}/{horizon / 3600:.2f}h "
              f"({100 * simulated / max(1., horizon):.1f}%) "
              f"events={state.get('event_transitions', 0)} phase={state['status']} last_done={rates}", flush=True)
        timing = state.get("stage_seconds", {})
        if timing:
            print(f"    wall={state.get('episode_wall_seconds_live', 0.) / 60:.1f}min "
                  f"init={timing.get('initialize_seconds', 0.):.1f}s "
                  f"policy+features={timing.get('policy_seconds_including_features', 0.):.1f}s "
                  f"environment={timing.get('environment_step_seconds', 0.):.1f}s", flush=True)


def choose_job(pending, active, slot, threshold, max_large):
    large = sum(item["slot"]["gpu"] == slot["gpu"] and item["job"]["satellites"] >= threshold for item in active.values())
    for index, job in enumerate(pending):
        if job["satellites"] < threshold or large < max_large:
            return pending.pop(index)
    return None


def resized_slots(args, control, active):
    gpus = [int(value) for value in args.gpus.split(",")] if args.device == "cuda" else []
    per_gpu = control.get("workers_per_gpu", args.workers_per_gpu)
    cpu_workers = control.get("cpu_workers", args.cpu_workers)
    memory = available_memory_gb()
    info = probe_gpus(gpus) if gpus else None
    requested = make_slots(gpus, per_gpu, cpu_workers, args.threads_per_worker,
                           available_cpus(), None, args.ram_per_worker_gb)
    occupied = {item["slot"]["id"] for item in active.values()}
    ram_room = float("inf") if memory is None else max(0., memory - 8.)
    gpu_room = {gpu: max(0., info[gpu]["free_gb"] - 2.) for gpu in gpus}
    result = []
    for slot in requested:
        if slot["id"] in occupied:
            result.append(slot)  # Never terminate or re-budget a current episode.
            continue
        if ram_room < args.ram_per_worker_gb:
            continue
        if slot["gpu"] >= 0 and gpu_room[slot["gpu"]] < args.gpu_memory_per_worker_gb:
            continue
        result.append(slot)
        ram_room -= args.ram_per_worker_gb
        if slot["gpu"] >= 0:
            gpu_room[slot["gpu"]] -= args.gpu_memory_per_worker_gb
    if not result:
        raise RuntimeError("No requested slots fit current free resources; existing workers are retained")
    return result


def schedule(root, identity, args, slots, completed):
    config, plan = identity["config"], identity["jobs"]
    fingerprint, execution = identity_digest(identity), execution_hashes()
    index = EpisodeIndex(root, plan, config, completed, fingerprint, execution)
    partials = index.rows
    pending = sorted([job for job in plan if job_key(job) not in completed],
                     key=lambda job: (job["tasks"], -job["satellites"], job["training_seed"], job["method"]))
    active, failed = {}, {}
    initial_done = sum(len(result["episodes"]) for result in completed.values()) + sum(map(len, partials.values()))
    started, last_report = time.monotonic(), -float("inf")
    last_summary_signature, draining = None, False
    control_id = read_control(root, os.getpid()).get("command_id")
    (root / "logs").mkdir(exist_ok=True)

    def update(status):
        nonlocal partials, last_summary_signature
        partials = index.scan(completed)
        done = sum(len(result["episodes"]) for result in completed.values()) + sum(map(len, partials.values()))
        signature = (done, len(completed))
        if signature != last_summary_signature:
            save_summaries(root, plan, completed, partials)
            last_summary_signature = signature
        elapsed = time.monotonic() - started
        throughput = (done - initial_done) / max(1., elapsed) * 3600
        states = []
        for key, item in active.items():
            path = root / "workers" / (key + ".json")
            state = read_json(path) if path.exists() else {}
            if state.get("pid") != item["process"].pid:
                state = dict(job=key, pid=item["process"].pid, slot=item["slot"]["id"], status="starting")
            states.append(state)
        total = len(plan) * config["episodes"]
        progress = dict(status=status, completed_jobs=len(completed), total_jobs=len(plan),
                        completed_episodes=done, total_episodes=total, running_jobs=len(active),
                        queued_jobs=len(pending), failed_jobs=len(failed), failures=failed,
                        session_elapsed_seconds=elapsed, new_episodes_per_hour=throughput,
                        estimated_remaining_hours=(total - done) / throughput if throughput > 0 and done - initial_done >= max(4, len(slots)) else None,
                        workers=states, updated_at=time.time(), target_worker_slots=len(slots),
                        max_large_per_gpu=args.max_large_per_gpu, draining=draining,
                        newly_completed_episodes=done - initial_done)
        atomic_json(root / "progress.json", progress)
        render_progress(progress)
        return done

    try:
        while pending or active:
            if STOP:
                break  # Emergency stop; normal operator pause uses the drain control.
            control = read_control(root, os.getpid())
            if control.get("command_id") != control_id:
                command_id = control.get("command_id")
                try:
                    new_slots = slots if control.get("action") == "drain" else resized_slots(args, control, active)
                    args.max_large_per_gpu = control.get("max_large_per_gpu", args.max_large_per_gpu)
                    slots = new_slots
                    draining = control.get("action") == "drain"
                    args.workers_per_gpu = control.get("workers_per_gpu", args.workers_per_gpu)
                    args.cpu_workers = control.get("cpu_workers", args.cpu_workers)
                    history = read_json(root / "parallel_execution.json")
                    history.append(dict(updated_at=time.time(), command_id=command_id, slots=slots,
                                        max_large_per_gpu=args.max_large_per_gpu, draining=draining,
                                        threads_per_worker=args.threads_per_worker))
                    atomic_json(root / "parallel_execution.json", history)
                    atomic_json(root / "execution_control_ack.json",
                                dict(command_id=command_id, accepted=True, slots=len(slots), draining=draining))
                    print(f"CONTROL accepted: slots={len(slots)} large/GPU={args.max_large_per_gpu} drain={draining}", flush=True)
                except (RuntimeError, ValueError, subprocess.SubprocessError) as error:
                    # Workers also see this file at episode boundaries. Revert
                    # rejected limits so they cannot repeatedly drain/restart.
                    atomic_json(root / "execution_control.json", dict(run_root=str(root), owner_pid=os.getpid(),
                                command_id=command_id, action="drain" if draining else "run",
                                workers_per_gpu=args.workers_per_gpu, cpu_workers=args.cpu_workers,
                                max_large_per_gpu=args.max_large_per_gpu))
                    atomic_json(root / "execution_control_ack.json",
                                dict(command_id=command_id, accepted=False, error=str(error)))
                    print(f"CONTROL rejected; current workers retained: {error}", flush=True)
                control_id = command_id
                last_report = -float("inf")
            for key, item in list(active.items()):
                code = item["process"].poll()
                if code is None:
                    continue
                item["log"].close()
                del active[key]
                if code == 0:
                    result = read_json(root / "jobs" / (key + ".json"))
                    validate_job(result, item["job"], config)
                    completed[key] = result
                    partials.pop(key, None)
                elif code == 75:
                    pending.append(item["job"])
                    print(f"DRAINED {key}: complete episode checkpoints retained.", flush=True)
                else:
                    failed[key] = dict(exit_code=code, log=str(root / "logs" / (key + ".log")))
                    print(f"FAILED {key}: exit={code}; no new jobs will be launched.", flush=True)
                last_report = -float("inf")
            if not failed and not draining:
                used = {item["slot"]["id"] for item in active.values()}
                for slot in slots:
                    if slot["id"] in used:
                        continue
                    job = choose_job(pending, active, slot, args.large_satellites, args.max_large_per_gpu)
                    if job is None:
                        continue
                    key = job_key(job)
                    environment = os.environ.copy()
                    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                        environment[variable] = str(args.threads_per_worker)
                    environment["PYTHONUNBUFFERED"] = "1"
                    log = (root / "logs" / (key + ".log")).open("a", encoding="utf-8")
                    command = [sys.executable, "-u", str(ROOT / "scripts/transmission_accelerated_worker.py"),
                               "--run-root", str(root), "--job", key, "--gpu-index", str(max(0, slot["gpu"])),
                               "--slot", slot["id"], "--threads", str(args.threads_per_worker),
                               "--progress-seconds", str(args.progress_seconds), "--parent-pid", str(os.getpid()),
                               "--cache-dir", str(args.cache_dir)]
                    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=(os.name == "posix"))
                    active[key] = dict(process=process, slot=slot, job=job, log=log)
                    print(f"START {slot['id']} pid={process.pid} {key}", flush=True)
            if time.monotonic() - last_report >= args.progress_seconds:
                done = update("failed" if failed else "draining" if draining else "running")
                if args.drain_after_new_episodes and done - initial_done >= args.drain_after_new_episodes and not draining:
                    control = dict(run_root=str(root), owner_pid=os.getpid(), action="drain",
                                   command_id=f"automatic-{time.time_ns()}")
                    atomic_json(root / "execution_control.json", control)
                last_report = time.monotonic()
            if not active and (failed or draining):
                break
            time.sleep(.2)
    finally:
        # SIGTERM is an emergency path. A normal drain reaches here with no live workers.
        for item in active.values():
            if item["process"].poll() is None:
                item["process"].terminate()
        deadline = time.monotonic() + 60
        for item in active.values():
            try:
                item["process"].wait(timeout=max(0., deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                item["process"].kill()
                item["process"].wait()
            item["log"].close()
        active.clear()
        completed.update(load_completed(root, plan, config))
        status = ("complete" if len(completed) == len(plan) else "interrupted" if STOP
                  else "failed" if failed else "drained" if draining else "failed")
        update(status)
    if len(completed) != len(plan) and not draining:
        raise RuntimeError("Evaluation incomplete. Completed episodes are retained; inspect logs before resume.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/transmission_day_v3_full_sweep.json")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=ROOT / "runs/thousand_paper_suite/training")
    parser.add_argument("--reuse-run", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clear-stale-lock", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--gpus", default="0,1", help="Logical GPU indices visible to PyTorch; CUDA_VISIBLE_DEVICES is preserved")
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--cpu-workers", type=int, default=2)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--max-large-per-gpu", type=int, default=2)
    parser.add_argument("--large-satellites", type=int, default=1024)
    parser.add_argument("--ram-per-worker-gb", type=float, default=16.)
    parser.add_argument("--gpu-memory-per-worker-gb", type=float, default=4.)
    parser.add_argument("--progress-seconds", type=float, default=15.)
    parser.add_argument("--policy", choices=["checkpoint", "priority"], default="checkpoint")
    parser.add_argument("--save-all-ledgers", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--validation-report", type=Path,
                        help="Passed baseline/accelerated neural prefix comparison from check_accelerated_equivalence.py")
    parser.add_argument("--drain-after-new-episodes", type=int, default=0,
                        help="Optional pilot: finish current episodes and pause after at least this many new completions")
    args = parser.parse_args()
    if args.drain_after_new_episodes < 0:
        parser.error("--drain-after-new-episodes must be nonnegative")
    root = args.run_root.expanduser().resolve()
    args.cache_dir = (args.cache_dir or root / "performance_cache").expanduser().resolve()
    if args.status:
        render_progress(read_json(root / "progress.json"))
        return
    for name in ("workers_per_gpu", "cpu_workers", "threads_per_worker", "max_large_per_gpu", "large_satellites",
                 "ram_per_worker_gb", "gpu_memory_per_worker_gb", "progress_seconds"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(args.threads_per_worker)
    import run_transmission_load as serial
    config = read_json(args.config)
    for field in ("satellite_counts", "task_counts", "training_seeds"):
        config[field] = serial.integer_list(None, config[field])
    if config["episodes"] < 1 or not config["methods"] or len(set(config["methods"])) != len(config["methods"]):
        raise ValueError("Episodes and method list must be positive/nonempty and distinct")
    if config["protocol"] != serial.EVENT_VERSION:
        raise ValueError("Parallel evaluation requires the event-v3 physical protocol")
    serial.EventTransmissionConfig(**config["env"])
    if config["catalog"].get("sampling_protocol"):
        from balanced_task_catalog import grid_options
        grid_options(config)
    if args.policy == "priority":
        config.update(methods=["priority"], training_seeds=[701])
    plan = serial.make_plan(config, args.policy, args.model_root.resolve())
    if args.plan_only:
        print(json.dumps(dict(config=config, jobs=plan, execution=dict(gpus=args.gpus,
              workers_per_gpu=args.workers_per_gpu, threads_per_worker=args.threads_per_worker)), ensure_ascii=False, indent=2))
        return
    identity = make_identity(config, plan, args.policy, args.device, args.save_all_ledgers)
    validation = None
    if args.policy == "checkpoint" and not args.prepare_only:
        report_path = args.validation_report
        if report_path is None and args.resume and (root / "validation_report.json").exists():
            report_path = root / "validation_report.json"
        if report_path is None:
            raise ValueError("Run check_accelerated_equivalence.py first and supply --validation-report; existing runs are untouched")
        validation = read_json(report_path)
        validate_neural_report(validation, identity)
    source = args.reuse_run.expanduser().resolve() if args.reuse_run else None
    if source is not None:
        if source == root:
            raise ValueError("Use a new parallel output directory; --reuse-run is read-only")
        if source_is_active(source):
            raise RuntimeError("Source serial/parallel run is still active. Stop that exact run before migration.")
        if canonical_identity(read_json(source / "manifest.json")["identity"]) != canonical_identity(identity):
            raise ValueError("Source configuration, physics/policy code or checkpoints differ; cannot import results")
    slots = []
    if not args.prepare_only:
        gpus = [int(value) for value in args.gpus.split(",")] if args.device == "cuda" else []
        if len(gpus) != len(set(gpus)) or any(value < 0 for value in gpus):
            raise ValueError("GPU indices must be distinct and nonnegative")
        gpu_info = probe_gpus(gpus) if gpus else None
        memory = available_memory_gb()
        slots = make_slots(gpus, args.workers_per_gpu, args.cpu_workers, args.threads_per_worker,
                           available_cpus(), memory, args.ram_per_worker_gb, gpu_info, args.gpu_memory_per_worker_gb)
        print(f"Resources: CPUs={available_cpus()} available_RAM={memory}GiB workers={len(slots)} "
              f"threads/worker={args.threads_per_worker} slots={[slot['id'] for slot in slots]}", flush=True)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        previous = read_json(manifest_path)
        if not args.resume or previous.get("runner") != PARALLEL_VERSION:
            raise ValueError("Use --resume for an existing parallel run, or choose a new directory")
        if canonical_identity(previous["identity"]) != canonical_identity(identity) or previous["execution_source_sha256"] != execution_hashes():
            raise ValueError("Resume inputs/physics/policy/execution code changed")
    elif any(root.iterdir()):
        raise ValueError("Use an empty dedicated parallel output directory")
    lock = acquire_lock(root, args.clear_stale_lock)
    try:
        if not manifest_path.exists():
            atomic_json(manifest_path, dict(protocol=config["protocol"], runner=PARALLEL_VERSION,
                        identity=identity, execution_source_sha256=execution_hashes(),
                        reuse_source=str(source) if source else None, created_at=time.time(),
                        note="unchanged physical/policy semantics; timing includes process/GPU contention"))
        if validation is not None:
            atomic_json(root / "validation_report.json", validation)
        if not (root / "progress.json").exists():
            atomic_json(root / "progress.json", dict(status="preparing", completed_jobs=0, total_jobs=len(plan),
                        completed_episodes=0, total_episodes=len(plan) * config["episodes"], running_jobs=0,
                        queued_jobs=len(plan), failed_jobs=0, workers=[], updated_at=time.time()))
        prepare_scenarios(root, config, source)
        if source is not None:
            count = import_completed(source, root, plan, config)
            imported = read_json(root / "migration.json")
            print(f"Reused exact scenario bytes, {count} complete jobs and "
                  f"{imported['imported_completed_episodes']} episode checkpoints; old run files were not modified.", flush=True)
        if args.prepare_only:
            print(f"Scenario prepared: {root / 'scenario'}")
            return
        execution_path = root / "parallel_execution.json"
        history = read_json(execution_path) if execution_path.exists() else []
        history.append(dict(started_at=time.time(), slots=slots, threads_per_worker=args.threads_per_worker,
                            max_large_per_gpu=args.max_large_per_gpu, large_satellites=args.large_satellites,
                            resource_budgets_note="planning caps, not measured per-worker memory requirements"))
        atomic_json(execution_path, history)
        atomic_json(root / "execution_control.json", dict(run_root=str(root), owner_pid=os.getpid(), action="run",
                    workers_per_gpu=args.workers_per_gpu, cpu_workers=args.cpu_workers,
                    max_large_per_gpu=args.max_large_per_gpu, command_id=f"startup-{time.time_ns()}"))
        schedule(root, identity, args, slots, load_completed(root, plan, config))
    except BaseException as error:
        progress_path = root / "progress.json"
        if progress_path.exists():
            progress = read_json(progress_path)
            progress.update(status="interrupted" if STOP else "failed", error=str(error), updated_at=time.time())
            atomic_json(progress_path, progress)
        raise
    finally:
        if lock.exists() and lock.read_text().strip() == str(os.getpid()):
            lock.unlink()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    main()


