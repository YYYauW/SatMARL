"""One isolated GPU/CPU process per evaluation job; save every completed episode."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import io
import os
from pathlib import Path
import signal
import time
import traceback

from transmission_parallel_common import (
    ROOT, atomic_json, configured_options, execution_hashes, identity_digest,
    job_key, read_json, sha, validate_job, validate_row,
)

STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True


def run_job(args):
    def stopping():
        return STOP or (os.name == "posix" and args.parent_pid is not None and os.getppid() != args.parent_pid)

    if stopping():
        return 130
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(args.threads)
    import run_transmission_load as serial
    manifest = read_json(args.run_root / "manifest.json")
    identity, execution = manifest["identity"], manifest["execution_source_sha256"]
    if execution_hashes() != execution:
        raise RuntimeError("Parallel execution source changed")
    for name, digest in identity["source_sha256"].items():
        if sha(ROOT / name.replace("\\", "/")) != digest:
            raise RuntimeError(f"Physical/policy source changed: {name}")
    config = identity["config"]
    job = next(job for job in identity["jobs"] if job_key(job) == args.job)
    fingerprint = identity_digest(identity)
    status_path = args.run_root / "workers" / (args.job + ".json")
    checkpoint_directory = args.run_root / "episode_results" / args.job
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    started_job = time.perf_counter()
    state = dict(job=args.job, pid=os.getpid(), gpu_index=args.gpu_index, slot=args.slot,
                 status="initializing", completed_episodes=0, total_episodes=config["episodes"],
                 episode=1, simulated_seconds=0., simulated_horizon_seconds=config["env"]["max_steps"] * config["env"]["step_duration_seconds"],
                 event_transitions=0, last_observation_success_rate=None,
                 last_observation_and_downlink_success_rate=None)

    def progress(**changes):
        state.update(changes, updated_at=time.time(), worker_elapsed_seconds=time.perf_counter() - started_job)
        atomic_json(status_path, state)

    progress()
    torch, device, checkpoint, metrics, choose = None, None, None, {}, None
    if identity["policy"] == "checkpoint":
        import torch
        from marl_common import resolve_device
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        if identity["device"] == "cuda":
            torch.cuda.set_device(args.gpu_index)
        device = resolve_device(identity["device"])
        raw = Path(job["checkpoint"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != identity["checkpoint_sha256"][job["checkpoint"]]:
            raise RuntimeError("Checkpoint changed after the run was created")
        checkpoint = torch.load(io.BytesIO(raw), map_location=device, weights_only=False)
        metrics = dict(checkpoint.get("metrics") or {})
    event_mode = config["protocol"] == serial.EVENT_VERSION
    config_class = serial.EventTransmissionConfig if event_mode else serial.TransmissionConfig
    env_class = serial.EventTransmissionEnv if event_mode else serial.TransmissionEnv
    rows, representative = [], None
    try:
        for episode in range(config["episodes"]):
            saved_path = checkpoint_directory / f"episode{episode:03d}.json"
            if saved_path.exists():
                saved = read_json(saved_path)
                if saved["identity_sha256"] != fingerprint or saved["execution_source_sha256"] != execution:
                    raise RuntimeError("Episode checkpoint belongs to different inputs/code")
                validate_row(saved["row"], job, episode, config)
                rows.append(saved["row"])
                if episode == 0:
                    representative = saved["representative"]
                progress(completed_episodes=len(rows))
                continue
            if stopping():
                progress(status="interrupted")
                return 130
            progress(status="initializing", episode=episode + 1, completed_episodes=len(rows),
                     simulated_seconds=0., event_transitions=0)
            options = configured_options(config, job, episode, metrics, args.run_root / "scenario")
            started = time.perf_counter()
            env = env_class(config_class(**options))
            observations, _ = env.reset(seed=config["scenario_seed"] + episode)
            if checkpoint is not None and choose is None:
                from evaluate_marl import build_policy
                _, choose = build_policy(checkpoint, metrics, observations, device)
            reset = getattr(choose, "reset_strategy", None)
            if callable(reset):
                reset()
            last_report = 0.
            while env.step_count < env.config.max_steps:
                if stopping():
                    progress(status="interrupted", simulated_seconds=float(env.step_count * env.config.step_duration_seconds))
                    return 130
                actions = choose(observations) if choose is not None else serial.priority_actions(env, observations)
                observations, _, terminated, truncated, _ = env.step(actions)
                now = time.perf_counter()
                if now - last_report >= args.progress_seconds:
                    progress(status="running", simulated_seconds=float(env.step_count * env.config.step_duration_seconds),
                             event_transitions=getattr(env, "_event_count", int(env.step_count)))
                    last_report = now
                if all(terminated.values()) or all(truncated.values()):
                    break
            summary = env.summary()
            summary.setdefault("simulated_seconds", env.step_count * env.config.step_duration_seconds)
            summary.setdefault("event_transitions", env.step_count)
            for name in ("imaging_startup_count", "downlink_startup_count", "imaging_power_on_seconds", "radio_power_on_seconds"):
                summary.setdefault(name, 0.)
            summary["episode_wall_seconds"] = time.perf_counter() - started
            row = dict(method=job["method"], training_seed=job["training_seed"], satellites=job["satellites"],
                       tasks=job["tasks"], episode=episode, scenario_seed=config["scenario_seed"] + episode,
                       **{name: summary[name] for name in serial.METRICS})
            validate_row(row, job, episode, config)
            current_representative = None
            if episode == 0:
                current_representative = dict(summary=summary, acquisitions=env.acquisition_ledger(),
                                              task_outcomes=env.task_outcomes(), stations=[asdict(g) for g in env.ground_stations])
                if event_mode:
                    current_representative["radio_power_sessions"] = env.radio_power_ledger()
                representative = current_representative
            atomic_json(saved_path, dict(identity_sha256=fingerprint, execution_source_sha256=execution,
                                         row=row, representative=current_representative))
            if identity["save_all_ledgers"]:
                atomic_json(args.run_root / "ledgers" / args.job / f"episode{episode:03d}.json",
                            dict(acquisitions=env.acquisition_ledger(), task_outcomes=env.task_outcomes()))
            rows.append(row)
            progress(status="episode_complete", completed_episodes=len(rows), simulated_seconds=row["simulated_seconds"],
                     event_transitions=row["event_transitions"],
                     last_observation_success_rate=row["observation_success_rate"],
                     last_observation_and_downlink_success_rate=row["observation_and_downlink_success_rate"])
            print(f"{args.job} episode {episode + 1}/{config['episodes']} "
                  f"obs={row['observation_success_tasks']}/{job['tasks']} ({row['observation_success_rate']:.2%}) "
                  f"obs+downlink={row['observation_and_downlink_success_tasks']}/{job['tasks']} "
                  f"({row['observation_and_downlink_success_rate']:.2%}) "
                  f"wall={row['episode_wall_seconds']:.1f}s", flush=True)
        result = dict(status="complete", protocol=config["protocol"], job=job, episodes=rows,
                      aggregate=serial.aggregate(rows), representative=representative,
                      execution=dict(mode="parallel", gpu_index=args.gpu_index, threads=args.threads))
        validate_job(result, job, config)
        atomic_json(args.run_root / "jobs" / (args.job + ".json"), result)
        progress(status="complete", completed_episodes=len(rows))
        return 0
    except BaseException as error:
        progress(status="failed", error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--slot", required=True)
    parser.add_argument("--progress-seconds", type=float, default=15.)
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        code = run_job(args)
    except BaseException:
        traceback.print_exc()
        code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()
