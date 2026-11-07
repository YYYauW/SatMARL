"""Inspect, or explicitly cancel, one verified frozen serial/v1/v2 evaluation.

Default inspection never sends signals or writes files. Immediate cancellation
retains completed episodes and original files; unsaved in-flight work is lost.
No kill-all, PID-only signal, SIGKILL, source update, or lock deletion is used.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import drain_legacy_transmission as legacy
from transmission_source_compat import accepted_digest

ROOT = Path(__file__).resolve().parents[1]
OWNERS = {
    "serial": "run_transmission_load.py",
    "transmission-parallel-v1": "run_transmission_parallel.py",
    "transmission-accelerated-v2": "run_transmission_accelerated.py",
}
WORKERS = {
    "transmission-parallel-v1": "transmission_parallel_worker.py",
    "transmission-accelerated-v2": "transmission_accelerated_worker.py",
}
FROZEN_SCIENCE = {name: value for name, value in legacy.FROZEN.items()
                  if not name.startswith("configs/") and name not in {
                      "scripts/run_transmission_parallel.py", "scripts/transmission_parallel_worker.py",
                      "scripts/transmission_parallel_common.py", "scripts/transmission_parallel_control.py"}}
FROZEN_SCIENCE.update({
    "src/sat_marl_env/config.py": "e82aadca2c4d58e746ffac7239b81e8a227d15e6ee840ffac76929d2a62020ad",
    "src/sat_marl_env/orbital.py": "9404a9a1dccf1ad0455ae36d7744651d58b2f139e2d07c8dd422f19dfcebd78e",
    "src/sat_marl_env/real_scenario.py": "e27a9326362e8cb84b8b48c385460dde74fcddeef642d56528a4f546e0691a0a",
    "scripts/build_kepler_ephemeris.py": "7c5bef9ca9b6f50e6b0d84438b0723e3921084a9da0d3fdc6f78811e8f17cbfc",
    "scripts/generate_task_catalogs.py": "ffb430166ff8b0921726813af608f568c05df245c6cb0603b4135ab4b816dfc0",
    "scripts/evaluate_marl.py": "a37c07d9151a1730638300cdcabd95aa1aefa64665c583ce7046877c45a7ed65",
    "scripts/marl_common.py": "f291f7d3328d144023aef4eae286044791dabb19a8232d6e9d0dfd160999f09a",
})
FROZEN_V2 = {
    "src/sat_marl_env/accelerated_transmission.py": "05bcbfe773e925555a021329a5345d4108bdeaab8c788377e5d1a73619aee164",
    "scripts/accelerated_features.py": "65b359112ce2ea063df17a407d1e806aee4f765823b6896b576046d48050df5e",
    "scripts/run_transmission_accelerated.py": "b5b7410a8892c22050e10ec1e7546f9a8d6d0653be4f77e26835d726578ff6d4",
    "scripts/transmission_accelerated_worker.py": "a47f5c2a829cf7c3ca9f56b654e204527b7b6bf8eef672ed8c07827149c68ceb",
    "scripts/transmission_accelerated_common.py": "b55d9c2f7bbe945bdfee9f073891b34426aa45b23fb820e62c277ed6c0560d35",
}
FROZEN_CONTROLS = {
    "scripts/transmission_accelerated_control.py": "f02c44d38f52b72152d068096c8bb72fa287f88667eaf59d1b41556bc62741db",
    "scripts/transmission_parallel_control.py": legacy.FROZEN["scripts/transmission_parallel_control.py"],
    "scripts/drain_legacy_transmission.py": "389ec54b025dddb37afe2153ea8ffb1e51f57aa61d972f76c9bdfc1d371f1a72",
}


def verify_manifest(root, project=ROOT):
    """Verify recognized published code, not just hashes chosen by a manifest."""
    root, project = Path(root).resolve(), Path(project).resolve()
    manifest = legacy.read_json(root / "manifest.json")
    version = manifest.get("runner", "serial")
    if version not in OWNERS:
        raise ValueError(f"Unsupported previous runner: {version}")
    identity = manifest["identity"]
    config = identity["config"]
    if (manifest.get("protocol") != "optical-ground-delivery-event-v3"
            or config.get("protocol") != manifest["protocol"]):
        raise ValueError("Previous run has an unsupported or inconsistent protocol")
    expected_science = dict(FROZEN_SCIENCE)
    if identity.get("policy") != "checkpoint":
        for name in ("scripts/evaluate_marl.py", "scripts/marl_common.py"):
            expected_science.pop(name)
    sources = legacy.normalized_hashes(identity["source_sha256"])
    if (set(sources) != set(expected_science)
            or any(not accepted_digest(name, recorded, expected_science[name])
                   for name, recorded in sources.items())):
        raise ValueError("Previous scientific identity differs from the frozen release")
    for name, expected in sources.items():
        target = (project / name).resolve()
        if not target.is_relative_to(project) or legacy.digest_file(target) != expected:
            raise ValueError(f"Previous scientific source changed: {name}")
    if version != "serial":
        expected_execution = (FROZEN_V2 if version == "transmission-accelerated-v2" else
                              {name: legacy.FROZEN[name] for name in legacy.EXECUTION_NAMES})
        if legacy.normalized_hashes(manifest["execution_source_sha256"]) != expected_execution:
            raise ValueError("Previous execution identity differs from the frozen release")
        for name, expected in expected_execution.items():
            if legacy.digest_file(project / name) != expected:
                raise ValueError(f"Previous execution source changed: {name}")
    jobs = {legacy.key_for_job(job): job for job in identity["jobs"]}
    if len(jobs) != len(identity["jobs"]):
        raise ValueError("Duplicate previous job identities")
    return manifest, version, jobs


def all_processes(processes):
    return [value for path in processes.proc.iterdir() if path.name.isdigit()
            if (value := processes.read(int(path.name))) is not None and value.state != "Z"]


@dataclass
class RunState:
    root: Path
    version: str
    owner: object
    workers: list
    manifest: dict
    jobs: dict


def inspect_run(root, project=ROOT, processes=None):
    root, project = Path(root).resolve(), Path(project).resolve()
    manifest, version, jobs = verify_manifest(root, project)
    processes = processes or legacy.LinuxProcesses()
    snapshot = all_processes(processes)
    owners = [process for process in snapshot
              if legacy.matches(process, root, project, OWNERS[version])]
    if len(owners) > 1:
        raise RuntimeError("Multiple exact owners match this run; no cancellation is safe")
    owner = owners[0] if owners else None
    lock = root / "runner.lock"
    if owner is not None and (not lock.is_file() or int(lock.read_text().strip()) != owner.pid):
        raise RuntimeError("Live owner does not match runner.lock; no signal sent")
    workers = []
    for process in snapshot:
        for name in WORKERS.values():
            if legacy.matches(process, root, project, name):
                if name != WORKERS.get(version):
                    raise RuntimeError("A worker of another runner is attached to this root")
                if legacy.option(process.argv, "--job") not in jobs:
                    raise RuntimeError("Worker job is absent from the previous manifest")
                if owner is not None and (process.ppid != owner.pid
                        or int(legacy.option(process.argv, "--parent-pid")) != owner.pid):
                    raise RuntimeError("Worker parent identity differs from the verified owner")
                workers.append(process)
    return RunState(root, version, owner, workers, manifest, jobs)


def report(state, emit=print):
    if state.owner is None:
        if state.workers:
            emit(f"ORPHAN {state.version}: {state.root}; workers={[p.pid for p in state.workers]}", flush=True)
        else:
            emit(f"STOPPED {state.version}: {state.root}; no live matching owner/worker", flush=True)
    else:
        emit(f"ACTIVE {state.version}: {state.root}; owner={state.owner.pid}; workers={[p.pid for p in state.workers]}", flush=True)


def discover(project=ROOT, processes=None):
    processes = processes or legacy.LinuxProcesses()
    snapshot = all_processes(processes)
    results = []
    for path in sorted((Path(project) / "runs").glob("*/manifest.json")):
        value = legacy.read_json(path)
        if value.get("runner", "serial") not in OWNERS:
            continue
        # Only relevant optical-delivery runs are eligible; other experiments stay out of scope.
        if value.get("protocol") != "optical-ground-delivery-event-v3":
            continue
        # Historical non-balanced diagnostic manifests are not this release.
        # They must not prevent discovery when already stopped. Any live exact
        # owner/worker still goes through strict frozen-source verification.
        if not any(legacy.matches(p, path.parent, project, script)
                   for p in snapshot for script in {*OWNERS.values(), *WORKERS.values()}):
            continue
        state = inspect_run(path.parent, project, processes)
        if state.owner is not None or state.workers:
            results.append(state)
    return results


def wait_stopped(state, project, processes, timeout_seconds=120., clock=time.monotonic,
                 sleep=time.sleep, emit=print):
    deadline, last_report = clock() + timeout_seconds, float("-inf")
    while True:
        current = inspect_run(state.root, project, processes)
        if current.owner is not None and (current.owner.pid, current.owner.start_ticks) != (
                state.owner.pid, state.owner.start_ticks):
            raise RuntimeError("A replacement owner appeared; no further signal sent")
        if current.owner is None and not current.workers:
            emit("Previous owner and all matching workers stopped. Completed files and original locks were retained.", flush=True)
            return True
        if clock() >= deadline:
            raise RuntimeError("Previous processes are still alive after the wait; no SIGKILL sent. Do not start the replacement.")
        if clock() - last_report >= 15.:
            emit(f"Waiting for shutdown: owner={None if current.owner is None else current.owner.pid}, workers={[p.pid for p in current.workers]}", flush=True)
            last_report = clock()
        sleep(.25)


def verify_control(project, name):
    if legacy.digest_file(Path(project) / name) != FROZEN_CONTROLS[name]:
        raise ValueError(f"Frozen control tool changed: {name}")


def cancel(state, project=ROOT, processes=None, *, apply=False, after_current_episode=False,
           timeout_seconds=120., clock=time.monotonic, sleep=time.sleep, emit=print):
    project = Path(project).resolve()
    processes = processes or legacy.LinuxProcesses()
    report(state, emit)
    if state.owner is None:
        if state.workers:
            raise RuntimeError("Owner is absent but orphan workers remain; no signal sent and run is not stopped")
        return True
    if not apply:
        emit("Inspection only. Add --apply to cancel this exact run. No process or file was changed.", flush=True)
        return False
    journal = legacy.journal_path(state.root)
    if journal.exists() and legacy.safe_journal_read(journal).get("status") != "done":
        raise RuntimeError("An unfinished legacy drain journal exists; recover its controller before cancellation")
    if state.owner.state in {"T", "t"} or any(worker.state in {"T", "t"} for worker in state.workers):
        raise RuntimeError("Previous processes are paused; recover the existing drain controller before cancellation")
    current = inspect_run(state.root, project, processes)
    def stable_identity(process):
        return None if process is None else (process.pid, process.ppid, process.uid,
                                             process.start_ticks, process.argv, process.cwd)
    if current.manifest != state.manifest or stable_identity(current.owner) != stable_identity(state.owner):
        raise RuntimeError("Previous manifest or owner changed; no signal sent")
    if after_current_episode:
        if state.version == "transmission-parallel-v1":
            verify_control(project, "scripts/drain_legacy_transmission.py")
            result = legacy.main(["--run-root", str(state.root), "--drain", "--apply"])
            if result:
                raise RuntimeError("Legacy episode-boundary drain did not complete")
        else:
            control = ("scripts/transmission_accelerated_control.py" if state.version == "transmission-accelerated-v2"
                       else "scripts/transmission_parallel_control.py")
            verify_control(project, control)
            arguments = (["--drain", "--apply"] if state.version == "transmission-accelerated-v2"
                         else ["--after-current-job", "--stop", "--apply"])
            if state.version == "serial":
                emit("Serial output is saved per whole job; waiting for its next complete job, not only one episode.", flush=True)
            subprocess.run([sys.executable, str(project / control), "--run-root", str(state.root)] + arguments,
                           cwd=project, check=True)
        # A boundary drain can take hours; never turn its wait into forced termination.
        return wait_stopped(state, project, processes, timeout_seconds=float("inf"), clock=clock, sleep=sleep, emit=emit)
    emit("Cancelling this exact previous run now. Completed episodes remain; unsaved in-flight work is discarded. Serial runs save only whole jobs.", flush=True)
    processes.signal(state.owner, signal.SIGTERM)
    return wait_stopped(state, project, processes, timeout_seconds=timeout_seconds, clock=clock, sleep=sleep, emit=emit)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--print-root", action="store_true", help="Print one unique verified active run root or refuse")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--after-current-episode", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=120.)
    args = parser.parse_args(argv)
    if not sys.platform.startswith("linux"):
        parser.error("This exact-process cancellation tool is Linux-only")
    if not 0 < args.timeout_seconds <= 120:
        parser.error("Immediate shutdown wait must be greater than zero and at most 120 seconds")
    if args.inspect and args.apply:
        parser.error("--inspect and --apply cannot be combined")
    if args.discover or args.print_root:
        if args.run_root or args.apply or args.after_current_episode:
            parser.error("Discovery is read-only; do not combine it with a run root or cancellation")
        states = discover(args.project.resolve())
        if args.print_root:
            if len(states) != 1 or states[0].owner is None:
                parser.error(f"Expected one active exact owner and no orphan-only run, found {len(states)} candidate runs; inspect --discover")
            print(states[0].root)
        else:
            for state in states:
                report(state)
            if not states:
                print("No verified active previous run. No process or file was changed.")
        return 0
    if args.run_root is None:
        parser.error("Select --run-root or --discover")
    state = inspect_run(args.run_root.expanduser(), args.project.resolve())
    arguments = dict(apply=args.apply, after_current_episode=args.after_current_episode,
                     timeout_seconds=args.timeout_seconds)
    if args.apply and state.owner is not None and not args.after_current_episode:
        # Share the old drainer's advisory lock so two operators cannot STOP/TERM
        # the same legacy tree concurrently. Boundary delegation owns its lock.
        with legacy.ControlMutex(state.root):
            cancel(state, args.project, **arguments)
    else:
        cancel(state, args.project, **arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
