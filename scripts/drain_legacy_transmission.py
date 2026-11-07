"""Drain the frozen v1 evaluator at saved-episode boundaries without hot updates.

Inspection is the default. Linux-only --drain --apply freezes the verified parent,
allows each current episode to finish, then stops all processes at saved boundaries.
Before the commit point, interruption/timeout resumes every process paused by this
tool. No legacy source, result, manifest, or lock file is modified. A durable
sibling journal and advisory control lock support explicit crash recovery.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import stat
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
VERSION = "transmission-parallel-v1"
FROZEN = {
    "src/sat_marl_env/env.py": "582434fd49fae83b1a73c7e4b3cadc61a3e9ce83c9d60c5049e38b7f15236b38",
    "src/sat_marl_env/entities.py": "655f13989636cf3ba839d0eb4f1923e6c1c0ccca4a0417f1f0d25b0185bda04d",
    "src/sat_marl_env/transmission.py": "772ca744d8860c08f54fbeea846cef34a528a3effee6eb0f63e975463f8b7bf8",
    "src/sat_marl_env/event_transmission.py": "a4ed827ed14813fd37526e079efa1a3c888b3217a08ccef10bd820f8d071a6ff",
    "configs/transmission_day_v3.json": "889f9aecd3b902c711c4b5ec0e1d134ca4d306be50c92d6e03d7899cd1ee78c9",
    "scripts/run_transmission_load.py": "248c03c92376e53829905fe4fd747e2a1c59bbfc822cd33a327cb4a2f930107d",
    "scripts/balanced_task_catalog.py": "1fe7b8053f8e3e0a8185bdbaa0052a56feccbe8a896d80aab046739812579aa9",
    "configs/transmission_day_v3_full_sweep.json": "d89c7125891a52ef6b5ce08ff0d131503846804a2e40703e641b12f22693314a",
    "scripts/run_transmission_parallel.py": "3ed6f483a0292d07c06f756c3a29e0a920245de7473ef545ecae56d2e8ade185",
    "scripts/transmission_parallel_worker.py": "7428e1100f9d2e9cebc7e7c456df4b56db90ed3c4a44d02a68b3657588b96fcb",
    "scripts/transmission_parallel_common.py": "a035d93af1ea7ce0f3f5e01703be7e703487fbb35947ac1b27a2c6c6906c60ef",
    "scripts/transmission_parallel_control.py": "249ca9987a2d5fef7dcdc88e8f70ebb667c3b583b4347c258846f56814d5a4b7",
}
EXECUTION_NAMES = {
    "scripts/run_transmission_parallel.py", "scripts/transmission_parallel_worker.py",
    "scripts/transmission_parallel_common.py",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_hashes(value):
    return {key.replace("\\", "/"): digest for key, digest in value.items()}


def identity_digest(identity):
    identity = json.loads(json.dumps(identity))
    identity["source_sha256"] = normalized_hashes(identity["source_sha256"])
    return hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def key_for_job(job):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", job["method"]):
        raise ValueError("Unsafe job method")
    return f"{job['method']}_seed{job['training_seed']}_n{job['satellites']}_m{job['tasks']}"


def verify_release(root, project=ROOT):
    """Read-only verification against the published 20261018 v1 release."""
    manifest = read_json(root / "manifest.json")
    if manifest.get("runner") != VERSION:
        raise ValueError("Only the frozen transmission-parallel-v1 release can use this drain tool")
    execution = normalized_hashes(manifest["execution_source_sha256"])
    if execution != {name: FROZEN[name] for name in EXECUTION_NAMES}:
        raise ValueError("Legacy execution hashes differ from the frozen v1 release")
    for name, expected in FROZEN.items():
        if digest_file(project / name) != expected:
            raise ValueError(f"Frozen v1 source changed: {name}; do not hot-update a running v1 evaluator")
    identity = manifest["identity"]
    for name, expected in normalized_hashes(identity["source_sha256"]).items():
        target = (project / name).resolve()
        if not target.is_relative_to(project.resolve()) or digest_file(target) != expected:
            raise ValueError(f"Scientific source mismatch: {name}")
        if name in FROZEN and expected != FROZEN[name]:
            raise ValueError(f"Scientific identity is not the frozen release: {name}")
    if manifest.get("protocol") != identity["config"]["protocol"]:
        raise ValueError("Manifest protocol mismatch")
    jobs = {key_for_job(job): job for job in identity["jobs"]}
    if len(jobs) != len(identity["jobs"]):
        raise ValueError("Duplicate job identities")
    return manifest, jobs


def validate_saved(saved, job, episode, manifest):
    """The frozen v1 row validator, copied to avoid importing mutable old code."""
    if saved["identity_sha256"] != identity_digest(manifest["identity"]):
        raise ValueError("Saved episode identity hash mismatch")
    if normalized_hashes(saved["execution_source_sha256"]) != normalized_hashes(manifest["execution_source_sha256"]):
        raise ValueError("Saved episode execution hash mismatch")
    row, config = saved["row"], manifest["identity"]["config"]
    for field in ("method", "training_seed", "satellites", "tasks"):
        if row[field] != job[field]:
            raise ValueError(f"Saved episode job mismatch: {field}")
    if row["episode"] != episode or row["scenario_seed"] != config["scenario_seed"] + episode:
        raise ValueError("Saved episode/scenario index mismatch")
    for field, value in row.items():
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise ValueError(f"Non-finite row metric: {field}")
    observed, delivered, total = (row[field] for field in (
        "observation_success_tasks", "observation_and_downlink_success_tasks", "tasks"))
    if not 0 <= delivered <= observed <= total or total <= 0:
        raise ValueError("Invalid success-task inclusion")
    for field, count in (("observation_success_rate", observed),
                         ("observation_and_downlink_success_rate", delivered)):
        if not math.isclose(row[field], count / total, abs_tol=1e-10, rel_tol=0):
            raise ValueError("Invalid success-rate denominator")
    horizon = config["env"]["max_steps"] * config["env"]["step_duration_seconds"]
    if not math.isclose(row["simulated_seconds"], horizon, abs_tol=1e-6, rel_tol=0):
        raise ValueError("Saved episode is not a complete physical horizon")
    if row["max_rolling_imaging_seconds"] > config["env"]["imaging_budget_seconds"] + 1e-6:
        raise ValueError("Saved imaging budget violation")
    residual = row["committed_data_mb"] - row["downlinked_data_mb"] - row["pending_data_mb"]
    if abs(residual) > 1e-5 * max(1., row["committed_data_mb"]):
        raise ValueError("Saved data conservation violation")


@dataclass(frozen=True)
class Process:
    pid: int
    ppid: int
    uid: int
    start_ticks: int
    state: str
    argv: tuple
    cwd: Path


def option(argv, name):
    values = []
    for index, argument in enumerate(argv):
        if argument == name and index + 1 < len(argv):
            values.append(argv[index + 1])
        elif argument.startswith(name + "="):
            values.append(argument.split("=", 1)[1])
    if len(values) != 1:
        raise ValueError(f"Expected exactly one {name}")
    return values[0]


class LinuxProcesses:
    """Use Linux process identity plus pidfds, never a naked PID-only signal."""
    def __init__(self, proc=Path("/proc")):
        self.proc = Path(proc)

    def read(self, pid):
        try:
            path = self.proc / str(pid)
            text = (path / "stat").read_text()
            fields = text[text.rfind(")") + 2:].split()
            argv = tuple(part for part in (path / "cmdline").read_bytes().decode().split("\0") if part)
            return Process(pid, int(fields[1]), path.stat().st_uid, int(fields[19]), fields[0], argv,
                           (path / "cwd").resolve(strict=True))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            return None

    def children(self, pid):
        result = []
        for path in self.proc.iterdir():
            if path.name.isdigit():
                process = self.read(int(path.name))
                if process is not None and process.ppid == pid:
                    result.append(process)
        return result

    def signal(self, expected, signum, reparented_from=None):
        if reparented_from is not None and signum != signal.SIGCONT:
            raise RuntimeError("Reparenting is permitted only for recovery SIGCONT")
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            raise RuntimeError("Safe signalling requires Linux Python with pidfd_open and pidfd_send_signal")
        try:
            descriptor = os.pidfd_open(expected.pid)
        except ProcessLookupError:
            return False
        try:
            current = self.read(expected.pid)
            if current is None:
                return False
            if (current.start_ticks, current.uid, current.argv, current.cwd) != (
                    expected.start_ticks, expected.uid, expected.argv, expected.cwd):
                raise RuntimeError(f"Process identity changed; refusing to signal PID {expected.pid}")
            if current.ppid != expected.ppid and not safe_reparented_recovery(
                    expected, reparented_from, self):
                raise RuntimeError(f"Parent identity changed; refusing to signal PID {expected.pid}")
            try:
                signal.pidfd_send_signal(descriptor, signum)
            except ProcessLookupError:
                return False
            return True
        finally:
            os.close(descriptor)


def original_parent_is_gone(parent, processes):
    if parent is None:
        return False
    current = processes.read(parent.pid)
    return current is None or current.state == "Z" or current.start_ticks != parent.start_ticks


def safe_reparented_recovery(worker, parent, processes):
    """A dead parent's original worker may need CONT to observe its orphan flag."""
    if parent is None or worker.ppid != parent.pid or not original_parent_is_gone(parent, processes):
        return False
    try:
        return (int(option(worker.argv, "--parent-pid")) == parent.pid
                and any(Path(argument).name == "transmission_parallel_worker.py" for argument in worker.argv))
    except ValueError:
        return False


def matches(process, root, project, script, parent_pid=None, jobs=None):
    if process is None or process.pid <= 1 or process.uid != os.getuid():
        return False
    try:
        candidates = []
        for argument in process.argv[1:]:
            if argument.endswith(".py"):
                path = Path(argument)
                candidates.append((path if path.is_absolute() else process.cwd / path).resolve())
        if candidates != [(project / "scripts" / script).resolve()]:
            return False
        value = Path(option(process.argv, "--run-root"))
        if (value if value.is_absolute() else process.cwd / value).resolve() != root.resolve():
            return False
        if parent_pid is not None:
            if process.ppid != parent_pid or int(option(process.argv, "--parent-pid")) != parent_pid:
                return False
            if option(process.argv, "--job") not in jobs:
                return False
        return True
    except (ValueError, OSError):
        return False


def boot_identity():
    """PID/start-tick identities must never survive a reboot."""
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def journal_path(root):
    root = Path(root).resolve()
    return root.with_name(root.name + ".drain-state.json")


def atomic_journal(path, value):
    """Durable atomic sibling write, outside the legacy result directory."""
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def safe_journal_read(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError("Drain journal is not a regular file")
    if os.name == "posix" and (info.st_uid != os.getuid() or info.st_mode & 0o022):
        raise RuntimeError("Drain journal must be owned by this user and not group/world-writable")
    return read_json(path)


class ControlMutex:
    """One active drain/recovery controller per run, automatically released on kill."""
    def __init__(self, root):
        self.path = journal_path(root).with_suffix(".control.lock")
        self.descriptor = None

    def __enter__(self):
        import fcntl
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        self.descriptor = os.open(self.path, flags, 0o600)
        info = os.fstat(self.descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
            os.close(self.descriptor)
            self.descriptor = None
            raise RuntimeError("Unsafe drain control lock ownership or permissions")
        try:
            fcntl.flock(self.descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            os.close(self.descriptor)
            self.descriptor = None
            raise RuntimeError("Another drain/recovery controller already owns this run") from error
        return self

    def __exit__(self, kind, value, traceback):
        if self.descriptor is not None:
            os.close(self.descriptor)
            self.descriptor = None


def manifest_fingerprint(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class DrainJournal:
    def __init__(self, root, manifest, project=ROOT):
        self.path = journal_path(root)
        if self.path.exists() and safe_journal_read(self.path).get("status") != "done":
            raise RuntimeError("An unfinished drain journal exists; inspect --recover-paused before another drain")
        self.data = dict(format="legacy-v1-drain-journal-v1", run_root=str(Path(root).resolve()),
                         project=str(Path(project).resolve()), manifest_sha256=manifest_fingerprint(manifest),
                         uid=os.getuid(), hostname=socket.gethostname(), boot_id=boot_identity(),
                         controller_pid=os.getpid(), status="draining", committed=False, records={})
        self.write()

    def write(self):
        self.data["updated_at"] = time.time()
        atomic_journal(self.path, self.data)

    def record(self, process, role, phase):
        self.data["records"][str(process.pid)] = dict(
            pid=process.pid, ppid=process.ppid, uid=process.uid, start_ticks=process.start_ticks,
            argv=list(process.argv), cwd=str(process.cwd), role=role, phase=phase,
            prior_state=process.state)
        self.write()

    def phase(self, process, phase):
        self.data["records"][str(process.pid)]["phase"] = phase
        self.write()

    def finish(self, committed, failures):
        self.data.update(status="recovery_failed" if failures else "done", committed=committed,
                         outcome="termination_requested" if committed else "resumed", failures=failures)
        self.write()


def recover_journal(root, manifest, jobs, processes, project=ROOT, apply=False, emit=print):
    """Only CONT exact stopped identities recorded by this controller; never TERM."""
    path = journal_path(root)
    data = safe_journal_read(path)
    expected = dict(format="legacy-v1-drain-journal-v1", run_root=str(Path(root).resolve()),
                    project=str(Path(project).resolve()), manifest_sha256=manifest_fingerprint(manifest),
                    uid=os.getuid(), hostname=socket.gethostname(), boot_id=boot_identity())
    for field, value in expected.items():
        if data.get(field) != value:
            raise RuntimeError(f"Drain recovery journal mismatch: {field}; no signal sent")
    if data.get("status") == "done":
        emit("Drain journal is already done; no signal sent.", flush=True)
        return []
    if data.get("status") == "committing":
        emit("A termination request may already be queued; recovery only CONTinues recorded processes and may finish shutdown.", flush=True)
    parent_records = [record for record in data["records"].values() if record["role"] == "parent"]
    if len(parent_records) > 1:
        raise RuntimeError("Multiple parent identities in recovery journal")
    original_parent = None
    if parent_records:
        record = parent_records[0]
        original_parent = Process(record["pid"], record["ppid"], record["uid"], record["start_ticks"],
                                  record["prior_state"], tuple(record["argv"]), Path(record["cwd"]))
        if not matches(original_parent, root, project, "run_transmission_parallel.py"):
            raise RuntimeError("Journal parent does not belong to this run")
    pending, expired = [], []
    for record in data["records"].values():
        if record["phase"] not in {"stop_requested", "paused"}:
            continue
        process = Process(record["pid"], record["ppid"], record["uid"], record["start_ticks"],
                          record["prior_state"], tuple(record["argv"]), Path(record["cwd"]))
        role = record["role"]
        if role not in {"parent", "worker"} or record["prior_state"] in {"T", "t"}:
            raise RuntimeError("Journal cannot claim an unrelated or previously paused process")
        script = "run_transmission_parallel.py" if role == "parent" else "transmission_parallel_worker.py"
        if not matches(process, root, project, script,
                       process.ppid if role == "worker" else None, jobs):
            raise RuntimeError(f"Journal process is not this run's {role}; no signal sent")
        current = processes.read(process.pid)
        if current is None or current.state == "Z":
            expired.append(record)
            continue
        if role == "parent" and current.start_ticks != process.start_ticks:
            # The original parent is dead. Never signal the unrelated new owner
            # of its PID, but still allow its recorded orphan workers to recover.
            expired.append(record)
            continue
        if (current.start_ticks, current.uid, current.argv, current.cwd) != (
                process.start_ticks, process.uid, process.argv, process.cwd):
            raise RuntimeError(f"PID {process.pid} identity changed; no recovery signal sent")
        recovery_parent = None
        if current.ppid != process.ppid:
            if role != "worker" or not safe_reparented_recovery(process, original_parent, processes):
                raise RuntimeError(f"PID {process.pid} parent identity changed; no recovery signal sent")
            recovery_parent = original_parent
        pending.append((process, current, record, recovery_parent))
    # Validate every record before making the first state change, then restore
    # workers first and the supervisor last. pidfds recheck identity at syscall.
    pending.sort(key=lambda item: item[2]["role"] == "parent")
    restored = []
    for process, current, record, recovery_parent in pending:
        if current.state in {"T", "t"}:
            emit(f"{'Resuming' if apply else 'Would resume'} recorded {record['role']} PID={process.pid}", flush=True)
            if apply:
                if recovery_parent is not None:
                    processes.signal(process, signal.SIGCONT, reparented_from=recovery_parent)
                else:
                    processes.signal(process, signal.SIGCONT)
                restored.append(process.pid)
        if apply:
            record["phase"] = "resumed"
            atomic_journal(path, data)
    if apply:
        for record in expired:
            record["phase"] = "exited"
        data.update(status="done", outcome="explicit_recovery", updated_at=time.time())
        atomic_journal(path, data)
    return restored


class Drain:
    def __init__(self, root, manifest, jobs, processes, project=ROOT, report_seconds=15., poll_seconds=.2,
                 timeout_seconds=None, emit=print, clock=time.monotonic, sleep=time.sleep):
        self.root, self.project = Path(root), Path(project)
        self.manifest, self.jobs, self.processes = manifest, jobs, processes
        self.report_seconds, self.poll_seconds, self.timeout_seconds = report_seconds, poll_seconds, timeout_seconds
        self.emit, self.clock, self.sleep = emit, clock, sleep
        self.paused = {}
        self.parent = None
        self.committed = False
        self.journal = None
        # main supplies a fresh release check immediately before the commit.
        # The injectable no-op keeps signal orchestration unit-testable in isolation.
        self.precommit = lambda: None

    def owner(self):
        pid = int((self.root / "runner.lock").read_text().strip())
        process = self.processes.read(pid)
        if not matches(process, self.root, self.project, "run_transmission_parallel.py"):
            raise RuntimeError("No matching same-user legacy parent; no signal sent")
        if process.state in {"T", "t"}:
            raise RuntimeError("Parent was already paused by another controller; refusing ownership")
        return process

    def workers(self, parent):
        children = self.processes.children(parent.pid)
        for process in children:
            if not matches(process, self.root, self.project, "transmission_parallel_worker.py", parent.pid, self.jobs):
                raise RuntimeError(f"Unverified direct child PID {process.pid}; refusing to drain")
            if process.state in {"T", "t"}:
                raise RuntimeError(f"Child PID {process.pid} was already paused by another controller")
        return children

    def complete_prefix(self, key):
        count = 0
        for episode in range(self.manifest["identity"]["config"]["episodes"]):
            path = self.root / "episode_results" / key / f"episode{episode:03d}.json"
            if not path.exists():
                break
            validate_saved(read_json(path), self.jobs[key], episode, self.manifest)
            count += 1
        return count

    def same_live(self, expected, recover_reparented=False):
        current = self.processes.read(expected.pid)
        if current is None or current.state == "Z":
            return None
        if (current.start_ticks, current.uid, current.argv, current.cwd) != (
                expected.start_ticks, expected.uid, expected.argv, expected.cwd):
            raise RuntimeError(f"Process identity changed for PID {expected.pid}; no signal sent")
        if current.ppid != expected.ppid and not (
                recover_reparented and safe_reparented_recovery(expected, self.parent, self.processes)):
            raise RuntimeError(f"Parent identity changed for PID {expected.pid}; no signal sent")
        return current

    def pause(self, process):
        current = self.same_live(process)
        if current is None:
            return False
        if current.state in {"T", "t"}:
            raise RuntimeError(f"PID {process.pid} was paused by another controller; refusing ownership")
        if self.journal is not None:
            role = "parent" if process.pid == self.parent.pid else "worker"
            self.journal.record(process, role, "stop_requested")
        # Register before the signal syscall: a terminal interrupt can arrive
        # immediately after SIGSTOP is delivered and before Python returns.
        self.paused[process.pid] = process
        if not self.processes.signal(process, signal.SIGSTOP):
            self.paused.pop(process.pid, None)
            return False
        deadline = self.clock() + 10.
        while True:
            current = self.same_live(process)
            if current is None:
                self.paused.pop(process.pid, None)
                if self.journal is not None:
                    self.journal.phase(process, "exited")
                return False
            if current.state in {"T", "t"}:
                if self.journal is not None:
                    self.journal.phase(process, "paused")
                return True
            if self.clock() >= deadline:
                raise RuntimeError(f"SIGSTOP not acknowledged for PID {process.pid}")
            self.sleep(.05)

    def recover(self):
        # Resume children first, parent last. No TERM before the all-ready commit.
        failures = []
        order = [pid for pid in self.paused if self.parent is None or pid != self.parent.pid]
        if self.parent is not None and self.parent.pid in self.paused:
            order.append(self.parent.pid)
        for pid in order:
            process = self.paused[pid]
            try:
                if self.parent is not None and process.pid == self.parent.pid and original_parent_is_gone(self.parent, self.processes):
                    current = None
                else:
                    current = self.same_live(process, recover_reparented=True)
                if current is not None:
                    if current.ppid != process.ppid:
                        self.processes.signal(process, signal.SIGCONT, reparented_from=self.parent)
                    else:
                        self.processes.signal(process, signal.SIGCONT)
                    if self.journal is not None:
                        self.journal.phase(process, "resumed")
                elif self.journal is not None:
                    self.journal.phase(process, "exited")
            except BaseException as error:
                failures.append(f"PID {pid}: {error}")
        self.paused.clear()
        if self.journal is not None:
            self.journal.finish(self.committed, failures)
        if failures:
            raise RuntimeError("Could not recover every verified process; inspect immediately: " + "; ".join(failures))

    def inspect(self):
        parent = self.owner()
        workers = self.workers(parent)
        self.emit(f"Verified legacy v1 parent PID={parent.pid}, run={self.root}", flush=True)
        for process in workers:
            key = option(process.argv, "--job")
            count = self.complete_prefix(key)
            self.emit(f"  PID={process.pid} {key}: complete_saved_episodes={count}", flush=True)
        return parent, workers

    def report(self, waiting):
        self.emit("Draining v1: parent dispatch is paused; its progress.json is intentionally not updating.", flush=True)
        for process, key, baseline in waiting:
            path = self.root / "workers" / (key + ".json")
            state = read_json(path) if path.exists() else {}
            detail = (f"ep={state.get('episode', '?')} sim={state.get('simulated_seconds', 0) / 3600:.2f}h "
                      f"phase={state.get('status', '?')}") if state.get("pid") == process.pid else "state unavailable"
            self.emit(f"  PID={process.pid} {key}: waiting for saved episode {baseline + 1}; {detail}", flush=True)

    def run(self):
        self.parent, _ = self.inspect()
        started = self.clock()
        last_report = -float("inf")
        try:
            if not self.pause(self.parent):
                raise RuntimeError("Parent exited before dispatch could be paused; inspect before retrying")
            # Rescan after STOP acknowledgement to include a child launched during
            # the inspection/stop interval. No further jobs can now be dispatched.
            workers = self.workers(self.parent)
            waiting = [(process, option(process.argv, "--job"),
                        self.complete_prefix(option(process.argv, "--job"))) for process in workers]
            total_episodes = self.manifest["identity"]["config"]["episodes"]
            while waiting:
                if self.same_live(self.parent) is None:
                    raise RuntimeError("Parent disappeared during drain")
                remaining = []
                for process, key, baseline in waiting:
                    current = self.same_live(process)
                    count = self.complete_prefix(key)
                    if current is None:
                        if count < total_episodes:
                            raise RuntimeError(f"Worker PID={process.pid} exited before its next saved boundary")
                        self.emit(f"  Finished naturally: PID={process.pid} {key}", flush=True)
                    elif count > baseline or count == total_episodes:
                        self.pause(process)
                        self.emit(f"  Boundary saved and worker paused: PID={process.pid} {key}, saved={count}", flush=True)
                    else:
                        remaining.append((process, key, baseline))
                waiting = remaining
                if not waiting:
                    break
                if self.timeout_seconds is not None and self.clock() - started >= self.timeout_seconds:
                    self.emit("Drain timeout before commit: resuming all paused processes, no termination requested.", flush=True)
                    return False
                if self.clock() - last_report >= self.report_seconds:
                    self.report(waiting)
                    last_report = self.clock()
                self.sleep(self.poll_seconds)
            # Every current worker has reached a validated saved boundary. Queue
            # the parent's STOP flag before resuming anything, preventing dispatch
            # or treating intentional worker exits as normal scheduler failures.
            self.precommit()
            if self.journal is not None:
                self.journal.data["status"] = "committing"
                self.journal.write()
            if not self.processes.signal(self.parent, signal.SIGTERM):
                raise RuntimeError("Parent exited at commit")
            self.committed = True
            for process in list(self.paused.values()):
                if process.pid != self.parent.pid and self.same_live(process) is not None:
                    self.processes.signal(process, signal.SIGTERM)
            self.emit("All boundaries saved. Termination requested; resuming paused processes for graceful shutdown.", flush=True)
        finally:
            self.recover()
        deadline = self.clock() + 90.
        while self.clock() < deadline:
            if self.same_live(self.parent) is None:
                remaining = [process.pid for process in workers
                             if (current := self.processes.read(process.pid)) is not None
                             and current.start_ticks == process.start_ticks and current.state != "Z"]
                if remaining:
                    raise RuntimeError(f"Parent stopped but old workers remain {remaining}; do not start the replacement")
                self.emit("Legacy parent stopped. Saved episodes and original lock/source files were retained.", flush=True)
                return True
            self.sleep(.2)
        raise RuntimeError("Parent is still shutting down. Do not start the replacement until all old workers exit.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--inspect", action="store_true", help="Read-only, also the default")
    parser.add_argument("--drain", action="store_true", help="Wait for all current episodes to be saved, then stop v1")
    parser.add_argument("--recover-paused", action="store_true", help="Recover only verified STOPped identities in the sibling drain journal")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--timeout-hours", type=float, default=72.)
    parser.add_argument("--progress-seconds", type=float, default=15.)
    args = parser.parse_args(argv)
    if not sys.platform.startswith("linux"):
        parser.error("This process inspection/drain tool is Linux-only")
    if args.timeout_hours <= 0 or args.progress_seconds <= 0:
        parser.error("Timeout and progress interval must be positive")
    if args.inspect and (args.drain or args.apply or args.recover_paused):
        parser.error("--inspect cannot be combined with --drain, --recover-paused or --apply")
    if args.drain and args.recover_paused:
        parser.error("Choose --drain or --recover-paused")
    if args.apply and not (args.drain or args.recover_paused):
        parser.error("--apply requires --drain or --recover-paused")
    root = args.run_root.expanduser().resolve()
    manifest, jobs = verify_release(root)
    if args.recover_paused:
        if args.apply:
            with ControlMutex(root):
                recover_journal(root, manifest, jobs, LinuxProcesses(), apply=True)
        else:
            recover_journal(root, manifest, jobs, LinuxProcesses())
            print("Recovery inspection only. Add --apply to resume these recorded paused processes.")
        return 0
    controller = Drain(root, manifest, jobs, LinuxProcesses(), timeout_seconds=args.timeout_hours * 3600,
                       report_seconds=args.progress_seconds)
    def precommit():
        current, _ = verify_release(root)
        if current != manifest or int((root / "runner.lock").read_text().strip()) != controller.parent.pid:
            raise RuntimeError("Manifest or owner lock changed while waiting; resuming without terminating")
    controller.precommit = precommit
    if not args.drain or not args.apply:
        controller.inspect()
        print("Inspection only. Use --drain --apply to wait at saved-episode boundaries and stop this exact v1 run.")
        return 0
    # A terminal SIGTERM should recover any STOPped processes just like Ctrl+C.
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Drain interrupted by signal {signum}")
    old_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        with ControlMutex(root):
            controller.journal = DrainJournal(root, manifest)
            return 0 if controller.run() else 2
    finally:
        signal.signal(signal.SIGTERM, old_handler)


if __name__ == "__main__":
    raise SystemExit(main())
