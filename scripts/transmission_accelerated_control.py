"""Live concurrency changes or lossless episode-boundary drain for v2 only."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

from transmission_accelerated_common import (
    PARALLEL_VERSION, atomic_json, execution_hashes, process_matches_run,
    read_control, read_json,
)


def make_command(root, pid, previous, *, drain=False, workers=None, cpu_workers=None, max_large=None):
    result = dict(previous, run_root=str(Path(root).resolve()), owner_pid=pid,
                  action="drain" if drain else "run", command_id=f"operator-{time.time_ns()}",
                  requested_at=time.time())
    for key, value in (("workers_per_gpu", workers), ("cpu_workers", cpu_workers), ("max_large_per_gpu", max_large)):
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Concurrency limits must be positive integers")
            result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--drain", action="store_true")
    parser.add_argument("--resume-dispatch", action="store_true")
    parser.add_argument("--workers-per-gpu", type=int)
    parser.add_argument("--cpu-workers", type=int)
    parser.add_argument("--max-large-per-gpu", type=int)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = args.run_root.expanduser().resolve()
    manifest = read_json(root / "manifest.json")
    if manifest.get("runner") != PARALLEL_VERSION:
        raise ValueError("This control tool applies only to the accelerated v2 runner; it never signals v1")
    if args.status:
        from run_transmission_accelerated import render_progress
        render_progress(read_json(root / "progress.json"))
        ack = root / "execution_control_ack.json"
        if ack.exists():
            print("Last control acknowledgement:", read_json(ack))
        return
    if manifest.get("execution_source_sha256") != execution_hashes():
        raise ValueError("Execution files changed; no control message written")
    if args.drain and (args.resume_dispatch or args.workers_per_gpu or args.cpu_workers or args.max_large_per_gpu):
        parser.error("Use a separate drain command without concurrency changes")
    if not (args.drain or args.resume_dispatch or args.workers_per_gpu or args.cpu_workers or args.max_large_per_gpu):
        parser.error("Select --status, --drain or a concurrency change")
    lock = root / "runner.lock"
    if not lock.exists():
        raise ValueError("No live owner lock. Resume the v2 run with its original command before controlling it")
    pid = int(lock.read_text().strip())
    if not process_matches_run(pid, root, {"run_transmission_accelerated.py"}):
        raise ValueError("Cannot verify the live owner; no file written and no signal sent")
    state = make_command(root, pid, read_control(root, pid), drain=args.drain,
                         workers=args.workers_per_gpu, cpu_workers=args.cpu_workers,
                         max_large=args.max_large_per_gpu)
    print("Requested execution control:", state)
    if not args.apply:
        print("Inspection only. Add --apply to submit. Current workers are not stopped.")
        return
    # Recheck immediately before committing; no PID-only signalling is used.
    if int(lock.read_text().strip()) != pid or not process_matches_run(pid, root, {"run_transmission_accelerated.py"}):
        raise ValueError("Owner changed; request not submitted")
    atomic_json(root / "execution_control.json", state)
    print("Submitted. Current episodes continue. See --status and the scheduler log for acknowledgement.")


if __name__ == "__main__":
    main()
