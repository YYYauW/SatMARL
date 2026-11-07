"""Inspect or signal only the verified owner process of one specified run."""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import signal
import time

from transmission_parallel_common import process_matches_run, read_json


def completed_count(root):
    return sum(read_json(path).get("status") == "complete" for path in (root / "jobs").glob("*.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--after-current-job", action="store_true", help="For the old serial runner, wait for one more complete job")
    args = parser.parse_args()
    root = args.run_root.expanduser().resolve()
    lock = root / "runner.lock"
    if not lock.exists():
        print("No owner lock; this run is already stopped or finished.")
        return
    pid = int(lock.read_text().strip())
    names = {"run_transmission_load.py", "run_transmission_parallel.py"}
    if not process_matches_run(pid, root, names):
        print(f"No matching live owner for PID {pid}; no signal sent, no lock deleted.")
        return
    baseline = completed_count(root)
    print(f"Verified owner PID={pid}, run={root}, complete_jobs={baseline}", flush=True)
    if not args.stop or not args.apply:
        print("Inspection only. Add --stop --apply to request termination of this exact process.")
        return
    if args.after_current_job:
        if not process_matches_run(pid, root, {"run_transmission_load.py"}):
            raise RuntimeError("--after-current-job applies only to the old serial runner")
        print("Waiting for the next complete serial job so its twenty episodes are saved...", flush=True)
        last_report = 0.
        while process_matches_run(pid, root, names):
            current = completed_count(root)
            if current > baseline:
                break
            if time.monotonic() - last_report >= 30:
                print(f"Still waiting: PID={pid}, complete_jobs={current}", flush=True)
                last_report = time.monotonic()
            time.sleep(.5)
    if process_matches_run(pid, root, names):
        print("Sending SIGTERM to the verified run owner. Old serial unsaved work may need rerunning; completed files remain.", flush=True)
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 15
    while process_matches_run(pid, root, names) and time.monotonic() < deadline:
        time.sleep(.2)
    if process_matches_run(pid, root, names):
        raise RuntimeError("Owner is still shutting down. Wait and inspect before launching the replacement.")
    print("Owner stopped; original files and any old lock were retained.")


if __name__ == "__main__":
    main()
