"""Compare real checkpoint decisions on identical original/accelerated prefixes.

This is a read-only check of source run assets. It does not complete an experiment
or claim full-day CUDA validation. The output report gates v2 neural execution.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
from pathlib import Path
import sys
import time

from transmission_accelerated_common import (
    ROOT, atomic_json, configured_options, execution_hashes, identity_digest,
    job_key, read_json, sha, verify_assets,
)


def exact(left, right, path="value"):
    import numpy as np
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if (not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray)
                or left.dtype != right.dtype or left.shape != right.shape
                or not np.array_equal(left, right, equal_nan=True)):
            raise AssertionError(f"Array mismatch: {path}")
        return
    if isinstance(left, dict):
        if not isinstance(right, dict) or list(left) != list(right):
            raise AssertionError(f"Dictionary keys/order mismatch: {path}")
        for name in left:
            exact(left[name], right[name], f"{path}.{name}")
        return
    if isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise AssertionError(f"Sequence mismatch: {path}")
        for index, (a, b) in enumerate(zip(left, right)):
            exact(a, b, f"{path}[{index}]")
        return
    if left != right:
        raise AssertionError(f"Scalar mismatch: {path}: {left!r} != {right!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--transitions", type=int, default=40)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--satellites", default="128,1024")
    parser.add_argument("--tasks", type=int, default=1000)
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()
    if args.transitions < 20 or args.threads < 1:
        parser.error("Use at least 20 transitions and one CPU thread")
    source, report = args.source_run.expanduser().resolve(), args.report.expanduser().resolve()
    if report.exists():
        raise FileExistsError("Preserve the existing validation report; choose a fresh report filename")
    if report.is_relative_to(source):
        raise ValueError("Write the report outside the active source run")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    import run_transmission_load as serial
    import torch
    import evaluate_marl
    from accelerated_features import build_policy
    from marl_common import resolve_device
    from sat_marl_env.accelerated_transmission import AcceleratedEventTransmissionEnv
    manifest = read_json(source / "manifest.json")
    identity, config = manifest["identity"], manifest["identity"]["config"]
    if identity["policy"] != "checkpoint" or identity["device"] != args.device:
        raise ValueError("Compare using the same real checkpoint policy/device as the source experiment")
    if not 0 <= args.episode < config["episodes"]:
        raise ValueError("Episode outside source plan")
    for name, digest in identity["source_sha256"].items():
        path = (ROOT / name.replace("\\", "/")).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != digest:
            raise ValueError(f"Original physical/policy code changed: {name}")
    verify_assets(source, read_json(source / "scenario_sha256.json"))
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.device == "cuda":
        torch.cuda.set_device(args.gpu)
    device = resolve_device(args.device)
    sizes = {int(value) for value in args.satellites.split(",")}
    jobs = [job for job in identity["jobs"] if job["satellites"] in sizes and job["tasks"] == args.tasks]
    if not jobs or {job["satellites"] for job in jobs} != sizes:
        raise ValueError("Requested validation cases are absent from the source plan")
    output = dict(status="running", identity_sha256=identity_digest(identity),
                  execution_source_sha256=execution_hashes(), checker_sha256=sha(Path(__file__)),
                  device=args.device, gpu=args.gpu, transitions=args.transitions,
                  checked_checkpoint_sha256=[], cases=[], started_at=time.time(),
                  note="Exact real-model prefix comparison, not a full-day neural equivalence proof or a throughput benchmark")
    atomic_json(report, output)
    try:
        for job in jobs:
            raw = Path(job["checkpoint"]).read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != identity["checkpoint_sha256"][job["checkpoint"]]:
                raise ValueError("Checkpoint differs from the source manifest")
            checkpoint = torch.load(io.BytesIO(raw), map_location=device, weights_only=False)
            metrics = dict(checkpoint.get("metrics") or {})
            options = configured_options(config, job, args.episode, metrics, source / "scenario")
            baseline = serial.EventTransmissionEnv(serial.EventTransmissionConfig(**options))
            optimized = AcceleratedEventTransmissionEnv(serial.EventTransmissionConfig(**options),
                                                        cache_dir=report.parent / "acceleration_validation_cache")
            old_obs, old_info = baseline.reset(seed=config["scenario_seed"] + args.episode)
            new_obs, new_info = optimized.reset(seed=config["scenario_seed"] + args.episode)
            exact((old_obs, old_info), (new_obs, new_info), "reset")
            _, old_choose = evaluate_marl.build_policy(checkpoint, metrics, old_obs, device)
            _, new_choose = build_policy(evaluate_marl, checkpoint, metrics, new_obs, device)
            count = 0
            for count in range(1, args.transitions + 1):
                old_action, new_action = old_choose(old_obs), new_choose(new_obs)
                exact(old_action, new_action, f"action[{count}]")
                old_result, new_result = baseline.step(old_action), optimized.step(new_action)
                exact(old_result, new_result, f"step[{count}]")
                old_obs, new_obs = old_result[0], new_result[0]
                if all(old_result[2].values()) or all(old_result[3].values()):
                    break
            exact(baseline.summary(), optimized.summary(), "summary")
            exact(baseline.acquisition_ledger(), optimized.acquisition_ledger(), "acquisitions")
            exact(baseline.task_outcomes(), optimized.task_outcomes(), "task_outcomes")
            exact(baseline.radio_power_ledger(), optimized.radio_power_ledger(), "radio_power")
            output["cases"].append(dict(job=job_key(job), checked_transitions=count,
                                        simulated_seconds=baseline.step_count, checkpoint_sha256=digest))
            output["checked_checkpoint_sha256"] = sorted({*output["checked_checkpoint_sha256"], digest})
            atomic_json(report, output)
            print(f"PASS {job_key(job)}: {count} transitions; sim={baseline.step_count:.3f}s", flush=True)
            del old_choose, new_choose, checkpoint, baseline, optimized
        output.update(status="passed", completed_at=time.time())
        atomic_json(report, output)
        print(f"Passed exact real-model prefix comparison. Report: {report}", flush=True)
    except BaseException as error:
        output.update(status="failed", error=str(error), completed_at=time.time())
        atomic_json(report, output)
        raise


if __name__ == "__main__":
    main()
