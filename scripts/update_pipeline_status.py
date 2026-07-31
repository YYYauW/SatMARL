from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--status", choices=["running", "complete", "failed"], required=True
    )
    parser.add_argument("--stage", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--metrics-url", required=True)
    parser.add_argument(
        "--tensorboard-url", default="http://127.0.0.1:6006"
    )
    args = parser.parse_args()

    path = args.root / "pipeline.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {"stages": []}
    stages = list(payload.get("stages") or [])
    if not stages or stages[-1].get("name") != args.stage:
        stages.append(
            {
                "name": args.stage,
                "status": args.status,
                "updated_at": time.time(),
            }
        )
    else:
        stages[-1].update(
            {"status": args.status, "updated_at": time.time()}
        )
    payload.update(
        {
            "status": args.status,
            "stage": args.stage,
            "message": args.message,
            "updated_at": time.time(),
            "tensorboard_url": args.tensorboard_url,
            "training_runs": [
                {
                    "method": args.method,
                    "seed": args.seed,
                    "metrics": args.metrics_url,
                }
            ],
            "stages": stages,
        }
    )
    atomic_write(path, payload)


if __name__ == "__main__":
    main()
