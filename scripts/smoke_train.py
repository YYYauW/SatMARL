from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(root / "scripts" / "train_dqn.py"),
        "--satellites",
        "16",
        "--tasks",
        "64",
        "--max-steps",
        "32",
        "--episodes",
        "2",
        "--learning-starts",
        "32",
        "--batch-size",
        "32",
        "--store-agents",
        "16",
        "--run-dir",
        str(root / "runs" / "smoke"),
    ]
    raise SystemExit(subprocess.call(cmd, cwd=root))


if __name__ == "__main__":
    main()
