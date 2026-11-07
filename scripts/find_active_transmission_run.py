"""Read-only discovery of verified live v1/v2 owners below this project's runs/."""
from __future__ import annotations

import argparse
from pathlib import Path

from transmission_accelerated_common import ROOT, read_json, source_is_active


def find_runs(project, kind="legacy"):
    versions = ({"transmission-parallel-v1"} if kind == "legacy" else
                {"transmission-accelerated-v2"} if kind == "accelerated" else
                {"transmission-parallel-v1", "transmission-accelerated-v2"})
    results = []
    for path in sorted((Path(project) / "runs").glob("*/manifest.json")):
        value = read_json(path)
        if value.get("runner") in versions and source_is_active(path.parent):
            results.append((path.parent.resolve(), value["runner"]))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--kind", choices=["legacy", "accelerated", "all"], default="legacy")
    parser.add_argument("--print-root", action="store_true", help="Print exactly one verified active run or refuse")
    args = parser.parse_args()
    results = find_runs(args.project.resolve(), args.kind)
    if args.print_root:
        if len(results) != 1:
            parser.error(f"Expected one active {args.kind} run, found {len(results)}. Inspect without --print-root.")
        print(results[0][0])
    else:
        for root, version in results:
            print(f"ACTIVE {version}: {root}")
        if not results:
            print("No matching verified active run found. No process or file was changed.")


if __name__ == "__main__":
    main()
