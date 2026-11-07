"""Read-only runtime breakdown for newly computed v2 episodes, excluding imports."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from transmission_accelerated_common import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.expanduser().resolve()
    groups = defaultdict(lambda: dict(count=0, wall_seconds=0., stage_seconds=defaultdict(float),
                                      environment_seconds=defaultdict(float), hits=0, misses=0))
    for path in (root / "performance").glob("*/episode*.json"):
        value = read_json(path)
        key = value["job"]
        item = groups[key]
        item["count"] += 1
        item["wall_seconds"] += value["wall_seconds"]
        for name, seconds in value["stage_seconds"].items():
            item["stage_seconds"][name] += seconds
        environment = value.get("environment", {})
        for name, seconds in environment.get("stage_seconds", {}).items():
            item["environment_seconds"][name] += seconds
        cache = environment.get("orbit_cache", {})
        item["hits"] += cache.get("hits", 0)
        item["misses"] += cache.get("misses", 0)
    print("New accelerated episodes only; imported v1 wall times are excluded.")
    print("Wall seconds below are worker time, not total scheduler elapsed time.")
    print("Environment substages are inclusive and must not be summed together.")
    for key, item in sorted(groups.items()):
        count, wall = item["count"], item["wall_seconds"]
        print(f"\n{key}: {count} new episodes, mean wall={wall / count / 60:.2f} min")
        for name, seconds in item["stage_seconds"].items():
            print(f"  {name}: {seconds / count:.2f} s/episode ({100 * seconds / max(1e-9, wall):.1f}% of worker wall)")
        print("  Inclusive environment substages:")
        for name, seconds in sorted(item["environment_seconds"].items(), key=lambda pair: -pair[1]):
            print(f"    {name}: {seconds / count:.2f} s/episode")
        total = item["hits"] + item["misses"]
        if total:
            print(f"  Exact orbital-query cache hit ratio: {item['hits'] / total:.2%}")
    if not groups:
        print("No new completed v2 episode yet. Running-stage timings are visible in --status.")


if __name__ == "__main__":
    main()
