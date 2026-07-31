from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sat_marl_env.orbital import geodetic_to_ecef
from sat_marl_env.real_scenario import load_ephemeris_cache, load_task_catalog, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a real-orbit SatMARL scenario and audit raw visibility."
    )
    parser.add_argument("--ephemeris-cache", type=Path, required=True)
    parser.add_argument("--task-catalog", type=Path, required=True)
    parser.add_argument("--satellites", type=int, required=True)
    parser.add_argument("--tasks", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--lookahead-steps", type=int, default=30)
    parser.add_argument("--step-duration-seconds", type=float, default=30.0)
    parser.add_argument("--earth-radius-km", type=float, default=6378.137)
    parser.add_argument("--min-elevation-deg", type=float, default=3.0)
    parser.add_argument("--max-off-nadir-deg", type=float, default=45.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    ephemeris = load_ephemeris_cache(
        args.ephemeris_cache,
        expected_satellites=args.satellites,
        required_steps=args.max_steps + args.lookahead_steps,
        expected_step_duration_seconds=args.step_duration_seconds,
    )
    rows = load_task_catalog(args.task_catalog)
    if len(rows) < args.tasks:
        parser.error(
            f"Task catalog has {len(rows)} rows, fewer than --tasks={args.tasks}."
        )
    rows = rows[: args.tasks]
    targets = np.asarray(
        [
            geodetic_to_ecef(
                float(row["latitude_deg"]),
                float(row["longitude_deg"]),
                args.earth_radius_km,
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    target_up = targets / np.linalg.norm(targets, axis=1, keepdims=True)
    previous = np.zeros((args.satellites, args.tasks), dtype=bool)
    window_starts = 0
    visible_edges = 0
    satellite_steps_with_opportunity = 0
    off_nadir_sum = 0.0

    for step in range(args.max_steps):
        satellites = ephemeris.position_ecef_km[step]
        los = targets[None, :, :] - satellites[:, None, :]
        ranges = np.maximum(np.linalg.norm(los, axis=2), 1e-9)
        los_unit = los / ranges[:, :, None]
        elevation = np.degrees(
            np.arcsin(
                np.clip(
                    np.sum(-los_unit * target_up[None, :, :], axis=2),
                    -1.0,
                    1.0,
                )
            )
        )
        nadir = -satellites / np.linalg.norm(satellites, axis=1, keepdims=True)
        off_nadir = np.degrees(
            np.arccos(
                np.clip(
                    np.sum(nadir[:, None, :] * los_unit, axis=2),
                    -1.0,
                    1.0,
                )
            )
        )
        visible = (elevation >= args.min_elevation_deg) & (
            off_nadir <= args.max_off_nadir_deg
        )
        window_starts += int(np.count_nonzero(visible & ~previous))
        edge_count = int(np.count_nonzero(visible))
        visible_edges += edge_count
        satellite_steps_with_opportunity += int(
            np.count_nonzero(np.any(visible, axis=1))
        )
        off_nadir_sum += float(np.sum(off_nadir[visible]))
        previous = visible

    satellite_steps = args.max_steps * args.satellites
    report = {
        "status": "valid",
        "ephemeris": ephemeris.summary(),
        "task_catalog": {
            "path": str(args.task_catalog.expanduser().resolve()),
            "sha256": sha256_file(args.task_catalog),
            "rows_used": args.tasks,
        },
        "constraints": {
            "point_observation_seconds": 5.0,
            "step_duration_seconds": args.step_duration_seconds,
            "min_elevation_deg": args.min_elevation_deg,
            "max_off_nadir_deg": args.max_off_nadir_deg,
        },
        "raw_geometric_visibility": {
            "visibility_windows": window_starts,
            "visible_satellite_target_steps": visible_edges,
            "satellite_steps_with_opportunity": satellite_steps_with_opportunity,
            "satellite_step_opportunity_rate": (
                satellite_steps_with_opportunity / max(1, satellite_steps)
            ),
            "mean_off_nadir_deg_when_visible": (
                off_nadir_sum / max(1, visible_edges)
            ),
            "mean_visible_targets_per_satellite_step": (
                visible_edges / max(1, satellite_steps)
            ),
        },
        "note": (
            "This audit covers Earth occultation, target elevation, and off-nadir "
            "geometry. The environment additionally enforces payload compatibility, "
            "sunlight, resolution/swath, slew, energy, storage, deadlines, downlink, "
            "and multi-agent contention."
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
