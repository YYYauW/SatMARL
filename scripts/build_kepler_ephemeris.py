from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sat_marl_env.entities import OrbitalElements
from sat_marl_env.orbital import propagate_kepler
from sat_marl_env.real_scenario import EPHEMERIS_FORMAT


ELEMENT_FIELDS = (
    "name",
    "plane_id",
    "semi_major_axis_km",
    "eccentricity",
    "inclination_deg",
    "raan_deg",
    "argument_of_perigee_deg",
    "mean_anomaly_deg",
)


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def greenwich_sidereal_angle_rad(value: datetime) -> float:
    julian_date = value.timestamp() / 86400.0 + 2440587.5
    centuries = (julian_date - 2451545.0) / 36525.0
    gmst_deg = (
        280.46061837
        + 360.98564736629 * (julian_date - 2451545.0)
        + 0.000387933 * centuries * centuries
        - centuries * centuries * centuries / 38710000.0
    )
    return math.radians(gmst_deg % 360.0)


def canonical_sha256(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        records, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def generate_walker_delta(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.satellites % args.planes != 0:
        raise ValueError(
            "Walker-delta generation requires --satellites divisible by --planes."
        )
    if not 0 <= args.walker_phasing < args.planes:
        raise ValueError("--walker-phasing must satisfy 0 <= F < number of planes.")
    satellites_per_plane = args.satellites // args.planes
    semi_major_axis = args.earth_radius_km + args.altitude_km
    records: list[dict[str, Any]] = []
    # Plane-interleaved ordering matches the MARL environment's agent indexing.
    for slot in range(satellites_per_plane):
        for plane_id in range(args.planes):
            raan = (
                args.raan_offset_deg + 360.0 * plane_id / args.planes
            ) % 360.0
            phase_shift = (
                360.0 * args.walker_phasing * plane_id / args.satellites
            )
            mean_anomaly = (
                args.mean_anomaly_offset_deg
                + 360.0 * slot / satellites_per_plane
                + phase_shift
            ) % 360.0
            records.append(
                {
                    "name": f"WALKER-P{plane_id:02d}-S{slot:03d}",
                    "plane_id": plane_id,
                    "semi_major_axis_km": semi_major_axis,
                    "eccentricity": args.eccentricity,
                    "inclination_deg": args.inclination_deg,
                    "raan_deg": raan,
                    "argument_of_perigee_deg": args.argument_of_perigee_deg,
                    "mean_anomaly_deg": mean_anomaly,
                }
            )
    return records


def read_elements(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Orbital-element CSV has no header: {path}")
        missing = set(ELEMENT_FIELDS).difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Orbital-element CSV is missing columns: {sorted(missing)}"
            )
        records: list[dict[str, Any]] = []
        for row_index, row in enumerate(reader, start=2):
            try:
                records.append(
                    {
                        "name": str(row["name"]).strip(),
                        "plane_id": int(row["plane_id"]),
                        "semi_major_axis_km": float(row["semi_major_axis_km"]),
                        "eccentricity": float(row["eccentricity"]),
                        "inclination_deg": float(row["inclination_deg"]),
                        "raan_deg": float(row["raan_deg"]) % 360.0,
                        "argument_of_perigee_deg": float(
                            row["argument_of_perigee_deg"]
                        )
                        % 360.0,
                        "mean_anomaly_deg": float(row["mean_anomaly_deg"]) % 360.0,
                    }
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid orbital elements at CSV row {row_index}: {exc}"
                ) from exc
    if not records:
        raise ValueError(f"Orbital-element CSV contains no satellites: {path}")
    return records


def validate_elements(
    records: list[dict[str, Any]], args: argparse.Namespace
) -> None:
    if len(records) != args.satellites:
        raise ValueError(
            f"Orbital elements contain {len(records)} satellites, but "
            f"--satellites={args.satellites}."
        )
    names: set[str] = set()
    for record in records:
        name = str(record["name"])
        if not name or name in names:
            raise ValueError(f"Satellite names must be nonempty and unique: {name!r}")
        names.add(name)
        if not 0 <= int(record["plane_id"]) < args.planes:
            raise ValueError(
                f"{name} has plane_id={record['plane_id']} outside "
                f"[0, {args.planes - 1}]."
            )
        if float(record["semi_major_axis_km"]) <= args.earth_radius_km:
            raise ValueError(f"{name} has a non-orbital semi-major axis.")
        if not 0 <= float(record["eccentricity"]) < 1:
            raise ValueError(f"{name} requires 0 <= eccentricity < 1.")
        if not 0 <= float(record["inclination_deg"]) <= 180:
            raise ValueError(f"{name} has invalid inclination.")


def write_elements(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ELEMENT_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def eci_to_ecef_state(
    position_eci: np.ndarray,
    velocity_eci: np.ndarray,
    theta_rad: float,
    earth_rotation_rate_rad_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
    rotation = np.asarray(
        [[cos_t, sin_t, 0.0], [-sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    omega_cross_position = np.asarray(
        [
            -earth_rotation_rate_rad_s * position_eci[1],
            earth_rotation_rate_rad_s * position_eci[0],
            0.0,
        ],
        dtype=np.float64,
    )
    return (
        rotation @ position_eci,
        rotation @ (velocity_eci - omega_cross_position),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a reproducible six-Keplerian-element constellation cache "
            "from a CSV catalog or a Walker-delta definition."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--elements-output", type=Path, default=None)
    parser.add_argument("--elements-csv", type=Path, default=None)
    parser.add_argument("--start-utc", required=True)
    parser.add_argument("--satellites", type=int, default=64)
    parser.add_argument("--planes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--lookahead-steps", type=int, default=30)
    parser.add_argument("--step-duration-seconds", type=float, default=30.0)
    parser.add_argument("--earth-radius-km", type=float, default=6378.137)
    parser.add_argument("--earth-mu-km3-s2", type=float, default=398600.4418)
    parser.add_argument(
        "--earth-rotation-rate-rad-s", type=float, default=7.2921159e-5
    )
    parser.add_argument("--altitude-km", type=float, default=550.0)
    parser.add_argument("--eccentricity", type=float, default=0.001)
    parser.add_argument("--inclination-deg", type=float, default=97.6)
    parser.add_argument("--walker-phasing", type=int, default=1)
    parser.add_argument("--raan-offset-deg", type=float, default=0.0)
    parser.add_argument("--argument-of-perigee-deg", type=float, default=0.0)
    parser.add_argument("--mean-anomaly-offset-deg", type=float, default=0.0)
    args = parser.parse_args()

    if args.satellites < 1 or args.planes < 1:
        parser.error("--satellites and --planes must be positive")
    if args.max_steps < 1 or args.lookahead_steps < 0:
        parser.error("--max-steps must be positive and --lookahead-steps nonnegative")
    if args.step_duration_seconds <= 0:
        parser.error("--step-duration-seconds must be positive")
    try:
        records = (
            read_elements(args.elements_csv.expanduser().resolve())
            if args.elements_csv is not None
            else generate_walker_delta(args)
        )
        validate_elements(records, args)
    except ValueError as exc:
        parser.error(str(exc))

    start = parse_utc(args.start_utc)
    total_steps = args.max_steps + args.lookahead_steps
    names = np.asarray([record["name"] for record in records], dtype=np.str_)
    plane_ids = np.asarray(
        [int(record["plane_id"]) for record in records], dtype=np.int32
    )
    position_eci = np.empty((total_steps, args.satellites, 3), dtype=np.float64)
    velocity_eci = np.empty_like(position_eci)
    position_ecef = np.empty_like(position_eci)
    velocity_ecef = np.empty_like(position_eci)
    gmst0 = greenwich_sidereal_angle_rad(start)

    for satellite_id, record in enumerate(records):
        elements = OrbitalElements(
            semi_major_axis_km=float(record["semi_major_axis_km"]),
            eccentricity=float(record["eccentricity"]),
            inclination_deg=float(record["inclination_deg"]),
            raan_deg=float(record["raan_deg"]),
            argument_of_perigee_deg=float(record["argument_of_perigee_deg"]),
            mean_anomaly_deg=float(record["mean_anomaly_deg"]),
        )
        for step in range(total_steps):
            elapsed = step * args.step_duration_seconds
            position_i, velocity_i, _ = propagate_kepler(
                elements, elapsed, args.earth_mu_km3_s2, 0.0
            )
            theta = gmst0 + args.earth_rotation_rate_rad_s * elapsed
            position_e, velocity_e = eci_to_ecef_state(
                position_i,
                velocity_i,
                theta,
                args.earth_rotation_rate_rad_s,
            )
            position_eci[step, satellite_id] = position_i
            velocity_eci[step, satellite_id] = velocity_i
            position_ecef[step, satellite_id] = position_e
            velocity_ecef[step, satellite_id] = velocity_e
        print(
            f"[{satellite_id + 1:04d}/{args.satellites:04d}] "
            f"{record['name']} plane={record['plane_id']}",
            flush=True,
        )

    output = args.output.expanduser().resolve()
    elements_output = (
        args.elements_output.expanduser().resolve()
        if args.elements_output is not None
        else output.with_suffix(".elements.csv")
    )
    write_elements(elements_output, records)
    elements_sha256 = canonical_sha256(records)
    metadata = {
        "format": EPHEMERIS_FORMAT,
        "mode": "keplerian_six_element_cache",
        "source": (
            "orbital_elements_csv"
            if args.elements_csv is not None
            else "walker_delta_six_elements"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "start_utc": start.isoformat().replace("+00:00", "Z"),
        "step_duration_seconds": float(args.step_duration_seconds),
        "max_steps": int(args.max_steps),
        "lookahead_steps": int(args.lookahead_steps),
        "total_steps": int(total_steps),
        "satellites": int(args.satellites),
        "planes": int(args.planes),
        "elements_file": str(elements_output),
        "elements_sha256": elements_sha256,
        "constellation": {
            "type": "custom_elements" if args.elements_csv else "walker_delta",
            "altitude_km": None if args.elements_csv else float(args.altitude_km),
            "inclination_deg": (
                None if args.elements_csv else float(args.inclination_deg)
            ),
            "eccentricity": (
                None if args.elements_csv else float(args.eccentricity)
            ),
            "walker_phasing": (
                None if args.elements_csv else int(args.walker_phasing)
            ),
            "raan_offset_deg": (
                None if args.elements_csv else float(args.raan_offset_deg)
            ),
        },
        "earth_model": {
            "radius_km": float(args.earth_radius_km),
            "mu_km3_s2": float(args.earth_mu_km3_s2),
            "rotation_rate_rad_s": float(args.earth_rotation_rate_rad_s),
            "gmst_at_epoch_deg": math.degrees(gmst0),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        satellite_names=names,
        plane_ids=plane_ids,
        position_eci_km=position_eci,
        velocity_eci_km_s=velocity_eci,
        position_ecef_km=position_ecef,
        velocity_ecef_km_s=velocity_ecef,
        orbital_elements_json=np.asarray(json.dumps(records, ensure_ascii=False)),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Saved orbital elements: {elements_output}")
    print(f"Saved ephemeris cache: {output}")


if __name__ == "__main__":
    main()
