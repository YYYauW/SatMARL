from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EPHEMERIS_FORMAT = "satmarl-ephemeris-v1"


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def read_tle_catalog(path: Path) -> list[tuple[str, str, str]]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    entries: list[tuple[str, str, str]] = []
    index = 0
    while index < len(lines):
        if lines[index].startswith("1 ") and index + 1 < len(lines):
            name = f"SAT-{len(entries):04d}"
            line1, line2 = lines[index], lines[index + 1]
            index += 2
        elif (
            index + 2 < len(lines)
            and lines[index + 1].startswith("1 ")
            and lines[index + 2].startswith("2 ")
        ):
            name = lines[index].removeprefix("0 ").strip()
            line1, line2 = lines[index + 1], lines[index + 2]
            index += 3
        else:
            raise ValueError(
                f"Malformed TLE catalog near line {index + 1} in {path}."
            )
        if not line1.startswith("1 ") or not line2.startswith("2 "):
            raise ValueError(f"Malformed TLE pair for {name!r} in {path}.")
        entries.append((name, line1, line2))
    if not entries:
        raise ValueError(f"No TLE records found in {path}.")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a reproducible SatMARL ECI/ECEF cache from an archived TLE "
            "catalog using Skyfield's SGP4 propagator."
        )
    )
    parser.add_argument("--tle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-utc", required=True, help="ISO-8601 UTC timestamp.")
    parser.add_argument("--satellites", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--lookahead-steps", type=int, default=30)
    parser.add_argument("--step-duration-seconds", type=float, default=30.0)
    parser.add_argument("--max-tle-age-days", type=float, default=14.0)
    parser.add_argument(
        "--allow-stale-tle",
        action="store_true",
        help="Allow propagation farther from the TLE epoch; recorded as a warning.",
    )
    args = parser.parse_args()

    try:
        from skyfield.api import EarthSatellite, load
        from skyfield.framelib import itrs
    except ImportError as exc:
        raise SystemExit(
            "Skyfield is required only for cache generation. Install with "
            "`python -m pip install -e '.[real-orbit]'`."
        ) from exc

    tle_path = args.tle.expanduser().resolve()
    output = args.output.expanduser().resolve()
    start = parse_utc(args.start_utc)
    if args.satellites < 1:
        parser.error("--satellites must be positive")
    if args.max_steps < 1 or args.lookahead_steps < 0:
        parser.error("--max-steps must be positive and --lookahead-steps nonnegative")
    if args.step_duration_seconds <= 0:
        parser.error("--step-duration-seconds must be positive")

    records = read_tle_catalog(tle_path)
    if len(records) < args.satellites:
        parser.error(
            f"TLE catalog has {len(records)} records, fewer than "
            f"--satellites={args.satellites}."
        )
    records = records[: args.satellites]
    total_steps = args.max_steps + args.lookahead_steps
    datetimes = [
        start + timedelta(seconds=index * args.step_duration_seconds)
        for index in range(total_steps)
    ]

    ts = load.timescale()
    times = ts.from_datetimes(datetimes)
    position_eci = np.empty((total_steps, args.satellites, 3), dtype=np.float64)
    velocity_eci = np.empty_like(position_eci)
    position_ecef = np.empty_like(position_eci)
    velocity_ecef = np.empty_like(position_eci)
    names: list[str] = []
    epochs: list[datetime] = []
    stale: list[dict[str, float | str]] = []

    for satellite_id, (name, line1, line2) in enumerate(records):
        satellite = EarthSatellite(line1, line2, name, ts)
        epoch = satellite.epoch.utc_datetime().astimezone(timezone.utc)
        age_days = abs((start - epoch).total_seconds()) / 86400.0
        if age_days > args.max_tle_age_days:
            stale.append({"satellite": name, "age_days": age_days})
        geocentric = satellite.at(times)
        position_eci[:, satellite_id, :] = geocentric.position.km.T
        velocity_eci[:, satellite_id, :] = geocentric.velocity.km_per_s.T
        ecef_position, ecef_velocity = geocentric.frame_xyz_and_velocity(itrs)
        position_ecef[:, satellite_id, :] = ecef_position.km.T
        velocity_ecef[:, satellite_id, :] = ecef_velocity.km_per_s.T
        names.append(name)
        epochs.append(epoch)
        print(
            f"[{satellite_id + 1:04d}/{args.satellites:04d}] "
            f"{name}: TLE age={age_days:.2f} days",
            flush=True,
        )

    if stale and not args.allow_stale_tle:
        worst = max(float(item["age_days"]) for item in stale)
        raise SystemExit(
            f"{len(stale)} TLE records exceed --max-tle-age-days="
            f"{args.max_tle_age_days:g} (worst={worst:.2f}). Use a TLE snapshot "
            "closer to --start-utc, or explicitly pass --allow-stale-tle."
        )

    tle_sha256 = hashlib.sha256(tle_path.read_bytes()).hexdigest()
    metadata = {
        "format": EPHEMERIS_FORMAT,
        "mode": "tle_sgp4_cache",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": "archived_tle_skyfield_sgp4",
        "tle_file": str(tle_path),
        "tle_sha256": tle_sha256,
        "start_utc": start.isoformat().replace("+00:00", "Z"),
        "end_utc": datetimes[-1].isoformat().replace("+00:00", "Z"),
        "step_duration_seconds": float(args.step_duration_seconds),
        "max_steps": int(args.max_steps),
        "lookahead_steps": int(args.lookahead_steps),
        "total_steps": int(total_steps),
        "satellites": int(args.satellites),
        "tle_epoch_min_utc": min(epochs).isoformat().replace("+00:00", "Z"),
        "tle_epoch_max_utc": max(epochs).isoformat().replace("+00:00", "Z"),
        "max_tle_age_days": float(args.max_tle_age_days),
        "stale_tle_records": stale,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        satellite_names=np.asarray(names, dtype=np.str_),
        position_eci_km=position_eci,
        velocity_eci_km_s=velocity_eci,
        position_ecef_km=position_ecef,
        velocity_ecef_km_s=velocity_ecef,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Saved ephemeris cache: {output}")


if __name__ == "__main__":
    main()
