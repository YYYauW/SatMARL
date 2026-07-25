from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


FIELDS = (
    "target_id",
    "target_name",
    "target_type",
    "latitude_deg",
    "longitude_deg",
    "priority",
    "release_step",
    "deadline_step",
    "duration",
    "required_mode",
    "required_resolution_m",
    "required_swath_km",
    "min_sun_elevation_deg",
    "energy_cost",
    "original_data_mb",
    "compression_ratio",
    "cooperation_mode",
    "required_observers",
    "max_coordination_gap_steps",
    "observation_duration_seconds",
    "area_width_km",
    "area_height_km",
    "area_orientation_deg",
    "coverage_threshold",
)

MODE_WEIGHTS = {"optical": 0.60, "sar": 0.30, "infrared": 0.10}
COOPERATION_WEIGHTS = {
    "single": 0.70,
    "sequential": 0.20,
    "simultaneous": 0.10,
}
AREA_SIZE_WEIGHTS = {"small": 1.0 / 3.0, "medium": 1.0 / 3.0, "large": 1.0 / 3.0}
AREA_SIZE_RANGES_KM = {
    "small": (12.0, 50.0),
    "medium": (50.0, 140.0),
    "large": (140.0, 300.0),
}
MODE_REQUIREMENTS = {
    "optical": {
        "resolution_m": (2.0, 10.0),
        "swath_km": (10.0, 65.0),
        "min_sun_elevation_deg": 8.0,
    },
    "sar": {
        "resolution_m": (8.0, 22.0),
        "swath_km": (25.0, 110.0),
        "min_sun_elevation_deg": -90.0,
    },
    "infrared": {
        "resolution_m": (18.0, 40.0),
        "swath_km": (45.0, 145.0),
        "min_sun_elevation_deg": -90.0,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def closest_factor_grid(count: int) -> tuple[int, int]:
    """Return an exact, near-square equal-area latitude/longitude grid."""

    root = int(math.sqrt(count))
    for latitude_bands in range(root, 0, -1):
        if count % latitude_bands == 0:
            return latitude_bands, count // latitude_bands
    return 1, count


def to_unit_vectors(points: np.ndarray) -> np.ndarray:
    latitude = np.radians(points[:, 0])
    longitude = np.radians(points[:, 1])
    cos_latitude = np.cos(latitude)
    return np.column_stack(
        (
            cos_latitude * np.cos(longitude),
            cos_latitude * np.sin(longitude),
            np.sin(latitude),
        )
    )


def minimum_angular_separation_deg(
    first: np.ndarray, second: np.ndarray | None = None, chunk_size: int = 256
) -> float:
    first_vectors = to_unit_vectors(first)
    second_vectors = (
        first_vectors if second is None else to_unit_vectors(second)
    )
    same = second is None
    largest_dot = -1.0
    for start in range(0, len(first_vectors), chunk_size):
        stop = min(len(first_vectors), start + chunk_size)
        dots = first_vectors[start:stop] @ second_vectors.T
        if same:
            rows = np.arange(start, stop)
            dots[np.arange(stop - start), rows] = -1.0
        largest_dot = max(largest_dot, float(np.max(dots)))
    return float(math.degrees(math.acos(float(np.clip(largest_dot, -1.0, 1.0)))))


def stratified_sphere(
    count: int,
    rng: np.random.Generator,
    *,
    forbidden_points: np.ndarray | None = None,
    minimum_separation_deg: float = 0.2,
    attempts_per_cell: int = 256,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Sample one point per equal-area grid cell without orbit alignment."""

    latitude_bands, longitude_sectors = closest_factor_grid(count)
    cells = [
        (latitude_band, longitude_sector)
        for latitude_band in range(latitude_bands)
        for longitude_sector in range(longitude_sectors)
    ]
    rng.shuffle(cells)
    longitude_offset = float(rng.uniform(-180.0, 180.0))
    cosine_limit = math.cos(math.radians(minimum_separation_deg))
    forbidden_vectors = (
        to_unit_vectors(forbidden_points)
        if forbidden_points is not None and len(forbidden_points)
        else np.empty((0, 3), dtype=np.float64)
    )
    accepted_points: list[tuple[float, float]] = []
    accepted_vectors: list[np.ndarray] = []

    for latitude_band, longitude_sector in cells:
        z_low = -1.0 + 2.0 * latitude_band / latitude_bands
        z_high = -1.0 + 2.0 * (latitude_band + 1) / latitude_bands
        longitude_low = (
            -180.0
            + 360.0 * longitude_sector / longitude_sectors
            + longitude_offset
        )
        longitude_high = longitude_low + 360.0 / longitude_sectors
        selected: tuple[float, float] | None = None
        selected_vector: np.ndarray | None = None
        for _ in range(attempts_per_cell):
            z = float(rng.uniform(z_low, z_high))
            latitude = math.degrees(math.asin(float(np.clip(z, -1.0, 1.0))))
            longitude = float(rng.uniform(longitude_low, longitude_high))
            longitude = ((longitude + 180.0) % 360.0) - 180.0
            vector = to_unit_vectors(
                np.asarray([[latitude, longitude]], dtype=np.float64)
            )[0]
            if (
                forbidden_vectors.size
                and float(np.max(forbidden_vectors @ vector)) > cosine_limit
            ):
                continue
            if accepted_vectors:
                existing = np.asarray(accepted_vectors)
                if float(np.max(existing @ vector)) > cosine_limit:
                    continue
            selected = (latitude, longitude)
            selected_vector = vector
            break
        if selected is None or selected_vector is None:
            raise RuntimeError(
                "Could not satisfy the requested spatial separation. "
                "Reduce --minimum-separation-deg or change the catalog size."
            )
        accepted_points.append(selected)
        accepted_vectors.append(selected_vector)

    points = np.asarray(accepted_points, dtype=np.float64)
    rng.shuffle(points)
    return points, (latitude_bands, longitude_sectors)


def quota_counts(count: int, weights: dict[str, float]) -> dict[str, int]:
    labels = list(weights)
    raw = np.asarray([weights[label] * count for label in labels], dtype=np.float64)
    allocated = np.floor(raw).astype(int)
    remainder = count - int(np.sum(allocated))
    order = np.argsort(-(raw - allocated), kind="stable")
    for index in order[:remainder]:
        allocated[index] += 1
    return {label: int(value) for label, value in zip(labels, allocated)}


def balanced_labels(
    count: int, weights: dict[str, float], rng: np.random.Generator
) -> np.ndarray:
    counts = quota_counts(count, weights)
    values = np.asarray(
        [label for label, amount in counts.items() for _ in range(amount)],
        dtype=np.str_,
    )
    rng.shuffle(values)
    return values


def stratified_uniform(
    count: int, low: float, high: float, rng: np.random.Generator
) -> np.ndarray:
    quantiles = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
    rng.shuffle(quantiles)
    return low + (high - low) * quantiles


def balanced_integer_values(
    count: int, low: int, high: int, rng: np.random.Generator
) -> np.ndarray:
    choices = np.arange(low, high + 1, dtype=np.int32)
    values = np.resize(choices, count)
    rng.shuffle(values)
    return values


def build_records(
    split: str,
    points: np.ndarray,
    rng: np.random.Generator,
    *,
    max_steps: int,
    min_window_steps: int,
    max_window_steps: int,
    area_fraction: float = 0.0,
) -> list[dict[str, Any]]:
    count = len(points)
    modes = balanced_labels(count, MODE_WEIGHTS, rng)
    area_count = int(round(count * area_fraction))
    target_types = np.asarray(
        ["area"] * area_count + ["point"] * (count - area_count),
        dtype=np.str_,
    )
    rng.shuffle(target_types)
    area_indices = np.flatnonzero(target_types == "area")
    point_indices = np.flatnonzero(target_types == "point")
    cooperation = np.full(count, "auto", dtype="<U12")
    cooperation[point_indices] = balanced_labels(
        len(point_indices), COOPERATION_WEIGHTS, rng
    )
    area_size_classes = balanced_labels(
        len(area_indices), AREA_SIZE_WEIGHTS, rng
    )
    area_width = np.zeros(count, dtype=np.float64)
    area_height = np.zeros(count, dtype=np.float64)
    area_orientation = np.zeros(count, dtype=np.float64)
    for size_class, bounds in AREA_SIZE_RANGES_KM.items():
        local = np.flatnonzero(area_size_classes == size_class)
        indices = area_indices[local]
        area_width[indices] = stratified_uniform(len(indices), *bounds, rng)
        area_height[indices] = stratified_uniform(len(indices), *bounds, rng)
    if len(area_indices):
        area_orientation[area_indices] = stratified_uniform(
            len(area_indices), 0.0, 360.0, rng
        )
    priorities = balanced_integer_values(count, 1, 10, rng)
    windows = np.rint(
        stratified_uniform(
            count, float(min_window_steps), float(max_window_steps), rng
        )
    ).astype(np.int32)
    release_quantiles = stratified_uniform(count, 0.0, 1.0, rng)
    original_data = stratified_uniform(count, 20.0, 90.0, rng)
    compression = stratified_uniform(count, 0.18, 0.45, rng)
    energy = stratified_uniform(count, 4.5, 8.4, rng)
    resolution = np.empty(count, dtype=np.float64)
    swath = np.empty(count, dtype=np.float64)
    minimum_sun = np.empty(count, dtype=np.float64)

    for mode, requirements in MODE_REQUIREMENTS.items():
        indices = np.flatnonzero(modes == mode)
        resolution[indices] = stratified_uniform(
            len(indices), *requirements["resolution_m"], rng
        )
        swath[indices] = stratified_uniform(
            len(indices), *requirements["swath_km"], rng
        )
        minimum_sun[indices] = float(requirements["min_sun_elevation_deg"])

    records: list[dict[str, Any]] = []
    for index in range(count):
        window = int(np.clip(windows[index], min_window_steps, max_window_steps))
        latest_release = max_steps - 1 - window
        release = int(
            min(
                latest_release,
                math.floor(release_quantiles[index] * (latest_release + 1)),
            )
        )
        deadline = release + window
        cooperation_mode = str(cooperation[index])
        target_type = str(target_types[index])
        if target_type == "area":
            cooperation_mode = "auto"
        records.append(
            {
                "target_id": f"{split}_{index + 1:06d}",
                "target_name": f"{split.title()}-Target-{index + 1:06d}",
                "target_type": target_type,
                "latitude_deg": f"{points[index, 0]:.8f}",
                "longitude_deg": f"{points[index, 1]:.8f}",
                "priority": int(priorities[index]),
                "release_step": release,
                "deadline_step": deadline,
                "duration": 1,
                "required_mode": str(modes[index]),
                "required_resolution_m": f"{resolution[index]:.4f}",
                "required_swath_km": f"{swath[index]:.4f}",
                "min_sun_elevation_deg": f"{minimum_sun[index]:.1f}",
                "energy_cost": f"{energy[index]:.4f}",
                "original_data_mb": f"{original_data[index]:.4f}",
                "compression_ratio": f"{compression[index]:.6f}",
                "cooperation_mode": cooperation_mode,
                "required_observers": (
                    0
                    if target_type == "area"
                    else 1 if cooperation_mode == "single" else 2
                ),
                "max_coordination_gap_steps": 120,
                "observation_duration_seconds": (
                    "20.0" if target_type == "area" else "5.0"
                ),
                "area_width_km": (
                    f"{area_width[index]:.4f}" if target_type == "area" else ""
                ),
                "area_height_km": (
                    f"{area_height[index]:.4f}" if target_type == "area" else ""
                ),
                "area_orientation_deg": (
                    f"{area_orientation[index]:.4f}"
                    if target_type == "area"
                    else ""
                ),
                "coverage_threshold": (
                    "0.95" if target_type == "area" else ""
                ),
            }
        )
    rng.shuffle(records)
    return records


def write_catalog(path: Path, records: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing catalog: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)


def summarize_catalog(records: list[dict[str, Any]]) -> dict[str, Any]:
    latitudes = np.asarray(
        [float(record["latitude_deg"]) for record in records], dtype=np.float64
    )
    longitudes = np.asarray(
        [float(record["longitude_deg"]) for record in records], dtype=np.float64
    )
    priorities = np.asarray(
        [int(record["priority"]) for record in records], dtype=np.int32
    )
    releases = np.asarray(
        [int(record["release_step"]) for record in records], dtype=np.int32
    )
    deadlines = np.asarray(
        [int(record["deadline_step"]) for record in records], dtype=np.int32
    )
    longitude_radians = np.radians(longitudes)
    longitude_resultant = math.hypot(
        float(np.mean(np.cos(longitude_radians))),
        float(np.mean(np.sin(longitude_radians))),
    )
    area_size_counts: Counter[str] = Counter()
    for record in records:
        if record["target_type"] != "area":
            continue
        maximum_extent = max(
            float(record["area_width_km"]),
            float(record["area_height_km"]),
        )
        if maximum_extent <= 50.0:
            area_size_counts["small"] += 1
        elif maximum_extent <= 140.0:
            area_size_counts["medium"] += 1
        else:
            area_size_counts["large"] += 1
    return {
        "rows": len(records),
        "unique_target_ids": len({record["target_id"] for record in records}),
        "unique_coordinates": len(
            {
                (record["latitude_deg"], record["longitude_deg"])
                for record in records
            }
        ),
        "mode_counts": dict(
            sorted(Counter(record["required_mode"] for record in records).items())
        ),
        "cooperation_counts": dict(
            sorted(
                Counter(
                    record["cooperation_mode"] for record in records
                ).items()
            )
        ),
        "target_type_counts": dict(
            sorted(Counter(record["target_type"] for record in records).items())
        ),
        "area_size_counts": dict(sorted(area_size_counts.items())),
        "priority_counts": {
            str(value): int(np.count_nonzero(priorities == value))
            for value in range(1, 11)
        },
        "spatial": {
            "mean_sin_latitude": float(np.mean(np.sin(np.radians(latitudes)))),
            "std_sin_latitude": float(np.std(np.sin(np.radians(latitudes)))),
            "north_fraction": float(np.mean(latitudes >= 0.0)),
            "longitude_resultant_length": longitude_resultant,
        },
        "time_windows": {
            "release_min": int(np.min(releases)),
            "release_max": int(np.max(releases)),
            "window_min": int(np.min(deadlines - releases)),
            "window_max": int(np.max(deadlines - releases)),
            "window_mean": float(np.mean(deadlines - releases)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate algorithm-agnostic, equal-area stratified SatMARL train "
            "and test task catalogs plus an auditable distribution manifest."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/targets"))
    parser.add_argument("--train-count", type=int, default=3072)
    parser.add_argument("--test-count", type=int, default=3072)
    parser.add_argument("--train-seed", type=int, default=2701)
    parser.add_argument("--test-seed", type=int, default=2702)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--min-window-steps", type=int, default=40)
    parser.add_argument("--max-window-steps", type=int, default=160)
    parser.add_argument("--minimum-separation-deg", type=float, default=0.2)
    parser.add_argument(
        "--area-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of continuous oriented area requests. Their single- or "
            "multi-satellite requirement is derived by the environment."
        ),
    )
    parser.add_argument(
        "--file-prefix",
        default="",
        help="Optional prefix, e.g. area_ creates area_train_requests.csv.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.train_count < 1 or args.test_count < 1:
        parser.error("Catalog sizes must be positive.")
    if not (
        1
        <= args.min_window_steps
        <= args.max_window_steps
        < args.max_steps
    ):
        parser.error(
            "Require 1 <= min window <= max window < max simulation steps."
        )
    if args.train_seed == args.test_seed:
        parser.error("Training and test seeds must differ.")
    if args.minimum_separation_deg < 0.0:
        parser.error("--minimum-separation-deg must be nonnegative.")
    if not 0.0 <= args.area_fraction <= 1.0:
        parser.error("--area-fraction must be within [0, 1].")

    train_rng = np.random.default_rng(args.train_seed)
    test_rng = np.random.default_rng(args.test_seed)
    train_points, train_grid = stratified_sphere(
        args.train_count,
        train_rng,
        minimum_separation_deg=args.minimum_separation_deg,
    )
    test_points, test_grid = stratified_sphere(
        args.test_count,
        test_rng,
        forbidden_points=train_points,
        minimum_separation_deg=args.minimum_separation_deg,
    )
    train_records = build_records(
        "train",
        train_points,
        train_rng,
        max_steps=args.max_steps,
        min_window_steps=args.min_window_steps,
        max_window_steps=args.max_window_steps,
        area_fraction=args.area_fraction,
    )
    test_records = build_records(
        "test",
        test_points,
        test_rng,
        max_steps=args.max_steps,
        min_window_steps=args.min_window_steps,
        max_window_steps=args.max_window_steps,
        area_fraction=args.area_fraction,
    )

    output_dir = args.output_dir.expanduser().resolve()
    train_path = output_dir / f"{args.file_prefix}train_requests.csv"
    test_path = output_dir / f"{args.file_prefix}test_requests.csv"
    manifest_path = output_dir / f"{args.file_prefix}catalog_manifest.json"
    write_catalog(train_path, train_records, args.overwrite)
    write_catalog(test_path, test_records, args.overwrite)

    train_coordinates = {
        (record["latitude_deg"], record["longitude_deg"])
        for record in train_records
    }
    test_coordinates = {
        (record["latitude_deg"], record["longitude_deg"])
        for record in test_records
    }
    manifest = {
        "format": "satmarl-task-catalog-v1",
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "design": {
            "spatial_sampling": (
                "one jittered point per equal-area sin(latitude)-longitude cell"
            ),
            "orbit_aligned": False,
            "geographic_demand_model": "global synthetic benchmark",
            "mode_weights": MODE_WEIGHTS,
            "cooperation_weights": COOPERATION_WEIGHTS,
            "area_fraction": args.area_fraction,
            "area_size_weights": AREA_SIZE_WEIGHTS,
            "area_size_ranges_km": AREA_SIZE_RANGES_KM,
            "area_collaboration": (
                "derived at reset from target dimensions, payload swath, "
                "observation duration, resolution, and compatible satellites"
            ),
            "priority_design": "balanced integer quotas from 1 through 10",
            "window_design": (
                f"stratified uniform [{args.min_window_steps}, "
                f"{args.max_window_steps}] steps"
            ),
            "minimum_within_and_cross_split_separation_deg": (
                args.minimum_separation_deg
            ),
        },
        "parameters": {
            "train_count": args.train_count,
            "test_count": args.test_count,
            "train_seed": args.train_seed,
            "test_seed": args.test_seed,
            "max_steps": args.max_steps,
            "train_equal_area_grid": list(train_grid),
            "test_equal_area_grid": list(test_grid),
        },
        "train": summarize_catalog(train_records),
        "test": summarize_catalog(test_records),
        "split_audit": {
            "exact_coordinate_overlap": len(
                train_coordinates.intersection(test_coordinates)
            ),
            "minimum_train_separation_deg": minimum_angular_separation_deg(
                train_points
            ),
            "minimum_test_separation_deg": minimum_angular_separation_deg(
                test_points
            ),
            "minimum_cross_split_separation_deg": (
                minimum_angular_separation_deg(train_points, test_points)
            ),
        },
    }
    manifest["files"] = {
        "train": {
            "path": train_path.name,
            "sha256": sha256_file(train_path),
        },
        "test": {
            "path": test_path.name,
            "sha256": sha256_file(test_path),
        },
    }
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite manifest: {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Saved training catalog: {train_path}")
    print(f"Saved test catalog:     {test_path}")
    print(f"Saved manifest:         {manifest_path}")


if __name__ == "__main__":
    main()
