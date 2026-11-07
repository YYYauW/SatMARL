"""Nested equal-area point requests for the full-method day-long load sweep.

Every complete block visits each fixed coarse cell once. A per-cell permutation
of refined cells spreads successive visits; the first ten 1000-point blocks
also cover every 40-by-250 refined cell once. Episodes jitter positions within
the same strata. Request attributes and release bins are balanced per block.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
from pathlib import Path

import numpy as np

from generate_task_catalogs import build_records, quota_counts, stratified_uniform

SAMPLING_PROTOCOL = "nested_equal_area_batches_v1"


def grid_options(config):
    specification = config["catalog"]
    if specification.get("sampling_protocol") != SAMPLING_PROTOCOL:
        raise ValueError("Unknown balanced catalogue sampling protocol")
    if specification["spatial_profile"] != "global_uniform":
        raise ValueError("Nested equal-area sampling requires global_uniform")
    grid = specification["spatial_grid"]
    names = ("latitude_bands", "longitude_sectors",
             "subcell_latitude_bands", "subcell_longitude_sectors")
    values = tuple(grid[name] for name in names)
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in values):
        raise ValueError("Equal-area grid dimensions must be positive integers")
    latitude_bands, longitude_sectors, sub_latitude, sub_longitude = values
    block_size = latitude_bands * longitude_sectors
    if (not config["task_counts"] or any(not isinstance(count, int) or isinstance(count, bool)
            or count < block_size or count % block_size for count in config["task_counts"])):
        raise ValueError(f"Balanced task counts must be positive multiples of {block_size}; use the original v3 config for a smoke test")
    bins = specification["release_time_bins"]
    horizon = config["env"]["max_steps"]
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 1:
        raise ValueError("release_time_bins must be a positive integer")
    if config["env"]["step_duration_seconds"] != 1 or horizon % bins:
        raise ValueError("Balanced release bins require integral second-based equal-length bins")
    if not 0 < specification["window_steps"][0] <= specification["window_steps"][1] < horizon / bins:
        raise ValueError("Every request window must fit inside the final release-time bin")
    return latitude_bands, longitude_sectors, sub_latitude, sub_longitude


def equal_bin_quotas(count, bins):
    """Integer bin totals with discrepancy below one for every cumulative M."""
    edges = np.arange(bins + 1, dtype=np.int64) * count // bins
    return np.diff(edges)


def _rng(config, episode, stream, block):
    # Independent streams keep earlier blocks stable when the requested maximum M changes.
    return np.random.default_rng(np.random.SeedSequence(
        [config["scenario_seed"], episode, stream, block]))


def generate_records(config, episode):
    latitude_bands, longitude_sectors, sub_latitude, sub_longitude = grid_options(config)
    block_size = latitude_bands * longitude_sectors
    cycle_size = sub_latitude * sub_longitude
    specification, environment = config["catalog"], config["env"]
    bins = specification["release_time_bins"]
    bin_seconds = environment["max_steps"] // bins
    cell_ids = np.arange(block_size)
    cell_latitude = cell_ids // longitude_sectors
    cell_longitude = cell_ids % longitude_sectors
    records, permutations = [], None
    for block in range(max(config["task_counts"]) // block_size):
        cycle, within_cycle = divmod(block, cycle_size)
        if within_cycle == 0:
            permutation_rng = _rng(config, episode, 913, cycle)
            permutations = np.asarray([permutation_rng.permutation(cycle_size) for _ in cell_ids])
        refined_cells = permutations[:, within_cycle]
        # A tiny interior margin preserves cell membership after eight-decimal CSV rounding.
        jitter = _rng(config, episode, 914, block).uniform(1e-6, 1 - 1e-6, (block_size, 2))
        z = -1 + 2 * (cell_latitude + (refined_cells // sub_longitude + jitter[:, 0]) / sub_latitude) / latitude_bands
        longitude = -180 + 360 * (cell_longitude + (refined_cells % sub_longitude + jitter[:, 1]) / sub_longitude) / longitude_sectors
        points = np.column_stack((np.degrees(np.arcsin(z)), longitude))
        attributes_rng = _rng(config, episode, 915, block)
        current = build_records(
            "test", points, attributes_rng, max_steps=environment["max_steps"],
            min_window_steps=specification["window_steps"][0],
            max_window_steps=specification["window_steps"][1],
            cooperative_observers_min=specification["cooperative_observers"][0],
            cooperative_observers_max=specification["cooperative_observers"][1],
            cooperation_weights=specification["fractions"],
        )
        release_counts = (equal_bin_quotas((block + 1) * block_size, bins)
                          - equal_bin_quotas(block * block_size, bins))
        release_bins = np.repeat(np.arange(bins), release_counts)
        attributes_rng.shuffle(release_bins)
        sizes = stratified_uniform(block_size, environment["original_data_min_mb"],
                                   environment["original_data_max_mb"], _rng(config, episode, 916, block))
        for index, row in enumerate(current):
            original_cell = int(row["target_id"].rsplit("_", 1)[1]) - 1
            number = block * block_size + index + 1
            window = row["deadline_step"] - row["release_step"]
            release_bin = int(release_bins[index])
            lower = release_bin * bin_seconds
            upper = min((release_bin + 1) * bin_seconds - 1, environment["max_steps"] - 1 - window)
            release = int(attributes_rng.integers(lower, upper + 1))
            raw = round(float(sizes[index]), 4)
            row.update(
                target_id=f"test_{number:06d}", target_name=f"Test-Target-{number:06d}",
                release_step=release, deadline_step=release + window,
                required_mode="optical", original_data_mb=raw, compression_ratio=.25,
                memory_required_mb=raw / 4, required_resolution_m=2.0,
                required_swath_km=10.0, min_sun_elevation_deg=8.0,
                observation_duration_seconds=environment["point_observation_seconds"],
                max_coordination_gap_steps=environment["sequential_max_gap_steps"],
                spatial_cell_id=original_cell, sampling_block=block,
            )
        # Keep blocks in order: every requested prefix contains whole balanced blocks.
        records.extend(current)
    return records


def _spatial_counts(rows, latitude_bands, longitude_sectors):
    latitude = np.asarray([float(row["latitude_deg"]) for row in rows])
    longitude = np.asarray([float(row["longitude_deg"]) for row in rows])
    if not np.all(np.isfinite(latitude)) or not np.all(np.isfinite(longitude)):
        raise ValueError("Non-finite target coordinates")
    if not np.all((latitude > -90) & (latitude < 90) & (longitude >= -180) & (longitude < 180)):
        raise ValueError("Coordinates outside the balanced global grid")
    lat_bin = np.minimum(latitude_bands - 1, np.floor((np.sin(np.radians(latitude)) + 1) * latitude_bands / 2).astype(int))
    lon_bin = np.minimum(longitude_sectors - 1, np.floor((longitude + 180) * longitude_sectors / 360).astype(int))
    return np.bincount(lat_bin * longitude_sectors + lon_bin,
                       minlength=latitude_bands * longitude_sectors)


def audit_rows(config, rows):
    latitude_bands, longitude_sectors, sub_latitude, sub_longitude = grid_options(config)
    block_size = latitude_bands * longitude_sectors
    specification = config["catalog"]
    if len(rows) != max(config["task_counts"]):
        raise ValueError("Wrong balanced master catalogue size")
    if len({row["target_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate target identifiers")
    expected_types = quota_counts(block_size, specification["fractions"])
    bin_seconds = config["env"]["max_steps"] // specification["release_time_bins"]
    scales = []
    for count in config["task_counts"]:
        subset = rows[:count]
        coarse = _spatial_counts(subset, latitude_bands, longitude_sectors)
        refined = _spatial_counts(subset, latitude_bands * sub_latitude, longitude_sectors * sub_longitude)
        types = Counter(row["cooperation_mode"] for row in subset)
        priorities = Counter(int(row["priority"]) for row in subset)
        releases = np.bincount([int(row["release_step"]) // bin_seconds for row in subset],
                               minlength=specification["release_time_bins"])
        if not np.all(coarse == count // block_size):
            raise ValueError(f"Nonuniform coarse spatial prefix at M={count}")
        if int(refined.max()) - int(refined.min()) > 1:
            raise ValueError(f"Unbalanced refined spatial prefix at M={count}")
        if any(types[name] != per_block * (count // block_size) for name, per_block in expected_types.items()):
            raise ValueError(f"Request-type counts changed at M={count}")
        if priorities != Counter({priority: count // 10 for priority in range(1, 11)}):
            raise ValueError(f"Unbalanced priorities at M={count}")
        if not np.array_equal(releases, equal_bin_quotas(count, specification["release_time_bins"])):
            raise ValueError(f"Unbalanced request release times at M={count}")
        for row in subset:
            release, deadline = int(row["release_step"]), int(row["deadline_step"])
            if not (0 <= release <= deadline < config["env"]["max_steps"]
                    and specification["window_steps"][0] <= deadline - release <= specification["window_steps"][1]):
                raise ValueError("Request time-window constraint changed")
        scales.append(dict(
            tasks=count, spatial_grid_min_count=int(coarse.min()), spatial_grid_max_count=int(coarse.max()),
            spatial_cell_count_sha256=hashlib.sha256(coarse.astype("<i8").tobytes()).hexdigest(),
            refined_grid_min_count=int(refined.min()), refined_grid_max_count=int(refined.max()),
            single_requests=int(types["single"]), multi_observation_requests=count - int(types["single"]),
            priority_counts=[priorities[priority] for priority in range(1, 11)],
            release_bin_counts=releases.tolist(),
            mean_raw_data_mb=float(np.mean([float(row["original_data_mb"]) for row in subset])),
            spatially_balanced=True,
        ))
    return scales


def balance_report(config, catalog_paths):
    grid_options(config)
    if not catalog_paths:
        raise ValueError("At least one balanced catalogue is required")
    episodes = []
    signatures = {count: set() for count in config["task_counts"]}
    for episode, path in enumerate(catalog_paths):
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            scales = audit_rows(config, list(csv.DictReader(handle)))
        for scale in scales:
            signatures[scale["tasks"]].add(scale["spatial_cell_count_sha256"])
        episodes.append(dict(episode=episode, catalog=Path(path).name, scales=scales))
    return dict(
        sampling_protocol=SAMPLING_PROTOCOL,
        spatial_grid=config["catalog"]["spatial_grid"],
        release_time_bins=config["catalog"]["release_time_bins"],
        episode_count=len(episodes), all_prefixes_balanced=True,
        spatial_counts_identical_across_episodes=all(len(values) == 1 for values in signatures.values()),
        episodes=episodes,
    )
