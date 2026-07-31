from __future__ import annotations

import math

import numpy as np


Point = tuple[float, float]


def _radical_inverse(index: int, base: int) -> float:
    value = 0.0
    factor = 1.0 / base
    while index:
        index, digit = divmod(index, base)
        value += digit * factor
        factor /= base
    return value


def low_discrepancy_rectangle_samples(
    width_km: float, height_km: float, count: int
) -> np.ndarray:
    """Return deterministic non-grid samples over a task-local rectangle."""

    if width_km <= 0.0 or height_km <= 0.0 or count < 1:
        raise ValueError("Area dimensions and sample count must be positive.")
    values = np.empty((count, 2), dtype=np.float64)
    for row in range(count):
        index = row + 1
        values[row, 0] = (_radical_inverse(index, 2) - 0.5) * width_km
        values[row, 1] = (_radical_inverse(index, 3) - 0.5) * height_km
    return values


def oriented_strip_polygon(
    center_u_km: float,
    center_v_km: float,
    length_km: float,
    width_km: float,
    relative_heading_deg: float,
) -> list[Point]:
    """Build a directed rectangular strip in task-local coordinates."""

    heading = math.radians(relative_heading_deg)
    along = np.asarray([math.cos(heading), math.sin(heading)], dtype=np.float64)
    across = np.asarray([-math.sin(heading), math.cos(heading)], dtype=np.float64)
    center = np.asarray([center_u_km, center_v_km], dtype=np.float64)
    half_length = max(0.0, length_km) / 2.0
    half_width = max(0.0, width_km) / 2.0
    return [
        tuple(center - along * half_length - across * half_width),
        tuple(center + along * half_length - across * half_width),
        tuple(center + along * half_length + across * half_width),
        tuple(center - along * half_length + across * half_width),
    ]


def polygon_area(polygon: list[Point]) -> float:
    if len(polygon) < 3:
        return 0.0
    points = np.asarray(polygon, dtype=np.float64)
    return 0.5 * abs(
        float(
            np.dot(points[:, 0], np.roll(points[:, 1], -1))
            - np.dot(points[:, 1], np.roll(points[:, 0], -1))
        )
    )


def _clip_edge(
    polygon: list[Point], axis: int, bound: float, keep_greater: bool
) -> list[Point]:
    if not polygon:
        return []

    def inside(point: Point) -> bool:
        value = point[axis]
        return value >= bound - 1e-12 if keep_greater else value <= bound + 1e-12

    output: list[Point] = []
    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside != previous_inside:
            denominator = current[axis] - previous[axis]
            fraction = 0.0 if abs(denominator) < 1e-12 else (
                bound - previous[axis]
            ) / denominator
            intersection = (
                previous[0] + fraction * (current[0] - previous[0]),
                previous[1] + fraction * (current[1] - previous[1]),
            )
            output.append(intersection)
        if current_inside:
            output.append(current)
        previous = current
        previous_inside = current_inside
    return output


def clip_polygon_to_target(
    polygon: list[Point], target_width_km: float, target_height_km: float
) -> list[Point]:
    """Clip a strip polygon to the continuous rectangular target boundary."""

    half_width = target_width_km / 2.0
    half_height = target_height_km / 2.0
    clipped = _clip_edge(polygon, 0, -half_width, True)
    clipped = _clip_edge(clipped, 0, half_width, False)
    clipped = _clip_edge(clipped, 1, -half_height, True)
    return _clip_edge(clipped, 1, half_height, False)


def strip_sample_mask(
    samples: np.ndarray,
    center_u_km: float,
    center_v_km: float,
    length_km: float,
    width_km: float,
    relative_heading_deg: float,
) -> int:
    """Encode task samples covered by a directed strip as a Python bitset."""

    heading = math.radians(relative_heading_deg)
    along = np.asarray([math.cos(heading), math.sin(heading)], dtype=np.float64)
    across = np.asarray([-math.sin(heading), math.cos(heading)], dtype=np.float64)
    delta = samples - np.asarray([center_u_km, center_v_km], dtype=np.float64)
    selected = (
        (np.abs(delta @ along) <= max(0.0, length_km) / 2.0 + 1e-9)
        & (np.abs(delta @ across) <= max(0.0, width_km) / 2.0 + 1e-9)
    )
    mask = 0
    for index in np.flatnonzero(selected):
        mask |= 1 << int(index)
    return mask


def uncovered_centroid(samples: np.ndarray, covered_mask: int) -> Point:
    uncovered = np.fromiter(
        (
            not bool((covered_mask >> index) & 1)
            for index in range(len(samples))
        ),
        dtype=bool,
        count=len(samples),
    )
    if not np.any(uncovered):
        return 0.0, 0.0
    center = np.mean(samples[uncovered], axis=0)
    return float(center[0]), float(center[1])


def best_strip_center(
    samples: np.ndarray,
    covered_mask: int,
    length_km: float,
    width_km: float,
    relative_heading_deg: float,
    target_width_km: float,
    target_height_km: float,
    max_candidates: int = 48,
) -> Point:
    """Choose a complementary strip center without turning cells into actions."""

    if covered_mask == 0:
        return 0.0, 0.0
    uncovered_indices = np.fromiter(
        (
            index
            for index in range(len(samples))
            if not bool((covered_mask >> index) & 1)
        ),
        dtype=np.int64,
    )
    if not len(uncovered_indices):
        return 0.0, 0.0
    stride = max(1, int(math.ceil(len(uncovered_indices) / max_candidates)))
    candidate_points = samples[uncovered_indices[::stride]]
    centroid = np.mean(samples[uncovered_indices], axis=0, keepdims=True)
    candidate_points = np.vstack((centroid, candidate_points))
    best_center = (float(centroid[0, 0]), float(centroid[0, 1]))
    best_score: tuple[int, float, float] | None = None
    target_area = max(target_width_km * target_height_km, 1e-9)
    for center_u, center_v in candidate_points:
        mask = strip_sample_mask(
            samples,
            float(center_u),
            float(center_v),
            length_km,
            width_km,
            relative_heading_deg,
        )
        marginal = (mask & ~covered_mask).bit_count()
        footprint = oriented_strip_polygon(
            float(center_u),
            float(center_v),
            length_km,
            width_km,
            relative_heading_deg,
        )
        inside_area = polygon_area(
            clip_polygon_to_target(
                footprint, target_width_km, target_height_km
            )
        )
        outside_area = max(0.0, length_km * width_km - inside_area)
        estimated_new_area = marginal / len(samples) * target_area
        redundant_area = max(0.0, inside_area - estimated_new_area)
        score = (marginal, -outside_area, -redundant_area)
        if best_score is None or score > best_score:
            best_score = score
            best_center = float(center_u), float(center_v)
    return best_center
