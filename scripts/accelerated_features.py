"""Exact-order NumPy feature preparation for isolated evaluation workers.

No observation or action is retained between decisions. Group membership, row
order, scalar utility expressions, dtype conversion and NumPy reduction order
match ``marl_common``. Neural inference, masks, argmax and bids remain in the
original ``evaluate_marl.build_policy`` implementation.
"""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Any, Callable

import numpy as np


def _factors(rows: list[dict], masks: np.ndarray | None = None,
             candidate_views: list[np.ndarray] | None = None) -> np.ndarray:
    if not rows:
        return np.empty((0, 0), dtype=np.float32)
    candidate_k = int(rows[0]["candidates"].shape[0])
    output = np.zeros((len(rows), candidate_k, 4), dtype=np.float32)
    groups: dict[int, list[tuple[int, int, float, float]]] = {}
    for row, obs in enumerate(rows):
        ids = np.asarray(obs["candidate_ids"], dtype=np.int64)
        candidates = (candidate_views[row] if candidate_views is not None
                      else np.asarray(obs["candidates"], dtype=np.float32))
        mask = masks[row] if masks is not None else np.asarray(obs["action_mask"], dtype=bool)
        for slot in range(candidate_k):
            task_id = int(ids[slot])
            if task_id < 0 or slot + 2 >= len(mask) or not bool(mask[slot + 2]):
                continue
            feature = candidates[slot]
            # Keep the original scalar expression and operand dtypes. Changing
            # this to a vectorized dot product can change floating-point bits.
            utility = float(feature[0] * feature[19]
                            - 0.10 * feature[22] - 0.05 * feature[21])
            required = max(1.0, round(float(feature[14]) * 2.0))
            groups.setdefault(task_id, []).append((row, slot, utility, required))
    count_scale = max(1.0, float(np.log1p(max(2, len(rows)))))
    for entries in groups.values():
        count = len(entries)
        # Identical list order and float64 NumPy mean to the original code.
        mean_utility = float(np.mean([entry[2] for entry in entries]))
        density = float(np.log1p(count) / count_scale)
        for row, slot, utility, required in entries:
            # NumPy casts each scalar on assignment, avoiding an additional
            # four-element ndarray allocation on every feasible edge.
            output[row, slot] = (density, mean_utility,
                                 utility - mean_utility,
                                 min(1.0, count / required))
    return output.reshape(len(rows), -1)


def opportunity_graph_features(observations: dict[str, dict]) -> np.ndarray:
    return _factors(list(observations.values()))


def _weak(self_features: np.ndarray, action_means: np.ndarray,
          task_opportunity: np.ndarray, latitude_bins: int,
          longitude_bins: int) -> np.ndarray:
    plane_keys = np.round(self_features[:, 1], decimals=6)
    latitude_index = np.clip(
        ((self_features[:, 2] + 1.0) * 0.5 * latitude_bins).astype(np.int32),
        0, latitude_bins - 1,
    )
    longitude_index = np.clip(
        ((self_features[:, 3] + 1.0) * 0.5 * longitude_bins).astype(np.int32),
        0, longitude_bins - 1,
    )
    region_keys = latitude_index * longitude_bins + longitude_index
    output = np.zeros((len(self_features), 12), dtype=np.float32)
    for keys, start, include_actions in ((plane_keys, 0, True),
                                         (region_keys, 8, False)):
        groups: dict[float | int, list[int]] = {}
        for row, key in enumerate(keys):
            groups.setdefault(key.item(), []).append(row)
        for members in groups.values():
            index = np.asarray(members, dtype=np.int64)
            population = self_features[index]
            # Separate means preserve the original NumPy striding and
            # summation order. Do not replace them with bincount/reduceat.
            base = np.asarray([
                float(np.mean(population[:, 5])),
                float(np.mean(population[:, 6])),
                float(np.mean(population[:, 17])),
                float(np.mean(task_opportunity[index])),
            ], dtype=np.float32)
            output[index, start:start + 4] = base
            if include_actions:
                output[index, start + 4:start + 4 + action_means.shape[1]] = (
                    np.mean(action_means[index], axis=0).astype(np.float32)
                )
    return output


def weak_mean_field_features(observations: dict[str, dict], latitude_bins: int = 6,
                             longitude_bins: int = 12) -> np.ndarray:
    rows = list(observations.values())
    if not rows:
        return np.empty((0, 12), dtype=np.float32)
    self_features = np.stack([np.asarray(row["self"], dtype=np.float32) for row in rows])
    action_means = np.stack([np.asarray(row["neighbor_action_mean"], dtype=np.float32)
                             for row in rows])
    task_opportunity = np.asarray([
        float(np.any(np.asarray(row["action_mask"])[2:])) for row in rows
    ], dtype=np.float32)
    return _weak(self_features, action_means, task_opportunity,
                 latitude_bins, longitude_bins)


def _local(observations: dict[str, dict]):
    agents = list(observations)
    rows = list(observations.values())
    # Keep the general original concat->cast order even for unusual mixed
    # dtypes. This also preserves the original empty-input ValueError.
    local = np.stack([
        np.concatenate([row["self"].ravel(), row["neighbors"].ravel(),
                        row["neighbor_action_mean"].ravel(),
                        row["candidates"].ravel()]).astype(np.float32, copy=False)
        for row in rows
    ])
    global_state = np.asarray(rows[0]["global"], dtype=np.float32)
    masks = np.stack([row["action_mask"] for row in rows]).astype(bool)
    return agents, rows, local, global_state, masks


def _uniform_float32(rows: list[dict]) -> bool:
    names = ("self", "neighbors", "neighbor_action_mean", "candidates")
    shapes = {name: rows[0][name].shape for name in names}
    return all(all(row[name].dtype == np.float32 and row[name].shape == shapes[name]
                   for name in names) for row in rows)


def _candidate_views(rows: list[dict], local: np.ndarray,
                     uniform_float32: bool) -> list[np.ndarray] | None:
    # Reuse the already flattened float32 candidate buffer only when the cast
    # path is provably identical. Mixed dtypes retain their original conversion.
    if not uniform_float32:
        return None
    width = rows[0]["candidates"].size
    if width == 0:
        return [np.asarray(row["candidates"], dtype=np.float32) for row in rows]
    return [local[index, -width:].reshape(row["candidates"].shape)
            for index, row in enumerate(rows)]


def _flatten(observations: dict[str, dict], hierarchical: bool,
             record: Callable[[str, float], None] | None = None):
    start = perf_counter() if record is not None else 0.0
    agents, rows, local, global_state, masks = _local(observations)
    uniform_float32 = _uniform_float32(rows)
    factors = _factors(rows, masks, _candidate_views(rows, local, uniform_float32))
    if record is not None:
        record("graph", perf_counter() - start)
    if not hierarchical:
        return agents, np.concatenate([local, factors], axis=1), global_state, masks
    start = perf_counter() if record is not None else 0.0
    # These matrices were already concatenated in local. Float32 input is the
    # execution fast path; unusual input dtypes use the original direct casts.
    if uniform_float32:
        self_width = rows[0]["self"].size
        neighbor_width = rows[0]["neighbors"].size
        action_width = rows[0]["neighbor_action_mean"].size
        action_start = self_width + neighbor_width
        self_features = local[:, :self_width]
        action_means = local[:, action_start:action_start + action_width]
        opportunities = np.any(masks[:, 2:], axis=1).astype(np.float32)
        weak = _weak(self_features, action_means, opportunities, 6, 12)
    else:
        weak = weak_mean_field_features(observations)
    if record is not None:
        record("weak", perf_counter() - start)
    # One final concatenation instead of a graph concat followed by another
    # concat for the twelve weak-interaction outputs. No arithmetic changes.
    actor_obs = np.concatenate([local, factors, weak], axis=1)
    return agents, actor_obs, global_state, masks


def flatten_opportunity_graph_all(observations: dict[str, dict]):
    return _flatten(observations, False)


def flatten_hierarchical_coalition_all(observations: dict[str, dict]):
    return _flatten(observations, True)


@contextmanager
def install_features(evaluate_module: Any, *, profile: bool = True):
    """Temporarily install feature builders while constructing a policy.

    ``evaluate_marl`` captures the selected builder in its graph-policy closure,
    so the returned policy continues using it after original globals restore.
    This process-local helper is intended for one policy in a worker, not for
    simultaneous policy construction from multiple threads.
    """
    counters: dict[str, float | int] = {
        "feature_calls": 0, "feature_seconds": 0.0,
        "graph_calls": 0, "graph_seconds": 0.0,
        "weak_calls": 0, "weak_seconds": 0.0,
        "policy_calls": 0, "policy_seconds": 0.0,
    }

    def record(name: str, seconds: float) -> None:
        counters[name + "_calls"] += 1
        counters[name + "_seconds"] += seconds

    def make_flatten(hierarchical: bool):
        def flatten(current):
            start = perf_counter() if profile else 0.0
            result = _flatten(current, hierarchical, record if profile else None)
            if profile:
                record("feature", perf_counter() - start)
            return result
        return flatten

    old_graph = evaluate_module.flatten_opportunity_graph_all
    old_hierarchical = evaluate_module.flatten_hierarchical_coalition_all
    evaluate_module.flatten_opportunity_graph_all = make_flatten(False)
    evaluate_module.flatten_hierarchical_coalition_all = make_flatten(True)
    try:
        yield counters
    finally:
        evaluate_module.flatten_opportunity_graph_all = old_graph
        evaluate_module.flatten_hierarchical_coalition_all = old_hierarchical


def build_policy(evaluate_module: Any, checkpoint: dict, metrics: dict,
                 observations: dict, device: Any, *, profile: bool = True):
    """Build the unmodified policy with exact-order accelerated CPU features.

    Returns the usual ``(algorithm_id, choose)`` pair. ``choose`` preserves
    ``reset_strategy`` and exposes a JSON-serializable cumulative snapshot as
    ``choose.acceleration_profile()``. Policy time includes feature time;
    graph/weak are components of feature time, not additional wall time.
    """
    with install_features(evaluate_module, profile=profile) as counters:
        algo, original_choose = evaluate_module.build_policy(
            checkpoint, metrics, observations, device
        )

    def choose(current):
        if not profile:
            return original_choose(current)
        start = perf_counter()
        try:
            return original_choose(current)
        finally:
            counters["policy_calls"] += 1
            counters["policy_seconds"] += perf_counter() - start

    if hasattr(original_choose, "reset_strategy"):
        choose.reset_strategy = original_choose.reset_strategy
    choose.acceleration_profile = lambda: dict(counters)
    return algo, choose
