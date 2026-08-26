from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime
from typing import Any

import numpy as np

from .area_coverage import (
    best_strip_center,
    clip_polygon_to_target,
    low_discrepancy_rectangle_samples,
    oriented_strip_polygon,
    polygon_area,
    strip_sample_mask,
    uncovered_centroid,
)
from .config import EnvConfig
from .entities import (
    DataPacketState,
    GroundStationState,
    OrbitalElements,
    SatelliteState,
    TaskReservationState,
    TaskState,
)
from .orbital import (
    ecef_to_spherical,
    geodetic_to_ecef,
    propagate_kepler,
    satellite_is_sunlit,
    sun_elevation_deg,
    sun_unit_ecef,
    target_attitude_ecef,
    target_attitude_lvlh,
    target_geometry,
)
from .real_scenario import EphemerisCache, load_ephemeris_cache, load_task_catalog

try:
    from gymnasium import spaces
except Exception:  # pragma: no cover - gymnasium is optional at import time.
    spaces = None


ActionValue = int | Mapping[str, float | int]
Plan = dict[str, Any]
Claim = tuple[str, float, float, Plan]
DownlinkRequest = tuple[str, int, float]


class SatTaskingEnv:
    """Parallel MARL environment with dynamic orbit geometry and local coupling.

    Orbit positions are propagated from six Keplerian elements with a two-body
    model. Earth rotation, line-of-sight occultation, target and station
    elevation, sunlight, three-axis slew time, payload timing, compressed data,
    segmented downlink, station capacity, and cooperative tasks are enforced.
    """

    self_dim = 24
    task_dim = 24
    neighbor_dim = 13
    neighbor_action_dim = 4
    global_dim = 12

    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self._ephemeris: EphemerisCache | None = None
        if self.config.ephemeris_cache_path:
            self._ephemeris = load_ephemeris_cache(
                self.config.ephemeris_cache_path,
                expected_satellites=self.config.num_satellites,
                required_steps=(
                    self.config.max_steps + self.config.planning_lookahead_steps
                ),
                expected_step_duration_seconds=self.config.step_duration_seconds,
            )
            start_utc = self._ephemeris.metadata.get("start_utc")
            if start_utc:
                start = datetime.fromisoformat(str(start_utc).replace("Z", "+00:00"))
                self.config.epoch_day_of_year = float(start.timetuple().tm_yday)
                self.config.epoch_utc_hour = (
                    start.hour
                    + start.minute / 60.0
                    + (start.second + start.microsecond / 1e6) / 3600.0
                )
        self._task_catalog = (
            load_task_catalog(self.config.task_catalog_path)
            if self.config.task_catalog_path
            else None
        )
        if (
            self._task_catalog is not None
            and len(self._task_catalog) < self.config.num_tasks
        ):
            raise ValueError(
                f"Task catalog has {len(self._task_catalog)} rows, but "
                f"num_tasks={self.config.num_tasks}."
            )
        self.possible_agents = [
            f"satellite_{idx}" for idx in range(self.config.num_satellites)
        ]
        self.agents = list(self.possible_agents)
        self._rng = np.random.default_rng(self.config.random_seed)
        self._step = 0
        self._satellites: list[SatelliteState] = []
        self._tasks: list[TaskState] = []
        self._ground_stations: list[GroundStationState] = []
        self._last_candidate_ids: dict[str, np.ndarray] = {}
        self._last_action_masks: dict[str, np.ndarray] = {}
        self._plan_cache: dict[tuple[int, int, int], tuple[bool, str, Plan]] = {}
        self._decision_count = 0
        self._decision_opportunities = 0
        self._task_decision_opportunities = 0
        self._downlink_only_opportunities = 0
        self._forced_idle_actions = 0
        self._avoidable_idle_actions = 0
        self._sat_geometry: list[dict[str, Any]] = []
        self._task_ecef = np.empty((0, 3), dtype=np.float64)
        self._task_sun_elevation = np.empty(0, dtype=np.float64)
        self._station_ecef = np.empty((0, 3), dtype=np.float64)
        self._station_cache: list[list[tuple[int, float]]] = []
        self._candidate_sets: list[set[int]] = []
        self._station_sets: list[set[int]] = []
        self._local_neighbor_ids: list[set[int]] = []
        self._area_samples: dict[int, np.ndarray] = {}
        self._last_ground_utilization = 0.0
        self._task_reward_total = 0.0
        self._downlink_reward_total = 0.0
        self._team_reward_total = 0.0
        self._reservations: dict[int, TaskReservationState] = {}
        self._satellite_reservations: dict[int, int] = {}
        self._next_reservation_version = 1
        self._reservation_created_count = 0
        self._reservation_committed_count = 0
        self._reservation_expired_count = 0
        self._reservation_member_waste = 0
        self._decision_count = 0
        self._decision_opportunities = 0
        self._task_decision_opportunities = 0
        self._downlink_only_opportunities = 0
        self._forced_idle_actions = 0
        self._avoidable_idle_actions = 0

    @property
    def step_count(self) -> int:
        return self._step

    @property
    def satellites(self) -> tuple[SatelliteState, ...]:
        return tuple(self._satellites)

    @property
    def tasks(self) -> tuple[TaskState, ...]:
        return tuple(self._tasks)

    @property
    def ground_stations(self) -> tuple[GroundStationState, ...]:
        return tuple(self._ground_stations)

    def reset(
        self, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        elif self.config.random_seed is not None:
            self._rng = np.random.default_rng(self.config.random_seed)

        self._step = 0
        self.agents = list(self.possible_agents)
        self._satellites = self._build_satellites()
        self._tasks = self._build_tasks()
        self._derive_area_requirements()
        self._ground_stations = self._build_ground_stations()
        self._last_ground_utilization = 0.0
        self._task_reward_total = 0.0
        self._downlink_reward_total = 0.0
        self._team_reward_total = 0.0
        self._reservations = {}
        self._satellite_reservations = {}
        self._next_reservation_version = 1
        self._reservation_created_count = 0
        self._reservation_committed_count = 0
        self._reservation_expired_count = 0
        self._reservation_member_waste = 0
        self._decision_count = 0
        self._decision_opportunities = 0
        self._task_decision_opportunities = 0
        self._downlink_only_opportunities = 0
        self._forced_idle_actions = 0
        self._avoidable_idle_actions = 0
        self._expire_tasks()
        observations = self._observe_all()
        infos = {agent: {"agent_index": idx} for idx, agent in enumerate(self.agents)}
        return observations, infos

    def step(
        self, actions: Mapping[str, ActionValue]
    ) -> tuple[
        dict[str, dict[str, np.ndarray]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        if not self._satellites:
            raise RuntimeError("Call reset() before step().")

        self._expire_tasks()
        self._expire_collaboration_progress()
        self._expire_reservations()
        rewards = {agent: 0.0 for agent in self.agents}
        infos: dict[str, dict[str, Any]] = {agent: {} for agent in self.agents}
        claims: dict[int, list[Claim]] = defaultdict(list)
        downlink_requests: dict[int, list[DownlinkRequest]] = defaultdict(list)

        for agent in self.agents:
            sat_idx = self._agent_index(agent)
            sat = self._satellites[sat_idx]
            action, bid = self._parse_action(actions.get(agent, 0))
            sat.last_action_category = self._action_category(action)
            previous_mask = self._last_action_masks.get(agent)
            has_opportunity = bool(
                previous_mask is not None and np.any(previous_mask[1:])
            )
            has_task_opportunity = bool(
                previous_mask is not None and np.any(previous_mask[2:])
            )
            downlink_only_opportunity = bool(
                has_opportunity and not has_task_opportunity
            )
            self._decision_count += 1
            self._decision_opportunities += int(has_opportunity)
            self._task_decision_opportunities += int(has_task_opportunity)
            self._downlink_only_opportunities += int(downlink_only_opportunity)

            if action == 0:
                if self._satellite_available(sat) and has_opportunity:
                    rewards[agent] += self.config.idle_penalty
                    self._avoidable_idle_actions += 1
                    infos[agent]["event"] = "avoidable_idle"
                else:
                    rewards[agent] += self.config.forced_idle_penalty
                    self._forced_idle_actions += 1
                    infos[agent]["event"] = (
                        "busy" if not self._satellite_available(sat) else "forced_idle"
                    )
                continue

            if not self._satellite_available(sat):
                self._reject_action(
                    agent, sat, rewards, infos, "satellite_busy", window_miss=False
                )
                continue

            if action == 1:
                station = self._best_visible_station(sat_idx)
                if sat.storage <= 1e-8:
                    self._reject_action(
                        agent, sat, rewards, infos, "no_data", window_miss=False
                    )
                elif station is None:
                    self._reject_action(
                        agent, sat, rewards, infos, "no_ground_station", window_miss=True
                    )
                else:
                    station_id, elevation = station
                    downlink_requests[station_id].append((agent, station_id, elevation))
                    infos[agent].update(
                        {"event": "downlink_requested", "station_id": station_id}
                    )
                continue

            task_id = self._action_to_task_id(agent, action)
            if task_id is None:
                self._reject_action(
                    agent, sat, rewards, infos, "invalid_action", window_miss=False
                )
                continue

            task = self._tasks[task_id]
            reserved_task_id = self._satellite_reservations.get(sat.sat_id)
            if reserved_task_id is not None and reserved_task_id != task_id:
                self._reject_action(
                    agent,
                    sat,
                    rewards,
                    infos,
                    "reservation_conflict",
                    window_miss=False,
                )
                continue
            feasible, reason, plan = self._plan_task(sat, task)
            if not feasible:
                self._reject_action(
                    agent,
                    sat,
                    rewards,
                    infos,
                    reason,
                    window_miss=reason
                    in {
                        "before_release",
                        "deadline_missed",
                        "earth_occulted",
                        "below_min_elevation",
                        "insufficient_sunlight",
                        "slew_window_overrun",
                    },
                )
                continue

            if bid is None:
                bid = self._default_bid(sat, task, plan)
            claims[task_id].append(
                (agent, float(bid), float(plan["off_nadir_deg"]), plan)
            )
            infos[agent].update(
                {
                    "event": "claimed",
                    "task_id": task_id,
                    "maneuver_steps": int(plan["maneuver_steps"]),
                    "observation_start_step": int(plan["observation_start_step"]),
                    "quality": float(plan["quality"]),
                }
            )

        self._resolve_downlinks(downlink_requests, rewards, infos)
        completed_values = self._resolve_task_claims(claims, rewards, infos)

        if completed_values:
            team_bonus = (
                self.config.team_reward_weight
                * float(np.sum(completed_values))
                / max(1, len(self.agents))
            )
            for agent in self.agents:
                rewards[agent] += team_bonus
                infos[agent]["team_bonus"] = team_bonus
            self._team_reward_total += team_bonus * len(self.agents)

        self._advance_dynamics()
        self._step += 1
        self._expire_tasks()
        self._expire_collaboration_progress()
        self._expire_reservations(rewards, infos)

        terminated = self._all_tasks_finished()
        truncated = self._step >= self.config.max_steps
        observations = self._observe_all()
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        for agent in self.agents:
            infos[agent]["done"] = terminated or truncated
        return observations, rewards, terminations, truncations, infos

    def action_space(self, agent: str | None = None):
        if spaces is None:
            return None
        return spaces.Discrete(self.config.candidate_k + 2)

    def observation_space(self, agent: str | None = None):
        if spaces is None:
            return None
        cfg = self.config
        return spaces.Dict(
            {
                "self": spaces.Box(-1.0, 1.0, shape=(self.self_dim,), dtype=np.float32),
                "candidates": spaces.Box(
                    -1.0, 1.0, shape=(cfg.candidate_k, self.task_dim), dtype=np.float32
                ),
                "candidate_ids": spaces.Box(
                    -1, cfg.num_tasks, shape=(cfg.candidate_k,), dtype=np.int32
                ),
                "neighbors": spaces.Box(
                    -1.0,
                    1.0,
                    shape=(cfg.neighbor_k, self.neighbor_dim),
                    dtype=np.float32,
                ),
                "neighbor_action_mean": spaces.Box(
                    0.0, 1.0, shape=(self.neighbor_action_dim,), dtype=np.float32
                ),
                "global": spaces.Box(
                    0.0, 1.0, shape=(self.global_dim,), dtype=np.float32
                ),
                "action_mask": spaces.MultiBinary(cfg.candidate_k + 2),
            }
        )

    def render(self) -> str:
        summary = self.summary()
        return (
            f"step={self._step} completed={summary['completed_tasks']} "
            f"expired={summary['expired_tasks']} available={summary['available_tasks']} "
            f"busy={summary['busy_satellites']} packets={summary['pending_packets']}"
        )

    def summary(self) -> dict[str, Any]:
        completed = [task for task in self._tasks if task.completed_by is not None]
        expired = [task for task in self._tasks if task.expired]
        area_tasks = [task for task in self._tasks if task.is_area]
        area_completed = [
            task for task in area_tasks if task.completed_by is not None
        ]
        total_inside = float(
            sum(task.inside_imaged_area_km2 for task in area_tasks)
        )
        total_outside = float(
            sum(task.outside_imaged_area_km2 for task in area_tasks)
        )
        total_redundant = float(
            sum(task.redundant_imaged_area_km2 for task in area_tasks)
        )
        qualities = [
            quality for task in completed for quality in task.observation_qualities
        ]
        return {
            "config": asdict(self.config),
            "scenario": self._scenario_metadata(),
            "step": self._step,
            "completed_tasks": len(completed),
            "cooperative_completed_tasks": sum(
                task.required_observers > 1 for task in completed
            ),
            "area_tasks": len(area_tasks),
            "area_completed_tasks": len(area_completed),
            "area_cooperative_completed_tasks": sum(
                task.min_contributing_satellites > 1 for task in area_completed
            ),
            "mean_area_coverage": float(
                np.mean([task.coverage_fraction for task in area_tasks])
            )
            if area_tasks
            else 0.0,
            "priority_weighted_area_coverage": (
                float(
                    sum(task.priority * task.coverage_fraction for task in area_tasks)
                    / max(1e-9, sum(task.priority for task in area_tasks))
                )
                if area_tasks
                else 0.0
            ),
            "area_strip_count": int(sum(task.strip_count for task in area_tasks)),
            "area_inside_imaged_km2": total_inside,
            "area_outside_imaged_km2": total_outside,
            "area_redundant_imaged_km2": total_redundant,
            "area_outside_ratio": total_outside
            / max(1e-9, total_inside + total_outside),
            "area_redundancy_ratio": total_redundant / max(1e-9, total_inside),
            "expired_tasks": len(expired),
            "available_tasks": sum(task.available for task in self._tasks),
            "active_tasks": sum(self._task_active(task) for task in self._tasks),
            "total_priority_completed": float(sum(task.priority for task in completed)),
            "mean_observation_quality": float(np.mean(qualities)) if qualities else 0.0,
            "mean_energy": float(np.mean([sat.energy for sat in self._satellites])),
            "mean_storage": float(np.mean([sat.storage for sat in self._satellites])),
            "pending_packets": int(sum(len(sat.packets) for sat in self._satellites)),
            "pending_data_mb": float(sum(sat.storage for sat in self._satellites)),
            "downlinked_data_mb": float(
                sum(station.delivered_mb for station in self._ground_stations)
            ),
            "busy_satellites": int(
                sum(not self._satellite_available(sat) for sat in self._satellites)
            ),
            "total_conflicts": int(sum(sat.conflict_count for sat in self._satellites)),
            "total_ground_conflicts": int(
                sum(sat.ground_conflict_count for sat in self._satellites)
            ),
            "total_invalid_actions": int(sum(sat.invalid_count for sat in self._satellites)),
            "total_window_misses": int(
                sum(sat.window_miss_count for sat in self._satellites)
            ),
            "ground_station_utilization": float(self._last_ground_utilization),
            "task_reward_total": float(self._task_reward_total),
            "downlink_reward_total": float(self._downlink_reward_total),
            "team_reward_total": float(self._team_reward_total),
            "active_reservations": len(self._reservations),
            "reserved_satellites": len(self._satellite_reservations),
            "reservation_created": self._reservation_created_count,
            "reservation_committed": self._reservation_committed_count,
            "reservation_expired": self._reservation_expired_count,
            "reservation_member_waste": self._reservation_member_waste,
            "reservation_commit_rate": self._reservation_committed_count
            / max(1, self._reservation_created_count),
            "mean_reservation_fill": float(
                np.mean(
                    [
                        len(reservation.member_bids)
                        / max(1, self._tasks[task_id].required_observers)
                        for task_id, reservation in self._reservations.items()
                    ]
                )
            )
            if self._reservations
            else 0.0,
            "decision_count": self._decision_count,
            "decision_opportunities": self._decision_opportunities,
            "decision_opportunity_rate": self._decision_opportunities
            / max(1, self._decision_count),
            "task_decision_opportunities": self._task_decision_opportunities,
            "task_decision_opportunity_rate": self._task_decision_opportunities
            / max(1, self._decision_count),
            "downlink_only_opportunities": self._downlink_only_opportunities,
            "forced_idle_actions": self._forced_idle_actions,
            "avoidable_idle_actions": self._avoidable_idle_actions,
        }

    def _scenario_metadata(self) -> dict[str, Any]:
        target_catalog = self.config.task_catalog_path
        return {
            "orbit_source": (
                self._ephemeris.summary()
                if self._ephemeris is not None
                else {"mode": "synthetic_two_body"}
            ),
            "task_source": {
                "mode": "external_catalog" if target_catalog else "synthetic",
                "path": str(target_catalog) if target_catalog else None,
                "rows": len(self._task_catalog) if self._task_catalog is not None else None,
            },
        }

    def _propagate_satellite(
        self, sat: SatelliteState, step: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._ephemeris is not None:
            return self._ephemeris.state(step, sat.sat_id)
        return propagate_kepler(
            sat.orbital_elements,
            step * self.config.step_duration_seconds,
            self.config.earth_mu_km3_s2,
            self.config.earth_rotation_rate_rad_s,
        )

    def _build_satellites(self) -> list[SatelliteState]:
        cfg = self.config
        satellites: list[SatelliteState] = []
        sats_per_plane = max(1, int(math.ceil(cfg.num_satellites / cfg.num_planes)))
        plane_altitudes = self._rng.uniform(
            cfg.orbit_altitude_min_km, cfg.orbit_altitude_max_km, cfg.num_planes
        )
        plane_eccentricities = self._rng.uniform(
            0.0, cfg.orbit_eccentricity_max, cfg.num_planes
        )
        plane_inclinations = self._rng.uniform(
            cfg.orbit_inclination_min_deg,
            cfg.orbit_inclination_max_deg,
            cfg.num_planes,
        )
        plane_argp = self._rng.uniform(0.0, 360.0, cfg.num_planes)

        for sat_id in range(cfg.num_satellites):
            plane_id = (
                int(self._ephemeris.plane_ids[sat_id])
                if self._ephemeris is not None
                and self._ephemeris.plane_ids is not None
                else sat_id % cfg.num_planes
            )
            slot = sat_id // cfg.num_planes
            mean_anomaly = (360.0 * slot / sats_per_plane) % 360.0
            external_elements = (
                self._ephemeris.orbital_elements[sat_id]
                if self._ephemeris is not None
                and self._ephemeris.orbital_elements is not None
                else None
            )
            if external_elements is not None:
                elements = OrbitalElements(
                    semi_major_axis_km=float(
                        external_elements["semi_major_axis_km"]
                    ),
                    eccentricity=float(external_elements["eccentricity"]),
                    inclination_deg=float(external_elements["inclination_deg"]),
                    raan_deg=float(external_elements["raan_deg"]),
                    argument_of_perigee_deg=float(
                        external_elements["argument_of_perigee_deg"]
                    ),
                    mean_anomaly_deg=float(
                        external_elements["mean_anomaly_deg"]
                    ),
                )
                mean_anomaly = elements.mean_anomaly_deg
                semi_major_axis = elements.semi_major_axis_km
            else:
                semi_major_axis = (
                    cfg.earth_radius_km + float(plane_altitudes[plane_id])
                )
                elements = OrbitalElements(
                    semi_major_axis_km=semi_major_axis,
                    eccentricity=float(plane_eccentricities[plane_id]),
                    inclination_deg=float(plane_inclinations[plane_id]),
                    raan_deg=float(360.0 * plane_id / cfg.num_planes),
                    argument_of_perigee_deg=float(plane_argp[plane_id]),
                    mean_anomaly_deg=float(mean_anomaly),
                )
            mean_motion = math.sqrt(cfg.earth_mu_km3_s2 / semi_major_axis**3)
            orbit_rate = mean_motion * cfg.step_duration_seconds / (2.0 * math.pi)
            selector = sat_id % 10
            base_resolution = float(
                self._rng.uniform(
                    cfg.payload_resolution_min_m, cfg.payload_resolution_max_m
                )
            )
            base_swath = float(
                self._rng.uniform(cfg.payload_swath_min_km, cfg.payload_swath_max_km)
            )
            if selector < 6:
                mode = "optical"
                resolution = max(0.2, 0.55 * base_resolution)
                swath = 0.7 * base_swath
            elif selector < 9:
                mode = "sar"
                resolution = base_resolution
                swath = base_swath
            else:
                mode = "infrared"
                resolution = min(40.0, 2.2 * base_resolution)
                swath = 1.3 * base_swath
            satellites.append(
                SatelliteState(
                    sat_id=sat_id,
                    plane_id=plane_id,
                    orbital_elements=elements,
                    phase=mean_anomaly / 360.0,
                    orbit_rate=float(orbit_rate),
                    energy=float(cfg.max_energy * self._rng.uniform(0.7, 1.0)),
                    storage=0.0,
                    payload_mode=mode,
                    swath_width_km=float(swath),
                    field_of_view_deg=float(
                        self._rng.uniform(cfg.payload_fov_min_deg, cfg.payload_fov_max_deg)
                    ),
                    resolution_m=float(resolution),
                    roll_deg=float(self._rng.uniform(-2.0, 2.0)),
                    pitch_deg=float(self._rng.uniform(-2.0, 2.0)),
                    yaw_deg=float(self._rng.uniform(-1.0, 1.0)),
                )
            )
        return satellites

    def _sample_random_task_schedule(self, window: int) -> tuple[int, int]:
        cfg = self.config
        latest_release = max(0, cfg.max_steps - window - 1)
        release = int(self._rng.integers(0, latest_release + 1)) if latest_release else 0
        deadline = min(cfg.max_steps - 1, release + window)
        return release, deadline

    def _sample_global_task_point(self) -> tuple[float, float]:
        latitude = math.degrees(math.asin(float(self._rng.uniform(-0.9, 0.9))))
        longitude = float(self._rng.uniform(-180.0, 180.0))
        return latitude, longitude

    def _wrap_longitude(self, longitude_deg: float) -> float:
        return float(((longitude_deg + 180.0) % 360.0) - 180.0)

    def _sample_task_mode(self, anchor_mode: str | None) -> str:
        cfg = self.config
        if (
            anchor_mode is not None
            and self._rng.random() < cfg.curriculum_payload_match_probability
        ):
            return anchor_mode
        mode_roll = self._rng.random()
        return "optical" if mode_roll < 0.6 else "sar" if mode_roll < 0.9 else "infrared"

    def _use_curriculum_task_layout(self) -> bool:
        cfg = self.config
        layout = cfg.task_layout.lower()
        if layout in {"curriculum", "curriculum_visible", "visible_curriculum"}:
            return True
        if layout in {"mixed", "mixed_curriculum"}:
            return self._rng.random() < cfg.curriculum_visible_fraction
        return False

    def _schedule_around_anchor(self, window: int, anchor_step: int) -> tuple[int, int]:
        cfg = self.config
        latest_release = max(0, cfg.max_steps - window - 1)
        low = max(0, anchor_step - window + 1)
        high = min(anchor_step, latest_release)
        if high < low:
            return self._sample_random_task_schedule(window)
        release = int(self._rng.integers(low, high + 1))
        deadline = min(cfg.max_steps - 1, release + window)
        return release, deadline

    def _sample_curriculum_task(
        self, window: int
    ) -> tuple[int, int, float, float, str | None]:
        cfg = self.config
        if not self._satellites:
            release, deadline = self._sample_random_task_schedule(window)
            latitude, longitude = self._sample_global_task_point()
            return release, deadline, latitude, longitude, None

        last_sample: tuple[int, int, float, float, str | None] | None = None
        attempts = max(8, min(32, len(self._satellites)))
        for _ in range(attempts):
            sat = self._satellites[int(self._rng.integers(0, len(self._satellites)))]
            anchor_step = int(self._rng.integers(0, cfg.max_steps))
            time_jitter = max(0, cfg.curriculum_time_jitter_steps)
            if time_jitter:
                anchor_step += int(self._rng.integers(-time_jitter, time_jitter + 1))
                anchor_step = int(np.clip(anchor_step, 0, cfg.max_steps - 1))

            elapsed = anchor_step * cfg.step_duration_seconds
            _, _, sat_ecef = self._propagate_satellite(sat, anchor_step)
            latitude, longitude, _ = ecef_to_spherical(sat_ecef, cfg.earth_radius_km)
            jitter_deg = max(0.0, cfg.curriculum_ground_track_jitter_deg)
            if jitter_deg > 0.0:
                radius = jitter_deg * math.sqrt(float(self._rng.random()))
                theta = float(self._rng.uniform(0.0, 2.0 * math.pi))
                latitude += radius * math.sin(theta)
                longitude += radius * math.cos(theta) / max(
                    math.cos(math.radians(latitude)), 0.2
                )
            latitude = float(np.clip(latitude, -85.0, 85.0))
            longitude = self._wrap_longitude(longitude)
            release, deadline = self._schedule_around_anchor(window, anchor_step)
            sample = (release, deadline, latitude, longitude, sat.payload_mode)
            last_sample = sample

            if sat.payload_mode != "optical":
                return sample
            target_ecef = geodetic_to_ecef(latitude, longitude, cfg.earth_radius_km)
            day = cfg.epoch_day_of_year + elapsed / 86400.0
            utc_hour = (cfg.epoch_utc_hour + elapsed / 3600.0) % 24.0
            if (
                sun_elevation_deg(target_ecef, sun_unit_ecef(day, utc_hour))
                >= cfg.optical_min_sun_elevation_deg
            ):
                return sample

        if last_sample is not None:
            return last_sample
        release, deadline = self._sample_random_task_schedule(window)
        latitude, longitude = self._sample_global_task_point()
        return release, deadline, latitude, longitude, None

    def _sample_task_definition(
        self, window: int
    ) -> tuple[int, int, float, float, str | None]:
        if self._use_curriculum_task_layout():
            return self._sample_curriculum_task(window)
        release, deadline = self._sample_random_task_schedule(window)
        latitude, longitude = self._sample_global_task_point()
        return release, deadline, latitude, longitude, None

    def _build_tasks(self) -> list[TaskState]:
        if self._task_catalog is not None:
            return self._build_catalog_tasks()

        cfg = self.config
        tasks: list[TaskState] = []
        min_window = max(1, cfg.min_task_window)
        max_window = max(min_window, cfg.max_task_window)
        for task_id in range(cfg.num_tasks):
            window = int(self._rng.integers(min_window, max_window + 1))
            release, deadline, latitude, longitude, anchor_mode = (
                self._sample_task_definition(window)
            )
            mode = self._sample_task_mode(anchor_mode)
            if mode == "optical":
                required_resolution = self._rng.uniform(2.0, 10.0)
                required_swath = self._rng.uniform(10.0, 65.0)
                min_sun = cfg.optical_min_sun_elevation_deg
            elif mode == "sar":
                required_resolution = self._rng.uniform(8.0, 22.0)
                required_swath = self._rng.uniform(25.0, 110.0)
                min_sun = -90.0
            else:
                required_resolution = self._rng.uniform(18.0, 40.0)
                required_swath = self._rng.uniform(45.0, 145.0)
                min_sun = -90.0

            is_area = self._rng.random() < cfg.area_task_fraction
            cooperation_mode = "coverage" if is_area else "single"
            required_observers = 1
            if not is_area and self._rng.random() < cfg.cooperative_task_probability:
                cooperation_mode = (
                    "simultaneous"
                    if self._rng.random() < cfg.simultaneous_task_fraction
                    else "sequential"
                )
                required_observers = int(
                    self._rng.integers(
                        cfg.cooperative_observers_min,
                        cfg.cooperative_observers_max + 1,
                    )
                )

            original_data = float(
                self._rng.uniform(cfg.original_data_min_mb, cfg.original_data_max_mb)
            )
            compression = float(
                self._rng.uniform(cfg.compression_ratio_min, cfg.compression_ratio_max)
            )
            area_width = (
                float(self._rng.uniform(cfg.area_width_min_km, cfg.area_width_max_km))
                if is_area
                else 0.0
            )
            area_height = (
                float(
                    self._rng.uniform(cfg.area_height_min_km, cfg.area_height_max_km)
                )
                if is_area
                else 0.0
            )
            tasks.append(
                TaskState(
                    task_id=task_id,
                    target_phase=(longitude + 180.0) / 360.0,
                    target_lat_deg=latitude,
                    target_lon_deg=longitude,
                    priority=float(
                        self._rng.uniform(cfg.task_priority_min, cfg.task_priority_max)
                    ),
                    release_step=release,
                    deadline_step=deadline,
                    energy_cost=float(cfg.base_energy_cost * self._rng.uniform(0.75, 1.4)),
                    duration=int(
                        self._rng.integers(cfg.task_duration_min, cfg.task_duration_max + 1)
                    ),
                    observation_duration_seconds=float(
                        cfg.area_observation_seconds
                        if is_area
                        else cfg.point_observation_seconds
                    ),
                    required_mode=mode,
                    required_resolution_m=float(required_resolution),
                    required_swath_km=float(required_swath),
                    min_sun_elevation_deg=float(min_sun),
                    original_data_mb=original_data,
                    compression_ratio=compression,
                    cooperation_mode=cooperation_mode,
                    required_observers=required_observers,
                    max_coordination_gap_steps=cfg.sequential_max_gap_steps,
                    target_type="area" if is_area else "point",
                    area_width_km=area_width,
                    area_height_km=area_height,
                    area_orientation_deg=(
                        float(self._rng.uniform(0.0, 360.0)) if is_area else 0.0
                    ),
                    coverage_threshold=(
                        cfg.area_coverage_threshold if is_area else 1.0
                    ),
                )
            )
        return tasks

    @staticmethod
    def _catalog_value(
        row: dict[str, str], key: str, default: float | int | str
    ) -> float | int | str:
        value = row.get(key, "")
        return default if value == "" else value

    def _build_catalog_tasks(self) -> list[TaskState]:
        cfg = self.config
        assert self._task_catalog is not None
        tasks: list[TaskState] = []
        min_window = max(1, cfg.min_task_window)
        max_window = max(min_window, cfg.max_task_window)

        for task_id, row in enumerate(self._task_catalog[: cfg.num_tasks]):
            latitude = float(row["latitude_deg"])
            longitude = self._wrap_longitude(float(row["longitude_deg"]))
            if not -90.0 <= latitude <= 90.0:
                raise ValueError(
                    f"Task catalog row {task_id + 2} has invalid latitude {latitude}."
                )

            window = int(self._rng.integers(min_window, max_window + 1))
            sampled_release, sampled_deadline = self._sample_random_task_schedule(window)
            release = int(self._catalog_value(row, "release_step", sampled_release))
            deadline = int(self._catalog_value(row, "deadline_step", sampled_deadline))
            if not (0 <= release <= deadline < cfg.max_steps):
                raise ValueError(
                    f"Task catalog row {task_id + 2} has invalid release/deadline "
                    f"{release}/{deadline} for max_steps={cfg.max_steps}."
                )

            mode = str(
                self._catalog_value(row, "required_mode", self._sample_task_mode(None))
            ).lower()
            if mode not in {"optical", "sar", "infrared"}:
                raise ValueError(
                    f"Task catalog row {task_id + 2} has unsupported mode {mode!r}."
                )
            if mode == "optical":
                default_resolution = float(self._rng.uniform(2.0, 10.0))
                default_swath = float(self._rng.uniform(10.0, 65.0))
                default_min_sun = cfg.optical_min_sun_elevation_deg
            elif mode == "sar":
                default_resolution = float(self._rng.uniform(8.0, 22.0))
                default_swath = float(self._rng.uniform(25.0, 110.0))
                default_min_sun = -90.0
            else:
                default_resolution = float(self._rng.uniform(18.0, 40.0))
                default_swath = float(self._rng.uniform(45.0, 145.0))
                default_min_sun = -90.0

            target_type = str(
                self._catalog_value(row, "target_type", "point")
            ).lower()
            if target_type not in {"point", "area"}:
                raise ValueError(
                    f"Task catalog row {task_id + 2} has unsupported "
                    f"target_type {target_type!r}."
                )
            cooperation_mode = str(
                self._catalog_value(row, "cooperation_mode", "")
            ).lower()
            if target_type == "area":
                cooperation_mode = "coverage"
            elif cooperation_mode == "":
                if self._rng.random() < cfg.cooperative_task_probability:
                    cooperation_mode = (
                        "simultaneous"
                        if self._rng.random() < cfg.simultaneous_task_fraction
                        else "sequential"
                    )
                else:
                    cooperation_mode = "single"
            if cooperation_mode not in {
                "single",
                "simultaneous",
                "sequential",
                "coverage",
                "auto",
            }:
                raise ValueError(
                    f"Task catalog row {task_id + 2} has unsupported "
                    f"cooperation_mode {cooperation_mode!r}."
                )
            default_observers = (
                1
                if cooperation_mode == "single"
                else int(
                    self._rng.integers(
                        cfg.cooperative_observers_min,
                        cfg.cooperative_observers_max + 1,
                    )
                )
            )
            required_observers = int(
                self._catalog_value(row, "required_observers", default_observers)
            )
            if cooperation_mode == "single":
                required_observers = 1
            if target_type == "area":
                required_observers = 1

            area_width = float(
                self._catalog_value(row, "area_width_km", 0.0)
            )
            area_height = float(
                self._catalog_value(row, "area_height_km", 0.0)
            )
            if target_type == "area" and (
                area_width <= 0.0 or area_height <= 0.0
            ):
                raise ValueError(
                    f"Task catalog row {task_id + 2} is an area request but "
                    "area_width_km/area_height_km are not positive."
                )
            coverage_threshold = float(
                self._catalog_value(
                    row, "coverage_threshold", cfg.area_coverage_threshold
                )
            )
            if target_type == "area" and not 0.0 < coverage_threshold <= 1.0:
                raise ValueError(
                    f"Task catalog row {task_id + 2} has invalid "
                    f"coverage_threshold {coverage_threshold}."
                )

            original_data = float(
                self._catalog_value(
                    row,
                    "original_data_mb",
                    float(
                        self._rng.uniform(
                            cfg.original_data_min_mb, cfg.original_data_max_mb
                        )
                    ),
                )
            )
            compression = float(
                self._catalog_value(
                    row,
                    "compression_ratio",
                    float(
                        self._rng.uniform(
                            cfg.compression_ratio_min, cfg.compression_ratio_max
                        )
                    ),
                )
            )
            tasks.append(
                TaskState(
                    task_id=task_id,
                    target_phase=(longitude + 180.0) / 360.0,
                    target_lat_deg=latitude,
                    target_lon_deg=longitude,
                    priority=float(
                        self._catalog_value(
                            row,
                            "priority",
                            float(
                                self._rng.uniform(
                                    cfg.task_priority_min, cfg.task_priority_max
                                )
                            ),
                        )
                    ),
                    release_step=release,
                    deadline_step=deadline,
                    energy_cost=float(
                        self._catalog_value(
                            row,
                            "energy_cost",
                            cfg.base_energy_cost * float(self._rng.uniform(0.75, 1.4)),
                        )
                    ),
                    duration=int(self._catalog_value(row, "duration", 1)),
                    observation_duration_seconds=float(
                        self._catalog_value(
                            row,
                            "observation_duration_seconds",
                            cfg.area_observation_seconds
                            if target_type == "area"
                            else cfg.point_observation_seconds,
                        )
                    ),
                    required_mode=mode,
                    required_resolution_m=float(
                        self._catalog_value(
                            row, "required_resolution_m", default_resolution
                        )
                    ),
                    required_swath_km=float(
                        self._catalog_value(row, "required_swath_km", default_swath)
                    ),
                    min_sun_elevation_deg=float(
                        self._catalog_value(
                            row, "min_sun_elevation_deg", default_min_sun
                        )
                    ),
                    original_data_mb=original_data,
                    compression_ratio=compression,
                    cooperation_mode=cooperation_mode,
                    required_observers=required_observers,
                    max_coordination_gap_steps=int(
                        self._catalog_value(
                            row,
                            "max_coordination_gap_steps",
                            cfg.sequential_max_gap_steps,
                        )
                    ),
                    target_type=target_type,
                    area_width_km=area_width,
                    area_height_km=area_height,
                    area_orientation_deg=float(
                        self._catalog_value(row, "area_orientation_deg", 0.0)
                    )
                    % 360.0,
                    coverage_threshold=(
                        coverage_threshold if target_type == "area" else 1.0
                    ),
                )
            )
        return tasks

    def _derive_area_requirements(self) -> None:
        """Derive cooperation from target area and compatible payload capacity."""

        cfg = self.config
        self._area_samples = {}
        for task in self._tasks:
            if not task.is_area:
                continue
            task.coverage_sample_count = max(64, int(cfg.area_coverage_samples))
            self._area_samples[task.task_id] = low_discrepancy_rectangle_samples(
                task.area_width_km,
                task.area_height_km,
                task.coverage_sample_count,
            )
            compatible = [
                sat
                for sat in self._satellites
                if sat.payload_mode == task.required_mode
                and sat.resolution_m <= task.required_resolution_m
            ]
            best_single_fraction = 0.0
            for sat in compatible:
                orbital_speed = math.sqrt(
                    cfg.earth_mu_km3_s2
                    / sat.orbital_elements.semi_major_axis_km
                )
                ground_speed = (
                    orbital_speed
                    * cfg.earth_radius_km
                    / sat.orbital_elements.semi_major_axis_km
                )
                strip_length = ground_speed * task.observation_duration_seconds
                strip_width = sat.swath_width_km
                aligned_area = max(
                    min(task.area_width_km, strip_length)
                    * min(task.area_height_km, strip_width),
                    min(task.area_width_km, strip_width)
                    * min(task.area_height_km, strip_length),
                )
                best_single_fraction = max(
                    best_single_fraction,
                    aligned_area / max(task.area_km2, 1e-9),
                )
            capacity = max(best_single_fraction, 1e-6)
            task.required_strips = min(
                cfg.area_max_required_strips,
                max(1, int(math.ceil(task.coverage_threshold / capacity))),
            )
            if task.required_strips == 1:
                task.cooperation_mode = "single"
                task.min_contributing_satellites = 1
            else:
                task.cooperation_mode = "coverage"
                task.min_contributing_satellites = min(
                    task.required_strips,
                    max(
                        1,
                        min(
                            len(compatible),
                            cfg.area_min_cooperative_satellites,
                        ),
                    ),
                )
            task.required_observers = task.min_contributing_satellites

    def _build_ground_stations(self) -> list[GroundStationState]:
        sites = [
            ("Kashgar", 39.47, 75.99),
            ("Sanya", 18.31, 109.31),
            ("Kiruna", 67.86, 20.23),
            ("Perth", -31.95, 115.86),
            ("Santiago", -33.45, -70.67),
            ("Alaska", 64.84, -147.72),
            ("Hartebeesthoek", -25.89, 27.69),
            ("Kourou", 5.24, -52.77),
        ]
        stations = []
        for station_id in range(self.config.num_ground_stations):
            name, lat, lon = sites[station_id % len(sites)]
            stations.append(
                GroundStationState(
                    station_id=station_id,
                    name=name,
                    latitude_deg=lat,
                    longitude_deg=lon,
                    min_elevation_deg=self.config.min_ground_elevation_deg,
                    channel_capacity=self.config.ground_station_channels,
                    rate_mb_per_step=self.config.ground_station_rate_mb_per_step,
                )
            )
        return stations

    def _observe_all(self) -> dict[str, dict[str, np.ndarray]]:
        self._refresh_geometry()
        self._plan_cache = {}
        candidate_cache = self._candidate_cache()
        self._prepare_local_neighborhoods(candidate_cache)
        global_features = self._global_features()
        observations: dict[str, dict[str, np.ndarray]] = {}
        self._last_candidate_ids = {}
        self._last_action_masks = {}
        for agent in self.agents:
            sat_idx = self._agent_index(agent)
            observation = self._observe_agent(sat_idx, candidate_cache, global_features)
            observations[agent] = observation
            self._last_candidate_ids[agent] = observation["candidate_ids"].copy()
            self._last_action_masks[agent] = observation["action_mask"].copy()
        return observations

    def _observe_agent(
        self,
        sat_idx: int,
        candidate_cache: list[np.ndarray],
        global_features: np.ndarray,
    ) -> dict[str, np.ndarray]:
        cfg = self.config
        sat = self._satellites[sat_idx]
        candidate_ids = candidate_cache[sat_idx]
        candidate_features = np.zeros((cfg.candidate_k, self.task_dim), dtype=np.float32)
        padded_ids = np.full(cfg.candidate_k, -1, dtype=np.int32)
        action_mask = np.zeros(cfg.candidate_k + 2, dtype=np.int8)
        action_mask[0] = 1
        reserved_task_id = self._satellite_reservations.get(sat_idx)
        if self._satellite_available(sat) and reserved_task_id is None:
            action_mask[1] = int(sat.storage > 1e-8 and bool(self._station_cache[sat_idx]))

        for row, task_id in enumerate(candidate_ids[: cfg.candidate_k]):
            task = self._tasks[int(task_id)]
            padded_ids[row] = task.task_id
            feasible, _, plan = self._plan_task(sat, task)
            candidate_features[row] = self._task_features(sat, task, plan)
            reservation_compatible = (
                reserved_task_id is None or int(task_id) == reserved_task_id
            )
            action_mask[row + 2] = int(feasible and reservation_compatible)

        neighbors, action_mean = self._neighbor_features(sat_idx, candidate_cache)
        return {
            "self": self._self_features(sat),
            "candidates": candidate_features,
            "candidate_ids": padded_ids,
            "neighbors": neighbors,
            "neighbor_action_mean": action_mean,
            "global": global_features,
            "action_mask": action_mask,
        }

    def _refresh_geometry(self) -> None:
        cfg = self.config
        elapsed = self._step * cfg.step_duration_seconds
        day = cfg.epoch_day_of_year + elapsed / 86400.0
        utc_hour = (cfg.epoch_utc_hour + elapsed / 3600.0) % 24.0
        sun_unit = sun_unit_ecef(day, utc_hour)
        self._sat_geometry = []
        for sat in self._satellites:
            position_eci, velocity_eci, position_ecef = self._propagate_satellite(
                sat, self._step
            )
            velocity_ecef = (
                self._ephemeris.ecef_velocity(self._step, sat.sat_id)
                if self._ephemeris is not None
                else self._ecef_velocity_from_eci(
                    position_ecef, velocity_eci, elapsed
                )
            )
            lat, lon, altitude = ecef_to_spherical(position_ecef, cfg.earth_radius_km)
            if self._ephemeris is not None:
                sat.phase = float(
                    (math.atan2(position_ecef[1], position_ecef[0]) / (2.0 * math.pi))
                    % 1.0
                )
            else:
                mean_motion = math.sqrt(
                    cfg.earth_mu_km3_s2 / sat.orbital_elements.semi_major_axis_km**3
                )
                mean_anomaly = (
                    math.radians(sat.orbital_elements.mean_anomaly_deg)
                    + mean_motion * elapsed
                )
                sat.phase = float((mean_anomaly / (2.0 * math.pi)) % 1.0)
            self._sat_geometry.append(
                {
                    "eci": position_eci,
                    "velocity_eci": velocity_eci,
                    "ecef": position_ecef,
                    "velocity_ecef": velocity_ecef,
                    "latitude_deg": lat,
                    "longitude_deg": lon,
                    "altitude_km": altitude,
                    "sunlit": satellite_is_sunlit(
                        position_ecef, sun_unit, cfg.earth_radius_km
                    ),
                }
            )

        self._task_ecef = np.asarray(
            [
                geodetic_to_ecef(
                    task.target_lat_deg, task.target_lon_deg, cfg.earth_radius_km
                )
                for task in self._tasks
            ],
            dtype=np.float64,
        )
        self._task_sun_elevation = np.asarray(
            [sun_elevation_deg(position, sun_unit) for position in self._task_ecef],
            dtype=np.float64,
        )
        self._station_ecef = np.asarray(
            [
                geodetic_to_ecef(
                    station.latitude_deg, station.longitude_deg, cfg.earth_radius_km
                )
                for station in self._ground_stations
            ],
            dtype=np.float64,
        )
        self._station_cache = []
        for sat_idx, sat_geo in enumerate(self._sat_geometry):
            visible: list[tuple[int, float]] = []
            for station in self._ground_stations:
                elevation, _, _, earth_visible = target_geometry(
                    sat_geo["ecef"], self._station_ecef[station.station_id]
                )
                if earth_visible and elevation >= station.min_elevation_deg:
                    visible.append((station.station_id, float(elevation)))
            visible.sort(key=lambda item: item[1], reverse=True)
            self._station_cache.append(visible)

    def _candidate_cache(self) -> list[np.ndarray]:
        cfg = self.config
        empty = np.empty(0, dtype=np.int32)
        active_ids = np.asarray(
            [
                task.task_id
                for task in self._tasks
                if task.available
                and task.deadline_step >= self._step
                and task.release_step <= self._step + cfg.planning_lookahead_steps
            ],
            dtype=np.int32,
        )
        if active_ids.size == 0:
            return [empty for _ in self._satellites]

        targets = self._task_ecef[active_ids]
        target_up = targets / np.maximum(np.linalg.norm(targets, axis=1, keepdims=True), 1e-9)
        mode_codes = np.asarray(
            [self._mode_index(self._tasks[int(task_id)].required_mode) for task_id in active_ids],
            dtype=np.int8,
        )
        required_resolution = np.asarray(
            [self._tasks[int(task_id)].required_resolution_m for task_id in active_ids],
            dtype=np.float64,
        )
        required_swath = np.asarray(
            [
                0.0
                if self._tasks[int(task_id)].is_area
                else self._tasks[int(task_id)].required_swath_km
                for task_id in active_ids
            ],
            dtype=np.float64,
        )
        area_requests = np.asarray(
            [self._tasks[int(task_id)].is_area for task_id in active_ids],
            dtype=bool,
        )
        min_sun_elevation = np.asarray(
            [self._tasks[int(task_id)].min_sun_elevation_deg for task_id in active_ids],
            dtype=np.float64,
        )
        priorities = np.asarray(
            [self._tasks[int(task_id)].priority for task_id in active_ids],
            dtype=np.float64,
        )
        deadlines = np.asarray(
            [self._tasks[int(task_id)].deadline_step for task_id in active_ids],
            dtype=np.float64,
        )
        releases = np.asarray(
            [self._tasks[int(task_id)].release_step for task_id in active_ids],
            dtype=np.float64,
        )
        active_positions = np.full(len(self._tasks), -1, dtype=np.int32)
        active_positions[active_ids] = np.arange(active_ids.size, dtype=np.int32)
        observed_by_satellite: list[list[int]] = [
            [] for _ in self._satellites
        ]
        for task in self._tasks:
            for observer in task.observed_by:
                observed_by_satellite[observer].append(task.task_id)
        cache: list[np.ndarray] = []
        for sat_idx, sat in enumerate(self._satellites):
            if not self._satellite_available(sat):
                cache.append(empty)
                continue
            compatible = mode_codes == self._mode_index(sat.payload_mode)
            unobserved = np.ones(active_ids.size, dtype=bool)
            for observed_task_id in observed_by_satellite[sat_idx]:
                position = active_positions[observed_task_id]
                if position >= 0 and not area_requests[position]:
                    unobserved[position] = False
            # A target visible from 500--650 km LEO must be within roughly 25
            # geocentric degrees of the sub-satellite point.  A conservative
            # 35-degree cap creates a lossless spatial shortlist before the
            # expensive attitude, payload, sunlight, and quality checks.
            offset_ecef: list[np.ndarray] = []
            for offset in range(cfg.planning_lookahead_steps + 1):
                planned_step = self._step + offset
                elapsed = planned_step * cfg.step_duration_seconds
                _, _, sat_ecef = self._propagate_satellite(sat, planned_step)
                offset_ecef.append(sat_ecef)
            subpoint_units = np.stack(
                [position / max(float(np.linalg.norm(position)), 1e-9) for position in offset_ecef]
            )
            central_cos = target_up @ subpoint_units.T
            spatially_reachable = np.max(central_cos, axis=1) >= math.cos(
                math.radians(35.0)
            )
            eligible = np.flatnonzero(compatible & unobserved & spatially_reachable)
            if eligible.size == 0:
                cache.append(empty)
                continue

            local_targets = targets[eligible]
            local_target_up = target_up[eligible]
            local_modes = mode_codes[eligible]
            local_resolution = required_resolution[eligible]
            local_swath = required_swath[eligible]
            local_min_sun = min_sun_elevation[eligible]
            local_priorities = priorities[eligible]
            local_deadlines = deadlines[eligible]
            local_releases = releases[eligible]
            best_scores = np.full(eligible.size, -np.inf, dtype=np.float64)
            for offset, sat_ecef in enumerate(offset_ecef):
                planned_step = self._step + offset
                elapsed = planned_step * cfg.step_duration_seconds
                los = local_targets - sat_ecef
                ranges = np.maximum(np.linalg.norm(los, axis=1), 1e-9)
                los_unit = los / ranges[:, None]
                elevation = np.degrees(
                    np.arcsin(np.clip(np.sum(-los_unit * local_target_up, axis=1), -1.0, 1.0))
                )
                nadir = -sat_ecef / max(float(np.linalg.norm(sat_ecef)), 1e-9)
                off_nadir = np.degrees(
                    np.arccos(np.clip(los_unit @ nadir, -1.0, 1.0))
                )
                cos_off = np.maximum(np.cos(np.radians(off_nadir)), 0.1)
                altitude = max(
                    1.0, float(np.linalg.norm(sat_ecef)) - cfg.earth_radius_km
                )
                effective_resolution = sat.resolution_m * ranges / altitude
                fov_footprint = 2.0 * ranges * math.tan(
                    math.radians(sat.field_of_view_deg / 2.0)
                )
                effective_swath = np.minimum(
                    sat.swath_width_km / cos_off, fov_footprint
                )
                day = cfg.epoch_day_of_year + elapsed / 86400.0
                utc_hour = (cfg.epoch_utc_hour + elapsed / 3600.0) % 24.0
                sun_unit = sun_unit_ecef(day, utc_hour)
                sun_elevations = np.degrees(
                    np.arcsin(np.clip(local_target_up @ sun_unit, -1.0, 1.0))
                )
                valid = (
                    (local_releases <= planned_step)
                    & (local_deadlines >= planned_step)
                    & (elevation >= cfg.min_observation_elevation_deg)
                    & (off_nadir <= cfg.max_off_nadir_deg)
                    & (effective_resolution <= local_resolution)
                    & (effective_swath >= local_swath)
                    & (
                        (local_modes != self._mode_index("optical"))
                        | (sun_elevations >= local_min_sun)
                    )
                )
                quality = np.power(cos_off, cfg.quality_off_nadir_power)
                urgency = 1.0 / (1.0 + np.maximum(0.0, local_deadlines - planned_step))
                scores = local_priorities * quality + urgency - 0.015 * off_nadir - 0.01 * offset
                best_scores[valid] = np.maximum(best_scores[valid], scores[valid])

            visible = np.flatnonzero(np.isfinite(best_scores))
            if visible.size == 0:
                cache.append(empty)
                continue

            scores = best_scores[visible]
            count = min(cfg.candidate_k, visible.size)
            if visible.size > count:
                local = np.argpartition(scores, -count)[-count:]
                local = local[np.argsort(scores[local])[::-1]]
            else:
                local = np.argsort(scores)[::-1]
            selected_ids = active_ids[eligible[visible[local]]].astype(
                np.int32, copy=False
            )
            reserved_task_id = self._satellite_reservations.get(sat_idx)
            if (
                reserved_task_id is not None
                and reserved_task_id not in selected_ids
                and self._tasks[reserved_task_id].available
            ):
                # Preserve the leased task in the bounded candidate list so a
                # size-invariant actor can observe reservation progress.
                selected_ids = np.concatenate(
                    [np.asarray([reserved_task_id], dtype=np.int32), selected_ids]
                )[: cfg.candidate_k]
            cache.append(selected_ids)
        for sat_idx, reserved_task_id in self._satellite_reservations.items():
            if sat_idx >= len(cache):
                continue
            task = self._tasks[reserved_task_id]
            if not task.available or reserved_task_id in cache[sat_idx]:
                continue
            feasible, _, _ = self._plan_task(self._satellites[sat_idx], task)
            if feasible:
                cache[sat_idx] = np.concatenate(
                    [
                        np.asarray([reserved_task_id], dtype=np.int32),
                        cache[sat_idx],
                    ]
                )[: cfg.candidate_k]
        return cache

    def _self_features(self, sat: SatelliteState) -> np.ndarray:
        cfg = self.config
        geo = self._sat_geometry[sat.sat_id]
        busy_remaining = max(0, sat.busy_until_step - self._step)
        cooldown_remaining = max(0, sat.cooldown_until_step - self._step)
        return np.asarray(
            [
                sat.phase,
                sat.plane_id / max(1, cfg.num_planes - 1),
                np.clip(geo["latitude_deg"] / 90.0, -1.0, 1.0),
                np.clip(geo["longitude_deg"] / 180.0, -1.0, 1.0),
                np.clip(geo["altitude_km"] / 1000.0, 0.0, 1.0),
                np.clip(sat.energy / cfg.max_energy, 0.0, 1.0),
                np.clip(sat.storage / cfg.max_storage, 0.0, 1.0),
                np.clip(sat.roll_deg / cfg.max_roll_deg, -1.0, 1.0),
                np.clip(sat.pitch_deg / cfg.max_pitch_deg, -1.0, 1.0),
                np.clip(sat.yaw_deg / cfg.max_yaw_deg, -1.0, 1.0),
                np.clip(sat.roll_rate_deg_s / cfg.max_angular_rate_deg_s, -1.0, 1.0),
                np.clip(sat.pitch_rate_deg_s / cfg.max_angular_rate_deg_s, -1.0, 1.0),
                np.clip(sat.yaw_rate_deg_s / cfg.max_angular_rate_deg_s, -1.0, 1.0),
                self._mode_code(sat.payload_mode),
                float(sat.payload_on),
                np.clip(busy_remaining / max(1, cfg.max_steps), 0.0, 1.0),
                np.clip(cooldown_remaining / max(1, cfg.max_steps), 0.0, 1.0),
                float(self._satellite_available(sat)),
                float(bool(self._station_cache[sat.sat_id])),
                np.clip(len(sat.packets) / 10.0, 0.0, 1.0),
                np.clip(sat.downlinked_mb / max(1.0, cfg.max_storage * 5.0), 0.0, 1.0),
                np.clip(sat.swath_width_km / 180.0, 0.0, 1.0),
                np.clip(sat.resolution_m / 40.0, 0.0, 1.0),
                float(geo["sunlit"]),
            ],
            dtype=np.float32,
        )

    def _task_features(
        self, sat: SatelliteState, task: TaskState, plan: Plan | None = None
    ) -> np.ndarray:
        cfg = self.config
        if plan is None:
            _, _, plan = self._plan_task(sat, task)
        observation_progress = (
            task.coverage_fraction
            if task.is_area
            else len(task.observed_by) / max(1, task.required_observers)
        )
        reservation = self._reservations.get(task.task_id)
        reservation_progress = (
            len(reservation.member_bids) / max(1, task.required_observers)
            if reservation is not None
            else 0.0
        )
        progress = max(observation_progress, reservation_progress)
        return np.asarray(
            [
                task.priority / cfg.task_priority_max,
                float(self._task_active(task)),
                max(0, task.release_step - self._step) / max(1, cfg.max_steps),
                max(0, task.deadline_step - self._step) / max(1, cfg.max_steps),
                np.clip(task.target_lat_deg / 90.0, -1.0, 1.0),
                np.clip(task.target_lon_deg / 180.0, -1.0, 1.0),
                self._mode_code(task.required_mode),
                np.clip(task.required_resolution_m / 40.0, 0.0, 1.0),
                np.clip(
                    (
                        task.area_width_km / 300.0
                        if task.is_area
                        else task.required_swath_km / 180.0
                    ),
                    0.0,
                    1.0,
                ),
                np.clip(task.original_data_mb / max(1.0, cfg.original_data_max_mb), 0.0, 1.0),
                np.clip(task.compression_ratio, 0.0, 1.0),
                np.clip(task.compressed_data_mb / cfg.max_storage, 0.0, 1.0),
                np.clip(
                    (
                        task.area_height_km / 300.0
                        if task.is_area
                        else task.duration / max(1, cfg.task_duration_max)
                    ),
                    0.0,
                    1.0,
                ),
                self._cooperation_code(task.cooperation_mode),
                task.required_observers
                / max(1, cfg.cooperative_observers_max, cfg.area_max_required_strips),
                np.clip(progress, 0.0, 1.0),
                np.clip(float(plan.get("elevation_deg", -90.0)) / 90.0, -1.0, 1.0),
                np.clip(float(plan.get("off_nadir_deg", 90.0)) / cfg.max_off_nadir_deg, 0.0, 1.0),
                np.clip(float(plan.get("sun_elevation_deg", -90.0)) / 90.0, -1.0, 1.0),
                np.clip(float(plan.get("quality", 0.0)), 0.0, 1.0),
                np.clip(float(plan.get("effective_resolution_m", 40.0)) / 40.0, 0.0, 1.0),
                np.clip(
                    (
                        float(plan.get("observation_offset_steps", cfg.max_steps))
                        + float(plan.get("maneuver_steps", 0))
                    )
                    / max(1, cfg.max_steps),
                    0.0,
                    1.0,
                ),
                np.clip(float(plan.get("total_energy", cfg.max_energy)) / cfg.max_energy, 0.0, 1.0),
                task.task_id / max(1, cfg.num_tasks - 1),
            ],
            dtype=np.float32,
        )

    def _neighbor_features(
        self, sat_idx: int, candidate_cache: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        features = np.zeros((cfg.neighbor_k, self.neighbor_dim), dtype=np.float32)
        action_mean = np.zeros(self.neighbor_action_dim, dtype=np.float32)
        if len(self._satellites) <= 1 or cfg.neighbor_k <= 0:
            return features, action_mean

        sat = self._satellites[sat_idx]
        own_tasks = self._candidate_sets[sat_idx]
        own_stations = self._station_sets[sat_idx]
        ranked: list[tuple[float, int, bool, bool, bool]] = []
        for other_idx in self._local_neighbor_ids[sat_idx]:
            other = self._satellites[other_idx]
            other_tasks = self._candidate_sets[other_idx]
            other_stations = self._station_sets[other_idx]
            same_plane = other.plane_id == sat.plane_id
            task_competitor = bool(own_tasks & other_tasks)
            station_competitor = bool(own_stations & other_stations)
            if not (same_plane or task_competitor or station_competitor):
                continue
            distance = self._phase_distance(sat.phase, other.phase)
            score = -distance
            score += cfg.same_plane_neighbor_bonus * float(same_plane)
            score += cfg.task_competitor_neighbor_bonus * float(task_competitor)
            score += cfg.station_competitor_neighbor_bonus * float(station_competitor)
            ranked.append(
                (score, other_idx, same_plane, task_competitor, station_competitor)
            )
        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = ranked[: cfg.neighbor_k]
        if not selected:
            return features, action_mean

        categories = np.zeros(self.neighbor_action_dim, dtype=np.float32)
        for row, (_, other_idx, same_plane, task_competitor, station_competitor) in enumerate(selected):
            other = self._satellites[other_idx]
            other_tasks = self._candidate_sets[other_idx]
            other_stations = self._station_sets[other_idx]
            task_overlap = len(own_tasks & other_tasks) / max(1, len(own_tasks))
            station_overlap = len(own_stations & other_stations) / max(1, len(own_stations))
            features[row] = np.asarray(
                [
                    self._signed_phase_delta(sat.phase, other.phase),
                    float(same_plane),
                    float(task_competitor),
                    float(station_competitor),
                    np.clip(other.energy / cfg.max_energy, 0.0, 1.0),
                    np.clip(other.storage / cfg.max_storage, 0.0, 1.0),
                    np.clip(other.roll_deg / cfg.max_roll_deg, -1.0, 1.0),
                    np.clip(other.pitch_deg / cfg.max_pitch_deg, -1.0, 1.0),
                    float(not self._satellite_available(other)),
                    float(task_overlap),
                    other.last_action_category / max(1, self.neighbor_action_dim - 1),
                    float(station_overlap),
                    np.clip(other.completed_count / 10.0, 0.0, 1.0),
                ],
                dtype=np.float32,
            )
            categories[other.last_action_category] += 1.0
        action_mean = categories / len(selected)
        return features, action_mean

    def _prepare_local_neighborhoods(
        self, candidate_cache: list[np.ndarray]
    ) -> None:
        """Build sparse relation edges once per step instead of scanning N squared."""

        plane_members: dict[int, list[int]] = defaultdict(list)
        task_members: dict[int, list[int]] = defaultdict(list)
        station_members: dict[int, list[int]] = defaultdict(list)
        self._candidate_sets = [
            set(int(value) for value in candidates) for candidates in candidate_cache
        ]
        self._station_sets = [
            {station_id for station_id, _ in visible}
            for visible in self._station_cache
        ]
        for sat_idx, sat in enumerate(self._satellites):
            plane_members[sat.plane_id].append(sat_idx)
            for task_id in self._candidate_sets[sat_idx]:
                task_members[task_id].append(sat_idx)
            for station_id in self._station_sets[sat_idx]:
                station_members[station_id].append(sat_idx)

        self._local_neighbor_ids = []
        for sat_idx, sat in enumerate(self._satellites):
            related = set(plane_members[sat.plane_id])
            for task_id in self._candidate_sets[sat_idx]:
                related.update(task_members[task_id])
            for station_id in self._station_sets[sat_idx]:
                related.update(station_members[station_id])
            related.discard(sat_idx)
            self._local_neighbor_ids.append(related)

    def _global_features(self) -> np.ndarray:
        total_tasks = max(1, len(self._tasks))
        total_sats = max(1, len(self._satellites))
        completed = sum(task.completed_by is not None for task in self._tasks)
        expired = sum(task.expired for task in self._tasks)
        available = sum(task.available for task in self._tasks)
        active = sum(self._task_active(task) for task in self._tasks)
        urgent = sum(
            task.available and 0 <= task.deadline_step - self._step <= 3
            for task in self._tasks
        )
        collaborative_progress = sum(
            (
                task.coverage_fraction
                if task.is_area
                else len(task.observed_by) / max(1, task.required_observers)
            )
            for task in self._tasks
            if task.available and task.required_observers > 1
        )
        collaborative_count = sum(
            task.available and task.required_observers > 1 for task in self._tasks
        )
        return np.asarray(
            [
                completed / total_tasks,
                expired / total_tasks,
                available / total_tasks,
                active / total_tasks,
                np.clip(np.mean([sat.energy for sat in self._satellites]) / self.config.max_energy, 0.0, 1.0),
                np.clip(np.mean([sat.storage for sat in self._satellites]) / self.config.max_storage, 0.0, 1.0),
                sum(not self._satellite_available(sat) for sat in self._satellites) / total_sats,
                urgent / total_tasks,
                collaborative_progress / max(1, collaborative_count),
                np.clip(sum(len(sat.packets) for sat in self._satellites) / max(1, total_sats * 5), 0.0, 1.0),
                np.clip(self._last_ground_utilization, 0.0, 1.0),
                self._step / max(1, self.config.max_steps),
            ],
            dtype=np.float32,
        )

    def _plan_task(self, sat: SatelliteState, task: TaskState) -> tuple[bool, str, Plan]:
        key = (self._step, sat.sat_id, task.task_id)
        cached = self._plan_cache.get(key)
        if cached is not None:
            return cached

        cfg = self.config
        start_offset = max(0, task.release_step - self._step)
        end_offset = min(
            cfg.planning_lookahead_steps, task.deadline_step - self._step
        )
        fallback_offset = max(0, min(start_offset, cfg.planning_lookahead_steps))
        fallback_geometry = self._task_geometry(sat, task, fallback_offset)
        fallback: Plan = {
            **fallback_geometry,
            "slew_deg": 0.0,
            "slew_seconds": 0.0,
            "slew_steps": 0,
            "stabilization_steps": 0,
            "warmup_steps": 0,
            "maneuver_steps": 0,
            "observation_offset_steps": fallback_offset,
            "observation_start_step": self._step + fallback_offset,
            "finish_step": self._step + fallback_offset + task.duration - 1,
            "observation_duration_seconds": float(task.observation_duration_seconds),
            "observation_start_seconds": float(
                (self._step + fallback_offset) * cfg.step_duration_seconds
            ),
            "observation_finish_seconds": float(
                (self._step + fallback_offset) * cfg.step_duration_seconds
                + task.observation_duration_seconds
            ),
            "total_energy": task.energy_cost,
            "storage_required_mb": task.compressed_data_mb
            / max(1, task.required_observers),
        }

        if not self._satellite_available(sat):
            result = (False, "satellite_busy", fallback)
        elif not task.available:
            result = (False, "task_not_available", fallback)
        elif sat.sat_id in task.observed_by and not task.is_area:
            result = (False, "already_contributed", fallback)
        elif self._step > task.deadline_step:
            result = (False, "deadline_missed", fallback)
        elif start_offset > cfg.planning_lookahead_steps:
            result = (False, "before_release", fallback)
        elif sat.payload_mode != task.required_mode:
            result = (False, "payload_mode_mismatch", fallback)
        else:
            result = (False, "no_future_visibility", fallback)
            for offset in range(start_offset, max(start_offset, end_offset) + 1):
                geometry = self._task_geometry(sat, task, offset)
                target_roll = float(geometry["target_roll_deg"])
                target_pitch = float(geometry["target_pitch_deg"])
                target_yaw = float(geometry["target_yaw_deg"])
                roll_seconds = self._axis_slew_seconds(
                    target_roll - sat.roll_deg, sat.roll_rate_deg_s
                )
                pitch_seconds = self._axis_slew_seconds(
                    target_pitch - sat.pitch_deg, sat.pitch_rate_deg_s
                )
                yaw_seconds = self._axis_slew_seconds(
                    target_yaw - sat.yaw_deg, sat.yaw_rate_deg_s
                )
                slew_seconds = max(roll_seconds, pitch_seconds, yaw_seconds)
                slew_steps = int(math.ceil(slew_seconds / cfg.step_duration_seconds))
                stabilization_steps = int(
                    math.ceil(cfg.stabilization_seconds / cfg.step_duration_seconds)
                )
                warmup_steps = 0 if sat.payload_on else int(
                    math.ceil(cfg.payload_warmup_seconds / cfg.step_duration_seconds)
                )
                maneuver_steps = slew_steps + stabilization_steps
                if offset < max(maneuver_steps, warmup_steps):
                    continue
                finish_step = self._step + offset + task.duration - 1
                slew_deg = (
                    abs(target_roll - sat.roll_deg)
                    + abs(target_pitch - sat.pitch_deg)
                    + abs(target_yaw - sat.yaw_deg)
                )
                observation_energy_scale = (
                    max(
                        1.0,
                        task.observation_duration_seconds
                        / max(cfg.point_observation_seconds, 1e-6),
                    )
                    if task.is_area
                    else 1.0
                )
                total_energy = (
                    task.energy_cost * observation_energy_scale
                    + slew_deg * cfg.slew_energy_cost_per_deg
                )
                if warmup_steps:
                    total_energy += cfg.payload_warmup_energy
                storage_required = task.compressed_data_mb / max(
                    1,
                    task.required_strips
                    if task.is_area
                    else task.required_observers,
                )
                plan: Plan = {
                    **geometry,
                    "slew_deg": float(slew_deg),
                    "slew_seconds": float(slew_seconds),
                    "slew_steps": slew_steps,
                    "stabilization_steps": stabilization_steps,
                    "warmup_steps": warmup_steps,
                    "maneuver_steps": maneuver_steps,
                    "observation_offset_steps": offset,
                    "observation_start_step": self._step + offset,
                    "finish_step": finish_step,
                    "observation_duration_seconds": float(
                        task.observation_duration_seconds
                    ),
                    "observation_start_seconds": float(
                        (self._step + offset) * cfg.step_duration_seconds
                    ),
                    "observation_finish_seconds": float(
                        (self._step + offset) * cfg.step_duration_seconds
                        + task.observation_duration_seconds
                    ),
                    "total_energy": float(total_energy),
                    "storage_required_mb": float(storage_required),
                }
                reason = "ok"
                if not bool(geometry["earth_visible"]):
                    reason = "earth_occulted"
                elif float(geometry["elevation_deg"]) < cfg.min_observation_elevation_deg:
                    reason = "below_min_elevation"
                elif float(geometry["off_nadir_deg"]) > cfg.max_off_nadir_deg:
                    reason = "off_nadir_limit"
                elif abs(target_roll) > cfg.max_roll_deg:
                    reason = "roll_limit"
                elif abs(target_pitch) > cfg.max_pitch_deg:
                    reason = "pitch_limit"
                elif abs(target_yaw) > cfg.max_yaw_deg:
                    reason = "yaw_limit"
                elif float(geometry["effective_resolution_m"]) > task.required_resolution_m:
                    reason = "resolution_limit"
                elif (
                    not task.is_area
                    and float(geometry["effective_swath_km"])
                    < task.required_swath_km
                ):
                    reason = "swath_limit"
                elif task.is_area and float(
                    geometry.get("marginal_coverage_fraction", 0.0)
                ) < cfg.area_min_marginal_coverage:
                    reason = "no_new_area_coverage"
                elif task.required_mode == "optical" and float(
                    geometry["sun_elevation_deg"]
                ) < task.min_sun_elevation_deg:
                    reason = "insufficient_sunlight"
                elif finish_step > task.deadline_step:
                    reason = "slew_window_overrun"
                elif sat.energy < total_energy:
                    reason = "insufficient_energy"
                elif sat.storage + storage_required > cfg.max_storage:
                    reason = "insufficient_storage"
                if reason == "ok":
                    result = (True, reason, plan)
                    break
                result = (False, reason, plan)

        self._plan_cache[key] = result
        return result

    def _task_geometry(
        self, sat: SatelliteState, task: TaskState, offset_steps: int = 0
    ) -> Plan:
        cfg = self.config
        planned_step = self._step + offset_steps
        elapsed = planned_step * cfg.step_duration_seconds
        if offset_steps == 0:
            sat_geo = self._sat_geometry[sat.sat_id]
        else:
            position_eci, velocity_eci, position_ecef = self._propagate_satellite(
                sat, planned_step
            )
            sat_geo = {
                "eci": position_eci,
                "velocity_eci": velocity_eci,
                "ecef": position_ecef,
                "velocity_ecef": (
                    self._ephemeris.ecef_velocity(planned_step, sat.sat_id)
                    if self._ephemeris is not None
                    else self._ecef_velocity_from_eci(
                        position_ecef, velocity_eci, elapsed
                    )
                ),
                "altitude_km": float(np.linalg.norm(position_ecef))
                - cfg.earth_radius_km,
            }
        center_u_km = 0.0
        center_v_km = 0.0
        target_lat_deg = task.target_lat_deg
        target_lon_deg = task.target_lon_deg
        if task.is_area:
            samples = self._area_samples[task.task_id]
            center_u_km, center_v_km = uncovered_centroid(
                samples, task.coverage_mask
            )
            target_lat_deg, target_lon_deg = self._task_local_to_geodetic(
                task, center_u_km, center_v_km
            )
            target_ecef = geodetic_to_ecef(
                target_lat_deg, target_lon_deg, cfg.earth_radius_km
            )
        else:
            target_ecef = self._task_ecef[task.task_id]
        elevation, off_nadir, slant_range, earth_visible = target_geometry(
            sat_geo["ecef"], target_ecef
        )
        if self._ephemeris is not None:
            target_roll, target_pitch, target_yaw = target_attitude_ecef(
                sat_geo["ecef"],
                self._ephemeris.ecef_velocity(planned_step, sat.sat_id),
                target_ecef,
            )
        else:
            target_roll, target_pitch, target_yaw = target_attitude_lvlh(
                sat_geo["eci"],
                sat_geo["velocity_eci"],
                sat_geo["ecef"],
                target_ecef,
                cfg.earth_rotation_rate_rad_s,
                elapsed,
            )
        cos_off = max(math.cos(math.radians(off_nadir)), 0.1)
        altitude = max(1.0, float(sat_geo["altitude_km"]))
        effective_resolution = sat.resolution_m * slant_range / altitude
        fov_footprint = 2.0 * slant_range * math.tan(
            math.radians(sat.field_of_view_deg / 2.0)
        )
        effective_swath = min(sat.swath_width_km / cos_off, fov_footprint)
        resolution_quality = min(1.0, task.required_resolution_m / max(effective_resolution, 1e-6))
        quality = max(0.0, cos_off) ** cfg.quality_off_nadir_power * resolution_quality
        day = cfg.epoch_day_of_year + elapsed / 86400.0
        utc_hour = (cfg.epoch_utc_hour + elapsed / 3600.0) % 24.0
        sun_elevation = (
            self._task_sun_elevation[task.task_id]
            if offset_steps == 0
            else sun_elevation_deg(target_ecef, sun_unit_ecef(day, utc_hour))
        )
        plan: Plan = {
            "elevation_deg": float(elevation),
            "off_nadir_deg": float(off_nadir),
            "slant_range_km": float(slant_range),
            "earth_visible": bool(earth_visible),
            "sun_elevation_deg": float(sun_elevation),
            "effective_resolution_m": float(effective_resolution),
            "effective_swath_km": float(effective_swath),
            "quality": float(np.clip(quality, 0.0, 1.0)),
            "target_roll_deg": float(target_roll),
            "target_pitch_deg": float(target_pitch),
            "target_yaw_deg": float(target_yaw),
        }
        if task.is_area:
            ground_heading, ground_speed = self._ground_track_heading_speed(
                sat_geo["ecef"],
                sat_geo["velocity_ecef"],
                task.target_lat_deg,
                task.target_lon_deg,
            )
            relative_heading = (
                ground_heading - task.area_orientation_deg + 180.0
            ) % 360.0 - 180.0
            strip_length = max(
                1e-6, ground_speed * task.observation_duration_seconds
            )
            strip_width = max(1e-6, effective_swath)
            optimized_u, optimized_v = best_strip_center(
                self._area_samples[task.task_id],
                task.coverage_mask,
                strip_length,
                strip_width,
                relative_heading,
                task.area_width_km,
                task.area_height_km,
            )
            if (
                abs(optimized_u - center_u_km) > 1e-6
                or abs(optimized_v - center_v_km) > 1e-6
            ):
                center_u_km, center_v_km = optimized_u, optimized_v
                target_lat_deg, target_lon_deg = self._task_local_to_geodetic(
                    task, center_u_km, center_v_km
                )
                target_ecef = geodetic_to_ecef(
                    target_lat_deg, target_lon_deg, cfg.earth_radius_km
                )
                elevation, off_nadir, slant_range, earth_visible = target_geometry(
                    sat_geo["ecef"], target_ecef
                )
                if self._ephemeris is not None:
                    target_roll, target_pitch, target_yaw = target_attitude_ecef(
                        sat_geo["ecef"],
                        sat_geo["velocity_ecef"],
                        target_ecef,
                    )
                else:
                    target_roll, target_pitch, target_yaw = target_attitude_lvlh(
                        sat_geo["eci"],
                        sat_geo["velocity_eci"],
                        sat_geo["ecef"],
                        target_ecef,
                        cfg.earth_rotation_rate_rad_s,
                        elapsed,
                    )
                cos_off = max(math.cos(math.radians(off_nadir)), 0.1)
                effective_resolution = sat.resolution_m * slant_range / altitude
                fov_footprint = 2.0 * slant_range * math.tan(
                    math.radians(sat.field_of_view_deg / 2.0)
                )
                effective_swath = min(
                    sat.swath_width_km / cos_off, fov_footprint
                )
                strip_width = max(1e-6, effective_swath)
                resolution_quality = min(
                    1.0,
                    task.required_resolution_m
                    / max(effective_resolution, 1e-6),
                )
                quality = (
                    max(0.0, cos_off) ** cfg.quality_off_nadir_power
                    * resolution_quality
                )
                sun_elevation = sun_elevation_deg(
                    target_ecef, sun_unit_ecef(day, utc_hour)
                )
                plan.update(
                    {
                        "elevation_deg": float(elevation),
                        "off_nadir_deg": float(off_nadir),
                        "slant_range_km": float(slant_range),
                        "earth_visible": bool(earth_visible),
                        "sun_elevation_deg": float(sun_elevation),
                        "effective_resolution_m": float(
                            effective_resolution
                        ),
                        "effective_swath_km": float(effective_swath),
                        "quality": float(np.clip(quality, 0.0, 1.0)),
                        "target_roll_deg": float(target_roll),
                        "target_pitch_deg": float(target_pitch),
                        "target_yaw_deg": float(target_yaw),
                    }
                )
            footprint = oriented_strip_polygon(
                center_u_km,
                center_v_km,
                strip_length,
                strip_width,
                relative_heading,
            )
            clipped = clip_polygon_to_target(
                footprint, task.area_width_km, task.area_height_km
            )
            strip_area = strip_length * strip_width
            inside_area = polygon_area(clipped)
            outside_area = max(0.0, strip_area - inside_area)
            sample_mask = strip_sample_mask(
                self._area_samples[task.task_id],
                center_u_km,
                center_v_km,
                strip_length,
                strip_width,
                relative_heading,
            )
            marginal_mask = sample_mask & ~task.coverage_mask
            marginal_fraction = (
                marginal_mask.bit_count() / task.coverage_sample_count
            )
            projected_fraction = min(
                1.0,
                (task.coverage_mask | sample_mask).bit_count()
                / task.coverage_sample_count,
            )
            new_inside_area = marginal_fraction * task.area_km2
            plan.update(
                {
                    "target_lat_deg": float(target_lat_deg),
                    "target_lon_deg": float(target_lon_deg),
                    "strip_center_u_km": float(center_u_km),
                    "strip_center_v_km": float(center_v_km),
                    "ground_track_heading_deg": float(ground_heading),
                    "strip_relative_heading_deg": float(relative_heading),
                    "strip_length_km": float(strip_length),
                    "strip_width_km": float(strip_width),
                    "strip_area_km2": float(strip_area),
                    "inside_area_km2": float(inside_area),
                    "outside_area_km2": float(outside_area),
                    "outside_ratio": float(
                        outside_area / max(strip_area, 1e-9)
                    ),
                    "redundant_area_km2": float(
                        max(0.0, inside_area - new_inside_area)
                    ),
                    "coverage_sample_mask": sample_mask,
                    "marginal_coverage_fraction": float(marginal_fraction),
                    "projected_coverage_fraction": float(projected_fraction),
                    "strip_footprint_local": footprint,
                }
            )
        return plan

    def _ecef_velocity_from_eci(
        self,
        position_ecef: np.ndarray,
        velocity_eci: np.ndarray,
        elapsed_seconds: float,
    ) -> np.ndarray:
        theta = self.config.earth_rotation_rate_rad_s * elapsed_seconds
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        rotation = np.asarray(
            [[cos_t, sin_t, 0.0], [-sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]]
        )
        inertial_velocity_ecef = rotation @ velocity_eci
        earth_rate = np.asarray(
            [0.0, 0.0, self.config.earth_rotation_rate_rad_s]
        )
        return inertial_velocity_ecef - np.cross(earth_rate, position_ecef)

    def _ground_track_heading_speed(
        self,
        position_ecef: np.ndarray,
        velocity_ecef: np.ndarray,
        latitude_deg: float,
        longitude_deg: float,
    ) -> tuple[float, float]:
        latitude = math.radians(latitude_deg)
        longitude = math.radians(longitude_deg)
        east = np.asarray([-math.sin(longitude), math.cos(longitude), 0.0])
        north = np.asarray(
            [
                -math.sin(latitude) * math.cos(longitude),
                -math.sin(latitude) * math.sin(longitude),
                math.cos(latitude),
            ]
        )
        radius_scale = self.config.earth_radius_km / max(
            float(np.linalg.norm(position_ecef)), 1e-9
        )
        east_speed = float(np.dot(velocity_ecef, east)) * radius_scale
        north_speed = float(np.dot(velocity_ecef, north)) * radius_scale
        heading = math.degrees(math.atan2(east_speed, north_speed)) % 360.0
        return heading, math.hypot(east_speed, north_speed)

    def _task_local_to_geodetic(
        self, task: TaskState, u_km: float, v_km: float
    ) -> tuple[float, float]:
        orientation = math.radians(task.area_orientation_deg)
        north_km = u_km * math.cos(orientation) - v_km * math.sin(
            orientation
        )
        east_km = u_km * math.sin(orientation) + v_km * math.cos(
            orientation
        )
        latitude = math.radians(task.target_lat_deg)
        target_latitude = task.target_lat_deg + math.degrees(
            north_km / self.config.earth_radius_km
        )
        target_longitude = task.target_lon_deg + math.degrees(
            east_km
            / (
                self.config.earth_radius_km
                * max(abs(math.cos(latitude)), 1e-4)
            )
        )
        return float(np.clip(target_latitude, -89.9999, 89.9999)), (
            self._wrap_longitude(target_longitude)
        )

    def _resolve_task_claims(
        self,
        claims: dict[int, list[Claim]],
        rewards: dict[str, float],
        infos: dict[str, dict[str, Any]],
    ) -> list[float]:
        completed_values: list[float] = []
        for task_id, contenders in claims.items():
            task = self._tasks[task_id]
            if not task.available:
                for agent, _, _, _ in contenders:
                    rewards[agent] += self.config.duplicate_penalty
                    infos[agent]["event"] = "duplicate_task"
                continue

            if task.is_area:
                ranked = sorted(
                    contenders,
                    key=lambda claim: (
                        claim[1],
                        float(
                            claim[3].get("marginal_coverage_fraction", 0.0)
                        ),
                        -float(claim[3].get("outside_ratio", 1.0)),
                    ),
                    reverse=True,
                )
                winners: list[Claim] = []
                projected_mask = task.coverage_mask
                for agent, bid, off_nadir, original_plan in ranked:
                    plan = dict(original_plan)
                    sample_mask = int(plan["coverage_sample_mask"])
                    marginal_mask = sample_mask & ~projected_mask
                    marginal_fraction = (
                        marginal_mask.bit_count() / task.coverage_sample_count
                    )
                    if marginal_fraction < self.config.area_min_marginal_coverage:
                        continue
                    plan["marginal_coverage_fraction"] = marginal_fraction
                    plan["projected_coverage_fraction"] = min(
                        1.0,
                        (projected_mask | sample_mask).bit_count()
                        / task.coverage_sample_count,
                    )
                    plan["redundant_area_km2"] = max(
                        0.0,
                        float(plan["inside_area_km2"])
                        - marginal_fraction * task.area_km2,
                    )
                    winners.append((agent, bid, off_nadir, plan))
                    projected_mask |= sample_mask

                winner_agents = {claim[0] for claim in winners}
                completed_now = False
                for claim in winners:
                    completed_now = self._commit_observation(
                        task, claim, rewards, infos
                    ) or completed_now
                if completed_now:
                    completed_values.append(task.priority)
                for agent, _, _, _ in contenders:
                    if agent in winner_agents:
                        continue
                    rewards[agent] += self.config.conflict_penalty
                    sat = self._satellites[self._agent_index(agent)]
                    sat.conflict_count += 1
                    infos[agent].update(
                        {
                            "event": "overlapping_strip_conflict",
                            "task_id": task_id,
                            "winner": ",".join(sorted(winner_agents)),
                        }
                    )
                continue

            if task.cooperation_mode == "simultaneous":
                if self.config.coalition_reservations:
                    self._reserve_claims(task, contenders, infos)
                    contenders = self._current_reservation_claims(task)
                winners = self._select_simultaneous_group(
                    contenders, task.required_observers
                )
                if not winners:
                    current_agents = {
                        claim[0] for claim in claims.get(task_id, [])
                    }
                    for agent in current_agents:
                        reservation = self._reservations.get(task_id)
                        if reservation is not None:
                            infos[agent].update(
                                {
                                    "event": "coalition_reserved",
                                    "task_id": task_id,
                                    "reservation_version": reservation.version,
                                    "reserved_members": len(
                                        reservation.member_bids
                                    ),
                                    "required_observers": task.required_observers,
                                    "reservation_expires_step": (
                                        reservation.expires_step
                                    ),
                                }
                            )
                            continue
                        rewards[agent] += (
                            self.config.coordination_failure_penalty
                        )
                        infos[agent].update(
                            {
                                "event": "coordination_failed",
                                "task_id": task_id,
                                "required_observers": task.required_observers,
                            }
                        )
                    continue
                if self.config.coalition_reservations:
                    self._reservation_committed_count += 1
                    self._release_reservation(task_id)
            else:
                winners = [self._select_winner(contenders)]

            winner_agents = {claim[0] for claim in winners}
            completed_now = False
            for claim in winners:
                completed_now = self._commit_observation(
                    task, claim, rewards, infos
                ) or completed_now
            if completed_now:
                completed_values.append(task.priority)

            for agent, _, _, _ in contenders:
                if agent in winner_agents:
                    continue
                rewards[agent] += self.config.conflict_penalty
                sat = self._satellites[self._agent_index(agent)]
                sat.conflict_count += 1
                infos[agent].update(
                    {
                        "event": "lost_conflict",
                        "task_id": task_id,
                        "winner": ",".join(sorted(winner_agents)),
                    }
                )
        return completed_values

    def _commit_observation(
        self,
        task: TaskState,
        claim: Claim,
        rewards: dict[str, float],
        infos: dict[str, dict[str, Any]],
    ) -> bool:
        agent, _, _, plan = claim
        sat = self._satellites[self._agent_index(agent)]
        finish_step = int(plan["finish_step"])
        sat.energy = max(0.0, sat.energy - float(plan["total_energy"]))
        storage_required = float(plan["storage_required_mb"])
        sat.storage += storage_required
        sat.target_roll_deg = float(plan["target_roll_deg"])
        sat.target_pitch_deg = float(plan["target_pitch_deg"])
        sat.target_yaw_deg = float(plan["target_yaw_deg"])
        sat.busy_until_step = finish_step + 1
        payload_cooldown_steps = int(
            math.ceil(
                self.config.payload_cooldown_seconds
                / self.config.step_duration_seconds
            )
        )
        sat.payload_cooldown_until_step = finish_step + 1 + payload_cooldown_steps
        sat.cooldown_until_step = (
            sat.payload_cooldown_until_step + self.config.post_task_cooldown_steps
        )
        sat.payload_on = True
        sat.payload_ready_step = self._step + int(plan["warmup_steps"])
        sat.last_payload_use_step = finish_step
        sat.last_task_id = task.task_id
        sat.completed_count += 1
        sat.packets.append(
            DataPacketState(
                task_id=task.task_id,
                generated_step=finish_step,
                original_size_mb=task.original_data_mb
                / max(
                    1,
                    task.required_strips
                    if task.is_area
                    else task.required_observers,
                ),
                compressed_size_mb=storage_required,
                remaining_mb=storage_required,
                priority=task.priority,
                source_satellite_id=sat.sat_id,
            )
        )

        if task.is_area:
            return self._commit_area_observation(
                task, claim, rewards, infos, storage_required
            )

        task.observed_by.append(sat.sat_id)
        task.observation_steps.append(finish_step)
        task.observation_qualities.append(float(plan["quality"]))
        completed_now = len(task.observed_by) >= task.required_observers
        if completed_now:
            task.completed_by_satellites = list(task.observed_by)
            task.completed_by = task.observed_by[0]
            task.completed_step = max(task.observation_steps)

        observation_reward = (
            self._task_reward(task, finish_step)
            * float(plan["quality"])
            / max(1, task.required_observers)
        )
        if task.required_observers > 1 and not completed_now:
            observation_reward *= self.config.incomplete_cooperative_reward_scale
        completion_bonus = self.config.task_completion_bonus if completed_now else 0.0
        if completed_now and task.required_observers > 1:
            completion_bonus += self.config.cooperative_completion_bonus
        maneuver_penalty = (
            self.config.maneuver_penalty_per_step * float(plan["maneuver_steps"])
        )
        task_reward = observation_reward + completion_bonus
        rewards[agent] += task_reward + maneuver_penalty
        self._task_reward_total += task_reward
        infos[agent].update(
            {
                "event": "completed_task" if completed_now else "cooperative_observation",
                "task_id": task.task_id,
                "task_reward": observation_reward,
                "completion_bonus": completion_bonus,
                "finish_step": finish_step,
                "observation_start_step": int(plan["observation_start_step"]),
                "total_energy": float(plan["total_energy"]),
                "compressed_data_mb": storage_required,
                "quality": float(plan["quality"]),
                "cooperation_mode": task.cooperation_mode,
                "observers": list(task.observed_by),
            }
        )
        return completed_now

    def _commit_area_observation(
        self,
        task: TaskState,
        claim: Claim,
        rewards: dict[str, float],
        infos: dict[str, dict[str, Any]],
        storage_required: float,
    ) -> bool:
        agent, _, _, plan = claim
        sat = self._satellites[self._agent_index(agent)]
        finish_step = int(plan["finish_step"])
        sample_mask = int(plan["coverage_sample_mask"])
        marginal_mask = sample_mask & ~task.coverage_mask
        marginal_fraction = (
            marginal_mask.bit_count() / task.coverage_sample_count
        )
        task.coverage_mask |= sample_mask
        task.coverage_fraction = min(
            1.0, task.coverage_mask.bit_count() / task.coverage_sample_count
        )
        task.observed_by.append(sat.sat_id)
        task.observation_steps.append(finish_step)
        task.observation_qualities.append(float(plan["quality"]))
        task.strip_count += 1
        task.inside_imaged_area_km2 += float(plan["inside_area_km2"])
        task.outside_imaged_area_km2 += float(plan["outside_area_km2"])
        redundant_area = max(
            0.0,
            float(plan["inside_area_km2"])
            - marginal_fraction * task.area_km2,
        )
        task.redundant_imaged_area_km2 += redundant_area
        task.strip_headings_deg.append(
            float(plan["ground_track_heading_deg"])
        )
        task.strip_footprints_local.append(
            [
                (float(point[0]), float(point[1]))
                for point in plan["strip_footprint_local"]
            ]
        )
        contributors = sorted(set(task.observed_by))
        completed_now = (
            task.coverage_fraction + 1e-12 >= task.coverage_threshold
            and len(contributors) >= task.min_contributing_satellites
        )
        if completed_now:
            task.completed_by_satellites = contributors
            task.completed_by = contributors[0]
            task.completed_step = max(task.observation_steps)

        timeliness_multiplier = self._task_reward(
            task, finish_step
        ) / max(task.priority, 1e-9)
        coverage_reward = (
            task.priority
            * marginal_fraction
            * self.config.area_coverage_reward_scale
            * float(plan["quality"])
            * timeliness_multiplier
        )
        target_area = max(task.area_km2, 1e-9)
        outside_penalty = (
            self.config.area_outside_penalty_weight
            * float(plan["outside_area_km2"])
            / target_area
        )
        redundancy_penalty = (
            self.config.area_redundancy_penalty_weight
            * redundant_area
            / target_area
        )
        completion_bonus = 0.0
        if completed_now:
            completion_bonus = (
                self.config.task_completion_bonus
                + self.config.area_completion_bonus
            )
            if task.min_contributing_satellites > 1:
                completion_bonus += self.config.cooperative_completion_bonus
        maneuver_penalty = (
            self.config.maneuver_penalty_per_step
            * float(plan["maneuver_steps"])
        )
        task_reward = (
            coverage_reward
            + completion_bonus
            - outside_penalty
            - redundancy_penalty
        )
        rewards[agent] += task_reward + maneuver_penalty
        self._task_reward_total += task_reward
        infos[agent].update(
            {
                "event": "area_completed" if completed_now else "area_strip",
                "task_id": task.task_id,
                "target_type": "area",
                "task_reward": coverage_reward,
                "completion_bonus": completion_bonus,
                "outside_penalty": outside_penalty,
                "redundancy_penalty": redundancy_penalty,
                "finish_step": finish_step,
                "observation_start_step": int(plan["observation_start_step"]),
                "total_energy": float(plan["total_energy"]),
                "compressed_data_mb": storage_required,
                "quality": float(plan["quality"]),
                "cooperation_mode": task.cooperation_mode,
                "observers": contributors,
                "strip_heading_deg": float(
                    plan["ground_track_heading_deg"]
                ),
                "strip_length_km": float(plan["strip_length_km"]),
                "strip_width_km": float(plan["strip_width_km"]),
                "marginal_coverage": marginal_fraction,
                "coverage_fraction": task.coverage_fraction,
                "coverage_threshold": task.coverage_threshold,
                "inside_area_km2": float(plan["inside_area_km2"]),
                "outside_area_km2": float(plan["outside_area_km2"]),
                "redundant_area_km2": redundant_area,
            }
        )
        return completed_now

    def _resolve_downlinks(
        self,
        requests: dict[int, list[DownlinkRequest]],
        rewards: dict[str, float],
        infos: dict[str, dict[str, Any]],
    ) -> None:
        used_channels = 0
        total_channels = sum(
            station.channel_capacity for station in self._ground_stations
        )
        for station_id, contenders in requests.items():
            station = self._ground_stations[station_id]
            contenders.sort(
                key=lambda request: self._downlink_priority(
                    self._satellites[self._agent_index(request[0])], request[2]
                ),
                reverse=True,
            )
            winners = contenders[: station.channel_capacity]
            used_channels += len(winners)
            winner_agents = {request[0] for request in winners}
            for agent, _, elevation in winners:
                sat = self._satellites[self._agent_index(agent)]
                elevation_factor = max(0.2, math.sin(math.radians(elevation)))
                capacity = station.rate_mb_per_step * elevation_factor
                amount, value = self._downlink_packets(sat, capacity)
                station.delivered_mb += amount
                sat.downlinked_mb += amount
                rewards[agent] += value
                self._downlink_reward_total += value
                infos[agent].update(
                    {
                        "event": "downlink",
                        "station_id": station_id,
                        "station_name": station.name,
                        "elevation_deg": elevation,
                        "downlinked": amount,
                        "timeliness_reward": value,
                    }
                )
            for agent, _, _ in contenders:
                if agent in winner_agents:
                    continue
                sat = self._satellites[self._agent_index(agent)]
                sat.ground_conflict_count += 1
                station.conflict_count += 1
                rewards[agent] += self.config.ground_conflict_penalty
                infos[agent].update(
                    {
                        "event": "ground_station_conflict",
                        "station_id": station_id,
                        "station_name": station.name,
                    }
                )
        self._last_ground_utilization = used_channels / max(1, total_channels)

    def _downlink_packets(
        self, sat: SatelliteState, capacity_mb: float
    ) -> tuple[float, float]:
        remaining_capacity = capacity_mb
        amount = 0.0
        reward = 0.0
        sat.packets.sort(
            key=lambda packet: (
                packet.generated_step,
                -packet.priority,
            )
        )
        for packet in sat.packets:
            if remaining_capacity <= 1e-8 or packet.generated_step > self._step:
                continue
            chunk = min(packet.remaining_mb, remaining_capacity)
            age = max(0, self._step - packet.generated_step)
            timeliness = math.exp(-age / max(1e-6, self.config.data_timeliness_tau_steps))
            value_factor = 1.0 + packet.priority / max(1.0, self.config.task_priority_max)
            reward += (
                chunk
                * self.config.downlink_reward_per_unit
                * timeliness
                * value_factor
            )
            packet.remaining_mb -= chunk
            packet.delivered_mb += chunk
            remaining_capacity -= chunk
            amount += chunk
        sat.packets = [packet for packet in sat.packets if packet.remaining_mb > 1e-6]
        sat.storage = float(sum(packet.remaining_mb for packet in sat.packets))
        return float(amount), float(reward)

    def _downlink_priority(self, sat: SatelliteState, elevation: float) -> float:
        if not sat.packets:
            return elevation / 90.0
        oldest_age = max(0, self._step - min(packet.generated_step for packet in sat.packets))
        max_priority = max(packet.priority for packet in sat.packets)
        return max_priority + 0.05 * oldest_age + elevation / 90.0

    def _default_bid(
        self, sat: SatelliteState, task: TaskState, plan: Plan | None = None
    ) -> float:
        if plan is None:
            _, _, plan = self._plan_task(sat, task)
        energy_margin = max(0.0, sat.energy - float(plan["total_energy"])) / self.config.max_energy
        storage_margin = max(
            0.0,
            self.config.max_storage - sat.storage - float(plan["storage_required_mb"]),
        ) / self.config.max_storage
        deadline_margin = max(0, task.deadline_step - int(plan["finish_step"])) / max(
            1, self.config.max_steps
        )
        area_value = 0.0
        if task.is_area:
            area_value = (
                4.0 * float(plan.get("marginal_coverage_fraction", 0.0))
                - 1.5 * float(plan.get("outside_ratio", 0.0))
                - float(plan.get("redundant_area_km2", 0.0))
                / max(task.area_km2, 1e-9)
            )
        return (
            task.priority * float(plan["quality"])
            + 0.8 * energy_margin
            + 0.4 * storage_margin
            + 0.2 * deadline_margin
            - 0.03 * float(plan["maneuver_steps"])
            + area_value
        )

    def _task_reward(self, task: TaskState, finish_step: int) -> float:
        time_left = max(0, task.deadline_step - finish_step)
        deadline_bonus = self.config.deadline_bonus_weight * (
            time_left / max(1, self.config.max_steps)
        )
        return float(task.priority * (1.0 + deadline_bonus))

    def _advance_dynamics(self) -> None:
        cfg = self.config
        next_step = self._step + 1
        for sat in self._satellites:
            if self._sat_geometry[sat.sat_id]["sunlit"]:
                sat.energy = min(cfg.max_energy, sat.energy + cfg.solar_charge_per_step)
            sat.roll_deg, sat.roll_rate_deg_s = self._advance_axis(
                sat.roll_deg, sat.roll_rate_deg_s, sat.target_roll_deg
            )
            sat.pitch_deg, sat.pitch_rate_deg_s = self._advance_axis(
                sat.pitch_deg, sat.pitch_rate_deg_s, sat.target_pitch_deg
            )
            sat.yaw_deg, sat.yaw_rate_deg_s = self._advance_axis(
                sat.yaw_deg, sat.yaw_rate_deg_s, sat.target_yaw_deg
            )
            if next_step >= sat.busy_until_step:
                sat.target_roll_deg = 0.0
                sat.target_pitch_deg = 0.0
                sat.target_yaw_deg = 0.0
            if (
                sat.payload_on
                and next_step >= sat.payload_cooldown_until_step
                and sat.last_payload_use_step >= 0
                and next_step - sat.last_payload_use_step
                > cfg.payload_idle_shutdown_steps
            ):
                sat.payload_on = False
            sat.energy = float(np.clip(sat.energy, 0.0, cfg.max_energy))
            sat.storage = float(np.clip(sat.storage, 0.0, cfg.max_storage))

    def _advance_axis(
        self, angle: float, angular_rate: float, target: float
    ) -> tuple[float, float]:
        cfg = self.config
        error = target - angle
        if abs(error) < 1e-5 and abs(angular_rate) < 1e-5:
            return float(target), 0.0
        desired_rate = math.copysign(
            min(
                cfg.max_angular_rate_deg_s,
                math.sqrt(max(0.0, 2.0 * cfg.max_angular_accel_deg_s2 * abs(error))),
            ),
            error,
        )
        max_rate_change = cfg.max_angular_accel_deg_s2 * cfg.step_duration_seconds
        rate_change = float(np.clip(desired_rate - angular_rate, -max_rate_change, max_rate_change))
        new_rate = float(
            np.clip(
                angular_rate + rate_change,
                -cfg.max_angular_rate_deg_s,
                cfg.max_angular_rate_deg_s,
            )
        )
        delta = new_rate * cfg.step_duration_seconds
        if abs(delta) >= abs(error):
            return float(target), 0.0
        return float(angle + delta), new_rate

    def _axis_slew_seconds(self, delta_deg: float, current_rate_deg_s: float) -> float:
        cfg = self.config
        distance = abs(delta_deg)
        if distance <= 1e-6 and abs(current_rate_deg_s) <= 1e-6:
            return 0.0
        accel = max(1e-6, cfg.max_angular_accel_deg_s2)
        max_rate = max(1e-6, cfg.max_angular_rate_deg_s)
        braking = abs(current_rate_deg_s) / accel
        accel_time = max_rate / accel
        accel_distance = max_rate * max_rate / accel
        if distance <= accel_distance:
            profile = 2.0 * math.sqrt(distance / accel)
        else:
            profile = 2.0 * accel_time + (distance - accel_distance) / max_rate
        return float(profile + braking)

    def _expire_tasks(self) -> None:
        for task in self._tasks:
            if task.completed_by is None and self._step > task.deadline_step:
                task.expired = True

    def _expire_collaboration_progress(self) -> None:
        for task in self._tasks:
            if (
                task.available
                and task.cooperation_mode == "sequential"
                and task.observation_steps
                and self._step - max(task.observation_steps) > task.max_coordination_gap_steps
            ):
                task.observed_by.clear()
                task.observation_steps.clear()
                task.observation_qualities.clear()

    def _reserve_claims(
        self,
        task: TaskState,
        contenders: list[Claim],
        infos: dict[str, dict[str, Any]],
    ) -> None:
        """Merge asynchronous proposals into one versioned task lease."""

        reservation = self._reservations.get(task.task_id)
        if reservation is None:
            reservation = TaskReservationState(
                task_id=task.task_id,
                version=self._next_reservation_version,
                created_step=self._step,
                expires_step=self._step + max(1, self.config.reservation_ttl_steps),
            )
            self._next_reservation_version += 1
            self._reservations[task.task_id] = reservation
            self._reservation_created_count += 1

        for agent, bid, _, _ in contenders:
            sat_id = self._agent_index(agent)
            existing_task = self._satellite_reservations.get(sat_id)
            if existing_task is not None and existing_task != task.task_id:
                infos[agent].update(
                    {
                        "event": "reservation_conflict",
                        "reserved_task_id": existing_task,
                    }
                )
                continue
            reservation.member_bids[sat_id] = max(
                float(bid), reservation.member_bids.get(sat_id, -math.inf)
            )
            self._satellite_reservations[sat_id] = task.task_id
            infos[agent]["reservation_version"] = reservation.version

    def _current_reservation_claims(self, task: TaskState) -> list[Claim]:
        """Replan leased members at the current physical state."""

        reservation = self._reservations.get(task.task_id)
        if reservation is None:
            return []
        claims: list[Claim] = []
        for sat_id, bid in reservation.member_bids.items():
            sat = self._satellites[sat_id]
            if not self._satellite_available(sat):
                continue
            feasible, _, plan = self._plan_task(sat, task)
            if not feasible:
                continue
            claims.append(
                (
                    self.possible_agents[sat_id],
                    float(bid),
                    float(plan["off_nadir_deg"]),
                    plan,
                )
            )
        return claims

    def _release_reservation(self, task_id: int) -> TaskReservationState | None:
        reservation = self._reservations.pop(task_id, None)
        if reservation is None:
            return None
        for sat_id in reservation.member_ids:
            if self._satellite_reservations.get(sat_id) == task_id:
                self._satellite_reservations.pop(sat_id, None)
        return reservation

    def _expire_reservations(
        self,
        rewards: dict[str, float] | None = None,
        infos: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        expired_task_ids = [
            task_id
            for task_id, reservation in self._reservations.items()
            if self._step > reservation.expires_step
            or not self._tasks[task_id].available
        ]
        for task_id in expired_task_ids:
            reservation = self._release_reservation(task_id)
            if reservation is None:
                continue
            self._reservation_expired_count += 1
            self._reservation_member_waste += len(reservation.member_bids)
            if rewards is None or infos is None:
                continue
            for sat_id in reservation.member_ids:
                agent = self.possible_agents[sat_id]
                rewards[agent] += self.config.reservation_failure_penalty
                infos[agent].update(
                    {
                        "event": "reservation_expired",
                        "task_id": task_id,
                        "reservation_version": reservation.version,
                    }
                )

    def _reject_action(
        self,
        agent: str,
        sat: SatelliteState,
        rewards: dict[str, float],
        infos: dict[str, dict[str, Any]],
        reason: str,
        window_miss: bool,
    ) -> None:
        rewards[agent] += self.config.invalid_action_penalty
        sat.invalid_count += 1
        if window_miss:
            sat.window_miss_count += 1
            rewards[agent] += self.config.window_miss_penalty
        infos[agent].update({"event": "infeasible_claim", "reason": reason})

    def _select_winner(self, contenders: list[Claim]) -> Claim:
        max_bid = max(item[1] for item in contenders)
        best = [item for item in contenders if np.isclose(item[1], max_bid)]
        min_distance = min(item[2] for item in best)
        best = [item for item in best if np.isclose(item[2], min_distance)]
        return best[0] if len(best) == 1 else best[int(self._rng.integers(0, len(best)))]

    def _select_simultaneous_group(
        self, contenders: list[Claim], required: int
    ) -> list[Claim]:
        tolerance = self.config.simultaneous_tolerance_steps
        best_group: list[Claim] = []
        best_score = -math.inf
        for anchor in contenders:
            anchor_finish = int(anchor[3]["finish_step"])
            compatible = [
                claim
                for claim in contenders
                if abs(int(claim[3]["finish_step"]) - anchor_finish) <= tolerance
            ]
            compatible.sort(key=lambda claim: claim[1], reverse=True)
            group = compatible[:required]
            if len(group) == required:
                score = sum(claim[1] for claim in group)
                if score > best_score:
                    best_score = score
                    best_group = group
        return best_group

    def _best_visible_station(self, sat_idx: int) -> tuple[int, float] | None:
        return self._station_cache[sat_idx][0] if self._station_cache[sat_idx] else None

    def _parse_action(self, raw_action: ActionValue) -> tuple[int, float | None]:
        if isinstance(raw_action, Mapping):
            action = int(raw_action.get("action", 0))
            raw_bid = raw_action.get("bid")
            return action, None if raw_bid is None else float(raw_bid)
        return int(raw_action), None

    def _action_to_task_id(self, agent: str, action: int) -> int | None:
        idx = action - 2
        if idx < 0 or idx >= self.config.candidate_k:
            return None
        candidate_ids = self._last_candidate_ids.get(agent)
        if candidate_ids is None or idx >= len(candidate_ids):
            return None
        task_id = int(candidate_ids[idx])
        return None if task_id < 0 else task_id

    def _all_tasks_finished(self) -> bool:
        return all(not task.available for task in self._tasks)

    def _task_active(self, task: TaskState) -> bool:
        return task.available and task.release_step <= self._step <= task.deadline_step

    def _satellite_available(self, sat: SatelliteState) -> bool:
        return self._step >= sat.busy_until_step and self._step >= sat.cooldown_until_step

    def _in_downlink_window(self, sat: SatelliteState) -> bool:
        return bool(self._station_cache[sat.sat_id]) if self._station_cache else False

    def _look_angle_deg(self, sat: SatelliteState, task: TaskState) -> float:
        return float(self._task_geometry(sat, task)["target_roll_deg"])

    def _agent_index(self, agent: str) -> int:
        try:
            return int(agent.rsplit("_", 1)[1])
        except Exception as exc:
            raise KeyError(f"Unknown agent id: {agent}") from exc

    @staticmethod
    def _action_category(action: int) -> int:
        if action == 0:
            return 0
        if action == 1:
            return 1
        if action >= 2:
            return 2
        return 3

    @staticmethod
    def _mode_code(mode: str) -> float:
        return {"optical": 0.0, "sar": 0.5, "infrared": 1.0}.get(mode, 1.0)

    @staticmethod
    def _mode_index(mode: str) -> int:
        return {"optical": 0, "sar": 1, "infrared": 2}.get(mode, 3)

    @staticmethod
    def _cooperation_code(mode: str) -> float:
        return {
            "single": 0.0,
            "sequential": 0.5,
            "simultaneous": 1.0,
            "coverage": 0.75,
        }.get(mode, 0.0)

    @staticmethod
    def _phase_distance(a: float, b: float) -> float:
        delta = abs(a - b) % 1.0
        return min(delta, 1.0 - delta)

    @staticmethod
    def _signed_phase_delta(a: float, b: float) -> float:
        return ((b - a + 0.5) % 1.0) - 0.5
