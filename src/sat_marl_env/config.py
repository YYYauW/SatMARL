from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EnvConfig:
    """Configuration for the multi-agent satellite tasking environment."""

    num_satellites: int = 32
    num_tasks: int = 768
    num_planes: int = 4
    num_ground_stations: int = 4
    max_steps: int = 240
    candidate_k: int = 24
    neighbor_k: int = 6
    random_seed: int | None = 7

    # Optional reproducible scenario inputs.  The ephemeris cache may be built
    # from simulated Keplerian six-element constellations (the primary path),
    # archived TLEs/SGP4, or another external propagator.  The task catalog
    # stores target coordinates and optional request attributes.  Leaving both
    # unset preserves the original synthetic scenario generator.
    ephemeris_cache_path: str | None = None
    task_catalog_path: str | None = None

    # Task generation. The realistic default keeps targets globally random.
    # curriculum_visible is for training: it samples many targets near future
    # satellite ground tracks so agents see more meaningful action choices.
    task_layout: str = "global_random"
    curriculum_visible_fraction: float = 0.0
    curriculum_ground_track_jitter_deg: float = 5.0
    curriculum_time_jitter_steps: int = 3
    curriculum_payload_match_probability: float = 0.8

    # Simulation epoch and two-body orbit propagation.
    # Decisions are made every 30 seconds, while a point acquisition itself
    # lasts five seconds.  This keeps the MARL horizon tractable without
    # pretending that an image acquisition occupies a full decision interval.
    step_duration_seconds: float = 30.0
    epoch_day_of_year: float = 182.0
    epoch_utc_hour: float = 0.0
    earth_radius_km: float = 6378.137
    earth_mu_km3_s2: float = 398600.4418
    earth_rotation_rate_rad_s: float = 7.2921159e-5
    orbit_altitude_min_km: float = 500.0
    orbit_altitude_max_km: float = 650.0
    orbit_eccentricity_max: float = 0.005
    orbit_inclination_min_deg: float = 45.0
    orbit_inclination_max_deg: float = 98.2

    # Observation geometry and payload diversity.
    min_observation_elevation_deg: float = 3.0
    max_off_nadir_deg: float = 45.0
    optical_min_sun_elevation_deg: float = 8.0
    quality_off_nadir_power: float = 1.6
    payload_swath_min_km: float = 20.0
    payload_swath_max_km: float = 120.0
    payload_resolution_min_m: float = 0.5
    payload_resolution_max_m: float = 12.0
    payload_fov_min_deg: float = 45.0
    payload_fov_max_deg: float = 45.0

    # Energy and onboard compressed data storage.
    max_energy: float = 100.0
    max_storage: float = 160.0
    solar_charge_per_step: float = 1.8
    base_energy_cost: float = 6.0
    original_data_min_mb: float = 20.0
    original_data_max_mb: float = 90.0
    compression_ratio_min: float = 0.18
    compression_ratio_max: float = 0.45
    data_timeliness_tau_steps: float = 28.0

    # Three-axis attitude dynamics and payload timing.
    max_roll_deg: float = 45.0
    max_pitch_deg: float = 35.0
    max_yaw_deg: float = 20.0
    max_angular_rate_deg_s: float = 1.2
    max_angular_accel_deg_s2: float = 0.18
    stabilization_seconds: float = 8.0
    payload_warmup_seconds: float = 35.0
    payload_cooldown_seconds: float = 30.0
    payload_idle_shutdown_steps: int = 8
    payload_warmup_energy: float = 1.5
    slew_energy_cost_per_deg: float = 0.035
    attitude_relax_deg_per_step: float = 5.0
    point_observation_seconds: float = 5.0
    task_duration_min: int = 1
    task_duration_max: int = 1
    min_task_window: int = 40
    max_task_window: int = 160
    post_task_cooldown_steps: int = 0
    planning_lookahead_steps: int = 30

    # Ground-station contention and segmented downlink.
    min_ground_elevation_deg: float = 10.0
    ground_station_channels: int = 2
    ground_station_rate_mb_per_step: float = 32.0

    # Cooperative tasks.
    cooperative_task_probability: float = 0.30
    simultaneous_task_fraction: float = 0.15
    cooperative_observers_min: int = 2
    cooperative_observers_max: int = 2
    simultaneous_tolerance_steps: int = 2
    sequential_max_gap_steps: int = 120

    idle_penalty: float = -0.01
    forced_idle_penalty: float = 0.0
    invalid_action_penalty: float = -1.0
    conflict_penalty: float = -0.35
    ground_conflict_penalty: float = -0.2
    coordination_failure_penalty: float = -0.3
    duplicate_penalty: float = -0.5
    maneuver_penalty_per_step: float = -0.03
    window_miss_penalty: float = -0.25
    # Task-aligned shaping: completion dominates data-volume reward so that a
    # policy cannot beat a scheduler merely by transmitting more megabytes.
    downlink_reward_per_unit: float = 0.005
    team_reward_weight: float = 0.08
    deadline_bonus_weight: float = 0.35
    task_completion_bonus: float = 2.0
    cooperative_completion_bonus: float = 4.0
    incomplete_cooperative_reward_scale: float = 0.20

    task_priority_min: float = 1.0
    task_priority_max: float = 10.0
    min_deadline: int = 12

    same_plane_neighbor_bonus: float = 0.35
    task_competitor_neighbor_bonus: float = 0.8
    station_competitor_neighbor_bonus: float = 0.6
