from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class OrbitalElements:
    semi_major_axis_km: float
    eccentricity: float
    inclination_deg: float
    raan_deg: float
    argument_of_perigee_deg: float
    mean_anomaly_deg: float


@dataclass(slots=True)
class DataPacketState:
    task_id: int
    generated_step: int
    original_size_mb: float
    compressed_size_mb: float
    remaining_mb: float
    priority: float
    source_satellite_id: int
    delivered_mb: float = 0.0


@dataclass(slots=True)
class SatelliteState:
    sat_id: int
    plane_id: int
    orbital_elements: OrbitalElements
    phase: float
    orbit_rate: float
    energy: float
    storage: float
    payload_mode: str
    swath_width_km: float
    field_of_view_deg: float
    resolution_m: float
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    roll_rate_deg_s: float = 0.0
    pitch_rate_deg_s: float = 0.0
    yaw_rate_deg_s: float = 0.0
    target_roll_deg: float = 0.0
    target_pitch_deg: float = 0.0
    target_yaw_deg: float = 0.0
    payload_on: bool = False
    payload_ready_step: int = 0
    payload_cooldown_until_step: int = 0
    last_payload_use_step: int = -1
    busy_until_step: int = 0
    cooldown_until_step: int = 0
    last_task_id: int | None = None
    last_action_category: int = 0
    completed_count: int = 0
    conflict_count: int = 0
    ground_conflict_count: int = 0
    invalid_count: int = 0
    window_miss_count: int = 0
    downlinked_mb: float = 0.0
    packets: list[DataPacketState] = field(default_factory=list)

    @property
    def attitude_deg(self) -> float:
        """Backward-compatible alias for the former one-axis attitude."""

        return self.roll_deg

    @attitude_deg.setter
    def attitude_deg(self, value: float) -> None:
        self.roll_deg = float(value)


@dataclass(slots=True)
class TaskState:
    task_id: int
    target_phase: float
    priority: float
    release_step: int
    deadline_step: int
    energy_cost: float
    storage_gain: float = 0.0
    duration: int = 1
    observation_duration_seconds: float = 5.0
    target_lat_deg: float = 0.0
    target_lon_deg: float = 0.0
    required_mode: str = "optical"
    required_resolution_m: float = 10.0
    required_swath_km: float = 0.0
    min_sun_elevation_deg: float = 8.0
    original_data_mb: float = 20.0
    compression_ratio: float = 0.3
    cooperation_mode: str = "single"
    required_observers: int = 1
    max_coordination_gap_steps: int = 10
    observed_by: list[int] = field(default_factory=list)
    observation_steps: list[int] = field(default_factory=list)
    observation_qualities: list[float] = field(default_factory=list)
    completed_by_satellites: list[int] = field(default_factory=list)
    completed_by: int | None = None
    completed_step: int | None = None
    expired: bool = False
    target_type: str = "point"
    area_width_km: float = 0.0
    area_height_km: float = 0.0
    area_orientation_deg: float = 0.0
    coverage_threshold: float = 1.0
    coverage_sample_count: int = 0
    coverage_mask: int = 0
    coverage_fraction: float = 0.0
    required_strips: int = 1
    min_contributing_satellites: int = 1
    strip_count: int = 0
    inside_imaged_area_km2: float = 0.0
    outside_imaged_area_km2: float = 0.0
    redundant_imaged_area_km2: float = 0.0
    strip_headings_deg: list[float] = field(default_factory=list)
    strip_footprints_local: list[list[tuple[float, float]]] = field(
        default_factory=list
    )

    @property
    def compressed_data_mb(self) -> float:
        if self.storage_gain > 0.0:
            return self.storage_gain
        return self.original_data_mb * self.compression_ratio

    @property
    def available(self) -> bool:
        return self.completed_by is None and not self.expired

    @property
    def is_area(self) -> bool:
        return self.target_type == "area"

    @property
    def area_km2(self) -> float:
        return max(0.0, self.area_width_km) * max(0.0, self.area_height_km)


@dataclass(slots=True)
class GroundStationState:
    station_id: int
    name: str
    latitude_deg: float
    longitude_deg: float
    min_elevation_deg: float
    channel_capacity: int
    rate_mb_per_step: float
    delivered_mb: float = 0.0
    conflict_count: int = 0
