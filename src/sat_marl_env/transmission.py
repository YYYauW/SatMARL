"""Versioned optical observation / ground-delivery evaluation environment.

The archived environment and its default configuration remain unchanged.
Rates are decimal Mbps; data volumes are decimal MB; all intervals are seconds.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from .config import EnvConfig
from .entities import GroundStationState
from .env import SatTaskingEnv
from .orbital import propagate_kepler, target_geometry


MODEL_VERSION = "optical-ground-delivery-v2"


@dataclass(slots=True)
class TransmissionConfig(EnvConfig):
    # Engineering reference: Wuhan-1 8 Tbit storage and 900 Mbps X-band channel.
    max_storage: float = 1_000_000.0
    original_data_min_mb: float = 2000.0
    original_data_max_mb: float = 8000.0
    compression_ratio_min: float = 0.25
    compression_ratio_max: float = 0.25
    num_ground_stations: int = 5
    downlink_rate_mbps: float = 900.0
    downlink_efficiency: float = 0.8
    max_ground_elevation_deg: float = 90.0
    contact_scan_seconds: float = 1.0
    contact_root_tolerance_seconds: float = 0.01
    imaging_window_seconds: float = 5400.0
    imaging_budget_seconds: float = 1800.0
    optical_only: bool = True
    optical_resolution_m: float = 0.5
    optical_swath_km: float = 20.0
    # Coordinates are disclosed simulation proxies, not surveyed antenna sites.
    ground_station_sites: tuple = (
        ("Miyun_proxy", 40.45, 116.85),
        ("Kashgar_proxy", 39.47, 75.99),
        ("Sanya_proxy", 18.31, 109.31),
        ("Lijiang_proxy", 26.88, 100.23),
        ("Mohe_proxy", 52.97, 122.53),
    )

    def __post_init__(self):
        finite_positive = (
            self.downlink_rate_mbps, self.contact_scan_seconds,
            self.contact_root_tolerance_seconds, self.imaging_window_seconds,
            self.max_storage, self.step_duration_seconds,
            self.original_data_min_mb, self.original_data_max_mb,
            self.optical_resolution_m, self.optical_swath_km,
        )
        if not all(math.isfinite(x) and x > 0 for x in finite_positive):
            raise ValueError("Rates, sizes, time steps and contact tolerances must be positive/finite")
        if not 0 < self.downlink_efficiency <= 1:
            raise ValueError("downlink_efficiency must be in (0, 1]")
        if not 0 <= self.imaging_budget_seconds <= self.imaging_window_seconds:
            raise ValueError("Require 0 <= imaging budget <= rolling window")
        if not 0 <= self.min_ground_elevation_deg < self.max_ground_elevation_deg <= 90:
            raise ValueError("Require 0 <= minimum elevation < maximum elevation <= 90")
        if self.ground_station_channels < 1:
            raise ValueError("Each station must have at least one receiving antenna")
        if not 1 <= self.num_ground_stations <= len(self.ground_station_sites):
            raise ValueError("Explicit station coordinates required; stations cannot be duplicated")
        for name, lat, lon in self.ground_station_sites:
            if not name or not math.isfinite(lat) or not math.isfinite(lon) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError("Station sites require a name and valid latitude/longitude")
        if self.original_data_min_mb > self.original_data_max_mb:
            raise ValueError("Original-data range is reversed")
        if self.compression_ratio_min != .25 or self.compression_ratio_max != .25:
            raise ValueError("This protocol fixes compressed/raw size to 1/4")
        if self.area_task_fraction:
            raise ValueError("This protocol supports point-target requests only")
        # Keep the archived alias meaningful in exported configuration snapshots.
        self.ground_station_rate_mb_per_step = (
            self.downlink_rate_mbps*self.downlink_efficiency/8*self.step_duration_seconds)


def rolling_work_seconds(intervals, end: float, window: float) -> float:
    """Lebesgue duration in [end-window, end], not a count of decision slots."""
    return sum(max(0.0, min(b, end) - max(a, end - window)) for a, b in intervals)


def peak_rolling_work(intervals, window: float) -> float:
    """Exact maximum for a finite set of nonoverlapping imaging intervals."""
    if not intervals:
        return 0.0
    endpoints = {x for a, b in intervals for x in (a, b, a + window, b + window)}
    return max(rolling_work_seconds(intervals, t, window) for t in endpoints)


def elevation_windows(
    elevation_at: Callable[[float], float], start: float, end: float,
    minimum: float, maximum: float, scan: float = 1.0, tolerance: float = .01,
) -> list[tuple[float, float]]:
    """Find elevation-band intervals by scanning and bisecting both boundaries.

    Accuracy is tolerance after bracketing. An entire grazing contact shorter
    than scan can be missed; the recorded scan resolution is part of the protocol.
    """
    if end <= start:
        return []
    times = np.linspace(start, end, max(1, math.ceil((end-start)/scan)) + 1)
    values = [float(elevation_at(float(t))) for t in times]
    cuts = [start, end]
    for left, right, va, vb in zip(times[:-1], times[1:], values[:-1], values[1:]):
        for level in (minimum, maximum):
            if (va < level) == (vb < level):
                continue
            a, b, fa = float(left), float(right), va - level
            while b-a > tolerance:
                mid = .5*(a+b)
                fm = elevation_at(mid)-level
                if (fa < 0) == (fm < 0):
                    a, fa = mid, fm
                else:
                    b = mid
            cuts.append(.5*(a+b))
    cuts = sorted(set(cuts))
    windows = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        value = elevation_at(.5*(a+b))
        if minimum <= value <= maximum:
            if windows and abs(windows[-1][1]-a) < tolerance:
                windows[-1] = (windows[-1][0], b)
            else:
                windows.append((a, b))
    return windows


@dataclass(slots=True)
class Acquisition:
    acquisition_id: int
    task_id: int
    satellite_id: int
    start_seconds: float
    finish_seconds: float
    raw_data_mb: float
    compressed_data_mb: float
    delivered_mb: float = 0.0
    delivered_time_seconds: float | None = None


class TransmissionEnv(SatTaskingEnv):
    def __init__(self, config: TransmissionConfig | None = None):
        self._sessions = {}
        self._contacts = {}
        self._position_cache = {}
        self._acquisitions: list[Acquisition] = []
        self._imaging_intervals = {}
        self._closing_records = {}
        self._contact_antenna_seconds = 0.0
        super().__init__(config or TransmissionConfig())

    def reset(self, *args, **kwargs):
        self._sessions.clear()
        self._contacts.clear()
        self._position_cache.clear()
        self._acquisitions.clear()
        self._imaging_intervals.clear()
        self._closing_records.clear()
        self._contact_antenna_seconds = 0.0
        return super().reset(*args, **kwargs)

    def _build_satellites(self):
        satellites = super()._build_satellites()
        if self.config.optical_only:
            # Draws are unchanged so that the original orbit/attitude seed is retained.
            for sat in satellites:
                sat.payload_mode = "optical"
                sat.resolution_m = self.config.optical_resolution_m
                sat.swath_width_km = self.config.optical_swath_km
        return satellites

    def _build_tasks(self):
        tasks = super()._build_tasks()
        for task in tasks:
            if task.is_area:
                raise ValueError("Point-target downlink protocol cannot consume area requests")
            if not math.isfinite(task.original_data_mb) or task.original_data_mb <= 0:
                raise ValueError(f"Task {task.task_id} needs a positive original_data_mb")
            if self._task_catalog is not None:
                row = self._task_catalog[task.task_id]
                if row.get("memory_required_mb", "") != "":
                    memory = float(row["memory_required_mb"])
                    if not math.isfinite(memory) or not math.isclose(
                            memory, task.original_data_mb*.25, rel_tol=1e-8, abs_tol=1e-6):
                        raise ValueError("memory_required_mb must equal original_data_mb/4 per observation")
            task.compression_ratio = .25
            task.storage_gain = 0.0
            if self.config.optical_only:
                task.required_mode = "optical"
                # Requirements remain catalog attributes. Do not silently relax them.
                task.min_sun_elevation_deg = self.config.optical_min_sun_elevation_deg
        return tasks

    def _build_ground_stations(self):
        cfg = self.config
        return [GroundStationState(
            station_id=i, name=name, latitude_deg=float(lat), longitude_deg=float(lon),
            min_elevation_deg=cfg.min_ground_elevation_deg,
            channel_capacity=cfg.ground_station_channels,
            rate_mb_per_step=cfg.downlink_rate_mbps*cfg.downlink_efficiency/8*cfg.step_duration_seconds,
        ) for i, (name, lat, lon) in enumerate(cfg.ground_station_sites[:cfg.num_ground_stations])]

    def _position_at(self, sat, seconds):
        key = (sat.sat_id, float(seconds))
        if key in self._position_cache:
            return self._position_cache[key]
        if self._ephemeris is None:
            value = propagate_kepler(
                sat.orbital_elements, seconds, self.config.earth_mu_km3_s2,
                self.config.earth_rotation_rate_rad_s,
            )[2]
        else:
            # Cubic Hermite interpolation of cached ECEF positions and velocities.
            cache = self._ephemeris
            dt = cache.step_duration_seconds
            x = seconds/dt
            index = min(int(math.floor(x)), cache.num_steps-2)
            if index < 0 or x > cache.num_steps-1:
                raise ValueError("Contact search is outside the ephemeris horizon")
            u = x-index
            p0, p1 = cache.position_ecef_km[index:index+2, sat.sat_id]
            v0, v1 = cache.velocity_ecef_km_s[index:index+2, sat.sat_id]
            value = ((2*u**3-3*u**2+1)*p0 + (u**3-2*u**2+u)*dt*v0
                     + (-2*u**3+3*u**2)*p1 + (u**3-u**2)*dt*v1)
        self._position_cache[key] = value
        return value

    def _elevation_at(self, sat, station, seconds):
        position = self._position_at(sat, seconds)
        target = self._station_ecef[station.station_id]
        return target_geometry(position, target)[0]

    def _refresh_geometry(self):
        super()._refresh_geometry()
        self._position_cache.clear()
        self._contacts = {}
        start = self._step*self.config.step_duration_seconds
        end = min(start+self.config.step_duration_seconds,
                  self.config.max_steps*self.config.step_duration_seconds)
        cfg = self.config
        # For LEO paths the station-elevation rate is bounded well below 4 deg/s.
        # Use a conservative slant-range bound only to skip obviously remote sites.
        for sat, geo in zip(self._satellites, self._sat_geometry):
            visible = []
            for station in self._ground_stations:
                elevation, _, slant, _ = target_geometry(
                    geo["ecef"], self._station_ecef[station.station_id])
                # 20 km/s safely exceeds LEO ECEF motion for the supported altitudes.
                margin = math.degrees(20.0*max(0.0, end-start)/max(1.0, slant-20.0*(end-start)))
                if elevation+margin < station.min_elevation_deg:
                    continue
                windows = elevation_windows(
                    lambda t, s=sat, g=station: self._elevation_at(s, g, t),
                    start, end, station.min_elevation_deg, cfg.max_ground_elevation_deg,
                    cfg.contact_scan_seconds, cfg.contact_root_tolerance_seconds,
                )
                if windows:
                    self._contacts[sat.sat_id, station.station_id] = windows
                    mid = .5*(windows[0][0]+windows[0][1])
                    visible.append((station.station_id, self._elevation_at(sat, station, mid)))
            visible.sort(key=lambda item: item[1], reverse=True)
            self._station_cache[sat.sat_id] = visible
        for sat_id, station_id in list(self._sessions.items()):
            windows = self._contacts.get((sat_id, station_id), [])
            if not windows or windows[0][0] > start+cfg.contact_root_tolerance_seconds:
                del self._sessions[sat_id]

    def _satellite_available(self, sat):
        return sat.sat_id not in self._sessions and super()._satellite_available(sat)

    def _observation_storage_mb(self, task):
        # Each independent acquisition makes its own image, irrespective of nu_j.
        return task.original_data_mb*.25

    def _additional_plan_failure(self, sat, task, plan):
        start = float(plan["observation_start_seconds"])
        finish = float(plan["observation_finish_seconds"])
        dt = self.config.step_duration_seconds
        if finish > self.config.max_steps*dt + 1e-8:
            return "acquisition_after_horizon"
        # Task deadlines are end-of-slot bounds, matching archived step indexing.
        if finish > (task.deadline_step+1)*dt + 1e-8:
            return "acquisition_after_deadline"
        intervals = self._imaging_intervals.get(sat.sat_id, [])
        if any(start < b-1e-8 and finish > a+1e-8 for a, b in intervals):
            return "imaging_overlap"
        if peak_rolling_work([*intervals, (start, finish)], self.config.imaging_window_seconds) > self.config.imaging_budget_seconds+1e-8:
            return "imaging_duty_limit"
        return "ok"

    def _commit_observation(self, task, claim, rewards, infos):
        agent, _, _, plan = claim
        sat = self._satellites[self._agent_index(agent)]
        reason = self._additional_plan_failure(sat, task, plan)
        if reason != "ok":
            raise RuntimeError(f"Infeasible accepted acquisition: {reason}")
        closed = super()._commit_observation(task, claim, rewards, infos)
        start = float(plan["observation_start_seconds"])
        finish = float(plan["observation_finish_seconds"])
        record = Acquisition(len(self._acquisitions), task.task_id, sat.sat_id,
                             start, finish, task.original_data_mb, task.original_data_mb*.25)
        self._acquisitions.append(record)
        self._imaging_intervals.setdefault(sat.sat_id, []).append((start, finish))
        packet = sat.packets[-1]
        packet.original_size_mb = record.raw_data_mb
        packet.generated_time_seconds = finish
        packet.generated_step = int(math.ceil(finish/self.config.step_duration_seconds))
        packet.acquisition_id = record.acquisition_id
        # Support durations spanning several physical steps without releasing early.
        release_step = int(math.ceil(finish/self.config.step_duration_seconds))
        sat.busy_until_step = max(sat.busy_until_step, release_step)
        cooldown_steps = int(math.ceil(self.config.payload_cooldown_seconds/self.config.step_duration_seconds))
        sat.cooldown_until_step = max(sat.cooldown_until_step, release_step+cooldown_steps+self.config.post_task_cooldown_steps)
        sat.payload_cooldown_until_step = max(sat.payload_cooldown_until_step, release_step+cooldown_steps)
        infos[agent]["event"] = "observation_scheduled"
        infos[agent]["acquisition_id"] = record.acquisition_id
        if closed:
            # Capture the exact valid closing set before partial-progress expiry.
            ids = []
            for sat_id in task.completed_by_satellites:
                ids.append(next(r.acquisition_id for r in reversed(self._acquisitions)
                                if r.task_id == task.task_id and r.satellite_id == sat_id))
            self._closing_records[task.task_id] = ids
        return closed

    def _transmit_interval(self, sat, start, end):
        """FIFO with exact readiness and finish times, including multiple packets."""
        rate = self.config.downlink_rate_mbps*self.config.downlink_efficiency/8
        cursor, amount, reward = start, 0.0, 0.0
        sat.packets.sort(key=lambda p: (
            p.generated_time_seconds if p.generated_time_seconds is not None else p.generated_step*self.config.step_duration_seconds,
            -p.priority, p.task_id))
        for packet in sat.packets:
            ready = packet.generated_time_seconds
            if ready is None:
                ready = packet.generated_step*self.config.step_duration_seconds
            cursor = min(end, max(cursor, ready))
            if cursor >= end:
                break
            chunk = min(packet.remaining_mb, rate*(end-cursor))
            if chunk <= 1e-8:
                continue
            cursor += chunk/rate
            packet.remaining_mb = max(0.0, packet.remaining_mb-chunk)
            packet.delivered_mb += chunk
            amount += chunk
            age_steps = max(0.0, cursor-ready)/self.config.step_duration_seconds
            reward += (chunk*self.config.downlink_reward_per_unit
                       * math.exp(-age_steps/max(1e-6, self.config.data_timeliness_tau_steps))
                       * (1+packet.priority/max(1.0, self.config.task_priority_max)))
            if packet.acquisition_id is not None:
                record = self._acquisitions[packet.acquisition_id]
                record.delivered_mb += chunk
                if packet.remaining_mb <= 1e-6:
                    record.delivered_time_seconds = cursor
        sat.packets = [p for p in sat.packets if p.remaining_mb > 1e-6]
        sat.storage = sum(p.remaining_mb for p in sat.packets)
        return amount, reward, cursor

    def _resolve_downlinks(self, requests, rewards, infos):
        """Nonpreemptive sessions, antenna-time reuse, and station fallback.

        One satellite uses one receiver at a time. A receiver can serve another
        requester immediately after a queue drains or an elevation window closes.
        """
        cfg = self.config
        start = self._step*cfg.step_duration_seconds
        end = (self._step+1)*cfg.step_duration_seconds
        tolerance = cfg.contact_root_tolerance_seconds
        requested = {self._agent_index(a) for group in requests.values() for a, _, _ in group}
        active = set(self._sessions)
        participants = requested | active
        occupied = {g.station_id: [] for g in self._ground_stations}
        total_seconds = 0.0

        def free_intervals(station_id, a, b):
            # Split at occupied boundaries; station capacity is independent receivers.
            cuts = sorted({a, b, *(x for c, d in occupied[station_id] for x in (c, d) if a < x < b)})
            return [(c, d) for c, d in zip(cuts[:-1], cuts[1:])
                    if sum(x <= .5*(c+d) < y for x, y in occupied[station_id])
                    < self._ground_stations[station_id].channel_capacity]

        order = sorted(participants, key=lambda i: (
            i not in active, -self._downlink_priority(self._satellites[i], 90.0), i))
        for sat_id in order:
            sat = self._satellites[sat_id]
            agent = self.agents[sat_id]
            options = ([self._sessions[sat_id]] if sat_id in active else
                       [g for g, _ in self._station_cache[sat_id]])
            previous_station = self._sessions.pop(sat_id, None)
            best = None
            for station_id in options:
                windows = self._contacts.get((sat_id, station_id), [])
                available = [(c, d) for a, b in windows for c, d in free_intervals(station_id, a, b)]
                if available:
                    score = sum(b-a for a, b in available)
                    if best is None or score > best[0]:
                        best = score, station_id, available
            if best is None:
                sat.ground_conflict_count += 1
                for station_id in options[:1]:
                    self._ground_stations[station_id].conflict_count += 1
                rewards[agent] += cfg.ground_conflict_penalty
                infos[agent].update(event="ground_station_conflict")
                continue
            _, station_id, intervals = best
            station = self._ground_stations[station_id]
            amount, value, used = 0.0, 0.0, 0.0
            for a, b in intervals:
                if not sat.packets:
                    break
                part, rew, finish = self._transmit_interval(sat, a, b)
                if part <= 0:
                    continue
                # A pending packet waiting to become available occupies this session.
                occupied[station_id].append((a, finish))
                used += finish-a
                amount += part
                value += rew
                if sat.packets and abs(b-end) <= tolerance:
                    self._sessions[sat_id] = station_id
            station.delivered_mb += amount
            sat.downlinked_mb += amount
            total_seconds += used
            rewards[agent] += value
            self._downlink_reward_total += value
            infos[agent].update(event="downlink", station_id=station_id,
                                station_name=station.name, downlinked=amount,
                                transmission_seconds=used, timeliness_reward=value,
                                session_continued=previous_station is not None)
        total_antennas = sum(g.channel_capacity for g in self._ground_stations)
        self._contact_antenna_seconds += total_seconds
        self._last_ground_utilization = total_seconds/max(1e-8, total_antennas*(end-start))

    def _all_tasks_finished(self):
        # All methods retain the same physical horizon, including delivery time.
        return False

    def task_outcomes(self):
        now = self._step*self.config.step_duration_seconds
        results = []
        for task in self._tasks:
            records = [self._acquisitions[i] for i in self._closing_records.get(task.task_id, [])]
            observed = bool(records) and all(r.finish_seconds <= now+1e-8 for r in records)
            delivered = observed and all(r.delivered_time_seconds is not None and
                                        r.delivered_time_seconds <= now+1e-8 for r in records)
            results.append(dict(task_id=task.task_id, required_observers=task.required_observers,
                                original_data_mb=task.original_data_mb,
                                memory_per_observation_mb=task.original_data_mb*.25,
                                observation_success=observed,
                                observation_and_downlink_success=delivered,
                                observation_finish_seconds=max((r.finish_seconds for r in records), default=None),
                                delivery_finish_seconds=max((r.delivered_time_seconds for r in records), default=None) if delivered else None,
                                closing_acquisition_ids=[r.acquisition_id for r in records]))
        return results

    def summary(self):
        result = super().summary()
        outcomes = self.task_outcomes()
        observed = [o for o in outcomes if o["observation_success"]]
        delivered = [o for o in outcomes if o["observation_and_downlink_success"]]
        admitted_count = result["completed_tasks"]
        result.update(
            transmission_model=MODEL_VERSION,
            admitted_completed_tasks=admitted_count,
            observation_success_tasks=len(observed),
            observation_and_downlink_success_tasks=len(delivered),
            observation_success_rate=len(observed)/max(1, len(outcomes)),
            observation_and_downlink_success_rate=len(delivered)/max(1, len(outcomes)),
            observed_but_undelivered_tasks=len(observed)-len(delivered),
            delivered_priority=sum(self._tasks[o["task_id"]].priority for o in delivered),
            total_generated_data_mb=sum(r.compressed_data_mb for r in self._acquisitions
                                        if r.finish_seconds <= self._step*self.config.step_duration_seconds),
            committed_data_mb=sum(r.compressed_data_mb for r in self._acquisitions),
            contact_antenna_seconds=self._contact_antenna_seconds,
            mean_ground_station_utilization=self._contact_antenna_seconds/max(
                1e-8, self._step*self.config.step_duration_seconds
                * sum(g.channel_capacity for g in self._ground_stations)),
            imaging_window_seconds=self.config.imaging_window_seconds,
            imaging_budget_seconds=self.config.imaging_budget_seconds,
            max_rolling_imaging_seconds=max((peak_rolling_work(v, self.config.imaging_window_seconds)
                                             for v in self._imaging_intervals.values()), default=0.0),
        )
        # In v2 the generic aliases report execution-finished observations.
        result["completed_tasks"] = len(observed)
        result["cooperative_completed_tasks"] = sum(o["required_observers"] > 1 for o in observed)
        result["total_priority_completed"] = sum(self._tasks[o["task_id"]].priority for o in observed)
        return result

    def acquisition_ledger(self):
        return [asdict(r) for r in self._acquisitions]
