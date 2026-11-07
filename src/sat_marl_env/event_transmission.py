"""Continuous-second activity timing with event-driven state advancement.

Orbit-cache sampling and candidate refresh are independent of execution timing.
The archived step-based environments retain their original behavior.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .env import SatTaskingEnv
from .orbital import satellite_is_sunlit, sun_unit_ecef
from .real_scenario import load_ephemeris_cache
from .transmission import TransmissionConfig, TransmissionEnv, rolling_work_seconds

MODEL_VERSION = "optical-ground-delivery-event-v3"


@dataclass(slots=True)
class EventTransmissionConfig(TransmissionConfig):
    # In v3 a time index denotes ONE SECOND; transitions have variable duration.
    max_steps: int = 86400
    step_duration_seconds: float = 1.0
    planning_lookahead_steps: int = 900
    decision_clock: str = "event_seconds"
    ephemeris_sampling_seconds: float = 30.0
    candidate_refresh_seconds: float = 30.0
    opportunity_scan_seconds: float = 5.0
    wait_recheck_seconds: float = 5.0
    timing_tolerance_seconds: float = 0.001
    max_event_transitions: int = 2_000_000
    # Representative planning profile; not a single named spacecraft data sheet.
    max_angular_rate_deg_s: float = 1.0
    max_angular_accel_deg_s2: float = 0.25
    stabilization_seconds: float = 5.0
    payload_warmup_seconds: float = 60.0
    downlink_warmup_seconds: float = 420.0
    payload_cooldown_seconds: float = 0.0
    payload_idle_shutdown_steps: int = 240
    point_observation_seconds: float = 5.0
    solar_charge_per_step: float = 0.06
    idle_penalty: float = -0.0003333333333333333
    maneuver_penalty_per_step: float = -0.001
    min_task_window: int = 240
    max_task_window: int = 960
    data_timeliness_tau_steps: float = 840.0
    reservation_ttl_steps: int = 60
    simultaneous_tolerance_steps: int = 60
    sequential_max_gap_steps: int = 3600

    def __post_init__(self):
        TransmissionConfig.__post_init__(self)
        if self.step_duration_seconds != 1 or self.decision_clock != "event_seconds":
            raise ValueError("Event time indices must be continuous seconds, not fixed decision slots")
        for value in (self.ephemeris_sampling_seconds, self.candidate_refresh_seconds,
                      self.opportunity_scan_seconds, self.wait_recheck_seconds,
                      self.timing_tolerance_seconds, self.max_angular_rate_deg_s,
                      self.max_angular_accel_deg_s2):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Sampling, timing and attitude limits must be positive")
        for value in (self.stabilization_seconds, self.payload_warmup_seconds,
                      self.downlink_warmup_seconds, self.payload_cooldown_seconds):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Startup, settling and cooldown times must be nonnegative")
        if self.point_observation_seconds != 5:
            raise ValueError("This point-target protocol fixes exposure to five seconds")


class ContinuousOrbit:
    """Hermite interpolation of coarse-cache ECI/ECEF states at fractional seconds."""
    def __init__(self, raw):
        self.raw = raw
        self.metadata = raw.metadata
        self.plane_ids = raw.plane_ids
        self.orbital_elements = raw.orbital_elements

    def _interpolate(self, seconds, satellite_id, frame):
        raw, dt = self.raw, self.raw.step_duration_seconds
        x = np.asarray(seconds, dtype=float)/dt
        if np.any(x < 0) or np.any(x > raw.num_steps-1+1e-9):
            raise ValueError("Requested time outside the cached ephemeris")
        idx = np.minimum(np.floor(x).astype(int), raw.num_steps-2)
        u = np.expand_dims(x-idx, -1)
        positions = getattr(raw, f"position_{frame}_km")
        velocities = getattr(raw, f"velocity_{frame}_km_s")
        p0, p1 = positions[idx, satellite_id], positions[idx+1, satellite_id]
        v0, v1 = velocities[idx, satellite_id], velocities[idx+1, satellite_id]
        p = ((2*u**3-3*u**2+1)*p0+(u**3-2*u**2+u)*dt*v0
             +(-2*u**3+3*u**2)*p1+(u**3-u**2)*dt*v1)
        v = ((6*u*u-6*u)*p0/dt+(3*u*u-4*u+1)*v0
             +(-6*u*u+6*u)*p1/dt+(3*u*u-2*u)*v1)
        return p, v

    def state(self, seconds, satellite_id):
        p, v = self._interpolate(seconds, satellite_id, "eci")
        ecef, _ = self._interpolate(seconds, satellite_id, "ecef")
        return p, v, ecef

    def ecef_velocity(self, seconds, satellite_id):
        return self._interpolate(seconds, satellite_id, "ecef")[1]

    def summary(self):
        return {**self.raw.summary(), "execution_time": "continuous_seconds",
                "interpolation": "cubic_Hermite"}


def axis_maneuver_seconds(angle_deg, maximum_rate, maximum_acceleration):
    """Rest-to-rest triangular/trapezoidal angular-velocity profile."""
    distance = abs(angle_deg)
    threshold = maximum_rate**2/maximum_acceleration
    if distance <= threshold:
        return 2*math.sqrt(distance/maximum_acceleration)
    return distance/maximum_rate+maximum_rate/maximum_acceleration


def axis_profile(angle, rate, acceleration, elapsed):
    """Position/rate along a rest-to-rest profile, without timestep rounding."""
    sign, distance = (1 if angle >= 0 else -1), abs(angle)
    if distance == 0:
        return 0.0, 0.0
    ta = min(rate/acceleration, math.sqrt(distance/acceleration))
    peak = acceleration*ta
    cruise = max(0.0, (distance-acceleration*ta*ta)/peak)
    total = 2*ta+cruise
    t = max(0.0, min(elapsed, total))
    if t <= ta:
        pos, vel = .5*acceleration*t*t, acceleration*t
    elif t <= ta+cruise:
        pos, vel = .5*acceleration*ta*ta+peak*(t-ta), peak
    else:
        remaining = total-t
        pos, vel = distance-.5*acceleration*remaining*remaining, acceleration*remaining
    return sign*pos, sign*vel


class EventTransmissionEnv(TransmissionEnv):
    def __init__(self, config=None):
        self._profiles = {}
        self._activity_records = []
        self._all_contacts = {}
        self._radio_cycles = {}
        self._candidate_snapshot = None
        self._candidate_snapshot_time = -math.inf
        self._pending_downlink = None
        self._next_time = 0.0
        self._event_count = 0
        self._image_starts = 0
        self._radio_starts = 0
        self._imaging_power_seconds = 0.0
        self._radio_power_seconds = 0.0
        self._elapsed_intervals = []
        super().__init__(config or EventTransmissionConfig())

    def _load_ephemeris_input(self):
        raw = load_ephemeris_cache(self.config.ephemeris_cache_path,
                                   expected_satellites=self.config.num_satellites)
        required = self.config.max_steps+self.config.planning_lookahead_steps
        if (raw.num_steps-1)*raw.step_duration_seconds < required:
            raise ValueError("Orbit cache must cover the episode plus planning lookahead")
        return ContinuousOrbit(raw)

    def reset(self, *args, **kwargs):
        self._profiles.clear()
        self._activity_records.clear()
        self._all_contacts.clear()
        self._radio_cycles.clear()
        self._candidate_snapshot = None
        self._candidate_snapshot_time = -math.inf
        self._pending_downlink = None
        self._event_count = self._image_starts = self._radio_starts = 0
        self._imaging_power_seconds = self._radio_power_seconds = 0.0
        self._elapsed_intervals.clear()
        return super().reset(*args, **kwargs)

    def _position_at(self, sat, seconds):
        if self._ephemeris is not None:
            return self._ephemeris._interpolate(seconds, sat.sat_id, "ecef")[0]
        return self._propagate_satellite(sat, seconds)[2]

    def _build_contact_timeline(self):
        cfg = self.config
        times = np.linspace(0, cfg.max_steps,
                            max(1, math.ceil(cfg.max_steps/cfg.contact_scan_seconds))+1)
        for sat in self._satellites:
            if self._ephemeris is not None:
                positions = self._ephemeris._interpolate(times, sat.sat_id, "ecef")[0]
            else:
                positions = np.asarray([self._position_at(sat, float(t)) for t in times])
            combined = []
            for station in self._ground_stations:
                target = self._station_ecef[station.station_id]
                los = positions-target
                up = target/np.linalg.norm(target)
                elevations = np.degrees(np.arcsin(np.clip(
                    (los@up)/np.maximum(1e-9, np.linalg.norm(los, axis=1)), -1, 1)))
                cuts = [0.0, float(cfg.max_steps)]
                for level in (cfg.min_ground_elevation_deg, cfg.max_ground_elevation_deg):
                    crossing = np.flatnonzero((elevations[:-1] < level) != (elevations[1:] < level))
                    for index in crossing:
                        a, b = float(times[index]), float(times[index+1])
                        fa = elevations[index]-level
                        while b-a > cfg.contact_root_tolerance_seconds:
                            middle = (a+b)/2
                            fm = self._elevation_at(sat, station, middle)-level
                            if (fa < 0) == (fm < 0):
                                a, fa = middle, fm
                            else:
                                b = middle
                        cuts.append((a+b)/2)
                cuts = sorted(set(cuts))
                windows = []
                for a, b in zip(cuts[:-1], cuts[1:]):
                    value = self._elevation_at(sat, station, (a+b)/2)
                    if cfg.min_ground_elevation_deg <= value <= cfg.max_ground_elevation_deg:
                        if windows and abs(windows[-1][1]-a) <= cfg.contact_root_tolerance_seconds:
                            windows[-1] = (windows[-1][0], b)
                        else:
                            windows.append((a, b))
                combined.extend((max(0.0, a-cfg.downlink_warmup_seconds), b) for a, b in windows)
                self._all_contacts[sat.sat_id, station.station_id] = windows
            # One radio stays on across overlapping preheat/contact periods.
            merged = []
            for a, b in sorted(combined):
                if merged and a <= merged[-1]["end"]:
                    merged[-1]["end"] = max(merged[-1]["end"], b)
                else:
                    merged.append(dict(planned_on=a, end=b, actual_on=None, ready=None))
            self._radio_cycles[sat.sat_id] = merged

    def _ensure_radio_power(self, now):
        for sat in self._satellites:
            if not sat.packets:
                continue
            for cycle in self._radio_cycles.get(sat.sat_id, []):
                if cycle["planned_on"] <= now < cycle["end"] and cycle["actual_on"] is None:
                    cycle["actual_on"] = now
                    cycle["ready"] = now+self.config.downlink_warmup_seconds
                    self._radio_starts += 1

    def _radio_ready(self, sat_id, now):
        return any(c["ready"] is not None and c["ready"] <= now+1e-8 < c["end"]
                   for c in self._radio_cycles.get(sat_id, []))

    def _refresh_geometry(self):
        SatTaskingEnv._refresh_geometry(self)
        if not self._all_contacts:
            self._build_contact_timeline()
        now = float(self._step)
        self._ensure_radio_power(now)
        for sat in self._satellites:
            self._station_cache[sat.sat_id] = [
                (g, elevation) for g, elevation in self._station_cache[sat.sat_id]
                if elevation <= self.config.max_ground_elevation_deg]
        # Instantaneous visibility; future AOS is a separately scheduled event.
        for sat_id in list(self._sessions):
            station_id = self._sessions[sat_id]
            if not any(a <= now+1e-8 < b for a, b in self._all_contacts[sat_id, station_id]):
                del self._sessions[sat_id]

    def _observe_agent(self, sat_idx, candidate_cache, global_features):
        result = super()._observe_agent(sat_idx, candidate_cache, global_features)
        if not self._radio_ready(sat_idx, float(self._step)):
            result["action_mask"][1] = 0
        return result

    def _candidate_cache(self):
        cfg, now = self.config, float(self._step)
        if (self._candidate_snapshot is not None and
                now-self._candidate_snapshot_time < cfg.candidate_refresh_seconds-1e-8):
            return [np.asarray([j for j in ids if self._tasks[int(j)].available], dtype=np.int32)
                    for ids in self._candidate_snapshot]
        active = [t for t in self._tasks if t.available and t.deadline_step >= now
                  and t.release_step <= now+cfg.planning_lookahead_steps]
        empty = np.empty(0, dtype=np.int32)
        snapshot = []
        if not active:
            snapshot = [empty for _ in self._satellites]
        else:
            ids = np.asarray([t.task_id for t in active], dtype=np.int32)
            targets = self._task_ecef[ids]
            up = targets/np.linalg.norm(targets, axis=1, keepdims=True)
            releases = np.asarray([t.release_step for t in active])
            deadlines = np.asarray([t.deadline_step for t in active])
            for sat in self._satellites:
                # Geometric candidates are retained while busy for immediate
                # local refinement at the exact activity-completion event.
                scores = np.full(len(ids), -np.inf)
                for offset in np.arange(0, cfg.planning_lookahead_steps+1e-8,
                                        cfg.candidate_refresh_seconds):
                    time = now+float(offset)
                    position = self._position_at(sat, time)
                    los = position-targets
                    ranges = np.maximum(1e-8, np.linalg.norm(los, axis=1))
                    elevation = np.degrees(np.arcsin(np.clip(np.sum(los*up, axis=1)/ranges, -1, 1)))
                    toward = -los/ranges[:, None]
                    nadir = -position/np.linalg.norm(position)
                    off = np.degrees(np.arccos(np.clip(toward@nadir, -1, 1)))
                    valid = ((releases <= time) & (deadlines >= time)
                             & (elevation >= cfg.min_observation_elevation_deg)
                             & (off <= cfg.max_off_nadir_deg))
                    value = np.asarray([t.priority for t in active])-.015*off-.01*offset/30
                    scores[valid] = np.maximum(scores[valid], value[valid])
                candidates = [index for index in np.argsort(scores)[::-1]
                              if math.isfinite(scores[index]) and
                              sat.sat_id not in self._tasks[int(ids[index])].observed_by]
                selected = ids[candidates[:cfg.candidate_k]]
                leased = self._satellite_reservations.get(sat.sat_id)
                if leased is not None and self._tasks[leased].available and leased not in selected:
                    selected = np.concatenate(([leased], selected))[:cfg.candidate_k].astype(np.int32)
                snapshot.append(selected)
        self._candidate_snapshot, self._candidate_snapshot_time = snapshot, now
        return snapshot

    def _geometry_failure(self, sat, task, geo):
        cfg = self.config
        for failed, reason in (
            (not geo["earth_visible"], "earth_occulted"),
            (geo["elevation_deg"] < cfg.min_observation_elevation_deg, "below_min_elevation"),
            (geo["off_nadir_deg"] > cfg.max_off_nadir_deg, "off_nadir_limit"),
            (abs(geo["target_roll_deg"]) > cfg.max_roll_deg, "roll_limit"),
            (abs(geo["target_pitch_deg"]) > cfg.max_pitch_deg, "pitch_limit"),
            (abs(geo["target_yaw_deg"]) > cfg.max_yaw_deg, "yaw_limit"),
            (geo["effective_resolution_m"] > task.required_resolution_m, "resolution_limit"),
            (geo["effective_swath_km"] < task.required_swath_km, "swath_limit"),
            (geo["sun_elevation_deg"] < task.min_sun_elevation_deg, "insufficient_sunlight"),
        ):
            if failed:
                return reason
        return "ok"

    def _plan_task(self, sat, task):
        now, cfg = float(self._step), self.config
        key = (now, sat.sat_id, task.task_id)
        if key in self._plan_cache:
            return self._plan_cache[key]
        fallback = {**self._task_geometry(sat, task, 0), "slew_deg": 0., "slew_seconds": 0.,
                    "slew_steps": 0., "stabilization_steps": 0., "warmup_steps": 0.,
                    "maneuver_steps": 0., "observation_offset_steps": 0.,
                    "observation_start_step": now, "finish_step": now+5,
                    "observation_duration_seconds": 5., "observation_start_seconds": now,
                    "observation_finish_seconds": now+5, "total_energy": task.energy_cost,
                    "storage_required_mb": self._observation_storage_mb(task)}
        initial = "ok"
        if not self._satellite_available(sat): initial = "satellite_busy"
        elif not task.available: initial = "task_not_available"
        elif sat.sat_id in task.observed_by: initial = "already_contributed"
        elif sat.payload_mode != task.required_mode: initial = "payload_mode_mismatch"
        if initial != "ok":
            result = False, initial, fallback
            self._plan_cache[key] = result
            return result
        lower = max(now, float(task.release_step))
        upper = min(now+cfg.planning_lookahead_steps, float(task.deadline_step)-5,
                    float(cfg.max_steps)-5)

        def evaluate(start):
            geo = self._task_geometry(sat, task, start-now)
            deltas = [geo[f"target_{axis}_deg"]-getattr(sat, f"{axis}_deg")
                      for axis in ("roll", "pitch", "yaw")]
            slew = max(axis_maneuver_seconds(d, cfg.max_angular_rate_deg_s,
                                             cfg.max_angular_accel_deg_s2) for d in deltas)
            warmup = (max(0., sat.payload_ready_step-now) if sat.payload_on
                      else cfg.payload_warmup_seconds)
            prepare = max(slew+cfg.stabilization_seconds, warmup)
            energy = task.energy_cost+sum(abs(d) for d in deltas)*cfg.slew_energy_cost_per_deg
            if not sat.payload_on:
                energy += cfg.payload_warmup_energy
            plan = {**geo, "slew_deg": sum(abs(d) for d in deltas), "slew_seconds": slew,
                    "slew_steps": slew, "stabilization_steps": cfg.stabilization_seconds,
                    "warmup_steps": warmup, "maneuver_steps": slew+cfg.stabilization_seconds,
                    "observation_offset_steps": start-now, "observation_start_step": start,
                    "finish_step": start+5, "observation_duration_seconds": 5.,
                    "observation_start_seconds": start, "observation_finish_seconds": start+5,
                    "total_energy": energy, "storage_required_mb": self._observation_storage_mb(task),
                    "payload_cold_start": not sat.payload_on,
                    "slew_start_seconds": start-cfg.stabilization_seconds-slew}
            reason = self._geometry_failure(sat, task, geo)
            if reason == "ok" and start-now < prepare-1e-8:
                reason = "preparation_incomplete"
            if reason == "ok" and sat.energy < energy: reason = "insufficient_energy"
            if reason == "ok" and sat.storage+plan["storage_required_mb"] > cfg.max_storage:
                reason = "insufficient_storage"
            if reason == "ok": reason = self._additional_plan_failure(sat, task, plan)
            if reason == "ok":
                # Check the entire five-second acquisition at <=1-second spacing.
                for offset in (1., 2., 3., 4., 5.):
                    end_geo = self._task_geometry(sat, task, start+offset-now)
                    reason = self._geometry_failure(sat, task, end_geo)
                    if reason != "ok": break
            return reason, plan

        result = False, "no_future_visibility", fallback
        previous = lower
        samples = list(np.arange(lower, upper+1e-8, cfg.opportunity_scan_seconds)) if lower <= upper else []
        if samples and samples[-1] < upper: samples.append(upper)
        for sample in samples:
            reason, plan = evaluate(float(sample))
            result = False, reason, plan
            if reason == "ok":
                left, right = previous, float(sample)
                while right-left > cfg.timing_tolerance_seconds:
                    middle = (left+right)/2
                    why, _ = evaluate(middle)
                    if why == "ok": right = middle
                    else: left = middle
                reason, plan = evaluate(right)
                result = True, reason, plan
                break
            previous = float(sample)
        self._plan_cache[key] = result
        return result

    def _additional_plan_failure(self, sat, task, plan):
        start, finish = plan["observation_start_seconds"], plan["observation_finish_seconds"]
        if finish > self.config.max_steps+1e-8: return "acquisition_after_horizon"
        if finish > task.deadline_step+1e-8: return "acquisition_after_deadline"
        intervals = self._imaging_intervals.get(sat.sat_id, [])
        if any(start < b-1e-8 and finish > a+1e-8 for a, b in intervals):
            return "imaging_overlap"
        # Commitments are chronological; existing intervals already passed budget checks.
        if rolling_work_seconds([*intervals, (start, finish)], finish,
                                self.config.imaging_window_seconds) > self.config.imaging_budget_seconds+1e-8:
            return "imaging_duty_limit"
        return "ok"

    def _plan_finish_index(self, plan):
        return float(plan["finish_step"])

    def _commit_observation(self, task, claim, rewards, infos):
        agent, _, _, plan = claim
        sat = self._satellites[self._agent_index(agent)]
        origins = [getattr(sat, f"{axis}_deg") for axis in ("roll", "pitch", "yaw")]
        was_on, ready = sat.payload_on, sat.payload_ready_step
        closed = super()._commit_observation(task, claim, rewards, infos)
        finish = plan["observation_finish_seconds"]
        sat.busy_until_step = finish
        sat.cooldown_until_step = finish+self.config.payload_cooldown_seconds
        sat.payload_cooldown_until_step = sat.cooldown_until_step
        power_on = (None if was_on else max(float(self._step),
                    plan["observation_start_seconds"]-self.config.payload_warmup_seconds))
        sat.payload_on = was_on or power_on <= float(self._step)+1e-8
        sat.payload_ready_step = ready if was_on else power_on+self.config.payload_warmup_seconds
        sat.last_payload_use_step = finish
        sat.packets[-1].generated_step = finish
        startup_logged = was_on or sat.payload_on
        self._profiles[sat.sat_id] = dict(plan=plan, origins=origins, task_id=task.task_id,
                                         power_on=power_on, startup_logged=startup_logged)
        self._activity_records.append(dict(
            acquisition_id=len(self._acquisitions)-1,
            slew_start_seconds=plan["slew_start_seconds"], slew_seconds=plan["slew_seconds"],
            stabilization_seconds=self.config.stabilization_seconds,
            startup_seconds=0 if was_on else self.config.payload_warmup_seconds,
            payload_power_on_seconds=power_on,
            payload_ready_seconds=sat.payload_ready_step))
        if not was_on and sat.payload_on: self._image_starts += 1
        infos[agent].update(slew_seconds=plan["slew_seconds"],
                            stabilization_seconds=self.config.stabilization_seconds,
                            startup_seconds=0 if was_on else self.config.payload_warmup_seconds,
                            observation_start_seconds=plan["observation_start_seconds"],
                            observation_finish_seconds=finish)
        return closed

    def _resolve_downlinks(self, requests, rewards, infos):
        # Integrate only after new observation commitments determine the next event.
        self._pending_downlink = requests, rewards, infos

    def _assign_downlinks(self):
        if self._pending_downlink is None: return
        requests, rewards, infos = self._pending_downlink
        counts = {g.station_id: 0 for g in self._ground_stations}
        for station_id in self._sessions.values(): counts[station_id] += 1
        contenders = {self._agent_index(a) for group in requests.values() for a, _, _ in group}
        for sat_id in sorted(contenders, key=lambda i: -self._downlink_priority(self._satellites[i], 90)):
            if sat_id in self._sessions or not self._radio_ready(sat_id, float(self._step)):
                continue
            chosen = next((g for g, _ in self._station_cache[sat_id]
                           if counts[g] < self._ground_stations[g].channel_capacity), None)
            if chosen is not None:
                self._sessions[sat_id] = chosen
                counts[chosen] += 1
            else:
                self._satellites[sat_id].ground_conflict_count += 1
                rewards[self.agents[sat_id]] += self.config.ground_conflict_penalty
                infos[self.agents[sat_id]]["event"] = "ground_station_conflict"

    def _choose_next_time(self):
        now, cfg = float(self._step), self.config
        events = [float(cfg.max_steps)]
        future = lambda time: time > now+1e-8 and time <= cfg.max_steps
        relevant = [t for t in self._tasks if t.available]
        if relevant:
            nearest = min(t.release_step for t in relevant)
            if nearest > now+cfg.planning_lookahead_steps:
                events.append(nearest-cfg.planning_lookahead_steps)
            else:
                # Opportunity polling is an explicit wait duration; it does not
                # quantize slew, acquisition, startup, or completion timestamps.
                has_choice = any(np.any(m[1:]) for m in self._last_action_masks.values())
                events.append(now+(cfg.wait_recheck_seconds if has_choice else cfg.candidate_refresh_seconds))
            events.extend(t.release_step for t in relevant if future(t.release_step))
            events.extend(t.deadline_step+cfg.timing_tolerance_seconds for t in relevant
                          if future(t.deadline_step+cfg.timing_tolerance_seconds))
        for sat in self._satellites:
            events.extend(t for t in (sat.busy_until_step, sat.cooldown_until_step,
                                     sat.payload_ready_step) if future(t))
            if sat.payload_on and sat.last_payload_use_step >= 0:
                shutdown = sat.last_payload_use_step+cfg.payload_idle_shutdown_steps
                if future(shutdown): events.append(shutdown)
            profile = self._profiles.get(sat.sat_id)
            if profile:
                plan = profile["plan"]
                if profile["power_on"] is not None and future(profile["power_on"]):
                    events.append(profile["power_on"])
                events.extend(t for t in (plan["slew_start_seconds"],
                    plan["observation_start_seconds"]-cfg.stabilization_seconds,
                    plan["observation_start_seconds"], plan["observation_finish_seconds"]) if future(t))
            if sat.packets:
                for g in self._ground_stations:
                    for a, b in self._all_contacts[sat.sat_id, g.station_id]:
                        if future(a): events.append(a)
                        if a <= now+1e-8 < b: events.append(b)
                for cycle in self._radio_cycles.get(sat.sat_id, []):
                    for t in (cycle["planned_on"], cycle["ready"], cycle["end"]):
                        if t is not None and future(t): events.append(t)
            if sat.sat_id in self._sessions:
                ready_packets = [p for p in sat.packets if p.generated_time_seconds <= now+1e-8]
                if ready_packets:
                    packet = min(ready_packets, key=lambda p: (p.generated_time_seconds, -p.priority))
                    events.append(now+packet.remaining_mb/(cfg.downlink_rate_mbps*cfg.downlink_efficiency/8))
        for lease in self._reservations.values():
            expiry = lease.expires_step+cfg.timing_tolerance_seconds
            if future(expiry): events.append(expiry)
        valid = [t for t in events if future(t)]
        return min(valid) if valid else float(cfg.max_steps)

    def _advance_dynamics(self):
        now, cfg = float(self._step), self.config
        self._ensure_radio_power(now)
        self._assign_downlinks()
        end = self._choose_next_time()
        if end <= now: raise RuntimeError("Event clock failed to advance")
        self._next_time = end
        dt = end-now
        self._elapsed_intervals.append(dt)
        self._event_count += 1
        if self._event_count > cfg.max_event_transitions:
            raise RuntimeError("Event guard exceeded before the physical horizon")
        antenna_seconds = 0.0
        if self._pending_downlink:
            _, rewards, infos = self._pending_downlink
            for agent in self.agents:
                event = infos[agent].get("event")
                if event == "avoidable_idle":
                    rewards[agent] += cfg.idle_penalty*(dt-1)
                elif event == "forced_idle":
                    rewards[agent] += cfg.forced_idle_penalty*(dt-1)
            for sat_id, station_id in list(self._sessions.items()):
                sat, station, agent = self._satellites[sat_id], self._ground_stations[station_id], self.agents[sat_id]
                contact_end = next(b for a, b in self._all_contacts[sat_id, station_id] if a <= now+1e-8 < b)
                amount, reward, used_until = self._transmit_interval(sat, now, min(end, contact_end))
                station.delivered_mb += amount
                sat.downlinked_mb += amount
                rewards[agent] += reward
                self._downlink_reward_total += reward
                used = max(0., used_until-now) if amount else 0.
                antenna_seconds += used
                infos[agent].update(event="downlink", station_id=station_id,
                    downlinked=amount, transmission_start_seconds=now,
                    transmission_finish_seconds=used_until, transmission_seconds=used)
                if not sat.packets or end >= contact_end-1e-8:
                    del self._sessions[sat_id]
        self._contact_antenna_seconds += antenna_seconds
        antennas = sum(g.channel_capacity for g in self._ground_stations)
        self._last_ground_utilization = antenna_seconds/max(1e-8, antennas*dt)
        for sat in self._satellites:
            # Integrate the inherited normalized solar model independently of
            # policy calls, using midpoint quadrature with <=60-second cells.
            cells = max(1, math.ceil(dt/60))
            cell_dt = dt/cells
            sunlight = 0.0
            for cell in range(cells):
                middle = now+(cell+.5)*cell_dt
                sun = sun_unit_ecef(cfg.epoch_day_of_year+middle/86400,
                                   (cfg.epoch_utc_hour+middle/3600)%24)
                if satellite_is_sunlit(self._position_at(sat, middle), sun, cfg.earth_radius_km):
                    sunlight += cell_dt
            sat.energy = min(cfg.max_energy, sat.energy+cfg.solar_charge_per_step*sunlight)
            if sat.payload_on:
                self._imaging_power_seconds += dt
            for cycle in self._radio_cycles.get(sat.sat_id, []):
                if cycle["actual_on"] is not None:
                    self._radio_power_seconds += max(0., min(end, cycle["end"])-max(now, cycle["actual_on"]))
            profile = self._profiles.get(sat.sat_id)
            if profile:
                plan = profile["plan"]
                if not profile["startup_logged"] and end >= profile["power_on"]-1e-8:
                    sat.payload_on = True
                    profile["startup_logged"] = True
                    self._image_starts += 1
                for axis, origin in zip(("roll", "pitch", "yaw"), profile["origins"]):
                    target = plan[f"target_{axis}_deg"]
                    offset, velocity = axis_profile(target-origin, cfg.max_angular_rate_deg_s,
                        cfg.max_angular_accel_deg_s2, end-plan["slew_start_seconds"])
                    setattr(sat, f"{axis}_deg", origin+offset)
                    setattr(sat, f"{axis}_rate_deg_s", velocity)
                if end >= plan["observation_finish_seconds"]-1e-8:
                    # End-of-acquisition pointing becomes the next transition's origin.
                    geo = self._task_geometry(sat, self._tasks[profile["task_id"]],
                                              plan["observation_finish_seconds"]-now)
                    for axis in ("roll", "pitch", "yaw"):
                        setattr(sat, f"{axis}_deg", geo[f"target_{axis}_deg"])
                        setattr(sat, f"{axis}_rate_deg_s", 0.)
                    del self._profiles[sat.sat_id]
            if (sat.payload_on and sat.last_payload_use_step >= 0 and
                    end >= sat.last_payload_use_step+cfg.payload_idle_shutdown_steps and
                    sat.sat_id not in self._profiles):
                sat.payload_on = False
        self._pending_downlink = None

    def _advance_clock(self):
        self._step = self._next_time

    def summary(self):
        result = super().summary()
        result.update(transmission_model=MODEL_VERSION, simulated_seconds=float(self._step),
            simulated_hours=float(self._step)/3600, event_transitions=self._event_count,
            imaging_startup_count=self._image_starts, downlink_startup_count=self._radio_starts,
            imaging_power_on_seconds=self._imaging_power_seconds,
            radio_power_on_seconds=self._radio_power_seconds,
            minimum_event_interval_seconds=min(self._elapsed_intervals, default=0.),
            maximum_event_interval_seconds=max(self._elapsed_intervals, default=0.))
        return result

    def acquisition_ledger(self):
        activities = {r["acquisition_id"]: r for r in self._activity_records}
        return [{**r, **activities.get(r["acquisition_id"], {})}
                for r in super().acquisition_ledger()]

    def radio_power_ledger(self):
        return [dict(satellite_id=sat_id, **cycle)
                for sat_id, cycles in self._radio_cycles.items()
                for cycle in cycles if cycle["actual_on"] is not None]
