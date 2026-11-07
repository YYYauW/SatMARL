"""Opt-in, numerically faithful acceleration for the frozen event-v3 protocol.

The original environment remains the reference implementation.  This subclass
only reuses pure computations.  Event times, candidate sorting, interpolation,
planning and execution formulas are unchanged.  No action mask or task plan is
kept across observations.  Contact files contain immutable geometric windows;
radio state is reconstructed privately for every environment/reset.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import struct
import sys
import tempfile
from time import perf_counter

import numpy as np

from .event_transmission import ContinuousOrbit, EventTransmissionEnv
from .orbital import (
    ecef_to_spherical, geodetic_to_ecef, satellite_is_sunlit,
    sun_elevation_deg, sun_unit_ecef, target_geometry,
)

ACCELERATION_VERSION = "event-v3-pure-cache-v1"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(1024*1024), b""):
            digest.update(data)
    return digest.hexdigest()


class CachedContinuousOrbit(ContinuousOrbit):
    """Bounded per-event LRU around the original exact Hermite operation.

    Vector time queries (whole-day station scans) are deliberately not cached.
    Float keys retain all 64 bits, including signed zero.  Returned arrays are
    independent writable arrays, as in the reference, so a caller cannot alter
    a later cached answer by modifying an earlier return value.
    """
    def __init__(self, raw, max_entries=16384):
        super().__init__(raw)
        self.max_entries = max(0, int(max_entries))
        self._cache = OrderedDict()
        self.hits = self.misses = self.bypasses = self.evictions = 0
        self.peak_entries = 0
        self.event_clears = 0

    def clear_event(self):
        self._cache.clear()
        self.event_clears += 1

    def _interpolate(self, seconds, satellite_id, frame):
        if (not self.max_entries or np.ndim(seconds) != 0
                or not isinstance(satellite_id, (int, np.integer))
                or isinstance(satellite_id, (bool, np.bool_))):
            self.bypasses += 1
            return super()._interpolate(seconds, satellite_id, frame)
        key = (int(satellite_id), frame, struct.pack("!d", float(seconds)))
        result = self._cache.get(key)
        if result is not None:
            self.hits += 1
            self._cache.move_to_end(key)
            return tuple(item.copy() for item in result)
        self.misses += 1
        result = super()._interpolate(seconds, satellite_id, frame)
        self._cache[key] = tuple(item.copy() for item in result)
        if len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
            self.evictions += 1
        self.peak_entries = max(self.peak_entries, len(self._cache))
        return result

    def cache_snapshot(self):
        return dict(hits=self.hits, misses=self.misses, bypasses=self.bypasses,
                    evictions=self.evictions, entries=len(self._cache),
                    peak_entries=self.peak_entries, max_entries=self.max_entries,
                    event_clears=self.event_clears)


class AcceleratedEventTransmissionEnv(EventTransmissionEnv):
    """Compatible event-v3 environment with independent pure-data caches.

    ``cache_dir`` optionally enables cross-process geometric contact reuse.
    ``profile=False`` disables wall-clock timing; inexpensive cache counters stay
    enabled. ``orbit_cache_max_entries=0`` disables scalar interpolation reuse.
    Stage timings are cumulative/inclusive (geometry includes contact setup).
    ``performance_snapshot()`` is separate from the scientific summary/ledgers.
    """
    def __init__(self, config=None, cache_dir=None,
                 orbit_cache_max_entries=16384, profile=True):
        self._accel_cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
        self._accel_orbit_limit = int(orbit_cache_max_entries)
        if self._accel_orbit_limit < 0:
            raise ValueError("orbit_cache_max_entries must be nonnegative")
        self._accel_profile = bool(profile)
        self._accel_stage_seconds = {}
        self._accel_stage_calls = {}
        self._accel_counts = dict(task_coordinates_hit=0, task_coordinates_miss=0,
            station_coordinates_hit=0, station_coordinates_miss=0,
            contact_disk_hit=0, contact_disk_miss=0, contact_disk_invalid=0,
            contact_disk_write=0, contact_disk_error=0)
        self._accel_task_signature = None
        self._accel_station_signature = None
        self._accel_task_coordinates = None
        self._accel_station_coordinates = None
        self._accel_event = None
        self._accel_source_sha = None
        self._accel_contact_key = None
        self._accel_contact_identity = None
        self._accel_contact_signature = None
        super().__init__(config)

    def _timed(self, name, operation, *args, **kwargs):
        self._accel_stage_calls[name] = self._accel_stage_calls.get(name, 0)+1
        if not self._accel_profile:
            return operation(*args, **kwargs)
        started = perf_counter()
        try:
            return operation(*args, **kwargs)
        finally:
            self._accel_stage_seconds[name] = (
                self._accel_stage_seconds.get(name, 0.)+perf_counter()-started)

    def _load_ephemeris_input(self):
        original = super()._load_ephemeris_input()
        # raw.sha256 is the loaded NPZ content hash, not a path/mtime proxy.
        return CachedContinuousOrbit(original.raw, self._accel_orbit_limit)

    def reset(self, *args, **kwargs):
        self._accel_event = None
        self._accel_contact_signature = None
        self._accel_contact_key = None
        self._accel_contact_identity = None
        if isinstance(self._ephemeris, CachedContinuousOrbit):
            self._ephemeris.clear_event()
        return self._timed("reset", super().reset, *args, **kwargs)

    def _refresh_geometry(self):
        return self._timed("geometry", self._refresh_geometry_exact)

    def _refresh_geometry_exact(self):
        cfg = self.config
        if self._accel_event != float(self._step):
            if isinstance(self._ephemeris, CachedContinuousOrbit):
                self._ephemeris.clear_event()
            self._accel_event = float(self._step)
        elapsed = self._step * cfg.step_duration_seconds
        day = cfg.epoch_day_of_year + elapsed / 86400.0
        utc_hour = (cfg.epoch_utc_hour + elapsed / 3600.0) % 24.0
        sun_unit = sun_unit_ecef(day, utc_hour)
        self._sat_geometry = []
        for sat in self._satellites:
            position_eci, velocity_eci, position_ecef = self._propagate_satellite(sat, self._step)
            velocity_ecef = (
                self._ephemeris.ecef_velocity(self._step, sat.sat_id)
                if self._ephemeris is not None
                else self._ecef_velocity_from_eci(position_ecef, velocity_eci, elapsed))
            lat, lon, altitude = ecef_to_spherical(position_ecef, cfg.earth_radius_km)
            if self._ephemeris is not None:
                sat.phase = float((math.atan2(position_ecef[1], position_ecef[0])
                                   / (2.0 * math.pi)) % 1.0)
            else:
                mean_motion = math.sqrt(cfg.earth_mu_km3_s2 / sat.orbital_elements.semi_major_axis_km**3)
                mean_anomaly = math.radians(sat.orbital_elements.mean_anomaly_deg) + mean_motion * elapsed
                sat.phase = float((mean_anomaly / (2.0 * math.pi)) % 1.0)
            self._sat_geometry.append({
                "eci": position_eci, "velocity_eci": velocity_eci,
                "ecef": position_ecef, "velocity_ecef": velocity_ecef,
                "latitude_deg": lat, "longitude_deg": lon, "altitude_km": altitude,
                "sunlit": satellite_is_sunlit(position_ecef, sun_unit, cfg.earth_radius_km)})

        # Coordinate signatures also notice explicit test-side scene changes.
        task_signature = (cfg.earth_radius_km, tuple(
            (t.target_lat_deg, t.target_lon_deg) for t in self._tasks))
        if task_signature != self._accel_task_signature:
            self._accel_task_coordinates = np.asarray([
                geodetic_to_ecef(t.target_lat_deg, t.target_lon_deg, cfg.earth_radius_km)
                for t in self._tasks], dtype=np.float64)
            self._accel_task_signature = task_signature
            self._accel_counts["task_coordinates_miss"] += 1
        else:
            self._accel_counts["task_coordinates_hit"] += 1
        self._task_ecef = self._accel_task_coordinates
        self._task_sun_elevation = np.asarray([
            sun_elevation_deg(position, sun_unit) for position in self._task_ecef], dtype=np.float64)
        station_signature = (cfg.earth_radius_km, tuple(
            (g.latitude_deg, g.longitude_deg) for g in self._ground_stations))
        if station_signature != self._accel_station_signature:
            self._accel_station_coordinates = np.asarray([
                geodetic_to_ecef(g.latitude_deg, g.longitude_deg, cfg.earth_radius_km)
                for g in self._ground_stations], dtype=np.float64)
            self._accel_station_signature = station_signature
            self._accel_counts["station_coordinates_miss"] += 1
        else:
            self._accel_counts["station_coordinates_hit"] += 1
        self._station_ecef = self._accel_station_coordinates
        self._station_cache = []
        for sat_geo in self._sat_geometry:
            visible = []
            for station in self._ground_stations:
                elevation, _, _, earth_visible = target_geometry(
                    sat_geo["ecef"], self._station_ecef[station.station_id])
                if earth_visible and elevation >= station.min_elevation_deg:
                    visible.append((station.station_id, float(elevation)))
            visible.sort(key=lambda item: item[1], reverse=True)
            self._station_cache.append(visible)

        if not self._all_contacts:
            self._build_contact_timeline()
        now = float(self._step)
        self._ensure_radio_power(now)
        for sat in self._satellites:
            self._station_cache[sat.sat_id] = [
                (g, elevation) for g, elevation in self._station_cache[sat.sat_id]
                if elevation <= cfg.max_ground_elevation_deg]
        for sat_id in list(self._sessions):
            station_id = self._sessions[sat_id]
            if not any(a <= now+1e-8 < b for a, b in self._all_contacts[sat_id, station_id]):
                del self._sessions[sat_id]

    def _candidate_cache(self):
        return self._timed("candidate", self._candidate_cache_exact)

    def _candidate_cache_exact(self):
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
            # Identical array, constructed once rather than once per sat/time.
            priorities = np.asarray([t.priority for t in active])
            for sat in self._satellites:
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
                    value = priorities-.015*off-.01*offset/30
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

    def _plan_task(self, sat, task):
        return self._timed("planning", super()._plan_task, sat, task)

    def _observe_all(self):
        return self._timed("observation", super()._observe_all)

    def _observe_agent(self, sat_idx, candidate_cache, global_features):
        return self._timed("features", super()._observe_agent,
                           sat_idx, candidate_cache, global_features)

    def _prepare_local_neighborhoods(self, candidate_cache):
        return self._timed("neighborhood", super()._prepare_local_neighborhoods,
                           candidate_cache)

    def _advance_dynamics(self):
        return self._timed("dynamics", super()._advance_dynamics)

    def _contact_identity(self):
        # Exclude only data that cannot affect geometric contact windows.  All
        # physical and numerical configuration (including warmup) remains keyed.
        config = asdict(self.config)
        for name in ("task_catalog_path", "ephemeris_cache_path", "random_seed", "num_tasks"):
            config.pop(name, None)
        orbit = (dict(kind="external", sha256=self._ephemeris.raw.sha256)
                 if self._ephemeris is not None else
                 dict(kind="generated", elements=[asdict(s.orbital_elements) for s in self._satellites]))
        stations = [dict(station_id=g.station_id, name=g.name,
            latitude_deg=g.latitude_deg, longitude_deg=g.longitude_deg,
            min_elevation_deg=g.min_elevation_deg, channel_capacity=g.channel_capacity,
            rate_mb_per_step=g.rate_mb_per_step) for g in self._ground_stations]
        if self._accel_source_sha is None:
            directory = Path(__file__).resolve().parent
            self._accel_source_sha = {name: _file_sha(directory/name) for name in (
                "accelerated_transmission.py", "event_transmission.py", "transmission.py",
                "env.py", "orbital.py", "real_scenario.py")}
        return dict(version=ACCELERATION_VERSION, config=config, orbit=orbit,
                    stations=stations, source_sha256=self._accel_source_sha,
                    numpy=np.__version__, machine=platform.machine(), byteorder=sys.byteorder)

    def _decode_contacts(self, payload, identity, key):
        if not isinstance(payload, dict):
            raise ValueError("Contact cache must be a JSON object")
        if payload.get("identity_sha256") != key or payload.get("identity") != identity:
            raise ValueError("Contact identity mismatch")
        data = payload["windows"]
        if payload.get("windows_sha256") != hashlib.sha256(_canonical(data)).hexdigest():
            raise ValueError("Contact content checksum mismatch")
        ns, ng, horizon = len(self._satellites), len(self._ground_stations), float(self.config.max_steps)
        contacts = {}
        if len(data) != ns*ng:
            raise ValueError("Incomplete contact cache")
        for satellite, station, windows in data:
            if (not isinstance(satellite, int) or not 0 <= satellite < ns or
                    not isinstance(station, int) or not 0 <= station < ng or
                    (satellite, station) in contacts):
                raise ValueError("Invalid contact satellite/station")
            checked, last = [], -math.inf
            for a, b in windows:
                if not (math.isfinite(a) and math.isfinite(b) and 0 <= a < b <= horizon
                        and a >= last):
                    raise ValueError("Invalid contact interval")
                checked.append((float(a), float(b)))
                last = b
            contacts[satellite, station] = checked
        return contacts

    def _restore_radio_templates(self):
        # Reproduce the original merge ordering; never load actual_on/ready.
        self._radio_cycles = {}
        for sat in self._satellites:
            combined = []
            for station in self._ground_stations:
                windows = self._all_contacts[sat.sat_id, station.station_id]
                combined.extend((max(0.0, a-self.config.downlink_warmup_seconds), b)
                                for a, b in windows)
            merged = []
            for a, b in sorted(combined):
                if merged and a <= merged[-1]["end"]:
                    merged[-1]["end"] = max(merged[-1]["end"], b)
                else:
                    merged.append(dict(planned_on=a, end=b, actual_on=None, ready=None))
            self._radio_cycles[sat.sat_id] = merged

    def _build_contact_timeline(self):
        return self._timed("contact", self._build_contact_timeline_cached)

    def _build_contact_timeline_cached(self):
        if self._accel_cache_dir is None:
            return super()._build_contact_timeline()
        # Canonical JSON also normalizes tuples to lists before identity checks.
        identity = json.loads(_canonical(self._contact_identity()))
        key = hashlib.sha256(_canonical(identity)).hexdigest()
        self._accel_contact_key, self._accel_contact_identity = key, identity
        path = self._accel_cache_dir / (key+".contacts.json")
        if path.is_file():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                self._all_contacts = self._decode_contacts(payload, identity, key)
                self._restore_radio_templates()
                self._accel_counts["contact_disk_hit"] += 1
                return
            except (OSError, ValueError, TypeError, KeyError):
                self._accel_counts["contact_disk_invalid"] += 1
        self._accel_counts["contact_disk_miss"] += 1
        super()._build_contact_timeline()
        data = [[satellite, station, windows]
                for (satellite, station), windows in sorted(self._all_contacts.items())]
        payload = dict(identity_sha256=key, identity=identity, windows=data,
                       windows_sha256=hashlib.sha256(_canonical(data)).hexdigest())
        # Each process has its own temporary path; readers see one complete
        # validated file after replace, never a partially written JSON document.
        temp_path = None
        try:
            self._accel_cache_dir.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix=key+".", suffix=".tmp", dir=self._accel_cache_dir)
            temp_path = Path(name)
            with os.fdopen(fd, "wb") as handle:
                handle.write(_canonical(payload))
                handle.flush()
                os.fsync(handle.fileno())
            self._decode_contacts(json.loads(temp_path.read_text(encoding="utf-8")), identity, key)
            os.replace(temp_path, path)
            self._accel_counts["contact_disk_write"] += 1
        except (OSError, ValueError, TypeError):
            # Cache storage is optional: a full reference computation is valid.
            self._accel_counts["contact_disk_error"] += 1
        finally:
            if temp_path is not None and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    self._accel_counts["contact_disk_error"] += 1

    def performance_snapshot(self):
        orbit = (self._ephemeris.cache_snapshot()
                 if isinstance(self._ephemeris, CachedContinuousOrbit) else None)
        return dict(version=ACCELERATION_VERSION, profile_enabled=self._accel_profile,
                    stage_timing_semantics="cumulative_inclusive",
                    stage_seconds=dict(self._accel_stage_seconds),
                    stage_calls=dict(self._accel_stage_calls), cache=dict(self._accel_counts),
                    orbit_cache=orbit, contact_key=self._accel_contact_key)
