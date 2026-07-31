from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


EPHEMERIS_FORMAT = "satmarl-ephemeris-v1"
LEGACY_EPHEMERIS_FORMAT = "satmarl-tle-ephemeris-v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class EphemerisCache:
    """Time-indexed satellite states from six elements, TLEs, or another source."""

    path: Path
    satellite_names: tuple[str, ...]
    plane_ids: np.ndarray | None
    orbital_elements: tuple[dict[str, Any], ...] | None
    position_eci_km: np.ndarray
    velocity_eci_km_s: np.ndarray
    position_ecef_km: np.ndarray
    velocity_ecef_km_s: np.ndarray
    metadata: dict[str, Any]
    sha256: str

    @property
    def num_steps(self) -> int:
        return int(self.position_ecef_km.shape[0])

    @property
    def num_satellites(self) -> int:
        return int(self.position_ecef_km.shape[1])

    @property
    def step_duration_seconds(self) -> float:
        return float(self.metadata["step_duration_seconds"])

    def state(
        self, step: int, satellite_id: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if step < 0 or step >= self.num_steps:
            raise IndexError(
                f"Ephemeris step {step} is outside [0, {self.num_steps - 1}]."
            )
        if satellite_id < 0 or satellite_id >= self.num_satellites:
            raise IndexError(
                f"Satellite {satellite_id} is outside [0, {self.num_satellites - 1}]."
            )
        return (
            self.position_eci_km[step, satellite_id],
            self.velocity_eci_km_s[step, satellite_id],
            self.position_ecef_km[step, satellite_id],
        )

    def ecef_velocity(self, step: int, satellite_id: int) -> np.ndarray:
        return self.velocity_ecef_km_s[step, satellite_id]

    def summary(self) -> dict[str, Any]:
        summary = {
            "mode": self.metadata.get("mode", "external_ephemeris_cache"),
            "source": self.metadata.get("source"),
            "format": self.metadata.get("format"),
            "path": str(self.path),
            "sha256": self.sha256,
            "satellites": self.num_satellites,
            "planes": (
                int(np.max(self.plane_ids)) + 1
                if self.plane_ids is not None and self.plane_ids.size
                else None
            ),
            "steps": self.num_steps,
            "step_duration_seconds": self.step_duration_seconds,
            "start_utc": self.metadata.get("start_utc"),
            "elements_sha256": self.metadata.get("elements_sha256"),
            "constellation": self.metadata.get("constellation"),
        }
        if self.metadata.get("tle_sha256"):
            summary.update(
                {
                    "tle_sha256": self.metadata.get("tle_sha256"),
                    "tle_epoch_min_utc": self.metadata.get("tle_epoch_min_utc"),
                    "tle_epoch_max_utc": self.metadata.get("tle_epoch_max_utc"),
                }
            )
        return summary


def load_ephemeris_cache(
    path: str | Path,
    *,
    expected_satellites: int | None = None,
    required_steps: int | None = None,
    expected_step_duration_seconds: float | None = None,
) -> EphemerisCache:
    cache_path = Path(path).expanduser().resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(f"Ephemeris cache not found: {cache_path}")

    with np.load(cache_path, allow_pickle=False) as payload:
        required = {
            "satellite_names",
            "position_eci_km",
            "velocity_eci_km_s",
            "position_ecef_km",
            "velocity_ecef_km_s",
            "metadata_json",
        }
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(
                f"Ephemeris cache {cache_path} is missing arrays: {sorted(missing)}"
            )
        names = tuple(str(item) for item in payload["satellite_names"].tolist())
        plane_ids = (
            np.asarray(payload["plane_ids"], dtype=np.int32)
            if "plane_ids" in payload.files
            else None
        )
        orbital_elements = (
            tuple(json.loads(str(payload["orbital_elements_json"].item())))
            if "orbital_elements_json" in payload.files
            else None
        )
        position_eci = np.asarray(payload["position_eci_km"], dtype=np.float64)
        velocity_eci = np.asarray(payload["velocity_eci_km_s"], dtype=np.float64)
        position_ecef = np.asarray(payload["position_ecef_km"], dtype=np.float64)
        velocity_ecef = np.asarray(payload["velocity_ecef_km_s"], dtype=np.float64)
        metadata_raw = payload["metadata_json"]
        metadata = json.loads(str(metadata_raw.item()))

    if metadata.get("format") not in {
        EPHEMERIS_FORMAT,
        LEGACY_EPHEMERIS_FORMAT,
    }:
        raise ValueError(
            f"Unsupported ephemeris format {metadata.get('format')!r}; "
            f"expected {EPHEMERIS_FORMAT!r}."
        )
    if position_eci.ndim != 3 or position_eci.shape[-1] != 3:
        raise ValueError("position_eci_km must have shape [steps, satellites, 3].")
    if velocity_eci.shape != position_eci.shape:
        raise ValueError("velocity_eci_km_s must match position_eci_km.")
    if position_ecef.shape != position_eci.shape:
        raise ValueError("position_ecef_km must match position_eci_km.")
    if velocity_ecef.shape != position_eci.shape:
        raise ValueError("velocity_ecef_km_s must match position_eci_km.")
    if len(names) != position_eci.shape[1]:
        raise ValueError("satellite_names length does not match ephemeris arrays.")
    if plane_ids is not None and plane_ids.shape != (position_eci.shape[1],):
        raise ValueError("plane_ids must have shape [satellites].")
    if orbital_elements is not None and len(orbital_elements) != position_eci.shape[1]:
        raise ValueError(
            "orbital_elements_json length does not match ephemeris arrays."
        )
    if not (
        np.all(np.isfinite(position_eci))
        and np.all(np.isfinite(velocity_eci))
        and np.all(np.isfinite(position_ecef))
        and np.all(np.isfinite(velocity_ecef))
    ):
        raise ValueError("Ephemeris cache contains NaN or infinite values.")

    cache = EphemerisCache(
        path=cache_path,
        satellite_names=names,
        plane_ids=plane_ids,
        orbital_elements=orbital_elements,
        position_eci_km=position_eci,
        velocity_eci_km_s=velocity_eci,
        position_ecef_km=position_ecef,
        velocity_ecef_km_s=velocity_ecef,
        metadata=metadata,
        sha256=sha256_file(cache_path),
    )
    if expected_satellites is not None and cache.num_satellites != expected_satellites:
        raise ValueError(
            f"Ephemeris has {cache.num_satellites} satellites, but the environment "
            f"was configured for {expected_satellites}."
        )
    if required_steps is not None and cache.num_steps < required_steps:
        raise ValueError(
            f"Ephemeris has {cache.num_steps} steps, but at least {required_steps} "
            "are required for the horizon plus planning lookahead."
        )
    if expected_step_duration_seconds is not None and not np.isclose(
        cache.step_duration_seconds,
        expected_step_duration_seconds,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            f"Ephemeris step is {cache.step_duration_seconds}s, but the environment "
            f"uses {expected_step_duration_seconds}s."
        )
    return cache


def load_task_catalog(path: str | Path) -> list[dict[str, str]]:
    catalog_path = Path(path).expanduser().resolve()
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Task catalog not found: {catalog_path}")
    with catalog_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Task catalog has no header: {catalog_path}")
        aliases = {
            "lat": "latitude_deg",
            "latitude": "latitude_deg",
            "lon": "longitude_deg",
            "lng": "longitude_deg",
            "longitude": "longitude_deg",
        }
        rows: list[dict[str, str]] = []
        for raw in reader:
            row = {
                aliases.get(str(key).strip().lower(), str(key).strip().lower()): (
                    "" if value is None else str(value).strip()
                )
                for key, value in raw.items()
                if key is not None
            }
            if not any(row.values()):
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"Task catalog contains no data rows: {catalog_path}")
    required = {"latitude_deg", "longitude_deg"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(
            f"Task catalog must contain latitude/longitude columns; missing "
            f"{sorted(missing)} in {catalog_path}."
        )
    return rows
