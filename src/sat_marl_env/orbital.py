from __future__ import annotations

import math

import numpy as np

from .entities import OrbitalElements


def geodetic_to_ecef(latitude_deg: float, longitude_deg: float, radius_km: float) -> np.ndarray:
    lat = math.radians(latitude_deg)
    lon = math.radians(longitude_deg)
    cos_lat = math.cos(lat)
    return radius_km * np.asarray(
        [cos_lat * math.cos(lon), cos_lat * math.sin(lon), math.sin(lat)],
        dtype=np.float64,
    )


def ecef_to_spherical(position_km: np.ndarray, radius_km: float) -> tuple[float, float, float]:
    radius = float(np.linalg.norm(position_km))
    lat = math.degrees(math.asin(float(position_km[2]) / max(radius, 1e-9)))
    lon = math.degrees(math.atan2(float(position_km[1]), float(position_km[0])))
    return lat, lon, radius - radius_km


def propagate_kepler(
    elements: OrbitalElements,
    elapsed_seconds: float,
    mu_km3_s2: float,
    earth_rotation_rate_rad_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate osculating elements with a two-body Kepler model."""

    a = elements.semi_major_axis_km
    e = elements.eccentricity
    inc = math.radians(elements.inclination_deg)
    raan = math.radians(elements.raan_deg)
    argp = math.radians(elements.argument_of_perigee_deg)
    mean_motion = math.sqrt(mu_km3_s2 / (a**3))
    mean_anomaly = math.radians(elements.mean_anomaly_deg) + mean_motion * elapsed_seconds
    mean_anomaly = (mean_anomaly + math.pi) % (2.0 * math.pi) - math.pi

    eccentric_anomaly = mean_anomaly
    for _ in range(8):
        denominator = 1.0 - e * math.cos(eccentric_anomaly)
        eccentric_anomaly -= (
            eccentric_anomaly - e * math.sin(eccentric_anomaly) - mean_anomaly
        ) / max(denominator, 1e-10)

    cos_e = math.cos(eccentric_anomaly)
    sin_e = math.sin(eccentric_anomaly)
    radius = a * (1.0 - e * cos_e)
    true_anomaly = math.atan2(math.sqrt(1.0 - e * e) * sin_e, cos_e - e)
    p = a * (1.0 - e * e)
    position_pf = np.asarray(
        [radius * math.cos(true_anomaly), radius * math.sin(true_anomaly), 0.0]
    )
    velocity_pf = math.sqrt(mu_km3_s2 / p) * np.asarray(
        [-math.sin(true_anomaly), e + math.cos(true_anomaly), 0.0]
    )

    cos_o, sin_o = math.cos(raan), math.sin(raan)
    cos_i, sin_i = math.cos(inc), math.sin(inc)
    cos_w, sin_w = math.cos(argp), math.sin(argp)
    rotation = np.asarray(
        [
            [cos_o * cos_w - sin_o * sin_w * cos_i, -cos_o * sin_w - sin_o * cos_w * cos_i, sin_o * sin_i],
            [sin_o * cos_w + cos_o * sin_w * cos_i, -sin_o * sin_w + cos_o * cos_w * cos_i, -cos_o * sin_i],
            [sin_w * sin_i, cos_w * sin_i, cos_i],
        ]
    )
    position_eci = rotation @ position_pf
    velocity_eci = rotation @ velocity_pf

    theta = earth_rotation_rate_rad_s * elapsed_seconds
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    eci_to_ecef = np.asarray(
        [[cos_t, sin_t, 0.0], [-sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]]
    )
    position_ecef = eci_to_ecef @ position_eci
    return position_eci, velocity_eci, position_ecef


def target_geometry(
    satellite_ecef: np.ndarray,
    target_ecef: np.ndarray,
) -> tuple[float, float, float, bool]:
    los = target_ecef - satellite_ecef
    slant_range = float(np.linalg.norm(los))
    los_unit = los / max(slant_range, 1e-9)
    target_up = target_ecef / max(float(np.linalg.norm(target_ecef)), 1e-9)
    sat_to_target_elevation = math.degrees(
        math.asin(float(np.clip(np.dot(-los_unit, target_up), -1.0, 1.0)))
    )
    nadir = -satellite_ecef / max(float(np.linalg.norm(satellite_ecef)), 1e-9)
    off_nadir = math.degrees(
        math.acos(float(np.clip(np.dot(nadir, los_unit), -1.0, 1.0)))
    )
    earth_visible = sat_to_target_elevation >= 0.0
    return sat_to_target_elevation, off_nadir, slant_range, earth_visible


def target_attitude_lvlh(
    satellite_eci: np.ndarray,
    velocity_eci: np.ndarray,
    satellite_ecef: np.ndarray,
    target_ecef: np.ndarray,
    earth_rotation_rate_rad_s: float,
    elapsed_seconds: float,
) -> tuple[float, float, float]:
    theta = earth_rotation_rate_rad_s * elapsed_seconds
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    ecef_to_eci = np.asarray(
        [[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]]
    )
    los_eci = ecef_to_eci @ (target_ecef - satellite_ecef)
    los_eci /= max(float(np.linalg.norm(los_eci)), 1e-9)
    radial = satellite_eci / max(float(np.linalg.norm(satellite_eci)), 1e-9)
    nadir = -radial
    along = velocity_eci - np.dot(velocity_eci, radial) * radial
    along /= max(float(np.linalg.norm(along)), 1e-9)
    cross = np.cross(along, nadir)
    cross /= max(float(np.linalg.norm(cross)), 1e-9)
    nadir_component = max(1e-9, float(np.dot(los_eci, nadir)))
    roll = math.degrees(math.atan2(float(np.dot(los_eci, cross)), nadir_component))
    pitch = math.degrees(math.atan2(float(np.dot(los_eci, along)), nadir_component))
    return roll, pitch, 0.0


def target_attitude_ecef(
    satellite_ecef: np.ndarray,
    velocity_ecef: np.ndarray,
    target_ecef: np.ndarray,
) -> tuple[float, float, float]:
    """Return LVLH pointing angles using a frame-consistent ECEF state."""

    los = target_ecef - satellite_ecef
    los /= max(float(np.linalg.norm(los)), 1e-9)
    radial = satellite_ecef / max(float(np.linalg.norm(satellite_ecef)), 1e-9)
    nadir = -radial
    along = velocity_ecef - np.dot(velocity_ecef, radial) * radial
    along /= max(float(np.linalg.norm(along)), 1e-9)
    cross = np.cross(along, nadir)
    cross /= max(float(np.linalg.norm(cross)), 1e-9)
    nadir_component = max(1e-9, float(np.dot(los, nadir)))
    roll = math.degrees(math.atan2(float(np.dot(los, cross)), nadir_component))
    pitch = math.degrees(math.atan2(float(np.dot(los, along)), nadir_component))
    return roll, pitch, 0.0


def sun_unit_ecef(day_of_year: float, utc_hour: float) -> np.ndarray:
    declination = math.radians(
        23.44 * math.sin(2.0 * math.pi * (day_of_year - 81.0) / 365.25)
    )
    subsolar_longitude = math.radians((12.0 - utc_hour) * 15.0)
    return np.asarray(
        [
            math.cos(declination) * math.cos(subsolar_longitude),
            math.cos(declination) * math.sin(subsolar_longitude),
            math.sin(declination),
        ],
        dtype=np.float64,
    )


def sun_elevation_deg(target_ecef: np.ndarray, sun_ecef_unit: np.ndarray) -> float:
    target_up = target_ecef / max(float(np.linalg.norm(target_ecef)), 1e-9)
    return math.degrees(
        math.asin(float(np.clip(np.dot(target_up, sun_ecef_unit), -1.0, 1.0)))
    )


def satellite_is_sunlit(
    satellite_ecef: np.ndarray,
    sun_ecef_unit: np.ndarray,
    earth_radius_km: float,
) -> bool:
    projection = float(np.dot(satellite_ecef, sun_ecef_unit))
    if projection >= 0.0:
        return True
    perpendicular = satellite_ecef - projection * sun_ecef_unit
    return float(np.linalg.norm(perpendicular)) > earth_radius_km
