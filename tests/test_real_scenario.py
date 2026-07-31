from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sat_marl_env import EnvConfig, SatTaskingEnv
from sat_marl_env.real_scenario import EPHEMERIS_FORMAT, load_ephemeris_cache


class RealScenarioTest(unittest.TestCase):
    def _write_fixture(self, root: Path) -> tuple[Path, Path]:
        radius = 6378.137 + 550.0
        steps = 5
        eci = np.zeros((steps, 1, 3), dtype=np.float64)
        ecef = np.zeros_like(eci)
        velocity = np.zeros_like(eci)
        eci[:, 0, 0] = radius
        ecef[:, 0, 0] = radius
        velocity[:, 0, 1] = 7.5
        metadata = {
            "format": EPHEMERIS_FORMAT,
            "mode": "tle_sgp4_cache",
            "start_utc": "2026-07-20T12:00:00Z",
            "step_duration_seconds": 30.0,
            "tle_sha256": "fixture",
        }
        ephemeris = root / "ephemeris.npz"
        np.savez_compressed(
            ephemeris,
            satellite_names=np.asarray(["FIXTURE-SAT"]),
            position_eci_km=eci,
            velocity_eci_km_s=velocity,
            position_ecef_km=ecef,
            velocity_ecef_km_s=velocity,
            metadata_json=np.asarray(json.dumps(metadata)),
        )
        catalog = root / "targets.csv"
        with catalog.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "target_id",
                    "latitude_deg",
                    "longitude_deg",
                    "priority",
                    "release_step",
                    "deadline_step",
                    "required_mode",
                    "required_resolution_m",
                    "required_swath_km",
                    "min_sun_elevation_deg",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "target_id": "nadir",
                    "latitude_deg": 0,
                    "longitude_deg": 0,
                    "priority": 9,
                    "release_step": 0,
                    "deadline_step": 3,
                    "required_mode": "optical",
                    "required_resolution_m": 20,
                    "required_swath_km": 1,
                    "min_sun_elevation_deg": -90,
                }
            )
        return ephemeris, catalog

    def test_external_ephemeris_and_catalog_drive_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ephemeris, catalog = self._write_fixture(Path(temporary))
            cfg = EnvConfig(
                num_satellites=1,
                num_tasks=1,
                num_planes=1,
                num_ground_stations=1,
                max_steps=4,
                planning_lookahead_steps=1,
                candidate_k=1,
                neighbor_k=0,
                ephemeris_cache_path=str(ephemeris),
                task_catalog_path=str(catalog),
                task_layout="catalog",
                stabilization_seconds=0.0,
                payload_warmup_seconds=0.0,
                min_observation_elevation_deg=0.0,
                max_off_nadir_deg=45.0,
                random_seed=1,
            )
            env = SatTaskingEnv(cfg)
            observations, _ = env.reset(seed=1)
            np.testing.assert_allclose(
                env._sat_geometry[0]["ecef"], [6378.137 + 550.0, 0.0, 0.0]
            )
            self.assertEqual(env.tasks[0].target_lat_deg, 0.0)
            self.assertEqual(env.tasks[0].target_lon_deg, 0.0)
            self.assertEqual(env.summary()["scenario"]["orbit_source"]["mode"], "tle_sgp4_cache")
            self.assertEqual(int(observations["satellite_0"]["candidate_ids"][0]), 0)
            self.assertEqual(int(observations["satellite_0"]["action_mask"][1]), 0)
            self.assertEqual(int(observations["satellite_0"]["action_mask"][2]), 1)

    def test_cache_shape_validation_rejects_satellite_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ephemeris, _ = self._write_fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "configured for 2"):
                load_ephemeris_cache(ephemeris, expected_satellites=2)

    def test_walker_six_element_builder_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "walker.npz"
            command = [
                sys.executable,
                str(ROOT / "scripts" / "build_kepler_ephemeris.py"),
                "--output",
                str(output),
                "--start-utc",
                "2026-07-20T00:00:00Z",
                "--satellites",
                "4",
                "--planes",
                "2",
                "--max-steps",
                "4",
                "--lookahead-steps",
                "1",
                "--altitude-km",
                "550",
                "--inclination-deg",
                "97.6",
                "--walker-phasing",
                "1",
            ]
            completed = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            cache = load_ephemeris_cache(
                output,
                expected_satellites=4,
                required_steps=5,
                expected_step_duration_seconds=30.0,
            )
            self.assertEqual(
                cache.summary()["mode"], "keplerian_six_element_cache"
            )
            np.testing.assert_array_equal(cache.plane_ids, [0, 1, 0, 1])
            self.assertIsNotNone(cache.orbital_elements)
            assert cache.orbital_elements is not None
            self.assertAlmostEqual(
                float(cache.orbital_elements[0]["inclination_deg"]), 97.6
            )
            self.assertAlmostEqual(
                float(cache.orbital_elements[1]["raan_deg"]), 180.0
            )
            radii = np.linalg.norm(cache.position_eci_km[0], axis=1)
            np.testing.assert_allclose(radii, 6378.137 + 550.0, atol=10.0)
            self.assertTrue(output.with_suffix(".elements.csv").is_file())


if __name__ == "__main__":
    unittest.main()
