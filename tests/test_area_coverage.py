from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sat_marl_env import EnvConfig, SatTaskingEnv
from sat_marl_env.area_coverage import (
    clip_polygon_to_target,
    low_discrepancy_rectangle_samples,
    oriented_strip_polygon,
    polygon_area,
    strip_sample_mask,
)
from sat_marl_env.entities import TaskState


class AreaCoverageTest(unittest.TestCase):
    def test_directed_strip_uses_continuous_polygon_clipping(self) -> None:
        footprint = oriented_strip_polygon(0.0, 0.0, 100.0, 10.0, 0.0)
        clipped = clip_polygon_to_target(footprint, 40.0, 20.0)
        self.assertAlmostEqual(polygon_area(footprint), 1000.0, places=6)
        self.assertAlmostEqual(polygon_area(clipped), 400.0, places=6)
        diagonal = oriented_strip_polygon(0.0, 0.0, 100.0, 10.0, 45.0)
        self.assertNotAlmostEqual(
            polygon_area(clip_polygon_to_target(diagonal, 40.0, 20.0)),
            polygon_area(clipped),
        )

    def test_low_discrepancy_union_mask_is_not_a_regular_grid(self) -> None:
        samples = low_discrepancy_rectangle_samples(40.0, 20.0, 256)
        self.assertEqual(samples.shape, (256, 2))
        self.assertGreater(len(np.unique(samples[:, 0])), 200)
        first = strip_sample_mask(samples, -10.0, 0.0, 20.0, 20.0, 0.0)
        second = strip_sample_mask(samples, 10.0, 0.0, 20.0, 20.0, 0.0)
        self.assertGreater((first | second).bit_count() / len(samples), 0.98)
        self.assertLess((first & second).bit_count() / len(samples), 0.02)

    def _area_env(self, width: float, height: float) -> SatTaskingEnv:
        cfg = EnvConfig(
            num_satellites=2,
            num_tasks=1,
            num_planes=1,
            num_ground_stations=1,
            max_steps=8,
            candidate_k=2,
            neighbor_k=1,
            min_observation_elevation_deg=0.0,
            stabilization_seconds=0.0,
            payload_warmup_seconds=0.0,
            area_coverage_samples=256,
            area_observation_seconds=20.0,
            random_seed=23,
        )
        env = SatTaskingEnv(cfg)
        env.reset(seed=23)
        for sat in env._satellites:
            sat.orbital_elements.mean_anomaly_deg = 0.0
            sat.energy = cfg.max_energy
            sat.storage = 0.0
            sat.payload_mode = "optical"
            sat.payload_on = True
            sat.resolution_m = 1.0
            sat.swath_width_km = 80.0
        env._refresh_geometry()
        env._tasks[0] = TaskState(
            task_id=0,
            target_phase=0.0,
            target_lat_deg=float(env._sat_geometry[0]["latitude_deg"]),
            target_lon_deg=float(env._sat_geometry[0]["longitude_deg"]),
            priority=5.0,
            release_step=0,
            deadline_step=7,
            energy_cost=1.0,
            observation_duration_seconds=20.0,
            required_mode="optical",
            required_resolution_m=5.0,
            min_sun_elevation_deg=-90.0,
            original_data_mb=20.0,
            compression_ratio=0.2,
            target_type="area",
            area_width_km=width,
            area_height_km=height,
            area_orientation_deg=30.0,
            coverage_threshold=0.95,
            cooperation_mode="coverage",
        )
        env._derive_area_requirements()
        env._observe_all()
        return env

    def test_small_area_becomes_single_satellite_one_shot(self) -> None:
        env = self._area_env(20.0, 20.0)
        task = env.tasks[0]
        self.assertEqual(task.required_strips, 1)
        self.assertEqual(task.min_contributing_satellites, 1)
        self.assertEqual(task.cooperation_mode, "single")
        self.assertEqual(
            int(env._last_action_masks["satellite_0"][2]), 1
        )
        env.step({"satellite_0": 2, "satellite_1": 0})
        self.assertIsNotNone(task.completed_by)
        self.assertGreaterEqual(task.coverage_fraction, task.coverage_threshold)
        self.assertEqual(env.summary()["area_completed_tasks"], 1)

    def test_large_area_derives_multi_satellite_requirement(self) -> None:
        env = self._area_env(260.0, 260.0)
        task = env.tasks[0]
        self.assertGreater(task.required_strips, 1)
        self.assertEqual(task.min_contributing_satellites, 2)
        self.assertEqual(task.cooperation_mode, "coverage")
        self.assertTrue(math.isclose(task.coverage_fraction, 0.0))

    def test_complementary_strips_complete_joint_area_request(self) -> None:
        env = self._area_env(40.0, 20.0)
        task = env._tasks[0]
        task.required_strips = 2
        task.required_observers = 2
        task.min_contributing_satellites = 2
        task.cooperation_mode = "coverage"
        task.coverage_threshold = 0.95
        samples = env._area_samples[0]

        def claim(agent: str, center_u: float):
            mask = strip_sample_mask(
                samples, center_u, 0.0, 20.0, 20.0, 0.0
            )
            footprint = oriented_strip_polygon(
                center_u, 0.0, 20.0, 20.0, 0.0
            )
            plan = {
                "finish_step": 0,
                "total_energy": 1.0,
                "storage_required_mb": 1.0,
                "target_roll_deg": 0.0,
                "target_pitch_deg": 0.0,
                "target_yaw_deg": 0.0,
                "warmup_steps": 0,
                "quality": 1.0,
                "observation_start_step": 0,
                "maneuver_steps": 0,
                "ground_track_heading_deg": 0.0,
                "strip_length_km": 20.0,
                "strip_width_km": 20.0,
                "inside_area_km2": 400.0,
                "outside_area_km2": 0.0,
                "outside_ratio": 0.0,
                "coverage_sample_mask": mask,
                "marginal_coverage_fraction": mask.bit_count() / len(samples),
                "strip_footprint_local": footprint,
            }
            return agent, 1.0, 0.0, plan

        rewards = {agent: 0.0 for agent in env.agents}
        infos = {agent: {} for agent in env.agents}
        values = env._resolve_task_claims(
            {0: [claim("satellite_0", -10.0), claim("satellite_1", 10.0)]},
            rewards,
            infos,
        )
        self.assertEqual(values, [task.priority])
        self.assertIsNotNone(task.completed_by)
        self.assertEqual(task.completed_by_satellites, [0, 1])
        self.assertGreaterEqual(task.coverage_fraction, 0.98)
        self.assertEqual(task.strip_count, 2)


if __name__ == "__main__":
    unittest.main()
