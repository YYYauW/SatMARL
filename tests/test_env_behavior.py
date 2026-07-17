from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sat_marl_env import EnvConfig, SatTaskingEnv
from sat_marl_env.entities import DataPacketState, TaskState


class SatTaskingEnvBehaviorTest(unittest.TestCase):
    def _coincident_two_satellite_env(self, cooperation_mode: str = "single"):
        cfg = EnvConfig(
            num_satellites=2,
            num_tasks=1,
            num_planes=1,
            num_ground_stations=1,
            max_steps=6,
            candidate_k=2,
            neighbor_k=1,
            min_observation_elevation_deg=0.0,
            stabilization_seconds=0.0,
            payload_warmup_seconds=0.0,
            random_seed=3,
        )
        env = SatTaskingEnv(cfg)
        env.reset(seed=3)
        for sat in env._satellites:
            sat.orbital_elements.mean_anomaly_deg = 0.0
            sat.energy = cfg.max_energy
            sat.storage = 0.0
            sat.payload_mode = "optical"
            sat.payload_on = True
            sat.resolution_m = 1.0
            sat.swath_width_km = 100.0
            sat.roll_deg = 0.0
            sat.pitch_deg = 0.0
            sat.yaw_deg = 0.0
        env._refresh_geometry()
        latitude = float(env._sat_geometry[0]["latitude_deg"])
        longitude = float(env._sat_geometry[0]["longitude_deg"])
        required = 2 if cooperation_mode == "simultaneous" else 1
        env._tasks[0] = TaskState(
            task_id=0,
            target_phase=0.0,
            target_lat_deg=latitude,
            target_lon_deg=longitude,
            priority=5.0,
            release_step=0,
            deadline_step=5,
            energy_cost=1.0,
            required_mode="optical",
            required_resolution_m=5.0,
            required_swath_km=10.0,
            min_sun_elevation_deg=-90.0,
            original_data_mb=10.0,
            compression_ratio=0.2,
            cooperation_mode=cooperation_mode,
            required_observers=required,
        )
        return cfg, env, env._observe_all()

    def test_competing_satellites_complete_task_once(self) -> None:
        _, env, observations = self._coincident_two_satellite_env()
        self.assertEqual(int(observations["satellite_0"]["candidate_ids"][0]), 0)
        self.assertEqual(int(observations["satellite_1"]["candidate_ids"][0]), 0)

        observations, _, _, _, _ = env.step(
            {"satellite_0": 2, "satellite_1": 2}
        )
        summary = env.summary()
        self.assertEqual(summary["completed_tasks"], 1)
        self.assertEqual(summary["total_conflicts"], 1)
        self.assertTrue((observations["satellite_0"]["candidate_ids"] == -1).all())

    def test_simultaneous_task_requires_two_satellites(self) -> None:
        _, env, observations = self._coincident_two_satellite_env("simultaneous")
        self.assertEqual(int(observations["satellite_0"]["action_mask"][2]), 1)
        _, rewards, _, _, infos = env.step({"satellite_0": 2, "satellite_1": 2})
        task = env.tasks[0]
        self.assertEqual(task.completed_by_satellites, [0, 1])
        self.assertEqual(env.summary()["cooperative_completed_tasks"], 1)
        self.assertGreater(env.summary()["task_reward_total"], 0.0)
        self.assertTrue(any(info.get("completion_bonus", 0.0) > 0.0 for info in infos.values()))
        self.assertGreater(sum(rewards.values()), 0.0)

    def test_ground_station_capacity_and_segmented_downlink(self) -> None:
        cfg = EnvConfig(
            num_satellites=2,
            num_tasks=1,
            num_planes=1,
            num_ground_stations=1,
            max_steps=4,
            ground_station_channels=1,
            ground_station_rate_mb_per_step=5.0,
            random_seed=9,
        )
        env = SatTaskingEnv(cfg)
        observations, _ = env.reset(seed=9)
        for sat in env._satellites:
            packet = DataPacketState(
                task_id=0,
                generated_step=0,
                original_size_mb=20.0,
                compressed_size_mb=10.0,
                remaining_mb=10.0,
                priority=5.0,
                source_satellite_id=sat.sat_id,
            )
            sat.packets = [packet]
            sat.storage = 10.0
        env._station_cache = [[(0, 45.0)], [(0, 45.0)]]
        env._last_candidate_ids = {
            agent: np.full(cfg.candidate_k, -1, dtype=np.int32)
            for agent in env.agents
        }

        env.step({"satellite_0": 1, "satellite_1": 1})
        summary = env.summary()
        self.assertGreater(summary["downlinked_data_mb"], 0.0)
        self.assertLess(summary["downlinked_data_mb"], 5.0)
        self.assertEqual(summary["total_ground_conflicts"], 1)
        self.assertEqual(summary["pending_packets"], 2)

    def test_local_neighbors_only_use_defined_relations(self) -> None:
        cfg = EnvConfig(
            num_satellites=4,
            num_tasks=1,
            num_planes=2,
            num_ground_stations=1,
            neighbor_k=3,
            max_steps=4,
            random_seed=4,
        )
        env = SatTaskingEnv(cfg)
        env.reset(seed=4)
        env._station_cache = [[], [], [], []]
        env._satellites[2].last_action_category = 2
        empty_candidates = [np.empty(0, dtype=np.int32) for _ in env.satellites]
        env._prepare_local_neighborhoods(empty_candidates)
        features, action_mean = env._neighbor_features(0, empty_candidates)
        self.assertEqual(float(features[0, 1]), 1.0)
        self.assertTrue((features[1:] == 0.0).all())
        self.assertAlmostEqual(float(action_mean[2]), 1.0)

    def test_orbit_is_propagated_from_six_elements(self) -> None:
        cfg = EnvConfig(num_satellites=1, num_tasks=1, max_steps=3, random_seed=5)
        env = SatTaskingEnv(cfg)
        observations, _ = env.reset(seed=5)
        before = env._sat_geometry[0]["ecef"].copy()
        elements = env.satellites[0].orbital_elements
        self.assertGreater(elements.semi_major_axis_km, cfg.earth_radius_km)
        env.step({"satellite_0": 0})
        after = env._sat_geometry[0]["ecef"]
        self.assertFalse(np.allclose(before, after))
        self.assertEqual(observations["satellite_0"]["self"].shape, (env.self_dim,))

    def test_point_target_uses_five_second_acquisition_and_45_degree_fov(self) -> None:
        cfg, env, _ = self._coincident_two_satellite_env()
        sat = env.satellites[0]
        task = env.tasks[0]
        self.assertEqual(sat.field_of_view_deg, 45.0)
        self.assertEqual(task.observation_duration_seconds, 5.0)
        feasible, reason, plan = env._plan_task(sat, task)
        self.assertTrue(feasible, reason)
        self.assertAlmostEqual(
            float(plan["observation_finish_seconds"])
            - float(plan["observation_start_seconds"]),
            5.0,
        )

    def test_default_horizon_covers_a_complete_leo_orbit(self) -> None:
        cfg = EnvConfig()
        self.assertGreaterEqual(cfg.max_steps * cfg.step_duration_seconds, 7200.0)
        self.assertEqual(cfg.payload_fov_min_deg, 45.0)
        self.assertEqual(cfg.payload_fov_max_deg, 45.0)
        self.assertGreaterEqual(cfg.num_tasks, 3 * cfg.max_steps)

    def test_forced_idle_is_not_penalized(self) -> None:
        cfg = EnvConfig(num_satellites=1, num_tasks=1, max_steps=2, random_seed=12)
        env = SatTaskingEnv(cfg)
        env.reset(seed=12)
        env._tasks[0].expired = True
        observations = env._observe_all()
        self.assertFalse(np.any(observations["satellite_0"]["action_mask"][1:]))
        _, rewards, _, _, infos = env.step({"satellite_0": 0})
        self.assertEqual(rewards["satellite_0"], cfg.forced_idle_penalty)
        self.assertEqual(infos["satellite_0"]["event"], "forced_idle")

    def test_decision_statistics_reset_between_episodes(self) -> None:
        cfg = EnvConfig(num_satellites=2, num_tasks=4, max_steps=2, random_seed=13)
        env = SatTaskingEnv(cfg)
        env.reset(seed=13)
        env.step({agent: 0 for agent in env.agents})
        self.assertGreater(env.summary()["decision_count"], 0)
        env.reset(seed=14)
        summary = env.summary()
        self.assertEqual(summary["decision_count"], 0)
        self.assertEqual(summary["forced_idle_actions"], 0)


if __name__ == "__main__":
    unittest.main()
