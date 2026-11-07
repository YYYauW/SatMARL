from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from sat_marl_env import EnvConfig
from sat_marl_env.entities import DataPacketState
from sat_marl_env.transmission import (
    TransmissionConfig, TransmissionEnv, elevation_windows, peak_rolling_work,
)


class TransmissionTest(unittest.TestCase):
    def env(self, satellites=1, **kwargs):
        cfg = TransmissionConfig(num_satellites=satellites, num_planes=1,
                                 num_tasks=1, num_ground_stations=1, max_steps=8,
                                 candidate_k=2, neighbor_k=1, **kwargs)
        env = TransmissionEnv(cfg)
        env.reset(seed=3)
        return env

    def add_packet(self, env, sat_id, size, ready=0.0):
        sat = env._satellites[sat_id]
        packet = DataPacketState(0, 0, size*4, size, size, 5.0, sat_id,
                                 generated_time_seconds=ready)
        sat.packets.append(packet)
        sat.storage += size
        return packet

    def resolve(self, env, ids):
        rewards = {a: 0.0 for a in env.agents}
        infos = {a: {} for a in env.agents}
        env._resolve_downlinks({0: [(env.agents[i], 0, 45.0) for i in ids]}, rewards, infos)
        return infos

    def test_defaults_and_archived_values_are_separate(self):
        self.assertEqual(EnvConfig().max_storage, 160)
        self.assertEqual(EnvConfig().ground_station_rate_mb_per_step, 32)
        cfg = TransmissionConfig()
        self.assertEqual(cfg.imaging_budget_seconds, 1800)
        self.assertEqual(cfg.imaging_window_seconds, 5400)
        self.assertEqual(cfg.downlink_rate_mbps, 900)

    def test_window_crossing_both_bounds(self):
        windows = elevation_windows(lambda t: 5+2*t, 0, 30, 10, 50, 1, .001)
        self.assertAlmostEqual(windows[0][0], 2.5, delta=.001)
        self.assertAlmostEqual(windows[0][1], 22.5, delta=.001)

    def test_zenith_exclusion_splits_pass(self):
        windows = elevation_windows(lambda t: 90-2*abs(t-15), 0, 30, 10, 80, 1, .001)
        self.assertEqual(len(windows), 2)
        self.assertAlmostEqual(windows[0][1], 10, delta=.001)
        self.assertAlmostEqual(windows[1][0], 20, delta=.001)

    def test_no_contact_and_full_contact(self):
        self.assertEqual(elevation_windows(lambda _: 5, 0, 30, 10, 90), [])
        self.assertEqual(elevation_windows(lambda _: 45, 0, 30, 10, 90), [(0, 30)])

    def test_exact_rolling_budget_crosses_90_minute_boundary(self):
        self.assertEqual(peak_rolling_work([(0, 1800), (5400, 7200)], 5400), 1800)
        self.assertEqual(peak_rolling_work([(0, 1800), (5395, 5400)], 5400), 1805)

    def test_budget_checks_future_reservations(self):
        env = self.env()
        env.config.max_steps = 300
        env._imaging_intervals[0] = [(0, 1800)]
        task = env.tasks[0]
        task.deadline_step = 299
        reason = env._additional_plan_failure(env.satellites[0], task,
                    {"observation_start_seconds": 2000, "observation_finish_seconds": 2005})
        self.assertEqual(reason, "imaging_duty_limit")
        self.assertEqual(env._additional_plan_failure(env.satellites[0], task,
                    {"observation_start_seconds": 5400, "observation_finish_seconds": 5405}), "ok")

    def test_per_acquisition_data_not_divided_by_observers(self):
        env = self.env()
        task = env.tasks[0]
        task.original_data_mb, task.required_observers = 4000, 5
        self.assertEqual(env._observation_storage_mb(task), 1000)
        self.assertEqual(task.compression_ratio, .25)

    def test_full_slot_capacity_and_multiple_packets(self):
        env = self.env()
        for _ in range(4):
            self.add_packet(env, 0, 1000)
        env._contacts = {(0, 0): [(0, 30)]}
        env._station_cache = [[(0, 45)]]
        self.resolve(env, [0])
        self.assertAlmostEqual(env.summary()["downlinked_data_mb"], 2700)
        self.assertEqual(len(env.satellites[0].packets), 2)
        self.assertAlmostEqual(env.satellites[0].packets[0].remaining_mb, 300)

    def test_partial_contact_not_whole_slot(self):
        env = self.env()
        self.add_packet(env, 0, 4000)
        env._contacts = {(0, 0): [(12, 22)]}
        env._station_cache = [[(0, 45)]]
        self.resolve(env, [0])
        self.assertAlmostEqual(env.satellites[0].downlinked_mb, 900)
        self.assertNotIn(0, env._sessions)

    def test_no_transmission_before_acquisition_finishes(self):
        env = self.env()
        packet = self.add_packet(env, 0, 2000, ready=20)
        amount, _, _ = env._transmit_interval(env.satellites[0], 0, 30)
        self.assertAlmostEqual(amount, 900)
        self.assertAlmostEqual(packet.remaining_mb, 1100)

    def test_future_packet_never_extends_allocated_contact(self):
        env = self.env()
        self.add_packet(env, 0, 900)
        future = self.add_packet(env, 0, 900, ready=100)
        amount, _, finish = env._transmit_interval(env.satellites[0], 0, 30)
        self.assertAlmostEqual(amount, 900)
        self.assertLessEqual(finish, 30)
        self.assertAlmostEqual(future.remaining_mb, 900)

    def test_session_continues_without_a_second_action(self):
        env = self.env()
        self.add_packet(env, 0, 6000)
        env._contacts = {(0, 0): [(0, 30)]}
        env._station_cache = [[(0, 45)]]
        self.resolve(env, [0])
        self.assertIn(0, env._sessions)
        self.assertFalse(env._satellite_available(env.satellites[0]))
        env._step = 1
        env._contacts = {(0, 0): [(30, 40)]}
        self.resolve(env, [])
        self.assertAlmostEqual(env.satellites[0].downlinked_mb, 3600)
        self.assertNotIn(0, env._sessions)

    def test_receiver_reuse_after_queue_drains(self):
        env = self.env(satellites=2, ground_station_channels=1)
        self.add_packet(env, 0, 900)
        self.add_packet(env, 1, 900)
        env._contacts = {(0, 0): [(0, 30)], (1, 0): [(0, 30)]}
        env._station_cache = [[(0, 45)], [(0, 45)]]
        self.resolve(env, [0, 1])
        self.assertAlmostEqual(sum(s.downlinked_mb for s in env.satellites), 1800)
        self.assertEqual(env.summary()["total_ground_conflicts"], 0)

    def test_antenna_capacity_not_exceeded(self):
        env = self.env(satellites=2, ground_station_channels=1)
        for i in (0, 1):
            self.add_packet(env, i, 6000)
        env._contacts = {(0, 0): [(0, 30)], (1, 0): [(0, 30)]}
        env._station_cache = [[(0, 45)], [(0, 45)]]
        self.resolve(env, [0, 1])
        self.assertAlmostEqual(sum(s.downlinked_mb for s in env.satellites), 2700)
        self.assertEqual(env.summary()["total_ground_conflicts"], 1)

    def test_station_fallback(self):
        env = self.env(satellites=2, ground_station_channels=1)
        second = env._ground_stations[0]
        from dataclasses import replace
        env._ground_stations.append(replace(second, station_id=1))
        for i in (0, 1):
            self.add_packet(env, i, 6000)
        env._contacts = {(i, g): [(0, 30)] for i in (0, 1) for g in (0, 1)}
        env._station_cache = [[(0, 45), (1, 40)], [(0, 45), (1, 40)]]
        self.resolve(env, [0, 1])
        self.assertAlmostEqual(sum(s.downlinked_mb for s in env.satellites), 5400)

    def test_observation_and_delivery_metrics(self):
        env = self.env()
        task = env.tasks[0]
        task.original_data_mb = 4000
        task.required_observers = 1
        task.cooperation_mode = "single"
        plan = dict(finish_step=1, total_energy=0, storage_required_mb=1000,
                    target_roll_deg=0, target_pitch_deg=0, target_yaw_deg=0,
                    warmup_steps=0, observation_start_seconds=30,
                    observation_finish_seconds=35, quality=1, slew_steps=0,
                    maneuver_steps=0, observation_start_step=1)
        rewards, infos = {"satellite_0": 0.}, {"satellite_0": {}}
        env._commit_observation(task, ("satellite_0", 0, 0, plan), rewards, infos)
        self.assertEqual(env.summary()["admitted_completed_tasks"], 1)
        self.assertEqual(env.summary()["observation_success_tasks"], 0)
        env._step = 2
        self.assertEqual(env.summary()["observation_success_tasks"], 1)
        self.assertEqual(env.summary()["observation_and_downlink_success_tasks"], 0)
        env._transmit_interval(env.satellites[0], 60, 65)
        self.assertEqual(env.summary()["observation_and_downlink_success_tasks"], 0)
        env._transmit_interval(env.satellites[0], 65, 80)
        env._step = 3
        self.assertEqual(env.summary()["observation_and_downlink_success_tasks"], 1)

    def test_reset_clears_duty_and_delivery_records(self):
        env = self.env()
        env._sessions[0] = 0
        env._imaging_intervals[0] = [(0, 30)]
        env.reset(seed=3)
        self.assertFalse(env._sessions)
        self.assertFalse(env._imaging_intervals)
        self.assertFalse(env._acquisitions)

    def test_joint_delivery_needs_all_contributors(self):
        env = self.env(satellites=2)
        task = env.tasks[0]
        task.original_data_mb = 4000
        task.required_observers, task.cooperation_mode = 2, "sequential"
        task.deadline_step = 7
        plan = dict(finish_step=1, total_energy=0, storage_required_mb=1000,
                    target_roll_deg=0, target_pitch_deg=0, target_yaw_deg=0,
                    warmup_steps=0, observation_start_seconds=30,
                    observation_finish_seconds=35, quality=1, slew_steps=0,
                    maneuver_steps=0, observation_start_step=1)
        rewards = {a: 0. for a in env.agents}
        infos = {a: {} for a in env.agents}
        for sat_id in (0, 1):
            env._commit_observation(task, (env.agents[sat_id], 0, 0, plan), rewards, infos)
        env._step = 2
        self.assertEqual(env.summary()["observation_success_tasks"], 1)
        self.assertAlmostEqual(sum(s.storage for s in env.satellites), 2000)
        env._transmit_interval(env.satellites[0], 60, 80)
        env._step = 3
        self.assertEqual(env.summary()["observation_and_downlink_success_tasks"], 0)
        env._transmit_interval(env.satellites[1], 90, 110)
        env._step = 4
        self.assertEqual(env.summary()["observation_and_downlink_success_tasks"], 1)

    def test_full_horizon_available_for_delivery(self):
        env = self.env()
        env._tasks[0].expired = True
        _, _, term, trunc, _ = env.step({"satellite_0": 0})
        self.assertFalse(term["satellite_0"])
        self.assertFalse(trunc["satellite_0"])

    def test_real_geometry_observe_then_deliver_end_to_end(self):
        env = self.env(payload_warmup_seconds=0, stabilization_seconds=0,
                       payload_cooldown_seconds=0, optical_min_sun_elevation_deg=-90)
        sat, task, station = env.satellites[0], env.tasks[0], env.ground_stations[0]
        geo = env._sat_geometry[0]
        task.target_lat_deg = station.latitude_deg = geo["latitude_deg"]
        task.target_lon_deg = station.longitude_deg = geo["longitude_deg"]
        task.release_step, task.deadline_step = 0, 7
        task.required_observers, task.cooperation_mode = 1, "single"
        task.required_mode, task.required_resolution_m, task.required_swath_km = "optical", 2, 10
        task.min_sun_elevation_deg = -90
        task.original_data_mb = 4000
        sat.payload_on = True
        sat.roll_deg = sat.pitch_deg = sat.yaw_deg = 0
        sat.roll_rate_deg_s = sat.pitch_rate_deg_s = sat.yaw_rate_deg_s = 0
        obs = env._observe_all()
        self.assertEqual(obs["satellite_0"]["action_mask"][2], 1)
        obs, _, _, _, _ = env.step({"satellite_0": 2})
        self.assertEqual(env.summary()["observation_success_tasks"], 1)
        self.assertEqual(env.summary()["observation_and_downlink_success_tasks"], 0)
        self.assertEqual(obs["satellite_0"]["action_mask"][1], 1)
        env.step({"satellite_0": 1})
        self.assertEqual(env.summary()["observation_and_downlink_success_tasks"], 1)
        self.assertAlmostEqual(env.summary()["downlinked_data_mb"], 1000)
        self.assertAlmostEqual(env.summary()["pending_data_mb"], 0)

    def test_invalid_config_rejected(self):
        for settings in ({"downlink_rate_mbps": 0}, {"downlink_efficiency": 2},
                         {"max_ground_elevation_deg": 5},
                         {"imaging_budget_seconds": 5401},
                         {"num_ground_stations": 6}):
            with self.assertRaises(ValueError):
                TransmissionConfig(**settings)


if __name__ == "__main__":
    unittest.main()
