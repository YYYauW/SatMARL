from __future__ import annotations
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from sat_marl_env.entities import DataPacketState
from sat_marl_env.event_transmission import (
    ContinuousOrbit, EventTransmissionConfig, EventTransmissionEnv,
    axis_maneuver_seconds, axis_profile,
)
from sat_marl_env.orbital import ecef_to_spherical


class EventTransmissionTest(unittest.TestCase):
    def env(self, tasks=2, **kwargs):
        options = dict(num_satellites=1, num_planes=1, num_tasks=tasks,
                       num_ground_stations=1, max_steps=180,
                       planning_lookahead_steps=60, candidate_k=2, neighbor_k=1,
                       random_seed=3, optical_min_sun_elevation_deg=-90)
        options.update(kwargs)
        env = EventTransmissionEnv(EventTransmissionConfig(**options))
        env.reset(seed=3)
        return env

    def scene(self, **kwargs):
        env = self.env(**kwargs)
        sat = env.satellites[0]
        sat.roll_deg = sat.pitch_deg = sat.yaw_deg = 0
        sat.roll_rate_deg_s = sat.pitch_rate_deg_s = sat.yaw_rate_deg_s = 0
        for index, task in enumerate(env.tasks):
            position = env._position_at(sat, 60+index*20)
            lat, lon, _ = ecef_to_spherical(position, env.config.earth_radius_km)
            task.target_lat_deg, task.target_lon_deg = lat, lon
            task.release_step, task.deadline_step = 0, env.config.max_steps
            task.required_mode, task.required_observers = "optical", 1
            task.cooperation_mode = "single"
            task.required_resolution_m, task.required_swath_km = 2, 10
            task.min_sun_elevation_deg = -90
            task.original_data_mb = 4000
        env._candidate_snapshot = None
        return env, env._observe_all()

    def action(self, obs, task_id):
        ids = obs["satellite_0"]["candidate_ids"]
        return int(np.flatnonzero(ids == task_id)[0])+2

    def finish(self, env):
        target = env._acquisitions[-1].finish_seconds
        while env.step_count < target-1e-8:
            env.step({"satellite_0": 0})

    def test_default_horizon_is_a_day(self):
        cfg = EventTransmissionConfig()
        self.assertEqual(cfg.max_steps*cfg.step_duration_seconds, 86400)
        self.assertEqual(cfg.point_observation_seconds, 5)
        self.assertEqual(cfg.decision_clock, "event_seconds")
        self.assertEqual(cfg.payload_cooldown_seconds, 0)

    def test_fractional_triangular_maneuver_time(self):
        value = axis_maneuver_seconds(2, 1, .25)
        self.assertAlmostEqual(value, 2*math.sqrt(8))
        self.assertNotEqual(value % 30, 0)

    def test_trapezoidal_maneuver_time(self):
        self.assertAlmostEqual(axis_maneuver_seconds(10, 1, .25), 14)
        self.assertAlmostEqual(axis_maneuver_seconds(-10, 1, .25), 14)

    def test_axis_profile_obeys_rate_and_end_state(self):
        duration = axis_maneuver_seconds(10, 1, .25)
        for t in np.linspace(0, duration, 101):
            _, velocity = axis_profile(10, 1, .25, float(t))
            self.assertLessEqual(abs(velocity), 1+1e-9)
        position, velocity = axis_profile(10, 1, .25, duration)
        self.assertAlmostEqual(position, 10)
        self.assertAlmostEqual(velocity, 0)

    def test_cold_start_is_parallel_to_slew_and_settling(self):
        env, _ = self.scene()
        valid, reason, plan = env._plan_task(env.satellites[0], env.tasks[0])
        self.assertTrue(valid, reason)
        self.assertAlmostEqual(plan["observation_start_seconds"], 60, delta=.002)
        self.assertEqual(plan["warmup_steps"], 60)
        self.assertEqual(plan["stabilization_steps"], 5)
        self.assertEqual(plan["observation_finish_seconds"]-plan["observation_start_seconds"], 5)

    def test_warm_payload_does_not_repeat_startup(self):
        env, obs = self.scene()
        env.step({"satellite_0": self.action(obs, 0)})
        self.finish(env)
        self.assertEqual(env.summary()["imaging_startup_count"], 1)
        self.assertTrue(env.satellites[0].payload_on)
        valid, reason, plan = env._plan_task(env.satellites[0], env.tasks[1])
        self.assertTrue(valid, reason)
        self.assertEqual(plan["warmup_steps"], 0)
        self.assertLess(plan["observation_start_seconds"]-env.step_count, 60)
        obs = env._observe_all()
        env.step({"satellite_0": self.action(obs, 1)})
        self.finish(env)
        self.assertEqual(env.summary()["imaging_startup_count"], 1)
        self.assertEqual(env.acquisition_ledger()[1]["startup_seconds"], 0)

    def test_completion_releases_satellite_at_exact_second(self):
        env, obs = self.scene()
        env.step({"satellite_0": self.action(obs, 0)})
        finish = env._acquisitions[0].finish_seconds
        self.finish(env)
        self.assertAlmostEqual(env.step_count, finish)
        self.assertAlmostEqual(finish, 65, delta=.002)
        self.assertTrue(env._satellite_available(env.satellites[0]))
        self.assertEqual(env.summary()["observation_success_tasks"], 1)

    def test_new_imaging_power_cycle_requires_one_new_startup(self):
        env, obs = self.scene(payload_idle_shutdown_steps=10)
        env.step({"satellite_0": self.action(obs, 0)})
        self.finish(env)
        off = env.satellites[0].last_payload_use_step+10
        while env.step_count < off-1e-8:
            env.step({"satellite_0": 0})
        self.assertFalse(env.satellites[0].payload_on)
        sat, task = env.satellites[0], env.tasks[1]
        lat, lon, _ = ecef_to_spherical(env._position_at(sat, env.step_count+60),
                                       env.config.earth_radius_km)
        task.target_lat_deg, task.target_lon_deg = lat, lon
        env._candidate_snapshot = None
        env._observe_all()
        valid, reason, plan = env._plan_task(sat, task)
        self.assertTrue(valid, reason)
        self.assertEqual(plan["warmup_steps"], 60)

    def test_future_observation_schedules_power_on_once_near_acquisition(self):
        env, _ = self.scene(max_steps=300, planning_lookahead_steps=180)
        sat, task = env.satellites[0], env.tasks[0]
        task.release_step = 120
        lat, lon, _ = ecef_to_spherical(env._position_at(sat, 120), env.config.earth_radius_km)
        task.target_lat_deg, task.target_lon_deg = lat, lon
        env._candidate_snapshot = None
        obs = env._observe_all()
        env.step({"satellite_0": self.action(obs, 0)})
        self.assertAlmostEqual(env.acquisition_ledger()[0]["payload_power_on_seconds"], 60, delta=.002)
        if env.step_count < 60-1e-8:
            self.assertFalse(sat.payload_on)
            self.assertEqual(env._image_starts, 0)
        while env.step_count < 60-1e-8:
            env.step({"satellite_0": 0})
        self.assertEqual(env._image_starts, 1)
        self.assertTrue(sat.payload_on)

    def test_three_axes_can_maneuver_in_parallel(self):
        values = [axis_maneuver_seconds(angle, 1, .25) for angle in (2, 3, 10)]
        self.assertEqual(max(values), 14)
        self.assertLess(max(values), sum(values))

    def test_radio_startup_once_for_multiple_packets_and_resume(self):
        env = self.env(downlink_warmup_seconds=7.5)
        sat = env.satellites[0]
        env._radio_cycles = {0: [dict(planned_on=0, end=100, actual_on=None, ready=None),
                                  dict(planned_on=110, end=180, actual_on=None, ready=None)]}
        def add():
            sat.packets.append(DataPacketState(0, 0, 40, 10, 10, 1, 0,
                                              generated_time_seconds=0))
            sat.storage += 10
        add(); add()
        env._ensure_radio_power(0)
        self.assertFalse(env._radio_ready(0, 7.4))
        self.assertTrue(env._radio_ready(0, 7.5))
        env._transmit_interval(sat, 7.5, 8.5)
        self.assertFalse(sat.packets)
        add()
        env._ensure_radio_power(20)
        self.assertTrue(env._radio_ready(0, 20))
        self.assertEqual(env._radio_starts, 1)
        env._ensure_radio_power(110)
        self.assertEqual(env._radio_starts, 2)

    def test_radio_preheat_can_finish_before_aos(self):
        env = self.env(downlink_warmup_seconds=20)
        env.satellites[0].packets.append(DataPacketState(0, 0, 40, 10, 10, 1, 0,
                                                        generated_time_seconds=0))
        env._radio_cycles = {0: [dict(planned_on=80, end=180, actual_on=None, ready=None)]}
        env._ensure_radio_power(79)
        self.assertEqual(env._radio_starts, 0)
        env._ensure_radio_power(80)
        self.assertTrue(env._radio_ready(0, 100))

    def test_downlink_completes_at_fractional_second_and_keeps_radio_warm(self):
        env = self.env(tasks=1, downlink_warmup_seconds=7.5)
        sat, station = env.satellites[0], env.ground_stations[0]
        station.latitude_deg = env._sat_geometry[0]["latitude_deg"]
        station.longitude_deg = env._sat_geometry[0]["longitude_deg"]
        env.tasks[0].expired = True
        env._all_contacts.clear()
        env._candidate_snapshot = None
        def add():
            sat.packets.append(DataPacketState(0, 0, 3600, 900, 900, 1, 0,
                                              generated_time_seconds=0))
            sat.storage += 900
        add()
        env._observe_all()
        env.step({"satellite_0": 0})
        self.assertAlmostEqual(env.step_count, 7.5)
        env.step({"satellite_0": 1})
        self.assertAlmostEqual(env.step_count, 17.5)
        self.assertFalse(sat.packets)
        self.assertNotIn(0, env._sessions)
        add()
        env._observe_all()
        env.step({"satellite_0": 1})
        self.assertAlmostEqual(env.step_count, 27.5)
        self.assertEqual(env._radio_starts, 1)

    def test_full_day_clock_with_coarse_external_cache(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
        from run_transmission_load import prepare_orbit
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"orbit.npz"
            config = dict(protocol="optical-ground-delivery-event-v3",
                          env=dict(max_steps=86400, step_duration_seconds=1,
                                   planning_lookahead_steps=60, ephemeris_sampling_seconds=30),
                          planes={"1": 1}, start_utc="2026-08-01T00:00:00Z",
                          orbit=dict(altitude_km=550, inclination_deg=97.6,
                                     eccentricity=.001, walker_phasing=0))
            prepare_orbit(config, path, 1)
            env = self.env(tasks=1, max_steps=86400, ephemeris_cache_path=str(path),
                           planning_lookahead_steps=60)
            self.assertEqual(env._ephemeris.raw.step_duration_seconds, 30)
            env.tasks[0].expired = True
            env._candidate_snapshot = None
            env._observe_all()
            _, _, _, truncated, _ = env.step({"satellite_0": 0})
            self.assertTrue(truncated["satellite_0"])
            self.assertEqual(env.summary()["simulated_seconds"], 86400)
            self.assertEqual(env.summary()["simulated_hours"], 24)

    def test_radio_and_imaging_have_independent_startups(self):
        env, obs = self.scene()
        env.step({"satellite_0": self.action(obs, 0)})
        self.assertEqual(env._image_starts, 1)
        self.assertEqual(env._radio_starts, 0)

    def test_no_30_second_rounding_of_arbitrary_settling(self):
        env, _ = self.scene(stabilization_seconds=3.75, payload_warmup_seconds=0)
        valid, reason, plan = env._plan_task(env.satellites[0], env.tasks[0])
        self.assertTrue(valid, reason)
        self.assertEqual(plan["stabilization_steps"], 3.75)
        self.assertLess(plan["observation_start_seconds"], 60)
        self.assertNotAlmostEqual(plan["observation_start_seconds"] % 30, 0)

    def test_full_five_seconds_must_fit_before_deadline(self):
        env, _ = self.scene()
        env.tasks[0].deadline_step = 64.9
        env._plan_cache.clear()
        valid, _, _ = env._plan_task(env.satellites[0], env.tasks[0])
        self.assertFalse(valid)

    def test_data_conservation_after_scheduled_acquisition(self):
        env, obs = self.scene()
        env.step({"satellite_0": self.action(obs, 0)})
        self.finish(env)
        summary = env.summary()
        self.assertAlmostEqual(summary["committed_data_mb"],
                               summary["pending_data_mb"]+summary["downlinked_data_mb"])

    def test_reset_clears_power_cycles_and_event_clock(self):
        env, obs = self.scene()
        env.step({"satellite_0": self.action(obs, 0)})
        env.reset(seed=3)
        self.assertEqual(env.step_count, 0)
        self.assertEqual(env._image_starts, 0)
        self.assertEqual(env._radio_starts, 0)
        self.assertFalse(env._activity_records)

    def test_hermite_cache_accepts_fractional_seconds(self):
        class Raw:
            step_duration_seconds = 30
            num_steps = 2
            metadata, orbital_elements, plane_ids = {}, None, None
            position_eci_km = position_ecef_km = np.asarray([[[0., 0, 0]], [[30., 60, 90]]])
            velocity_eci_km_s = velocity_ecef_km_s = np.asarray([[[1., 2, 3]], [[1., 2, 3]]])
        cache = ContinuousOrbit(Raw())
        p, v, _ = cache.state(2.75, 0)
        np.testing.assert_allclose(p, [2.75, 5.5, 8.25])
        np.testing.assert_allclose(v, [1, 2, 3])

    def test_invalid_clock_rejected(self):
        with self.assertRaises(ValueError):
            EventTransmissionConfig(step_duration_seconds=30)
        with self.assertRaises(ValueError):
            EventTransmissionConfig(downlink_warmup_seconds=-1)


if __name__ == "__main__":
    unittest.main()
