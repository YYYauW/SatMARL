from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
spec = importlib.util.spec_from_file_location("transmission_runner_test", ROOT/"scripts/run_transmission_load.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTest(unittest.TestCase):
    def config(self):
        return json.loads((ROOT/"configs/transmission_load_v2.json").read_text(encoding="utf-8"))

    def test_main_grid_is_full_cross_product(self):
        config = self.config()
        jobs = runner.make_plan(config, "checkpoint", ROOT/"models")
        self.assertEqual(len(jobs), 3*2*4*4)
        for n in config["satellite_counts"]:
            counts = {j["tasks"] for j in jobs if j["satellites"] == n}
            self.assertEqual(counts, set(config["task_counts"]))

    def test_per_target_memory_and_nested_demand(self):
        config = self.config()
        config["task_counts"] = [8, 16]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"catalog.csv"
            runner.prepare_catalog(config, path, 0)
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 16)
            for row in rows:
                self.assertAlmostEqual(float(row["memory_required_mb"])*4,
                                       float(row["original_data_mb"]))
                self.assertEqual(row["required_mode"], "optical")
            original_hash = runner.sha(path)
            runner.prepare_catalog(config, path, 0)
            self.assertEqual(runner.sha(path), original_hash)

    def test_invalid_lists(self):
        for value in ("0", "8,8", "-1,4"):
            with self.assertRaises(ValueError):
                runner.integer_list(value, [])

    def test_day_catalog_covers_the_day_with_second_units(self):
        config = json.loads((ROOT/"configs/transmission_day_v3.json").read_text(encoding="utf-8"))
        config["task_counts"] = [128]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"targets.csv"
            runner.prepare_catalog(config, path, 0)
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        releases = [int(row["release_step"]) for row in rows]
        self.assertLess(min(releases), 3600)
        self.assertGreater(max(releases), 23*3600)
        for row in rows:
            duration = int(row["deadline_step"])-int(row["release_step"])
            self.assertGreaterEqual(duration, 240)
            self.assertLessEqual(duration, 960)
            self.assertEqual(row["observation_duration_seconds"], "5.0")
            self.assertEqual(row["max_coordination_gap_steps"], "3600")

    def test_merge_treats_training_seed_as_unit(self):
        rows = []
        for seed, value in ((701, 10), (702, 20)):
            rows.append(dict(method="full", satellites=512, tasks=6144,
                             training_seed=seed,
                             **{metric+"_mean": value for metric in runner.METRICS}))
        merged = runner.merge_training_seeds(rows)
        self.assertEqual(merged[0]["training_seed_count"], 2)
        self.assertEqual(merged[0]["observation_success_tasks_mean"], 15)


if __name__ == "__main__":
    unittest.main()
