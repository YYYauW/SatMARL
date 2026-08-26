from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


RUNNER = load_module(
    "run_thousand_paper_suite_for_test",
    ROOT / "scripts" / "run_thousand_paper_suite.py",
)
SUMMARY = load_module(
    "summarize_thousand_paper_suite_for_test",
    ROOT / "scripts" / "summarize_thousand_paper_suite.py",
)


class ThousandPaperSuiteTest(unittest.TestCase):
    def suite(self, temporary: str):
        root = Path(temporary)
        arguments = Namespace(
            config=ROOT / "configs" / "thousand_paper_suite.json",
            run_root=root / "runs",
            scenario_root=root / "scenario",
            data_root=root / "catalogs",
            profile="smoke",
            stage="all",
            methods=(
                "full,mlp_opportunity,ippo,mappo,qmix,ps_dqn,no_reservation,"
                "no_factor_messages,no_learned_bids"
            ),
            seeds="701",
            gpus="0,1",
            resume=True,
            dry_run=True,
            fail_fast=True,
            poll_seconds=0.01,
        )
        return RUNNER.Suite(arguments)

    def test_commands_cover_rl_baselines_and_mechanism_ablations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            suite = self.suite(temporary)
            commands = {
                job.job_id: job.command for job in suite.train_jobs()
            }
            self.assertIn("scripts/train_ppo.py", commands["train:ippo:seed701"])
            self.assertIn("scripts/train_ppo.py", commands["train:mappo:seed701"])
            self.assertIn("scripts/train_qmix.py", commands["train:qmix:seed701"])
            self.assertIn("scripts/train_dqn.py", commands["train:ps_dqn:seed701"])
            self.assertIn("mlp", commands["train:mlp_opportunity:seed701"])
            self.assertIn(
                "--no-coalition-reservations",
                commands["train:no_reservation:seed701"],
            )
            for command in commands.values():
                self.assertIn("--cooperative-observers-min", command)
                self.assertIn("--cooperative-observers-max", command)
            no_balance = suite.train_command("no_opportunity_balance", 701)
            self.assertIn("--no-opportunity-balancing", no_balance)
            self.assertIn("--no-semantic-opportunity-balancing", no_balance)
            self.assertNotIn("--semantic-opportunity-balancing", no_balance)
            no_factors = commands["train:no_factor_messages:seed701"]
            self.assertIn("--no-graph-factor-messages", no_factors)
            self.assertNotIn("--no-learned-resource-bids", no_factors)
            no_bids = commands["train:no_learned_bids:seed701"]
            self.assertIn("--no-learned-resource-bids", no_bids)
            self.assertNotIn("--no-graph-factor-messages", no_bids)

    def test_multi_job_gpu_slots_are_unique_and_keep_physical_gpu_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            suite = self.suite(temporary)
            suite.config["resources"]["jobs_per_gpu"] = 2
            slots = suite.worker_slots()
            self.assertEqual(
                slots,
                [("0:0", "0"), ("0:1", "0"), ("1:0", "1"), ("1:1", "1")],
            )
            self.assertEqual(len({slot_id for slot_id, _ in slots}), 4)

    def test_smoke_plan_has_two_scales_and_declared_stress_axes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            suite = self.suite(temporary)
            evaluations = suite.evaluation_jobs()
            self.assertEqual(len(evaluations), 18)
            self.assertEqual(
                {int(job.markers[0].stem.removeprefix("n")) for job in evaluations},
                {8, 16},
            )
            specifications = suite.stress_specs()
            self.assertEqual(
                {item["group"] for item in specifications},
                {"geography", "cooperation", "task_density"},
            )
            self.assertEqual(len(specifications), 10)

    def test_summary_uses_training_seed_as_statistical_unit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            for method, offset in (("full", 3.0), ("mappo", 0.0)):
                for seed in (701, 702, 703):
                    path = (
                        run_root
                        / "training"
                        / method
                        / f"seed_{seed}"
                        / "evaluation"
                        / "base"
                        / "n512.json"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    aggregate = {
                        metric: {"mean": offset + seed / 1000.0, "std": 0.1}
                        for metric in SUMMARY.METRICS
                    }
                    path.write_text(
                        json.dumps(
                            {
                                "status": "complete",
                                "env_config": {"num_tasks": 24576},
                                "benchmark": {"episodes": 20, "aggregate": aggregate},
                                "algorithm": {"network": {"parameter_count": 1234}},
                            }
                        ),
                        encoding="utf-8",
                    )
            rows = SUMMARY.collect_rows(run_root)
            self.assertEqual(len(rows), 6)
            grouped = SUMMARY.aggregate_across_training_seeds(rows)
            self.assertTrue(all(row["training_seeds"] == 3 for row in grouped))
            comparisons = SUMMARY.paired_comparisons(rows)
            completed = next(
                row
                for row in comparisons
                if row["metric"] == "completed_tasks"
            )
            self.assertEqual(completed["paired_training_seeds"], 3)
            self.assertAlmostEqual(completed["mean_difference"], 3.0)
            self.assertAlmostEqual(completed["ci95_low"], 3.0)
            self.assertAlmostEqual(completed["ci95_high"], 3.0)


if __name__ == "__main__":
    unittest.main()
