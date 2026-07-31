from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_point_aaai_suite", ROOT / "scripts" / "run_point_aaai_suite.py"
)
assert SPEC is not None and SPEC.loader is not None
suite = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = suite
SPEC.loader.exec_module(suite)


class PointAAAISuiteTests(unittest.TestCase):
    def arguments(self, root: Path) -> argparse.Namespace:
        return argparse.Namespace(
            run_root=root,
            seeds=(701, 702, 703),
            scales=(64, 128, 256),
            sparsity_tasks=(768, 1536, 3072),
            tasks_per_satellite=48,
            max_steps=240,
            candidate_k=24,
            neighbor_k=6,
            step_seconds=30,
            lookahead_steps=30,
            episodes=300,
            checkpoint_every=10,
            resume=False,
            eval_seed=12001,
            eval_episodes=20,
            base_planes=8,
        )

    def test_core_and_ablation_matrix(self) -> None:
        specs = suite.method_specs(("core", "ablations"), (768, 1536, 3072))
        self.assertEqual(
            [item.name for item in specs[:6]],
            list(suite.CORE_METHODS),
        )
        self.assertEqual(
            [item.name for item in specs[6:]],
            list(suite.ABLATION_METHODS),
        )
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(Path(directory))
            train_jobs, eval_jobs = suite.build_jobs(args, specs)
        self.assertEqual(len(train_jobs), (6 + 4) * 3)
        self.assertEqual(len(eval_jobs), 6 * 3 * 3 + 4 * 3)
        ablation_scales = {
            job.scale for job in eval_jobs if job.method.phase == "ablations"
        }
        self.assertEqual(ablation_scales, {64})

    def test_gpu_zero_is_valid(self) -> None:
        self.assertEqual(suite.parse_gpu_csv("0,1"), (0, 1))

    def test_oasis_command_matches_submitted_point_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.arguments(root)
            spec = suite.MethodSpec(
                "oasis_graph", "oasis", "opportunity_graph"
            )
            job = suite.TrainJob(spec, 701, root / "run")
            command = suite.training_command(
                args, job, root / "n64.npz", root / "train.csv"
            )
        rendered = " ".join(command)
        self.assertIn("--architecture opportunity_graph", rendered)
        self.assertIn("--satellites 64", rendered)
        self.assertIn("--tasks 3072", rendered)
        self.assertIn("--point-observation-seconds 5", rendered)
        self.assertIn("--fov-deg 45", rendered)
        self.assertIn("--ephemeris-cache", rendered)
        self.assertIn("--task-catalog", rendered)
        self.assertNotIn("--area-task-fraction", rendered)
        self.assertNotIn("fast_slow_graph", rendered)

    def test_duration_ablation_disables_only_duration_correction(self) -> None:
        spec = next(
            item
            for item in suite.method_specs(("ablations",), (768, 1536, 3072))
            if item.name == "no_duration_correction"
        )
        self.assertEqual(spec.flags, ("--no-opportunity-gae",))

    def test_sparsity_reuses_core_3072_point(self) -> None:
        specs = suite.method_specs(("sparsity",), (768, 1536, 3072))
        self.assertEqual(
            {item.tasks for item in specs},
            {768, 1536},
        )
        self.assertEqual(len(specs), 4)


if __name__ == "__main__":
    unittest.main()
