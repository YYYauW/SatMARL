from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from oasis_common import (
    OpportunityCurriculum,
    compute_opportunity_gae,
    decision_opportunities,
    opportunity_balance_weights,
    semantic_opportunity_balance_weights,
    semantic_opportunity_classes,
)


class OpportunityLearningTest(unittest.TestCase):
    def test_forced_idle_rows_are_not_decisions(self) -> None:
        masks = np.asarray([[[1, 0, 0], [1, 1, 0]], [[1, 0, 1], [1, 0, 0]]])
        active = decision_opportunities(masks)
        np.testing.assert_array_equal(active, [[False, True], [True, False]])

    def test_duration_corrected_return_aggregates_idle_interval(self) -> None:
        rewards = np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        values = np.zeros_like(rewards)
        dones = np.asarray([[0.0], [0.0], [0.0], [1.0]], dtype=np.float32)
        active = np.asarray([[True], [False], [True], [False]])
        advantages, returns, durations = compute_opportunity_gae(
            rewards, values, dones, active, gamma=0.5, gae_lambda=0.0
        )
        self.assertAlmostEqual(float(returns[0, 0]), 1.0 + 0.5 * 2.0)
        self.assertAlmostEqual(float(returns[2, 0]), 3.0 + 0.5 * 4.0)
        self.assertEqual(int(durations[0, 0]), 2)
        self.assertEqual(int(durations[2, 0]), 2)
        self.assertEqual(float(advantages[1, 0]), 0.0)

    def test_balancing_gives_equal_mass_to_occupied_strata(self) -> None:
        masks = np.zeros((1, 4, 6), dtype=bool)
        masks[:, :, 0] = True
        masks[0, 0, 1] = True
        masks[0, 1, 1] = True
        masks[0, 2, 1:3] = True
        masks[0, 3, 1:6] = True
        weights = opportunity_balance_weights(masks)
        self.assertAlmostEqual(float(weights[0, :2].sum()), float(weights[0, 2]))
        self.assertAlmostEqual(float(weights[0, 2]), float(weights[0, 3]))

    def test_semantic_classes_separate_downlink_and_task_choices(self) -> None:
        masks = np.zeros((1, 4, 5), dtype=bool)
        masks[:, :, 0] = True
        masks[0, 1, 1] = True
        masks[0, 2, 2] = True
        masks[0, 3, 1:4] = True
        classes = semantic_opportunity_classes(masks)
        np.testing.assert_array_equal(classes, [[0, 1, 2, 3]])

    def test_semantic_balancing_equalizes_occupied_class_mass(self) -> None:
        masks = np.zeros((1, 5, 5), dtype=bool)
        masks[:, :, 0] = True
        masks[0, :2, 1] = True
        masks[0, 2:4, 2] = True
        masks[0, 4, 2:4] = True
        weights = semantic_opportunity_balance_weights(masks)
        self.assertAlmostEqual(float(weights[0, :2].sum()), float(weights[0, 2:4].sum()))
        self.assertAlmostEqual(float(weights[0, 2:4].sum()), float(weights[0, 4]))

    def test_curriculum_moves_toward_target_and_real_distribution(self) -> None:
        controller = OpportunityCurriculum(start_fraction=0.8, end_fraction=0.0)
        first = controller.value(1, 10)
        after_low_rate = controller.observe(0.01, 1, 10)
        final = controller.value(10, 10)
        self.assertGreaterEqual(after_low_rate, 0.0)
        self.assertLessEqual(after_low_rate, first)
        self.assertEqual(final, 0.0)


if __name__ == "__main__":
    unittest.main()
