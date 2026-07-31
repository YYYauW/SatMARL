from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
SCRIPTS = PROJECT_ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from marl_common import (
    ActorCritic,
    FastSlowOpportunityGraphActorCritic,
    OpportunityGraphActorCritic,
    QMixer,
    flatten_local_all,
    flatten_opportunity_graph_all,
    opportunity_graph_features,
    slow_strategy_features,
)
from sat_marl_env import EnvConfig, SatTaskingEnv


class MarlAlgorithmTest(unittest.TestCase):
    def test_actor_critic_and_local_observation_dimensions(self) -> None:
        env = SatTaskingEnv(
            EnvConfig(num_satellites=4, num_tasks=8, num_planes=2, max_steps=4)
        )
        observations, _ = env.reset(seed=2)
        _, local, global_state, masks = flatten_local_all(observations)
        expected_local = (
            env.self_dim
            + env.config.neighbor_k * env.neighbor_dim
            + env.neighbor_action_dim
            + env.config.candidate_k * env.task_dim
        )
        self.assertEqual(local.shape, (4, expected_local))
        model = ActorCritic(
            expected_local,
            expected_local + len(global_state),
            masks.shape[1],
            hidden_dim=32,
        )
        logits = model.policy_logits(torch.as_tensor(local))
        critic_input = np.concatenate(
            [local, np.repeat(global_state[None, :], 4, axis=0)], axis=1
        )
        values = model.values(torch.as_tensor(critic_input))
        self.assertEqual(tuple(logits.shape), (4, masks.shape[1]))
        self.assertEqual(tuple(values.shape), (4,))

    def test_qmix_is_monotonic_in_agent_q_values(self) -> None:
        torch.manual_seed(4)
        mixer = QMixer(num_agents=5, state_dim=7, embed_dim=16)
        states = torch.randn(3, 7)
        lower = torch.randn(3, 5)
        higher = lower + torch.rand(3, 5)
        with torch.no_grad():
            lower_total = mixer(lower, states)
            higher_total = mixer(higher, states)
        self.assertTrue(torch.all(higher_total >= lower_total - 1e-6))

    def test_opportunity_graph_actor_is_constellation_size_invariant(self) -> None:
        widths = []
        for satellites in (4, 8):
            env = SatTaskingEnv(
                EnvConfig(
                    num_satellites=satellites,
                    num_tasks=64,
                    num_planes=2,
                    max_steps=8,
                    candidate_k=6,
                    neighbor_k=2,
                    task_layout="mixed",
                    curriculum_visible_fraction=1.0,
                )
            )
            observations, _ = env.reset(seed=13)
            _, actor_obs, global_state, masks = flatten_opportunity_graph_all(
                observations
            )
            widths.append(actor_obs.shape[1])
            model = OpportunityGraphActorCritic(
                critic_dim=actor_obs.shape[1] + len(global_state),
                candidate_k=6,
                neighbor_k=2,
                hidden_dim=32,
            )
            logits = model.policy_logits(torch.as_tensor(actor_obs))
            self.assertEqual(tuple(logits.shape), (satellites, masks.shape[1]))
        self.assertEqual(widths[0], widths[1])

    def test_opportunity_graph_messages_detect_shared_task_factor(self) -> None:
        def row(task_id: int) -> dict[str, np.ndarray]:
            candidates = np.zeros((2, 24), dtype=np.float32)
            candidates[0, 0] = 0.8
            candidates[0, 14] = 0.5
            candidates[0, 19] = 0.9
            return {
                "self": np.zeros(24, dtype=np.float32),
                "neighbors": np.zeros((1, 13), dtype=np.float32),
                "neighbor_action_mean": np.zeros(4, dtype=np.float32),
                "candidates": candidates,
                "candidate_ids": np.asarray([task_id, -1], dtype=np.int32),
                "action_mask": np.asarray([1, 0, 1, 0], dtype=np.int8),
                "global": np.zeros(12, dtype=np.float32),
            }

        messages = opportunity_graph_features(
            {"satellite_0": row(7), "satellite_1": row(7)}
        ).reshape(2, 2, 4)
        self.assertGreater(float(messages[0, 0, 0]), 0.0)
        self.assertAlmostEqual(float(messages[0, 0, 1]), float(messages[1, 0, 1]))
        self.assertEqual(float(messages[0, 1].sum()), 0.0)

    def test_fast_slow_graph_has_fixed_width_and_valid_logits(self) -> None:
        widths = []
        for satellites in (4, 8):
            env = SatTaskingEnv(
                EnvConfig(
                    num_satellites=satellites,
                    num_tasks=64,
                    num_planes=2,
                    max_steps=8,
                    candidate_k=6,
                    neighbor_k=2,
                    task_layout="mixed",
                    curriculum_visible_fraction=1.0,
                )
            )
            observations, _ = env.reset(seed=29)
            _, graph_obs, global_state, masks = flatten_opportunity_graph_all(
                observations
            )
            slow_obs = slow_strategy_features(graph_obs, 6, 2)
            actor_obs = np.concatenate([graph_obs, slow_obs], axis=1)
            widths.append(actor_obs.shape[1])
            model = FastSlowOpportunityGraphActorCritic(
                critic_dim=actor_obs.shape[1] + len(global_state),
                candidate_k=6,
                neighbor_k=2,
                hidden_dim=32,
                intent_dim=16,
            )
            logits = model.policy_logits(torch.as_tensor(actor_obs))
            self.assertEqual(tuple(logits.shape), (satellites, masks.shape[1]))
        self.assertEqual(widths[0], widths[1])


if __name__ == "__main__":
    unittest.main()
