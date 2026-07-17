from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sat_marl_env import EnvConfig, SatTaskingEnv


def masked_random_actions(env: SatTaskingEnv, observations, rng: np.random.Generator):
    actions = {}
    for agent, obs in observations.items():
        valid = np.flatnonzero(obs["action_mask"])
        actions[agent] = int(rng.choice(valid)) if len(valid) else 0
    return actions


def greedy_priority_actions(env: SatTaskingEnv, observations):
    actions = {}
    for agent, obs in observations.items():
        mask = obs["action_mask"]
        task_rows = np.flatnonzero(mask[2:]) + 2
        if len(task_rows) == 0:
            actions[agent] = 1 if mask[1] else 0
            continue
        priorities = obs["candidates"][task_rows - 2, 0]
        actions[agent] = int(task_rows[int(np.argmax(priorities))])
    return actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--satellites", type=int, default=32)
    parser.add_argument("--tasks", type=int, default=128)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--policy", choices=["random", "greedy"], default="greedy")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    cfg = EnvConfig(
        num_satellites=args.satellites,
        num_tasks=args.tasks,
        max_steps=args.steps,
        random_seed=args.seed,
    )
    env = SatTaskingEnv(cfg)
    rng = np.random.default_rng(args.seed)
    observations, _ = env.reset(seed=args.seed)

    for _ in range(args.steps):
        if args.policy == "random":
            actions = masked_random_actions(env, observations, rng)
        else:
            actions = greedy_priority_actions(env, observations)

        observations, rewards, terminations, truncations, infos = env.step(actions)
        if all(terminations.values()) or all(truncations.values()):
            break

    summary = env.summary()
    print(env.render())
    print(f"policy={args.policy}")
    print(f"completed_tasks={summary['completed_tasks']}")
    print(f"expired_tasks={summary['expired_tasks']}")
    print(f"total_priority_completed={summary['total_priority_completed']:.2f}")
    print(f"total_conflicts={summary['total_conflicts']}")
    print(f"total_invalid_actions={summary['total_invalid_actions']}")


if __name__ == "__main__":
    main()

