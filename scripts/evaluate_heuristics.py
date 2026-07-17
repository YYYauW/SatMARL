from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from marl_common import atomic_write_json

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sat_marl_env import EnvConfig, SatTaskingEnv


POLICIES = ("random_feasible", "greedy_priority", "earliest_deadline", "rolling_assignment")


def task_score(env: SatTaskingEnv, task_id: int, policy: str) -> float:
    task = env.tasks[task_id]
    if policy == "earliest_deadline":
        slack = max(1, task.deadline_step - env.step_count)
        return 1000.0 / slack + task.priority
    return task.priority


def independent_actions(
    env: SatTaskingEnv,
    observations: dict,
    policy: str,
    rng: np.random.Generator,
) -> dict[str, int]:
    actions: dict[str, int] = {}
    for agent, observation in observations.items():
        valid = np.flatnonzero(observation["action_mask"])
        task_actions = valid[valid >= 2]
        if policy == "random_feasible":
            actions[agent] = int(rng.choice(valid)) if len(valid) else 0
            continue
        if len(task_actions):
            scored = []
            for action in task_actions:
                task_id = int(observation["candidate_ids"][int(action) - 2])
                scored.append((task_score(env, task_id, policy), int(action)))
            actions[agent] = max(scored)[1]
        elif 1 in valid:
            actions[agent] = 1
        else:
            actions[agent] = 0
    return actions


def rolling_assignment_actions(env: SatTaskingEnv, observations: dict) -> dict[str, int]:
    agents = list(observations)
    task_ids = sorted(
        {
            int(task_id)
            for observation in observations.values()
            for task_id in observation["candidate_ids"]
            if int(task_id) >= 0
        }
    )
    actions = {agent: 0 for agent in agents}
    if not task_ids:
        for agent in agents:
            if bool(observations[agent]["action_mask"][1]):
                actions[agent] = 1
        return actions

    task_col = {task_id: col for col, task_id in enumerate(task_ids)}
    score = np.full((len(agents), len(task_ids) + len(agents)), -1e6, dtype=np.float64)
    score[:, len(task_ids) :] = 0.0
    action_lookup: dict[tuple[int, int], int] = {}
    for row, agent in enumerate(agents):
        observation = observations[agent]
        for action in np.flatnonzero(observation["action_mask"]):
            if action < 2:
                continue
            task_id = int(observation["candidate_ids"][int(action) - 2])
            task = env.tasks[task_id]
            feasible, _, plan = env._plan_task(env.satellites[row], task)
            if not feasible:
                continue
            quality = float(plan.get("quality", 1.0))
            wait = float(plan.get("observation_start_step", env.step_count)) - env.step_count
            value = task.priority * quality / (1.0 + max(0.0, wait))
            score[row, task_col[task_id]] = value
            action_lookup[(row, task_col[task_id])] = int(action)
    rows, cols = linear_sum_assignment(-score)
    for row, col in zip(rows, cols, strict=True):
        action = action_lookup.get((int(row), int(col)))
        if action is not None and score[row, col] > 0.0:
            actions[agents[int(row)]] = action
        elif bool(observations[agents[int(row)]]["action_mask"][1]):
            actions[agents[int(row)]] = 1
    return actions


def run_episode(cfg: EnvConfig, policy: str, seed: int) -> dict[str, float]:
    env = SatTaskingEnv(cfg)
    observations, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    reward_total = 0.0
    steps = 0
    started = time.perf_counter()
    for _ in range(cfg.max_steps):
        actions = (
            rolling_assignment_actions(env, observations)
            if policy == "rolling_assignment"
            else independent_actions(env, observations, policy, rng)
        )
        observations, rewards, terminations, truncations, _ = env.step(actions)
        reward_total += float(np.mean(list(rewards.values())))
        steps += 1
        if all(terminations.values()) or all(truncations.values()):
            break
    summary = env.summary()
    return {
        "mean_episode_reward": reward_total / max(1, steps),
        "completed_tasks": float(summary["completed_tasks"]),
        "total_priority_completed": float(summary["total_priority_completed"]),
        "mean_observation_quality": float(summary["mean_observation_quality"]),
        "decision_opportunity_rate": float(summary["decision_opportunity_rate"]),
        "conflicts": float(summary["total_conflicts"]),
        "invalid_actions": float(summary["total_invalid_actions"]),
        "runtime_seconds": time.perf_counter() - started,
    }


def aggregate(records: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        key: {
            "mean": float(np.mean([record[key] for record in records])),
            "std": float(np.std([record[key] for record in records])),
        }
        for key in records[0]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--satellites", type=int, default=64)
    parser.add_argument("--tasks", type=int, default=256)
    parser.add_argument("--planes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=96)
    parser.add_argument("--candidate-k", type=int, default=24)
    parser.add_argument("--neighbor-k", type=int, default=8)
    parser.add_argument("--step-duration-seconds", type=float, default=30.0)
    parser.add_argument("--point-observation-seconds", type=float, default=5.0)
    parser.add_argument("--fov-deg", type=float, default=45.0)
    parser.add_argument("--max-off-nadir-deg", type=float, default=45.0)
    parser.add_argument("--min-task-window", type=int, default=40)
    parser.add_argument("--max-task-window", type=int, default=160)
    parser.add_argument("--planning-lookahead-steps", type=int, default=30)
    parser.add_argument(
        "--task-layout",
        choices=["global_random", "curriculum_visible", "mixed", "mixed_curriculum"],
        default="global_random",
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=24001)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "baselines" / "summary.json")
    args = parser.parse_args()
    cfg = EnvConfig(
        num_satellites=args.satellites,
        num_tasks=args.tasks,
        num_planes=args.planes,
        max_steps=args.max_steps,
        candidate_k=args.candidate_k,
        neighbor_k=args.neighbor_k,
        step_duration_seconds=args.step_duration_seconds,
        point_observation_seconds=args.point_observation_seconds,
        payload_fov_min_deg=args.fov_deg,
        payload_fov_max_deg=args.fov_deg,
        max_off_nadir_deg=args.max_off_nadir_deg,
        min_task_window=args.min_task_window,
        max_task_window=args.max_task_window,
        planning_lookahead_steps=args.planning_lookahead_steps,
        task_layout=args.task_layout,
        random_seed=args.seed,
    )
    payload = {
        "status": "running",
        "started_at": time.time(),
        "config": asdict(cfg),
        "eval_episodes": args.episodes,
        "policies": {},
    }
    atomic_write_json(args.output, payload)
    for policy in POLICIES:
        records = [run_episode(cfg, policy, args.seed + episode) for episode in range(args.episodes)]
        payload["policies"][policy] = {"aggregate": aggregate(records), "episodes": records}
        payload["updated_at"] = time.time()
        atomic_write_json(args.output, payload)
    payload["status"] = "complete"
    payload["updated_at"] = time.time()
    atomic_write_json(args.output, payload)


if __name__ == "__main__":
    main()
