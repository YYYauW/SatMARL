from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sat_marl_env import EnvConfig, SatTaskingEnv
from marl_common import resolve_device, scenario_seed


class QNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int,
        hidden_layers: int = 2,
        activation: str = "relu",
        layer_norm: bool = True,
    ):
        super().__init__()
        activation_class = {"relu": nn.ReLU, "silu": nn.SiLU, "tanh": nn.Tanh}[
            activation
        ]
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim)]
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(activation_class())
        for _ in range(max(1, hidden_layers) - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), activation_class()])
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self._items: deque[tuple[np.ndarray, int, float, np.ndarray, np.ndarray, float]] = (
            deque(maxlen=capacity)
        )

    def __len__(self) -> int:
        return len(self._items)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        next_mask: np.ndarray,
        done: bool,
    ) -> None:
        self._items.append(
            (
                obs.astype(np.float32, copy=False),
                int(action),
                float(reward),
                next_obs.astype(np.float32, copy=False),
                next_mask.astype(np.float32, copy=False),
                float(done),
            )
        )

    def sample(self, batch_size: int):
        batch = random.sample(self._items, batch_size)
        obs, actions, rewards, next_obs, next_masks, dones = zip(*batch)
        return (
            torch.as_tensor(np.asarray(obs), dtype=torch.float32),
            torch.as_tensor(actions, dtype=torch.long),
            torch.as_tensor(rewards, dtype=torch.float32),
            torch.as_tensor(np.asarray(next_obs), dtype=torch.float32),
            torch.as_tensor(np.asarray(next_masks), dtype=torch.bool),
            torch.as_tensor(dones, dtype=torch.float32),
        )


def flatten_agent_obs(
    agent_obs: dict[str, np.ndarray], include_global: bool = True
) -> np.ndarray:
    parts = [agent_obs["self"].ravel()]
    if include_global:
        parts.append(agent_obs["global"].ravel())
    parts.extend(
        [
            agent_obs["neighbors"].ravel(),
            agent_obs["neighbor_action_mean"].ravel(),
            agent_obs["candidates"].ravel(),
        ]
    )
    return np.concatenate(parts).astype(np.float32, copy=False)


def flatten_all(
    observations: dict[str, dict[str, np.ndarray]], include_global: bool = True
):
    agents = list(observations.keys())
    vecs = np.stack(
        [flatten_agent_obs(observations[agent], include_global) for agent in agents]
    )
    masks = np.stack([observations[agent]["action_mask"] for agent in agents]).astype(bool)
    return agents, vecs, masks


def choose_actions(
    net: QNetwork,
    observations: dict[str, dict[str, np.ndarray]],
    epsilon: float,
    device: torch.device,
    rng: np.random.Generator,
    include_global: bool = True,
) -> tuple[dict[str, int], list[str], np.ndarray, np.ndarray, np.ndarray]:
    agents, vecs, masks = flatten_all(observations, include_global)
    with torch.no_grad():
        q_values = net(torch.as_tensor(vecs, dtype=torch.float32, device=device))
        q_values = q_values.cpu().numpy()
    q_values[~masks] = -1e9
    greedy = np.argmax(q_values, axis=1).astype(np.int64)
    actions = {}
    for row, agent in enumerate(agents):
        if rng.random() < epsilon:
            valid = np.flatnonzero(masks[row])
            actions[agent] = int(rng.choice(valid)) if len(valid) else 0
        else:
            actions[agent] = int(greedy[row])
    return actions, agents, vecs, masks, greedy


def epsilon_by_step(args: argparse.Namespace, total_env_steps: int) -> float:
    if total_env_steps >= args.epsilon_decay_steps:
        return args.epsilon_end
    frac = total_env_steps / max(1, args.epsilon_decay_steps)
    return args.epsilon_start + frac * (args.epsilon_end - args.epsilon_start)


def optimize(
    net: QNetwork,
    target_net: QNetwork,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    args: argparse.Namespace,
    device: torch.device,
) -> float | None:
    if len(replay) < max(args.batch_size, args.learning_starts):
        return None

    obs, actions, rewards, next_obs, next_masks, dones = replay.sample(args.batch_size)
    obs = obs.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_obs = next_obs.to(device)
    next_masks = next_masks.to(device)
    dones = dones.to(device)

    q = net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        target_next_q = target_net(next_obs).masked_fill(~next_masks, -1e9)
        if args.double_dqn:
            online_next_q = net(next_obs).masked_fill(~next_masks, -1e9)
            next_actions = online_next_q.argmax(dim=1, keepdim=True)
            next_values = target_next_q.gather(1, next_actions).squeeze(1)
        else:
            next_values = target_next_q.max(dim=1).values
        target = rewards + args.gamma * (1.0 - dones) * next_values
    if args.loss_type == "huber":
        loss = F.smooth_l1_loss(q, target, beta=args.huber_delta)
    else:
        loss = F.mse_loss(q, target)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
    optimizer.step()
    return float(loss.detach().cpu().item())


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    last_error: Exception | None = None
    for attempt in range(20):
        try:
            tmp.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.05 * (attempt + 1))
    raise last_error if last_error is not None else RuntimeError("atomic write failed")


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    result = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def save_checkpoint(
    path: Path,
    net: QNetwork,
    target_net: QNetwork,
    optimizer: torch.optim.Optimizer,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
    torch.save(
        {
            "policy": net.state_dict(),
            "target": target_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
        },
        tmp,
    )
    tmp.replace(path)


def load_checkpoint(
    path: Path,
    net: QNetwork,
    target_net: QNetwork,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    net.load_state_dict(checkpoint["policy"])
    target_net.load_state_dict(checkpoint["target"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return dict(checkpoint.get("metrics") or {})


def make_config(args: argparse.Namespace) -> EnvConfig:
    return EnvConfig(
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
        random_seed=args.seed,
        task_layout=args.task_layout,
        curriculum_visible_fraction=args.curriculum_visible_fraction,
        curriculum_ground_track_jitter_deg=args.curriculum_ground_track_jitter_deg,
        curriculum_time_jitter_steps=args.curriculum_time_jitter_steps,
        curriculum_payload_match_probability=args.curriculum_payload_match_probability,
        min_observation_elevation_deg=args.min_observation_elevation_deg,
        max_off_nadir_deg=args.max_off_nadir_deg,
        optical_min_sun_elevation_deg=args.optical_min_sun_elevation_deg,
        min_task_window=args.min_task_window,
        max_task_window=args.max_task_window,
        planning_lookahead_steps=args.planning_lookahead_steps,
    )


def algorithm_metadata(
    args: argparse.Namespace, env: SatTaskingEnv, net: QNetwork
) -> dict[str, Any]:
    parameter_count = sum(parameter.numel() for parameter in net.parameters())
    return {
        "name": "parameter_shared_double_dqn" if args.double_dqn else "parameter_shared_dqn",
        "family": "value_based_marl",
        "training_mode": "shared_replay_centralized_training",
        "execution_mode": (
            "decentralized_local_observation"
            if args.local_observation_only
            else "decentralized_with_broadcast_global_summary"
        ),
        "parameter_sharing": True,
        "agent_count": env.config.num_satellites,
        "action_masking": True,
        "coordination": "environment_auction_and_hard_feasibility",
        "credit_assignment": "individual_reward_plus_team_bonus",
        "replay_sampling": "uniform_over_decision_opportunities",
        "network": {
            "input_dim": env.self_dim
            + (0 if args.local_observation_only else env.global_dim)
            + env.config.neighbor_k * env.neighbor_dim
            + env.neighbor_action_dim
            + env.config.candidate_k * env.task_dim,
            "output_dim": env.config.candidate_k + 2,
            "hidden_dim": args.hidden_dim,
            "hidden_layers": args.hidden_layers,
            "activation": args.activation,
            "layer_norm": args.layer_norm,
            "parameter_count": parameter_count,
        },
        "optimizer": {
            "name": args.optimizer,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.grad_clip,
        },
        "temporal_difference": {
            "double_dqn": args.double_dqn,
            "gamma": args.gamma,
            "loss": args.loss_type,
            "huber_delta": args.huber_delta,
            "target_update_steps": args.target_update_steps,
            "target_tau": args.target_tau,
        },
        "exploration": {
            "type": "linear_epsilon_greedy",
            "epsilon_start": args.epsilon_start,
            "epsilon_end": args.epsilon_end,
            "epsilon_decay_steps": args.epsilon_decay_steps,
        },
        "sampling": {
            "buffer_size": args.buffer_size,
            "batch_size": args.batch_size,
            "learning_starts": args.learning_starts,
            "store_agents_per_step": args.store_agents,
            "train_frequency": args.train_frequency,
            "gradient_steps": args.gradient_steps,
            "reward_scale": args.reward_scale,
            "reward_clip": args.reward_clip,
        },
    }


def update_target_network(
    target_net: QNetwork, net: QNetwork, tau: float
) -> None:
    if tau >= 1.0:
        target_net.load_state_dict(net.state_dict())
        return
    with torch.no_grad():
        for target_parameter, parameter in zip(
            target_net.parameters(), net.parameters(), strict=True
        ):
            target_parameter.mul_(1.0 - tau).add_(parameter, alpha=tau)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--satellites", type=int, default=128)
    parser.add_argument("--tasks", type=int, default=512)
    parser.add_argument("--planes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=96)
    parser.add_argument("--candidate-k", type=int, default=16)
    parser.add_argument("--neighbor-k", type=int, default=6)
    parser.add_argument("--step-duration-seconds", type=float, default=30.0)
    parser.add_argument("--point-observation-seconds", type=float, default=5.0)
    parser.add_argument("--fov-deg", type=float, default=45.0)
    parser.add_argument(
        "--task-layout",
        choices=["global_random", "curriculum_visible", "mixed", "mixed_curriculum"],
        default="global_random",
    )
    parser.add_argument("--curriculum-visible-fraction", type=float, default=0.0)
    parser.add_argument("--curriculum-ground-track-jitter-deg", type=float, default=5.0)
    parser.add_argument("--curriculum-time-jitter-steps", type=int, default=3)
    parser.add_argument("--curriculum-payload-match-probability", type=float, default=0.8)
    parser.add_argument("--min-observation-elevation-deg", type=float, default=3.0)
    parser.add_argument("--max-off-nadir-deg", type=float, default=40.0)
    parser.add_argument("--optical-min-sun-elevation-deg", type=float, default=8.0)
    parser.add_argument("--min-task-window", type=int, default=16)
    parser.add_argument("--max-task-window", type=int, default=48)
    parser.add_argument("--planning-lookahead-steps", type=int, default=12)
    parser.add_argument("--scenario-seed-cycle", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--activation", choices=["relu", "silu", "tanh"], default="relu")
    parser.add_argument("--layer-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--local-observation-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--buffer-size", type=int, default=60000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--optimizer", choices=["adamw", "adam"], default="adamw")
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--loss-type", choices=["huber", "mse"], default="huber")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--double-dqn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--reward-clip", type=float, default=0.0)
    parser.add_argument("--epsilon-start", type=float, default=0.8)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=50000)
    parser.add_argument("--target-update-steps", type=int, default=800)
    parser.add_argument("--target-tau", type=float, default=1.0)
    parser.add_argument("--store-agents", type=int, default=96)
    parser.add_argument("--train-frequency", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--metrics-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "runs" / "realistic")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--stop-file", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)

    env = SatTaskingEnv(make_config(args))
    observations, _ = env.reset(seed=args.seed)
    include_global = not args.local_observation_only
    _, sample_vecs, sample_masks = flatten_all(observations, include_global)
    input_dim = sample_vecs.shape[1]
    action_dim = sample_masks.shape[1]

    net = QNetwork(
        input_dim,
        action_dim,
        args.hidden_dim,
        args.hidden_layers,
        args.activation,
        args.layer_norm,
    ).to(device)
    target_net = QNetwork(
        input_dim,
        action_dim,
        args.hidden_dim,
        args.hidden_layers,
        args.activation,
        args.layer_norm,
    ).to(device)
    target_net.load_state_dict(net.state_dict())
    optimizer_class = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_class(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    replay = ReplayBuffer(args.buffer_size)

    metrics_path = args.run_dir / "metrics.json"
    checkpoint_path = args.run_dir / "checkpoints" / "latest.pt"
    resume_path = args.resume_from or checkpoint_path
    stop_file = args.stop_file or (args.run_dir / "STOP")
    history: list[dict[str, Any]] = []
    loss_window: deque[float] = deque(maxlen=200)
    total_env_steps = 0
    total_agent_steps = 0
    started_at = time.time()
    start_episode = 1
    resumed_from: str | None = None

    metrics: dict[str, Any] = {
        "status": "running",
        "started_at": started_at,
        "updated_at": started_at,
        "device": str(device),
        "project_root": str(PROJECT_ROOT),
        "args": serializable_args(args),
        "algorithm": algorithm_metadata(args, env, net),
        "env_config": asdict(env.config),
        "input_dim": input_dim,
        "action_dim": action_dim,
        "episode": 0,
        "total_env_steps": 0,
        "total_agent_steps": 0,
        "history": history,
        "last": {},
    }

    if args.resume and resume_path.exists():
        previous = load_checkpoint(resume_path, net, target_net, optimizer, device)
        previous_input_dim = previous.get("input_dim")
        previous_action_dim = previous.get("action_dim")
        if previous_input_dim and previous_input_dim != input_dim:
            raise ValueError(
                f"Checkpoint input_dim={previous_input_dim} does not match {input_dim}."
            )
        if previous_action_dim and previous_action_dim != action_dim:
            raise ValueError(
                f"Checkpoint action_dim={previous_action_dim} does not match {action_dim}."
            )
        total_env_steps = int(previous.get("total_env_steps", 0))
        total_agent_steps = int(previous.get("total_agent_steps", 0))
        previous_episode = int(previous.get("episode", 0))
        start_episode = previous_episode + 1
        history = list(previous.get("history") or [])[-600:]
        for item in history[-200:]:
            value = item.get("loss")
            if value is not None:
                loss_window.append(float(value))
        resumed_from = str(resume_path)
        metrics.update(previous)
        metrics.update(
            {
                "status": "running",
                "updated_at": time.time(),
                "device": str(device),
                "project_root": str(PROJECT_ROOT),
                "args": serializable_args(args),
                "algorithm": algorithm_metadata(args, env, net),
                "env_config": asdict(env.config),
                "input_dim": input_dim,
                "action_dim": action_dim,
                "episode": previous_episode,
                "total_env_steps": total_env_steps,
                "total_agent_steps": total_agent_steps,
                "history": history,
                "resumed_from": resumed_from,
                "resumed_at": time.time(),
                "replay_size": 0,
            }
        )
    elif args.resume:
        metrics["resume_warning"] = f"No checkpoint found at {resume_path}; started fresh."

    atomic_write_json(metrics_path, metrics)

    try:
        for episode in range(start_episode, args.episodes + 1):
            if stop_file.exists():
                metrics.update(
                    {
                        "status": "stopped",
                        "updated_at": time.time(),
                        "episode": episode - 1,
                        "total_env_steps": total_env_steps,
                        "total_agent_steps": total_agent_steps,
                        "history": history,
                        "stop_file": str(stop_file),
                    }
                )
                atomic_write_json(metrics_path, metrics)
                save_checkpoint(checkpoint_path, net, target_net, optimizer, metrics)
                break

            observations, _ = env.reset(
                seed=scenario_seed(args.seed, episode, args.scenario_seed_cycle)
            )
            episode_reward = 0.0
            episode_steps = 0
            last_loss = None
            episode_started_at = time.time()

            for _ in range(args.max_steps):
                epsilon = epsilon_by_step(args, total_env_steps)
                actions, agents, vecs, masks, _ = choose_actions(
                    net, observations, epsilon, device, rng, include_global
                )
                next_observations, rewards, terminations, truncations, infos = env.step(
                    actions
                )
                next_agents, next_vecs, next_masks = flatten_all(
                    next_observations, include_global
                )
                next_index = {agent: idx for idx, agent in enumerate(next_agents)}
                done = all(terminations.values()) or all(truncations.values())

                decision_rows = np.flatnonzero(np.count_nonzero(masks, axis=1) > 1)
                store_count = min(args.store_agents, len(decision_rows))
                if store_count > 0:
                    chosen_rows = rng.choice(
                        decision_rows, size=store_count, replace=False
                    )
                    for row in chosen_rows:
                        agent = agents[int(row)]
                        next_row = next_index[agent]
                        stored_reward = rewards[agent] * args.reward_scale
                        if args.reward_clip > 0.0:
                            stored_reward = float(
                                np.clip(stored_reward, -args.reward_clip, args.reward_clip)
                            )
                        replay.add(
                            vecs[row],
                            actions[agent],
                            stored_reward,
                            next_vecs[next_row],
                            next_masks[next_row],
                            done,
                        )

                if total_env_steps % max(1, args.train_frequency) == 0:
                    for _ in range(args.gradient_steps):
                        loss = optimize(net, target_net, optimizer, replay, args, device)
                        if loss is not None:
                            last_loss = loss
                            loss_window.append(loss)

                total_env_steps += 1
                total_agent_steps += len(agents)
                episode_steps += 1
                episode_reward += float(np.mean(list(rewards.values())))
                observations = next_observations

                if total_env_steps % args.target_update_steps == 0:
                    update_target_network(target_net, net, args.target_tau)
                if done:
                    break

            summary = env.summary()
            mean_loss = float(np.mean(loss_window)) if loss_window else None
            episode_seconds = max(1e-6, time.time() - episode_started_at)
            episode_record = {
                "episode": episode,
                "env_steps": episode_steps,
                "mean_reward": episode_reward / max(1, episode_steps),
                "reward_per_opportunity": episode_reward
                * len(env.agents)
                / max(1, summary["decision_opportunities"]),
                "decision_opportunities": summary["decision_opportunities"],
                "decision_opportunity_rate": summary["decision_opportunity_rate"],
                "forced_idle_actions": summary["forced_idle_actions"],
                "avoidable_idle_actions": summary["avoidable_idle_actions"],
                "completed_tasks": summary["completed_tasks"],
                "expired_tasks": summary["expired_tasks"],
                "active_tasks": summary["active_tasks"],
                "available_tasks": summary["available_tasks"],
                "total_priority_completed": summary["total_priority_completed"],
                "busy_satellites": summary["busy_satellites"],
                "conflicts": summary["total_conflicts"],
                "invalid_actions": summary["total_invalid_actions"],
                "window_misses": summary["total_window_misses"],
                "mean_energy": summary["mean_energy"],
                "mean_storage": summary["mean_storage"],
                "mean_observation_quality": summary["mean_observation_quality"],
                "cooperative_completed_tasks": summary["cooperative_completed_tasks"],
                "ground_conflicts": summary["total_ground_conflicts"],
                "pending_data_mb": summary["pending_data_mb"],
                "downlinked_data_mb": summary["downlinked_data_mb"],
                "ground_station_utilization": summary["ground_station_utilization"],
                "epsilon": epsilon_by_step(args, total_env_steps),
                "loss": last_loss,
                "mean_loss": mean_loss,
                "total_env_steps": total_env_steps,
                "total_agent_steps": total_agent_steps,
                "replay_size": len(replay),
                "episode_seconds": episode_seconds,
                "env_steps_per_second": episode_steps / episode_seconds,
                "agent_steps_per_second": (
                    episode_steps * len(env.agents) / episode_seconds
                ),
                "timestamp": time.time(),
            }
            history.append(episode_record)

            if episode % args.metrics_every == 0:
                metrics.update(
                    {
                        "status": "running",
                        "updated_at": time.time(),
                        "episode": episode,
                        "total_env_steps": total_env_steps,
                        "total_agent_steps": total_agent_steps,
                        "epsilon": episode_record["epsilon"],
                        "replay_size": len(replay),
                        "resumed_from": resumed_from,
                        "last": episode_record,
                        "history": history,
                    }
                )
                atomic_write_json(metrics_path, metrics)

            if episode % args.checkpoint_every == 0:
                save_checkpoint(checkpoint_path, net, target_net, optimizer, metrics)

            if stop_file.exists():
                metrics.update(
                    {
                        "status": "stopped",
                        "updated_at": time.time(),
                        "episode": episode,
                        "total_env_steps": total_env_steps,
                        "total_agent_steps": total_agent_steps,
                        "history": history,
                        "stop_file": str(stop_file),
                    }
                )
                atomic_write_json(metrics_path, metrics)
                save_checkpoint(checkpoint_path, net, target_net, optimizer, metrics)
                break

        else:
            metrics.update(
                {
                    "status": "complete",
                    "updated_at": time.time(),
                    "episode": args.episodes,
                    "total_env_steps": total_env_steps,
                    "total_agent_steps": total_agent_steps,
                    "history": history,
                }
            )
            atomic_write_json(metrics_path, metrics)
            save_checkpoint(checkpoint_path, net, target_net, optimizer, metrics)
    except KeyboardInterrupt:
        metrics["status"] = "interrupted"
        metrics["updated_at"] = time.time()
        atomic_write_json(metrics_path, metrics)
        save_checkpoint(checkpoint_path, net, target_net, optimizer, metrics)
    except Exception as exc:
        metrics["status"] = "failed"
        metrics["updated_at"] = time.time()
        metrics["error"] = repr(exc)
        atomic_write_json(metrics_path, metrics)
        raise


if __name__ == "__main__":
    main()
