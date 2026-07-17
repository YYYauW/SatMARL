from __future__ import annotations

import argparse
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

from marl_common import (
    AgentQNetwork,
    QMixer,
    atomic_torch_save,
    atomic_write_json,
    flatten_local_all,
    parameter_count,
    resolve_device,
    scenario_seed,
    serialize_args,
)
from sat_marl_env import EnvConfig, SatTaskingEnv


Transition = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]


class JointReplayBuffer:
    def __init__(self, capacity: int):
        self.items: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.items)

    def add(
        self,
        observations: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
        reward: float,
        next_observations: np.ndarray,
        next_state: np.ndarray,
        next_masks: np.ndarray,
        done: bool,
    ) -> None:
        self.items.append(
            (
                observations.astype(np.float16),
                state.astype(np.float32),
                actions.astype(np.int16),
                float(reward),
                next_observations.astype(np.float16),
                next_state.astype(np.float32),
                next_masks.astype(bool),
                float(done),
            )
        )

    def sample(self, batch_size: int):
        batch = random.sample(self.items, batch_size)
        fields = list(zip(*batch))
        return (
            torch.as_tensor(np.asarray(fields[0]), dtype=torch.float32),
            torch.as_tensor(np.asarray(fields[1]), dtype=torch.float32),
            torch.as_tensor(np.asarray(fields[2]), dtype=torch.long),
            torch.as_tensor(np.asarray(fields[3]), dtype=torch.float32),
            torch.as_tensor(np.asarray(fields[4]), dtype=torch.float32),
            torch.as_tensor(np.asarray(fields[5]), dtype=torch.float32),
            torch.as_tensor(np.asarray(fields[6]), dtype=torch.bool),
            torch.as_tensor(np.asarray(fields[7]), dtype=torch.float32),
        )


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


def epsilon_by_step(args: argparse.Namespace, total_env_steps: int) -> float:
    fraction = min(1.0, total_env_steps / max(1, args.epsilon_decay_steps))
    return args.epsilon_start + fraction * (args.epsilon_end - args.epsilon_start)


def update_target(target: nn.Module, online: nn.Module, tau: float) -> None:
    if tau >= 1.0:
        target.load_state_dict(online.state_dict())
        return
    with torch.no_grad():
        for target_parameter, parameter in zip(
            target.parameters(), online.parameters(), strict=True
        ):
            target_parameter.mul_(1.0 - tau).add_(parameter, alpha=tau)


def optimize(
    agent_net: AgentQNetwork,
    mixer: QMixer,
    target_agent_net: AgentQNetwork,
    target_mixer: QMixer,
    optimizer: torch.optim.Optimizer,
    replay: JointReplayBuffer,
    args: argparse.Namespace,
    device: torch.device,
) -> float | None:
    if len(replay) < max(args.batch_size, args.learning_starts):
        return None
    (
        observations,
        states,
        actions,
        rewards,
        next_observations,
        next_states,
        next_masks,
        dones,
    ) = replay.sample(args.batch_size)
    observations = observations.to(device)
    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_observations = next_observations.to(device)
    next_states = next_states.to(device)
    next_masks = next_masks.to(device)
    dones = dones.to(device)
    batch_size, num_agents, obs_dim = observations.shape

    all_q = agent_net(observations.reshape(batch_size * num_agents, obs_dim)).view(
        batch_size, num_agents, -1
    )
    chosen_q = all_q.gather(2, actions.unsqueeze(-1)).squeeze(-1)
    mixed_q = mixer(chosen_q, states)

    with torch.no_grad():
        target_next_q = target_agent_net(
            next_observations.reshape(batch_size * num_agents, obs_dim)
        ).view(batch_size, num_agents, -1)
        target_next_q = target_next_q.masked_fill(~next_masks, -1e9)
        if args.double_q:
            online_next_q = agent_net(
                next_observations.reshape(batch_size * num_agents, obs_dim)
            ).view(batch_size, num_agents, -1)
            online_next_q = online_next_q.masked_fill(~next_masks, -1e9)
            next_actions = online_next_q.argmax(dim=2, keepdim=True)
            target_agent_values = target_next_q.gather(2, next_actions).squeeze(-1)
        else:
            target_agent_values = target_next_q.max(dim=2).values
        target_total = target_mixer(target_agent_values, next_states)
        td_target = rewards + args.gamma * (1.0 - dones) * target_total
    loss = F.smooth_l1_loss(mixed_q, td_target, beta=args.huber_delta)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(
        list(agent_net.parameters()) + list(mixer.parameters()), args.grad_clip
    )
    optimizer.step()
    return float(loss.detach().cpu())


def algorithm_metadata(
    args: argparse.Namespace,
    env: SatTaskingEnv,
    agent_net: AgentQNetwork,
    mixer: QMixer,
    input_dim: int,
    state_dim: int,
    action_dim: int,
) -> dict[str, Any]:
    return {
        "name": "qmix",
        "family": "value_decomposition_marl",
        "training_mode": "centralized_monotonic_value_mixing",
        "execution_mode": "decentralized_local_q_values",
        "parameter_sharing": True,
        "agent_count": env.config.num_satellites,
        "action_masking": True,
        "coordination": "qmix_team_value_plus_environment_auction",
        "credit_assignment": "monotonic_joint_action_value_team_reward_sum",
        "network": {
            "agent_input_dim": input_dim,
            "mixer_state_dim": state_dim,
            "output_dim": action_dim,
            "hidden_dim": args.hidden_dim,
            "hidden_layers": args.hidden_layers,
            "mixer_embed_dim": args.mixer_embed_dim,
            "parameter_count": parameter_count(agent_net, mixer),
        },
        "optimizer": {
            "name": "adamw",
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.grad_clip,
        },
        "temporal_difference": {
            "double_q": args.double_q,
            "gamma": args.gamma,
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
            "joint_replay_capacity": args.buffer_size,
            "batch_size": args.batch_size,
            "learning_starts": args.learning_starts,
            "gradient_steps": args.gradient_steps,
            "reward_scale": args.reward_scale,
            "reward_clip": args.reward_clip,
        },
    }


def save_checkpoint(
    path: Path,
    agent_net: AgentQNetwork,
    mixer: QMixer,
    target_agent_net: AgentQNetwork,
    target_mixer: QMixer,
    optimizer: torch.optim.Optimizer,
    metrics: dict[str, Any],
) -> None:
    atomic_torch_save(
        path,
        {
            "algorithm_id": "qmix",
            "agent_net": agent_net.state_dict(),
            "mixer": mixer.state_dict(),
            "target_agent_net": target_agent_net.state_dict(),
            "target_mixer": target_mixer.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--satellites", type=int, default=64)
    parser.add_argument("--tasks", type=int, default=256)
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
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--activation", choices=["relu", "silu", "tanh"], default="relu")
    parser.add_argument("--layer-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mixer-embed-dim", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-starts", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--double-q", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--reward-clip", type=float, default=10.0)
    parser.add_argument("--epsilon-start", type=float, default=0.8)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=50000)
    parser.add_argument("--target-update-steps", type=int, default=800)
    parser.add_argument("--target-tau", type=float, default=1.0)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--metrics-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
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
    _, local_sample, state_sample, mask_sample = flatten_local_all(observations)
    input_dim = local_sample.shape[1]
    state_dim = len(state_sample)
    action_dim = mask_sample.shape[1]
    num_agents = len(local_sample)

    agent_net = AgentQNetwork(
        input_dim,
        action_dim,
        args.hidden_dim,
        args.hidden_layers,
        args.activation,
        args.layer_norm,
    ).to(device)
    target_agent_net = AgentQNetwork(
        input_dim,
        action_dim,
        args.hidden_dim,
        args.hidden_layers,
        args.activation,
        args.layer_norm,
    ).to(device)
    mixer = QMixer(num_agents, state_dim, args.mixer_embed_dim).to(device)
    target_mixer = QMixer(num_agents, state_dim, args.mixer_embed_dim).to(device)
    target_agent_net.load_state_dict(agent_net.state_dict())
    target_mixer.load_state_dict(mixer.state_dict())
    optimizer = torch.optim.AdamW(
        list(agent_net.parameters()) + list(mixer.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    replay = JointReplayBuffer(args.buffer_size)

    metrics_path = args.run_dir / "metrics.json"
    checkpoint_path = args.run_dir / "checkpoints" / "latest.pt"
    stop_file = args.stop_file or args.run_dir / "STOP"
    history: list[dict[str, Any]] = []
    loss_window: deque[float] = deque(maxlen=200)
    start_episode = 1
    total_env_steps = 0
    total_agent_steps = 0
    started_at = time.time()
    metrics: dict[str, Any] = {
        "status": "running",
        "started_at": started_at,
        "updated_at": started_at,
        "device": str(device),
        "project_root": str(PROJECT_ROOT),
        "args": serialize_args(args),
        "env_config": asdict(env.config),
        "algorithm": algorithm_metadata(
            args, env, agent_net, mixer, input_dim, state_dim, action_dim
        ),
        "input_dim": input_dim,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "episode": 0,
        "total_env_steps": 0,
        "total_agent_steps": 0,
        "replay_size": 0,
        "history": history,
        "last": {},
    }

    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        agent_net.load_state_dict(checkpoint["agent_net"])
        mixer.load_state_dict(checkpoint["mixer"])
        target_agent_net.load_state_dict(checkpoint["target_agent_net"])
        target_mixer.load_state_dict(checkpoint["target_mixer"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        previous = dict(checkpoint.get("metrics") or {})
        start_episode = int(previous.get("episode", 0)) + 1
        total_env_steps = int(previous.get("total_env_steps", 0))
        total_agent_steps = int(previous.get("total_agent_steps", 0))
        history = list(previous.get("history") or [])[-600:]
        metrics.update(previous)
        metrics.update(
            {
                "status": "running",
                "updated_at": time.time(),
                "args": serialize_args(args),
                "env_config": asdict(env.config),
                "algorithm": algorithm_metadata(
                    args, env, agent_net, mixer, input_dim, state_dim, action_dim
                ),
                "history": history,
                "replay_size": 0,
                "resumed_at": time.time(),
            }
        )
    atomic_write_json(metrics_path, metrics)

    try:
        for episode in range(start_episode, args.episodes + 1):
            if stop_file.exists():
                metrics.update(
                    {"status": "stopped", "updated_at": time.time(), "episode": episode - 1}
                )
                atomic_write_json(metrics_path, metrics)
                save_checkpoint(
                    checkpoint_path,
                    agent_net,
                    mixer,
                    target_agent_net,
                    target_mixer,
                    optimizer,
                    metrics,
                )
                break
            episode_started = time.time()
            observations, _ = env.reset(
                seed=scenario_seed(args.seed, episode, args.scenario_seed_cycle)
            )
            raw_episode_reward = 0.0
            episode_steps = 0
            last_loss = None
            for _ in range(args.max_steps):
                agents, local_obs, state, masks = flatten_local_all(observations)
                with torch.no_grad():
                    q_values = agent_net(
                        torch.as_tensor(local_obs, dtype=torch.float32, device=device)
                    ).cpu().numpy()
                q_values[~masks] = -1e9
                greedy_actions = np.argmax(q_values, axis=1).astype(np.int64)
                epsilon = epsilon_by_step(args, total_env_steps)
                actions = greedy_actions.copy()
                explore = rng.random(num_agents) < epsilon
                for row in np.flatnonzero(explore):
                    valid = np.flatnonzero(masks[row])
                    actions[row] = int(rng.choice(valid)) if len(valid) else 0
                action_dict = {
                    agent: int(actions[row]) for row, agent in enumerate(agents)
                }
                next_observations, rewards, terminations, truncations, _ = env.step(
                    action_dict
                )
                _, next_local, next_state, next_masks = flatten_local_all(
                    next_observations
                )
                mean_reward = float(np.mean(list(rewards.values())))
                team_reward = float(np.sum(list(rewards.values())))
                raw_episode_reward += mean_reward
                learning_reward = team_reward * args.reward_scale
                if args.reward_clip > 0:
                    learning_reward = float(
                        np.clip(
                            learning_reward, -args.reward_clip, args.reward_clip
                        )
                    )
                done = all(terminations.values()) or all(truncations.values())
                replay.add(
                    local_obs,
                    state,
                    actions,
                    learning_reward,
                    next_local,
                    next_state,
                    next_masks,
                    done,
                )
                for _ in range(args.gradient_steps):
                    loss = optimize(
                        agent_net,
                        mixer,
                        target_agent_net,
                        target_mixer,
                        optimizer,
                        replay,
                        args,
                        device,
                    )
                    if loss is not None:
                        last_loss = loss
                        loss_window.append(loss)
                total_env_steps += 1
                total_agent_steps += num_agents
                episode_steps += 1
                observations = next_observations
                if total_env_steps % args.target_update_steps == 0:
                    update_target(target_agent_net, agent_net, args.target_tau)
                    update_target(target_mixer, mixer, args.target_tau)
                if done:
                    break

            summary = env.summary()
            episode_seconds = max(1e-6, time.time() - episode_started)
            record = {
                "episode": episode,
                "env_steps": episode_steps,
                "mean_reward": raw_episode_reward / max(1, episode_steps),
                "reward_per_opportunity": raw_episode_reward
                * num_agents
                / max(1, summary["decision_opportunities"]),
                "decision_opportunities": summary["decision_opportunities"],
                "decision_opportunity_rate": summary["decision_opportunity_rate"],
                "forced_idle_actions": summary["forced_idle_actions"],
                "avoidable_idle_actions": summary["avoidable_idle_actions"],
                "completed_tasks": summary["completed_tasks"],
                "cooperative_completed_tasks": summary["cooperative_completed_tasks"],
                "expired_tasks": summary["expired_tasks"],
                "total_priority_completed": summary["total_priority_completed"],
                "mean_observation_quality": summary["mean_observation_quality"],
                "conflicts": summary["total_conflicts"],
                "ground_conflicts": summary["total_ground_conflicts"],
                "invalid_actions": summary["total_invalid_actions"],
                "window_misses": summary["total_window_misses"],
                "busy_satellites": summary["busy_satellites"],
                "pending_data_mb": summary["pending_data_mb"],
                "downlinked_data_mb": summary["downlinked_data_mb"],
                "epsilon": epsilon_by_step(args, total_env_steps),
                "loss": last_loss,
                "mean_loss": float(np.mean(loss_window)) if loss_window else None,
                "replay_size": len(replay),
                "total_env_steps": total_env_steps,
                "total_agent_steps": total_agent_steps,
                "episode_seconds": episode_seconds,
                "env_steps_per_second": episode_steps / episode_seconds,
                "agent_steps_per_second": episode_steps
                * num_agents
                / episode_seconds,
                "timestamp": time.time(),
            }
            history.append(record)
            metrics.update(
                {
                    "status": "running",
                    "updated_at": time.time(),
                    "episode": episode,
                    "total_env_steps": total_env_steps,
                    "total_agent_steps": total_agent_steps,
                    "epsilon": record["epsilon"],
                    "replay_size": len(replay),
                    "last": record,
                    "history": history,
                }
            )
            if episode % args.metrics_every == 0:
                atomic_write_json(metrics_path, metrics)
            if episode % args.checkpoint_every == 0:
                save_checkpoint(
                    checkpoint_path,
                    agent_net,
                    mixer,
                    target_agent_net,
                    target_mixer,
                    optimizer,
                    metrics,
                )
        else:
            metrics.update(
                {"status": "complete", "updated_at": time.time(), "episode": args.episodes}
            )
            atomic_write_json(metrics_path, metrics)
            save_checkpoint(
                checkpoint_path,
                agent_net,
                mixer,
                target_agent_net,
                target_mixer,
                optimizer,
                metrics,
            )
    except Exception as exc:
        metrics.update(
            {"status": "failed", "updated_at": time.time(), "error": repr(exc)}
        )
        atomic_write_json(metrics_path, metrics)
        raise


if __name__ == "__main__":
    main()
