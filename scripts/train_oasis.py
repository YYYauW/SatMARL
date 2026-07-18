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
from torch.distributions import Categorical

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from marl_common import (
    ActorCritic,
    FastSlowOpportunityGraphActorCritic,
    OpportunityGraphActorCritic,
    atomic_torch_save,
    atomic_write_json,
    critic_observations,
    flatten_local_all,
    flatten_opportunity_graph_all,
    masked_logits,
    parameter_count,
    resolve_device,
    scenario_seed,
    serialize_args,
    slow_strategy_features,
)
from oasis_common import (
    OpportunityCurriculum,
    compute_opportunity_gae,
    decision_opportunities,
    opportunity_balance_weights,
    semantic_opportunity_balance_weights,
    semantic_opportunity_classes,
)
from sat_marl_env import EnvConfig, SatTaskingEnv


def make_config(args: argparse.Namespace) -> EnvConfig:
    return EnvConfig(
        num_satellites=args.satellites,
        num_tasks=args.tasks,
        num_planes=args.planes,
        max_steps=args.max_steps,
        candidate_k=args.candidate_k,
        neighbor_k=args.neighbor_k,
        step_duration_seconds=args.step_duration_seconds,
        random_seed=args.seed,
        task_layout=args.task_layout,
        curriculum_visible_fraction=args.curriculum_visible_fraction,
        curriculum_ground_track_jitter_deg=args.curriculum_ground_track_jitter_deg,
        curriculum_time_jitter_steps=args.curriculum_time_jitter_steps,
        curriculum_payload_match_probability=args.curriculum_payload_match_probability,
        min_observation_elevation_deg=args.min_observation_elevation_deg,
        max_off_nadir_deg=args.max_off_nadir_deg,
        payload_fov_min_deg=args.fov_deg,
        payload_fov_max_deg=args.fov_deg,
        point_observation_seconds=args.point_observation_seconds,
        optical_min_sun_elevation_deg=args.optical_min_sun_elevation_deg,
        min_task_window=args.min_task_window,
        max_task_window=args.max_task_window,
        planning_lookahead_steps=args.planning_lookahead_steps,
    )


def algorithm_metadata(
    args: argparse.Namespace,
    env: SatTaskingEnv,
    model: nn.Module,
    actor_dim: int,
    critic_dim: int,
    action_dim: int,
) -> dict[str, Any]:
    centralized = True
    return {
        "name": (
            "oasis_fast_slow_graph"
            if args.architecture == "fast_slow_graph"
            else "oasis_graph"
            if args.architecture == "opportunity_graph"
            else "oasis_mappo"
        ),
        "family": "opportunity_aware_asynchronous_marl",
        "training_mode": "centralized_critic_decentralized_actor",
        "execution_mode": "decentralized_local_observation",
        "parameter_sharing": True,
        "agent_count": env.config.num_satellites,
        "action_masking": True,
        "coordination": (
            "learned_task_factor_bids_plus_hard_feasibility"
            if args.architecture in {"opportunity_graph", "fast_slow_graph"}
            and args.learned_resource_bids
            else "task_factor_policy_plus_environment_tiebreak"
            if args.architecture in {"opportunity_graph", "fast_slow_graph"}
            else "environment_auction_and_hard_feasibility"
        ),
        "architecture": args.architecture,
        "learned_resource_bids": (
            args.architecture in {"opportunity_graph", "fast_slow_graph"}
            and args.learned_resource_bids
        ),
        "opportunity_graph": {
            "factor_nodes": "tasks",
            "edge_message_dim": 4,
            "factor_messages": args.graph_factor_messages,
            "complexity": "O(active_satellites * candidate_k)",
            "size_invariant_parameters": args.architecture
            in {"opportunity_graph", "fast_slow_graph"},
        },
        "temporal_hierarchy": {
            "enabled": args.architecture == "fast_slow_graph",
            "slow_interval_steps": args.slow_interval,
            "slow_interval_seconds": args.slow_interval * args.step_duration_seconds,
            "intent_dim": args.slow_intent_dim,
            "slow_input": "persistent_resource_and_task_factor_snapshot",
            "fast_policy": "opportunity_conditioned_task_and_downlink_decisions",
        },
        "credit_assignment": "duration_corrected_opportunity_gae",
        "policy_samples": "balanced_decision_opportunities_only",
        "opportunity_learning": {
            "asynchronous_decision_epochs": True,
            "duration_corrected_discount": True,
            "opportunity_balancing": args.opportunity_balancing,
            "semantic_opportunity_balancing": args.semantic_opportunity_balancing,
            "adaptive_curriculum": args.adaptive_curriculum,
            "target_opportunity_rate": args.target_opportunity_rate,
            "cross_episode_min_decision_samples": args.min_decision_samples,
            "max_buffered_episodes": args.max_buffered_episodes,
        },
        "network": {
            "actor_input_dim": actor_dim,
            "critic_input_dim": critic_dim,
            "output_dim": action_dim,
            "hidden_dim": args.hidden_dim,
            "hidden_layers": args.hidden_layers,
            "activation": args.activation,
            "layer_norm": args.layer_norm,
            "parameter_count": parameter_count(model),
        },
        "optimizer": {
            "name": "adamw",
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.grad_clip,
        },
        "ppo": {
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_coef": args.clip_coef,
            "value_clip_coef": args.value_clip_coef,
            "entropy_coef": args.entropy_coef,
            "value_coef": args.value_coef,
            "update_epochs": args.update_epochs,
            "minibatch_size": args.minibatch_size,
            "normalize_advantages": args.normalize_advantages,
            "reward_scale": args.reward_scale,
            "reward_clip": args.reward_clip,
        },
    }


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = np.zeros(rewards.shape[1], dtype=np.float32)
    for step in range(rewards.shape[0] - 1, -1, -1):
        next_values = (
            values[step + 1]
            if step + 1 < rewards.shape[0]
            else np.zeros(rewards.shape[1], dtype=np.float32)
        )
        nonterminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_values * nonterminal - values[step]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[step] = gae
    return advantages, advantages + values


def ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    rollout: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, float]:
    actor_obs = torch.as_tensor(
        rollout["actor_obs"].reshape(-1, rollout["actor_obs"].shape[-1]),
        dtype=torch.float32,
        device=device,
    )
    critic_obs = torch.as_tensor(
        rollout["critic_obs"].reshape(-1, rollout["critic_obs"].shape[-1]),
        dtype=torch.float32,
        device=device,
    )
    masks = torch.as_tensor(
        rollout["masks"].reshape(-1, rollout["masks"].shape[-1]),
        dtype=torch.bool,
        device=device,
    )
    actions = torch.as_tensor(
        rollout["actions"].reshape(-1), dtype=torch.long, device=device
    )
    old_log_probs = torch.as_tensor(
        rollout["log_probs"].reshape(-1), dtype=torch.float32, device=device
    )
    old_values = torch.as_tensor(
        rollout["values"].reshape(-1), dtype=torch.float32, device=device
    )
    advantages = torch.as_tensor(
        rollout["advantages"].reshape(-1), dtype=torch.float32, device=device
    )
    returns = torch.as_tensor(
        rollout["returns"].reshape(-1), dtype=torch.float32, device=device
    )
    opportunity_rows = torch.as_tensor(
        rollout["opportunities"].reshape(-1), dtype=torch.bool, device=device
    )
    sample_weights = torch.as_tensor(
        rollout["opportunity_weights"].reshape(-1),
        dtype=torch.float32,
        device=device,
    )
    decision_rows = opportunity_rows
    actor_obs = actor_obs[decision_rows]
    critic_obs = critic_obs[decision_rows]
    masks = masks[decision_rows]
    actions = actions[decision_rows]
    old_log_probs = old_log_probs[decision_rows]
    old_values = old_values[decision_rows]
    advantages = advantages[decision_rows]
    returns = returns[decision_rows]
    sample_weights = sample_weights[decision_rows]
    if len(actions) == 0:
        return {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "decision_samples": 0,
        }
    if args.normalize_advantages:
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )

    sample_count = len(actions)
    losses: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    kls: list[float] = []
    clip_fractions: list[float] = []
    for _ in range(args.update_epochs):
        permutation = rng.permutation(sample_count)
        for start in range(0, sample_count, args.minibatch_size):
            rows = permutation[start : start + args.minibatch_size]
            row_tensor = torch.as_tensor(rows, dtype=torch.long, device=device)
            logits = masked_logits(
                model.policy_logits(actor_obs[row_tensor]), masks[row_tensor]
            )
            distribution = Categorical(logits=logits)
            new_log_probs = distribution.log_prob(actions[row_tensor])
            entropy = distribution.entropy().mean()
            new_values = model.values(critic_obs[row_tensor])

            log_ratio = new_log_probs - old_log_probs[row_tensor]
            ratio = log_ratio.exp()
            unclipped = ratio * advantages[row_tensor]
            clipped = (
                torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                * advantages[row_tensor]
            )
            row_weights = sample_weights[row_tensor]
            row_weights = row_weights / row_weights.mean().clamp_min(1e-8)
            policy_loss = -(torch.min(unclipped, clipped) * row_weights).mean()

            value_delta = new_values - old_values[row_tensor]
            clipped_values = old_values[row_tensor] + torch.clamp(
                value_delta, -args.value_clip_coef, args.value_clip_coef
            )
            value_loss_unclipped = (new_values - returns[row_tensor]).pow(2)
            value_loss_clipped = (clipped_values - returns[row_tensor]).pow(2)
            value_loss = 0.5 * (
                torch.max(value_loss_unclipped, value_loss_clipped) * row_weights
            ).mean()
            loss = (
                policy_loss
                + args.value_coef * value_loss
                - args.entropy_coef * entropy
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            with torch.no_grad():
                approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    (torch.abs(ratio - 1.0) > args.clip_coef).float().mean()
                )
            losses.append(float(loss.detach().cpu()))
            policy_losses.append(float(policy_loss.detach().cpu()))
            value_losses.append(float(value_loss.detach().cpu()))
            entropies.append(float(entropy.detach().cpu()))
            kls.append(float(approximate_kl.detach().cpu()))
            clip_fractions.append(float(clip_fraction.detach().cpu()))

    return {
        "loss": float(np.mean(losses)),
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
        "approx_kl": float(np.mean(kls)),
        "clip_fraction": float(np.mean(clip_fractions)),
        "decision_samples": sample_count,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: dict[str, Any],
    algorithm: str,
) -> None:
    atomic_torch_save(
        path,
        {
            "algorithm_id": algorithm,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--satellites", type=int, default=64)
    parser.add_argument("--tasks", type=int, default=768)
    parser.add_argument("--planes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--candidate-k", type=int, default=24)
    parser.add_argument("--neighbor-k", type=int, default=6)
    parser.add_argument(
        "--task-layout",
        choices=["global_random", "curriculum_visible", "mixed", "mixed_curriculum"],
        default="mixed",
    )
    parser.add_argument("--curriculum-visible-fraction", type=float, default=0.85)
    parser.add_argument("--curriculum-ground-track-jitter-deg", type=float, default=5.0)
    parser.add_argument("--curriculum-time-jitter-steps", type=int, default=3)
    parser.add_argument("--curriculum-payload-match-probability", type=float, default=0.8)
    parser.add_argument("--min-observation-elevation-deg", type=float, default=3.0)
    parser.add_argument("--step-duration-seconds", type=float, default=30.0)
    parser.add_argument("--point-observation-seconds", type=float, default=5.0)
    parser.add_argument("--fov-deg", type=float, default=45.0)
    parser.add_argument("--max-off-nadir-deg", type=float, default=45.0)
    parser.add_argument("--optical-min-sun-elevation-deg", type=float, default=8.0)
    parser.add_argument("--min-task-window", type=int, default=40)
    parser.add_argument("--max-task-window", type=int, default=160)
    parser.add_argument("--planning-lookahead-steps", type=int, default=30)
    parser.add_argument("--scenario-seed-cycle", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument(
        "--architecture",
        choices=["mlp", "opportunity_graph", "fast_slow_graph"],
        default="mlp",
        help="Choose MLP, graph, or bi-timescale graph actor.",
    )
    parser.add_argument(
        "--graph-factor-messages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ablation switch for satellite--task contention messages.",
    )
    parser.add_argument(
        "--learned-resource-bids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ablation switch for using policy logits as conflict-resolution bids.",
    )
    parser.add_argument(
        "--slow-interval",
        type=int,
        default=8,
        help="Refresh strategic intent every K environment steps for fast_slow_graph.",
    )
    parser.add_argument("--slow-intent-dim", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--activation", choices=["relu", "silu", "tanh"], default="relu")
    parser.add_argument("--layer-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument(
        "--opportunity-gae", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--opportunity-balancing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--semantic-opportunity-balancing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Balance downlink-only, single-task, and multi-task decisions separately.",
    )
    parser.add_argument(
        "--adaptive-curriculum", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--curriculum-end-fraction", type=float, default=0.15)
    parser.add_argument("--target-opportunity-rate", type=float, default=0.12)
    parser.add_argument("--curriculum-feedback-gain", type=float, default=1.5)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-clip-coef", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--min-decision-samples", type=int, default=512)
    parser.add_argument("--max-buffered-episodes", type=int, default=16)
    parser.add_argument("--normalize-advantages", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--reward-clip", type=float, default=10.0)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--metrics-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-file", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.slow_interval < 1:
        parser.error("--slow-interval must be at least 1")
    if args.slow_intent_dim < 1:
        parser.error("--slow-intent-dim must be at least 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)
    centralized = True

    env = SatTaskingEnv(make_config(args))
    curriculum = OpportunityCurriculum(
        start_fraction=args.curriculum_visible_fraction,
        end_fraction=args.curriculum_end_fraction,
        target_opportunity_rate=args.target_opportunity_rate,
        feedback_gain=args.curriculum_feedback_gain,
    )
    observations, _ = env.reset(seed=args.seed)
    graph_actor = args.architecture in {"opportunity_graph", "fast_slow_graph"}
    fast_slow_actor = args.architecture == "fast_slow_graph"
    flatten_actor = (
        flatten_opportunity_graph_all if graph_actor else flatten_local_all
    )

    def base_actor_observations(current_observations):
        flattened = flatten_actor(current_observations)
        if graph_actor and not args.graph_factor_messages:
            agents, local_obs, global_state, masks = flattened
            local_obs = local_obs.copy()
            local_obs[:, -env.config.candidate_k * 4 :] = 0.0
            return agents, local_obs, global_state, masks
        return flattened

    agents_sample, graph_sample, global_sample, masks_sample = base_actor_observations(
        observations
    )
    if fast_slow_actor:
        slow_sample = slow_strategy_features(
            graph_sample, env.config.candidate_k, env.config.neighbor_k
        )
        local_sample = np.concatenate([graph_sample, slow_sample], axis=1)
    else:
        local_sample = graph_sample
    actor_dim = local_sample.shape[1]
    critic_dim = actor_dim + len(global_sample) if centralized else actor_dim
    action_dim = masks_sample.shape[1]
    if fast_slow_actor:
        model = FastSlowOpportunityGraphActorCritic(
            critic_dim=critic_dim,
            candidate_k=env.config.candidate_k,
            neighbor_k=env.config.neighbor_k,
            hidden_dim=args.hidden_dim,
            intent_dim=args.slow_intent_dim,
            activation=args.activation,
            layer_norm=args.layer_norm,
        ).to(device)
    elif graph_actor:
        model = OpportunityGraphActorCritic(
            critic_dim=critic_dim,
            candidate_k=env.config.candidate_k,
            neighbor_k=env.config.neighbor_k,
            hidden_dim=args.hidden_dim,
            activation=args.activation,
            layer_norm=args.layer_norm,
        ).to(device)
    else:
        model = ActorCritic(
            actor_dim,
            critic_dim,
            action_dim,
            args.hidden_dim,
            args.hidden_layers,
            args.activation,
            args.layer_norm,
        ).to(device)
    algorithm_id = (
        "oasis_fast_slow_graph"
        if fast_slow_actor
        else "oasis_graph"
        if graph_actor
        else "oasis"
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    metrics_path = args.run_dir / "metrics.json"
    checkpoint_path = args.run_dir / "checkpoints" / "latest.pt"
    stop_file = args.stop_file or args.run_dir / "STOP"
    history: list[dict[str, Any]] = []
    loss_window: deque[float] = deque(maxlen=100)
    pending_rollouts: list[dict[str, np.ndarray]] = []
    pending_decision_samples = 0
    latest_update_metrics: dict[str, float | int | bool] = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "decision_samples": 0,
        "update_performed": False,
        "update_batch_episodes": 0,
    }
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
            args, env, model, actor_dim, critic_dim, action_dim
        ),
        "input_dim": actor_dim,
        "critic_input_dim": critic_dim,
        "action_dim": action_dim,
        "episode": 0,
        "total_env_steps": 0,
        "total_agent_steps": 0,
        "history": history,
        "last": {},
    }

    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
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
                    args, env, model, actor_dim, critic_dim, action_dim
                ),
                "history": history,
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
                save_checkpoint(checkpoint_path, model, optimizer, metrics, algorithm_id)
                break

            episode_started = time.time()
            if args.adaptive_curriculum:
                env.config.task_layout = "mixed"
                env.config.curriculum_visible_fraction = curriculum.value(
                    episode, args.episodes
                )
            else:
                env.config.curriculum_visible_fraction = args.curriculum_visible_fraction
            observations, _ = env.reset(
                seed=scenario_seed(args.seed, episode, args.scenario_seed_cycle)
            )
            actor_rows: list[np.ndarray] = []
            critic_rows: list[np.ndarray] = []
            mask_rows: list[np.ndarray] = []
            action_rows: list[np.ndarray] = []
            log_prob_rows: list[np.ndarray] = []
            value_rows: list[np.ndarray] = []
            reward_rows: list[np.ndarray] = []
            done_rows: list[np.ndarray] = []
            raw_episode_reward = 0.0
            slow_snapshot: np.ndarray | None = None
            slow_refreshes = 0

            for step_index in range(args.max_steps):
                agents, graph_obs, global_state, masks = base_actor_observations(
                    observations
                )
                if fast_slow_actor:
                    if slow_snapshot is None or step_index % args.slow_interval == 0:
                        slow_snapshot = slow_strategy_features(
                            graph_obs, env.config.candidate_k, env.config.neighbor_k
                        )
                        slow_refreshes += 1
                    local_obs = np.concatenate([graph_obs, slow_snapshot], axis=1)
                else:
                    local_obs = graph_obs
                critic_obs = critic_observations(
                    local_obs, global_state, centralized
                )
                with torch.no_grad():
                    local_tensor = torch.as_tensor(
                        local_obs, dtype=torch.float32, device=device
                    )
                    critic_tensor = torch.as_tensor(
                        critic_obs, dtype=torch.float32, device=device
                    )
                    mask_tensor = torch.as_tensor(masks, dtype=torch.bool, device=device)
                    policy_logits = masked_logits(
                        model.policy_logits(local_tensor), mask_tensor
                    )
                    distribution = Categorical(logits=policy_logits)
                    sampled_actions = distribution.sample()
                    log_probs = distribution.log_prob(sampled_actions)
                    values = model.values(critic_tensor)
                action_array = sampled_actions.cpu().numpy().astype(np.int64)
                if graph_actor and args.learned_resource_bids:
                    bid_array = policy_logits.cpu().numpy()
                    action_dict = {
                        agent: (
                            {
                                "action": int(action_array[row]),
                                "bid": float(bid_array[row, action_array[row]]),
                            }
                            if int(action_array[row]) >= 2
                            else int(action_array[row])
                        )
                        for row, agent in enumerate(agents)
                    }
                else:
                    action_dict = {
                        agent: int(action_array[row]) for row, agent in enumerate(agents)
                    }
                next_observations, rewards, terminations, truncations, _ = env.step(
                    action_dict
                )
                reward_array = np.asarray(
                    [rewards[agent] for agent in agents], dtype=np.float32
                )
                raw_episode_reward += float(np.mean(reward_array))
                learning_rewards = reward_array * args.reward_scale
                if args.reward_clip > 0:
                    learning_rewards = np.clip(
                        learning_rewards, -args.reward_clip, args.reward_clip
                    )
                done = all(terminations.values()) or all(truncations.values())

                actor_rows.append(local_obs)
                critic_rows.append(critic_obs)
                mask_rows.append(masks)
                action_rows.append(action_array)
                log_prob_rows.append(log_probs.cpu().numpy())
                value_rows.append(values.cpu().numpy())
                reward_rows.append(learning_rewards)
                done_rows.append(
                    np.full(len(agents), float(done), dtype=np.float32)
                )
                observations = next_observations
                total_env_steps += 1
                total_agent_steps += len(agents)
                if done:
                    break

            rollout = {
                "actor_obs": np.asarray(actor_rows, dtype=np.float32),
                "critic_obs": np.asarray(critic_rows, dtype=np.float32),
                "masks": np.asarray(mask_rows, dtype=bool),
                "actions": np.asarray(action_rows, dtype=np.int64),
                "log_probs": np.asarray(log_prob_rows, dtype=np.float32),
                "values": np.asarray(value_rows, dtype=np.float32),
                "rewards": np.asarray(reward_rows, dtype=np.float32),
                "dones": np.asarray(done_rows, dtype=np.float32),
            }
            opportunities = decision_opportunities(rollout["masks"])
            opportunity_classes = semantic_opportunity_classes(rollout["masks"])
            if args.opportunity_gae:
                advantages, returns, durations = compute_opportunity_gae(
                    rollout["rewards"],
                    rollout["values"],
                    rollout["dones"],
                    opportunities,
                    args.gamma,
                    args.gae_lambda,
                )
            else:
                advantages, returns = compute_gae(
                    rollout["rewards"],
                    rollout["values"],
                    rollout["dones"],
                    args.gamma,
                    args.gae_lambda,
                )
                durations = opportunities.astype(np.int32)
            rollout["opportunities"] = opportunities
            if args.semantic_opportunity_balancing:
                rollout["opportunity_weights"] = semantic_opportunity_balance_weights(
                    rollout["masks"], opportunities
                )
            elif args.opportunity_balancing:
                rollout["opportunity_weights"] = opportunity_balance_weights(
                    rollout["masks"], opportunities
                )
            else:
                rollout["opportunity_weights"] = opportunities.astype(np.float32)
            rollout["advantages"] = advantages
            rollout["returns"] = returns
            episode_decision_samples = int(np.count_nonzero(opportunities))
            task_decision_samples = int(np.count_nonzero(opportunity_classes >= 2))
            downlink_only_samples = int(np.count_nonzero(opportunity_classes == 1))
            pending_rollouts.append(rollout)
            pending_decision_samples += episode_decision_samples
            buffered_episodes = len(pending_rollouts)
            should_update = (
                pending_decision_samples >= max(1, args.min_decision_samples)
                or buffered_episodes >= max(1, args.max_buffered_episodes)
                or episode == args.episodes
            )
            if should_update:
                combined_rollout = {
                    key: np.concatenate([item[key] for item in pending_rollouts], axis=0)
                    for key in rollout
                }
                update_metrics = ppo_update(
                    model, optimizer, combined_rollout, args, device, rng
                )
                update_metrics.update(
                    {
                        "update_performed": True,
                        "update_batch_episodes": buffered_episodes,
                    }
                )
                latest_update_metrics = update_metrics
                loss_window.append(float(update_metrics["loss"]))
                pending_rollouts.clear()
                pending_decision_samples = 0
            else:
                update_metrics = {
                    **latest_update_metrics,
                    "update_performed": False,
                    "decision_samples": 0,
                    "update_batch_episodes": 0,
                }
            summary = env.summary()
            next_curriculum_fraction = env.config.curriculum_visible_fraction
            if args.adaptive_curriculum:
                next_curriculum_fraction = curriculum.observe(
                    summary["decision_opportunity_rate"], episode, args.episodes
                )
            episode_seconds = max(1e-6, time.time() - episode_started)
            episode_steps = len(actor_rows)
            record = {
                "episode": episode,
                "env_steps": episode_steps,
                "mean_reward": raw_episode_reward / max(1, episode_steps),
                "reward_per_opportunity": raw_episode_reward
                * len(env.agents)
                / max(1, summary["decision_opportunities"]),
                "decision_opportunities": summary["decision_opportunities"],
                "episode_decision_samples": episode_decision_samples,
                "task_decision_samples": task_decision_samples,
                "downlink_only_decision_samples": downlink_only_samples,
                "task_decision_opportunity_rate": task_decision_samples
                / max(1, episode_steps * len(env.agents)),
                "buffered_decision_samples": pending_decision_samples,
                "buffered_episodes": len(pending_rollouts),
                "decision_opportunity_rate": summary["decision_opportunity_rate"],
                "active_agents_mean": float(np.mean(np.sum(opportunities, axis=1))),
                "slow_strategy_refreshes": slow_refreshes,
                "slow_interval_steps": args.slow_interval if fast_slow_actor else 0,
                "mean_decision_interval": float(
                    np.mean(durations[durations > 0])
                )
                if np.any(durations > 0)
                else 0.0,
                "curriculum_visible_fraction": env.config.curriculum_visible_fraction,
                "next_curriculum_visible_fraction": next_curriculum_fraction,
                "target_opportunity_rate": args.target_opportunity_rate,
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
                "task_reward_total": summary["task_reward_total"],
                "downlink_reward_total": summary["downlink_reward_total"],
                "team_reward_total": summary["team_reward_total"],
                "loss": update_metrics["loss"],
                "mean_loss": float(np.mean(loss_window)) if loss_window else 0.0,
                **update_metrics,
                "total_env_steps": total_env_steps,
                "total_agent_steps": total_agent_steps,
                "episode_seconds": episode_seconds,
                "env_steps_per_second": episode_steps / episode_seconds,
                "agent_steps_per_second": episode_steps
                * len(env.agents)
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
                    "last": record,
                    "history": history,
                }
            )
            if episode % args.metrics_every == 0:
                atomic_write_json(metrics_path, metrics)
            if episode % args.checkpoint_every == 0:
                save_checkpoint(checkpoint_path, model, optimizer, metrics, algorithm_id)

        else:
            metrics.update(
                {
                    "status": "complete",
                    "updated_at": time.time(),
                    "episode": args.episodes,
                }
            )
            atomic_write_json(metrics_path, metrics)
            save_checkpoint(checkpoint_path, model, optimizer, metrics, algorithm_id)
    except Exception as exc:
        metrics.update(
            {"status": "failed", "updated_at": time.time(), "error": repr(exc)}
        )
        atomic_write_json(metrics_path, metrics)
        raise


if __name__ == "__main__":
    main()
