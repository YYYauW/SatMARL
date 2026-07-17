from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate_policy import collect_events, collect_frame, config_from_checkpoint
from marl_common import (
    ActorCritic,
    AgentQNetwork,
    OpportunityGraphActorCritic,
    flatten_local_all,
    flatten_opportunity_graph_all,
    resolve_device,
)
from sat_marl_env import SatTaskingEnv
from train_dqn import QNetwork, flatten_all


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.eval.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def algorithm_id(checkpoint: dict, metrics: dict) -> str:
    explicit = checkpoint.get("algorithm_id")
    if explicit:
        return str(explicit)
    name = str((metrics.get("algorithm") or {}).get("name", "ps_dqn"))
    return "ps_dqn" if "dqn" in name else name


def build_policy(
    checkpoint: dict,
    metrics: dict,
    observations: dict,
    device: torch.device,
) -> tuple[str, Callable[[dict], dict[str, int]]]:
    algo = algorithm_id(checkpoint, metrics)
    train_args = metrics.get("args") or {}
    if algo == "ps_dqn":
        include_global = not bool(train_args.get("local_observation_only", False))
        _, vectors, masks = flatten_all(observations, include_global)
        network = QNetwork(
            vectors.shape[1],
            masks.shape[1],
            int(train_args.get("hidden_dim", 256)),
            int(train_args.get("hidden_layers", 2)),
            str(train_args.get("activation", "relu")),
            bool(train_args.get("layer_norm", True)),
        ).to(device)
        network.load_state_dict(checkpoint["policy"])
        network.eval()

        def choose(current: dict) -> dict[str, int]:
            agents, current_vectors, current_masks = flatten_all(
                current, include_global
            )
            with torch.no_grad():
                q_values = network(
                    torch.as_tensor(
                        current_vectors, dtype=torch.float32, device=device
                    )
                ).cpu().numpy()
            q_values[~current_masks] = -1e9
            actions = np.argmax(q_values, axis=1)
            return {agent: int(actions[row]) for row, agent in enumerate(agents)}

        return algo, choose

    agents, local, global_state, masks = flatten_local_all(observations)
    if algo == "oasis_graph":
        graph_agents, graph_local, graph_global, graph_masks = (
            flatten_opportunity_graph_all(observations)
        )
        candidate_k = int(train_args.get("candidate_k", graph_masks.shape[1] - 2))
        neighbor_k = int(train_args.get("neighbor_k", 6))
        model = OpportunityGraphActorCritic(
            critic_dim=graph_local.shape[1] + len(graph_global),
            candidate_k=candidate_k,
            neighbor_k=neighbor_k,
            hidden_dim=int(train_args.get("hidden_dim", 256)),
            activation=str(train_args.get("activation", "relu")),
            layer_norm=bool(train_args.get("layer_norm", True)),
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        factor_messages = bool(train_args.get("graph_factor_messages", True))
        learned_resource_bids = bool(train_args.get("learned_resource_bids", True))

        def choose_graph(current: dict) -> dict[str, int | dict[str, float | int]]:
            current_agents, current_local, _, current_masks = (
                flatten_opportunity_graph_all(current)
            )
            if not factor_messages:
                current_local = current_local.copy()
                current_local[:, -candidate_k * 4 :] = 0.0
            with torch.no_grad():
                logits = model.policy_logits(
                    torch.as_tensor(
                        current_local, dtype=torch.float32, device=device
                    )
                ).cpu().numpy()
            logits[~current_masks] = -1e9
            actions = np.argmax(logits, axis=1)
            return {
                agent: (
                    {"action": int(actions[row]), "bid": float(logits[row, actions[row]])}
                    if int(actions[row]) >= 2 and learned_resource_bids
                    else int(actions[row])
                )
                for row, agent in enumerate(current_agents)
            }

        return algo, choose_graph

    if algo in {"ippo", "mappo", "oasis"}:
        critic_dim = local.shape[1] + (
            len(global_state) if algo in {"mappo", "oasis"} else 0
        )
        model = ActorCritic(
            local.shape[1],
            critic_dim,
            masks.shape[1],
            int(train_args.get("hidden_dim", 256)),
            int(train_args.get("hidden_layers", 2)),
            str(train_args.get("activation", "relu")),
            bool(train_args.get("layer_norm", True)),
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        def choose(current: dict) -> dict[str, int]:
            current_agents, current_local, _, current_masks = flatten_local_all(
                current
            )
            with torch.no_grad():
                logits = model.policy_logits(
                    torch.as_tensor(
                        current_local, dtype=torch.float32, device=device
                    )
                ).cpu().numpy()
            logits[~current_masks] = -1e9
            actions = np.argmax(logits, axis=1)
            return {
                agent: int(actions[row])
                for row, agent in enumerate(current_agents)
            }

        return algo, choose

    if algo == "qmix":
        agent_network = AgentQNetwork(
            local.shape[1],
            masks.shape[1],
            int(train_args.get("hidden_dim", 256)),
            int(train_args.get("hidden_layers", 2)),
            str(train_args.get("activation", "relu")),
            bool(train_args.get("layer_norm", True)),
        ).to(device)
        agent_network.load_state_dict(checkpoint["agent_net"])
        agent_network.eval()

        def choose(current: dict) -> dict[str, int]:
            current_agents, current_local, _, current_masks = flatten_local_all(
                current
            )
            with torch.no_grad():
                q_values = agent_network(
                    torch.as_tensor(
                        current_local, dtype=torch.float32, device=device
                    )
                ).cpu().numpy()
            q_values[~current_masks] = -1e9
            actions = np.argmax(q_values, axis=1)
            return {
                agent: int(actions[row])
                for row, agent in enumerate(current_agents)
            }

        return algo, choose
    raise ValueError(f"Unsupported algorithm: {algo}")


def task_payload(env: SatTaskingEnv) -> list[dict]:
    return [
        {
            "id": task.task_id,
            "phase": round(float(task.target_phase), 6),
            "latitude": round(float(task.target_lat_deg), 5),
            "longitude": round(float(task.target_lon_deg), 5),
            "priority": round(float(task.priority), 3),
            "release": task.release_step,
            "deadline": task.deadline_step,
            "duration": task.duration,
            "required_mode": task.required_mode,
            "required_resolution_m": round(float(task.required_resolution_m), 3),
            "required_swath_km": round(float(task.required_swath_km), 3),
            "original_data_mb": round(float(task.original_data_mb), 3),
            "compressed_data_mb": round(float(task.compressed_data_mb), 3),
            "cooperation_mode": task.cooperation_mode,
            "required_observers": task.required_observers,
            "observed_by": list(task.observed_by),
            "completed_by": task.completed_by,
            "completed_step": task.completed_step,
            "expired": task.expired,
        }
        for task in env.tasks
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--satellites", type=int, default=None)
    parser.add_argument("--tasks", type=int, default=None)
    parser.add_argument("--planes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument("--neighbor-k", type=int, default=None)
    parser.add_argument(
        "--eval-task-layout",
        choices=["global_random", "curriculum_visible", "mixed", "mixed_curriculum"],
        default=None,
    )
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument(
        "--capture-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store the representative orbit animation; disable for large-scale benchmarks.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    metrics = dict(checkpoint.get("metrics") or {})
    eval_episodes = max(1, args.eval_episodes)
    benchmark_rows: list[dict] = []
    representative_env: SatTaskingEnv | None = None
    representative_frames: list[dict] = []
    representative_schedule: list[dict] = []
    representative_summary: dict = {}
    choose_actions: Callable[[dict], dict[str, int]] | None = None
    algo = "unknown"
    summary_keys = (
        "completed_tasks",
        "cooperative_completed_tasks",
        "expired_tasks",
        "total_priority_completed",
        "mean_observation_quality",
        "downlinked_data_mb",
        "task_reward_total",
        "downlink_reward_total",
        "team_reward_total",
        "total_conflicts",
        "total_ground_conflicts",
        "total_invalid_actions",
        "total_window_misses",
        "decision_opportunity_rate",
        "task_decision_opportunity_rate",
        "avoidable_idle_actions",
        "forced_idle_actions",
    )

    for episode_index in range(eval_episodes):
        episode_seed = args.seed + episode_index
        env = SatTaskingEnv(config_from_checkpoint(metrics, args))
        observations, _ = env.reset(seed=episode_seed)
        if choose_actions is None:
            algo, choose_actions = build_policy(
                checkpoint, metrics, observations, device
            )
        capture_representative = episode_index == 0
        capture_frames = capture_representative and args.capture_frames
        frames = [collect_frame(env)] if capture_frames else []
        schedule_rows: list[dict] = []
        total_reward = 0.0
        episode_steps = 0
        for _ in range(env.config.max_steps):
            actions = choose_actions(observations)
            observations, rewards, terminations, truncations, infos = env.step(actions)
            mean_reward = float(np.mean(list(rewards.values())))
            total_reward += mean_reward
            episode_steps += 1
            if capture_representative:
                events = collect_events(infos)
                if capture_frames:
                    frames.append(collect_frame(env, mean_reward, events))
                for event in events:
                    schedule_rows.append(
                        {
                            "decision_step": env.step_count - 1,
                            "satellite": event.get("agent"),
                            **{key: value for key, value in event.items() if key != "agent"},
                        }
                    )
            if all(terminations.values()) or all(truncations.values()):
                break

        episode_summary = env.summary()
        row = {
            "seed": episode_seed,
            "episode_steps": episode_steps,
            "mean_episode_reward": total_reward / max(1, episode_steps),
            **{key: episode_summary[key] for key in summary_keys},
        }
        benchmark_rows.append(row)
        if capture_representative:
            representative_env = env
            representative_frames = frames
            representative_schedule = schedule_rows
            representative_summary = episode_summary

    assert representative_env is not None
    aggregate: dict[str, dict[str, float]] = {}
    aggregate_keys = ("mean_episode_reward", *summary_keys)
    for key in aggregate_keys:
        values = np.asarray([float(row[key]) for row in benchmark_rows])
        aggregate[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    env = representative_env
    final_summary = representative_summary
    frames = representative_frames
    schedule = representative_schedule
    benchmark = {
        "episodes": eval_episodes,
        "seed_start": args.seed,
        "seed_end": args.seed + eval_episodes - 1,
        "aggregate": aggregate,
        "episodes_detail": benchmark_rows,
    }
    created_at = time.time()
    payload = {
        "status": "complete",
        "created_at": created_at,
        "algorithm_id": algo,
        "checkpoint": str(args.checkpoint),
        "checkpoint_episode": metrics.get("episode"),
        "seed": args.seed,
        "env_config": asdict(env.config),
        "algorithm": metrics.get("algorithm") or {"name": algo},
        "training_args": metrics.get("args") or {},
        "final_summary": final_summary,
        "mean_episode_reward": aggregate["mean_episode_reward"]["mean"],
        "benchmark": benchmark,
        "tasks": task_payload(env),
        "ground_stations": [
            {
                "id": station.station_id,
                "name": station.name,
                "latitude": station.latitude_deg,
                "longitude": station.longitude_deg,
                "channels": station.channel_capacity,
                "rate_mb_per_step": station.rate_mb_per_step,
                "delivered_mb": round(float(station.delivered_mb), 3),
                "conflicts": station.conflict_count,
            }
            for station in env.ground_stations
        ],
        "satellite_specs": [
            {
                "id": sat.sat_id,
                "plane": sat.plane_id,
                "orbital_elements": asdict(sat.orbital_elements),
                "payload_mode": sat.payload_mode,
                "swath_width_km": sat.swath_width_km,
                "field_of_view_deg": sat.field_of_view_deg,
                "resolution_m": sat.resolution_m,
            }
            for sat in env.satellites
        ],
        "frames": frames,
        "schedule": schedule,
    }
    atomic_write_json(args.output, payload)
    schedule_path = args.output.with_name("schedule.csv")
    schedule_fields = (
        "decision_step",
        "satellite",
        "event",
        "task_id",
        "observation_start_step",
        "finish_step",
        "quality",
        "task_reward",
        "completion_bonus",
        "total_energy",
        "compressed_data_mb",
        "observers",
        "station_id",
        "station_name",
        "elevation_deg",
        "downlinked",
        "winner",
        "reason",
    )
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    with schedule_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=schedule_fields, extrasaction="ignore")
        writer.writeheader()
        for row in schedule:
            normalized = dict(row)
            if isinstance(normalized.get("observers"), list):
                normalized["observers"] = ",".join(
                    str(value) for value in normalized["observers"]
                )
            writer.writerow(normalized)
    summary_path = args.output.with_name("summary.json")
    atomic_write_json(
        summary_path,
        {
            "status": "complete",
            "created_at": created_at,
            "algorithm_id": algo,
            "checkpoint_episode": metrics.get("episode"),
            "seed": args.seed,
            "eval_episodes": eval_episodes,
            "final_summary": final_summary,
            "mean_episode_reward": payload["mean_episode_reward"],
            "benchmark": benchmark,
            "rollout": str(args.output),
            "schedule": str(schedule_path),
        },
    )
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
