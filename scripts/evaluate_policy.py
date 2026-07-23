from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sat_marl_env import EnvConfig, SatTaskingEnv
from train_dqn import QNetwork, flatten_all


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def config_from_checkpoint(metrics: dict[str, Any], args: argparse.Namespace) -> EnvConfig:
    cfg = dict(metrics.get("env_config") or {})
    cfg.update(
        {
            "num_satellites": args.satellites or cfg.get("num_satellites", 256),
            "num_tasks": args.tasks or cfg.get("num_tasks", 1024),
            "num_planes": args.planes or cfg.get("num_planes", 8),
            "max_steps": args.max_steps or cfg.get("max_steps", 128),
            "candidate_k": args.candidate_k or cfg.get("candidate_k", 16),
            "neighbor_k": args.neighbor_k or cfg.get("neighbor_k", 6),
            "random_seed": args.seed,
        }
    )
    eval_task_layout = getattr(args, "eval_task_layout", None)
    if eval_task_layout:
        cfg["task_layout"] = eval_task_layout
        if eval_task_layout == "global_random":
            cfg["curriculum_visible_fraction"] = 0.0
    ephemeris_cache = getattr(args, "ephemeris_cache", None)
    task_catalog = getattr(args, "task_catalog", None)
    if (ephemeris_cache is None) != (task_catalog is None):
        raise ValueError(
            "--ephemeris-cache and --task-catalog must be supplied together"
        )
    if ephemeris_cache is not None:
        cfg["ephemeris_cache_path"] = str(ephemeris_cache.resolve())
        cfg["task_catalog_path"] = str(task_catalog.resolve())
        cfg["task_layout"] = "catalog"
    allowed = {field.name for field in EnvConfig.__dataclass_fields__.values()}
    return EnvConfig(**{key: value for key, value in cfg.items() if key in allowed})


def choose_greedy(net: QNetwork, observations, device: torch.device) -> dict[str, int]:
    agents, vecs, masks = flatten_all(observations)
    with torch.no_grad():
        q_values = net(torch.as_tensor(vecs, dtype=torch.float32, device=device))
        q_values = q_values.cpu().numpy()
    q_values[~masks] = -1e9
    greedy = np.argmax(q_values, axis=1).astype(np.int64)
    return {agent: int(greedy[row]) for row, agent in enumerate(agents)}


def collect_frame(env: SatTaskingEnv, step_reward: float = 0.0, events=None) -> dict[str, Any]:
    summary = env.summary()
    completed_ids = [
        task.task_id for task in env.tasks if task.completed_by is not None
    ]
    expired_ids = [task.task_id for task in env.tasks if task.expired]
    active_ids = [
        task.task_id
        for task in env.tasks
        if task.available and task.release_step <= env.step_count <= task.deadline_step
    ]
    satellites = [
        {
            "id": sat.sat_id,
            "plane": sat.plane_id,
            "phase": round(float(sat.phase), 6),
            "latitude": round(float(env._sat_geometry[sat.sat_id]["latitude_deg"]), 5),
            "longitude": round(float(env._sat_geometry[sat.sat_id]["longitude_deg"]), 5),
            "altitude_km": round(float(env._sat_geometry[sat.sat_id]["altitude_km"]), 3),
            "energy": round(float(sat.energy), 3),
            "storage": round(float(sat.storage), 3),
            "attitude": round(float(sat.attitude_deg), 3),
            "roll": round(float(sat.roll_deg), 3),
            "pitch": round(float(sat.pitch_deg), 3),
            "yaw": round(float(sat.yaw_deg), 3),
            "angular_rate": [
                round(float(sat.roll_rate_deg_s), 4),
                round(float(sat.pitch_rate_deg_s), 4),
                round(float(sat.yaw_rate_deg_s), 4),
            ],
            "payload_mode": sat.payload_mode,
            "swath_width_km": round(float(sat.swath_width_km), 3),
            "field_of_view_deg": round(float(sat.field_of_view_deg), 3),
            "resolution_m": round(float(sat.resolution_m), 3),
            "packets": len(sat.packets),
            "busy": env.step_count < sat.busy_until_step
            or env.step_count < sat.cooldown_until_step,
            "completed": sat.completed_count,
            "conflicts": sat.conflict_count,
        }
        for sat in env.satellites
    ]
    return {
        "step": env.step_count,
        "summary": summary,
        "step_reward": step_reward,
        "completed_ids": completed_ids,
        "expired_ids": expired_ids,
        "active_ids": active_ids,
        "satellites": satellites,
        "events": events or [],
    }


def collect_events(infos: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    events = []
    for agent, info in infos.items():
        event = info.get("event")
        if event not in {
            "completed_task",
            "cooperative_observation",
            "coordination_failed",
            "lost_conflict",
            "infeasible_claim",
            "downlink",
            "ground_station_conflict",
        }:
            continue
        row = {"agent": agent, "event": event}
        for key in (
            "task_id",
            "winner",
            "reason",
            "task_reward",
            "completion_bonus",
            "observation_start_step",
            "finish_step",
            "total_energy",
            "compressed_data_mb",
            "observers",
            "station_id",
            "station_name",
            "elevation_deg",
            "downlinked",
            "quality",
            "cooperation_mode",
        ):
            if key in info:
                value = info[key]
                if isinstance(value, np.generic):
                    value = value.item()
                row[key] = value
        events.append(row)
    return events[:200]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "runs" / "realistic" / "checkpoints" / "latest.pt")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "realistic" / "eval" / "rollout.json")
    parser.add_argument("--satellites", type=int, default=None)
    parser.add_argument("--tasks", type=int, default=None)
    parser.add_argument("--planes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument("--neighbor-k", type=int, default=None)
    parser.add_argument("--ephemeris-cache", type=Path, default=None)
    parser.add_argument("--task-catalog", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=7001)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    metrics = checkpoint.get("metrics") or {}
    env = SatTaskingEnv(config_from_checkpoint(metrics, args))
    observations, _ = env.reset(seed=args.seed)
    agents, sample_vecs, sample_masks = flatten_all(observations)
    net = QNetwork(
        input_dim=sample_vecs.shape[1],
        action_dim=sample_masks.shape[1],
        hidden_dim=int((metrics.get("args") or {}).get("hidden_dim", 256)),
        hidden_layers=int((metrics.get("args") or {}).get("hidden_layers", 2)),
        activation=str((metrics.get("args") or {}).get("activation", "relu")),
        layer_norm=bool((metrics.get("args") or {}).get("layer_norm", True)),
    ).to(device)
    net.load_state_dict(checkpoint["policy"])
    net.eval()

    frames = [collect_frame(env)]
    total_reward = 0.0
    for _ in range(env.config.max_steps):
        actions = choose_greedy(net, observations, device)
        observations, rewards, terminations, truncations, infos = env.step(actions)
        mean_reward = float(np.mean(list(rewards.values())))
        total_reward += mean_reward
        frames.append(collect_frame(env, mean_reward, collect_events(infos)))
        if all(terminations.values()) or all(truncations.values()):
            break

    final_summary = env.summary()
    payload = {
        "status": "complete",
        "created_at": time.time(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_episode": metrics.get("episode"),
        "seed": args.seed,
        "env_config": asdict(env.config),
        "algorithm": metrics.get("algorithm") or {},
        "training_args": metrics.get("args") or {},
        "final_summary": final_summary,
        "mean_episode_reward": total_reward / max(1, len(frames) - 1),
        "tasks": [
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
        ],
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
    }
    atomic_write_json(args.output, payload)
    summary_path = args.output.with_name("summary.json")
    atomic_write_json(
        summary_path,
        {
            "status": "complete",
            "created_at": payload["created_at"],
            "checkpoint_episode": payload["checkpoint_episode"],
            "seed": args.seed,
            "final_summary": final_summary,
            "mean_episode_reward": payload["mean_episode_reward"],
            "rollout": str(args.output),
        },
    )
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
