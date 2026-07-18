from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was explicitly requested, but this PyTorch build cannot access it."
        )
    device = torch.device(
        "cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu"
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    return device


def scenario_seed(base_seed: int, episode: int, seed_cycle: int = 0) -> int:
    if seed_cycle <= 0:
        return base_seed + episode
    return base_seed + ((episode - 1) % seed_cycle) + 1


def make_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    hidden_layers: int,
    activation: str = "relu",
    layer_norm: bool = True,
) -> nn.Sequential:
    activation_class = {"relu": nn.ReLU, "silu": nn.SiLU, "tanh": nn.Tanh}[
        activation
    ]
    layers: list[nn.Module] = []
    current_dim = input_dim
    for layer_index in range(max(1, hidden_layers)):
        layers.append(nn.Linear(current_dim, hidden_dim))
        if layer_norm and layer_index == 0:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(activation_class())
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(
        self,
        actor_dim: int,
        critic_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
        activation: str = "relu",
        layer_norm: bool = True,
    ):
        super().__init__()
        self.actor = make_mlp(
            actor_dim,
            action_dim,
            hidden_dim,
            hidden_layers,
            activation,
            layer_norm,
        )
        self.critic = make_mlp(
            critic_dim,
            1,
            hidden_dim,
            hidden_layers,
            activation,
            layer_norm,
        )

    def policy_logits(self, actor_obs: torch.Tensor) -> torch.Tensor:
        return self.actor(actor_obs)

    def values(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_obs).squeeze(-1)


class OpportunityGraphActorCritic(nn.Module):
    """Permutation-equivariant task scorer over a sparse opportunity graph.

    Each feasible satellite--task edge receives a shared candidate encoding and
    a four-dimensional resource-factor message.  The number of parameters and
    actor input width are independent of the constellation size; only the fixed
    candidate budget K enters the network shape.
    """

    def __init__(
        self,
        critic_dim: int,
        candidate_k: int,
        neighbor_k: int,
        hidden_dim: int = 256,
        activation: str = "relu",
        layer_norm: bool = True,
        self_dim: int = 24,
        task_dim: int = 24,
        neighbor_dim: int = 13,
        neighbor_action_dim: int = 4,
        factor_dim: int = 4,
    ):
        super().__init__()
        self.candidate_k = candidate_k
        self.task_dim = task_dim
        self.factor_dim = factor_dim
        self.context_dim = self_dim + neighbor_k * neighbor_dim + neighbor_action_dim
        self.actor_dim = self.context_dim + candidate_k * (task_dim + factor_dim)
        self.context_encoder = make_mlp(
            self.context_dim, hidden_dim, hidden_dim, 1, activation, layer_norm
        )
        self.candidate_encoder = make_mlp(
            task_dim + factor_dim, hidden_dim, hidden_dim, 1, activation, layer_norm
        )
        activation_class = {"relu": nn.ReLU, "silu": nn.SiLU, "tanh": nn.Tanh}[
            activation
        ]
        self.resource_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            activation_class(),
            nn.Linear(hidden_dim, 1),
        )
        self.wait_downlink_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation_class(),
            nn.Linear(hidden_dim, 2),
        )
        self.critic = make_mlp(
            critic_dim, 1, hidden_dim, 2, activation, layer_norm
        )

    def policy_logits(self, actor_obs: torch.Tensor) -> torch.Tensor:
        if actor_obs.shape[-1] != self.actor_dim:
            raise ValueError(
                f"Expected graph actor width {self.actor_dim}, got {actor_obs.shape[-1]}"
            )
        context = actor_obs[..., : self.context_dim]
        candidate_stop = self.context_dim + self.candidate_k * self.task_dim
        candidates = actor_obs[..., self.context_dim : candidate_stop].reshape(
            -1, self.candidate_k, self.task_dim
        )
        factors = actor_obs[..., candidate_stop:].reshape(
            -1, self.candidate_k, self.factor_dim
        )
        context_hidden = self.context_encoder(context)
        candidate_hidden = self.candidate_encoder(
            torch.cat([candidates, factors], dim=-1)
        )
        repeated_context = context_hidden.unsqueeze(1).expand(
            -1, self.candidate_k, -1
        )
        task_logits = self.resource_head(
            torch.cat([repeated_context, candidate_hidden], dim=-1)
        ).squeeze(-1)
        return torch.cat([self.wait_downlink_head(context_hidden), task_logits], dim=-1)

    def values(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_obs).squeeze(-1)


class FastSlowOpportunityGraphActorCritic(nn.Module):
    """Bi-timescale opportunity-graph actor with a persistent strategic intent.

    The slow encoder consumes a constellation-local snapshot that is refreshed
    every ``slow_interval`` environment steps.  Its latent intent conditions a
    fast, permutation-equivariant satellite--task scorer at every controllable
    opportunity.  Parameters remain independent of constellation size.
    """

    def __init__(
        self,
        critic_dim: int,
        candidate_k: int,
        neighbor_k: int,
        hidden_dim: int = 256,
        intent_dim: int = 64,
        activation: str = "relu",
        layer_norm: bool = True,
        self_dim: int = 24,
        task_dim: int = 24,
        neighbor_dim: int = 13,
        neighbor_action_dim: int = 4,
        factor_dim: int = 4,
    ):
        super().__init__()
        self.candidate_k = candidate_k
        self.task_dim = task_dim
        self.factor_dim = factor_dim
        self.context_dim = self_dim + neighbor_k * neighbor_dim + neighbor_action_dim
        self.graph_actor_dim = self.context_dim + candidate_k * (task_dim + factor_dim)
        # Slow features contain context plus mean/max pooled task-edge features.
        self.slow_feature_dim = self.context_dim + 2 * (task_dim + factor_dim)
        self.actor_dim = self.graph_actor_dim + self.slow_feature_dim
        self.context_encoder = make_mlp(
            self.context_dim, hidden_dim, hidden_dim, 1, activation, layer_norm
        )
        self.candidate_encoder = make_mlp(
            task_dim + factor_dim, hidden_dim, hidden_dim, 1, activation, layer_norm
        )
        self.slow_encoder = make_mlp(
            self.slow_feature_dim, intent_dim, hidden_dim, 2, activation, layer_norm
        )
        activation_class = {"relu": nn.ReLU, "silu": nn.SiLU, "tanh": nn.Tanh}[
            activation
        ]
        self.resource_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + intent_dim, hidden_dim),
            activation_class(),
            nn.Linear(hidden_dim, 1),
        )
        self.wait_downlink_head = nn.Sequential(
            nn.Linear(hidden_dim + intent_dim, hidden_dim),
            activation_class(),
            nn.Linear(hidden_dim, 2),
        )
        self.critic = make_mlp(
            critic_dim, 1, hidden_dim, 2, activation, layer_norm
        )

    def policy_logits(self, actor_obs: torch.Tensor) -> torch.Tensor:
        if actor_obs.shape[-1] != self.actor_dim:
            raise ValueError(
                f"Expected fast-slow actor width {self.actor_dim}, got {actor_obs.shape[-1]}"
            )
        graph_obs = actor_obs[..., : self.graph_actor_dim]
        slow_features = actor_obs[..., self.graph_actor_dim :]
        context = graph_obs[..., : self.context_dim]
        candidate_stop = self.context_dim + self.candidate_k * self.task_dim
        candidates = graph_obs[..., self.context_dim : candidate_stop].reshape(
            -1, self.candidate_k, self.task_dim
        )
        factors = graph_obs[..., candidate_stop:].reshape(
            -1, self.candidate_k, self.factor_dim
        )
        context_hidden = self.context_encoder(context)
        candidate_hidden = self.candidate_encoder(
            torch.cat([candidates, factors], dim=-1)
        )
        intent = self.slow_encoder(slow_features)
        repeated_context = context_hidden.unsqueeze(1).expand(
            -1, self.candidate_k, -1
        )
        repeated_intent = intent.unsqueeze(1).expand(-1, self.candidate_k, -1)
        task_logits = self.resource_head(
            torch.cat([repeated_context, candidate_hidden, repeated_intent], dim=-1)
        ).squeeze(-1)
        wait_downlink = self.wait_downlink_head(
            torch.cat([context_hidden, intent], dim=-1)
        )
        return torch.cat([wait_downlink, task_logits], dim=-1)

    def values(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_obs).squeeze(-1)


def slow_strategy_features(
    graph_actor_obs: np.ndarray,
    candidate_k: int,
    neighbor_k: int,
    self_dim: int = 24,
    task_dim: int = 24,
    neighbor_dim: int = 13,
    neighbor_action_dim: int = 4,
    factor_dim: int = 4,
) -> np.ndarray:
    """Compress a graph snapshot into fixed-width slow strategic features."""

    observations = np.asarray(graph_actor_obs, dtype=np.float32)
    context_dim = self_dim + neighbor_k * neighbor_dim + neighbor_action_dim
    candidate_stop = context_dim + candidate_k * task_dim
    expected = candidate_stop + candidate_k * factor_dim
    if observations.ndim != 2 or observations.shape[1] != expected:
        raise ValueError(
            f"Expected graph observations [agents, {expected}], got {observations.shape}"
        )
    context = observations[:, :context_dim]
    candidates = observations[:, context_dim:candidate_stop].reshape(
        -1, candidate_k, task_dim
    )
    factors = observations[:, candidate_stop:].reshape(-1, candidate_k, factor_dim)
    edges = np.concatenate([candidates, factors], axis=-1)
    pooled_mean = np.mean(edges, axis=1)
    pooled_max = np.max(edges, axis=1)
    return np.concatenate([context, pooled_mean, pooled_max], axis=1).astype(
        np.float32, copy=False
    )


class AgentQNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
        activation: str = "relu",
        layer_norm: bool = True,
    ):
        super().__init__()
        self.net = make_mlp(
            input_dim,
            action_dim,
            hidden_dim,
            hidden_layers,
            activation,
            layer_norm,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class QMixer(nn.Module):
    """Monotonic QMIX mixer conditioned on the aggregate environment state."""

    def __init__(self, num_agents: int, state_dim: int, embed_dim: int = 64):
        super().__init__()
        self.num_agents = num_agents
        self.embed_dim = embed_dim
        self.hyper_w1 = nn.Linear(state_dim, num_agents * embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_w_final = nn.Linear(state_dim, embed_dim)
        self.state_value = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1)
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        batch_size = agent_qs.shape[0]
        weights1 = torch.abs(self.hyper_w1(states)).view(
            batch_size, self.num_agents, self.embed_dim
        )
        biases1 = self.hyper_b1(states).view(batch_size, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), weights1) + biases1)
        final_weights = torch.abs(self.hyper_w_final(states)).view(
            batch_size, self.embed_dim, 1
        )
        value = self.state_value(states).view(batch_size, 1, 1)
        return (torch.bmm(hidden, final_weights) + value).view(batch_size)


def flatten_local_obs(agent_obs: dict[str, np.ndarray]) -> np.ndarray:
    """Observation available during decentralized execution."""

    return np.concatenate(
        [
            agent_obs["self"].ravel(),
            agent_obs["neighbors"].ravel(),
            agent_obs["neighbor_action_mean"].ravel(),
            agent_obs["candidates"].ravel(),
        ]
    ).astype(np.float32, copy=False)


def flatten_local_all(observations: dict[str, dict[str, np.ndarray]]):
    agents = list(observations)
    local = np.stack([flatten_local_obs(observations[agent]) for agent in agents])
    global_state = np.asarray(observations[agents[0]]["global"], dtype=np.float32)
    masks = np.stack(
        [observations[agent]["action_mask"] for agent in agents]
    ).astype(bool)
    return agents, local, global_state, masks


def opportunity_graph_features(
    observations: dict[str, dict[str, np.ndarray]],
) -> np.ndarray:
    """Build size-normalized messages on feasible satellite--task edges.

    The four edge features are log contender density, mean contender utility,
    the focal edge's relative utility, and observer-supply coverage.  Task IDs
    are used only to construct factors and never enter the learned network.
    """

    agents = list(observations)
    if not agents:
        return np.empty((0, 0), dtype=np.float32)
    candidate_k = int(observations[agents[0]]["candidates"].shape[0])
    factors = np.zeros((len(agents), candidate_k, 4), dtype=np.float32)
    groups: dict[int, list[tuple[int, int, float, float]]] = {}
    for agent_row, agent in enumerate(agents):
        obs = observations[agent]
        candidate_ids = np.asarray(obs["candidate_ids"], dtype=np.int64)
        candidates = np.asarray(obs["candidates"], dtype=np.float32)
        mask = np.asarray(obs["action_mask"], dtype=bool)
        for slot in range(candidate_k):
            task_id = int(candidate_ids[slot])
            if task_id < 0 or slot + 2 >= len(mask) or not bool(mask[slot + 2]):
                continue
            feature = candidates[slot]
            utility = float(
                feature[0] * feature[19]
                - 0.10 * feature[22]
                - 0.05 * feature[21]
            )
            required_observers = max(1.0, round(float(feature[14]) * 2.0))
            groups.setdefault(task_id, []).append(
                (agent_row, slot, utility, required_observers)
            )
    count_scale = max(1.0, float(np.log1p(max(2, len(agents)))))
    for entries in groups.values():
        count = len(entries)
        mean_utility = float(np.mean([entry[2] for entry in entries]))
        density = float(np.log1p(count) / count_scale)
        for agent_row, slot, utility, required in entries:
            factors[agent_row, slot] = np.asarray(
                [
                    density,
                    mean_utility,
                    utility - mean_utility,
                    min(1.0, count / required),
                ],
                dtype=np.float32,
            )
    return factors.reshape(len(agents), -1)


def flatten_opportunity_graph_all(
    observations: dict[str, dict[str, np.ndarray]],
):
    agents, local, global_state, masks = flatten_local_all(observations)
    factors = opportunity_graph_features(observations)
    actor_obs = np.concatenate([local, factors], axis=1).astype(np.float32, copy=False)
    return agents, actor_obs, global_state, masks


def critic_observations(
    local_observations: np.ndarray,
    global_state: np.ndarray,
    centralized: bool,
) -> np.ndarray:
    if not centralized:
        return local_observations
    repeated_global = np.repeat(global_state[None, :], len(local_observations), axis=0)
    return np.concatenate([local_observations, repeated_global], axis=1).astype(
        np.float32, copy=False
    )


def masked_logits(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~masks, -1e9)


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(20):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05 * (attempt + 1))


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def serialize_args(args: Any) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def parameter_count(*modules: nn.Module) -> int:
    return sum(
        parameter.numel()
        for module in modules
        for parameter in module.parameters()
    )
