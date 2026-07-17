from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def decision_opportunities(action_masks: np.ndarray) -> np.ndarray:
    """Return T x N active-decision indicators.

    Action zero is always the wait action.  A row is controllable when at least
    one additional action is feasible.
    """

    masks = np.asarray(action_masks, dtype=bool)
    if masks.ndim != 3:
        raise ValueError("action_masks must have shape [time, agents, actions]")
    return np.count_nonzero(masks, axis=-1) > 1


def opportunity_balance_weights(
    action_masks: np.ndarray, opportunities: np.ndarray | None = None
) -> np.ndarray:
    """Balance rare choice-set sizes without weighting forced-idle rows.

    Active decisions are stratified into one, two-to-three, and four-or-more
    feasible non-wait actions. Each occupied stratum contributes equal total
    mass and the active weights are normalized to mean one.
    """

    masks = np.asarray(action_masks, dtype=bool)
    active = (
        decision_opportunities(masks)
        if opportunities is None
        else np.asarray(opportunities, dtype=bool)
    )
    feasible_non_wait = np.maximum(0, np.count_nonzero(masks, axis=-1) - 1)
    strata = np.zeros_like(feasible_non_wait, dtype=np.int8)
    strata[(feasible_non_wait >= 2) & (feasible_non_wait <= 3)] = 1
    strata[feasible_non_wait >= 4] = 2
    weights = np.zeros_like(feasible_non_wait, dtype=np.float32)
    occupied = [level for level in range(3) if np.any(active & (strata == level))]
    for level in occupied:
        rows = active & (strata == level)
        weights[rows] = 1.0 / float(np.count_nonzero(rows))
    if occupied:
        weights[active] *= float(np.count_nonzero(active)) / float(len(occupied))
    return weights


def semantic_opportunity_classes(action_masks: np.ndarray) -> np.ndarray:
    """Classify controllable rows by scheduling semantics.

    Class zero is forced idle, class one is downlink-only, class two has one
    feasible observation task, and class three has multiple feasible tasks.
    Keeping downlink-only rows separate prevents abundant communication choices
    from dominating the rarer task-scheduling gradients.
    """

    masks = np.asarray(action_masks, dtype=bool)
    if masks.ndim != 3:
        raise ValueError("action_masks must have shape [time, agents, actions]")
    task_count = np.count_nonzero(masks[..., 2:], axis=-1)
    downlink = masks[..., 1] if masks.shape[-1] > 1 else np.zeros_like(task_count)
    classes = np.zeros_like(task_count, dtype=np.int8)
    classes[downlink & (task_count == 0)] = 1
    classes[task_count == 1] = 2
    classes[task_count >= 2] = 3
    return classes


def semantic_opportunity_balance_weights(
    action_masks: np.ndarray, opportunities: np.ndarray | None = None
) -> np.ndarray:
    """Give equal total mass to occupied semantic decision classes.

    The returned active weights have mean one, so enabling this option does not
    change the effective optimizer learning rate.
    """

    masks = np.asarray(action_masks, dtype=bool)
    active = (
        decision_opportunities(masks)
        if opportunities is None
        else np.asarray(opportunities, dtype=bool)
    )
    classes = semantic_opportunity_classes(masks)
    weights = np.zeros_like(classes, dtype=np.float32)
    occupied = [level for level in (1, 2, 3) if np.any(active & (classes == level))]
    for level in occupied:
        rows = active & (classes == level)
        weights[rows] = 1.0 / float(np.count_nonzero(rows))
    if occupied:
        weights[active] *= float(np.count_nonzero(active)) / float(len(occupied))
    return weights


def compute_opportunity_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    opportunities: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute duration-corrected GAE on each agent's decision epochs.

    Rewards between two consecutive controllable states are accumulated with
    time-step discounting. Bootstrapping uses gamma ** delta_t. Values and
    advantages on forced-idle rows are intentionally zero because those rows do
    not enter the actor or critic update.
    """

    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    active = np.asarray(opportunities, dtype=bool)
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("rewards, values, and dones must share shape [time, agents]")
    if active.shape != rewards.shape:
        raise ValueError("opportunities must share shape [time, agents]")

    time_steps, num_agents = rewards.shape
    advantages = np.zeros_like(rewards, dtype=np.float32)
    returns = np.zeros_like(rewards, dtype=np.float32)
    durations = np.zeros_like(rewards, dtype=np.int32)

    for agent in range(num_agents):
        epochs = np.flatnonzero(active[:, agent])
        next_gae = 0.0
        for position in range(len(epochs) - 1, -1, -1):
            start = int(epochs[position])
            next_epoch = (
                int(epochs[position + 1]) if position + 1 < len(epochs) else time_steps
            )
            stop = next_epoch
            terminal_offsets = np.flatnonzero(dones[start:stop, agent] > 0.5)
            if len(terminal_offsets):
                stop = start + int(terminal_offsets[0]) + 1

            duration = max(1, stop - start)
            durations[start, agent] = duration
            cumulative_reward = 0.0
            discount = 1.0
            for step in range(start, stop):
                cumulative_reward += discount * float(rewards[step, agent])
                discount *= gamma

            terminal = bool(np.any(dones[start:stop, agent] > 0.5))
            has_next_decision = next_epoch < time_steps and not terminal
            next_value = float(values[next_epoch, agent]) if has_next_decision else 0.0
            delta = (
                cumulative_reward
                + (gamma**duration) * next_value * float(has_next_decision)
                - float(values[start, agent])
            )
            continuation = float(has_next_decision)
            next_gae = delta + (gamma**duration) * gae_lambda * continuation * next_gae
            advantages[start, agent] = next_gae
            returns[start, agent] = next_gae + float(values[start, agent])

    return advantages, returns, durations


@dataclass(slots=True)
class OpportunityCurriculum:
    """Feedback controller for the fraction of visibility-aligned tasks."""

    start_fraction: float = 0.85
    end_fraction: float = 0.0
    target_opportunity_rate: float = 0.06
    feedback_gain: float = 1.5
    fraction: float | None = None

    def value(self, episode: int, total_episodes: int) -> float:
        progress = min(1.0, max(0.0, (episode - 1) / max(1, total_episodes - 1)))
        ceiling = self.start_fraction + progress * (
            self.end_fraction - self.start_fraction
        )
        if self.fraction is None:
            self.fraction = ceiling
        self.fraction = float(np.clip(self.fraction, self.end_fraction, ceiling))
        return self.fraction

    def observe(self, opportunity_rate: float, episode: int, total_episodes: int) -> float:
        current = self.value(episode, total_episodes)
        error = self.target_opportunity_rate - float(opportunity_rate)
        self.fraction = current + self.feedback_gain * error
        return self.value(min(total_episodes, episode + 1), total_episodes)
