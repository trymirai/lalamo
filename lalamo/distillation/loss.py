from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Bool, Float

__all__ = [
    "AdvantageBaseline",
    "distillation_advantages",
    "group_normalized_rewards",
    "masked_mean",
    "on_policy_distillation_loss",
    "policy_gradient_loss",
    "reward_advantages",
]


class AdvantageBaseline(StrEnum):
    NONE = "none"
    BATCH_MEAN = "batch_mean"


def masked_mean(
    values: Float[Array, "batch predictions"],
    mask: Bool[Array, "batch predictions"],
) -> Float[Array, ""]:
    num_selected = jnp.sum(mask)
    num_selected = eqx.error_if(num_selected, num_selected == 0, "The mask must select at least one prediction.")
    return jnp.sum(jnp.where(mask, values, 0.0)) / num_selected.astype(values.dtype)


def distillation_advantages(
    student_log_probs: Float[Array, "batch predictions"],
    teacher_log_probs: Float[Array, "batch predictions"],
    response_mask: Bool[Array, "batch predictions"],
    baseline: AdvantageBaseline,
) -> Float[Array, "batch predictions"]:
    """Advantage of each sampled token whose policy gradient descends the reverse KL from student to teacher.

    Both inputs hold the log-probability of the tokens the student sampled. A baseline that does not depend
    on the sampled token leaves the expected gradient unchanged and only lowers its variance.
    """
    log_prob_gaps = teacher_log_probs - student_log_probs
    match baseline:
        case AdvantageBaseline.NONE:
            baseline_value = jnp.zeros((), dtype=log_prob_gaps.dtype)
        case AdvantageBaseline.BATCH_MEAN:
            baseline_value = masked_mean(log_prob_gaps, response_mask)
    return jnp.where(response_mask, log_prob_gaps - baseline_value, 0.0)


def group_normalized_rewards(
    rewards: Float[Array, " rollouts"],
    group_size: int,
) -> Float[Array, " rollouts"]:
    """Standardizes rewards within consecutive groups of rollouts that answer the same prompt.

    A group whose rollouts all earned the same reward does not say which rollout was better, so it gets zeros.
    """
    grouped = rearrange(rewards, "(groups group_size) -> groups group_size", group_size=group_size)
    centered = grouped - jnp.mean(grouped, axis=1, keepdims=True)
    spread = jnp.std(grouped, axis=1, keepdims=True)
    is_informative = spread > 0
    normalized = jnp.where(is_informative, centered / jnp.where(is_informative, spread, 1.0), 0.0)
    return rearrange(normalized, "groups group_size -> (groups group_size)")


def reward_advantages(
    rewards: Float[Array, " batch"],
    response_mask: Bool[Array, "batch predictions"],
    group_size: int,
) -> Float[Array, "batch predictions"]:
    """Gives every response token of a rollout the group-normalized reward of the whole rollout."""
    rollout_advantages = rearrange(group_normalized_rewards(rewards, group_size), "batch -> batch 1")
    return jnp.where(response_mask, rollout_advantages, 0.0)


def policy_gradient_loss(
    student_log_probs: Float[Array, "batch predictions"],
    advantages: Float[Array, "batch predictions"],
    response_mask: Bool[Array, "batch predictions"],
) -> Float[Array, ""]:
    """Surrogate whose gradient raises the log-probability of sampled tokens in proportion to their advantage."""
    return -masked_mean(jax.lax.stop_gradient(advantages) * student_log_probs, response_mask)


def on_policy_distillation_loss(
    student_log_probs: Float[Array, "batch predictions"],
    teacher_log_probs: Float[Array, "batch predictions"],
    response_mask: Bool[Array, "batch predictions"],
    *,
    baseline: AdvantageBaseline,
) -> Float[Array, ""]:
    advantages = distillation_advantages(student_log_probs, teacher_log_probs, response_mask, baseline)
    return policy_gradient_loss(student_log_probs, advantages, response_mask)
