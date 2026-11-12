from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

__all__ = [
    "AdvantageBaseline",
    "distillation_advantages",
    "masked_mean",
    "on_policy_distillation_loss",
    "policy_gradient_loss",
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
