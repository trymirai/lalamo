from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Int

from lalamo.module import Keychain
from lalamo.modules import Decoder, DecoderForwardPassConfig

from .adapters import partition_adapters
from .loss import policy_gradient_loss
from .rollouts import Rollouts, sampled_token_log_probs

__all__ = [
    "add_gradients",
    "apply_gradients",
    "init_optimizer_state",
    "policy_gradients",
]


@eqx.filter_jit
def _policy_gradients(
    student: Decoder,
    rollouts: Rollouts,
    advantages: Float[Array, "batch predictions"],
    num_step_response_tokens: Int[Array, ""],
    forward_pass_config: DecoderForwardPassConfig,
    *,
    keychain: Keychain,
) -> Decoder:
    adapters, frozen = partition_adapters(student)
    response_mask = rollouts.response_mask
    share_of_step = jnp.sum(response_mask).astype(jnp.float32) / num_step_response_tokens.astype(jnp.float32)

    def micro_batch_loss(adapters: Decoder) -> Float[Array, ""]:
        student_log_probs = sampled_token_log_probs(
            eqx.combine(adapters, frozen),
            rollouts,
            forward_pass_config,
            keychain=keychain,
        )
        return share_of_step * policy_gradient_loss(student_log_probs, advantages, response_mask)

    return eqx.filter_grad(micro_batch_loss)(adapters)


def policy_gradients(
    student: Decoder,
    rollouts: Rollouts,
    advantages: Float[Array, "batch predictions"],
    num_step_response_tokens: Int[Array, ""],
    forward_pass_config: DecoderForwardPassConfig,
    *,
    keychain: Keychain,
) -> Decoder:
    """Gradient of the policy-gradient loss with respect to the adapters, for one micro-batch of an optimizer step.

    The loss is weighted by the micro-batch's share of the step's response tokens, so the gradients of all
    micro-batches of a step add up to the gradient of the loss over the whole step.
    """
    # The backward pass of the attention kernel multiplies float32 cotangents with lower-precision activations.
    # Strict dtype promotion rejects that when the product is lowered, so the compilation as a whole is relaxed.
    with jax.numpy_dtype_promotion("standard"):
        return _policy_gradients(
            student,
            rollouts,
            advantages,
            num_step_response_tokens,
            forward_pass_config,
            keychain=keychain,
        )


@eqx.filter_jit
def add_gradients(first: Decoder, second: Decoder) -> Decoder:
    return jax.tree.map(jnp.add, first, second)


def init_optimizer_state(student: Decoder, optimizer: optax.GradientTransformation) -> optax.OptState:
    adapters, _ = partition_adapters(student)
    return optimizer.init(cast("optax.Params", adapters))


@eqx.filter_jit
def _update_adapters(
    adapters: Decoder,
    optimizer_state: optax.OptState,
    gradients: Decoder,
    optimizer: optax.GradientTransformation,
) -> tuple[Decoder, optax.OptState]:
    # Optax mixes its integer step counter with float hyperparameters, which strict dtype promotion rejects.
    with jax.numpy_dtype_promotion("standard"):
        updates, optimizer_state = optimizer.update(
            cast("optax.Updates", gradients),
            optimizer_state,
            cast("optax.Params", adapters),
        )
    return cast("Decoder", optax.apply_updates(cast("optax.Params", adapters), updates)), optimizer_state


def apply_gradients(
    student: Decoder,
    optimizer_state: optax.OptState,
    gradients: Decoder,
    optimizer: optax.GradientTransformation,
) -> tuple[Decoder, optax.OptState]:
    # Only the adapters pass through the compiled update: a compiled function returns fresh arrays, so passing
    # the frozen weights through it would copy the whole model on every step.
    adapters, frozen = partition_adapters(student)
    updated_adapters, optimizer_state = _update_adapters(adapters, optimizer_state, gradients, optimizer)
    return eqx.combine(updated_adapters, frozen), optimizer_state
