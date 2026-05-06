from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.lax import DotAlgorithmPreset
from jaxtyping import Array, Float

from lalamo.utils.precision import use_dot_algorithm_preset
from lalamo.weight_matrix import Preconditioner

from .ldlq import block_ldl

__all__ = [
    "RoundFn",
    "yaqa_round_weights",
]

type RoundFn = Callable[
    [Float[Array, "output_channels input_channels"]],
    Float[Array, "output_channels input_channels"],
]


def yaqa_round_weights(
    weights: Float[Array, "output_channels input_channels"],
    preconditioner: Preconditioner,
    round_fn: RoundFn,
    *,
    input_block_size: int,
    output_block_size: int,
    max_iters: int = 15,
    atol: float = 1e-7,
) -> Float[Array, "output_channels input_channels"]:
    weights = weights.astype(jnp.float32)
    output_dim, input_dim = weights.shape

    input_block = preconditioner.input_block
    output_block = preconditioner.output_block
    if input_block is None and output_block is None:
        return round_fn(weights)

    if input_block is None:
        input_fisher_tril = None
    else:
        input_fisher_tril = block_ldl(input_block, input_block_size).tril
        input_fisher_tril = input_fisher_tril - jnp.identity(input_dim, dtype=input_fisher_tril.dtype)

    if output_block is None:
        output_fisher_triu = None
    else:
        output_fisher_triu = block_ldl(output_block, output_block_size).tril.T
        output_fisher_triu = output_fisher_triu - jnp.identity(output_dim, dtype=output_fisher_triu.dtype)

    def calculate_adjustment(
        quantized_weights: Float[Array, "output_channels input_channels"],
    ) -> Float[Array, "output_channels input_channels"]:
        residual = weights - quantized_weights
        adjustment = jnp.zeros_like(residual)

        with use_dot_algorithm_preset(DotAlgorithmPreset.F32_F32_F32):
            if input_fisher_tril is not None:
                adjustment += residual @ input_fisher_tril
            if output_fisher_triu is not None:
                adjustment += output_fisher_triu @ residual
            if input_fisher_tril is not None and output_fisher_triu is not None:
                adjustment += output_fisher_triu @ residual @ input_fisher_tril

        return adjustment

    def loop_body(
        state: tuple[Float[Array, "output_channels input_channels"], Float[Array, ""]],
    ) -> tuple[Float[Array, "output_channels input_channels"], Float[Array, ""]]:
        last_weights, _ = state
        adjustment = calculate_adjustment(last_weights)
        new_weights = round_fn(weights + adjustment)
        new_diff = jnp.max(jnp.abs(new_weights - last_weights))
        return new_weights, new_diff

    def loop_body_with_early_exit(
        iteration: int,
        state: tuple[Float[Array, "output_channels input_channels"], Float[Array, ""]],
    ) -> tuple[Float[Array, "output_channels input_channels"], Float[Array, ""]]:
        del iteration
        diff = state[1]
        return jax.lax.cond(diff < atol, lambda state: state, loop_body, state)

    result, _ = jax.lax.fori_loop(
        lower=0,
        upper=max_iters,
        body_fun=loop_body_with_early_exit,
        init_val=(round_fn(weights), jnp.inf),
    )
    return result
