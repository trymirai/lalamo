from functools import partial

import jax
import jax.numpy as jnp
from jax.lax import DotAlgorithmPreset
from jaxtyping import Array, Float

from lalamo.module import ShardingAxis
from lalamo.utils.precision import use_dot_algorithm_preset
from lalamo.weight_matrix import CompressionImplementation, Preconditioner, QuantizedSpec

from .ldlq import block_ldl

__all__ = [
    "yaqa_round_weights",
]


def yaqa_round_weights(
    weights: Float[Array, "*batch output_channels input_channels"],
    preconditioner: Preconditioner,
    spec: QuantizedSpec,
    *,
    max_iters: int = 15,
    atol: float = 1e-7,
) -> Float[Array, "*batch output_channels input_channels"]:
    if weights.ndim > 2:
        return jax.vmap(
            partial(
                yaqa_round_weights,
                spec=spec,
                max_iters=max_iters,
                atol=atol,
            ),
            in_axes=(0, 0),
            spmd_axis_name=ShardingAxis.EXPERT,
        )(weights, preconditioner)

    weights = weights.astype(jnp.float32)
    output_dim, input_dim = weights.shape

    input_block = preconditioner.input_block
    output_block = preconditioner.output_block
    if input_block is None and output_block is None:
        return spec.compress(weights, implementation=CompressionImplementation.TRAINING).decompress()

    if input_block is None:
        input_fisher_tril = None
    else:
        input_fisher_tril = block_ldl(input_block, spec.input_block_size).tril
        input_fisher_tril = input_fisher_tril - jnp.identity(input_dim, dtype=input_fisher_tril.dtype)

    if output_block is None:
        output_fisher_triu = None
    else:
        output_fisher_triu = block_ldl(output_block, spec.output_block_size).tril.T
        output_fisher_triu = output_fisher_triu - jnp.identity(output_dim, dtype=output_fisher_triu.dtype)

    def round_weights(
        candidate_weights: Float[Array, "output_channels input_channels"],
    ) -> Float[Array, "output_channels input_channels"]:
        return spec.compress(candidate_weights, implementation=CompressionImplementation.TRAINING).decompress()

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
        new_weights = round_weights(weights + adjustment)
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
        init_val=(round_weights(weights), jnp.inf),
    )
    return result
