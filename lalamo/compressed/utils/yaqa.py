from functools import partial

import jax
import jax.numpy as jnp
from jax.lax import DotAlgorithmPreset
from jaxtyping import Array, Bool, Float, Int

from lalamo.compressed.quantized_spec import QuantizedSpec
from lalamo.module import ShardingAxis
from lalamo.preconditioner import Preconditioner
from lalamo.utils.precision import use_dot_algorithm_preset
from lalamo.weight_matrix import CompressionImplementation

from .block_ldl import block_ldl

__all__ = [
    "ConvergenceError",
    "yaqa_round_weights",
]


class ConvergenceError(RuntimeError):
    pass


def yaqa_round_weights(
    weights: Float[Array, "*components output_channels input_channels"],
    preconditioner: Preconditioner,
    spec: QuantizedSpec,
    *,
    is_sharded: bool = True,
    max_iters: int = 2048,
    atol: float = 1e-7,
) -> Float[Array, "*components output_channels input_channels"]:
    result, has_converged = _yaqa_round_weights(
        weights,
        preconditioner,
        spec,
        is_sharded=is_sharded,
        max_iters=max_iters,
        atol=atol,
    )
    if not bool(jax.device_get(jnp.all(has_converged))):
        raise ConvergenceError(f"YAQA failed to converge in {max_iters} iterations.")

    return result


def _yaqa_round_weights(
    weights: Float[Array, "*components output_channels input_channels"],
    preconditioner: Preconditioner,
    spec: QuantizedSpec,
    *,
    is_sharded: bool,
    max_iters: int,
    atol: float,
) -> tuple[Float[Array, "*components output_channels input_channels"], Bool[Array, "*components"]]:
    if weights.ndim > 2:
        return jax.vmap(
            partial(
                _yaqa_round_weights,
                spec=spec,
                is_sharded=is_sharded,
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
        return spec.compress(
            weights,
            implementation=CompressionImplementation.TRAINING,
            is_sharded=is_sharded,
        ).decompress(), jnp.ones((), dtype=jnp.bool_)

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
        return spec.compress(
            candidate_weights,
            implementation=CompressionImplementation.TRAINING,
            is_sharded=is_sharded,
        ).decompress()

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

    def should_continue(
        state: tuple[Int[Array, ""], Float[Array, "output_channels input_channels"], Float[Array, ""]],
    ) -> Bool[Array, ""]:
        iteration, _last_weights, diff = state
        return (iteration < max_iters) & (diff >= atol)

    def loop_body(
        state: tuple[Int[Array, ""], Float[Array, "output_channels input_channels"], Float[Array, ""]],
    ) -> tuple[Int[Array, ""], Float[Array, "output_channels input_channels"], Float[Array, ""]]:
        iteration, last_weights, _diff = state
        adjustment = calculate_adjustment(last_weights)
        new_weights = round_weights(weights + adjustment)
        new_diff = jnp.max(jnp.abs(new_weights - last_weights))
        return iteration + 1, new_weights, new_diff

    _iteration, result, diff = jax.lax.while_loop(
        should_continue,
        loop_body,
        (jnp.array(0), round_weights(weights), jnp.inf),
    )
    return result, diff < atol
