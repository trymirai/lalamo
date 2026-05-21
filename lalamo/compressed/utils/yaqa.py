from functools import partial

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.lax import DotAlgorithmPreset
from jaxtyping import Array, Bool, Float, Int

from lalamo.compressed.quantized_spec import QuantizedSpec
from lalamo.preconditioner import Preconditioner
from lalamo.utils.precision import use_dot_algorithm_preset
from lalamo.utils.sharding import LogicalAxis, ShardingConfig, sharding_of
from lalamo.weight_matrix import CompressionImplementation

from .block_ldl import block_ldl

__all__ = [
    "ConvergenceError",
    "yaqa_round_blockwise",
    "yaqa_round_fixpoint",
]


class ConvergenceError(RuntimeError):
    pass


def yaqa_round_fixpoint(
    weights: Float[Array, "*components output_channels input_channels"],
    preconditioner: Preconditioner,
    spec: QuantizedSpec,
    *,
    sharding_config: ShardingConfig,
    max_iters: int | None = None,
    atol: float = 1e-7,
) -> Float[Array, "*components output_channels input_channels"]:
    input_sharding = sharding_of(weights)
    input_dtype = weights.dtype
    *_, output_dim, input_dim = weights.shape
    if max_iters is None:
        max_iters = 2 * (input_dim + output_dim)

    replicated_weight_sharding = sharding_config.make_sharding((None,) * weights.ndim)
    replicated_weights = jax.device_put(
        weights.astype(jnp.float32),
        replicated_weight_sharding,
    )

    def gather_preconditioner_leaf(leaf: Array) -> Array:
        return jax.device_put(
            leaf,
            sharding_config.make_sharding((None,) * leaf.ndim),
        )

    replicated_preconditioner = jtu.tree_map(gather_preconditioner_leaf, preconditioner)
    result, has_converged = _yaqa_round_fixpoint(
        replicated_weights,
        replicated_preconditioner,
        spec,
        sharding_config=sharding_config,
        max_iters=max_iters,
        atol=atol,
    )
    if not bool(jax.device_get(jnp.all(has_converged))):
        raise ConvergenceError(f"YAQA failed to converge in {max_iters} iterations.")

    return jax.device_put(result.astype(input_dtype), input_sharding)


def _yaqa_round_fixpoint(
    weights: Float[Array, "*components output_channels input_channels"],
    preconditioner: Preconditioner,
    spec: QuantizedSpec,
    *,
    sharding_config: ShardingConfig,
    max_iters: int,
    atol: float,
) -> tuple[Float[Array, "*components output_channels input_channels"], Bool[Array, "*components"]]:
    if weights.ndim > 2:
        spmd_axis_name = sharding_config.resolve_axis(LogicalAxis.MIXTURE)
        return jax.vmap(
            partial(
                _yaqa_round_fixpoint,
                spec=spec,
                sharding_config=sharding_config,
                max_iters=max_iters,
                atol=atol,
            ),
            in_axes=(0, 0),
            spmd_axis_name=spmd_axis_name,
        )(weights, preconditioner)

    output_dim, input_dim = weights.shape

    input_block = preconditioner.input_block
    output_block = preconditioner.output_block
    if input_block is None and output_block is None:
        rounded_weights = spec.compress(
            weights,
            implementation=CompressionImplementation.INFERENCE,
            sharding_config=sharding_config,
            is_sharded=False,
        ).decompress()
        return rounded_weights, jnp.ones((), dtype=jnp.bool_)

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
            implementation=CompressionImplementation.INFERENCE,
            sharding_config=sharding_config,
            is_sharded=False,
        ).decompress()

    def calculate_adjustment(
        quantized_weights: Float[Array, "output_channels input_channels"],
    ) -> Float[Array, "output_channels input_channels"]:
        residual = weights - quantized_weights
        adjustment = residual * jnp.array(0, dtype=residual.dtype)

        with use_dot_algorithm_preset(DotAlgorithmPreset.F32_F32_F32):
            if input_fisher_tril is not None:
                adjustment += residual @ input_fisher_tril
            if output_fisher_triu is not None:
                adjustment += output_fisher_triu @ residual
            if input_fisher_tril is not None and output_fisher_triu is not None:
                adjustment += output_fisher_triu @ residual @ input_fisher_tril

        return adjustment

    def objective(
        quantized_weights: Float[Array, "output_channels input_channels"],
    ) -> Float[Array, ""]:
        residual = weights - quantized_weights
        weighted_residual = residual

        with use_dot_algorithm_preset(DotAlgorithmPreset.F32_F32_F32):
            if input_block is not None:
                weighted_residual = weighted_residual @ input_block
            if output_block is not None:
                weighted_residual = output_block @ weighted_residual

        return jnp.sum(weighted_residual * residual)

    def should_continue(
        state: tuple[
            Int[Array, ""],
            Float[Array, "output_channels input_channels"],
            Float[Array, "output_channels input_channels"],
            Float[Array, "output_channels input_channels"],
            Float[Array, ""],
            Float[Array, ""],
            Float[Array, ""],
        ],
    ) -> Bool[Array, ""]:
        iteration, _previous_weights, _last_weights, _best_weights, _best_objective, diff, cycle_diff = state
        return (iteration < max_iters) & (diff >= atol) & (cycle_diff >= atol)

    def loop_body(
        state: tuple[
            Int[Array, ""],
            Float[Array, "output_channels input_channels"],
            Float[Array, "output_channels input_channels"],
            Float[Array, "output_channels input_channels"],
            Float[Array, ""],
            Float[Array, ""],
            Float[Array, ""],
        ],
    ) -> tuple[
        Int[Array, ""],
        Float[Array, "output_channels input_channels"],
        Float[Array, "output_channels input_channels"],
        Float[Array, "output_channels input_channels"],
        Float[Array, ""],
        Float[Array, ""],
        Float[Array, ""],
    ]:
        iteration, previous_weights, last_weights, best_weights, best_objective, _diff, _cycle_diff = state
        adjustment = calculate_adjustment(last_weights)
        new_weights = round_weights(weights + adjustment)
        new_objective = objective(new_weights)
        new_diff = jnp.max(jnp.abs(new_weights - last_weights))
        new_cycle_diff = jnp.max(jnp.abs(new_weights - previous_weights))
        is_best = new_objective < best_objective
        return (
            iteration + 1,
            last_weights,
            new_weights,
            jnp.where(is_best, new_weights, best_weights),
            jnp.minimum(new_objective, best_objective),
            new_diff,
            new_cycle_diff,
        )

    initial_weights = round_weights(weights)
    _iteration, _previous_weights, result, best_weights, _best_objective, diff, cycle_diff = jax.lax.while_loop(
        should_continue,
        loop_body,
        (
            jnp.array(0),
            initial_weights,
            initial_weights,
            initial_weights,
            objective(initial_weights),
            jnp.inf,
            jnp.inf,
        ),
    )
    return jnp.where(diff < atol, result, best_weights), (diff < atol) | (cycle_diff < atol)


def yaqa_round_blockwise(
    weights: Float[Array, "*components output_channels input_channels"],
    preconditioner: Preconditioner,
    spec: QuantizedSpec,
    *,
    sharding_config: ShardingConfig,
) -> Float[Array, "*components output_channels input_channels"]:
    input_sharding = sharding_of(weights)
    input_dtype = weights.dtype
    replicated_weight_sharding = sharding_config.make_sharding((None,) * weights.ndim)
    replicated_weights = jax.device_put(
        weights.astype(jnp.float32),
        replicated_weight_sharding,
    )

    def gather_preconditioner_leaf(leaf: Array) -> Array:
        return jax.device_put(
            leaf,
            sharding_config.make_sharding((None,) * leaf.ndim),
        )

    replicated_preconditioner = jtu.tree_map(gather_preconditioner_leaf, preconditioner)
    if replicated_weights.ndim > 2:
        result = jax.vmap(
            partial(
                _yaqa_round_blockwise_2d,
                spec=spec,
                sharding_config=sharding_config,
            ),
            in_axes=(0, 0),
        )(replicated_weights, replicated_preconditioner)
    else:
        result = _yaqa_round_blockwise_2d(
            replicated_weights,
            replicated_preconditioner,
            spec,
            sharding_config=sharding_config,
        )

    return jax.device_put(result.astype(input_dtype), input_sharding)


def _yaqa_round_blockwise_2d(
    weights: Float[Array, "output_channels input_channels"],
    preconditioner: Preconditioner,
    spec: QuantizedSpec,
    *,
    sharding_config: ShardingConfig,
) -> Float[Array, "output_channels input_channels"]:
    output_dim, input_dim = weights.shape
    if output_dim % spec.output_block_size != 0:
        raise ValueError(f"output_dim={output_dim} must be divisible by output_block_size={spec.output_block_size}")
    if input_dim % spec.input_block_size != 0:
        raise ValueError(f"input_dim={input_dim} must be divisible by input_block_size={spec.input_block_size}")

    input_block = preconditioner.input_block
    output_block = preconditioner.output_block
    if input_block is None:
        input_feedback = None
    else:
        input_feedback = block_ldl(input_block, spec.input_block_size).tril
        input_feedback = input_feedback - jnp.identity(input_dim, dtype=input_feedback.dtype)

    if output_block is None:
        output_feedback = None
    else:
        output_feedback = block_ldl(output_block, spec.output_block_size).tril.T
        output_feedback = output_feedback - jnp.identity(output_dim, dtype=output_feedback.dtype)

    quantized_weights = weights * jnp.array(0, dtype=weights.dtype)
    num_output_blocks = output_dim // spec.output_block_size
    num_input_blocks = input_dim // spec.input_block_size

    block_indices = jnp.arange(min(num_output_blocks, num_input_blocks))
    num_anti_diagonals = num_output_blocks + num_input_blocks - 1

    def quantize_anti_diagonal(
        step: Int[Array, ""],
        quantized_weights: Float[Array, "output_channels input_channels"],
    ) -> Float[Array, "output_channels input_channels"]:
        anti_diagonal = num_anti_diagonals - step - 1
        residual = weights - quantized_weights
        first_output_block = jnp.maximum(0, anti_diagonal - num_input_blocks + 1)
        first_input_block = anti_diagonal - first_output_block
        blocks_on_anti_diagonal = jnp.minimum(
            num_output_blocks - first_output_block,
            first_input_block + 1,
        )

        def target_block(block_index: Int[Array, ""]) -> Float[Array, "output_block_size input_block_size"]:
            output_block = jnp.minimum(first_output_block + block_index, num_output_blocks - 1)
            input_block = jnp.maximum(first_input_block - block_index, 0)
            output_start = output_block * spec.output_block_size
            input_start = input_block * spec.input_block_size
            target = jax.lax.dynamic_slice(
                weights,
                (output_start, input_start),
                (spec.output_block_size, spec.input_block_size),
            )

            with use_dot_algorithm_preset(DotAlgorithmPreset.F32_F32_F32):
                if input_feedback is not None:
                    residual_output_block = jax.lax.dynamic_slice(
                        residual,
                        (output_start, 0),
                        (spec.output_block_size, input_dim),
                    )
                    input_feedback_block = jax.lax.dynamic_slice(
                        input_feedback,
                        (0, input_start),
                        (input_dim, spec.input_block_size),
                    )
                    target += residual_output_block @ input_feedback_block
                if output_feedback is not None:
                    output_feedback_block = jax.lax.dynamic_slice(
                        output_feedback,
                        (output_start, 0),
                        (spec.output_block_size, output_dim),
                    )
                    residual_input_block = jax.lax.dynamic_slice(
                        residual,
                        (0, input_start),
                        (output_dim, spec.input_block_size),
                    )
                    target += output_feedback_block @ residual_input_block
                if input_feedback is not None and output_feedback is not None:
                    output_feedback_block = jax.lax.dynamic_slice(
                        output_feedback,
                        (output_start, 0),
                        (spec.output_block_size, output_dim),
                    )
                    input_feedback_block = jax.lax.dynamic_slice(
                        input_feedback,
                        (0, input_start),
                        (input_dim, spec.input_block_size),
                    )
                    target += output_feedback_block @ residual @ input_feedback_block

            return target

        rounded_blocks = spec.quantize_block(
            jax.vmap(target_block)(block_indices),
            sharding_config=sharding_config,
        )

        def update_block(
            block_index: Int[Array, ""],
            quantized_weights: Float[Array, "output_channels input_channels"],
        ) -> Float[Array, "output_channels input_channels"]:
            output_block = jnp.minimum(first_output_block + block_index, num_output_blocks - 1)
            input_block = jnp.maximum(first_input_block - block_index, 0)
            updated_weights = jax.lax.dynamic_update_slice(
                quantized_weights,
                rounded_blocks[block_index],
                (output_block * spec.output_block_size, input_block * spec.input_block_size),
            )
            return jnp.where(block_index < blocks_on_anti_diagonal, updated_weights, quantized_weights)

        return jax.lax.fori_loop(0, len(block_indices), update_block, quantized_weights)

    return jax.lax.fori_loop(0, num_anti_diagonals, quantize_anti_diagonal, quantized_weights)
