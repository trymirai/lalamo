from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.lax import dot_general
from jax.scipy.linalg import solve_triangular
from jaxtyping import Array, DTypeLike, Float, Key

from lalamo.module import Keychain, ParameterNorm, field
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import supports_dummy_arrays
from lalamo.utils.precision import use_dot_algorithm_preset
from lalamo.utils.sharding import LogicalAxis, ShardingConfig, sharding_of
from lalamo.weight_matrix import (
    CompressionImplementation,
    FullPrecisionMatrix,
    FullPrecisionSpec,
    Layout,
    MatmulConfig,
    WeightMatrix,
    WeightMatrixSpec,
)

__all__ = [
    "LowRankMatrix",
    "LowRankSpec",
]


@dataclass(frozen=True)
class LowRankSpec(WeightMatrixSpec):
    rank: int

    @supports_dummy_arrays()
    def compress(
        self,
        weights: Float[Array, "*components out_channels in_channels"],
        *,
        key: Key[Array, ""] | None = None,
        preconditioner: Preconditioner | None = None,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
        sharding_config: ShardingConfig,
        is_sharded: bool = True,
    ) -> "LowRankMatrix":
        if weights.ndim > 2 and preconditioner is not None:
            compress_component = partial(
                self.compress,
                key=key,
                implementation=implementation,
                sharding_config=sharding_config,
                is_sharded=is_sharded,
            )
            spmd_axis_name = sharding_config.resolve_axis(LogicalAxis.MIXTURE)
            return jax.vmap(compress_component, spmd_axis_name=spmd_axis_name)(
                weights,
                preconditioner=preconditioner,
            )

        input_factor = None
        output_factor = None
        if preconditioner is not None:
            input_block = preconditioner.input_block
            output_block = preconditioner.output_block
            if output_block is not None:
                output_factor = jnp.linalg.cholesky(output_block, upper=True)
                weights = output_factor @ weights
            if input_block is not None:
                input_factor = jnp.linalg.cholesky(input_block, upper=True)
                weights = weights @ input_factor.T

        svd_sharding = sharding_config.resolve_sharding((None,) * weights.ndim)
        weights = jax.device_put(weights, svd_sharding)
        u, singular_values, vh = jnp.linalg.svd(weights, full_matrices=False)
        *_, svd_rank = singular_values.shape
        truncated_rank = min(self.rank, svd_rank)
        if truncated_rank < self.rank:
            raise ValueError(f"Requested low-rank rank {self.rank}, but SVD only produced rank {truncated_rank}.")
        singular_value_roots = jnp.sqrt(singular_values[..., :truncated_rank])
        up_projection = u[..., :, :truncated_rank] * singular_value_roots[..., None, :]
        down_projection = singular_value_roots[..., :, None] * vh[..., :truncated_rank, :]
        if output_factor is not None:
            up_projection = solve_triangular(output_factor, up_projection, lower=False)
        if input_factor is not None:
            down_projection = solve_triangular(input_factor, down_projection.T, lower=False).T
        leading_dims = weights.ndim - 2
        up_projection_axes = Layout.OUTPUT_INPUT.weight_partition(leading_dims, is_sharded=is_sharded)
        if is_sharded:
            down_projection_axes = (LogicalAxis.MIXTURE,) * leading_dims + (None, None)
        else:
            down_projection_axes = Layout.OUTPUT_INPUT.weight_partition(leading_dims, is_sharded=False)
        up_projection_sharding = sharding_config.resolve_sharding(up_projection_axes)
        down_projection_sharding = sharding_config.resolve_sharding(down_projection_axes)
        return LowRankMatrix(
            spec=self,
            sharding_config=sharding_config,
            is_sharded=is_sharded,
            up_projection=jax.device_put(up_projection, up_projection_sharding),
            down_projection=jax.device_put(down_projection, down_projection_sharding),
        )


class LowRankMatrix(WeightMatrix[LowRankSpec]):
    down_projection: Float[Array, "*components rank input_channels"] = field(norm=ParameterNorm.SPECTRAL)
    up_projection: Float[Array, "*components output_channels rank"] = field(norm=ParameterNorm.SPECTRAL)

    def to_full_precision(self) -> FullPrecisionMatrix:
        return FullPrecisionSpec(layout=Layout.OUTPUT_INPUT).compress(
            self.decompress(),
            sharding_config=self.sharding_config,
            is_sharded=self.is_sharded,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        *batched_dims, output_dim, _ = self.up_projection.shape
        *_, input_dim = self.down_projection.shape
        return (*batched_dims, output_dim, input_dim)

    @property
    def dtype(self) -> DTypeLike:
        return self.up_projection.dtype

    def astype(self, dtype: DTypeLike) -> "LowRankMatrix":
        return LowRankMatrix(
            spec=self.spec,
            sharding_config=self.sharding_config,
            is_sharded=self.is_sharded,
            up_projection=self.up_projection.astype(dtype),
            down_projection=self.down_projection.astype(dtype),
        )

    def decompress(self) -> Float[Array, "*components out_channels in_channels"]:
        return self.up_projection @ self.down_projection

    def dot(
        self,
        vector: Float[Array, " source_channels"],
        *,
        keychain: Keychain,  # noqa: ARG002
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: bool = False,
    ) -> Float[Array, " target_channels"]:
        self._raise_if_batched()
        up_projection = self.up_projection.astype(vector.dtype)
        down_projection = self.down_projection.astype(vector.dtype)
        out_sharding = sharding_of(vector)

        with use_dot_algorithm_preset(forward_pass_config.precision):
            if transposed:
                rank_features = dot_general(
                    up_projection,
                    vector,
                    dimension_numbers=(((0,), (0,)), ((), ())),
                )
                result = dot_general(
                    down_projection,
                    rank_features,
                    dimension_numbers=(((0,), (0,)), ((), ())),
                    out_sharding=out_sharding,
                )
            else:
                rank_features = dot_general(
                    down_projection,
                    vector,
                    dimension_numbers=(((1,), (0,)), ((), ())),
                )
                result = dot_general(
                    up_projection,
                    rank_features,
                    dimension_numbers=(((1,), (0,)), ((), ())),
                    out_sharding=out_sharding,
                )

        return result
