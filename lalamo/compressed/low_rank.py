from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from jaxtyping import Array, DTypeLike, Float, Int, Key

from lalamo.module import Keychain, ParameterNorm, ShardingAxis, field
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import supports_dummy_arrays
from lalamo.utils.precision import use_dot_algorithm_preset
from lalamo.utils.sharding import make_sharding, reshard_as, use_out_sharding, with_sharding
from lalamo.weight_matrix import (
    CompressionImplementation,
    FullPrecisionMatrix,
    FullPrecisionSpec,
    Layout,
    MatmulConfig,
    WeightMatrix,
    WeightMatrixSpec,
    _ragged_dot,
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
        is_sharded: bool = True,
    ) -> "LowRankMatrix":
        if weights.ndim > 2 and preconditioner is not None:
            compress_component = partial(
                self.compress,
                key=key,
                implementation=implementation,
                is_sharded=is_sharded,
            )
            return jax.vmap(compress_component, spmd_axis_name=ShardingAxis.EXPERT)(
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
        if not is_sharded:
            up_projection = with_sharding(up_projection, make_sharding((None,) * up_projection.ndim))
            down_projection = with_sharding(down_projection, make_sharding((None,) * down_projection.ndim))
        return LowRankMatrix(
            spec=self,
            is_sharded=is_sharded,
            up_projection=up_projection,
            down_projection=down_projection,
        )


class LowRankMatrix(WeightMatrix[LowRankSpec]):
    down_projection: Float[Array, "*components rank input_channels"] = field(norm=ParameterNorm.SPECTRAL)
    up_projection: Float[Array, "*components output_channels rank"] = field(norm=ParameterNorm.SPECTRAL)

    def to_full_precision(self) -> FullPrecisionMatrix:
        return FullPrecisionSpec(layout=Layout.OUTPUT_INPUT).compress(self.decompress(), is_sharded=self.is_sharded)

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
            is_sharded=self.is_sharded,
            up_projection=self.up_projection.astype(dtype),
            down_projection=self.down_projection.astype(dtype),
        )

    @use_out_sharding((None, None))
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

        with use_dot_algorithm_preset(forward_pass_config.precision):
            if transposed:
                result = down_projection.T @ (up_projection.T @ vector)
            else:
                result = up_projection @ (down_projection @ vector)

        return result

    def ragged_dot(
        self,
        vectors: Float[Array, "tokens input_channels"],
        group_sizes: Int[Array, " experts"],
        *,
        keychain: Keychain,  # noqa: ARG002
        forward_pass_config: MatmulConfig = MatmulConfig(),
    ) -> Float[Array, "tokens output_channels"]:
        self._raise_if_not_batched()
        down_projection = self.down_projection.astype(vectors.dtype)
        up_projection = self.up_projection.astype(vectors.dtype)

        hidden = _ragged_dot(
            vectors,
            down_projection.swapaxes(-1, -2),
            group_sizes,
            precision=forward_pass_config.precision,
        )
        result = _ragged_dot(
            hidden,
            up_projection.swapaxes(-1, -2),
            group_sizes,
            precision=forward_pass_config.precision,
        )
        return reshard_as(result, vectors)
