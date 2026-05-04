from dataclasses import dataclass
from typing import Literal, overload

import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float

from lalamo.module import Keychain, ParameterNorm, field
from lalamo.utils.dummy_array import supports_dummy_arrays
from lalamo.utils.precision import use_dot_algorithm_preset
from lalamo.utils.sharding import use_out_sharding
from lalamo.weight_matrix import (
    CompressionImplementation,
    FullPrecisionMatrix,
    FullPrecisionSpec,
    Layout,
    MatmulConfig,
    Preconditioner,
    WeightMatrix,
    WeightMatrixSpec,
)

__all__ = [
    "LowRankMatrix",
    "LowRankSpec",
]


class IncoherenceProcessing(StrEnum):
    RANDOM_HADAMARD = "random_hadamard"


@dataclass(frozen=True)
class HybridSpec(WeightMatrixSpec):
    quantization_spec: WeightMatrixSpec
    adapter_spec: WeightMatrixSpec | None

    @supports_dummy_arrays()
    def compress(
        self,
        weights: Float[Array, "... out_channels in_channels"],
        preconditioner: Preconditioner | None = None,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,  # noqa: ARG002
    ) -> "LowRankMatrix":
        if preconditioner is not None:
            raise ValueError("Preconditioned rounding is not implemented yet.")

        u, singular_values, vh = jnp.linalg.svd(weights, full_matrices=False)
        *_, svd_rank = singular_values.shape
        truncated_rank = min(self.rank, svd_rank)
        if truncated_rank < self.rank:
            raise ValueError(f"Requested low-rank rank {self.rank}, but SVD only produced rank {truncated_rank}.")
        singular_value_roots = jnp.sqrt(singular_values[..., :truncated_rank])
        up_projection = u[..., :, :truncated_rank] * singular_value_roots[..., None, :]
        down_projection = singular_value_roots[..., :, None] * vh[..., :truncated_rank, :]
        return LowRankMatrix(
            spec=self,
            up_projection=up_projection,
            down_projection=down_projection,
        )


class LowRankMatrix(WeightMatrix[LowRankSpec]):
    down_projection: Float[Array, "... rank input_channels"] = field(norm=ParameterNorm.SPECTRAL)
    up_projection: Float[Array, "... output_channels rank"] = field(norm=ParameterNorm.SPECTRAL)

    def to_full_precision(self) -> FullPrecisionMatrix:
        return FullPrecisionSpec(layout=Layout.OUTPUT_INPUT).compress(self.decompress())

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
            up_projection=self.up_projection.astype(dtype),
            down_projection=self.down_projection.astype(dtype),
        )

    @use_out_sharding((None, None))
    def decompress(self) -> Float[Array, "... out_channels in_channels"]:
        return self.up_projection @ self.down_projection

    @overload
    def dot(
        self,
        vector: Float[Array, " in_channels"],
        *,
        keychain: Keychain,
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: Literal[False] = False,
    ) -> Float[Array, "... out_channels"]: ...

    @overload
    def dot(
        self,
        vector: Float[Array, " out_channels"],
        *,
        keychain: Keychain,
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: Literal[True],
    ) -> Float[Array, "... in_channels"]: ...

    def dot(
        self,
        vector: Float[Array, " channels"],
        *,
        keychain: Keychain,  # noqa: ARG002
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: bool = False,
    ) -> Float[Array, "... channels"]:
        self._raise_if_batched()
        up_projection = self.up_projection.astype(vector.dtype)
        down_projection = self.down_projection.astype(vector.dtype)

        with use_dot_algorithm_preset(forward_pass_config.precision):
            if transposed:
                result = down_projection.T @ (up_projection.T @ vector)
            else:
                result = up_projection @ (down_projection @ vector)

        return result
