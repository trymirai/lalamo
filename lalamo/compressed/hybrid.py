from dataclasses import dataclass
from typing import Literal, overload

from jaxtyping import Array, DTypeLike, Float

from lalamo.module import Keychain
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
    NONE = "none"


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


class HybridMatrix(WeightMatrix[HybridSpec]):
    quantized: WeightMatrix
    adapter: WeightMatrix | None

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
