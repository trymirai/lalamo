from dataclasses import dataclass
from enum import StrEnum

from jaxtyping import Array, DTypeLike, Float, Int, Key

import jax
import jax.numpy as jnp
import equinox as eqx
from lalamo.module import Keychain, field
from lalamo.utils.dummy_array import supports_dummy_arrays
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
    "HybridMatrix",
    "HybridSpec",
    "IncoherenceProcessing",
]


class IncoherenceProcessing(StrEnum):
    RANDOM_HADAMARD = "random_hadamard"




def _random_incoherence_factors(
    channels: int,
    key: Key[Array, ""],
) -> Int[Array, " channels"]:
    factors = jnp.where(jax.random.bernoulli(key, shape=(channels,)), 1, -1)
    return factors.astype(jnp.int32)


class RHTFactors(eqx.Module):
    input_factors: Int[Array, " in_channels"] = field(trainable=False)
    output_factors: Int[Array, " out_channels"] = field(trainable=False)

    @classmethod
    def random_init(
        cls,
        input_dim: int,
        output_dim: int,
        key: Key[Array, ""],
    ) -> "RHTFactors":
        input_key, output_key = jax.random.split(key, 2)
        return cls(
            input_factors=_random_incoherence_factors(channels=input_dim, key=input_key),
            output_factors=_random_incoherence_factors(
                channels=output_dim,
                key=output_key,
            ),
        )



@dataclass(frozen=True)
class HybridSpec(WeightMatrixSpec):
    quantization_spec: WeightMatrixSpec
    adapter_spec: WeightMatrixSpec | None
    incoherence_processing: IncoherenceProcessing | None

    @supports_dummy_arrays()
    def compress(
        self,
        weights: Float[Array, "... out_channels in_channels"],
        key: Key[Array, ""],
        preconditioner: Preconditioner | None = None,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
    ) -> "HybridMatrix":
        if preconditioner is not None:
            raise ValueError("Preconditioned rounding is not implemented yet.")

        quantized = self.quantization_spec.compress(
            weights,
            implementation=implementation,
        )
        if self.adapter_spec is not None:
            adapter = self.adapter_spec.compress(
                weights - quantized.decompress(),
                implementation=implementation,
            )
        else:
            adapter = None

        return HybridMatrix(
            spec=self,
            quantized=quantized,
            adapter=adapter,
        )


class HybridMatrix(WeightMatrix[HybridSpec]):
    quantized: WeightMatrix[WeightMatrixSpec]
    adapter: WeightMatrix[WeightMatrixSpec] | None
    incoherence_keys:

    def to_full_precision(self) -> FullPrecisionMatrix:
        return FullPrecisionSpec(layout=Layout.OUTPUT_INPUT).compress(self.decompress())

    @property
    def shape(self) -> tuple[int, ...]:
        return self.quantized.decompress().shape

    @property
    def dtype(self) -> DTypeLike:
        return self.quantized.dtype

    def astype(self, dtype: DTypeLike) -> "HybridMatrix":
        quantized = self.quantized.astype(dtype)
        adapter = self.adapter
        if adapter is not None:
            adapter = adapter.astype(dtype)
        return HybridMatrix(
            spec=self.spec,
            quantized=quantized,
            adapter=adapter,
        )

    @use_out_sharding((None, None))
    def decompress(self) -> Float[Array, "... out_channels in_channels"]:
        result = self.quantized.decompress()
        if self.adapter is None:
            return result
        return result + self.adapter.decompress()

    def switch_implementation(self, implementation: CompressionImplementation) -> "HybridMatrix":
        quantized = self.quantized.switch_implementation(implementation)
        adapter = self.adapter
        if adapter is not None:
            adapter = adapter.switch_implementation(implementation)
        return HybridMatrix(
            spec=self.spec,
            quantized=quantized,
            adapter=adapter,
        )

    def dot(
        self,
        vector: Float[Array, " channels"],
        *,
        keychain: Keychain,
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: bool = False,
    ) -> Float[Array, "... channels"]:
        quantized_keychain, adapter_keychain = keychain.split(2)
        result = self.quantized.dot(
            vector,
            keychain=quantized_keychain,
            forward_pass_config=forward_pass_config,
            transposed=transposed,
        )
        if self.adapter is not None:
            result += self.adapter.dot(
                vector,
                keychain=adapter_keychain,
                forward_pass_config=forward_pass_config,
                transposed=transposed,
            )
        return result
