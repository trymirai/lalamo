from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int, Key

from lalamo.compressed.hadamard import hadamard_transform
from lalamo.exportable import ExportResults
from lalamo.module import Keychain, field
from lalamo.utils.dummy_array import contains_dummy_arrays, dummy_array, supports_dummy_arrays
from lalamo.utils.parameter_path import ParameterPath
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
]

_HYBRID_FINAL_CACHE_MIN_OUTPUT_DIM = 10240


def _random_incoherence_signs(
    channels: int,
    key: Key[Array, ""],
) -> Int[Array, " channels"]:
    factors = jnp.where(jax.random.bernoulli(key, shape=(channels,)), 1, -1)
    return factors.astype(jnp.int32)


def _hadamard_transform_output_axis(
    inputs: Float[Array, "... out_channels in_channels"],
    block_size: Literal[32, 64, 128],
) -> Float[Array, "... out_channels in_channels"]:
    transposed = jnp.swapaxes(inputs, -1, -2)
    transformed_transposed = hadamard_transform(transposed, block_size)
    return jnp.swapaxes(transformed_transposed, -1, -2)


class IncoherenceSigns(eqx.Module):
    input_signs: Int[Array, " in_channels"] = field(trainable=False)
    output_signs: Int[Array, " out_channels"] = field(trainable=False)

    @classmethod
    def random_init(
        cls,
        input_dim: int,
        output_dim: int,
        key: Key[Array, ""] | None,
    ) -> "IncoherenceSigns":
        if key is None:
            raise ValueError("Cannot initialize random incoherence signs without a random key.")
        input_key, output_key = jax.random.split(key, 2)
        return cls(
            input_signs=_random_incoherence_signs(channels=input_dim, key=input_key),
            output_signs=_random_incoherence_signs(channels=output_dim, key=output_key),
        )

    def process_weights(
        self,
        weights: Float[Array, "... out_channels in_channels"],
        block_size: Literal[32, 64, 128],
    ) -> Float[Array, "... out_channels in_channels"]:
        signed_weights = weights * self.output_signs.astype(weights.dtype)[..., None]
        signed_weights = signed_weights * self.input_signs.astype(weights.dtype)
        return _hadamard_transform_output_axis(
            hadamard_transform(signed_weights, block_size),
            block_size,
        )

    def unprocess_weights(
        self,
        weights: Float[Array, "... out_channels in_channels"],
        block_size: Literal[32, 64, 128],
    ) -> Float[Array, "... out_channels in_channels"]:
        output_restored = _hadamard_transform_output_axis(weights, block_size)
        output_restored = output_restored * self.output_signs.astype(output_restored.dtype)[..., None]
        input_restored = hadamard_transform(output_restored, block_size)
        return input_restored * self.input_signs.astype(input_restored.dtype)

    def input_transform(
        self,
        vector: Float[Array, " channels"],
        block_size: Literal[32, 64, 128],
        *,
        transposed: bool,
    ) -> Float[Array, " channels"]:
        if transposed:
            signs = self.output_signs
        else:
            signs = self.input_signs
        return hadamard_transform(vector * signs.astype(vector.dtype), block_size)

    def output_transform(
        self,
        vector: Float[Array, " channels"],
        block_size: Literal[32, 64, 128],
        *,
        transposed: bool,
    ) -> Float[Array, " channels"]:
        if transposed:
            signs = self.input_signs
        else:
            signs = self.output_signs
        return hadamard_transform(vector, block_size) * signs.astype(vector.dtype)


@dataclass(frozen=True)
class HybridSpec(WeightMatrixSpec):
    quantization_spec: WeightMatrixSpec
    adapter_spec: WeightMatrixSpec | None
    incoherence_block_size: Literal[32, 64, 128] | None = 32

    @supports_dummy_arrays()
    def compress(
        self,
        weights: Float[Array, "... out_channels in_channels"],
        *,
        key: Key[Array, ""] | None = None,
        preconditioner: Preconditioner | None = None,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
    ) -> "HybridMatrix":
        if preconditioner is not None:
            raise ValueError("Preconditioned rounding is not implemented yet.")

        *_, output_dim, input_dim = weights.shape

        if key is not None:
            quantization_key, adapter_key, incoherence_key = jax.random.split(key, 3)
        else:
            quantization_key, adapter_key, incoherence_key = None, None, None

        if self.incoherence_block_size is not None:
            incoherence_signs = IncoherenceSigns.random_init(
                input_dim=input_dim,
                output_dim=output_dim,
                key=incoherence_key,
            )
            weights = incoherence_signs.process_weights(weights, self.incoherence_block_size)
        else:
            incoherence_signs = None

        quantized = self.quantization_spec.compress(
            weights,
            key=quantization_key,
            implementation=implementation,
        )
        if self.adapter_spec is not None:
            adapter = self.adapter_spec.compress(
                weights - quantized.decompress(),
                key=adapter_key,
                implementation=implementation,
            )
        else:
            adapter = None

        matrix = HybridMatrix(
            spec=self,
            quantized=quantized,
            adapter=adapter,
            incoherence_signs=incoherence_signs,
        )
        if implementation == CompressionImplementation.INFERENCE and _uses_final_cache(output_dim, matrix):
            return HybridMatrixForInference.from_hybrid(matrix)
        return matrix


class HybridMatrix(WeightMatrix[HybridSpec]):
    quantized: WeightMatrix[WeightMatrixSpec]
    adapter: WeightMatrix[WeightMatrixSpec] | None
    incoherence_signs: IncoherenceSigns | None

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
            incoherence_signs=self.incoherence_signs,
        )

    @use_out_sharding((None, None))
    def decompress(self) -> Float[Array, "... out_channels in_channels"]:
        result = self.quantized.decompress()
        if self.adapter is not None:
            result = result + self.adapter.decompress()
        if self.incoherence_signs is None:
            return result
        block_size = self.spec.incoherence_block_size
        assert block_size is not None
        return self.incoherence_signs.unprocess_weights(result, block_size)

    def switch_implementation(self, implementation: CompressionImplementation) -> "HybridMatrix":
        quantized = self.quantized.switch_implementation(implementation)
        adapter = self.adapter
        if adapter is not None:
            adapter = adapter.switch_implementation(implementation)
        matrix = HybridMatrix(
            spec=self.spec,
            quantized=quantized,
            adapter=adapter,
            incoherence_signs=self.incoherence_signs,
        )
        if implementation == CompressionImplementation.INFERENCE and _uses_final_cache(matrix.shape[-2], matrix):
            return HybridMatrixForInference.from_hybrid(matrix)
        return matrix

    def dot(
        self,
        vector: Float[Array, " channels"],
        *,
        keychain: Keychain,
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: bool = False,
    ) -> Float[Array, "... channels"]:
        block_size = self.spec.incoherence_block_size
        if self.incoherence_signs is not None:
            assert block_size is not None
            vector = self.incoherence_signs.input_transform(
                vector,
                block_size,
                transposed=transposed,
            )
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
        if self.incoherence_signs is not None:
            assert block_size is not None
            result = self.incoherence_signs.output_transform(
                result,
                block_size,
                transposed=transposed,
            )
        return result


def _full_precision_cache(matrix: HybridMatrix) -> FullPrecisionMatrix:
    if contains_dummy_arrays(matrix):
        weights = dummy_array(matrix.shape, matrix.dtype)
    else:
        weights = matrix.decompress()
    return FullPrecisionSpec(layout=Layout.OUTPUT_INPUT).compress(weights)


def _uses_final_cache(output_dim: int, matrix: HybridMatrix) -> bool:
    has_wrapper_work = matrix.incoherence_signs is not None or matrix.adapter is not None
    return has_wrapper_work and output_dim >= _HYBRID_FINAL_CACHE_MIN_OUTPUT_DIM


class HybridMatrixForInference(HybridMatrix):
    weights: FullPrecisionMatrix = field(trainable=False)

    @classmethod
    def from_hybrid(cls, matrix: HybridMatrix) -> "HybridMatrixForInference":
        return cls(
            spec=matrix.spec,
            quantized=matrix.quantized,
            adapter=matrix.adapter,
            incoherence_signs=matrix.incoherence_signs,
            weights=_full_precision_cache(matrix),
        )

    def _without_cache(self) -> HybridMatrix:
        return HybridMatrix(
            spec=self.spec,
            quantized=self.quantized,
            adapter=self.adapter,
            incoherence_signs=self.incoherence_signs,
        )

    def export(self) -> ExportResults:
        return self._without_cache().export()

    def load_exported(
        self,
        expored_data: ExportResults,
        allow_dtype_cast: bool = False,
        *,
        prefix: ParameterPath | None = None,
    ) -> "HybridMatrixForInference":
        loaded = self._without_cache().load_exported(
            expored_data,
            allow_dtype_cast=allow_dtype_cast,
            prefix=prefix,
        )
        return HybridMatrixForInference.from_hybrid(loaded)

    def to_full_precision(self) -> FullPrecisionMatrix:
        return self.weights

    @property
    def shape(self) -> tuple[int, ...]:
        return self.weights.shape

    @property
    def dtype(self) -> DTypeLike:
        return self.weights.dtype

    def astype(self, dtype: DTypeLike) -> "HybridMatrixForInference":
        return HybridMatrixForInference(
            spec=self.spec,
            quantized=self.quantized.astype(dtype),
            adapter=self.adapter.astype(dtype) if self.adapter is not None else None,
            incoherence_signs=self.incoherence_signs,
            weights=self.weights.astype(dtype),
        )

    @use_out_sharding((None, None))
    def decompress(self) -> Float[Array, "... out_channels in_channels"]:
        return self.weights.decompress()

    def switch_implementation(self, implementation: CompressionImplementation) -> HybridMatrix:
        if implementation == CompressionImplementation.INFERENCE:
            return self
        return self._without_cache().switch_implementation(implementation)

    def dot(
        self,
        vector: Float[Array, " channels"],
        *,
        keychain: Keychain,
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: bool = False,
    ) -> Float[Array, "... channels"]:
        return self.weights.dot(
            vector,
            keychain=keychain,
            forward_pass_config=forward_pass_config,
            transposed=transposed,
        )
