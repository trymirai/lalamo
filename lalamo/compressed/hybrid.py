from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.core import Tracer
from jaxtyping import Array, DTypeLike, Float, Int, Key

from lalamo.compressed.utils.hadamard import hadamard_transform
from lalamo.module import Keychain, field
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import supports_dummy_arrays
from lalamo.utils.sharding import use_out_sharding
from lalamo.weight_matrix import (
    CompressionImplementation,
    EmbeddingMatrix,
    FullPrecisionMatrix,
    FullPrecisionSpec,
    Layout,
    MatmulConfig,
    WeightMatrix,
    WeightMatrixSpec,
)

__all__ = [
    "HybridMatrix",
    "HybridSpec",
    "IncoherenceProcessingMode",
]


def _random_incoherence_signs(
    channels: int,
    key: Key[Array, ""],
) -> Int[Array, " channels"]:
    return jnp.where(jax.random.bernoulli(key, shape=(channels,)), 1, -1)


def _hadamard_transform_output_axis(
    inputs: Float[Array, "*components out_channels in_channels"],
    block_size: Literal[32, 64, 128],
) -> Float[Array, "*components out_channels in_channels"]:
    transposed = jnp.swapaxes(inputs, -1, -2)
    transformed_transposed = hadamard_transform(transposed, block_size)
    return jnp.swapaxes(transformed_transposed, -1, -2)


def _process_preconditioner_block(
    block: Float[Array, "*components channels channels"] | None,
    signs: Int[Array, " channels"] | None,
    block_size: Literal[32, 64, 128],
) -> Float[Array, "*components channels channels"] | None:
    if block is None or signs is None:
        return block
    signed_block = block * signs[..., None] * signs[None, ...]
    return _hadamard_transform_output_axis(
        hadamard_transform(signed_block, block_size),
        block_size,
    )


def _process_preconditioner(
    preconditioner: Preconditioner,
    incoherence_signs: "IncoherenceSigns",
    block_size: Literal[32, 64, 128],
) -> Preconditioner:
    return Preconditioner.init(
        input_block=_process_preconditioner_block(
            preconditioner.input_block,
            incoherence_signs.input_signs,
            block_size,
        ),
        output_block=_process_preconditioner_block(
            preconditioner.output_block,
            incoherence_signs.output_signs,
            block_size,
        ),
    )


class IncoherenceProcessingMode(StrEnum):
    INPUT = "input"
    OUTPUT = "output"
    INPUT_OUTPUT = "input_output"


class IncoherenceSigns(eqx.Module):
    input_signs: Int[Array, " in_channels"] | None = field(trainable=False)
    output_signs: Int[Array, " out_channels"] | None = field(trainable=False)

    @classmethod
    def random_init(
        cls,
        input_dim: int,
        output_dim: int,
        mode: IncoherenceProcessingMode,
        key: Key[Array, ""] | None,
    ) -> "IncoherenceSigns":
        if key is None:
            raise ValueError("Cannot initialize random incoherence signs without a random key.")
        input_key, output_key = jax.random.split(key, 2)
        return cls(
            input_signs=(
                None
                if mode == IncoherenceProcessingMode.OUTPUT
                else _random_incoherence_signs(channels=input_dim, key=input_key)
            ),
            output_signs=(
                None
                if mode == IncoherenceProcessingMode.INPUT
                else _random_incoherence_signs(channels=output_dim, key=output_key)
            ),
        )

    def process_weights(
        self,
        weights: Float[Array, "*components out_channels in_channels"],
        block_size: Literal[32, 64, 128],
    ) -> Float[Array, "*components out_channels in_channels"]:
        if self.input_signs is not None:
            weights = hadamard_transform(weights * self.input_signs, block_size)
        if self.output_signs is not None:
            weights = _hadamard_transform_output_axis(weights * self.output_signs[..., None], block_size)
        return weights

    def unprocess_weights(
        self,
        weights: Float[Array, "*components out_channels in_channels"],
        block_size: Literal[32, 64, 128],
    ) -> Float[Array, "*components out_channels in_channels"]:
        return self.unprocess_weight_input_axis(
            self.unprocess_weight_output_axis(weights, block_size),
            block_size,
        )

    def unprocess_weight_input_axis(
        self,
        weights: Float[Array, "*components out_channels in_channels"],
        block_size: Literal[32, 64, 128],
    ) -> Float[Array, "*components out_channels in_channels"]:
        if self.input_signs is None:
            return weights
        restored = hadamard_transform(weights, block_size)
        return restored * self.input_signs

    def unprocess_weight_output_axis(
        self,
        weights: Float[Array, "*components out_channels in_channels"],
        block_size: Literal[32, 64, 128],
    ) -> Float[Array, "*components out_channels in_channels"]:
        if self.output_signs is None:
            return weights
        restored = _hadamard_transform_output_axis(weights, block_size)
        return restored * self.output_signs[..., None]

    def input_transform(
        self,
        vector: Float[Array, " source_channels"],
        block_size: Literal[32, 64, 128],
        *,
        transposed: bool = False,
    ) -> Float[Array, " source_channels"]:
        if self.input_signs is None:
            return vector
        if transposed:
            return hadamard_transform(vector, block_size) * self.input_signs
        return hadamard_transform(vector * self.input_signs, block_size)

    def output_transform(
        self,
        vector: Float[Array, " target_channels"],
        block_size: Literal[32, 64, 128],
        *,
        transposed: bool = False,
    ) -> Float[Array, " target_channels"]:
        if self.output_signs is None:
            return vector
        if transposed:
            return hadamard_transform(vector * self.output_signs, block_size)
        return hadamard_transform(vector, block_size) * self.output_signs


@dataclass(frozen=True)
class HybridSpec(WeightMatrixSpec):
    quantization_spec: WeightMatrixSpec
    adapter_spec: WeightMatrixSpec | None
    incoherence_block_size: Literal[32, 64, 128] | None = 32
    incoherence_processing_mode: IncoherenceProcessingMode = IncoherenceProcessingMode.INPUT_OUTPUT

    @supports_dummy_arrays()
    def compress(
        self,
        weights: Float[Array, "*components out_channels in_channels"],
        *,
        key: Key[Array, ""] | None = None,
        preconditioner: Preconditioner | None = None,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
        is_sharded: bool = True,
    ) -> "HybridMatrix":
        *_, output_dim, input_dim = weights.shape
        if key is None and isinstance(weights, Tracer):
            key = jax.random.key(0)

        if key is not None:
            quantization_key, adapter_key, incoherence_key = jax.random.split(key, 3)
        else:
            quantization_key, adapter_key, incoherence_key = None, None, None

        incoherence_signs = None
        quantization_preconditioner = preconditioner
        if self.incoherence_block_size is not None:
            incoherence_signs = IncoherenceSigns.random_init(
                input_dim=input_dim,
                output_dim=output_dim,
                mode=self.incoherence_processing_mode,
                key=incoherence_key,
            )
            weights = incoherence_signs.process_weights(weights, self.incoherence_block_size)
            if preconditioner is not None:
                quantization_preconditioner = _process_preconditioner(
                    preconditioner,
                    incoherence_signs,
                    self.incoherence_block_size,
                )

        quantized = self.quantization_spec.compress(
            weights,
            key=quantization_key,
            preconditioner=quantization_preconditioner,
            implementation=implementation,
            is_sharded=is_sharded,
        )
        if self.adapter_spec is not None:
            residual = weights - quantized.decompress()
            adapter_preconditioner = preconditioner
            if incoherence_signs is not None:
                assert self.incoherence_block_size is not None
                residual = incoherence_signs.unprocess_weight_input_axis(
                    residual,
                    self.incoherence_block_size,
                )
                if preconditioner is not None:
                    assert quantization_preconditioner is not None
                    adapter_preconditioner = Preconditioner.init(
                        input_block=preconditioner.input_block,
                        output_block=quantization_preconditioner.output_block,
                    )
            adapter = self.adapter_spec.compress(
                residual,
                key=adapter_key,
                preconditioner=adapter_preconditioner,
                implementation=implementation,
                is_sharded=is_sharded,
            )
        else:
            adapter = None

        return HybridMatrix(
            spec=self,
            is_sharded=is_sharded,
            quantized=quantized,
            adapter=adapter,
            incoherence_signs=incoherence_signs,
        )


class HybridMatrix(EmbeddingMatrix[HybridSpec]):
    quantized: WeightMatrix[WeightMatrixSpec]
    adapter: WeightMatrix[WeightMatrixSpec] | None
    incoherence_signs: IncoherenceSigns | None

    def to_full_precision(self) -> FullPrecisionMatrix:
        return FullPrecisionSpec(layout=Layout.OUTPUT_INPUT).compress(self.decompress(), is_sharded=self.is_sharded)

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
            is_sharded=self.is_sharded,
            quantized=quantized,
            adapter=adapter,
            incoherence_signs=self.incoherence_signs,
        )

    def decompress(self) -> Float[Array, "*components out_channels in_channels"]:
        result = self.quantized.decompress()
        block_size = self.spec.incoherence_block_size
        if self.incoherence_signs is not None:
            assert block_size is not None
            result = self.incoherence_signs.unprocess_weights(result, block_size)
        if self.adapter is not None:
            adapter = self.adapter.decompress()
            if self.incoherence_signs is not None:
                assert block_size is not None
                adapter = self.incoherence_signs.unprocess_weight_output_axis(adapter, block_size)
            result = result + adapter
        return result

    def switch_implementation(self, implementation: CompressionImplementation) -> "HybridMatrix":
        quantized = self.quantized.switch_implementation(implementation)
        adapter = self.adapter
        if adapter is not None:
            adapter = adapter.switch_implementation(implementation)
        return HybridMatrix(
            spec=self.spec,
            is_sharded=self.is_sharded,
            quantized=quantized,
            adapter=adapter,
            incoherence_signs=self.incoherence_signs,
        )

    @use_out_sharding((None,))
    def lookup_embedding(
        self,
        index: int | Int[Array, ""],
        *,
        dtype: DTypeLike | None = None,
        keychain: Keychain,
        forward_pass_config: MatmulConfig = MatmulConfig(),
    ) -> Float[Array, " out_channels"]:
        self._raise_if_batched()
        if self.incoherence_signs is not None and self.incoherence_signs.input_signs is not None:
            raise ValueError("Hybrid embedding lookup is only supported when input RHT is disabled.")
        if not isinstance(self.quantized, EmbeddingMatrix):
            raise TypeError("Hybrid embedding lookup requires an embedding-compatible quantization matrix.")
        if self.adapter is not None:
            raise TypeError("Hybrid embedding lookup does not support adapters.")
        result = self.quantized.lookup_embedding(
            index,
            dtype=dtype,
            keychain=keychain,
            forward_pass_config=forward_pass_config,
        )
        if self.incoherence_signs is not None:
            assert self.spec.incoherence_block_size is not None
            result = self.incoherence_signs.output_transform(result, self.spec.incoherence_block_size)
        return result

    def dot(
        self,
        vector: Float[Array, " source_channels"],
        *,
        keychain: Keychain,
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: bool = False,
    ) -> Float[Array, " target_channels"]:
        if transposed:
            if self.adapter is not None:
                raise TypeError("Hybrid transposed matmul does not support adapters.")
            quantized_vector = vector
            if self.incoherence_signs is not None:
                assert self.spec.incoherence_block_size is not None
                quantized_vector = self.incoherence_signs.output_transform(
                    vector,
                    self.spec.incoherence_block_size,
                    transposed=True,
                )
            result = self.quantized.dot(
                quantized_vector,
                keychain=keychain,
                forward_pass_config=forward_pass_config,
                transposed=True,
            )
            if self.incoherence_signs is not None:
                assert self.spec.incoherence_block_size is not None
                result = self.incoherence_signs.input_transform(
                    result,
                    self.spec.incoherence_block_size,
                    transposed=True,
                )
            return result

        quantized_vector = vector
        if self.incoherence_signs is not None:
            assert self.spec.incoherence_block_size is not None
            quantized_vector = self.incoherence_signs.input_transform(vector, self.spec.incoherence_block_size)

        adapter = self.adapter
        if adapter is None:
            result = self.quantized.dot(
                quantized_vector,
                keychain=keychain,
                forward_pass_config=forward_pass_config,
            )
        else:
            quantized_keychain, adapter_keychain = keychain.split(2)
            result = self.quantized.dot(
                quantized_vector,
                keychain=quantized_keychain,
                forward_pass_config=forward_pass_config,
            )
            result = result + adapter.dot(
                vector,
                keychain=adapter_keychain,
                forward_pass_config=forward_pass_config,
            )
        if self.incoherence_signs is not None:
            assert self.spec.incoherence_block_size is not None
            return self.incoherence_signs.output_transform(result, self.spec.incoherence_block_size)
        return result
