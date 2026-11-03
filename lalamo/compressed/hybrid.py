from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp
from jax.core import Tracer
from jaxtyping import Array, DTypeLike, Float, Int, Key

from lalamo.module import Keychain
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import supports_dummy_arrays
from lalamo.utils.sharding import ShardingConfig
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

from .incoherence import HadamardSpec, IncoherenceProcessor, IncoherenceProcessorSpec

__all__ = [
    "HybridMatrix",
    "HybridSpec",
]


@dataclass(frozen=True)
class HybridSpec(WeightMatrixSpec):
    quantization_spec: WeightMatrixSpec
    adapter_spec: WeightMatrixSpec | None
    incoherence_spec: IncoherenceProcessorSpec | None = field(default_factory=HadamardSpec)

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
    ) -> "HybridMatrix":
        *_, output_dim, input_dim = weights.shape
        if key is None and isinstance(weights, Tracer):
            key = jax.random.key(0)

        if key is not None:
            quantization_key, adapter_key = jax.random.split(key, 2)
            incoherence_key = jax.random.key(
                42
            )  # We use the same shared keys for matrices of the same size to enable RHT fusion.
        else:
            quantization_key, adapter_key, incoherence_key = None, None, None

        processor = None
        quantization_preconditioner = preconditioner
        if self.incoherence_spec is not None:
            processor = self.incoherence_spec.init(input_dim, output_dim, incoherence_key, sharding_config)
            weights = processor.process_weights(weights, sharding_config)
            if preconditioner is not None:
                quantization_preconditioner = processor.process_preconditioner(preconditioner, sharding_config)

        quantized = self.quantization_spec.compress(
            weights,
            key=quantization_key,
            preconditioner=quantization_preconditioner,
            implementation=implementation,
            sharding_config=sharding_config,
            is_sharded=is_sharded,
        )
        if self.adapter_spec is not None:
            residual = weights - quantized.decompress()
            adapter_preconditioner = preconditioner
            if processor is not None:
                residual = processor.unprocess_input_axis(residual, sharding_config)
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
                sharding_config=sharding_config,
                is_sharded=is_sharded,
            )
        else:
            adapter = None

        return HybridMatrix(
            spec=self,
            sharding_config=sharding_config,
            is_sharded=is_sharded,
            quantized=quantized,
            adapter=adapter,
            # Expert modules slice and vmap every array along their component axes.
            incoherence_signs=None if processor is None else processor.with_components(weights.shape[:-2]),
        )


class HybridMatrix(EmbeddingMatrix[HybridSpec]):
    quantized: WeightMatrix[WeightMatrixSpec]
    adapter: WeightMatrix[WeightMatrixSpec] | None
    incoherence_signs: IncoherenceProcessor | None

    @classmethod
    def of(cls, quantized: WeightMatrix[WeightMatrixSpec], processor: IncoherenceProcessor) -> "HybridMatrix":
        spec = HybridSpec(quantized.spec, None, processor.spec)
        return cls(spec, quantized.sharding_config, quantized.is_sharded, quantized, None, processor)

    def to_full_precision(self) -> FullPrecisionMatrix:
        layout = getattr(self.quantized.spec, "layout", Layout.OUTPUT_INPUT)
        return FullPrecisionSpec(layout=layout).compress(
            self.decompress(),
            sharding_config=self.sharding_config,
            is_sharded=self.is_sharded,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.quantized.shape

    @property
    def dtype(self) -> DTypeLike:
        return self.quantized.dtype

    def astype(self, dtype: DTypeLike) -> "HybridMatrix":
        quantized = self.quantized.astype(dtype)
        adapter = self.adapter
        if adapter is not None:
            adapter = adapter.astype(dtype)
        return replace(self, quantized=quantized, adapter=adapter)

    def decompress(self) -> Float[Array, "*components out_channels in_channels"]:
        # Decoded in float32 and rounded once, after the rotation.
        result = self.quantized.astype(jnp.float32).decompress()
        if self.incoherence_signs is not None:
            result = self.incoherence_signs.unprocess_weights(result, self.sharding_config)
        if self.adapter is not None:
            adapter = self.adapter.decompress()
            if self.incoherence_signs is not None:
                adapter = self.incoherence_signs.unprocess_output_axis(adapter, self.sharding_config)
            result = result + adapter
        return result.astype(self.dtype)

    def switch_implementation(self, implementation: CompressionImplementation) -> "HybridMatrix":
        quantized = self.quantized.switch_implementation(implementation)
        adapter = self.adapter
        if adapter is not None:
            adapter = adapter.switch_implementation(implementation)
        return replace(self, quantized=quantized, adapter=adapter)

    def lookup_embedding(
        self,
        row_index: int | Int[Array, "*batch"],
        *,
        dtype: DTypeLike | None = None,
        keychain: Keychain,
        forward_pass_config: MatmulConfig = MatmulConfig(),
    ) -> Float[Array, "*batch out_channels"]:
        self._raise_if_batched()
        if not isinstance(self.quantized, EmbeddingMatrix):
            raise TypeError("Hybrid embedding lookup requires an embedding-compatible quantization matrix.")
        if self.adapter is not None:
            raise TypeError("Hybrid embedding lookup does not support adapters.")
        # Rows are decoded and rotated in float32 and rounded once, like decompress.
        rows = self.quantized.lookup_embedding(
            row_index,
            dtype=jnp.float32,
            keychain=keychain,
            forward_pass_config=forward_pass_config,
        )
        if self.incoherence_signs is not None:
            rows = self.incoherence_signs.unprocess_embedding(rows)
        return rows.astype(self.dtype if dtype is None else dtype)

    def dot(
        self,
        vector: Float[Array, " source_channels"],
        *,
        keychain: Keychain,
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: bool = False,
    ) -> Float[Array, " target_channels"]:
        if transposed and self.adapter is not None:
            raise TypeError("Hybrid transposed matmul does not support adapters.")

        quantized_keychain, adapter_keychain = keychain.split(2)
        transformed_vector = vector
        if self.incoherence_signs is not None:
            transformed_vector = self.incoherence_signs.input_transform(vector, transposed=transposed)
        result = self.quantized.dot(
            transformed_vector,
            keychain=quantized_keychain,
            forward_pass_config=forward_pass_config,
            transposed=transposed,
        )
        if self.adapter is not None:
            result = result + self.adapter.dot(
                vector,
                keychain=adapter_keychain,
                forward_pass_config=forward_pass_config,
            )

        if self.incoherence_signs is not None:
            result = self.incoherence_signs.output_transform(result, transposed=transposed)
        return result
