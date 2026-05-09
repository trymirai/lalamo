from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int, Key

from lalamo.compressed.utils.hadamard import hadamard_transform
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
]


def _random_incoherence_signs(
    channels: int,
    key: Key[Array, ""],
) -> Int[Array, " channels"]:
    factors = jnp.where(jax.random.bernoulli(key, shape=(channels,)), 1, -1)
    return factors.astype(jnp.int32)


def _hadamard_transform_output_axis(
    inputs: Float[Array, "*components out_channels in_channels"],
    block_size: Literal[32, 64, 128],
) -> Float[Array, "*components out_channels in_channels"]:
    transposed = jnp.swapaxes(inputs, -1, -2)
    transformed_transposed = hadamard_transform(transposed, block_size)
    return jnp.swapaxes(transformed_transposed, -1, -2)


def _process_preconditioner_block(
    block: Float[Array, "channels channels"],
    signs: Int[Array, " channels"],
    block_size: Literal[32, 64, 128],
) -> Float[Array, "channels channels"]:
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
    input_block = preconditioner.input_block
    output_block = preconditioner.output_block
    processed_input_block = None
    if input_block is not None:
        processed_input_block = _process_preconditioner_block(
            input_block,
            incoherence_signs.input_signs,
            block_size,
        )

    processed_output_block = None
    if output_block is not None:
        processed_output_block = _process_preconditioner_block(
            output_block,
            incoherence_signs.output_signs,
            block_size,
        )

    return Preconditioner.init(
        input_block=processed_input_block,
        output_block=processed_output_block,
    )


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
        weights: Float[Array, "*components out_channels in_channels"],
        block_size: Literal[32, 64, 128],
    ) -> Float[Array, "*components out_channels in_channels"]:
        signed_weights = weights * self.output_signs.astype(weights.dtype)[..., None]
        signed_weights = signed_weights * self.input_signs.astype(weights.dtype)
        return _hadamard_transform_output_axis(
            hadamard_transform(signed_weights, block_size),
            block_size,
        )

    def unprocess_weights(
        self,
        weights: Float[Array, "*components out_channels in_channels"],
        block_size: Literal[32, 64, 128],
    ) -> Float[Array, "*components out_channels in_channels"]:
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
        weights: Float[Array, "*components out_channels in_channels"],
        *,
        key: Key[Array, ""] | None = None,
        preconditioner: Preconditioner | None = None,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
    ) -> "HybridMatrix":
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
            if preconditioner is not None:
                preconditioner = _process_preconditioner(
                    preconditioner,
                    incoherence_signs,
                    self.incoherence_block_size,
                )
        else:
            incoherence_signs = None

        quantized = self.quantization_spec.compress(
            weights,
            key=quantization_key,
            preconditioner=preconditioner,
            implementation=implementation,
        )
        if self.adapter_spec is not None:
            adapter = self.adapter_spec.compress(
                weights - quantized.decompress(),
                key=adapter_key,
                preconditioner=preconditioner,
                implementation=implementation,
            )
        else:
            adapter = None

        return HybridMatrix(
            spec=self,
            quantized=quantized,
            adapter=adapter,
            incoherence_signs=incoherence_signs,
        )


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
    def decompress(self) -> Float[Array, "*components out_channels in_channels"]:
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
        return HybridMatrix(
            spec=self.spec,
            quantized=quantized,
            adapter=adapter,
            incoherence_signs=self.incoherence_signs,
        )

    def dot(
        self,
        vector: Float[Array, " channels"],
        *,
        keychain: Keychain,
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: bool = False,
    ) -> Float[Array, " channels"]:
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
