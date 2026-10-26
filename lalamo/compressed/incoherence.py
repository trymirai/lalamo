from abc import abstractmethod
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Generic, Literal, Self, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.lax import DotAlgorithmPreset
from jaxtyping import Array, Float, Int, Key

from lalamo.initializer import EmptyInitializer
from lalamo.kernels.hadamard import hadamard_transform
from lalamo.module import field
from lalamo.preconditioner import Preconditioner
from lalamo.utils.registry_abc import RegistryABC
from lalamo.utils.sharding import ShardingConfig, sharding_of

__all__ = [
    "HadamardProcessor",
    "HadamardSpec",
    "IncoherenceProcessingMode",
    "IncoherenceProcessor",
    "IncoherenceProcessorSpec",
    "KroneckerProcessor",
    "KroneckerSpec",
    "kronecker_transform",
]

Weights = Float[Array, "*components out_channels in_channels"]
Activations = Float[Array, "*batch channels"]


class IncoherenceProcessingMode(StrEnum):
    INPUT = "input"
    OUTPUT = "output"
    INPUT_OUTPUT = "input_output"


@dataclass(frozen=True)
class IncoherenceProcessorSpec(RegistryABC):
    @abstractmethod
    def init(
        self, input_dim: int, output_dim: int, key: Key[Array, ""] | None, sharding_config: ShardingConfig
    ) -> "IncoherenceProcessor": ...


SpecT_co = TypeVar("SpecT_co", bound=IncoherenceProcessorSpec, covariant=True)


class IncoherenceProcessor(eqx.Module, Generic[SpecT_co]):  # noqa: UP046
    spec: SpecT_co = field(static=True)

    @abstractmethod
    def process_weights(self, weights: Weights, sharding_config: ShardingConfig) -> Weights: ...

    @abstractmethod
    def process_preconditioner(
        self, preconditioner: Preconditioner, sharding_config: ShardingConfig
    ) -> Preconditioner: ...

    @abstractmethod
    def unprocess_input_axis(self, weights: Weights, sharding_config: ShardingConfig) -> Weights: ...

    @abstractmethod
    def unprocess_output_axis(self, weights: Weights, sharding_config: ShardingConfig) -> Weights: ...

    @abstractmethod
    def input_transform(self, vector: Activations, *, transposed: bool = False) -> Activations: ...

    @abstractmethod
    def output_transform(self, vector: Activations, *, transposed: bool = False) -> Activations: ...

    @abstractmethod
    def unprocess_embedding(self, rows: Activations) -> Activations: ...

    @abstractmethod
    def with_components(self, components: tuple[int, ...]) -> Self: ...

    def unprocess_weights(self, weights: Weights, sharding_config: ShardingConfig) -> Weights:
        return self.unprocess_input_axis(self.unprocess_output_axis(weights, sharding_config), sharding_config)


def _random_signs(channels: int, key: Key[Array, ""], sharding_config: ShardingConfig) -> Int[Array, " channels"]:
    signs = jnp.where(jax.random.bernoulli(key, shape=(channels,)), 1, -1)
    return jax.device_put(signs, sharding_config.make_sharding((None,)))


def _hadamard_transform_output_axis(inputs: Weights, block_size: Literal[32, 64, 128]) -> Weights:
    return jnp.swapaxes(hadamard_transform(jnp.swapaxes(inputs, -1, -2), block_size), -1, -2)


def _process_preconditioner_block(
    block: Float[Array, "*components channels channels"] | None,
    signs: Int[Array, " channels"] | None,
    block_size: Literal[32, 64, 128],
    sharding_config: ShardingConfig,
) -> Float[Array, "*components channels channels"] | None:
    if block is None or signs is None:
        return block
    block = jax.device_put(block, sharding_config.resolve_sharding((None,) * block.ndim))
    signs = jax.device_put(signs, sharding_config.resolve_sharding((None,))).astype(block.dtype)
    signed_block = block * signs[..., None] * signs[None, ...]
    return _hadamard_transform_output_axis(hadamard_transform(signed_block, block_size), block_size)


@dataclass(frozen=True)
class HadamardSpec(IncoherenceProcessorSpec):
    block_size: Literal[32, 64, 128] = 32
    processing_mode: IncoherenceProcessingMode = IncoherenceProcessingMode.INPUT_OUTPUT

    def init(
        self, input_dim: int, output_dim: int, key: Key[Array, ""] | None, sharding_config: ShardingConfig
    ) -> "HadamardProcessor":
        if key is None:
            raise ValueError("Cannot initialize random incoherence signs without a random key.")
        key = jax.device_put(key, sharding_config.make_sharding(()))
        input_key, output_key = jax.random.split(key, 2)
        input_signs = None
        if self.processing_mode != IncoherenceProcessingMode.OUTPUT:
            input_signs = _random_signs(input_dim, input_key, sharding_config)
        output_signs = None
        if self.processing_mode != IncoherenceProcessingMode.INPUT:
            output_signs = _random_signs(output_dim, output_key, sharding_config)
        return HadamardProcessor(self, input_signs, output_signs)


class HadamardProcessor(IncoherenceProcessor[HadamardSpec]):
    input_signs: Int[Array, "*components in_channels"] | None = field(trainable=False)
    output_signs: Int[Array, "*components out_channels"] | None = field(trainable=False)

    @classmethod
    def of(
        cls,
        input_signs: Int[Array, " in_channels"] | None,
        output_signs: Int[Array, " out_channels"] | None,
        block_size: Literal[32, 64, 128] = 32,
    ) -> Self:
        match input_signs is not None, output_signs is not None:
            case True, True:
                mode = IncoherenceProcessingMode.INPUT_OUTPUT
            case True, False:
                mode = IncoherenceProcessingMode.INPUT
            case False, True:
                mode = IncoherenceProcessingMode.OUTPUT
            case _:
                raise ValueError("A Hadamard processor needs input signs, output signs, or both.")
        return cls(HadamardSpec(block_size, mode), input_signs, output_signs)

    def process_weights(self, weights: Weights, sharding_config: ShardingConfig) -> Weights:
        input_sharding = sharding_of(weights)
        weights = jax.device_put(weights, sharding_config.resolve_sharding((None,) * weights.ndim))
        if self.input_signs is not None:
            input_signs = self.input_signs.astype(weights.dtype)
            weights = hadamard_transform(weights * input_signs[..., None, :], self.spec.block_size)
        if self.output_signs is not None:
            output_signs = self.output_signs.astype(weights.dtype)
            weights = _hadamard_transform_output_axis(weights * output_signs[..., None], self.spec.block_size)
        return jax.device_put(weights, input_sharding)

    def process_preconditioner(
        self, preconditioner: Preconditioner, sharding_config: ShardingConfig
    ) -> Preconditioner:
        block_size = self.spec.block_size
        return Preconditioner.init(
            input_block=_process_preconditioner_block(
                preconditioner.input_block, self.input_signs, block_size, sharding_config
            ),
            output_block=_process_preconditioner_block(
                preconditioner.output_block, self.output_signs, block_size, sharding_config
            ),
        )

    def unprocess_input_axis(self, weights: Weights, sharding_config: ShardingConfig) -> Weights:
        if self.input_signs is None:
            return weights
        input_sharding = sharding_of(weights)
        weights = jax.device_put(weights, sharding_config.resolve_sharding((None,) * weights.ndim))
        restored = hadamard_transform(weights, self.spec.block_size)
        return jax.device_put(restored * self.input_signs.astype(restored.dtype)[..., None, :], input_sharding)

    def unprocess_output_axis(self, weights: Weights, sharding_config: ShardingConfig) -> Weights:
        if self.output_signs is None:
            return weights
        input_sharding = sharding_of(weights)
        weights = jax.device_put(weights, sharding_config.resolve_sharding((None,) * weights.ndim))
        restored = _hadamard_transform_output_axis(weights, self.spec.block_size)
        return jax.device_put(restored * self.output_signs.astype(restored.dtype)[..., None], input_sharding)

    def input_transform(self, vector: Activations, *, transposed: bool = False) -> Activations:
        signs = self.output_signs if transposed else self.input_signs
        if signs is None:
            return vector
        return hadamard_transform(vector * signs.astype(vector.dtype), self.spec.block_size)

    def output_transform(self, vector: Activations, *, transposed: bool = False) -> Activations:
        signs = self.input_signs if transposed else self.output_signs
        if signs is None:
            return vector
        transformed = hadamard_transform(vector, self.spec.block_size)
        return transformed * signs.astype(transformed.dtype)

    def unprocess_embedding(self, rows: Activations) -> Activations:
        if self.input_signs is not None:
            raise ValueError("Hybrid embedding lookup is only supported when input RHT is disabled.")
        return self.output_transform(rows)

    def with_components(self, components: tuple[int, ...]) -> Self:
        def broadcast(signs: Array | None) -> Array | None:
            return None if signs is None else jnp.broadcast_to(signs, (*components, signs.shape[-1]))

        return replace(self, input_signs=broadcast(self.input_signs), output_signs=broadcast(self.output_signs))


def kronecker_transform(values: Array, orthogonal_factor: Array) -> Array:
    # Multiplies the last axis by kron(H, Q) without materializing it: H is the normalized Hadamard matrix of the
    # power-of-two part of the width, Q the orthogonal factor of its odd part.
    order = orthogonal_factor.shape[0]
    width = values.shape[-1] // order
    assert orthogonal_factor.shape == (order, order)
    assert width > 0 and width & (width - 1) == 0
    result = values.reshape(*values.shape[:-1], width, order)
    stride = 1
    while stride < width:
        grouped = result.reshape(*values.shape[:-1], -1, 2, stride, order)
        left, right = grouped[..., 0, :, :], grouped[..., 1, :, :]
        result = jnp.concatenate((left + right, left - right), axis=-2).reshape(result.shape)
        stride *= 2
    result = result / jnp.sqrt(jnp.asarray(width, dtype=values.dtype))
    if order == 1:
        return (result * orthogonal_factor[0, 0]).reshape(values.shape)
    return jnp.matmul(result, orthogonal_factor, precision=DotAlgorithmPreset.F32_F32_F32).reshape(values.shape)


@dataclass(frozen=True)
class KroneckerSpec(IncoherenceProcessorSpec):
    processing_mode: IncoherenceProcessingMode = IncoherenceProcessingMode.INPUT

    def __post_init__(self) -> None:
        if self.processing_mode == IncoherenceProcessingMode.INPUT_OUTPUT:
            raise ValueError("A Kronecker rotation acts on one axis")

    def init(
        self,
        input_dim: int,
        output_dim: int,
        key: Key[Array, ""] | None,  # noqa: ARG002
        sharding_config: ShardingConfig,
    ) -> "KroneckerProcessor":
        # The rotation is only ever loaded from saved parameters.
        channels = input_dim if self.processing_mode == IncoherenceProcessingMode.INPUT else output_dim
        initializer = EmptyInitializer(jnp.float32, sharding_config)
        order = channels // (channels & -channels)
        return KroneckerProcessor(
            self, initializer.zeros((channels,), dtype=jnp.int32), initializer.zeros((order, order))
        )


class KroneckerProcessor(IncoherenceProcessor[KroneckerSpec]):
    # Rotates the processed axis by kron(H, Q) after flipping the signs of its channels.
    signs: Int[Array, " channels"] = field(trainable=False)
    orthogonal_factor: Float[Array, "order order"] = field(trainable=False)

    @classmethod
    def of(
        cls,
        signs: Int[Array, " channels"],
        orthogonal_factor: Float[Array, "order order"],
        processing_mode: IncoherenceProcessingMode = IncoherenceProcessingMode.INPUT,
    ) -> Self:
        return cls(KroneckerSpec(processing_mode), signs, orthogonal_factor)

    @property
    def _is_input(self) -> bool:
        return self.spec.processing_mode == IncoherenceProcessingMode.INPUT

    def _rotate(self, values: Activations) -> Activations:
        signed = values.astype(jnp.float32) * self.signs.astype(jnp.float32)
        return kronecker_transform(signed, self.orthogonal_factor.T).astype(values.dtype)

    def _unrotate(self, values: Activations) -> Activations:
        unrotated = kronecker_transform(values.astype(jnp.float32), self.orthogonal_factor)
        return (unrotated * self.signs.astype(jnp.float32)).astype(values.dtype)

    def process_weights(self, weights: Weights, sharding_config: ShardingConfig) -> Weights:
        raise NotImplementedError("Kronecker rotations are loaded from saved parameters, not fitted")

    def process_preconditioner(
        self, preconditioner: Preconditioner, sharding_config: ShardingConfig
    ) -> Preconditioner:
        raise NotImplementedError("Kronecker rotations are loaded from saved parameters, not fitted")

    def unprocess_input_axis(self, weights: Weights, sharding_config: ShardingConfig) -> Weights:  # noqa: ARG002
        return self._unrotate(weights) if self._is_input else weights

    def unprocess_output_axis(self, weights: Weights, sharding_config: ShardingConfig) -> Weights:  # noqa: ARG002
        if self._is_input:
            return weights
        return jnp.swapaxes(self._unrotate(jnp.swapaxes(weights, -1, -2)), -1, -2)

    def input_transform(self, vector: Activations, *, transposed: bool = False) -> Activations:
        return vector if self._is_input == transposed else self._rotate(vector)

    def output_transform(self, vector: Activations, *, transposed: bool = False) -> Activations:
        return vector if self._is_input != transposed else self._unrotate(vector)

    def unprocess_embedding(self, rows: Activations) -> Activations:
        if self._is_input:
            raise ValueError("Hybrid embedding lookup rotates the output axis")
        return self._unrotate(rows)

    def with_components(self, components: tuple[int, ...]) -> Self:
        assert not components, "Kronecker rotations are not batched"
        return self
