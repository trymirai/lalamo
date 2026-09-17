from dataclasses import dataclass, replace
from typing import Self

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec
from jaxtyping import Array, DTypeLike, Float, Int, Key, UInt8

from lalamo.initializer import EmptyInitializer
from lalamo.module import Keychain, field
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import is_dummy_array
from lalamo.utils.precision import use_dot_algorithm_preset
from lalamo.utils.sharding import ShardingConfig, lookup_sharded_indices, sharding_of, with_sharding
from lalamo.weight_matrix import (
    CompressionImplementation,
    EmbeddingMatrix,
    FullPrecisionMatrix,
    FullPrecisionSpec,
    Layout,
    MatmulConfig,
    WeightMatrixSpec,
)

from .s_trellis import full_rotation
from .utils.packing import unpack_uint8_to_uint
from .utils.row_dot import row_batched_dot


@dataclass(frozen=True)
class SDirectionSpec(WeightMatrixSpec):
    layout: Layout

    def compress(
        self,
        weights: Float[Array, "out_channels in_channels"],
        *,
        key: Key[Array, ""] | None = None,  # noqa: ARG002
        preconditioner: Preconditioner | None = None,  # noqa: ARG002
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,  # noqa: ARG002
        sharding_config: ShardingConfig,
        is_sharded: bool = True,
    ) -> "SDirectionMatrix":
        if not is_dummy_array(weights):
            raise ValueError("S direction matrices must be loaded from saved parameters; fitting is not supported")
        rows, columns = self.layout.weight_shape((), *weights.shape)
        assert columns >= 1024
        initializer = EmptyInitializer(weights.dtype, sharding_config)
        return SDirectionMatrix(
            spec=self,
            sharding_config=sharding_config,
            is_sharded=is_sharded,
            codes=initializer.zeros((rows, columns // 1024 * 384), dtype=jnp.uint8),
            levels=initializer.zeros((8,), dtype=jnp.float32),
            unit_scale=initializer.zeros((), dtype=jnp.float32),
            mean_norm=initializer.zeros((), dtype=jnp.float32),
            tail=initializer.zeros((rows, columns % 1024)),
        ).switch_sharding_config(sharding_config)


class SDirectionMatrix(EmbeddingMatrix[SDirectionSpec]):
    codes: UInt8[Array, "rows bytes"]
    levels: Float[Array, "8"] = field(trainable=False)
    unit_scale: Float[Array, ""] = field(trainable=False)
    mean_norm: Float[Array, ""] = field(trainable=False)
    tail: Float[Array, "rows tail_columns"]

    def __check_init__(self) -> None:
        assert self.codes.dtype == jnp.uint8
        assert self.codes.shape[1] > 0 and self.codes.shape[1] % 384 == 0
        assert self.levels.shape == (8,) and self.levels.dtype == jnp.float32
        assert self.unit_scale.shape == self.mean_norm.shape == ()
        assert self.unit_scale.dtype == self.mean_norm.dtype == jnp.float32
        assert self.tail.shape[0] == self.codes.shape[0] and self.tail.shape[1] < 1024

    @property
    def shape(self) -> tuple[int, int]:
        return self.codes.shape[0], self.codes.shape[1] * 8 // 3 + self.tail.shape[1]

    @property
    def dtype(self) -> DTypeLike:
        return self.tail.dtype

    def astype(self, dtype: DTypeLike) -> Self:
        return replace(self, tail=self.tail.astype(dtype))

    def switch_sharding_config(self, sharding_config: ShardingConfig) -> Self:
        row_axis, _ = self.spec.layout.weight_partition(0, is_sharded=self.is_sharded)
        return replace(
            self,
            sharding_config=sharding_config,
            codes=with_sharding(self.codes, sharding_config.resolve_sharding((row_axis, None))),
            tail=with_sharding(self.tail, sharding_config.resolve_sharding((row_axis, None))),
            levels=with_sharding(self.levels, sharding_config.resolve_sharding((None,))),
            unit_scale=with_sharding(self.unit_scale, sharding_config.resolve_sharding(())),
            mean_norm=with_sharding(self.mean_norm, sharding_config.resolve_sharding(())),
        )

    def _decode(self, codes: Array, tail: Array, dtype: DTypeLike) -> Array:
        indices = unpack_uint8_to_uint(codes, 3)
        row_axes = tuple(sharding_of(codes).spec)[:-1]
        levels = self.levels.at[indices].get(out_sharding=PartitionSpec(*row_axes, None))
        # The producer folds in FP64, then rounds to FP32 and BF16 before any dtype override.
        with jax.enable_x64():
            values = levels.astype(jnp.float64) * self.unit_scale.astype(jnp.float64)
            values = values / jnp.linalg.norm(values, axis=-1, keepdims=True) * self.mean_norm.astype(jnp.float64)
            blocks = values.reshape(*values.shape[:-1], -1, 1024)
            folded = full_rotation(blocks, jnp.ones((1, 1), dtype=jnp.float64)).reshape(values.shape)
            folded = jax.lax.optimization_barrier(folded.astype(jnp.float32))
        folded = jax.lax.optimization_barrier(folded.astype(jnp.bfloat16)).astype(jnp.float32)
        return jnp.concatenate((folded, tail.astype(jnp.float32)), axis=-1).astype(dtype)

    def decompress(self) -> Array:
        return self.spec.layout.to_output_input(self._decode(self.codes, self.tail, self.dtype))

    def to_full_precision(self) -> FullPrecisionMatrix:
        return FullPrecisionSpec(self.spec.layout).compress(
            self.decompress(), sharding_config=self.sharding_config, is_sharded=self.is_sharded
        )

    def lookup_embedding(
        self,
        row_index: int | Int[Array, "*batch"],
        *,
        keychain: Keychain,  # noqa: ARG002
        dtype: DTypeLike | None = None,
        forward_pass_config: MatmulConfig = MatmulConfig(),  # noqa: ARG002
    ) -> Array:
        if self.spec.layout != Layout.INPUT_OUTPUT:
            raise ValueError("Embedding lookup requires input-output layout")
        return self._decode(
            lookup_sharded_indices(self.codes, row_index),
            lookup_sharded_indices(self.tail, row_index),
            self.dtype if dtype is None else dtype,
        )

    def dot(
        self,
        vector: Float[Array, " source_channels"],
        *,
        keychain: Keychain,  # noqa: ARG002
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: bool = False,
    ) -> Array:
        if transposed or self.spec.layout != Layout.OUTPUT_INPUT:
            weights = self.decompress().astype(vector.dtype)
            layout = Layout.INPUT_OUTPUT if transposed else Layout.OUTPUT_INPUT
            with use_dot_algorithm_preset(forward_pass_config.precision):
                return layout.matmul(weights, vector)
        return row_batched_dot(
            lambda row: self._decode(row[0], row[1], self.dtype),
            (self.codes, self.tail),
            vector,
            forward_pass_config.precision,
        )
