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
from lalamo.utils.sharding import ShardingConfig, lookup_sharded_indices, sharding_of
from lalamo.weight_matrix import (
    CompressionImplementation,
    EmbeddingMatrix,
    FullPrecisionMatrix,
    FullPrecisionSpec,
    Layout,
    MatmulConfig,
    WeightMatrixSpec,
)

from .qtip_gaussian import full_rotation
from .utils.packing import unpack_uint8_to_uint
from .utils.row_dot import row_batched_dot

# Rows are coded in 1024-column blocks of 3-bit level indices, each block rotated by one 1024-point Hadamard;
# the remaining columns are stored dense.
BLOCK_COLUMNS = 1024
LEVEL_BITS = 3
BLOCK_BYTES = BLOCK_COLUMNS * LEVEL_BITS // 8


@dataclass(frozen=True)
class DirectionSpec(WeightMatrixSpec):
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
    ) -> "DirectionMatrix":
        if not is_dummy_array(weights):
            raise ValueError("Direction matrices must be loaded from saved parameters; fitting is not supported")
        rows, columns = self.layout.weight_shape((), *weights.shape)
        assert columns >= BLOCK_COLUMNS
        row_axis, _ = self.layout.weight_partition(0, is_sharded=is_sharded)
        initializer = EmptyInitializer(weights.dtype, sharding_config)
        return DirectionMatrix(
            spec=self,
            sharding_config=sharding_config,
            is_sharded=is_sharded,
            codes=initializer.zeros((rows, columns // BLOCK_COLUMNS * BLOCK_BYTES), (row_axis, None), jnp.uint8),
            levels=initializer.zeros((1 << LEVEL_BITS,), dtype=jnp.float32),
            unit_scale=initializer.zeros((), dtype=jnp.float32),
            mean_norm=initializer.zeros((), dtype=jnp.float32),
            tail=initializer.zeros((rows, columns % BLOCK_COLUMNS), (row_axis, None)),
        )


class DirectionMatrix(EmbeddingMatrix[DirectionSpec]):
    codes: UInt8[Array, "rows bytes"]
    levels: Float[Array, " levels"] = field(trainable=False)
    unit_scale: Float[Array, ""] = field(trainable=False)
    mean_norm: Float[Array, ""] = field(trainable=False)
    tail: Float[Array, "rows tail_columns"]

    def __check_init__(self) -> None:
        assert self.codes.dtype == jnp.uint8
        assert self.codes.shape[1] > 0 and self.codes.shape[1] % BLOCK_BYTES == 0
        assert self.levels.shape == (1 << LEVEL_BITS,) and self.levels.dtype == jnp.float32
        assert self.unit_scale.shape == self.mean_norm.shape == ()
        assert self.unit_scale.dtype == self.mean_norm.dtype == jnp.float32
        assert self.tail.shape[0] == self.codes.shape[0] and self.tail.shape[1] < BLOCK_COLUMNS

    @property
    def shape(self) -> tuple[int, int]:
        return self.codes.shape[0], self.codes.shape[1] // BLOCK_BYTES * BLOCK_COLUMNS + self.tail.shape[1]

    @property
    def dtype(self) -> DTypeLike:
        return self.tail.dtype

    def astype(self, dtype: DTypeLike) -> Self:
        return replace(self, tail=self.tail.astype(dtype))

    def row_arrays(self) -> tuple[Array, Array]:
        return self.codes, self.tail

    def decode_rows(self, rows: tuple[Array, Array], dtype: DTypeLike) -> Array:
        codes, tail = rows
        indices = unpack_uint8_to_uint(codes, LEVEL_BITS)
        row_axes = tuple(sharding_of(codes).spec)[: codes.ndim - 1]
        levels = self.levels.at[indices].get(out_sharding=PartitionSpec(*row_axes, None))
        # The producer folds in FP64, then rounds to FP32 and BF16 before any dtype override.
        with jax.enable_x64():
            values = levels.astype(jnp.float64) * self.unit_scale.astype(jnp.float64)
            values = values / jnp.linalg.norm(values, axis=-1, keepdims=True) * self.mean_norm.astype(jnp.float64)
            blocks = values.reshape(*values.shape[:-1], -1, BLOCK_COLUMNS)
            folded = full_rotation(blocks, jnp.ones((1, 1), dtype=jnp.float64)).reshape(values.shape)
            folded = jax.lax.optimization_barrier(folded.astype(jnp.float32))
        folded = jax.lax.optimization_barrier(folded.astype(jnp.bfloat16)).astype(jnp.float32)
        return jnp.concatenate((folded, tail.astype(jnp.float32)), axis=-1).astype(dtype)

    def decompress(self) -> Array:
        return self.spec.layout.to_output_input(self.decode_rows(self.row_arrays(), self.dtype))

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
        rows = jax.tree.map(lambda array: lookup_sharded_indices(array, row_index), self.row_arrays())
        return self.decode_rows(rows, self.dtype if dtype is None else dtype)

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
            lambda rows: self.decode_rows(rows, self.dtype), self.row_arrays(), vector, forward_pass_config.precision
        )
